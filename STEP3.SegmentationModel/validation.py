import os, time, csv
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from scipy.ndimage import label
from functools import partial
from surface_distance import compute_surface_distances,compute_surface_dice_at_tolerance
import monai
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist
from monai.transforms import AsDiscrete,AsDiscreted,Compose,Invertd,SaveImaged
from monai import transforms, data
from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
import nibabel as nib

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='liver tumor validation')

# file dir
parser.add_argument('--data_root', default=None, type=str)
parser.add_argument('--datafold_dir', default=None, type=str)
parser.add_argument('--tumor_type', default='early', type=str)
parser.add_argument('--organ_type', default='liver', type=str)
parser.add_argument('--fold', default=0, type=int)

parser.add_argument('--save_dir', default='out', type=str)
parser.add_argument('--checkpoint', action='store_true')

parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--feature_size', default=16, type=int)
parser.add_argument('--val_overlap', default=0.75, type=float)
parser.add_argument('--num_classes', default=3, type=int)

parser.add_argument('--model', default='unet', type=str)
parser.add_argument('--swin_type', default='base', type=str)

def organ_region_filter_out(organ_mask, tumor_mask):
    ## dialtion
    organ_mask = ndimage.binary_closing(organ_mask, structure=np.ones((5,5,5)))
    organ_mask = ndimage.binary_dilation(organ_mask, structure=np.ones((5,5,5)))
    ## filter out
    tumor_mask = organ_mask * tumor_mask

    return tumor_mask

def denoise_pred(pred: np.ndarray):
    """
    # 0: background, 1: liver, 2: tumor.
    pred.shape: (3, H, W, D)
    """
    denoise_pred = np.zeros_like(pred)


    denoise_pred[1, ...] = pred[1, ...]

    denoise_pred[2, ...] = pred[1, ...] * pred[2,...]

    denoise_pred[0,...] = 1 - np.logical_or(denoise_pred[1,...], denoise_pred[2,...])

    return denoise_pred

def cal_dice(pred, true):
    intersection = np.sum(pred[true==1]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def cal_dice_nsd(pred, truth, spacing_mm=(1,1,1), tolerance=2):
    dice = cal_dice(pred, truth)
    # cal nsd
    surface_distances = compute_surface_distances(truth.astype(bool), pred.astype(bool), spacing_mm=spacing_mm)
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)

    return (dice, nsd)


def _get_model(args):
    inf_size = [96, 96, 96]
    print(args.model)
    if args.model == 'swinunetr':
        if args.swin_type == 'tiny':
            feature_size=12
        elif args.swin_type == 'small':
            feature_size=24
        elif args.swin_type == 'base':
            feature_size=48

        model = SwinUNETR_v2(in_channels=1,
                          out_channels=3,
                          img_size=(96, 96, 96),
                          feature_size=feature_size,
                          patch_size=2,
                          depths=[2, 2, 2, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=[7, 7, 7])
        
    elif args.model == 'unet':
        from monai.networks.nets import UNet 
        model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
    elif args.model == 'nnunet':
        from monai.networks.nets import DynUNet
        from dynunet_pipeline.create_network import get_kernels_strides
        from dynunet_pipeline.task_params import deep_supr_num
        task_id = 'custom'
        kernels, strides = get_kernels_strides(task_id)
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=deep_supr_num[task_id],
        )
        
    else:
        raise ValueError('Unsupported model ' + str(args.model))


    if args.checkpoint:
        checkpoint = torch.load(os.path.join(args.log_dir, 'model.pt'), map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)
        print('Use logdir weights')
    else:
        model_dict = torch.load(os.path.join(args.log_dir, 'model.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('Use logdir weights')

    model = model.cuda()
    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,  overlap=args.val_overlap, mode='gaussian')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    return model, model_inferer

def _get_loader(args):
    val_org_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "organ_pseudo"]),
            transforms.AddChanneld(keys=["image", "label", "organ_pseudo"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            transforms.SpatialPadd(keys=["image"], mode="minimum", spatial_size=[96, 96, 96]),
            transforms.ToTensord(keys=["image", "label", "organ_pseudo"]),
        ]
    )
    val_img=[]
    val_lbl=[]
    val_name=[]
    val_pseudo_lbl = []
    for line in open(os.path.join(args.datafold_dir, 'real_{}_val_{}.txt'.format(args.tumor_type, args.fold))):
        name = line.strip().split()[1].split('.')[0]
        val_img.append(args.data_root + line.strip().split()[0])
        val_lbl.append(args.data_root + line.strip().split()[1])
        val_pseudo_lbl.append('organ_pseudo_swin_new/'+args.organ_type + '/' + os.path.basename(line.strip().split()[1]))
        val_name.append(name)
    data_dicts_val = [{'image': image, 'label': label, 'organ_pseudo': organ_pseudo, 'name': name}
                for image, label, organ_pseudo, name in zip(val_img, val_lbl, val_pseudo_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))

    val_org_ds = data.Dataset(data_dicts_val, transform=val_org_transform)
    val_org_loader = data.DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4, sampler=None, pin_memory=True)

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=val_org_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=3),
        AsDiscreted(keys="label", to_onehot=3),
        AsDiscreted(keys="organ_pseudo", to_onehot=3),
    ])
    
    return val_org_loader, post_transforms

def main():
    args = parser.parse_args()
    model_name = args.log_dir.split('/')[-1]
    args.model_name = model_name
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    torch.cuda.set_device(0) #use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    ## loader and post_transform
    val_loader, post_transforms = _get_loader(args)

    ## NETWORK
    model, model_inferer = _get_model(args)

    organ_dice = []
    organ_nsd  = []
    tumor_dice = []
    tumor_nsd  = []
    header = ['name', 'organ_dice', 'organ_nsd', 'tumor_dice', 'tumor_nsd']
    rows = []

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            val_inputs = val_data["image"].cuda()
            name = val_data['label_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
            original_affine = val_data["label_meta_dict"]["affine"][0].numpy()
            pixdim = val_data['label_meta_dict']['pixdim'].cpu().numpy()
            spacing_mm = tuple(pixdim[0][1:4])
            # breakpoint()
            val_data['label'][val_data['label']==3] = 1
            val_data["pred"] = model_inferer(val_inputs)
            val_data = [post_transforms(i) for i in data.decollate_batch(val_data)]
            val_outputs, val_labels, val_organ_pseudo = val_data[0]['pred'], val_data[0]['label'], val_data[0]['organ_pseudo']
            
            # val_outpus.shape == val_labels.shape  (3, H, W, Z)
            val_outputs, val_labels = val_outputs.detach().cpu().numpy(), val_labels.detach().cpu().numpy()
            val_organ_pseudo = val_organ_pseudo.detach().cpu().numpy()
            val_outputs[1, ...] = val_organ_pseudo[1, ...]

            val_outputs = denoise_pred(val_outputs)

            current_liver_dice, current_liver_nsd = cal_dice_nsd(val_outputs[1,...], val_labels[1,...], spacing_mm=spacing_mm)
            current_tumor_dice, current_tumor_nsd = cal_dice_nsd(val_outputs[2,...], val_labels[2,...], spacing_mm=spacing_mm)

            organ_dice.append(current_liver_dice)
            organ_nsd.append(current_liver_nsd)
            tumor_dice.append(current_tumor_dice)
            tumor_nsd.append(current_tumor_nsd)


            row = [name, current_liver_dice, current_liver_nsd, current_tumor_dice, current_tumor_nsd]
            rows.append(row)

            print(name, val_outputs[0].shape, \
                'dice: [{:.3f}  {:.3f}]; nsd: [{:.3f}  {:.3f}]'.format(current_liver_dice, current_tumor_dice, current_liver_nsd, current_tumor_nsd), \
                'time {:.2f}s'.format(time.time() - start_time))

            # save the prediction
            output_dir = os.path.join(args.save_dir, args.model_name, str(args.val_overlap), 'pred')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # val_outputs = np.argmax(val_outputs, axis=0)
            val_outputs_ = np.zeros_like(val_outputs[0])
            val_outputs_[val_outputs[1]==1] = 1
            val_outputs_[val_outputs[2]==1] = 2

            nib.save(
                nib.Nifti1Image(val_outputs_.astype(np.uint8), original_affine), os.path.join(output_dir, f'{name}.nii.gz')
            )


        print("organ dice:", np.mean(organ_dice))
        print("organ nsd:", np.mean(organ_nsd))
        print("tumor dice:", np.mean(tumor_dice))
        print("tumor nsd",np.mean(tumor_nsd))
        rows.append(['average', organ_dice, organ_nsd, tumor_dice, tumor_nsd])

        # save metrics to cvs file
        csv_save = os.path.join(args.save_dir, args.model_name, str(args.val_overlap))
        if not os.path.exists(csv_save):
            os.makedirs(csv_save)
        csv_name = os.path.join(csv_save, 'metrics.csv')
        with open(csv_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

if __name__ == "__main__":
    main()
