### Tumor Generateion
import random
import cv2
import elasticdeform
import numpy as np
from scipy.ndimage import gaussian_filter
from TumorGeneration.ldm.ddpm.ddim import DDIMSampler
from .ldm.vq_gan_3d.model.vqgan import VQGAN
import matplotlib.pyplot as plt
import SimpleITK as sitk
from .ldm.ddpm import Unet3D, GaussianDiffusion, Tester
from hydra import initialize, compose
import torch
import yaml

# Random select location for tumors.
def random_select(mask_scan, organ_type):
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0].min(), np.where(np.any(mask_scan, axis=(0, 1)))[0].max()

    flag=0
    while 1:
        if flag<=10:
            z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start
        elif flag>10 and flag<=20:
            z = round(random.uniform(0.2, 0.8) * (z_end - z_start)) + z_start
        elif flag>20 and flag<=30:
            z = round(random.uniform(0.1, 0.9) * (z_end - z_start)) + z_start
        else:
            z = round(random.uniform(0.0, 1.0) * (z_end - z_start)) + z_start
        liver_mask = mask_scan[..., z]

        if organ_type == 'liver':
            kernel = np.ones((5,5), dtype=np.uint8)
            liver_mask = cv2.erode(liver_mask, kernel, iterations=1)
        if (liver_mask == 1).sum() > 0:
            break
        flag+=1

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

def center_select(mask_scan):
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0].min(), np.where(np.any(mask_scan, axis=(0, 1)))[0].max()
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0].min(), np.where(np.any(mask_scan, axis=(1, 2)))[0].max()
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0].min(), np.where(np.any(mask_scan, axis=(0, 2)))[0].max()

    z = round(0.5 * (z_end - z_start)) + z_start
    x = round(0.5 * (x_end - x_start)) + x_start
    y = round(0.5 * (y_end - y_start)) + y_start

    xyz = [x, y, z]
    potential_points = xyz

    return potential_points

# generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_fixed_geo(mask_scan, tumor_type, organ_type):
    if tumor_type == 'large':
        enlarge_x, enlarge_y, enlarge_z = 280, 280, 280
    else:
        enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'small':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo


    if tumor_type == 'medium':
        num_tumor = 1
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'large':
        num_tumor = 1
        for _ in range(num_tumor):
            # Large tumor
            
            x = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            y = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            z = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            sigma = random.randint(5, 10)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            if organ_type == 'liver' or organ_type == 'kidney' :
                point = random_select(mask_scan, organ_type)
                new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
                x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
                y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
                z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            else:
                x_start, x_end = np.where(np.any(geo, axis=(1, 2)))[0].min(), np.where(np.any(geo, axis=(1, 2)))[0].max()
                y_start, y_end = np.where(np.any(geo, axis=(0, 2)))[0].min(), np.where(np.any(geo, axis=(0, 2)))[0].max()
                z_start, z_end = np.where(np.any(geo, axis=(0, 1)))[0].min(), np.where(np.any(geo, axis=(0, 1)))[0].max()
                geo = geo[x_start:x_end, y_start:y_end, z_start:z_end]

                point = center_select(mask_scan)

                new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
                x_low = new_point[0] - geo.shape[0]//2
                y_low = new_point[1] - geo.shape[1]//2
                z_low = new_point[2] - geo.shape[2]//2
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_low+geo.shape[0], y_low:y_low+geo.shape[1], z_low:z_low+geo.shape[2]] += geo
    
    geo_mask = geo_mask[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]

    if ((tumor_type == 'medium') or (tumor_type == 'large')) and (organ_type == 'kidney'):
        if random.random() > 0.5:
            geo_mask = (geo_mask>=1)
        else:
            geo_mask = (geo_mask * mask_scan) >=1
    else:
        geo_mask = (geo_mask * mask_scan) >=1

    return geo_mask

def synt_model_prepare(device, vqgan_ckpt='TumorGeneration/model_weight/AutoencoderModel.ckpt', diffusion_ckpt='TumorGeneration/model_weight/', fold=0, organ='liver'):
    with initialize(config_path="diffusion_config/"):
        cfg=compose(config_name="ddpm.yaml")
    print('diffusion_ckpt',diffusion_ckpt)
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt)
    vqgan = vqgan.to(device)
    vqgan.eval()
    
    early_Unet3D = Unet3D(
            dim=cfg.diffusion_img_size,
            dim_mults=cfg.dim_mults,
            channels=cfg.diffusion_num_channels,
            out_dim=cfg.out_dim
            ).to(device)

    early_diffusion = GaussianDiffusion(
            early_Unet3D,
            vqgan_ckpt= vqgan_ckpt, # cfg.vqgan_ckpt,
            image_size=cfg.diffusion_img_size,
            num_frames=cfg.diffusion_depth_size,
            channels=cfg.diffusion_num_channels,
            timesteps=4,
            loss_type=cfg.loss_type,
            device=device
            ).to(device)
    
    noearly_Unet3D = Unet3D(
            dim=cfg.diffusion_img_size,
            dim_mults=cfg.dim_mults,
            channels=cfg.diffusion_num_channels,
            out_dim=cfg.out_dim
            ).to(device)
    
    noearly_diffusion = GaussianDiffusion(
            noearly_Unet3D,
            vqgan_ckpt= vqgan_ckpt,
            image_size=cfg.diffusion_img_size,
            num_frames=cfg.diffusion_depth_size,
            channels=cfg.diffusion_num_channels,
            timesteps=200,
            loss_type=cfg.loss_type,
            device=device
            ).to(device)
    
    early_tester = Tester(early_diffusion)
    early_tester.load(diffusion_ckpt+'{}_early.pt'.format(organ), map_location=device)

    noearly_checkpoint = torch.load(diffusion_ckpt+'{}_noearly.pt'.format(organ), map_location=device)
    noearly_diffusion.load_state_dict(noearly_checkpoint['ema'])
    noearly_sampler = DDIMSampler(noearly_diffusion, schedule="cosine")

    return vqgan, early_tester, noearly_sampler

def synthesize_early_tumor(ct_volume, organ_mask, organ_type, vqgan, tester):
    device=ct_volume.device

    # generate tumor mask
    tumor_types = ['tiny', 'small']
    tumor_probs = np.array([0.5, 0.5])
    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()
    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()
        
        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        
        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        tester.ema_model.eval()
        sample = tester.ema_model.sample(batch_size=volume.shape[0], cond=cond)

        if organ_type == 'liver' or organ_type == 'kidney' :
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = organ_mask + total_tumor_mask

    return final_volume_, organ_tumor_mask

def synthesize_medium_tumor(ct_volume, organ_mask, organ_type, vqgan, sampler, ddim_ts=50):
    device=ct_volume.device

    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()
    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = 'medium'
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()
        
        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        
        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        shape = masked_volume_feat.shape[-4:]
        samples_ddim, _ = sampler.sample(S=ddim_ts,
                                        conditioning=cond,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False)
        samples_ddim = (((samples_ddim + 1.0) / 2.0) * (vqgan.codebook.embeddings.max() -
                                                        vqgan.codebook.embeddings.min())) + vqgan.codebook.embeddings.min()

        sample = vqgan.decode(samples_ddim, quantize=True)
        
        if organ_type == 'liver' or organ_type == 'kidney':
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0

        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = torch.zeros_like(organ_mask)
        organ_tumor_mask[organ_mask==1] = 1
        organ_tumor_mask[total_tumor_mask==1] = 2

    return final_volume_, organ_tumor_mask

def synthesize_large_tumor(ct_volume, organ_mask, organ_type, vqgan, sampler, ddim_ts=50):
    device=ct_volume.device

    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()
    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = 'large'
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()
        
        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        
        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        shape = masked_volume_feat.shape[-4:]
        samples_ddim, _ = sampler.sample(S=ddim_ts,
                                        conditioning=cond,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False)
        samples_ddim = (((samples_ddim + 1.0) / 2.0) * (vqgan.codebook.embeddings.max() -
                                                        vqgan.codebook.embeddings.min())) + vqgan.codebook.embeddings.min()

        sample = vqgan.decode(samples_ddim, quantize=True)

        if organ_type == 'liver' or organ_type == 'kidney':
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0
            
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = torch.zeros_like(organ_mask)
        organ_tumor_mask[organ_mask==1] = 1
        organ_tumor_mask[total_tumor_mask==1] = 2

    return final_volume_, organ_tumor_mask