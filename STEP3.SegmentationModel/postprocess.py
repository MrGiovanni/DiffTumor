from scipy.ndimage import label 
import nibabel as nib
import cc3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def voxel2mm(N, spacing):
    return int(N * np.prod(spacing))

def size2r(size):
    r = np.cbrt(3 * size / (4 * np.pi))
    return r

def pred_size_threshold_postprocess2(pred_volume, spacing_mm=[1,1,1]):

    pred_les_numeric, pred_les_N = cc3d.connected_components(pred_volume, connectivity=26, return_N=True)
    
    for segid in range(1, pred_les_N+1):
        extracted_image = np.uint8(pred_les_numeric * (pred_les_numeric == segid)>0)
    
        pred_size = np.sum(extracted_image)
        # if pred_size < size_threshold:
        #     pred_les_numeric = pred_les_numeric * (1-extracted_image)
        #     continue
        size = voxel2mm(pred_size, spacing_mm)
        r = size2r(size)
        if r<3:
            pred_les_numeric = pred_les_numeric * (1-extracted_image)
    pred_volume_ = np.uint8(pred_les_numeric > 0)
        
    return pred_volume_

def pred_size_threshold_postprocess(pred_volume_cls, size_threshold=20):
    
    pred_volume_cls_binary = np.uint8(pred_volume_cls>0)
    pred_les_numeric, pred_les_N = cc3d.connected_components(pred_volume_cls_binary, connectivity=26, return_N=True)
        
    for segid in range(1, pred_les_N+1):
        extracted_image = np.uint8(pred_les_numeric * (pred_les_numeric == segid)>0)
        
        pred_size = np.sum(extracted_image)
        if pred_size < size_threshold:
            pred_les_numeric = pred_les_numeric * (1-extracted_image)
    pred_volume_cls = np.uint8(pred_les_numeric > 0) * pred_volume_cls
    
    return pred_volume_cls

def load_pred(pred_path):

    pred_volume = nib.load(pred_path)
    affine = pred_volume.affine
    pixdim = pred_volume.header['pixdim']
    spacing_mm = tuple(pixdim[1:4])
    pred_volume = pred_volume.get_fdata()  # HxWxC (Numeric)

    return pred_volume, spacing_mm, affine

file_root = 'out/pancreas/pancreas.segresnet/pancreas.segresnet/0.75/pred'
save_root = 'out/pancreas/pancreas.segresnet/pancreas.segresnet/0.75/pred_post2'
os.makedirs(save_root, exist_ok=True)
file_names = os.listdir(file_root)
file_names.sort()
for file in file_names:
    print(file)
    pred_volume, spacing_mm, affine = load_pred(os.path.join(file_root, file))
    # pred_volume = pred_size_threshold_postprocess(pred_volume)
    pred_volume = pred_size_threshold_postprocess2(pred_volume, spacing_mm)
    nib.save(
                nib.Nifti1Image(pred_volume.astype(np.uint8), affine), os.path.join(save_root, file)
            )
