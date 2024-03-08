import glob, os, six
import numpy as np
import nibabel as nib # nibabel==4.0.1
import SimpleITK as sitk
from radiomics import featureextractor
import warnings
warnings.filterwarnings('ignore')
import h5py
from scipy import ndimage
import imageio as io
import glob

def readh5(filename, dataset=None):
    fid = h5py.File(filename, 'r')
    if dataset is None:
        # load the first dataset in the h5 file
        dataset = list(fid)[0]
    return np.array(fid[dataset])

def voxel2R(A):
    return (np.array(A)/4*3/np.pi)**(1/3)

def pixel2voxel(A,res=[0.75,0.75,0.5]):
    return np.array(A)*(res[0]*res[1]*res[2])


def filter_liver_early_tumor(data_dir = '/mnt/ccvl15/zzhou82/PublicAbdominalData/04_LiTS/04_LiTS/label/',tumor_save_dir='04_LiTS/early_tumor_label/'):

    total_clot_size=[]
    total_clot_size_mmR=[]
    valid_ct_name=[]
    lael_paths = glob.glob(os.path.join(data_dir, '*'))
    lael_paths.sort()

    for label_path in lael_paths:
        print('label_path',label_path)
        file_name = os.path.basename(label_path)

        label = nib.load(os.path.join(label_path, name))
        pixdim = label.header['pixdim']
        original_affine = label.affine
        spacing_mm = tuple(pixdim[1:4])
        raw_label = label.get_fdata()

        tumor_mask = np.zeros_like(raw_label).astype(np.int16)
        organ_mask = np.zeros_like(raw_label).astype(np.int16)
        organ_mask[raw_label==1] = 1
        organ_mask[raw_label==2] = 1
        tumor_mask[raw_label==2] = 1
        
        save_path=os.path.join(tumor_save_dir, file_name)
        
        if len(np.unique(tumor_mask)) > 1: 
            label_numeric, gt_N = ndimage.label(tumor_mask)
            for segid in range(1, gt_N+1):
                extracted_label_numeric = np.uint8(label_numeric == segid)
                clot_size = np.sum(extracted_label_numeric)
                if clot_size < 8 :
                    continue
                clot_size_mm = pixel2voxel(clot_size, spacing_mm)
                clot_size_mmR = voxel2R(clot_size_mm)
                print('tumor clot_size_mmR',clot_size_mmR)
                if clot_size_mmR <= 10:
                    total_clot_size.append(clot_size)
                    total_clot_size_mmR.append(clot_size_mmR)
                    organ_mask[extracted_label_numeric==1] = 2
                    if not file_name in valid_ct_name:
                        valid_ct_name.append(file_name)
        nib.save(nib.Nifti1Image(organ_mask.astype(np.uint8), original_affine), save_path)

if __name__ == '__main__':
    data_dir = ''
    tumor_save_dir = ''
    os.makedirs(tumor_save_dir, exist_ok=True)
    filter_liver_early_tumor(data_dir, tumor_save_dir)

    file_list = glob.glob(tumor_save_dir+'/*')
    file_list.sort()
    f = open("04_LiTS_valid_names.txt","w")
    for label_path in file_list:
        name = os.path.basename(label_path)
        label_array = nib.load(label_path)
        label_array = label_array.get_fdata()
        if len(np.unique(label_array)) == 3:
            f.write(tumor_save_dir+name)
            f.write('\n')
            print('name', tumor_save_dir+name)
    f.close() 

    