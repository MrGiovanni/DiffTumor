import numpy as np
import nibabel as nib
import os
import pandas as pd
import csv
from scipy import ndimage
import argparse

def resample_image(image, original_spacing, target_spacing=(1, 1, 1), order=1):
    """
    Resample the image to the target spacing.

    Parameters:
    image (nibabel.Nifti1Image): Input image to resample.
    target_spacing (tuple): Target spacing in x, y, z directions.

    Returns:
    numpy.ndarray: Resampled image data.
    """
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(image.shape * resize_factor).astype(int)

    try:
        image = image.get_fdata()
    except:
        pass
    resampled_image = ndimage.zoom(image, resize_factor, order=order)

    return resampled_image, resize_factor 

def detection(tumor_mask, organ_mask, spacing, th=10, erode=True):
    """
    Returns 1 if there are tumors in the ct scan, 0 otherwise.
    
    tumor_mask: file path for the tumor segmentation mask, e.g., liver_tumor.nii.gz
    organ_mask: file path for the organ segmentation mask, e.g., liver.nii.gz
    th: detection threshold, in mm^3. Only considers a sample as positive if the total tumor 
    mask volume is bigger than th
    erode: performs binary erosion if True, denoises mask, avoiding false positives
    """
    array = nib.load(tumor_mask).get_fdata()
    
    if organ_mask is not None:
        array = array * nib.load(organ_mask).get_fdata()
        
    array, _ = resample_image(array, original_spacing=spacing,
                              target_spacing=(1, 1, 1), order=0)
    if erode:
        array = ndimage.binary_erosion(array, structure=np.ones((3, 3, 3)), iterations=1)
   
    if array.sum() > th:
        return 1
    else: 
        return 0
    
def get_spacing(ct_scan_path):
    """
    Get the spacing from a CT scan file.

    Parameters:
    ct_scan_path (str): Path to the CT scan file.

    Returns:
    tuple: The spacing of the CT scan in x, y, z directions.
    """
    ct_scan = nib.load(ct_scan_path)
    spacing = ct_scan.header.get_zooms()
    return spacing

def process_outputs(outputs_folder, ct_folder, th):
    """
    Process the outputs folder and generate a CSV file with detection results.

    Parameters:
    outputs_folder (str): Path to the outputs folder.
    ct_folder (str): Path to the destination folder.
    th (int): Detection threshold.
    """
    csv_file_path = os.path.join(outputs_folder, "tumor_detection_results.csv")
    csv_columns = ["Anon_MRN", "Anon_Acc_#", "Series", "liver tumor", "pancreas tumor", "kidney tumor"]
    
    # Erase the CSV file if it already exists
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        for anon_mrn in os.listdir(outputs_folder):
            if not os.path.isdir(os.path.join(outputs_folder,anon_mrn)):
                continue
            for anon_acc in os.listdir(os.path.join(outputs_folder,anon_mrn)):
                if not os.path.isdir(os.path.join(outputs_folder,anon_mrn,anon_acc)):
                    continue
                series=os.listdir(os.path.join(outputs_folder,anon_mrn,anon_acc))
                if len(series)>1:
                    raise ValueError ('More than 1 series in ', 
                                      os.path.join(outputs_folder,anon_mrn,anon_acc))
                series=series[0]
                row = {"Anon_MRN": anon_mrn,
                        "Anon_Acc_#": anon_acc,
                        "Series": series}
                ct_path = os.path.join(ct_folder,anon_mrn,anon_acc,series,'ct.nii.gz')
                spacing = get_spacing(ct_path)
                
                for organ in ['liver', 'kidney', 'pancreas']:
                    tumor_mask_path = os.path.join(outputs_folder,anon_mrn,anon_acc,series,
                                                   'predictions',organ+'_tumor.nii.gz')
                    organ_mask_path = os.path.join(outputs_folder,anon_mrn,anon_acc,series,
                                                   'predictions',organ+'.nii.gz')
                    row[f"{organ} tumor"] = detection(tumor_mask_path, organ_mask_path, spacing, th)
                    
                writer.writerow(row)
                print(row)
                
        print('CSV file saved at: ', csv_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect tumors from segmentation masks and generate a CSV file with the resulting binary labels.")
    parser.add_argument("--outputs_folder", type=str, help="Path to the outputs folder (segmentations)")
    parser.add_argument("--ct_folder", type=str, help="Path to the original CT scans folder (nifti)")
    parser.add_argument("--th", type=int, default=70, help="Detection threshold, mm^3. Minimum volume in an organ tumor segmentation mask for the sample to be considered positive for tumor. Default is 70 mm^3.")
    args = parser.parse_args()

    process_outputs(args.outputs_folder, args.ct_folder, args.th)
