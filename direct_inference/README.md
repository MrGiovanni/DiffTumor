<h1 align="center">DiffTumor</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Apply to New CT Scans</h3>
<p align="center">
    <a href='https://www.cs.jhu.edu/~alanlab/Pubs24/chen2024towards.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> 
    <a href='https://github.com/MrGiovanni/DiffTumor/blob/main/documents/cvpr_slides.pdf'><img src='https://img.shields.io/badge/Slides-PDF-orange'></a> 
    <a href='https://hub.jhu.edu/2024/05/30/researchers-create-artificial-tumors-for-cancer-research/'><img src='https://img.shields.io/badge/JHU-News-yellow'></a>
    <br/>
    <a href="https://github.com/MrGiovanni/DiffTumor"><img src="https://img.shields.io/github/stars/MrGiovanni/DiffTumor?style=social" /></a>
    <a href="https://twitter.com/bodymaps317"><img src="https://img.shields.io/twitter/follow/BodyMaps" alt="Follow on Twitter" /></a>
</p>

### Instructions

We use [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) for easier portability of code across diverse servers. Thus, the only requirement for running this code is Singularity and CUDA drivers.


#### 0- Download
```bash
git clone https://github.com/PedroRASB/ReportGeneration
cd ReportGeneration
wget https://huggingface.co/qicq1c/DiffTumor/resolve/main/difftumor_final.sif
wget https://huggingface.co/prasb/Singularities/resolve/main/PreAndPostProcessing.sif
wget https://huggingface.co/prasb/Singularities/resolve/main/overlay.img
```

#### 1- Convert data from dicom to nifti 

The code below creates one nifti file per study (one file per Anon_Acc_# folder inside source_folder). This file represents the largest series (more .dcm files) inside each Anon_Acc_# folder.

```bash
SOURCE=/path/to/source_folder/
CONVERTED=/path/to/destination_folder/
mkdir -p $CONVERTED
COMMAND="python ConvertDcm2Nii.py $SOURCE $CONVERTED"
singularity exec --bind $(pwd):$(pwd) --bind $SOURCE:$SOURCE --bind $CONVERTED:$CONVERTED PreAndPostProcessing.sif bash -c "$COMMAND;"
```

Details (click to open):

<details>
  <summary>Input data format: </summary>

```
source_folder/
├── Anon_MRN/
|    └── Anon_Acc_#/
|        ├── Series_1/
|        │   ├── file1.dcm
|        │   ├── file2.dcm
|        │   └── file1.dcm
|        ├── Series_2/
|        |   ├── file1.dcm
|        |   └── file2.dcm
|        └── ...
├──...
└──...
```

</details>

<details>
  <summary>Output data format</summary>
  
```
destination_folder/
├── Anon_MRN/
│    └── Anon_Acc_#/
│        └── Series_with_most_files/
│            └── ct.nii.gz
├──...
└──...
```
</details>

<details>
  <summary>Error handling</summary>
  
Error handling: if the longest series in Anon_Acc_# cannot be converted to nifti, the code will log an error in conversion_errors.txt and try convering the second-longest series. If no series in Anon_Acc_# can be successfully converted, the code will log "No series in Anon_Acc_# could be successfully converted" in conversion_errors.txt.

Check how many ct scans were successfully converted to nifti:
```bash
cd /path/to/destination_folder/
find . -type f -name "*.nii.gz" | wc -l
cd -
```
</details>




#### 2- Create and save per-voxel segmentations with DiffTumor
```bash
OUTPUTS=/path/to/outputs_folder/
mkdir -p $OUTPUTS
logs_file="logs_difftumor.txt"
echo 0 > $logs_file
bash RunDifftumor.sh $CONVERTED $OUTPUTS >> "$output_file" 2>&1
```

Logs and errors will be saved to logs_difftumor.txt.

<details>
  <summary>DiffTumor output data format</summary>
  
```
outputs_folder/
├── Anon_MRN/
│    └── Anon_Acc_#/
│        └── Series_with_most_files/
│            ├── ct.nii.gz
|            └── predictions
|                 ├── liver.nii.gz
|                 ├── pancreas.nii.gz
|                 ├── kidney.nii.gz
|                 ├── liver_tumor.nii.gz
|                 ├── pancreas_tumor.nii.gz
|                 └── kidney_tumor.nii.gz
├──...
└──...
```


</details>


#### 3- Convert per-voxel outputs to sample-level binary outputs
```bash
log="logs_conversion2binary.txt"
echo 0 > $log
COMMAND="python Segmentation2BinaryLabels.py --outputs_folder $OUTPUTS --ct_folder $CONVERTED --th 30"
singularity exec --bind $(pwd):$(pwd) --bind $OUTPUTS:$OUTPUTS --bind $CONVERTED:$CONVERTED PreAndPostProcessing.sif bash -c "$COMMAND;" >> "$log" 2>&1
```

Output: csv file with binary outputs at /path/to/outputs_folder/tumor_detection_results.csv. Logs and errors will be saved in logs_conversion2binary.txt.

Adjust --th (threshold) in the command above to balance between sensitivity and specificity (details below).

<details>
  <summary>Details about mask post-processing and the converseion to binary labels</summary>

Post-processing details:

  1- We remove any tumor detection outside of the organ mask. E.g., we certify that all liver tumors in liver_tumor.nii.gz are inside the liver (liver.nii.gz).

  2- We apply [binary erosion](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html), using a 3x3x3 mm cube as the structuring element. This operation denoises the segmentation mask, removing detections that are smaller than the structuring element. Thus, binary erosion helps avoid false positives.

  3- After erosion, we calculate the total volume of the tumors in the tumor mask, in mm^3. If it is above a threshold parameter, the sample is considered positive for tumor. 
  
  Higher thresholds (th) and binary erosion reduce the number of false positives but may reduce the model capacity to detect very small tumors, and increase false negatives.

</details>
