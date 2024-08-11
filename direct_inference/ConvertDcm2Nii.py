import os
import shutil
from pathlib import Path
import dicom2nifti
import argparse
import logging

# Function to clear log files at the start
def clear_logs():
    with open('conversion.log', 'w'):
        pass
    with open('conversion_errors.txt', 'w'):
        pass

# Clear logs at the start
clear_logs()

# Set up logging for successful conversions
logging.basicConfig(filename='conversion.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Set up a separate logger for errors
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('conversion_errors.txt')
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
error_logger.addHandler(error_handler)

flag=False

def dcm2nii(dcm_path, output_file):
    """
    Convert a directory of DICOM files to a NIfTI file.

    Parameters:
    - dcm_path: Path to the directory containing DICOM files
    - output_file: Path where the resulting NIfTI file will be saved
    """
    tmp = Path('temp_dicom_conversion')
    
    # Remove the temporary directory if it already exists
    if tmp.exists() and tmp.is_dir():
        shutil.rmtree(tmp)
    
    tmp.mkdir(parents=True, exist_ok=True)
    
    try:
        dicom2nifti.convert_directory(dcm_path, str(tmp), compression=True, reorient=True)
        nii_files = list(tmp.glob('*nii.gz'))
        if not nii_files:
            raise FileNotFoundError(f'No NIfTI file generated for {dcm_path}')
        nii = nii_files[0]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shutil.copy(nii, output_file)
        logging.info(f'Successfully converted {dcm_path} to {output_file}')
        return True
    except Exception as e:
        error_logger.error(f'{dcm_path}: {e}')
        if os.path.exists(os.path.dirname(output_file)):
            shutil.rmtree(os.path.dirname(output_file))
        return False
    finally:
        shutil.rmtree(tmp)

def convert_folder(source, destination, series_folders):
    """
    Convert DICOM files in the specified folder to a NIfTI file and save it to the destination.

    Parameters:
    - source: Path to the source folder
    - destination: Path to the destination folder
    - series_folders: List of series folders to be converted, sorted by number of files
    """
    for series_folder in series_folders:
        relative_path = os.path.relpath(series_folder, source)
        output_nifti_file = os.path.join(destination, relative_path, 'ct.nii.gz')
        os.makedirs(os.path.dirname(output_nifti_file), exist_ok=True)
        if dcm2nii(series_folder, output_nifti_file):
            return True
    return False

def find_series_with_max_files(series_folder):
    """
    Find all series in the given folder, sorted by the number of DICOM files.

    Parameters:
    - series_folder: Path to the folder containing series subfolders

    Returns:
    - A list of series folders sorted by the number of DICOM files (largest first)
    """
    series_list = []
    for series in os.listdir(series_folder):
        series_path = os.path.join(series_folder, series)
        if os.path.isdir(series_path):
            num_files = len([name for name in os.listdir(series_path) if name.endswith('.dcm')])
            series_list.append((num_files, series_path))
    series_list.sort(reverse=True, key=lambda x: x[0])
    return [series[1] for series in series_list]

def process_folders(source_folder, destination_folder):
    """
    Traverse through the source folder, identify Anon_Acc_# folders,
    find the series with the most DICOM files, and convert it to a NIfTI file.

    Parameters:
    - source_folder: Path to the source folder
    - destination_folder: Path to the destination folder
    """
    for anon_mrn in os.listdir(source_folder):
        anon_mrn_path = os.path.join(source_folder, anon_mrn)
        if os.path.isdir(anon_mrn_path):
            conversion_success = False
            for anon_acc in os.listdir(anon_mrn_path):
                anon_acc_path = os.path.join(anon_mrn_path, anon_acc)
                if os.path.isdir(anon_acc_path):
                    series_folders = find_series_with_max_files(anon_acc_path)
                    if series_folders:
                        if convert_folder(source_folder, destination_folder, series_folders):
                            conversion_success = True
            if not conversion_success:
                error_logger.error(f'No series in {anon_mrn_path} could be successfully converted.')

def main():
    """
    Main function to parse command-line arguments and start the conversion process.
    """
    parser = argparse.ArgumentParser(description="Convert DICOM series with the most files to NIfTI format.")
    parser.add_argument('source_folder', type=str, help='Path to the source folder containing DICOM files')
    parser.add_argument('destination_folder', type=str, help='Path to the destination folder to save NIfTI files')
    args = parser.parse_args()

    source_folder = args.source_folder
    destination_folder = args.destination_folder

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    process_folders(source_folder, destination_folder)

if __name__ == "__main__":
    main()
