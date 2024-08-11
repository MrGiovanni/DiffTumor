#!/bin/bash

# Get the path to destination_folder as input
destination_folder=$1

# Set the path for the output folder
output_folder=$2

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

export NCCL_DEBUG=INFO

# cache directories
export TORCH_HOME=cache
export TORCHVISION_HOME=cache

# Temporary directories for input and output
export temp_input_dir=tmp_inputs
export temp_output_dir=tmp_outputs
mkdir -p "$temp_input_dir"
mkdir -p "$temp_output_dir"
mkdir -p "$TORCH_HOME"
mkdir -p "$TORCHVISION_HOME"

# Function to copy and rename files for the segmentation model
prepare_input_file() {
    local input_file=$1
    local temp_case_file="$temp_input_dir/case00001.nii.gz"
    echo "Copying $input_file to $temp_case_file"
    cp "$input_file" "$temp_case_file"
    if [ $? -ne 0 ]; then
        echo "Failed to copy $input_file to $temp_case_file"
    else
        echo "Successfully copied $input_file to $temp_case_file"
    fi
}

# Function to run the segmentation model
run_segmentation() {
    echo "Running segmentation model on $temp_input_dir/case00001.nii.gz"
    singularity run --fakeroot --overlay overlay.img --nv -B $temp_input_dir:/workspace/inputs -B $temp_output_dir:/workspace/outputs difftumor_final.sif
    if [ $? -ne 0 ]; then
        echo "Segmentation model failed on $temp_input_dir/case00001.nii.gz"
        return 1
    else
        echo "Segmentation model succeeded on $temp_input_dir/case00001.nii.gz"
        return 0
    fi
}

# Function to move the results to the final output folder
move_results() {
    local series_folder=$1
    local output_series_folder="$output_folder/${series_folder#$destination_folder}"
    echo "Moving results to $output_series_folder"
    mkdir -p "$output_series_folder/predictions"
    if [ -f "$temp_output_dir/case00001/ct.nii.gz" ]; then
        echo "Found ct.nii.gz for $series_folder"
        cp "$temp_output_dir/case00001/ct.nii.gz" "$output_series_folder/"
        if [ $? -ne 0 ]; then
            echo "Failed to copy ct.nii.gz for $series_folder"
        else
            echo "Successfully copied ct.nii.gz for $series_folder"
        fi
    else
        echo "No ct.nii.gz file found for $series_folder"
    fi
    if [ -d "$temp_output_dir/case00001/predictions" ]; then
        echo "Found predictions folder for $series_folder"
        cp "$temp_output_dir/case00001/predictions/"* "$output_series_folder/predictions/"
        if [ $? -ne 0 ]; then
            echo "Failed to copy predictions for $series_folder"
        else
            echo "Successfully copied predictions for $series_folder"
        fi
    else
        echo "No predictions folder found for $series_folder"
    fi
}

# Clean up temporary directories
clean_temp_dirs() {
    echo "Cleaning temporary directories"
    rm -rf "$temp_input_dir"/*
    if [ $? -ne 0 ]; then
        echo "Failed to clean $temp_input_dir"
    else
        echo "Successfully cleaned $temp_input_dir"
    fi
    rm -rf "$temp_output_dir"/*
    if [ $? -ne 0 ]; then
        echo "Failed to clean $temp_output_dir"
    else
        echo "Successfully cleaned $temp_output_dir"
    fi
}

# Find all .nii.gz files and store them in an array
mapfile -t ct_files < <(find "$destination_folder" -type f -name "ct.nii.gz")

# Process each .nii.gz file in the array
for ct_file in "${ct_files[@]}"; do
    if [ -f "$ct_file" ]; then
        series_folder=$(dirname "$ct_file")
        echo "Processing $ct_file in $series_folder"
        prepare_input_file "$ct_file"
        run_segmentation
        if [ $? -ne 0 ]; then
            echo "Skipping $ct_file due to segmentation error"
            continue
        fi
        move_results "$series_folder"
        clean_temp_dirs
    fi
done

# Final cleanup
echo "Final cleanup"
rm -rf "$temp_input_dir"
rm -rf "$temp_output_dir"
