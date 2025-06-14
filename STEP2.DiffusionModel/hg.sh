#!/bin/bash
#SBATCH --job-name=diffusion_model

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -p general
#SBATCH -t 7-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

module load mamba/latest # only for Sol

# mamba create -n difftumor python=3.9
source activate difftumor
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install -r ../requirements.txt

vqgan_ckpt=pretrained_models/AutoencoderModel.ckpt
fold=0
datapath=/scratch/zzhou82/data/Task03_Liver/
tumorlabel=/scratch/zzhou82/data/preprocessed_labels/

python train.py dataset.name=liver_tumor_train dataset.fold=$fold dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['liver_tumor_data_early_fold'] dataset.uniform_sample=False model.results_folder_postfix="liver_early_tumor_fold"$fold""  model.vqgan_ckpt=$vqgan_ckpt

# sbatch --error=logs/diffusion_model.out --output=logs/diffusion_model.out hg.sh
