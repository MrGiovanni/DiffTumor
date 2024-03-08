#!/bin/bash
#SBATCH --job-name=autoencoder_model

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -t 0-05:00:00
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

datapath=/data/jliang12/zzhou82/datasets/PublicAbdominalData/
cache_rate=0.05
batch_size=4

# single GPU
gpu_num=1
python -W ignore train.py dataset.data_root_path=$datapath dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size model.gpus=$gpu_num

# sbatch --error=logs/autoencoder_model.out --output=logs/autoencoder_model.out hg.sh
