#!/bin/bash
#SBATCH --job-name=install

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -t 0-00:30:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

module load mamba/latest # only for Sol

# mamba create -n difftumor python=3.9
source activate difftumor

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# sbatch --error=logs/install.out --output=logs/install.out hg.sh