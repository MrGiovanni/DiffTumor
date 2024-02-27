#### Dependency
The code is tested on `python 3.8, Pytorch 1.12`.
```bash
conda create -n tumordiff python=3.8
conda activate tumordiff

git clone https://github.com/qic999/TumorDiffusion
cd TumorDiffusion
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install 'monai[nibabel]'
pip install monai==0.9.0
```

#### Traning
##### Reconstruction model
single gpu training
```python
datapath=/mnt/ccvl15/chongyu/
cache_rate=0.05
batch_size=6
CUDA_VISIBLE_DEVICES=0 python train.py dataset.data_root_path=$datapath dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size
```
multiple gpu training
```python
gpu_num=4
datapath=/mnt/ccvl15/chongyu/
cache_rate=0.05
batch_size=6
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py dataset.data_root_path=$datapath dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size model.gpus=$gpu_num
```
resume need add three parameter: resume, resume_version, resume_from_checkpoint  
resume_version is the version that is needed to resume.  
single gpu resume
```python
datapath=/mnt/ccvl15/chongyu/
cache_rate=0.05
batch_size=6
resume=True
resume_version=0
resume_checkpoint="checkpoints/vq_gan/synt/8k_96_d4_steplr/lightning_logs/version_0/checkpoints/epoch\=11-step\=30000-train/recon_loss\=0.21.ckpt"
CUDA_VISIBLE_DEVICES=0 python train.py dataset.data_root_path=$datapath dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size model.resume=$resume model.resume_version=$resume_version "model.resume_from_checkpoint=$resume_checkpoint"
```
multiple gpu resume
```python
gpu_num=4
datapath=/mnt/ccvl15/chongyu/
cache_rate=0.05
batch_size=6
resume=True
resume_version=0
resume_checkpoint="checkpoints/vq_gan/synt/8k_96_d4_steplr/lightning_logs/version_0/checkpoints/epoch\=11-step\=30000-train/recon_loss\=0.21.ckpt"
CUDA_VISIBLE_DEVICES=2,4,5,6 python train.py model.gpus=$gpu_num dataset.data_root_path=$datapath dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size model.resume=$resume model.resume_version=$resume_version "model.resume_from_checkpoint=$resume_checkpoint"
```