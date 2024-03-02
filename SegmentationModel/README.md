#### Dependency
The code is tested on `python 3.8, Pytorch 1.12`.
```bash
conda create -n tumordiff python=3.8
conda activate tumordiff

git clone https://github.com/qic999/TumorDiff4Seg.git
cd TumorDiff4Seg

cd TumorGeneration/
wget https://www.dropbox.com/scl/fi/gsg592onb0380v3ymy106/model_weight.tar.gz?rlkey=e9nilu2215founk7z15it9y7p
mv model_weight.tar.gz?rlkey=e9nilu2215founk7z15it9y7p model_weight.tar.gz
tar -xzvf model_weight.tar.gz
rm model_weight.tar.gz
cd ../

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python wandb scikit-image nilearn hydra-core==1.2.0 omegaconf tensorboard
pip install protobuf==3.20.1 sk-video SimpleITK matplotlib torchio torchstat
pip install tensorboardX==2.4.1 h5py einops einops-exts rotary-embedding-torch imageio timm tqdm elasticdeform
pip install 'monai[nibabel]'==0.9
pip install monai==0.9.0
pip install -r requirements.txt
```



#### Traning
```python
healthy_datapath=/scratch/zzhou82/data/NeurIPS-2023/
datapath=/data/jliang12/zzhou82/datasets/PublicAbdominalData/ 

cache_rate=0.05 # total 2053 CT scans
organ=$1
fold=$2
backbone=$3
logdir="runs/"+$1.$2.$3

python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --max_epochs 2000 --val_every 50 --batch_size=2 --save_checkpoint --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fg_thresh 30 --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir cross_eval/liver_aug_data_fold_scaleup/

```
