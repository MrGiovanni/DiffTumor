# Installation

## Dataset

please download these datasets and save to `<data-path>` (user-defined).

- 01 [AbdonmenAtlas](https://github.com/MrGiovanni/AbdomenAtlas)
- 02 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094)
- 03 [MSD-Pancreas](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 04 [KiTS](https://kits-challenge.org/kits23/#download-block)

```bash
wget https://www.dropbox.com/s/jnv74utwh99ikus/01_Multi-Atlas_Labeling.tar.gz # 01 Multi-Atlas_Labeling.tar.gz (1.53 GB)
wget https://www.dropbox.com/s/5yzdzb7el9r3o9i/02_TCIA_Pancreas-CT.tar.gz # 02 TCIA_Pancreas-CT.tar.gz (7.51 GB)
wget https://www.dropbox.com/s/lzrhirei2t2vuwg/03_CHAOS.tar.gz # 03 CHAOS.tar.gz (925.3 MB)
wget https://www.dropbox.com/s/2i19kuw7qewzo6q/04_LiTS.tar.gz # 04 LiTS.tar.gz (17.42 GB)
```


## Dependency
The code is tested on `python 3.8, Pytorch 1.12`.

##### Setup environment
```bash
conda create -n difftumor python=3.8
```

<details>
<summary style="margin-left: 25px;">[Optional] If You are using Agave/Sol</summary>
<div style="margin-left: 25px;">

```bash
module load anaconda3/5.3.0 # only for Agave

module load mamba/latest # only for Sol
mamba create -n difftumor python=3.8
```

</div>
</details>

```bash
source activate difftumor # or conda activate difftumor
```

#### Autoencoder 
```bash
conda create -n Autoencoder python=3.8
source activate Autoencoder (or conda activate Autoencoder)
cd AutoencoderModel
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install 'monai[nibabel]'
pip install monai==0.9.0
```

#### DiffusionModel
```bash
conda create -n DiffusionModel python=3.8
source activate DiffusionModel (or conda activate DiffusionModel)
cd DiffusionModel
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install wandb scikit-image nilearn hydra-core==1.2.0 omegaconf tensorboard
pip install protobuf==3.20.1 sk-video SimpleITK matplotlib torchio torchstat
pip install tensorboardX==2.4.1 h5py einops einops-exts rotary-embedding-torch imageio timm tqdm elasticdeform
pip install 'monai[nibabel]'==0.9
pip install monai==0.9.0
pip install -r requirements.txt
```

#### SegmentationModel
```bash
conda create -n SegmentationModel python=3.8
source activate SegmentationModel (or conda activate SegmentationModel)
cd SegmentationModel
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python wandb scikit-image nilearn hydra-core==1.2.0 omegaconf tensorboard
pip install protobuf==3.20.1 sk-video SimpleITK matplotlib torchio torchstat
pip install tensorboardX==2.4.1 h5py einops einops-exts rotary-embedding-torch imageio timm tqdm elasticdeform
pip install 'monai[nibabel]'==0.9
pip install monai==0.9.0
pip install -r requirements.txt
```
