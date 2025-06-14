# Installation

## Dataset

Please download these datasets and save to `datapath` (user-defined).

- 01 [AbdonmenAtlas 1.0](https://github.com/MrGiovanni/AbdomenAtlas)
- 02 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094)
- 03 [MSD-Pancreas](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 04 [KiTS](https://kits-challenge.org/kits23/#download-block)

It can be publicly available datasets (e.g., LiTS) or your private datasets. Currently, we only take data formatted in nii.gz. 
Download these datasets and save it to the datapath directory.
```bash
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/10_Decathlon/Task03_Liver.tar.gz?download=true # Task03_Liver.tar.gz (28.7 GB)
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/10_Decathlon/Task07_Pancreas.tar.gz?download=true # Task07_Pancreas.tar.gz (28.7 GB)
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/tree/main/05_KiTS.tar.gz?download=true # KiTS.tar.gz (28 GB)
```


## Dependency
The code is tested on `python 3.8, Pytorch 1.12`.

##### Setup environment

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
conda create -n difftumor python=3.8
source activate difftumor # or conda activate difftumor
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pip==24.0
pip install -r requirements.txt
```
