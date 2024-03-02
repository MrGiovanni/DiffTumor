# Installation

## Dataset

please download these datasets and save to `<data-path>` (user-defined).

- 01 [AbdonmenAtlas](https://github.com/MrGiovanni/AbdomenAtlas)
- 02 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094)
- 03 [MSD-Pancreas](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 04 [KiTS](https://kits-challenge.org/kits23/#download-block)

It can be publicly available datasets (e.g., LiTS) or your private datasets. Currently, we only take data formatted in nii.gz. 
Download these datasets and save it to the datapath directory.
```bash
wget https://www.dropbox.com/scl/fi/890cavm8n3pjyyy5df2lk/AbdomenAtlasMini1.0.tar.gz?rlkey=8d53plsdojlf9hjqqddbyfeb0 # 01 AbdonmenAtlas.tar.gz (300+ GB)
wget https://www.dropbox.com/s/2i19kuw7qewzo6q/04_LiTS.tar.gz # 02 LiTS.tar.gz (17.42 GB)
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
pip install -r requirements.txt
```
