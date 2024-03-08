# Generalizable Tumor Synthesis

Early-stage tumors present similar imaging characteristics in computed tomography (CT), whether they are located in the liver, pancreas, or kidneys. Leveraging this observation, we develop a generative AI model on a few examples of annotated tumors in a specific organ, e.g., the liver (in purple). This AI model (in purple), trained exclusively on liver tumors, can directly create synthetic tumors in those organs where CT volumes of annotated tumors are relatively scarce, e.g., the pancreas (in cyan) and kidneys (in blue and green). By integrating synthetic tumors into extensive CT volumes of healthy organs—routinely collected in clinical settings—we can substantially augment the training set for tumor segmentation. This enhancement can also significantly improve the AI generalizability across CT volumes sourced from diverse hospitals and patient demographics.

<p align="center"><img width="100%" src="figures/fig_teaser.png" /></p>

## Paper

<b>Towards Generalizable Tumor Synthesis</b> <br/>
[Qi Chen](https://scholar.google.com/citations?user=4Q5gs2MAAAAJ&hl=en)<sup>1,2</sup>, [Xiaoxi Chen](https://www.linkedin.com/in/xiaoxichensjtu/?originalSubdomain=cn)<sup>3</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, [Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong/)<sup>2</sup>, [Wei Chen](https://weichen582.github.io/)<sup>1</sup> and [Zongwei Zhou](https://www.zongweiz.com/)<sup>1,*</sup> <br/>
<sup>1 </sup>Johns Hopkins University,  <br/>
<sup>2 </sup>University of Science and Technology of China,  <br/>
<sup>3 </sup>Shanghai Jiao Tong University  <br/>
CVPR, 2024 <br/>
[paper](https://www.cs.jhu.edu/~alanlab/Pubs24/chen2024towards.pdf 
) | [code](https://github.com/MrGiovanni/DiffTumor) | [huggingface](https://huggingface.co/MrGiovanni/DiffTumor)

**We have documented common questions for the paper in [Frequently Asked Questions (FAQ)](documents/FAQ.md).**

**We have summarized publications related to tumor synthesis in [Awesome Synthetic Tumors](https://github.com/MrGiovanni/SyntheticTumors/blob/main/AWESOME.md) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re).**

## Visual Turing Test
**We have released videos for [Visual Turing Test](https://www.dropbox.com/scl/fo/mgw953fdysvto2y6bog1c/h?rlkey=pqe9ar4xzo52ggm5v8v43275v&dl=0).** Check to see if you could tell which is real tumor and which is synthetic tumor. 


## 0. Installation

```bash
git clone https://github.com/MrGiovanni/DiffTumor.git
cd DiffTumor
```

See [installation instructions](documents/INSTALL.md) to create an environment and obtain requirements.

## 1. Train Autoencoder Model
```
cd AutoencoderModel
gpu_num=1
datapath=/mnt/ccvl15/chongyu/
cache_rate=0.05
batch_size=4
python train.py dataset.data_root_path=$datapath dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size model.gpus=$gpu_num
```
We offer the pre-trained checkpoint of Autoencoder Model, which was trained on a combination of 17 publicly available CT datasets and 9,262 CT scans (see details in [SuPreM](https://github.com/MrGiovanni/SuPreM)).
```
wget https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/AutoencoderModel/AutoencoderModel.ckpt
```

## 2. Train Diffusion Model
Diffusoin Model need to be trained on tumor data with mask annotations. It can be publicly available datasets (e.g., LiTS, MSD-Pancreas, KiTS) or your private datasets. If you want to train a Diffusion Model that synthesize early tumors, you need to first process the data to filter out the early tumor data.

**Dataset Pre-Process**  
1. Download the dataset according to the dataset link.  
2. Modify `data_dir` and `tumor_save_dir` in [data_transfer.py](https://github.com/MrGiovanni/DiffTumor/blob/main/data_transfer.py).
3. `python -W ignore data_transfer.py`

Then we take the example of training Diffusion Model with early-stage liver tumors.
```
cd DiffusionModel
fold=0
python train.py dataset.name=liver_tumor_train dataset.fold=$fold dataset.dataset_list=['liver_tumor_data_early_fold'] dataset.uniform_sample=False model.results_folder_postfix="fold'$fold'_tumor_96_t4"  
```

We offer the pre-trained checkpoints of Diffusion Model, which were trained for early-stage and mid-/late- stage tumors for liver, pancreas and kidney, respectively.

| Tumor | Type | Download |
| ----  | ----  | ----     |
| liver  | early | [link]() |
| pancreas  | early | [link]() |
| kidney  | early | [link]() |
| liver  | mid&late | [link]() |
| pancreas  | mid&late | [link]() |
| kidney  | mid&late | [link]() |

## 3. Train Segmentation Model
```
cd SegmentationModel

healthy_datapath=/data/jliang12/zzhou82/datasets/PublicAbdominalData/
datapath=/data/jliang12/zzhou82/datasets/PublicAbdominalData/
cache_rate=1.0
batch_size=12
val_every=50
workers=12
organ=liver
fold=0

# U-Net
backbone=unet
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir="cross_eval/'$organ'_aug_data_fold/"
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir

# nnU-Net
backbone=nnunet
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir="cross_eval/'$organ'_aug_data_fold/"
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir

# Swin-UNETR
backbone=swinunetr
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir="cross_eval/'$organ'_aug_data_fold/"
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir

```

We offer the pre-trained checkpoints of Segmentation Model (U-Net, nnU-Net and Swin UNETR), which were trained on real and synthetic tumors for liver, pancreas and kidney.

<details>
<summary style="margin-left: 25px;">U-Net</summary>
<div style="margin-left: 25px;">

| Tumor | Download |
| ----  | ----     |
| liver  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/unet_synt_liver_tumors.pt) |
| pancreas  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/unet_synt_pancreas_tumors.pt) |
| kidney  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/unet_synt_kidney_tumors.pt) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">nnU-Net</summary>
<div style="margin-left: 25px;">

| Tumor | Download |
| ----  | ----     |
| liver  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/nnunet_synt_liver_tumors.pt) |
| pancreas  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/nnunet_synt_pancreas_tumors.pt) |
| kidney  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/nnunet_synt_kidney_tumors.pt) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Swin UNETR</summary>
<div style="margin-left: 25px;">

| Tumor | Download |
| ----  | ----     |
| liver  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/swinunetr_synt_liver_tumors.pt) |
| pancreas  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/swinunetr_synt_pancreas_tumors.pt) |
| kidney  | [link](https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/swinunetr_synt_kidney_tumors.pt) |

</div>
</details>

## 4. Evaluation

```
cd SegmentationModel
datapath=/mnt/ccvl15/zzhou82/PublicAbdominalData/
organ=liver
fold=0
datafold_dir="cross_eval/'$organ'_aug_data_fold/"

# U-Net
python -W ignore validation.py --model=unet --data_root $datapath --datafold_dir $datafold_dir --tumor_type tumor --organ_type $organ --fold $fold --log_dir $organ/$organ.fold$fold.unet --save_dir out/$organ/$organ.fold$fold.unet

# nnU-Net
python -W ignore validation.py --model=nnunet --data_root $datapath --datafold_dir $datafold_dir --tumor_type tumor --organ_type $organ --fold $fold --log_dir $organ/$organ.fold$fold.unet --save_dir out/$organ/$organ.fold$fold.unet

# Swin-UNETR
python -W ignore validation.py --model=swinunet --data_root $datapath --datafold_dir $datafold_dir --tumor_type tumor --organ_type $organ --fold $fold --log_dir $organ/$organ.fold$fold.unet --save_dir out/$organ/$organ.fold$fold.unet

```

## TODO

- [x] Upload the paper to arxiv
- [x] Upload the videos about Visual Turing Test
- [ ] Release the checkpoints of Autoencoder Model and Diffusion Model
- [ ] Release the checkpoints of Segmentation Model
- [ ] Build a large-scale, semi-synthetic dataset for abdominal tumor segmentation in solid and tubular organs (in progress)

## Citation

```
@misc{chen2024generalizable,
      title={Towards Generalizable Tumor Synthesis}, 
      author={Qi Chen and Xiaoxi Chen and Haorui Song and Zhiwei Xiong and Alan Yuille and Chen Wei and Zongwei Zhou},
      year={2024},
      eprint={2402.19470},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. The segmentation backbone is based on [Swin UNETR](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb); we appreciate the effort of the [MONAI Team](https://monai.io/) to provide and maintain open-source code to the community. We thank Camille Torrico and Alexa Delaney for improving the writing of this paper. Paper content is covered by patents pending.
