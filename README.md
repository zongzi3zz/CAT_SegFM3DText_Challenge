# CAT Code—CVPR 2025: Foundation Models for Text-guided 3D Biomedical Image Segmentation

  <p align="center">
    <a href='https://arxiv.org/abs/2406.07085'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/zongzi3zz/CAT/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
    <a href='https://youtu.be/pLiBnWpk5iY'>
      <img src='https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=YouTube' alt='Video'>
    </a>
  </p>
<br />

## 🎉 News
- **\[2024/09\]** CAT is accepted to NeurIPS 2024!

## 🛠️ Quick Start

### Installation

- It is recommended to build a Python-3.9 virtual environment using conda

  ```bash
  git clone https://github.com/zongzi3zz/CAT_SegFM3DText_Challenge.git
  cd CAT_SegFM3DText_Challenge
  conda env create -f environment.yml

### Dataset Preparation
- Please download dataset from [the competition website](https://www.codabench.org/competitions/5651/) and use `preprocess_data/npz2nii.py` to convert npz files into organized nii files and transform valid labels with `preprocess_data/trans_mask.py`
### Dataset Pre-Process
1. Modify [ORGAN_DATASET_DIR]  
2. `python -W ignore label_transfer_SegFM.py`
3. The example of data configure for training and evaluation can be seen in [datalist](https://github.com/zongzi3zz/CAT/tree/main/datalist)
4. The process for processing 10% data takes approximately 3 hours.
### Prompt Feats
We provide the prompt feats in [BaiduNetdisk](https://pan.baidu.com/s/1N_f58HGNRVWAM7vccZ6rLg) (code: `3ern`) and [GoogleDrive](https://drive.google.com/drive/folders/11noyz1l6y6sfi4yzSrhwvk60hPmz6qtB?usp=share_link).
### Train & Evaluation
The entire training process takes approximately 3 days using 8×A100 GPUs.
- **Train Pipeline**: 
  Set the parameter [`data_root`]() and run:
  ```shell
  bash scripts/train.sh
  ```
- **Evaluation**
  ```shell
  bash scripts/test.sh
  ```
- **Evaluation**
  ```shell
  python inference.py --single_infer_path npz_path
  ```

### Docker Image
The docker image can be found in [BaiduNetdisk](https://pan.baidu.com/s/1j_W-t5Txh21WxNwGtZKdBw)(code: `sc9d`) and [GoogleDrive](https://drive.google.com/file/d/1ixX9w7db_21q7xyv_VYPlKpfAuO1qaJ8/view?usp=sharing).
You can run the doceker with:
```shell
docker container run --gpus "device=0" -m 24G --name cat_segfm3dtext_challenge --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ cat_segfm3dtext_challenge:latest /bin/bash -c "sh predict.sh"
```

## Citation
If you find CAT useful, please cite using this BibTeX:
```bibtex
@article{huang2024cat,
  title={CAT: Coordinating Anatomical-Textual Prompts for Multi-Organ and Tumor Segmentation},
  author={Huang, Zhongzhen and Jiang, Yankai and Zhang, Rongzhao and Zhang, Shaoting and Zhang, Xiaofan},
  journal={arXiv preprint arXiv:2406.07085},
  year={2024}
}
```
