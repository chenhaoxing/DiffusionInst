## DiffusionInst: Diffusion Model for Instance Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusioninst-diffusion-model-for-instance/instance-segmentation-on-lvis-v1-0-val)](https://paperswithcode.com/sota/instance-segmentation-on-lvis-v1-0-val?p=diffusioninst-diffusion-model-for-instance)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusioninst-diffusion-model-for-instance/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=diffusioninst-diffusion-model-for-instance)

**DiffusionInst is the first work of diffusion model for instance segmentation.**
![](figure/arch.jpeg)
We hope our work could serve as a simple yet effective baseline, which could inspire designing more efficient diffusion frameworks for challenging discriminative tasks.


> [**DiffusionInst: Diffusion Model for Instance Segmentation**](https://arxiv.org/abs/2212.02773)               
> [Zhangxuan Gu](https://scholar.google.com/citations?user=Wkp3s68AAAAJ&hl=zh-CN&oi=ao), [Haoxing Chen](https://scholar.google.com/citations?hl=zh-CN&pli=1&user=BnS7HzAAAAAJ), Zhuoer Xu, Jun Lan, Changhua Meng, [Weiqiang Wang](https://scholar.google.com/citations?hl=zh-CN&user=yZ5iffAAAAAJ) 
> *[arXiv 2212.02773](https://arxiv.org/abs/2212.02773)*  

## Todo list:
- [x] Release source code.
- [x] Hyper-paramters tuning.
- [ ] Adding directly filter denoising.


## Getting Started
The installation instruction and usage are in [Getting Started with DiffusionInst](GETTING_STARTED.md).

## Model Performance
Method | Mask AP (1 step) | Mask AP (4 step) 
--- |:---:|:---:
COCO-val-Res50 | 37.3| 37.5 
COCO-val-Res101 | 41.0| 41.1 
COCO-val-Swin-B| 46.6| 46.8
LVIS-Res50 | 22.3| - 
LVIS-Res101| 27.0| - 
LVIS-Swin-B| 36.0| - 
COCO-testdev-Res50 | 37.1| - 
COCO-testdev-Res101 | 41.5| -
COCO-testdev-Swin-B| 47.6| -

![](figure/visual.jpeg)

## Citing DiffusionInst

If you use DiffusionInst in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@article{DiffusionInst,
      title={DiffusionInst: Diffusion Model for Instance Segmentation},
      author={Gu, Zhangxuan and Chen, Haoxing and Xu, Zhuoer and Lan, Jun and Meng, Changhua and Wang, Weiqiang},
      journal={arXiv preprint arXiv:2212.02773},
      year={2022}
}
```
## Acknowledgement
Many thanks to the nice work of DiffusionDet @[ShoufaChen](https://github.com/ShoufaChen). Our codes and configs follow [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet).

## Contacts
Please feel free to contact us if you have any problems.

Email: [haoxingchen@smail.nju.edu.cn](haoxingchen@smail.nju.edu.cn) or [guzhangxuan.gzx@antgroup.com](guzhangxuan.gzx@antgroup.com)

