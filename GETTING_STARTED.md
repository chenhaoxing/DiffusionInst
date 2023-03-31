## Getting Started with DiffusionInst



### Installation

The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN), and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
Thanks very much.

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install Detectron2 following https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation.

2. Prepare datasets
```
mkdir -p datasets/coco
mkdir -p datasets/lvis

ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017

ln -s /path_to_lvis_dataset/lvis_v1_train.json datasets/lvis/lvis_v1_train.json
ln -s /path_to_lvis_dataset/lvis_v1_val.json datasets/lvis/lvis_v1_val.json
```

3. Prepare pretrain models

DiffusionInst uses three backbones including ResNet-50, ResNet-101 and Swin-Base. The pretrained ResNet-50 model can be downloaded automatically by Detectron2. 

[DiffusionDet](https://github.com/ShoufaChen/DiffusionDet) provide pretrained [ResNet-101](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/torchvision-R-101.pkl) and [Swin-Base](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/swin_base_patch4_window7_224_22k.pkl) which are compatible with Detectron2. Please download them to `DiffusionInst_ROOT/models/` before training.

1. Train DiffusionInst
```
python train_net.py --num-gpus 8 \
    --config-file configs/diffinst.coco.res50.yaml
```

1. Evaluate DiffusionInst
```
python train_net.py --num-gpus 8 \
    --config-file configs/diffinst.coco.res50.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

* Evaluate with arbitrary number (e.g 300) of boxes by setting `MODEL.DiffusionInst.NUM_PROPOSALS 300`.
* Evaluate with 4 refinement steps by setting `MODEL.DiffusionInst.SAMPLE_STEP 4`.


### Inference Demo with Pre-trained Models
We provide a command line tool to run a simple demo following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/demo#detectron2-demo).

```bash
python demo.py --config-file configs/diffinst.coco.res50.yaml \
    --input image.jpg --opts MODEL.WEIGHTS diffinst_coco_res50.pth
```

We need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.
