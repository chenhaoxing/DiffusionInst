"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

Modified by Zhangxuan Gu, Haoxing Chen
Date: Nov 30, 2022
Contact: {guzhangxuan.gzx, chenhaoxing.chx}@antgroup.com
"""

from detectron2.config import CfgNode as CN

def add_diffusioninst_config(cfg):
    """
    Add config for DiffusionInst
    """
    cfg.MODEL.DiffusionInst = CN()
    cfg.MODEL.DiffusionInst.NUM_CLASSES = 80
    cfg.MODEL.DiffusionInst.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionInst.NHEADS = 8
    cfg.MODEL.DiffusionInst.DROPOUT = 0.0
    cfg.MODEL.DiffusionInst.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionInst.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionInst.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionInst.NUM_CLS = 1
    cfg.MODEL.DiffusionInst.NUM_REG = 3
    cfg.MODEL.DiffusionInst.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionInst.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionInst.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionInst.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionInst.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionInst.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionInst.USE_FOCAL = True
    cfg.MODEL.DiffusionInst.USE_FED_LOSS = False
    cfg.MODEL.DiffusionInst.ALPHA = 0.25
    cfg.MODEL.DiffusionInst.GAMMA = 2.0
    cfg.MODEL.DiffusionInst.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionInst.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionInst.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionInst.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionInst.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
