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

import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone#, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list
from detectron2.structures.masks import polygons_to_bitmask,BitMasks

__all__ = ["DiffusionInst"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


from detectron2.structures import Instances, ROIMasks


def clip_mask(Boxes,masks):
    boxes = Boxes.tensor.long()
    assert (len(boxes)==len(masks))
    m_out = []
    #import pdb;pdb.set_trace()
    k = torch.zeros(masks[0].size()).long().to(boxes.device)
    for i in range(len(masks)):
        #import pdb;pdb.set_trace()
        mask = masks[i]
        box = boxes[i]
        k[box[1]:box[3],box[0]:box[2]] = 1
        mask *= k
        m_out.append(mask)
        k *= 0
    return torch.stack(m_out)


# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mid_size, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    #import pdb;pdb.set_trace()
    output_boxes.clip(results.image_size)
    masks = results.pred_masks
    
    #masks = F.interpolate(masks.unsqueeze(1), size=new_size, mode='bilinear').squeeze(1)
    ################################################
    
    pred_global_masks = aligned_bilinear(masks.unsqueeze(1), 4)
    #import pdb;pdb.set_trace()
    pred_global_masks = pred_global_masks[:, :, :mid_size[0], :mid_size[1]]
    masks = F.interpolate(
                    pred_global_masks,
                    size=(new_size[0], new_size[1]),
                    mode='bilinear',
                    align_corners=False).squeeze(1)
    #################################################
    masks.gt_(0.5)
    #masks = masks.long()
    #import pdb;pdb.set_trace()
    masks = clip_mask(output_boxes,masks)
    #import pdb;pdb.set_trace()
    results.pred_masks = masks
    results = results[output_boxes.nonempty()]
    #import pdb;pdb.set_trace()
    #if results.has("pred_masks"):
        #if isinstance(results.pred_masks, ROIMasks):
        #    roi_masks = results.pred_masks
        #else:
            # pred_masks is a tensor of shape (N, 1, M, M)
        #    roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        #results.pred_masks = roi_masks.to_bitmasks(
        #    results.pred_boxes, output_height, output_width, mask_threshold
        #).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor
    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                           size=(oh, ow),
                           mode='bilinear',
                           align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode="replicate")
    return tensor[:, :, :oh - 1, :ow - 1]


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    num_instances = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances)
    return weight_splits, bias_splits

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@META_ARCH_REGISTRY.register()
class DiffusionInst(nn.Module):
    """
    Implement DiffusionInst
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionInst.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionInst.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionInst.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionInst.NUM_HEADS
        self.weight_nums = [64, 64, 8]
        self.bias_nums = [8, 8, 1]
        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionInst.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionInst.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.DiffusionInst.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionInst.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionInst.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionInst.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionInst.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionInst.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionInst.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionInst.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "mask"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        #import pdb;pdb.set_trace()
        outputs_class, outputs_coord,outputs_kernel,mask_feat = self.head(backbone_feats, x_boxes, t, None)
        
        #torch.Size([6, 1, 500, 80]), torch.Size([6, 1, 500, 153]), torch.Size([1, 8, 200, 304])
        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord,outputs_kernel,mask_feat

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        #import pdb;pdb.set_trace()
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        #tensor([ -1., 999.])
        times = list(reversed(times.int().tolist()))
        
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord,ensemble_kernel = [], [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            #import pdb;pdb.set_trace()
            preds, outputs_class, outputs_coord,outputs_kernel,mask_feat = self.model_predictions(backbone_feats, images_whwh, img, time_cond,self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                #true
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image, kernels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        outputs_kernel[-1], 
                                                                                        mask_feat,
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)
                ensemble_kernel.append(kernels_per_image)
        
        if self.use_ensemble and self.sampling_timesteps > 1:
            #import pdb;pdb.set_trace()
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            kernels_per_image = torch.cat(ensemble_kernel, dim=0)
            #import pdb;pdb.set_trace()
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                kernels_per_image = kernels_per_image[keep]
                
            num_instance = len(kernels_per_image)
            weights, biases = parse_dynamic_params(
                    kernels_per_image,
                    8,
                    self.weight_nums,
                    self.bias_nums)
            #import pdb;pdb.set_trace()
            mask_feat_head = mask_feat.repeat(1, num_instance, 1, 1)
            mask_logits = self.mask_heads_forward(
                    mask_feat_head, 
                    weights, 
                    biases, 
                    num_instance)
                #import pdb;pdb.set_trace()
            mask_logits = mask_logits.reshape(-1, 1,mask_feat.size(2), mask_feat.size(3)).squeeze(1).sigmoid()
            
            
            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.pred_masks = mask_logits
            results = [result]
        else:
            #import pdb;pdb.set_trace()
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            
            results = self.inference(box_cls, box_pred, outputs_kernel[-1], mask_feat,images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                #import pdb;pdb.set_trace()
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                mid_size = image_size
                r = detector_postprocess(results_per_image, height, width,mid_size)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        #import pdb;pdb.set_trace()
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]
            #import pdb;pdb.set_trace()
            outputs_class, outputs_coord,outputs_kernel,mask_feat = self.head(features, x_boxes, t, None)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_kernels': outputs_kernel[-1], 'mask_feat':mask_feat}

            if self.deep_supervision:
                 output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_kernels':c, 'mask_feat':mask_feat}
                                         for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1],outputs_kernel[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            #gt_masks = targets_per_image.gt_masks.tensor 
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            if len(targets_per_image) == 0:
                #import pdb;pdb.set_trace()
                bitmask = torch.zeros((h,w))#.to(self.device)
            else:
                bitmask = targets_per_image.gt_masks
            
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            #import pdb;pdb.set_trace()
             #BitMasks.from_polygon_masks(targets_per_image.gt_masks,h,w)
            target["masks"] = bitmask.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def mask_heads_forward(self, features, weights, biases, num_instances):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x,
                         w,
                         bias=b,
                         stride=1,
                         padding=0,
                         groups=num_instances)
            if i < n_layers - 1:
                x = F.relu(x)
        return x
    def inference(self, box_cls, box_pred, kernel, mask_feat, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes
            [1, 500, 153]
            [1, 8, 200, 304]

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        
        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
            #import pdb;pdb.set_trace()
            for i, (scores_per_image, box_pred_per_image, image_size,ker,mas) in enumerate(zip(
                    scores, box_pred, image_sizes,kernel, mask_feat
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                ker = ker.view(-1, 1, 153).repeat(1, self.num_classes, 1).view(-1, 153)
                #torch.Size([500, 4])
                
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                #torch.Size([40000, 4])
                ker = ker[topk_indices]
                box_pred_per_image = box_pred_per_image[topk_indices]
                #torch.Size([500, 4])
                
                
                if self.use_ensemble and self.sampling_timesteps > 1:
                    #import pdb;pdb.set_trace()
                    return box_pred_per_image, scores_per_image, labels_per_image, ker

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                    ker_per_image = ker[keep]
                
                num_instance = len(ker_per_image)
                weights, biases = parse_dynamic_params(
                    ker_per_image,
                    8,
                    self.weight_nums,
                    self.bias_nums)
                #import pdb;pdb.set_trace()
                mask_feat_head = mas.unsqueeze(0).repeat(1, num_instance, 1, 1)
                mask_logits = self.mask_heads_forward(
                    mask_feat_head, 
                    weights, 
                    biases, 
                    num_instance)
                #import pdb;pdb.set_trace()
                mask_logits = mask_logits.reshape(-1, 1, mas.size(1), mas.size(2)).squeeze(1).sigmoid()
                #mask_logits.gt_(0.5)
                #mask_logits = F.interpolate(mask_logits, size=image_size, mode='bilinear').squeeze(1)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                result.pred_masks = mask_logits
                results.append(result)

        else:
            import pdb;pdb.set_trace()
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
