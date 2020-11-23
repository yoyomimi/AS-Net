import mmcv
import numpy as np
import os
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

from scipy.spatial.distance import cdist

from libs.models.backbone import build_backbone
from libs.models.transformer import build_transformer
from libs.utils import box_ops
from libs.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                             accuracy, get_world_size, interpolate,
                             is_dist_avail_and_initialized)


class ASNet(nn.Module):
    """ This is the HOI Transformer module that performs HOI detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 num_classes=dict(
                     obj_labels=91,
                     rel_labels=117
                 ), 
                 num_queries=100,
                 rel_num_queries=16,
                 id_emb_dim=8,
                 aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: dict of number of sub clses, obj clses and relation clses, 
                         omitting the special no-object category
                         keys: ["obj_labels", "rel_labels"]
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.rel_num_queries = rel_num_queries
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # instance branch
        self.class_embed = nn.Linear(hidden_dim, num_classes['obj_labels'] + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # interaction branch
        self.rel_query_embed = nn.Embedding(rel_num_queries, hidden_dim)
        self.rel_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.rel_class_embed = nn.Linear(hidden_dim, num_classes['rel_labels'])
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # embedding
        self.rel_id_embed = MLP(hidden_dim, hidden_dim, id_emb_dim, 3)
        self.rel_src_embed = MLP(hidden_dim, hidden_dim, id_emb_dim, 3)
        self.rel_dst_embed = MLP(hidden_dim, hidden_dim, id_emb_dim, 3)
        # aux loss of each decoder layer
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # backbone
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        input_src = self.input_proj(src)
        
        # encoder + two parellel decoders
        rel_hs, hs = self.transformer(input_src, mask, self.query_embed.weight,
            self.rel_query_embed.weight, pos[-1])[:2]
        rel_hs = rel_hs[-1].unsqueeze(0)
        hs = hs[-1].unsqueeze(0)

        # FFN on top of the instance decoder
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        id_emb = self.rel_id_embed(hs)

        # FFN on top of the interaction decoder
        outputs_rel_class = self.rel_class_embed(rel_hs)
        outputs_rel_coord = self.rel_bbox_embed(rel_hs).sigmoid()
        src_emb = self.rel_src_embed(rel_hs)
        dst_emb = self.rel_dst_embed(rel_hs)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'id_emb': id_emb[-1]}
        rel_out = {'pred_logits': outputs_rel_class[-1], 'pred_boxes': outputs_rel_coord[-1],
                   'src_emb': src_emb[-1], 'dst_emb': dst_emb[-1]}
        output = {
            'pred_det': out,
            'pred_rel': rel_out
        }
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,
                outputs_rel_class, outputs_rel_coord, id_emb, src_emb, dst_emb)
        
        return output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_rel_class,
                      outputs_rel_coord, id_emb, src_emb, dst_emb):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        aux_output = []
        for idx in range(len(outputs_class)):
            out = {'pred_logits': outputs_class[idx], 'pred_boxes': outputs_coord[idx],
                   'id_emb': id_emb[idx]}
            if idx < len(outputs_rel_class):
                rel_out = {'pred_logits': outputs_rel_class[idx], 'pred_boxes': outputs_rel_coord[idx],
                           'src_emb': src_emb[idx], 'dst_emb': dst_emb[idx]}
            else:
                rel_out = None
            aux_output.append({
                'pred_det': out,
                'pred_rel': rel_out 
            })
        return aux_output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self,
                 rel_array_path,
                 use_emb=False):
        super().__init__()
        # use semantic embedding in the matching or not
        self.use_emb = use_emb
        # rel array to remove non-exist hoi categories in training
        self.rel_array_path = rel_array_path
        
    def get_matching_scores(self, s_cetr, o_cetr, s_scores, o_scores, rel_vec,
                            s_emb, o_emb, src_emb, dst_emb): 
        rel_s_centr = rel_vec[..., :2].unsqueeze(-1).repeat(1, 1, s_cetr.shape[0])
        rel_o_centr = rel_vec[..., 2:].unsqueeze(-1).repeat(1, 1, o_cetr.shape[0])
        s_cetr = s_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
        s_scores = s_scores.repeat(rel_vec.shape[0], 1)
        o_cetr = o_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
        o_scores = o_scores.repeat(rel_vec.shape[0], 1)
        dist_s_x = abs(rel_s_centr[..., 0, :] - s_cetr[..., 0])
        dist_s_y = abs(rel_s_centr[..., 1, :] - s_cetr[..., 1])
        dist_o_x = abs(rel_o_centr[..., 0, :] - o_cetr[..., 0])
        dist_o_y = abs(rel_o_centr[..., 1, :] - o_cetr[..., 1])
        dist_s = (1.0 / (dist_s_x + 1.0)) * (1.0 / (dist_s_y + 1.0))
        dist_o = (1.0 / (dist_o_x + 1.0)) * (1.0 / (dist_o_y + 1.0))
        # involving emb into the matching strategy
        if self.use_emb is True:
            s_emb_np = s_emb.data.cpu().numpy()
            o_emb_np = o_emb.data.cpu().numpy()
            src_emb_np = src_emb.data.cpu().numpy()
            dst_emb_np = dst_emb.data.cpu().numpy()
            dist_s_emb = torch.from_numpy(cdist(src_emb_np, s_emb_np, metric='euclidean')).to(rel_vec.device)
            dist_o_emb = torch.from_numpy(cdist(dst_emb_np, o_emb_np, metric='euclidean')).to(rel_vec.device)
            dist_s_emb = 1. / (dist_s_emb + 1.0)
            dist_o_emb = 1. / (dist_o_emb + 1.0)
            dist_s *= dist_s_emb
            dist_o *= dist_o_emb
        dist_s = dist_s * s_scores
        dist_o = dist_o * o_scores
        return dist_s, dist_o

    @torch.no_grad()
    def forward(self, outputs_dict, file_name, target_sizes,
                rel_topk=20, sub_cls=1):
        """ Perform the matching of postprocess to generate final predicted HOI triplets
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        outputs = outputs_dict['pred_det']
        # '(bs, num_queries,) bs=1
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        id_emb = outputs['id_emb'].flatten(0, 1)
        rel_outputs = outputs_dict['pred_rel']
        rel_out_logits, rel_out_bbox = rel_outputs['pred_logits'], \
            rel_outputs['pred_boxes']
        src_emb, dst_emb = rel_outputs['src_emb'].flatten(0, 1), \
            rel_outputs['dst_emb'].flatten(0, 1)
        assert len(out_logits) == len(target_sizes) == len(rel_out_logits) \
                == len(rel_out_bbox)
        assert target_sizes.shape[1] == 2
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        # parse instance detection results
        out_bbox = out_bbox * scale_fct[:, None, :]
        out_bbox_flat = out_bbox.flatten(0, 1)
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        labels_flat = labels.flatten(0, 1) # '(bs * num_queries, )
        scores_flat = scores.flatten(0, 1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox_flat)
        s_idx = torch.where(labels_flat==sub_cls)[0]
        o_idx = torch.arange(0, len(labels_flat)).long()
        # no detected human or object instances
        if len(s_idx) == 0 or len(o_idx) == 0:
            pred_out = {
                'file_name': file_name,
                'hoi_prediction': [],
                'predictions': []
            }
            return pred_out
        s_cetr = box_ops.box_xyxy_to_cxcywh(boxes[s_idx])[..., :2]
        o_cetr = box_ops.box_xyxy_to_cxcywh(boxes[o_idx])[..., :2]
        s_boxes, s_clses, s_scores = boxes[s_idx], labels_flat[s_idx], scores_flat[s_idx]
        o_boxes, o_clses, o_scores = boxes[o_idx], labels_flat[o_idx], scores_flat[o_idx]
        s_emb, o_emb = id_emb[s_idx], id_emb[o_idx]

        # parse interaction detection results
        rel_prob = rel_out_logits.sigmoid()
        topk = rel_prob.shape[-1]
        rel_scores = rel_prob.flatten(0, 1)
        hoi_labels  = torch.arange(0, topk).repeat(rel_scores.shape[0], 1).to(
            rel_prob.device) + 1
        rel_vec = rel_out_bbox * scale_fct[:, None, :]
        rel_vec_flat = rel_vec.flatten(0, 1)

        # matching distance in post-processing
        dist_s, dist_o = self.get_matching_scores(s_cetr, o_cetr, s_scores,
            o_scores, rel_vec_flat, s_emb, o_emb, src_emb, dst_emb)
        rel_s_scores, rel_s_ids = torch.max(dist_s, dim=-1)
        rel_o_scores, rel_o_ids = torch.max(dist_o, dim=-1)
        hoi_scores = rel_scores * s_scores[rel_s_ids].unsqueeze(-1) * \
            o_scores[rel_o_ids].unsqueeze(-1)

        # exclude non-exist hoi categories of training
        rel_array = torch.from_numpy(np.load(self.rel_array_path)).to(hoi_scores.device)
        valid_hoi_mask = rel_array[o_clses[rel_o_ids], 1:]
        hoi_scores = (valid_hoi_mask * hoi_scores).reshape(-1, 1)
        hoi_labels = hoi_labels.reshape(-1, 1)
        rel_s_ids = rel_s_ids.unsqueeze(-1).repeat(1, topk).reshape(-1, 1)
        rel_o_ids = rel_o_ids.unsqueeze(-1).repeat(1, topk).reshape(-1, 1)
        hoi_triplet = (torch.cat((rel_s_ids.float(), rel_o_ids.float(), hoi_labels.float(),
            hoi_scores), 1)).cpu().numpy()
        hoi_triplet = hoi_triplet[hoi_triplet[..., -1]>0.0]

        # remove repeated triplets
        hoi_triplet = hoi_triplet[np.argsort(-hoi_triplet[:,-1])]
        _, hoi_id = np.unique(hoi_triplet[:, [0, 1, 2]], axis=0, return_index=True)
        rel_triplet = hoi_triplet[hoi_id]
        rel_triplet = rel_triplet[np.argsort(-rel_triplet[:,-1])]

        # save topk hoi triplets
        rel_topk = min(rel_topk, len(rel_triplet))
        rel_triplet = rel_triplet[:rel_topk]
        hoi_labels, hoi_scores = rel_triplet[..., 2], rel_triplet[..., 3]
        rel_s_ids, rel_o_ids = np.array(rel_triplet[..., 0], dtype=np.int64), np.array(rel_triplet[..., 1], dtype=np.int64)
        sub_boxes, obj_boxes = s_boxes.cpu().numpy()[rel_s_ids], o_boxes.cpu().numpy()[rel_o_ids]
        sub_clses, obj_clses = s_clses.cpu().numpy()[rel_s_ids], o_clses.cpu().numpy()[rel_o_ids]
        sub_scores, obj_scores = s_scores.cpu().numpy()[rel_s_ids], o_scores.cpu().numpy()[rel_o_ids]
        self.end_time = time.time()
        
        # wtite to files
        pred_out = {}
        pred_out['file_name'] = file_name
        pred_out['hoi_prediction'] = []
        num_rel = len(hoi_labels)
        for i in range(num_rel):
            sid = i
            oid = i + num_rel
            hoi_dict = {
                'subject_id': sid,
                'object_id': oid,
                'category_id': hoi_labels[i],
                'score': hoi_scores[i]
            }
            pred_out['hoi_prediction'].append(hoi_dict)
        pred_out['predictions'] = []
        for i in range(num_rel):
            det_dict = {
                'bbox': sub_boxes[i],
		        'category_id': sub_clses[i],
                'score': sub_scores[i]
            }
            pred_out['predictions'].append(det_dict)
        for i in range(num_rel):
            det_dict = {
                'bbox': obj_boxes[i],
		        'category_id': obj_clses[i],
                'score': obj_scores[i]
            }
            pred_out['predictions'].append(det_dict)
        return pred_out 


def build_model(cfg, device):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    num_classes=dict(
        obj_labels=cfg.DATASET.OBJ_NUM_CLASSES,
        rel_labels=cfg.DATASET.REL_NUM_CLASSES
    )
    model = ASNet(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        rel_num_queries=cfg.TRANSFORMER.REL_NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
    )
    matcher = None
    criterion = None
    postprocessors = PostProcess(cfg.TEST.REL_ARRAY_PATH, cfg.TEST.USE_EMB)
    return model, criterion, postprocessors