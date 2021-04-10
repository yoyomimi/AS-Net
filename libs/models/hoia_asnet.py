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
from libs.models.matcher import build_matcher
from libs.models.transformer import build_transformer
from libs.utils import box_ops
from libs.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                             accuracy, get_world_size, interpolate,
                             is_dist_avail_and_initialized)


class ASNet_HOIA(nn.Module):
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
        self.rel_det_class_embed = nn.Linear(hidden_dim, num_classes['obj_labels'] + 1)
        self.rel_det_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
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
        outputs_class = self.rel_det_class_embed(hs)
        outputs_coord = self.rel_det_bbox_embed(hs).sigmoid()
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


class SetCriterion(nn.Module):
    """ This class computes the loss for HOI Transformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  
                 matcher, 
                 losses,
                 weight_dict,
                 eos_coef,
                 rel_eos_coef=0.1,
                 num_classes=dict(
                     obj_labels=90,
                     rel_labels=117
                 ),
                 neg_act_id=0):
        """ Create the criterion.
        Parameters:
            num_classes: dict of number of sub clses, obj clses and relation clses, 
                         omitting the special no-object category
                         keys: ["obj_labels", "rel_labels"]
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes['obj_labels']
        self.rel_classes = num_classes['rel_labels']
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs_dict, targets, indices_dict, num_boxes_dict, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_det' in outputs_dict
        outputs = outputs_dict['pred_det']
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        indices = indices_dict['det']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def loss_actions(self, outputs_dict, targets, indices_dict, num_boxes_dict, log=True,
                     neg_act_id=0, topk=5, alpha=0.25, gamma=2, loss_reduce='sum'):
        """Intereaction classificatioon loss (multi-label Focal Loss based on Sigmoid)
        targets dicts must contain the key "actions" containing a tensor of dim [nb_target_boxes]
        Return:
            losses keys:["rel_loss_ce", "rel_class_error"]
        """
        assert 'pred_rel' in outputs_dict
        outputs = outputs_dict['pred_rel']
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        indices = indices_dict['rel']
        idx = self._get_src_permutation_idx(indices)

        target_classes_obj = torch.cat([t["rel_labels"][J].to(src_logits.device) for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.zeros(src_logits.shape[0], src_logits.shape[1],
            self.rel_classes).type_as(src_logits).to(src_logits.device)
        target_classes[idx] = target_classes_obj.type_as(src_logits)
        losses = {}
        pred_sigmoid = src_logits.sigmoid()
        label = target_classes.long()
        pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)
        focal_weight = (alpha * label + (1 - alpha) * (1 - label)) * pt.pow(gamma)
        rel_loss = F.binary_cross_entropy_with_logits(src_logits,
            target_classes, reduction='none') * focal_weight
        if loss_reduce == 'mean':
            losses['rel_loss_ce'] = rel_loss.mean()
        else:
            losses['rel_loss_ce'] = rel_loss.sum()
        if log:
            _, pred = src_logits[idx].topk(topk, 1, True, True)
            acc = 0.0
            for tid, target in enumerate(target_classes_obj):
                tgt_idx = torch.where(target==1)[0]
                if len(tgt_idx) == 0:
                    continue
                acc_pred = 0.0
                for tgt_rel in tgt_idx:
                    acc_pred += (tgt_rel in pred[tid])
                acc += acc_pred / len(tgt_idx)
            rel_labels_error = 100 - 100 * acc / len(target_classes_obj)
            losses['rel_class_error'] = torch.from_numpy(np.array(
                rel_labels_error)).to(src_logits.device).float()
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        assert 'pred_det' in outputs_dict
        outputs = outputs_dict['pred_det']
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    @torch.no_grad()
    def loss_rel_cardinality(self, outputs_dict, targets, indices_dict, num_boxes_dict, neg_act_id=0):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        assert 'pred_rel' in outputs_dict
        outputs = outputs_dict['pred_rel']
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != neg_act_id).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'rel_cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_det' in outputs_dict
        outputs = outputs_dict['pred_det']
        assert 'pred_boxes' in outputs

        indices = indices_dict['det']
        num_boxes = num_boxes_dict['det']
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_rel_vecs(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """Compute the losses related to the interaction vector, the L1 regression loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_rel' in outputs_dict
        outputs = outputs_dict['pred_rel']
        assert 'pred_boxes' in outputs
        indices = indices_dict['rel']
        num_vecs = num_boxes_dict['rel']
        idx = self._get_src_permutation_idx(indices)
        self.out_idx = idx
        self.tgt_idx = self._get_tgt_permutation_idx(indices)      
        src_vecs = outputs['pred_boxes'][idx]
        target_vecs = torch.cat([t['rel_vecs'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_vecs, target_vecs, reduction='none')
        losses = {}
        losses['rel_loss_bbox'] = loss_bbox.sum() / num_vecs
        return losses


    def loss_emb_push(self, outputs_dict, targets, indices_dict, num_boxes_dict, margin=8):
        """id embedding push loss.
        """
        indices = indices_dict['det']
        idx = self._get_src_permutation_idx(indices)
        if len(idx) == 0:
            losses = {'loss_push': torch.Tensor([0.]).mean().to(idx.device)}
            return losses
        id_emb = outputs_dict['pred_det']['id_emb'][idx]
        n = id_emb.shape[0]
        m = [m.reshape(-1) for m in torch.meshgrid(torch.arange(n), torch.arange(n))]
        mask = torch.where(m[1] < m[0])[0]
        emb_cmp = id_emb[m[0][mask]] - id_emb[m[1][mask]]
        emb_dist = torch.pow(torch.sum(torch.pow(emb_cmp, 2), 1), 0.5)
        loss_push = torch.pow((margin - emb_dist).clamp(0), 2).mean()
        losses = {'loss_push': loss_push}
        return losses

    def loss_emb_pull(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """id embedding pull loss.
        """
        det_indices = indices_dict['det']
        rel_indices = indices_dict['rel']
                
        # get indices: det_idx1: [rel_idx1_src, rel_idx2_dst]
        det_pred_idx = self._get_src_permutation_idx(det_indices)
        target_det_centr = torch.cat([t['boxes'][i] for t, (_, i) in zip(
            targets, det_indices)], dim=0)[..., :2]
        rel_pred_idx = self._get_src_permutation_idx(rel_indices)
        if len(rel_pred_idx) == 0:
            losses = {'loss_pull': torch.Tensor([0.]).mean().to(rel_pred_idx.device)}
            return losses
        target_rel_centr = torch.cat([t['rel_vecs'][i] for t, (_, i) in zip(
            targets, rel_indices)], dim=0)
        src_emb = outputs_dict['pred_rel']['src_emb'][rel_pred_idx]
        dst_emb = outputs_dict['pred_rel']['dst_emb'][rel_pred_idx]
        id_emb = outputs_dict['pred_det']['id_emb'][det_pred_idx]

        ref_id_emb = []
        for i in range(len(src_emb)):
            ref_idx = torch.where(target_det_centr==target_rel_centr[i, :2])[0]
            if len(ref_idx) == 0:
                # to remove cur instead of setting to 0.
                losses = {'loss_pull': torch.Tensor([0.]).mean().to(ref_idx.device)}
                return losses
            ref_id_emb.append(id_emb[ref_idx[0]])
        for i in range(len(dst_emb)):
            ref_idx = torch.where(target_det_centr==target_rel_centr[i, 2:])[0]
            if len(ref_idx) == 0:
                losses = {'loss_pull': torch.Tensor([0.]).mean().to(ref_idx.device)}
                return losses
            ref_id_emb.append(id_emb[ref_idx[0]])
        pred_rel_emb = torch.cat([src_emb, dst_emb], 0)
        ref_id_emb = torch.stack(ref_id_emb, 0).to(pred_rel_emb.device)
        loss_pull = torch.pow((pred_rel_emb - ref_id_emb), 2).mean()
        losses = {'loss_pull': loss_pull}

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_neg_permutation_idx(self, neg_indices):
        # permute neg rel predictions following indices
        batch_idx = torch.cat([torch.full_like(neg_ind, i) for i, neg_ind in enumerate(neg_indices)])
        neg_idx = torch.cat([neg_ind for neg_ind in neg_indices])
        return batch_idx, neg_idx

    def get_loss(self, loss, outputs_dict, targets, indices_dict, num_boxes_dict, **kwargs):
        if outputs_dict['pred_rel'] is None:
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'boxes': self.loss_boxes
            }
        else:
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'boxes': self.loss_boxes,
                'actions': self.loss_actions,
                'rel_vecs': self.loss_rel_vecs,
                'rel_cardinality': self.loss_rel_cardinality,
                'emb_push': self.loss_emb_push,
                'emb_pull':self.loss_emb_pull
            }
        if loss not in loss_map:
            return {}
        return loss_map[loss](outputs_dict, targets, indices_dict, num_boxes_dict, **kwargs)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices_dict = self.matcher(outputs, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
            device=next(iter(outputs['pred_det'].values())).device)
        rel_num_boxes = sum(len(t["rel_labels"]) for t in targets)
        rel_num_boxes = torch.as_tensor([rel_num_boxes], dtype=torch.float,
            device=next(iter(outputs['pred_rel'].values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
            torch.distributed.all_reduce(rel_num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        rel_num_boxes = torch.clamp(rel_num_boxes / get_world_size(), min=1).item()
        num_boxes_dict = {
            'det': num_boxes,
            'rel': rel_num_boxes
        }
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets,
                                        indices_dict, num_boxes_dict))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs.keys():
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_dict = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'actions':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dict, num_boxes_dict, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


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
        valid_hoi_mask = rel_array[..., o_clses[rel_o_ids]-1].permute(1, 0)
        hoi_scores = (valid_hoi_mask * hoi_scores).reshape(-1, 1)
        hoi_labels = hoi_labels.reshape(-1, 1)
        rel_s_ids = rel_s_ids.unsqueeze(-1).repeat(1, topk).reshape(-1, 1)
        rel_o_ids = rel_o_ids.unsqueeze(-1).repeat(1, topk).reshape(-1, 1)
        hoi_triplet = (torch.cat((rel_s_ids.float(), rel_o_ids.float(), hoi_labels.float(),
            hoi_scores.float()), 1)).cpu().numpy()
        hoi_triplet = hoi_triplet[hoi_triplet[..., -1]>0.0]

        # remove repeated triplets
        if len(hoi_triplet) == 0:
            pred_out = {
                'file_name': file_name,
                'hoi_prediction': [],
                'predictions': []
            }
            return pred_out
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
    model = ASNet_HOIA(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        rel_num_queries=cfg.TRANSFORMER.REL_NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
    )
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.DET_CLS_COEF[0], 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF[0]}
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF[0]
    weight_dict.update({'rel_loss_ce': cfg.LOSS.REL_CLS_COEF, 'rel_loss_bbox': cfg.LOSS.BBOX_LOSS_COEF[1]})
    weight_dict.update({'loss_pull': 0.1, 'loss_push': 0.1})
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'actions', 'rel_vecs', 'rel_cardinality',
              'emb_pull', 'emb_push']
    criterion = SetCriterion(matcher=matcher, losses=losses, weight_dict=weight_dict,
                             eos_coef=cfg.LOSS.EOS_COEF, num_classes=num_classes)
    criterion.to(device)
    postprocessors = PostProcess(cfg.TEST.REL_ARRAY_PATH, cfg.TEST.USE_EMB)
    return model, criterion, postprocessors