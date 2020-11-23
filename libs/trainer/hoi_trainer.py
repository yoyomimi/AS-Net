import datetime
import logging
import math
import numpy as np
import sys
import time
from tqdm import tqdm

import torch
from torch import autograd

from libs.trainer.trainer import BaseTrainer
import libs.utils.misc as utils
from libs.utils.utils import save_checkpoint, write_dict_to_json


class HOITrainer(BaseTrainer):

    def __init__(self,
                 cfg,
                 model,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 postprocessors,
                 log_dir='output',
                 performance_indicator='mAP',
                 last_iter=-1,
                 rank=0,
                 device='cuda',
                 max_norm=0):

        super().__init__(cfg, model, criterion, optimizer, lr_scheduler, 
            log_dir, performance_indicator, last_iter, rank)
        self.postprocessors = postprocessors
        self.device = device
        self.max_norm = max_norm
    
    def evaluate(self, eval_loader, mode, rel_topk=100):
        self.model.eval()
        results = []
        count = 0
        for data in tqdm(eval_loader):
            imgs, targets, filenames = data
            imgs = [img.to(self.device) for img in imgs]
            # targets are list type
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            bs = len(imgs)
            target_sizes = targets[0]['size'].expand(bs, 2)
            target_sizes = target_sizes.to(self.device)
            outputs_dict = self.model(imgs)
            file_name = filenames[0]
            pred_out = self.postprocessors(outputs_dict, file_name, target_sizes,
                rel_topk=rel_topk)
            results.append(pred_out)
            count += 1

        # save the result
        result_path = f'{self.cfg.OUTPUT_ROOT}/pred.json'
        write_dict_to_json(results, result_path)

        # eval
        if mode == 'hico':
            from eval_tools.hico_eval import hico
            eval_tool = hico(annotation_file='data/hico/test_hico.json', 
                             train_annotation='data/hico/trainval_hico.json')
            mAP = eval_tool.evalution(results)
        else:
            mAP = 0.0

        return mAP