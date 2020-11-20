from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import argparse
import importlib
import logging
import os

import torch

import _init_paths
from configs import cfg
from configs import update_config
from libs.datasets.collate import collect
from libs.datasets.transform import EvalTransform


def parse_args():
    parser = argparse.ArgumentParser(description='HOI Detection Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/hico.yaml',
        required=True,
        type=str)    
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
          
    return args


def main_per_worker():
    args = parse_args()
    update_config(cfg, args)
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device(cfg.DEVICE)
    
    # model
    module = importlib.import_module(cfg.MODEL.FILE)
    model, criterion, postprocessors = getattr(module, 'build_model')(cfg, device)
    model = torch.nn.DataParallel(model).to(device)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # resume
        if 'state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['state_dict'], strict=True)
            logging.info(f'==> model pretrained from {resume_path} \n')
    module = importlib.import_module(cfg.MODEL.FILE)
    model, criterion, postprocessors = getattr(module, 'build_model')(cfg, device)

    # get datset
    module = importlib.import_module(cfg.DATASET.FILE)
    Dataset = getattr(module, cfg.DATASET.NAME)
    data_root = cfg.DATASET.ROOT # abs path in yaml
    if not os.path.exists(anno_root):
        logging.info(f'==> Cannot found data: {data_root}')
        raise FileNotFoundError
    eval_transform = EvalTransform(
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        max_size=cfg.DATASET.MAX_SIZE
    )
    logging.info(f'==> load val sub set: {data_root}')
    anno_root = os.path.join(data_root, 'anno.json')
    if not os.path.exists(anno_root):
        logging.info(f'==> Cannot found annotation: {anno_root}')
        raise FileNotFoundError
    eval_dataset = Dataset(data_root, anno_root, eval_transform)
    if eval_set is not None:
        logging.info(f'==> the size of eval dataset is {len(eval_dataset)}')
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collect,
        num_workers=cfg.WORKERS
    )
    
    # start evaluate in Trainer
    Trainer = get_trainer(
        cfg,
        model,
        criterion=criterion,
        optimizer=None,
        lr_scheduler=None,
        postprocessors=postprocessors,
        log_dir=OUTPUT_ROOT+'/output',
        performance_indicator=cfg.PI,
        last_iter=-1,
        rank=0,
        device=device,
        max_norm=None
    )
    logging.info(f'==> start eval...')
    
    assert cfg.TEST.MODE in ['hico', 'vcoco', 'hoia']
    Trainer.evaluate(eval_loader, cfg.TEST.MODE)


if __name__ == '__main__':
    main_per_worker()