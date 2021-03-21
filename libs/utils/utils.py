import importlib
import logging
import numpy as np
import os
import os.path as osp
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import ConcatDataset

from bisect import bisect_right
from functools import partial
from six.moves import map, zip

from libs.datasets.transform import TrainTransform
from libs.datasets.transform import EvalTransform


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def resource_path(relative_path):
    """To get the absolute path"""
    base_path = osp.abspath(".")

    return osp.join(base_path, relative_path)


def ensure_dir(root_dir, rank=0):
    if not osp.exists(root_dir) and rank == 0:
        print(f'=> creating {root_dir}')
        os.mkdir(root_dir)
    else:
        while not osp.exists(root_dir):
            print(f'=> wait for {root_dir} created')
            time.sleep(10)

    return root_dir


def create_logger(cfg, rank=0):
    # working_dir root
    abs_working_dir = resource_path('work_dirs')
    working_dir = ensure_dir(abs_working_dir, rank)
    # output_dir root
    output_root_dir = ensure_dir(os.path.join(working_dir, cfg.OUTPUT_ROOT), rank)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    final_output_dir = ensure_dir(os.path.join(output_root_dir, time_str), rank)
    # set up logger
    logger = setup_logger(final_output_dir, time_str, rank)

    return logger, final_output_dir


def setup_logger(final_output_dir, time_str, rank, phase='train'):
    log_file = f'{phase}_{time_str}_rank{rank}.log'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def get_model(cfg, device):
    module = importlib.import_module(cfg.MODEL.FILE)
    model, criterion, postprocessors = getattr(module, 'build_model')(cfg, device)

    return model, criterion, postprocessors

    
def get_optimizer(cfg, model):
    """Support two types of optimizers: SGD, Adam.
    """
    assert (cfg.TRAIN.OPTIMIZER in [
        'sgd',
        'adam',
    ])
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            nesterov=cfg.TRAIN.NESTEROV)
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    return optimizer


def load_checkpoint(cfg, model, optimizer, lr_scheduler, device, module_name='model'):
    last_iter = -1
    resume_path = cfg.MODEL.RESUME_PATH
    resume = cfg.TRAIN.RESUME
    if resume_path and resume:
        if osp.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location='cpu')
            # resume
            if 'state_dict' in checkpoint:
                model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                logging.info(f'==> model pretrained from {resume_path} \n')
            elif 'model' in checkpoint:
                if module_name == 'detr':
                    model.module.detr_head.load_state_dict(checkpoint['model'], strict=False)
                    logging.info(f'==> detr pretrained from {resume_path} \n')
                else:
                    model.module.load_state_dict(checkpoint['model'], strict=False)
                    logging.info(f'==> model pretrained from {resume_path} \n')
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info(f'==> optimizer resumed, continue training')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                last_iter = checkpoint['epoch']
                logging.info(f'==> last_epoch = {last_iter}')
            if 'epoch' in checkpoint:
                last_iter = checkpoint['epoch']
                logging.info(f'==> last_epoch = {last_iter}')
            # pre-train
        else:
            logging.error(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    else:
        logging.info("==> train model without resume")

    return model, optimizer, lr_scheduler, last_iter


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def get_lr_scheduler(cfg, optimizer, last_epoch=-1):
    """Support three types of optimizers: StepLR, MultiStepLR, MultiStepWithWarmup.
    """
    assert (cfg.TRAIN.LR_SCHEDULER in [
        'StepLR',
        'MultiStepLR',
        'MultiStepWithWarmup',
    ])
    if cfg.TRAIN.LR_SCHEDULER == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.TRAIN.LR_STEPS[0],
            cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch)
    elif cfg.TRAIN.LR_SCHEDULER == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_STEPS,
            cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch)
    elif cfg.TRAIN.LR_SCHEDULER == 'MultiStepWithWarmup':
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.TRAIN.LR_STEPS,
            cfg.TRAIN.LR_FACTOR,
            cfg.TRAIN.WARMUP_INIT_FACTOR,
            cfg.TRAIN.WARMUP_STEP,
            last_epoch)
    else:
        raise AttributeError(f'{cfg.TRAIN.LR_SCHEDULER} is not implemented')
    
    return lr_scheduler


def get_det_criterion(cfg):
        
    return critertion

def get_trainer(cfg, model, criterion, optimizer, lr_scheduler, postprocessors,
                log_dir, performance_indicator, last_iter, rank, device, max_norm):
    module = importlib.import_module(cfg.TRAINER.FILE)
    Trainer = getattr(module, cfg.TRAINER.NAME)(
        cfg,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        postprocessors=postprocessors,
        log_dir=log_dir,
        performance_indicator=performance_indicator,
        last_iter=last_iter,
        rank=rank,
        device=device,
        max_norm = max_norm
    )
    return Trainer

def list_to_set(data_list, name='train'):
    if len(data_list) == 0:
        dataset = None
        logging.warning(f"{name} dataset is None")
    elif len(data_list) == 1:
        dataset = data_list[0]
    else:
        dataset = ConcatDataset(data_list)
        
    if dataset is not None:
        logging.info(f'==> the size of {name} dataset is {len(dataset)}')
    return dataset

def get_dataset(cfg):
    train_transform = TrainTransform(
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        scales=cfg.DATASET.SCALES,
        max_size=cfg.DATASET.MAX_SIZE
    )
    eval_transform = EvalTransform(
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        max_size=cfg.DATASET.MAX_SIZE
    )
    module = importlib.import_module(cfg.DATASET.FILE)
    Dataset = getattr(module, cfg.DATASET.NAME)
    data_root = cfg.DATASET.ROOT # abs path in yaml
    # get train data list
    train_root = osp.join(data_root, 'train')
    train_set = [d for d in os.listdir(train_root) if osp.isdir(osp.join(train_root, d))]  
    if len(train_set) == 0:
        train_set = ['.']
    train_list = []
    for sub_set in train_set:
        train_sub_root = osp.join(train_root, sub_set)
        logging.info(f'==> load train sub set: {train_sub_root}')
        train_sub_set = Dataset(cfg, train_sub_root, train_transform)
        train_list.append(train_sub_set)
    # get eval data list
    eval_root = osp.join(data_root, 'test')
    eval_set = [d for d in os.listdir(eval_root) if osp.isdir(osp.join(eval_root, d))]
    if len(eval_set) == 0:
        eval_set = ['.']
    eval_list = []      
    for sub_set in eval_set:
        eval_sub_root = osp.join(eval_root, sub_set)
        logging.info(f'==> load val sub set: {eval_sub_root}')
        eval_sub_set = Dataset(cfg, eval_sub_root, eval_transform)
        eval_list.append(eval_sub_set)
    # concat dataset list
    train_dataset = list_to_set(train_list, 'train')
    eval_dataset = list_to_set(eval_list, 'eval')
    
    return train_dataset, eval_dataset

def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    logging.info(f'save model to {output_dir}')
    if is_best:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))

def load_eval_model(resume_path, model):
    if resume_path != '':
        if osp.exists(resume_path):
            print(f'==> model load from {resume_path}')
            checkpoint = torch.load(resume_path)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    return model

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def naive_np_nms(dets, thresh):  
    """Pure Python NMS baseline."""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = x1.argsort()[::-1]  
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
    return dets[keep]


def write_dict_to_json(mydict, f_path):
    import json
    import numpy
    class DateEnconding(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                numpy.uint16,numpy.uint32, numpy.uint64)):
                return int(obj)
            elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
                numpy.float64)):
                return float(obj)
            elif isinstance(obj, (numpy.ndarray,)): # add this line
                return obj.tolist() # add this line
            return json.JSONEncoder.default(self, obj)
    with open(f_path, 'w') as f:
        json.dump(mydict, f, cls=DateEnconding)
        print("write down det dict to %s!" %(f_path))
