from yacs.config import CfgNode as CN

INF = 1e8

_C = CN()

# working dir
_C.OUTPUT_ROOT = ''

# distribution
_C.DIST_BACKEND = 'nccl'
_C.DEVICE = 'cuda'
_C.WORKERS = 4
_C.PI = 'mAP'
_C.SEED = 42

# cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# dataset
_C.DATASET = CN()
_C.DATASET.FILE = 'hoi_det'
_C.DATASET.NAME = 'HICODetDataset'
_C.DATASET.ROOT = ''
_C.DATASET.MEAN = []
_C.DATASET.STD = []
_C.DATASET.MAX_SIZE = 1333
_C.DATASET.SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
_C.DATASET.IMG_NUM_PER_GPU = 2
_C.DATASET.SUB_NUM_CLASSES = 1
_C.DATASET.OBJ_NUM_CLASSES = 89
_C.DATASET.REL_NUM_CLASSES = 117

# model
_C.MODEL = CN()
# specific model 
_C.MODEL.FILE = ''
_C.MODEL.NAME = ''
# resume
_C.MODEL.RESUME_PATH = ''
_C.MODEL.MASKS = False

# backbone
_C.BACKBONE = CN()
_C.BACKBONE.NAME = 'resnet50'
_C.BACKBONE.DIALATION = False

# transformer
_C.TRANSFORMER = CN()
_C.TRANSFORMER.BRANCH_AGGREGATION = False
_C.TRANSFORMER.POSITION_EMBEDDING = 'sine' # choices=('sine', 'learned')
_C.TRANSFORMER.HIDDEN_DIM = 256
_C.TRANSFORMER.ENC_LAYERS = 6
_C.TRANSFORMER.DEC_LAYERS = 6
_C.TRANSFORMER.DIM_FEEDFORWARD = 2048
_C.TRANSFORMER.DROPOUT = 0.1
_C.TRANSFORMER.NHEADS = 8
_C.TRANSFORMER.NUM_QUERIES = 100
_C.TRANSFORMER.REL_NUM_QUERIES = 16
_C.TRANSFORMER.PRE_NORM = False

# matcher
_C.MATCHER = CN()
_C.MATCHER.COST_CLASS = 1
_C.MATCHER.COST_BBOX = 5
_C.MATCHER.COST_GIOU = 2

# LOSS
_C.LOSS = CN()
_C.LOSS.AUX_LOSS = True
_C.LOSS.DICE_LOSS_COEF = 1
_C.LOSS.DET_CLS_COEF = [1, 1]
_C.LOSS.REL_CLS_COEF = 1
_C.LOSS.BBOX_LOSS_COEF = [5, 5]
_C.LOSS.GIOU_LOSS_COEF = [2, 2]
_C.LOSS.EOS_COEF = 0.1

# trainer
_C.TRAINER = CN()
_C.TRAINER.FILE = ''
_C.TRAINER.NAME = ''

# train
_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = ''
_C.TRAIN.LR = 0.0001
_C.TRAIN.LR_BACKBONE = 0.00001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0001
# optimizer SGD
_C.TRAIN.NESTEROV = False
# learning rate scheduler
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_DROP = 70
_C.TRAIN.CLIP_MAX_NORM = 0.1
_C.TRAIN.MAX_EPOCH = 100
# train resume
_C.TRAIN.RESUME = False
# print freq
_C.TRAIN.PRINT_FREQ = 20
# save checkpoint during train
_C.TRAIN.SAVE_INTERVAL = 5000
_C.TRAIN.SAVE_EVERY_CHECKPOINT = False
# val when train
_C.TRAIN.VAL_WHEN_TRAIN = False

# test
_C.TEST = CN()
_C.TEST.REL_ARRAY_PATH = ''
_C.TEST.USE_EMB = False
_C.TEST.MODE = ''


def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.yaml_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    config.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)