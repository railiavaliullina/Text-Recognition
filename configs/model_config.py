from easydict import EasyDict
from configs.dataset_config import cfg as dataset_cfg

cfg = EasyDict()
cfg.in_channels = 3
cfg.num_classes = len(dataset_cfg.alphabet)
