import torch
import numpy as np

from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg
from datasets.MJSynthDataset import MJSynthDataset


def get_dataloaders():
    train_dataset = MJSynthDataset(cfg=dataset_cfg, dataset_type='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=train_cfg.batch_size)

    test_dataset = MJSynthDataset(cfg=dataset_cfg, dataset_type='test')
    test_dl = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=train_cfg.batch_size)

    val_dataset = MJSynthDataset(cfg=dataset_cfg, dataset_type='val')
    val_dl = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, batch_size=train_cfg.batch_size)
    return train_dl, test_dl, val_dl


# def collate_fn(data):
#     bs = len(data)
#     images, labels = zip(*data)
#     max_width = np.asarray([im.shape for im in images])[:, 2].max()
#     max_len = max([len(l) for l in labels])
#
#     features = np.zeros((bs, 3, 32, max_width))
#     labels_ = np.zeros((bs, max_len))
#
#     for i in range(bs):
#         _, h, w = data[i][0].shape
#         features[i] = np.concatenate([data[i][0], np.zeros((3, h, max_width - w))], -1)
#         len_ = len(data[i][1])
#         labels_[i] = np.concatenate([data[i][1], np.zeros((max_len - len_))], -1)
#
#     padded_features = torch.tensor(features, dtype=torch.float32)
#     padded_labels = torch.tensor(labels_, dtype=torch.long)
#     real_widths = torch.tensor(np.asarray([im.shape for im in images])[:, 2])
#     # padded_widths = torch.tensor([max_width] * len(data))
#     return padded_features, torch.tensor(labels), real_widths  # , padded_widths


def collate_fn(data):
    bs = len(data)
    images, labels = zip(*data)
    max_width = np.asarray([im.shape for im in images])[:, 2].max()
    max_len = max([len(l) for l in labels])

    features = np.zeros((bs, 3, 32, max_width))
    # labels_ = np.zeros((bs, max_len))

    for i in range(bs):
        _, h, w = data[i][0].shape
        features[i] = np.concatenate([data[i][0], np.zeros((3, h, max_width - w))], -1)
        # len_ = len(data[i][1])
        # labels_[i] = np.concatenate([data[i][1], np.zeros((max_len - len_))], -1)

    padded_features = torch.tensor(features, dtype=torch.float32)
    # padded_labels = torch.tensor(labels_, dtype=torch.long)
    real_widths = torch.tensor(np.asarray([im.shape for im in images])[:, 2])
    # padded_widths = torch.tensor([max_width] * len(data))
    return padded_features, labels, real_widths  # , padded_widths
