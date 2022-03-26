import os.path as osp

import torch

DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
CACHE_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'cache')
RESULTS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
NUM_WORKERS = 4
NUM_EPOCHS = 10
