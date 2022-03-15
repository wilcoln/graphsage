import os.path as osp
import torch

DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_WORKERS = 4

