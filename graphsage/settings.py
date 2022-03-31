import argparse
import os.path as osp

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--colab', action='store_true', default=False)
args = parser.parse_args()

DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
CACHE_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'cache')
RESULTS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
NUM_WORKERS = 4
NUM_EPOCHS = 1

if args.colab:
    DATA_DIR = '/content/data'
    CACHE_DIR = '/content/cache'
    RESULTS_DIR = '/content/results'
    BATCH_SIZE = 2048
