import argparse
import os.path as osp

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', help='Number of epochs', type=int)
parser.add_argument('--batch_size', help='Batch size', type=int)
parser.add_argument('--num_runs', help='Number of runs', type=int)
parser.add_argument('--colab', action='store_true', default=False)
parser.add_argument('--ignore_reddit', action='store_true', default=False)
parser.add_argument('--dataset', help='Dataset to use', type=str)
args = parser.parse_args()

DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
CACHE_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'cache')
RESULTS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512
NUM_WORKERS = 4
NUM_EPOCHS = 10

if args.colab:
    DATA_DIR = '/content/data'
    CACHE_DIR = '/content/cache'
    RESULTS_DIR = '/content/results'
    NUM_WORKERS = 2

if args.num_epochs is not None:
    # NUM_EPOCHS = args.num_epochs
    NUM_EPOCHS = 10

if args.batch_size is not None:
    # BATCH_SIZE = args.batch_size
    BATCH_SIZE = 512
