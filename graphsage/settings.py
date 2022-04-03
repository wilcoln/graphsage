import argparse
import os.path as osp

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', help='Number of epochs', type=int)
parser.add_argument('--batch_size', help='Batch size', type=int)
parser.add_argument('--num_runs', help='Number of runs', type=int)
parser.add_argument('--colab', action='store_true', help="whether we are running on google colab", default=False)
parser.add_argument('--dataset', help='Dataset to use', type=str)
parser.add_argument('--aggregator', help='Aggregator to use', type=str, default='mean')
parser.add_argument('--ignore_datasets', nargs='*', help='Datasets to ignore', type=str)
parser.add_argument('--ignore_aggregators', nargs='*', help='Aggregators to ignore', type=str)
parser.add_argument('--results_dir', help='Result directory to use', type=str)
parser.add_argument('--no-show', action='store_true', help="Do not show the figure at the end", default=False)
args = parser.parse_args()

DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
RESULTS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512
NUM_WORKERS = 4
NUM_EPOCHS = 10

if args.colab:
    DATA_DIR = '/content/data'
    RESULTS_DIR = '/content/results'
    NUM_WORKERS = 2

if args.num_epochs is not None:
    NUM_EPOCHS = args.num_epochs

if args.batch_size is not None:
    BATCH_SIZE = args.batch_size

# TODO: remove this when those are fixed
if args.ignore_aggregators is None:
    args.ignore_aggregators = ['lstm', 'bilstm']
