import argparse
import os.path as osp

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', help='Number of epochs', type=int)
parser.add_argument('--batch_size', help='Batch size', type=int)
parser.add_argument('--num_runs', help='Number of runs', type=int, default=1)
parser.add_argument('--colab', action='store_true', help="whether we are running on google colab", default=False)
parser.add_argument('--dataset', help='Dataset to use', type=str)
parser.add_argument('--aggregator', help='Aggregator to use', type=str, default='mean')
parser.add_argument('--ignore_datasets', nargs='*', help='Datasets to ignore', type=str)
parser.add_argument('--ignore_aggregators', nargs='*', help='Aggregators to ignore', type=str)
parser.add_argument('--results_dir', help='Result directory to use', type=str)
parser.add_argument('--no_show', action='store_true', help="Do not show the figure at the end", default=False)
parser.add_argument('--use_triple_loss', action='store_true', help="Use triple loss as unsup. loss", default=False)
parser.add_argument('--persistent_workers', action='store_true', help="Whether to make dataloader workers "
                                                                      "persistent", default=False)
parser.add_argument('--num_workers', help='Number of workers', type=int)
parser.add_argument('--lstm_num_inputs', help='Number of inputs for lstm aggregator', type=int)
parser.add_argument('--no_eval_train', action='store_true', help="whether to evaluate on the train set as well",
                    default=False)
parser.add_argument('--no_extensions', action='store_true', help="whether to include extension in the experiments",
                    default=False)
parser.add_argument('--std', action='store_true', help='Include standard deviation in table output', default=False)
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

if args.num_workers is not None:
    NUM_WORKERS = args.num_workers

if args.lstm_num_inputs is not None:
    LSTM_NUM_INPUTS = args.lstm_num_inputs

PERSISTENT_WORKERS = args.persistent_workers

NO_EVAL_TRAIN = args.no_eval_train

NUM_RUNS = args.num_runs
