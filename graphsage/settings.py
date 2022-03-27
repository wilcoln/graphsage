import os.path as osp

import torch

DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
CACHE_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'cache')
RESULTS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 1
SAINT_SAMPLER = ['NodeSampler','EdgeSampler','RandomWalkSampler']
SAINT_SAMPLER_ARGS = {'NodeSampler': {'sample_coverage':100},'EdgeSampler': {'sample_coverage':100},'RandomWalkSampler': {'walk_length': 2, 'num_steps': 5, 'sample_coverage':100}}
