import os.path as osp

import torch

import experiments.fig3.settings as fig3_settings
from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.datasets import Reddit
from graphsage.trainers import RawFeaturesTrainerForNodeLevelTask

if fig3_settings.DATASET == 'reddit':
    path = osp.join(settings.DATA_DIR, fig3_settings.DATASET.capitalize())
    dataset = Reddit(path)


if fig3_settings.DATASET in {'cora', 'citeseer', 'pubmed'}:
    dataset_name = fig3_settings.DATASET.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name)


def get(noise_prop):
    data = dataset[0]

    # Add noise to the feature matrix
    data.x = (1 - noise_prop)*data.x + noise_prop*torch.randn_like(data.x)

    return RawFeaturesTrainerForNodeLevelTask(data)


