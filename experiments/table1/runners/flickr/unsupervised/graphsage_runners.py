import os.path as osp

import torch

import experiments.table1.settings as table1_settings
from datasets import Flickr
from graphsage import settings
from graphsage.models import GraphSAGE
from graphsage.samplers import UniformLoader
from graphsage.trainers import UnsupervisedGraphSageTrainerForNodeLevelTask

device = settings.DEVICE

dataset_name = 'Flickr'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Flickr(path)


def get(aggregator):
    data = dataset[0]

    # Create model
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=table1_settings.HIDDEN_CHANNELS,
        num_layers=table1_settings.NUM_LAYERS,
        aggregator=aggregator,
    ).to(device)

    # Return trainer
    return UnsupervisedGraphSageTrainerForNodeLevelTask(
        dataset_name=dataset_name,
        model=model,
        data=data,
        loader=UniformLoader,
        num_epochs=settings.NUM_EPOCHS,
        optimizer=torch.optim.Adam(model.parameters(), lr=table1_settings.UNSUPERVISED_LEARNING_RATE),
        device=device,
    )
