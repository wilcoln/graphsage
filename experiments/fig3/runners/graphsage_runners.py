import copy
import os.path as osp

import torch

import experiments.fig3.settings as fig3_settings
from graphsage import settings
from graphsage.datasets import Reddit, Planetoid
from graphsage.models import GraphSAGE
from graphsage.samplers import UniformLoader
from graphsage.trainers import SupervisedGraphSageTrainerForNodeLevelTask

device = settings.DEVICE


if fig3_settings.DATASET == 'reddit':
    path = osp.join(settings.DATA_DIR, fig3_settings.DATASET.capitalize())
    dataset = Reddit(path)


if fig3_settings.DATASET in {'cora', 'citeseer', 'pubmed'}:
    dataset_name = fig3_settings.DATASET.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name)


def get(aggregator, noise_prop):
    data = dataset[0]

    # Add noise to the feature matrix
    data.x = (1 - noise_prop)*data.x + noise_prop*torch.randn_like(data.x)

    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=fig3_settings.HIDDEN_CHANNELS,
        out_channels=dataset.num_classes,
        num_layers=fig3_settings.NUM_LAYERS,
        aggregator=aggregator,
    ).to(device)

    # Return trainer
    return SupervisedGraphSageTrainerForNodeLevelTask(
        dataset_name=dataset_name,
        model=model,
        data=copy.copy(data),
        loader=UniformLoader,
        num_epochs=settings.NUM_EPOCHS,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=fig3_settings.LEARNING_RATE, weight_decay=1e-1),
        device=device,
    )

