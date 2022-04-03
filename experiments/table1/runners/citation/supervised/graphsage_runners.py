import copy
import os.path as osp
from types import SimpleNamespace

import torch

import experiments.table1.settings as table1_settings
from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.models import GraphSAGE
from graphsage.samplers import UniformLoader
from graphsage.trainers import SupervisedGraphSageTrainerForNodeLevelTask

device = settings.DEVICE


def get(dataset_name, aggregator):
    dataset_name = dataset_name.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name)

    # Send nodes to GPU for faster sampling
    data = dataset[0].to(device, 'x', 'y')

    # Create model
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=table1_settings.HIDDEN_CHANNELS,
        out_channels=dataset.num_classes,
        num_layers=table1_settings.NUM_LAYERS,
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
        optimizer=torch.optim.Adam(model.parameters(), lr=table1_settings.SUPERVISED_LEARNING_RATE),
        device=device,
    )


cora = SimpleNamespace(get=lambda aggregator: get('cora', aggregator))
citeseer = SimpleNamespace(get=lambda aggregator: get('citeseer', aggregator))
pubmed = SimpleNamespace(get=lambda aggregator: get('pubmed', aggregator))
