import copy
import os.path as osp

import torch
import torch_geometric.transforms as T

import experiments.table1.settings as table1_settings
from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.models.supervised import GraphSAGE
from graphsage.samplers import UniformLoader
from graphsage.trainers import SupervisedTrainerForNodeClassification

device = settings.DEVICE
dataset_name = 'Cora'  # 'Cora', 'CiteSeer', 'PubMed'
path = osp.join(settings.DATA_DIR, dataset_name)


def get(aggregator):
    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

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
    return SupervisedTrainerForNodeClassification(
        dataset_name=dataset_name,
        model=model,
        data=copy.copy(data),
        loader=UniformLoader,
        num_epochs=settings.NUM_EPOCHS,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=table1_settings.SUPERVISED_LEARNING_RATE),
        device=device,
    )
