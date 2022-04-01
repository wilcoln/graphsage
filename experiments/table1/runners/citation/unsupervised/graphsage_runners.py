import os.path as osp

import torch
import torch_geometric.transforms as T

import experiments.table1.settings as table1_settings
from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.models.unsupervised import GraphSAGE
from graphsage.samplers import UniformLoader, UniformSampler
from graphsage.trainers import UnsupervisedTrainerForNodeClassification

device = settings.DEVICE
dataset_name = 'Cora'  # 'Cora', 'CiteSeer', 'PubMed'
path = osp.join(settings.DATA_DIR, dataset_name)


def get(aggregator):
    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

    # Send nodes to GPU for faster sampling:
    data = dataset[0].to(device, 'x', 'y')

    # Create model
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=table1_settings.HIDDEN_CHANNELS,
        num_layers=table1_settings.NUM_LAYERS,
        aggregator=aggregator,
    ).to(device)

    # Return trainer
    return UnsupervisedTrainerForNodeClassification(
        dataset_name=dataset_name,
        model=model,
        data=data,
        sampler=UniformSampler,
        loader=UniformLoader,
        num_epochs=settings.NUM_EPOCHS,
        optimizer=torch.optim.Adam(model.parameters(), lr=table1_settings.UNSUPERVISED_LEARNING_RATE),
        device=device,
    )
