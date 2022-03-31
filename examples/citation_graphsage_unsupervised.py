import os.path as osp

import torch
import torch_geometric.transforms as T

from graphsage import settings
# Our own imports
from graphsage.datasets import Planetoid
from graphsage.models.unsupervised import GraphSAGE
from graphsage.samplers import UniformSampler, UniformLoader
from graphsage.trainers import UnsupervisedTrainerForNodeClassification

device = settings.DEVICE

dataset_name = 'Cora'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=64,
    num_layers=2,
    aggregator='mean',
).to(device)


UnsupervisedTrainerForNodeClassification(
    dataset_name=dataset_name,
    model=model,
    data=data,
    sampler=UniformSampler,
    loader=UniformLoader,
    num_epochs=settings.NUM_EPOCHS,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    device=device,
).run()


