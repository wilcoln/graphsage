import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.models.unsupervised import GraphSAGE
from graphsage.samplers import UniformSampler, UniformLoader
from graphsage.trainers import UnsupervisedTrainerForNodeClassification

device = settings.DEVICE

dataset_name = 'Reddit'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Reddit(path)

data = dataset[0]

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
