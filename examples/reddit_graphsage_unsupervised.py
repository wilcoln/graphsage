import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.models import GraphSAGE
from graphsage.samplers import UniformLoader
from graphsage.trainers import UnsupervisedGraphSageTrainerForNodeLevelTask

device = settings.DEVICE

dataset_name = 'Reddit'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Reddit(path)

data = dataset[0]

model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=64,
    num_layers=2,
    aggregator=settings.args.aggregator,
).to(device)

UnsupervisedGraphSageTrainerForNodeLevelTask(
    dataset_name=dataset_name,
    model=model,
    data=data,
    loader=UniformLoader,
    num_epochs=settings.NUM_EPOCHS,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    device=device,
).run()
