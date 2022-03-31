import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.models.unsupervised import GraphSAGE
from graphsage.samplers import UniformSampler, UniformLoader
from graphsage.trainers import UnsupervisedTrainerForNodeClassification

device = settings.DEVICE

path = osp.join(settings.DATA_DIR, 'Reddit')
dataset = Reddit(path)

data = dataset[0]

kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': True}

train_loader = UniformSampler(data.edge_index, sizes=[25, 10], shuffle=True, num_nodes=data.num_nodes, **kwargs)

subgraph_loader = UniformLoader(data, input_nodes=None, num_neighbors=[25, 10], shuffle=False, **kwargs)

model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=64,
    num_layers=2,
    aggregator='mean',
).to(device)


UnsupervisedTrainerForNodeClassification(
    model=model,
    data=data,
    num_epochs=settings.NUM_EPOCHS,
    train_loader=train_loader,
    subgraph_loader=subgraph_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    device=device,
).run()

