import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.samplers import UniformLoader
from graphsage.trainers import SupervisedTrainerForNodeClassification
from models.supervised import GraphSAGE

device = settings.DEVICE

path = osp.join(settings.DATA_DIR, 'Reddit')
dataset = Reddit(path)

data = dataset[0]

kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': True}
train_loader = UniformLoader(data, input_nodes=data.train_mask, num_neighbors=[25, 10], shuffle=False, **kwargs)

subgraph_loader = UniformLoader(data, input_nodes=None, num_neighbors=[25, 10], shuffle=False, **kwargs)

model = GraphSAGE(
    in_channels=dataset.num_features,
    hidden_channels=256,
    out_channels=dataset.num_classes,
    num_layers=2,
    aggregator='mean',
).to(device)

SupervisedTrainerForNodeClassification(
    model=model,
    data=data,
    num_epochs=settings.NUM_EPOCHS,
    train_loader=train_loader,
    subgraph_loader=subgraph_loader,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    device=device,
).run()
