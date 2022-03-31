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
    loader=UniformLoader,
    num_epochs=settings.NUM_EPOCHS,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    device=device,
).run()

