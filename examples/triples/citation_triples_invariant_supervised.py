import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.models.triples import MLP, InvariantModel
from graphsage.trainers.triples_models_trainers import TriplesTorchModuleTrainer
from graphsage.datasets.triples import pyg_graph_to_triples

device = settings.DEVICE
dataset_name = 'Cora'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Planetoid(path, dataset_name)
# Create the triples dataset
td = pyg_graph_to_triples(dataset)

# Train a triple model on the dataset
# region InvariantModel classifier
td.y = td.y[:, 0]

phi = MLP(
    in_channels=td.x.shape[1]//2,
    num_layers=1,
    hidden_channels=256,
).to(device)

rho = MLP(
    in_channels=256,
    num_layers=1,
    hidden_channels=td.num_classes,
).to(device)

model = InvariantModel(phi=phi, rho=rho).to(device)

TriplesTorchModuleTrainer(
    dataset_name=dataset_name,
    model=model,
    data=td,
    num_epochs=settings.NUM_EPOCHS,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device=device,
).run()
# endregion