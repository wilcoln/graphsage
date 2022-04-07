import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.models.triples import MLP
from graphsage.trainers.triples_models_trainers import TriplesTorchModuleTrainer
from graphsage.datasets.triples import pyg_graph_to_triples

device = settings.DEVICE
dataset_name = 'Cora'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Planetoid(path, dataset_name)
# Create the triples dataset
td = pyg_graph_to_triples(dataset)


# Train a triple model on the dataset
# region MLP classifier
td.y = td.y[:, 0]
model = MLP(
    in_channels=td.x.shape[1],
    num_layers=2,
    hidden_channels=256,
    out_channels=td.num_classes
).to(device)

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
