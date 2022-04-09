import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.datasets.triples import pyg_graph_to_triples
from graphsage.trainers.node_level_triples_models_trainers import UnsupervisedTriplesTorchModuleTrainer
from models.triples import TriplesMLP

device = settings.DEVICE
dataset_name = settings.args.dataset if settings.args.dataset is not None else 'cora'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Planetoid(path, dataset_name)
# Create the triples dataset
td = pyg_graph_to_triples(dataset)


# Train a triple model on the dataset
# region MLP classifier
model = TriplesMLP(
    in_channels=td.x.shape[1],
    num_layers=2,
    hidden_channels=256,
).to(device)

UnsupervisedTriplesTorchModuleTrainer(
    dataset_name=dataset_name,
    model=model,
    data=td,
    num_epochs=settings.NUM_EPOCHS,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device=device,
).run()
# endregion
