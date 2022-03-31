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

dataset = 'Cora'
path = osp.join(settings.DATA_DIR, dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

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


