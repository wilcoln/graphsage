import os.path as osp

import torch

from torch_geometric.loader import DataLoader

# Our own imports
from graphsage import settings
from graphsage.datasets import PPI
from graphsage.models.unsupervised import GraphSAGE
from graphsage.samplers import UniformSampler
from trainers import UnsupervisedTrainerForGraphClassification

device = settings.DEVICE

path = osp.join(settings.DATA_DIR, 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

train_data_list = [data for data in train_dataset]


train_loader_list = []

for curr_graph in train_data_list:
    _train_loader = UniformSampler(curr_graph.edge_index, sizes=[25, 10], batch_size=settings.BATCH_SIZE, shuffle=True,
                                    num_nodes=curr_graph.num_nodes)
    train_loader_list.append(_train_loader)


model = GraphSAGE(
    in_channels=train_dataset.num_node_features,
    hidden_channels=64,
    out_channels=train_dataset.num_classes,
    num_layers=2,
    aggregator='mean',
).to(device)

UnsupervisedTrainerForGraphClassification(
    model=model,
    num_epochs=settings.NUM_EPOCHS,
    train_loader=train_loader,
    train_loader_list=train_loader_list,
    train_data_list=train_data_list,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    device=device,
).run()
