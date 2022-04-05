import os.path as osp

import torch
from torch_geometric.loader import DataLoader

# Our own imports
from graphsage import settings
from graphsage.datasets import PPI
from graphsage.models import GraphSAGE
from graphsage.samplers import UniformLoader
from graphsage.trainers import UnsupervisedGraphSageTrainerForGraphLevelTask

device = settings.DEVICE

dataset_name = 'PPI'
path = osp.join(settings.DATA_DIR, dataset_name)
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

model = GraphSAGE(
    in_channels=train_dataset.num_node_features,
    hidden_channels=256,
    num_layers=2,
    aggregator=settings.args.aggregator,
).to(device)

UnsupervisedGraphSageTrainerForGraphLevelTask(
    dataset_name=dataset_name,
    model=model,
    num_epochs=settings.NUM_EPOCHS,
    loader=UniformLoader,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
    device=device,
).run()
