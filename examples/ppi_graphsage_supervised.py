import os.path as osp

import torch
from torch_geometric.loader import DataLoader

from graphsage import settings
from graphsage.datasets import PPI
from models.supervised import GraphSAGE
from trainers import SupervisedTrainerForGraphClassification

device = settings.DEVICE

dataset_name = 'PPI'
path = osp.join(settings.DATA_DIR, dataset_name)
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

model = GraphSAGE(
    in_channels=train_dataset.num_features,
    hidden_channels=256,
    out_channels=train_dataset.num_classes,
    num_layers=2,
    aggregator='mean',
).to(device)

SupervisedTrainerForGraphClassification(
    dataset_name=dataset_name,
    model=model,
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
    train_loader=DataLoader(train_dataset, batch_size=1, shuffle=True),
    val_loader=DataLoader(val_dataset, batch_size=2, shuffle=False),
    test_loader=DataLoader(test_dataset, batch_size=2, shuffle=False),
    device=device,
    num_epochs=settings.NUM_EPOCHS,
).run()
