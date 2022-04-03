import os.path as osp

import torch
from torch_geometric.loader import DataLoader

import experiments.table1.settings as table1_settings
from graphsage import settings
from graphsage.datasets import PPI
from graphsage.models import GraphSAGE
from graphsage.trainers import SupervisedGraphSageTrainerForGraphLevelTask

device = settings.DEVICE

dataset_name = 'PPI'
path = osp.join(settings.DATA_DIR, dataset_name)


def get(aggregator):
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

    # Create model
    model = GraphSAGE(
        in_channels=train_dataset.num_features,
        hidden_channels=table1_settings.HIDDEN_CHANNELS,
        out_channels=train_dataset.num_classes,
        num_layers=table1_settings.NUM_LAYERS,
        aggregator=aggregator,
    ).to(device)

    # Return trainer
    return SupervisedGraphSageTrainerForGraphLevelTask(
        dataset_name=dataset_name,
        model=model,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=table1_settings.SUPERVISED_LEARNING_RATE),
        train_loader=DataLoader(train_dataset, batch_size=1, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=2, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=2, shuffle=False),
        device=device,
        num_epochs=settings.NUM_EPOCHS,
    )
