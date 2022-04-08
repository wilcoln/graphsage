# Instantiate model and optimizers
import torch
from torch.utils.data import DataLoader

from graphsage import settings
from graphsage.datasets.triples import get_cycles
from graphsage.models.triples import GraphLevelTriplesMLP
from graphsage.trainers.graph_level_triples_models_trainers import GraphLevelTriplesTorchModuleTrainer

num_features = 50
train_dataset, val_dataset, test_dataset = get_cycles(num_features)

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = settings.DEVICE
model = GraphLevelTriplesMLP(
    in_channels=2*num_features,
    hidden_channels=100,
    out_channels=2,
    num_layers=2,
    rni=False).to(device)

GraphLevelTriplesTorchModuleTrainer(
    dataset_name='Cycles',
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=settings.NUM_EPOCHS,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device=device,
    num_prints=10,
).run()
