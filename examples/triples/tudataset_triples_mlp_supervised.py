import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from datasets.triples import pyg_graph_list_to_triples_graph_list
from graphsage import settings
from graphsage.models.triples import GraphLevelTriplesMLP
from graphsage.trainers.graph_level_triples_models_trainers import GraphLevelTriplesTorchModuleTrainer

name = settings.args.dataset if settings.args.dataset else 'mutag'
name = name.upper()

assert name in [
    'MUTAG',
    'ENZYMES',
    'PROTEINS',
    'IMDB-BINARY',
    'REDDIT-BINARY',
]

dataset = TUDataset(root='data/TUDataset', name=name)

dataset = dataset.shuffle()

num_classes = dataset.num_classes

if dataset.num_features:
    num_features = dataset.num_features
    processed_dataset = dataset
else:
    processed_dataset = []
    num_features = 10
    for graph in dataset:
        processed_dataset.append(
            Data(
                x=torch.zeros(graph.num_nodes, num_features),
                y=graph.y,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                num_nodes=graph.num_nodes
            )
        )

    dataset = processed_dataset

N_train = int(len(dataset) * 0.6)
N_val = int(len(dataset) * 0.2)
N_test = len(dataset) - N_train - N_val

train_dataset = dataset[:N_train]
val_dataset = dataset[N_train:N_train + N_val]
test_dataset = dataset[N_train + N_val:]

# dataset_name = 'PPI'
# path = osp.join(settings.DATA_DIR, dataset_name)
# dataset = train_dataset = PPI(path, split='train')
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')

train_dataset = pyg_graph_list_to_triples_graph_list(train_dataset)
val_dataset = pyg_graph_list_to_triples_graph_list(val_dataset)
test_dataset = pyg_graph_list_to_triples_graph_list(test_dataset)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = settings.DEVICE
model = GraphLevelTriplesMLP(
    in_channels=2 * num_features,
    hidden_channels=256,
    out_channels=num_classes,
    num_layers=2,
    rni=False).to(device)

GraphLevelTriplesTorchModuleTrainer(
    dataset_name='REDDIT-BINARY',
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=settings.NUM_EPOCHS,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    device=device,
    num_prints=10,
).run()
