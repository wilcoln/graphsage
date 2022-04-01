import os.path as osp
import time

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm

import experiments.fig2a.settings as fig2a_settings
from graphsage import settings
from graphsage.datasets import Reddit, Planetoid
from graphsage.models.supervised import GraphSAGE
from graphsage.samplers import UniformLoader

device = settings.DEVICE


if fig2a_settings.DATASET == 'reddit':
    path = osp.join(settings.DATA_DIR, fig2a_settings.DATASET.capitalize())
    dataset = Reddit(path)


if fig2a_settings.DATASET in {'cora', 'citeseer', 'pubmed'}:
    dataset_name = fig2a_settings.DATASET.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

kwargs = {'batch_size': fig2a_settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS}


def train(epoch, model, optimizer, train_loader):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    train_time = 0
    for batch in train_loader:
        start = time.time()
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size].to(device)
        y_hat = model(batch.x.to(device), batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.batch_size
        train_time += (time.time() - start) * batch.batch_size
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, train_time / total_examples


@torch.no_grad()
def test(model, subgraph_loader, data):
    model.eval()

    start = time.time()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    test_time = time.time() - start

    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return tuple(accs) + (test_time,)


def run(aggregator):
    data = dataset[0]

    train_loader = UniformLoader(data, input_nodes=data.train_mask,
                                 num_neighbors=[fig2a_settings.K1, fig2a_settings.K2], shuffle=True, **kwargs)

    subgraph_loader = UniformLoader(data, input_nodes=None,
                                    num_neighbors=[fig2a_settings.K1, fig2a_settings.K2], shuffle=False, **kwargs)

    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=fig2a_settings.HIDDEN_CHANNELS,
        out_channels=dataset.num_classes,
        num_layers=fig2a_settings.NUM_LAYERS,
        aggregator=aggregator,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=fig2a_settings.LEARNING_RATE)

    train_time = test_time = 0
    for epoch in range(1, settings.NUM_EPOCHS + 1):
        loss, epoch_train_time = train(epoch, model, optimizer, train_loader)
        train_acc, val_acc, test_acc, epoch_test_time = test(model, subgraph_loader, data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc:'
              f' {test_acc:.4f}')

        train_time += epoch_train_time
        test_time += epoch_test_time

    train_time /= settings.NUM_EPOCHS
    test_time /= settings.NUM_EPOCHS

    test_proportion = int(data.test_mask.sum()) / data.num_nodes
    test_time *= test_proportion

    return {
        'test_time': test_time,  # full test time
        'train_time': train_time,  # per batch
    }


class GraphSAGERunner:
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def run(self):
        return run(self.aggregator)


def get(aggregator):
    return GraphSAGERunner(aggregator)
