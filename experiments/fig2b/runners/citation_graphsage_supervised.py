import copy
import os.path as osp
import time

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets import Planetoid

from graphsage import settings
from graphsage.layers import SAGE
from graphsage.samplers import UniformLoader

device = settings.DEVICE
EPS = 1e-15
dataset = 'Cora'
path = osp.join(settings.DATA_DIR, dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': True}
train_loader = UniformLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = UniformLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGE(in_channels, hidden_channels))
        self.convs.append(SAGE(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


model = GraphSAGE(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_examples += batch.batch_size

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    start = time.time()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    test_time = time.time() - start

    y = data.y.to(y_hat.device)

    # Compute accuracy:
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))

    # Compute f1 score:
    f1s = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        f1s.append(f1_score(y[mask], y_hat[mask], average='micro'))

    return *accs, *f1s, test_time


def run():
    for epoch in range(1, settings.NUM_EPOCHS + 1):
        loss = train()
        train_acc, val_acc, test_acc, train_f1, val_f1, test_f1, test_time = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'test_time': test_time,
    }


class SampleSizeRunner:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def run(self):
        global train_loader, subgraph_loader, data, kwargs
        train_loader = UniformLoader(data, input_nodes=data.train_mask,
                                     num_neighbors=[self.sample_size, self.sample_size], shuffle=True, **kwargs)
        subgraph_loader = UniformLoader(copy.copy(data), input_nodes=None,
                                        num_neighbors=[self.sample_size, self.sample_size], shuffle=False, **kwargs)

        # No need to maintain these features during evaluation:
        del subgraph_loader.data.x, subgraph_loader.data.y
        # Add global node index information.
        subgraph_loader.data.num_nodes = data.num_nodes
        subgraph_loader.data.n_id = torch.arange(data.num_nodes)
        return run()


