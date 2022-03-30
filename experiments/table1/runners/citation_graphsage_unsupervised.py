import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
# pyg imports
import torch_geometric.transforms as T
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from torch_cluster import random_walk

from datasets import Planetoid
# Our own imports
from graphsage import settings
from graphsage.layers import SAGE
from graphsage.samplers import UniformSampler

device = settings.DEVICE
EPS = 1e-15
dataset = 'Cora'
path = osp.join(settings.DATA_DIR, dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class NeighborSampler(UniformSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)


train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                               shuffle=True, num_nodes=data.num_nodes)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGE(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

model = GraphSAGE(data.num_node_features, hidden_channels=64, num_layers=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
x, edge_index = data.x.to(device), data.edge_index.to(device)


def train():
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    model.eval()
    out = model.full_forward(x, edge_index).cpu()

    clf = SGDClassifier(loss="log", penalty="l2")
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    pred = clf.predict(out[data.test_mask])
    test_f1 = f1_score(data.y[data.test_mask], pred, average='micro')
    pred = clf.predict(out[data.val_mask])
    val_f1 = f1_score(data.y[data.val_mask], pred, average='micro')

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_f1, val_acc, test_f1, test_acc


def run():
    for epoch in range(1, settings.NUM_EPOCHS + 1):
        loss = train()
        val_f1, val_acc, test_f1, test_acc = test()
        # print epoch and results
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')

    return {
        'val_f1': val_f1,
        'val_acc': val_acc,
        'test_f1': test_f1,
        'test_acc': test_acc
    }
