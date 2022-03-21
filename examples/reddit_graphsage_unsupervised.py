# WARNING: Needs at least 2GB of GPU memory to run.
import json
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.linear_model import SGDClassifier
# pyg imports
from sklearn.metrics import f1_score
from torch_cluster import random_walk
from tqdm import tqdm

# Our own imports
from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.layers import SAGE
from graphsage.samplers import UniformSampler

device = settings.DEVICE

path = osp.join(settings.DATA_DIR, 'Reddit')
dataset = Reddit(path, transform=T.NormalizeFeatures())

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


train_loader = NeighborSampler(data.edge_index, sizes=[25, 10], batch_size=settings.BATCH_SIZE,
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

    def inference(self, x_all, subgraph_loader):
        xs = []
        for batch_size, n_id, adjs in tqdm(subgraph_loader):
            adjs = [adj.to(device) for adj in adjs]
            x = x_all[n_id].to(device)
            x = self.forward(x, adjs)
            xs.append(x)
        x_all = torch.cat(xs, dim=0)
        return x_all


model = GraphSAGE(data.num_node_features, hidden_channels=256, num_layers=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)


def train():
    model.train()

    total_loss = 0
    xs = []
    for batch_size, n_id, adjs in tqdm(train_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(data.x[n_id].to(settings.DEVICE), adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)
        xs.append(out)
    x_all = torch.cat(xs, dim=0)
    return x_all, total_loss / data.num_nodes


@torch.no_grad()
def test(out):
    if out is None:
        model.eval()
        out = model.inference(data.x, train_loader)
        out, _, _ = out.split(out.size(0) // 3, dim=0)

    out = out.cpu()
    clf = SGDClassifier(loss="log", penalty="l2")
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    # compute test and val f1 score
    pred = clf.predict(out[data.test_mask])
    test_f1 = f1_score(data.y[data.test_mask], pred, average='micro')
    pred = clf.predict(out[data.val_mask])
    val_f1 = f1_score(data.y[data.val_mask], pred, average='micro')

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_f1, val_acc, test_f1, test_acc


model_name = 'reddit_normalized_graphsage_unsupervised'

for epoch in range(1, 11):
    # check if model is saved if it is train if not load it
    model_path = osp.join(settings.CACHE_DIR, f'{model_name}_{epoch}.pth')
    output_path = model_path.replace(".pth", ".out")
    loss_path = model_path.replace(".pth", ".loss")
    if osp.exists(model_path):
        print(f'Loading model from {model_path}, output from {output_path}, and loss from {loss_path}')
        model.load_state_dict(torch.load(model_path))
        out = torch.load(output_path)
        loss = torch.load(loss_path)
        print('Done !')
    else:
        print(f'Training model {model_name}')
        out, loss = train()
        print(f'Saving model to {model_path}, output to {output_path}, and loss to {loss_path}')
        torch.save(model.state_dict(), model_path)
        torch.save(out, output_path)
        torch.save(loss, loss_path)
        print('Done !')

    val_f1, val_acc, test_f1, test_acc = test(out)
    # print epoch and results
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')
    # save results
    results = {
        'val_f1': val_f1,
        'test_f1': test_f1,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }
    results_path = osp.join(settings.RESULTS_DIR, f'{model_name}_{epoch}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f'Saved results to {results_path}')