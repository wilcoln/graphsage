import copy
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
# pyg imports
from sklearn.metrics import f1_score
from torch_cluster import random_walk
from tqdm import tqdm

# Our own imports
from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.layers import SAGE
from graphsage.samplers import UniformSampler, UniformLoader


# Define the device
device = settings.DEVICE

path = osp.join(settings.DATA_DIR, 'Reddit')
dataset = Reddit(path)

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

kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': True}
subgraph_loader = UniformLoader(copy.copy(data), input_nodes=None,
                                num_neighbors=[25, 10], shuffle=False, **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            # aggregator_type = ['mean', 'gcn', 'max', 'sum', 'lstm', 'bilstm']
            self.convs.append(SAGE(in_channels, hidden_channels, aggregator_type='mean'))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
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


model = GraphSAGE(data.num_node_features, hidden_channels=256, num_layers=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    total_loss = 0
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
    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    model.eval()
    out = model.inference(data.x, subgraph_loader).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    # compute test and val accuracies score

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    # compute test and val f1 score
    pred = clf.predict(out[data.test_mask])
    test_f1 = f1_score(data.y[data.test_mask], pred, average='micro')
    pred = clf.predict(out[data.val_mask])
    val_f1 = f1_score(data.y[data.val_mask], pred, average='micro')

    return val_f1, val_acc, test_f1, test_acc


for epoch in range(1, settings.NUM_EPOCHS + 1):
    loss = train()
    val_f1, val_acc, test_f1, test_acc = test()
    # print epoch and results
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')
