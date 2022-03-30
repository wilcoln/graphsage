import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from torch_cluster import random_walk
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Our own imports
from graphsage import settings
from graphsage.datasets import PPI
from graphsage.layers import SAGE
from graphsage.samplers import UniformSampler

device = settings.DEVICE
EPS = 1e-15

path = osp.join(settings.DATA_DIR, 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

train_data_list = [data for data in train_dataset]


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


train_loader_list = []

for curr_graph in train_data_list:
    _train_loader = NeighborSampler(curr_graph.edge_index, sizes=[10, 10], batch_size=256, shuffle=True,
                                    num_nodes=curr_graph.num_nodes)
    train_loader_list.append(_train_loader)


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


model = GraphSAGE(train_dataset.num_features, 121, train_dataset.num_classes).to(
    device)  # hidden channels = 21 to make it match with data.y's shape which is [#, 121]
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    total_num_nodes = 0

    for i in tqdm(range(len(train_loader_list))):  # loop over all 20 train loaders
        # add up current train loaders # of nodes to total_num_nodes
        total_num_nodes += train_data_list[i].num_nodes
        # update the value for x and edge index for the current data
        x, edge_index = train_data_list[i].x.to(device), train_data_list[i].edge_index.to(device)
        for batch_size, n_id, adjs in train_loader_list[i]:  # train over the current graph
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()  # set the gradients to zero

            out = model(x[n_id], adjs)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * out.size(0)

    return total_loss / total_num_nodes


def _loader_to_embeddings_and_labels(model, loader):
    xs, ys = [], []
    for data in loader:
        ys.append(torch.argmax(data.y, dim=1))
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        out = model.full_forward(x, edge_index)
        xs.append(out)
    return torch.cat(xs, dim=0).cpu(), torch.cat(ys, dim=0).cpu()


@torch.no_grad()
def test(train_loader, val_loader, test_loader):
    model.eval()

    # Create classifier
    clf = SGDClassifier(loss="log", penalty="l2")

    # Train classifier on train data
    train_embeddings, train_labels = _loader_to_embeddings_and_labels(model, train_loader)
    clf.fit(train_embeddings, train_labels)
    train_predictions = clf.predict(train_embeddings)
    train_f1 = f1_score(train_labels, train_predictions, average='micro')

    # Evaluate on validation set
    val_embeddings, val_labels = _loader_to_embeddings_and_labels(model, val_loader)
    val_predictions = clf.predict(val_embeddings)
    val_f1 = f1_score(val_labels, val_predictions, average='micro')

    # Evaluate on validation set
    test_embeddings, test_labels = _loader_to_embeddings_and_labels(model, test_loader)
    test_predictions = clf.predict(test_embeddings)
    test_f1 = f1_score(test_labels, test_predictions, average='micro')

    return train_f1, val_f1, test_f1


for epoch in range(1, settings.NUM_EPOCHS + 1):
    loss = train()
    train_f1, val_f1, test_f1 = test(train_loader, val_loader, test_loader)
    print('Epoch: {:03d}, Loss: {:.4f}, Train F1: {:.4f}, Val F1: {:.4f}, Test F1: {:.4f}'
          .format(epoch, loss, train_f1, val_f1, test_f1))
