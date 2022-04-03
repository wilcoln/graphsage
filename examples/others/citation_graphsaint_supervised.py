import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.layers import SAGE
from graphsage.samplers import GraphSAINTRandomWalkSampler

dataset = 'Cora'
path = osp.join(settings.DATA_DIR, dataset)
dataset = Planetoid(path, dataset)

data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

loader = GraphSAINTRandomWalkSampler(data, batch_size=settings.BATCH_SIZE, walk_length=2,
                                     num_steps=5, sample_coverage=100,
                                     save_dir=dataset.processed_dir,
                                     num_workers=settings.NUM_WORKERS, )


class GraphSAINT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = SAGE(in_channels, hidden_channels)
        self.conv2 = SAGE(hidden_channels, hidden_channels)
        self.conv3 = SAGE(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAINT(hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs


for epoch in range(1, settings.NUM_EPOCHS + 1):
    loss = train()
    accs = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
