import os.path as osp

import torch
import torch.nn.functional as F
# pyg imports
from torch_geometric.nn import global_mean_pool

# Our own imports
from graphsage import settings
from graphsage.datasets import Flickr
from graphsage.layers import SAGE
from graphsage.samplers import ShaDowKHopSampler

path = osp.join(settings.DATA_DIR, 'Flickr')
dataset = Flickr(path)
data = dataset[0]

kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': settings.PERSISTENT_WORKERS}
train_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                                 node_idx=data.train_mask, **kwargs)
val_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                               node_idx=data.val_mask, **kwargs)
test_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                                node_idx=data.test_mask, **kwargs)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # aggregator_type = ['mean', 'gcn', 'max', 'sum', 'lstm', 'bilstm']
        self.conv1 = SAGE(in_channels, hidden_channels, aggregator='mean')
        self.conv2 = SAGE(hidden_channels, hidden_channels, aggregator='mean')
        self.conv3 = SAGE(hidden_channels, hidden_channels, aggregator='mean')
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, root_n_id):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)

        # We merge both central node embeddings and subgraph embeddings:
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)

        x = self.lin(x)
        return x


device = settings.DEVICE
model = GraphSAGE(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_examples += data.num_graphs
    return total_correct / total_examples


for epoch in range(1, 5):  # 51 originally
    loss = train()
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ',
          f'Val: {val_acc:.4f} Test: {test_acc:.4f}')
