import os.path as osp

import torch
import torch.nn.functional as F
# pyg imports

# Our own imports
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from datasets import Reddit
from graphsage import settings
from graphsage.layers import SAGE
from graphsage.samplers import ShaDowKHopSampler
import experiments.table1.settings as table1_settings

dataset_name = 'Reddit'
device = settings.DEVICE
kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': settings.PERSISTENT_WORKERS}


def get(aggregator):
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Reddit(path)
    data = dataset[0]

    train_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                                     node_idx=data.train_mask, **kwargs)
    val_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                                   node_idx=data.val_mask, **kwargs)
    test_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                                    node_idx=data.test_mask, **kwargs)

    class Shadow(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, aggregator):
            super().__init__()
            self.conv1 = SAGE(in_channels, hidden_channels, aggregator=aggregator)
            self.conv2 = SAGE(hidden_channels, hidden_channels, aggregator=aggregator)
            self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

        def forward(self, x, edge_index, batch, root_n_id):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.3)
            x = self.conv2(x, edge_index).relu()
            x = F.dropout(x, p=0.3, training=self.training)

            # We merge both central node embeddings and subgraph embeddings:
            x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)

            x = self.lin(x)
            return x


    class ShadowRunner:
        def __init__(self):
            self.model = Shadow(dataset.num_features, table1_settings.HIDDEN_CHANNELS, dataset.num_classes, aggregator).to(device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=table1_settings.SUPERVISED_LEARNING_RATE)

        def train(self, epoch):
            self.model.train()
            total_loss = total_examples = 0
            for data in tqdm(train_loader):
                data = data.to(device)
                self.optimizer.zero_grad()
                out = self.model(data.x, data.edge_index, data.batch, data.root_n_id)
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * data.num_graphs
                total_examples += data.num_graphs
            return total_loss / total_examples


        @torch.no_grad()
        def test(self, loader):
            self.model.eval()
            total_correct = total_examples = 0
            for data in loader:
                data = data.to(device)
                out = self.model(data.x, data.edge_index, data.batch, data.root_n_id)
                total_correct += int((out.argmax(dim=-1) == data.y).sum())
                total_examples += data.num_graphs
            return total_correct / total_examples

        def run(self):
            best_val_acc = 0
            for epoch in range(1, settings.NUM_EPOCHS + 1):  # 51 originally
                loss = self.train(epoch)
                val_acc = self.test(val_loader)
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = self.test(test_loader)

            return {
                'test_f1': test_acc,
            }

    return ShadowRunner()
