import os.path as osp
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from tqdm import tqdm

from datasets import Planetoid
from graphsage import settings
from graphsage.layers import SAGE
from graphsage.samplers import GraphSAINTRandomWalkSampler
import experiments.table1.settings as table1_settings

device = settings.DEVICE


def get(dataset_name, aggregator):

    dataset_name = dataset_name.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name)

    device = settings.DEVICE
    data = dataset[0]

    class GraphSAINT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGE(in_channels, hidden_channels, aggregator=aggregator)
            self.conv2 = SAGE(hidden_channels, hidden_channels, aggregator=aggregator)
            self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

        def forward(self, x0, edge_index, edge_weight=None):
            x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
            x1 = F.dropout(x1, p=0.2, training=self.training)
            x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
            x2 = F.dropout(x2, p=0.2, training=self.training)
            x = torch.cat([x1, x2], dim=-1)
            x = self.lin(x)
            return x.log_softmax(dim=-1)


    class GraphSaintRunner:
        def __init__(self):
            row, col = data.edge_index
            data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

            self.loader = GraphSAINTRandomWalkSampler(data, batch_size=settings.BATCH_SIZE, walk_length=2,
                                                      num_steps=5, sample_coverage=5,
                                                      save_dir=dataset.processed_dir,
                                                      num_workers=settings.NUM_WORKERS, )

            self.model = GraphSAINT(
                in_channels=dataset.num_features,
                hidden_channels=table1_settings.HIDDEN_CHANNELS,
                out_channels=dataset.num_classes).to(device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=table1_settings.SUPERVISED_LEARNING_RATE)

        def train(self, epoch):
            self.model.train()

            total_loss = total_examples = 0
            for batch in tqdm(self.loader):
                batch = batch.to(device)
                self.optimizer.zero_grad()

                out = self.model(batch.x, batch.edge_index)
                loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_nodes
                total_examples += batch.num_nodes
            return total_loss / total_examples


        @torch.no_grad()
        def test(self):
            self.model.eval()

            # y_hats, ys = [], []
            # for batch in tqdm(self.loader):
            #     batch = batch.to(device)
            #     out = self.model(batch.x, batch.edge_index)
            #     y_hats.append(out.argmax(dim=-1))
            #     ys.append(batch.y)
            #
            # y_hats, ys = torch.cat(y_hats, dim=0), torch.cat(ys, dim=0)
            #
            # correct = y_hats.eq(ys.to(device))

            out = self.model(data.x.to(device), data.edge_index.to(device))
            pred = out.argmax(dim=-1)
            correct = pred.eq(data.y.to(device))

            accs = []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                accs.append(correct[mask].sum().item() / mask.sum().item())
            return tuple(accs)

        def run(self):
            for epoch in range(1, settings.NUM_EPOCHS + 1):
                loss = self.train(epoch)
                best_val_acc = best_test_acc = 0
                train_acc, val_acc, test_acc = self.test()
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test:'
                      f' {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
            return {
                'test_f1': best_test_acc,
            }

    return GraphSaintRunner()


cora = SimpleNamespace(get=lambda aggregator: get('cora', aggregator))
citeseer = SimpleNamespace(get=lambda aggregator: get('citeseer', aggregator))
pubmed = SimpleNamespace(get=lambda aggregator: get('pubmed', aggregator))
