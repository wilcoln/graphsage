import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader

# Our imports
from graphsage import settings
from graphsage.datasets import PPI
from graphsage.layers import SAGE
from extensions.GraphSAINT.graphsaint import *

device = settings.DEVICE
SAINT_SAMPLER = settings.SAINT_SAMPLER
SAINT_SAMPLER_ARGS = settings.SAINT_SAMPLER_ARGS


path = osp.join(settings.DATA_DIR, 'PPI')
dataset = PPI(path)
dataset[0].data
train_dataset = PPI(path, split='train')
train_dataset[0]
val_dataset = PPI(path, split='val')
val_dataset[0]
test_dataset = PPI(path, split='test')

#Set sampler settings
sampler_settings = SAINT_SAMPLER_ARGS[sampler]
sampler_settings.update({'batch_size': settings.BATCH_SIZE})#, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': True})
data = dataset[0]
data
sampler = 'RandomWalkSampler'

train_loader = SAINT_sampler(dataset.data, **sampler_settings)
val_loader = SAINT_sampler(val_dataset.data, **sampler_settings)
test_loader = SAINT_sampler(test_dataset.data, **sampler_settings)

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


model = GraphSAGE(train_dataset.num_features, 256, train_dataset.num_classes).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train(epoch):
    model.train()
    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch}')

    total_loss = total_correct = total_examples = 0

    for d in train_loader:
        d = d.to(device)
        optimizer.zero_grad()

        out = model(d.x, d.edge_index)
        # pred = (out > 0).float().to(device)
        loss = loss_op(out, d.y)

        # pred.shape
        # d.y.shape
        # 124*121
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * d.num_nodes
        total_examples += d.num_nodes
        # total_correct += int(pred.eq(d.y.to(device)).sum())


        pbar.update(1)

    return total_loss / total_examples#, total_correct / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 11):
    loss = train(epoch)
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
          f'Test: {test_f1:.4f}')