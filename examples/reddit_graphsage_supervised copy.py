import copy
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm

# Our imports
from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.layers import SAGE
from graphsage.samplers import UniformLoader
from extensions.GraphSAINT.graphsaint import *

device = 'cpu'#settings.DEVICE
torch.set_num_threads(2)
SAINT_SAMPLER = settings.SAINT_SAMPLER
SAINT_SAMPLER_ARGS = settings.SAINT_SAMPLER_ARGS
EPS = 1e-15
sampler = 'RandomWalkSampler'
for sampler in SAINT_SAMPLER:
    #Select GraphSAINT sampler
    if sampler == 'NodeSampler':
        SAINT_sampler = GraphSAINTNodeSampler
    elif sampler == 'EdgeSampler':
        SAINT_sampler = GraphSAINTEdgeSampler
    elif sampler == 'RandomWalkSampler':
        SAINT_sampler = GraphSAINTRandomWalkSampler
    else:
        raise ValueError(f'Specified SAINT sampler {sampler} not available')
    
    #Set sampler settings
    sampler_settings = SAINT_SAMPLER_ARGS[sampler]
    sampler_settings.update({'batch_size': 1})#, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': True})
    
    path = osp.join(settings.DATA_DIR, 'Reddit')
    dataset = Reddit(path)

    data = dataset[0]

    loader = SAINT_sampler(data, **sampler_settings)


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


    model = GraphSAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(epoch):
        model.train()
        pbar = tqdm(total=int(len(loader.dataset)))
        pbar.set_description(f'Epoch {epoch}')

        total_loss = total_correct = total_examples = 0

        for d in loader:
            d = d.to(device)
            optimizer.zero_grad()

            out = model(d.x, d.edge_index)
            pred = out.argmax(dim=-1)
            loss = F.cross_entropy(out[d.train_mask], d.y[d.train_mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * d.num_nodes
            total_examples += d.num_nodes
            total_correct += int(pred.eq(d.y.to(device)).sum())


            pbar.update(1)

        return total_loss / total_examples, total_correct / total_examples


    @torch.no_grad()
    def test():
        model.eval()

        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=-1)
        correct = pred.eq(data.y.to(device))

        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            accs.append(correct[mask].sum().item() / mask.sum().item())
        return accs


    for epoch in range(1, 11):
        loss, acc = train(epoch)
        print(f'SAINT sampler {sampler} - Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test()
        print(f'SAINT sampler {sampler} Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    print(f'Result: SAINT sampler {sampler} Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    