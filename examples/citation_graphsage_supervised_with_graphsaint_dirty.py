import copy
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm

# Our imports
from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.layers import SAGE
from graphsage.samplers import UniformLoader
from extensions.GraphSAINT.graphsaint import *

device = settings.DEVICE
###!!! TO BE DELETED
torch.set_num_threads(2)
SAINT_SAMPLER = settings.SAINT_SAMPLER
SAINT_SAMPLER_ARGS = settings.SAINT_SAMPLER_ARGS
EPS = 1e-15
sampler = 'NodeSampler'
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
    sampler_settings.update({'batch_size': settings.BATCH_SIZE})

    dataset = 'Cora'
    path = osp.join(settings.DATA_DIR, dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

    # Already send node features/labels to GPU for faster access during sampling:
    data = dataset[0].to(device, 'x', 'y')

    kwargs = {'batch_size': settings.BATCH_SIZE}#, 'num_workers': settings.NUM_WORKERS, 'persistent_workers': True}
    # train_loader = UniformLoader(data, input_nodes=data.train_mask,
    #                             num_neighbors=[25, 10], shuffle=True, **kwargs)

    # subgraph_loader = UniformLoader(copy.copy(data), input_nodes=None,
    #                                 num_neighbors=[-1], shuffle=False, **kwargs)

    subgraph_loader = UniformLoader(copy.copy(data), input_nodes=None,
                                        num_neighbors=[-1], shuffle=False, **kwargs)

    SAINT_SAMPLER = GraphSAINTEdgeSampler
    train_loader = SAINT_sampler(data, shuffle=True, sample_coverage=100, **kwargs)

    subgraph_loader = SAINT_sampler(copy.copy(data), sample_coverage=100,shuffle=False, **kwargs)

    train_loader.data#.batch_size = settings.BATCH_SIZE
    # No need to maintain these features during evaluation:
    del subgraph_loader.data.x, subgraph_loader.data.y
    # Add global node index information.
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)
    # subgraph_loader.data.batch_size = settings.BATCH_SIZE
    data.x.shape
    subgraph_loader.data.n_id
    data.n_id
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

        @torch.no_grad()
        def inference(self, x_all, subgraph_loader):
            pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
            pbar.set_description('Evaluating')

            # Compute representations of nodes layer by layer, using *all*
            # available edges. This leads to faster computation in contrast to
            # immediately computing the final representations of each batch:
            # x_all_subet = copy.copy(x_all)
            self = model
            conv = self.convs[0]
            x_all = data.x
            for i, conv in enumerate(self.convs):
                print(i,conv)
                xs = []
                for batch in subgraph_loader:
                    print(batch)
                    # break
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    x.shape
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                    xs.append(x.cpu())
                    pbar.update(settings.BATCH_SIZE)

                xs[0].shape
                len(xs)
                x_all = torch.cat(xs, dim=0)
                x_all.shape
                # x_all_temp = torch.cat(xs, dim=0)
            pbar.close()
            return x_all

    model = GraphSAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    def train(epoch):
        model.train()

        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = total_examples = 0


        for s in train_loader:            
            #Sampler
            # samples = SAINT_sampler(batch, **sampler_settings)
            # for s in samples:
            optimizer.zero_grad()
            y = s.y
            y_hat = model(s.x, s.edge_index.to(device))
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * s.batch_size
            total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            total_examples += s.batch_size
            pbar.update(s.batch_size)


            # y_hat = model(s.x, s.edge_index.to(device))
            # loss = F.cross_entropy(y_hat, y)
            # loss.backward()
            # optimizer.step()

            # total_loss += float(loss) * s.num_nodes
            # total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            # total_examples += s.num_nodes
            # pbar.update(s.num_nodes)
        pbar.close()

        return total_loss / total_examples, total_correct / total_examples


    @torch.no_grad()
    def test():
        model.eval()
        y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
        y = data.y.to(y_hat.device)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
        return accs


    for epoch in range(1, 11):
        loss, acc = train(epoch)
        print(f'SAINT sampler {sampler} - Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test()
        print(f'SAINT sampler {sampler} Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')