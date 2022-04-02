import torch
import torch.nn.functional as F
from tqdm import tqdm

from graphsage import settings
from graphsage.layers import SAGE

device = settings.DEVICE


def set_batch_size_attr(batch):
    try:
        getattr(batch, 'batch_size')
    except AttributeError:
        batch.batch_size = batch.num_graphs

    return batch


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, aggregator, num_layers, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = hidden_channels

        self.layers = torch.nn.ModuleList(
            [SAGE(in_channels, hidden_channels, aggregator)] +
            [
                SAGE(hidden_channels, hidden_channels, aggregator)
                for _ in range(num_layers - 2)
            ] +
            [SAGE(hidden_channels, out_channels, aggregator)]
        )

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset))
        pbar.set_description('Evaluating')

        xs = []
        for batch in subgraph_loader:
            set_batch_size_attr(batch)
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index)
                if i < len(self.layers) - 1:
                    x = x.relu_()
                    x = F.dropout(x, p=0.5, training=self.training)
            xs.append(x[:batch.batch_size])
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
