import torch
import torch.nn.functional as F
from tqdm import tqdm

from graphsage import settings
from graphsage.layers import SAGE

device = settings.DEVICE


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
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description('Evaluating')

        for i, layer in enumerate(self.layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(device)].to(device)
                x = layer(x, batch.edge_index.to(device))
                if i < len(self.layers) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
