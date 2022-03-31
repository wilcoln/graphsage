import torch.nn.functional as F

from ... import settings
from ...models.supervised import GraphSAGE as GraphSAGESupervised

device = settings.DEVICE


class GraphSAGE(GraphSAGESupervised):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, adj_list):
        for i, (edge_index, _, size) in enumerate(adj_list):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.layers[i]((x, x_target), edge_index)
            if i != len(self.layers) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x
