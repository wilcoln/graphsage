import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSumOfLosses(nn.Module):
    def __init__(self, num_losses):
        super(WeightedSumOfLosses, self).__init__()
        self.sigma = nn.Parameter(torch.ones(num_losses))

    def forward(self, *losses):
        assert len(losses) == len(self.sigma), 'Number of losses must match number of weights'
        l = 0.5 * torch.Tensor(losses) / self.sigma**2
        l = l.sum() + torch.log(self.sigma.prod())
        return l


class MLP(nn.Module):
    """ Multi-layer perceptron. """

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=2, batch_norm=True):
        assert num_layers >= 1, "num_layers must be at least 1"
        super(MLP, self).__init__()

        self.name = f'{num_layers}-MLP'
        self.batch_norm = batch_norm

        out_channels = out_channels if out_channels is not None else hidden_channels

        self.layers = nn.ModuleList(
            [nn.Linear(in_channels, hidden_channels)] +
            [
                nn.Linear(hidden_channels, hidden_channels)
                for _ in range(num_layers - 2)
            ] +
            [nn.Linear(hidden_channels, out_channels)]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers-1)] +
            [nn.BatchNorm1d(out_channels)]
        )

    def forward(self, x):
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if self.batch_norm:
                x = batch_norm(x)
            if i < len(self.layers) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class TriplesMLP(nn.Module):
    """ Multi-layer perceptron for triples. """

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=2):
        super().__init__()
        self.phi = MLP(in_channels//2, hidden_channels, out_channels, num_layers)  # node encoder

    def forward(self, x):
        # compute the representation for each data point
        u, v = torch.split(x, x.shape[1] // 2, dim=1)
        u_phi = self.phi(u)  # representation of source node u
        v_phi = self.phi(v)  # representation of target node v

        return torch.cat([u_phi, v_phi], dim=1)

    def reset_parameters(self):
        self.phi.reset_parameters()


class GraphLevelTriplesMLP(nn.Module):

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=2, rni=False):
        super().__init__()
        self.triple_encoder = TriplesMLP(in_channels, hidden_channels, num_layers=num_layers)
        self.graph_encoder = MLP(2*hidden_channels, hidden_channels, out_channels, num_layers, False)
        self.rni = rni

    def forward(self, x):
        # compute the representation for each data point
        if self.rni:
            half_embed_size = x.shape[1]//2
            random_embeddings = torch.rand(x.shape[0], half_embed_size).to(x.device)
            x = torch.hstack((random_embeddings, x[:, half_embed_size:]))  # randomize node feature

        hidden = self.triple_encoder(x)
        out = torch.sum(hidden, dim=0, keepdim=True)
        out = self.graph_encoder(out)
        return hidden, out

    def reset_parameters(self):
        self.phi.reset_parameters()
