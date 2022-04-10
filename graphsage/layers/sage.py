import random
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTMCell
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul
from graphsage import settings
import numpy as np

class SAGE(MessagePassing):
    """
    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggregator_type (str): ['mean', 'max', 'gcn', 'lstm', 'bilstm', 'sum'], mean by default
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
            For eg: to inherit the aggregation function implementation from
            `torch_geometric.nn.conv.MessagePassing`, set (aggr = 'func_name')
            where func_name is in ['mean', 'sum', 'add', 'min', 'max', 'mul'];
            additionally, set the flow direction of message passing by passing
            the flow argument as either (flow = 'source_to_target') or
            (flow = 'target_to_source')
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, aggregator: str = 'mean',
                 normalize: bool = True, root_weight: bool = True, **kwargs):

        super(SAGE, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.aggregator = aggregator

        assert self.aggregator in ['mean', 'max',  'sum', 'gcn', 'lstm', 'bilstm', 'max_pool', 'mean_pool', None]

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.aggregator == 'gcn':
            # Convolutional aggregator does not concatenate the root node
            # i.e it doesn't concatenate the nodes previous layer
            self.root_weight = False

        if self.aggregator == 'lstm':
            self.lstm_n_inputs = settings.LSTM_NUM_INPUTS
            self.lstm = LSTMCell(self.lstm_n_inputs * in_channels[0], in_channels[0])

        if self.aggregator == 'bilstm':
            self.bilstm = torch.nn.LSTM(in_channels[0], in_channels[0], bidirectional=True, batch_first=True)
            self.att = Linear(2 * in_channels[0], 1)

        if self.aggregator in {'max_pool', 'mean_pool'}:
            self.pool = Linear(in_channels[0], in_channels[0], bias=True)

        self.lin_l = Linear(in_channels[0], out_channels, bias=False)  # neighbours

        if self.root_weight:  # Not created for GCN
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)  # root itself

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialises learnable parameters
        """
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        if self.aggregator == 'lstm':
            self.lstm.reset_parameters()
        if self.aggregator == 'bilstm':
            self.bilstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """
        Computes GraphSAGE Layer
        """
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate internally calls message_and_aggregate()
        # if edge_index is a SparseTensor and message_and_aggregate() is implemented,
        # otherwise it calls message(), aggregate() separately
        if self.aggregator in {'bilstm'} and isinstance(edge_index, Tensor):
            num_nodes = int(edge_index.max()) + 1
            edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

        if self.aggregator == 'gcn':
            root_indices = torch.unique(edge_index[1])
            self_edge_index = torch.stack([root_indices, root_indices], dim=0)
            edge_index = torch.cat([edge_index, self_edge_index], dim=1)

        out = self.propagate(edge_index, x=x, size=size)

        if self.aggregator == 'lstm':   
            out =  F.pad(out, (0, 0, 0, x[1].shape[0] - out.shape[0]), value=0)

        out = self.lin_l(out)

        # updates node embeddings 
        x_r = x[1]  # x[1] -- root
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)  # root doesn't get added for GCN

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def _get_neighbors(self, node_idx, edge_index_i: Tensor, edge_index_j: Tensor) -> list:
        """Return the neighbors of a node"""
        return list(set(edge_index_j[edge_index_i == node_idx].cpu().numpy().tolist()))


    def aggregate(self, inputs: Tensor, index: Tensor, edge_index_j, edge_index_i,
                  ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                  aggr: Optional[str] = None) -> Tensor:
        if self.aggregator == 'lstm':
            # Get neighbours of all inputs
            node_neighbors_dict = {
                node: self._get_neighbors(node, edge_index_i, edge_index_j)
                for node in set(edge_index_j.cpu().numpy().tolist())
            }

            # Truncate neighbours to self.lstm_n_inputs randomly
            node_neighbors_dict = {
                node: random.sample(neighbors, self.lstm_n_inputs)
                if len(neighbors) > self.lstm_n_inputs else neighbors
                for node, neighbors in node_neighbors_dict.items()
            }

            node_neighbors_embeddings_dict = {}
            for node, neighbors in node_neighbors_dict.items():
                try:
                    node_neighbors_embeddings_dict[node] = torch.cat((inputs[neighbors], inputs[neighbors][0,:].repeat(self.lstm_n_inputs - len(neighbors),1)))
                except:
                    node_neighbors_embeddings_dict[node] = F.pad(inputs[neighbors], (0, 0, 0, self.lstm_n_inputs - len(neighbors)), value=np.random.randn())

            # # Get the node embeddings of the neighbours & pad them to self.lstm_n_inputs
            # node_neighbors_embeddings_dict = {
            #     node: torch.cat((inputs[neighbors], inputs[neighbors][0,:].repeat(self.lstm_n_inputs - len(neighbors),1)))# F.pad(inputs[neighbors], (0, 0, 0, self.lstm_n_inputs - len(neighbors)), value=0)
            #     for node, neighbors in node_neighbors_dict.items()
            # }

            del node_neighbors_dict

            # Flatten the embeddings
            node_neighbors_sequence_embeddings_dict = {}
            for node, neighbors_embeddings in node_neighbors_embeddings_dict.items():
                node_neighbors_sequence_embeddings_dict[node] = neighbors_embeddings.view(1, -1)
                del neighbors_embeddings

            del node_neighbors_embeddings_dict

            # Aggregate the embeddings using LSTM
            node_aggregated_neighbors_embeddings_dict = {}
            for node, neighbors_sequence_embeddings in node_neighbors_sequence_embeddings_dict.items():
                node_aggregated_neighbors_embeddings_dict[node], _ = self.lstm(neighbors_sequence_embeddings)
                del _
                del neighbors_sequence_embeddings

            del node_neighbors_sequence_embeddings_dict

            return torch.cat(list(node_aggregated_neighbors_embeddings_dict.values()), dim=0)

        elif self.aggregator in {'mean', 'max',  'sum'}:
            reduce = self.aggregator

        elif self.aggregator in 'gcn':
            reduce = 'mean'

        elif self.aggregator == 'max_pool':
            inputs = F.relu(self.pool(inputs))
            reduce = 'max'

        elif self.aggregator == 'mean_pool':
            inputs = F.relu(self.pool(inputs))
            reduce = 'mean'

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=reduce)


    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor, edge_index_j, edge_index_i) -> Tensor:
        """
        Performs both message passing and aggregation of messages from neighbours using the aggregator
        """
        adj_t = adj_t.set_value(None, layout=None)

        if self.aggregator in {'mean', 'max',  'sum'}:
            return matmul(adj_t, x[0], reduce=self.aggregator)

        if self.aggregator == 'gcn':
            return matmul(adj_t, x[0], reduce='mean')

        if self.aggregator == 'max_pool':
            x = F.relu(self.pool(x[0]))
            return matmul(adj_t, x, reduce='max')

        if self.aggregator == 'mean_pool':
            x = F.relu(self.pool(x[0]))
            return matmul(adj_t, x, reduce='mean')

        elif self.aggregator == 'lstm':
            x_j = x[0][edge_index_j]
            x, mask = to_dense_batch(x_j, edge_index_i)
            _, (rst, _) = self.lstm(x)
            out = rst.squeeze(0)
            return out

        elif self.aggregator == 'bilstm':
            x = torch.stack(x, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.bilstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)
