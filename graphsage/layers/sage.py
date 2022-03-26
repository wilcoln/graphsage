from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_sparse import SparseTensor, matmul

from torch_geometric.utils import to_dense_batch


class SAGE(MessagePassing):
    """
    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        aggregator_type (str): ['mean', 'max', 'gcn', 'lstm'], mean by default
        out_channels (int): Size of each output sample.
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
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, aggregator_type: str = 'mean', normalize: bool = False,
                 root_weight: bool = True, bias: bool = True):

        super(SAGE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.aggregator_type = aggregator_type
        self.bias = bias
        
        assert self.aggregator_type in ['mean', 'max', 'lstm', 'gcn']

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.aggregator_type == 'gcn':
          # GCN does not require self root node, but just the neighbours
          self.root_weight = False 

        if aggregator_type == 'max':
          self.lin_pool = Linear(in_channels[0], in_channels[0])

        if self.aggregator_type == 'lstm':
          self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias) # neighbours
        if self.root_weight: # Not created for GCN
            self.lin_r = Linear(in_channels[1], out_channels, bias=False) # root itself

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialises learnable parameters
        """
        self.lin_l.reset_parameters()
        if self.root_weight:
          self.lin_r.reset_parameters()
        if self.aggregator_type == 'lstm':
          self.lstm.reset_parameters()
        if self.aggregator_type == 'max':
          self.lin_pool.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """
        Computes GraphSAGE Layer
        """

        if isinstance(x, Tensor):
          x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1] # x[1] -- root
        if self.root_weight and x_r is not None: 
            out += self.lin_r(x_r) # root doesn't get added for GCN

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor, edge_index_j, edge_index_i) -> Tensor:
      """
        Performs message Passing and aggregates messages from neighbours using the aggregator_type
      """
      adj_t = adj_t.set_value(None, layout=None)
      if self.aggregator_type == 'mean' or self.aggregator_type == 'gcn':
        return matmul(adj_t, x[0], reduce='mean')

      elif self.aggregator_type == 'max':
        return matmul(adj_t, x[0], reduce='max')
        # return torch.stack(x[0], dim=-1).max(dim=-1)[0] - alternative implementation of max pool operation
        
      elif self.aggregator_type == 'lstm':
        x_j = x[0][edge_index_j]
        x, mask = to_dense_batch(x_j, edge_index_i)
        return self.lstm(x)