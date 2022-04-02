import torch
from torch_cluster import random_walk
from torch_geometric.data import Data
from torch_sparse import SparseTensor


def get_pos_neg_batches(batch, data):
    device = batch.x.device

    batch_edge_index = batch.edge_index
    batch_num_nodes = int(batch_edge_index.max()) + 1
    batch_edge_index = SparseTensor(
        row=batch_edge_index[0],
        col=batch_edge_index[1],
        sparse_sizes=(batch_num_nodes, batch_num_nodes)
    ).t()

    row, col, _ = batch_edge_index.coo()

    # For each node in `batch`, we sample a direct neighbor (as positive
    # example) and a random node (as negative example):
    pos_batch_n_id = random_walk(row, col, batch.n_id, walk_length=1, coalesced=False)[:, 1].cpu()

    neg_batch_n_id = torch.randint(0, data.num_nodes, (batch.n_id.numel(),), dtype=torch.long)

    pos_batch = Data(
        x=torch.index_select(data.x.cpu(), 0, pos_batch_n_id).to(device),
        n_id=pos_batch_n_id.to(device),
        edge_index=batch.edge_index.to(device),
    )
    neg_batch = Data(
        x=torch.index_select(data.x.cpu(), 0, neg_batch_n_id).to(device),
        n_id=neg_batch_n_id.to(device),
        edge_index=batch.edge_index.to(device),
    )

    return pos_batch, neg_batch

