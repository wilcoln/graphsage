import os
import os.path as osp

import torch

import graphsage.settings as settings


def singles_to_triples(data, edge_index):
    heads = data[edge_index[0]]  # (E, D)
    tails = data[edge_index[1]]  # (E, D)

    # Add triples
    triples = torch.cat([heads, tails], dim=1)  # (E, 2D)
    # add inverse triples
    inverse_triples = torch.cat([tails, heads], dim=1)  # (E, 2*D)
    triples = torch.cat([triples, inverse_triples], dim=0)  # (2E, 2D)
    # add self-loops
    self_loops = torch.cat([data, data], dim=1)  # (V, D)
    triples = torch.cat([triples, self_loops], dim=0)  # (2E + V, 2D)

    return triples  # triple


def get_triples_mask(mask, edge_index, triple=False):
    triples_mask = mask[edge_index[0]]
    inverse_triples_mask = mask[edge_index[1]]
    triples_mask = torch.cat([triples_mask, inverse_triples_mask], dim=0)
    triples_mask = triples_mask if triple else torch.zeros_like(triples_mask)
    triples_mask = torch.cat([triples_mask, mask], dim=0)
    return triples_mask


def pyg_graph_list_to_triples_graph_list(dataset):
    triples_graph_list = []
    for data in dataset:
        # Create triple features
        x = singles_to_triples(data.x, data.edge_index)
        # Create triples labels
        y = data.y
        triples_graph_list.append((x, y))

    return triples_graph_list


def pyg_graph_to_triples(dataset):
    dataset_name = dataset.name if hasattr(dataset, 'name') else dataset.__class__.__name__
    triples_data_dir = osp.join(settings.DATA_DIR, 'triples', dataset_name)
    triples_data_path = osp.join(triples_data_dir, f'{dataset_name}.pt')
    if not osp.exists(triples_data_path):
        print(f'Creating triples for {dataset_name}')
        os.makedirs(triples_data_dir)
        data = dataset[0]
        # Create triple features
        x = singles_to_triples(data.x, data.edge_index)
        # Create triples labels
        y = singles_to_triples(data.y.view(-1, 1), data.edge_index)
        # Get new split masks
        triple_train_mask = get_triples_mask(data.train_mask, data.edge_index, triple=True)
        train_mask = get_triples_mask(data.train_mask, data.edge_index)
        val_mask = get_triples_mask(data.val_mask, data.edge_index)
        test_mask = get_triples_mask(data.test_mask, data.edge_index)
        triple_dataset = TripleDataset(
            name=dataset_name, x=x, y=y, num_classes=dataset.num_classes,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
            triple_train_mask=triple_train_mask,
        )
        print(f'Saving triples dataset to: {triples_data_dir}')
        torch.save(triple_dataset, triples_data_path)
    else:
        print(f'Loading triples dataset from: {triples_data_dir}')
        triple_dataset = torch.load(triples_data_path)

    return triple_dataset


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, triple_train_mask, train_mask, val_mask, test_mask, name, num_classes):
        self.x = x
        self.y = y
        self.triple_train_mask = triple_train_mask
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.name = name
        self.num_classes = num_classes

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def mask2index(mask):
    indices = torch.arange(0, mask.size(0))
    return indices[mask]


def get_cycles(num_features):
    # Code inspired from practicals N° 1
    import networkx as nx

    # Part 1: Generating a cycle dataset

    # # Task 1.1
    from datasets.triples import pyg_graph_list_to_triples_graph_list

    def decompose(n):
        """Get disjoint decomposition of n"""
        result = []
        for i in range(3, n - 2):
            result.append((i, n - i))
        return result

    def generate_fcg(n):
        """Build the cycle graph of n nodes"""
        return nx.cycle_graph(n)

    def generate_dcg(n, m):
        """Build the disjoint graph of n and m nodes"""
        return nx.disjoint_union(generate_fcg(n), generate_fcg(m))

    fcg_list = []  # full cycle graphs
    dcg_list = []  # disjoint cycle graphs
    for n in range(6, 16):
        decompositions = decompose(n)
        for n1, n2 in decompositions:
            fcg_list.append(generate_fcg(n))
            dcg_list.append(generate_dcg(n1, n2))

    # # Task 1.2
    import torch
    from torch_geometric.utils.convert import from_networkx

    pyg_fcg_list = []
    pyg_dcg_list = []
    for fcg, dcg in zip(fcg_list, dcg_list):
        pyg_fcg = from_networkx(fcg)
        pyg_dcg = from_networkx(dcg)

        pyg_fcg.x = torch.ones(fcg.number_of_nodes(), num_features)
        pyg_fcg.y = torch.tensor([1])

        pyg_dcg.x = torch.ones(dcg.number_of_nodes(), num_features)
        pyg_dcg.y = torch.tensor([0])

        pyg_fcg_list.append(pyg_fcg)
        pyg_dcg_list.append(pyg_dcg)

    # Dataset
    half_data_size = len(pyg_dcg_list)
    shuffle = torch.randperm(half_data_size)

    # Shuffle dataset
    pyg_fcg_list = [pyg_fcg_list[i] for i in shuffle]
    pyg_dcg_list = [pyg_dcg_list[i] for i in shuffle]

    # Split dataset
    Ntrain = int(.8*half_data_size)
    train_dataset, test_dataset = [], []

    for i, (pyg_fcg, pyg_dcg) in enumerate(zip(pyg_fcg_list, pyg_dcg_list)):
        dataset = train_dataset if i < Ntrain else test_dataset
        dataset.append(pyg_fcg)
        dataset.append(pyg_dcg)

    # Create train and test loaders
    train_dataset = pyg_graph_list_to_triples_graph_list(train_dataset)
    val_dataset = pyg_graph_list_to_triples_graph_list(test_dataset[:len(test_dataset)//2])
    test_dataset = pyg_graph_list_to_triples_graph_list(test_dataset[len(test_dataset)//2:])

    return train_dataset, val_dataset, test_dataset
