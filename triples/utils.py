import os
import os.path as osp

import torch

import graphsage.settings as graphsage_settings


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


def pyg_graph_to_triples(dataset):
    triples_data_dir = osp.join(graphsage_settings.DATA_DIR, 'triples', dataset.name)
    triples_data_path = osp.join(triples_data_dir, f'{dataset.name}.pt')
    if not osp.exists(triples_data_path):
        print(f'Creating triples for {dataset.name}')
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
            name=dataset.name, x=x, y=y, num_classes=dataset.num_classes,
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
