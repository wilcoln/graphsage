import os
import os.path as osp
from argparse import Namespace

import torch
from icecream import ic

import graphsage.settings as graphsage_settings

def singles_to_triplets(data, edge_index):
    heads = data[edge_index[0]]  # (E, D)
    tails = data[edge_index[1]]  # (E, D)

    # Add triplets
    triplets = torch.cat([heads, tails], dim=1)  # (E, 2D)
    # add inverse triplets
    inverse_triplets = torch.cat([tails, heads], dim=1)  # (E, 2*D)
    triplets = torch.cat([triplets, inverse_triplets], dim=0)  # (2E, 2D)
    # add self-loops
    self_loops = torch.cat([data, data], dim=1)  # (V, D)
    triplets = torch.cat([triplets, self_loops], dim=0)  # (2E + V, 2D)

    return triplets  # triplet


def get_triplets_mask(mask, edge_index, train=False):
    triplets_mask = mask[edge_index[0]]
    inverse_triplets_mask = mask[edge_index[1]]
    triplets_mask = torch.cat([triplets_mask, inverse_triplets_mask], dim=0)
    triplets_mask = triplets_mask if train else torch.zeros_like(triplets_mask)
    triplets_mask = torch.cat([triplets_mask, mask], dim=0)
    return triplets_mask


def pyg_graph_to_triplets(dataset):
    triplets_data_dir = osp.join(graphsage_settings.DATA_DIR, 'triplets', dataset.name)
    triplets_data_path = osp.join(triplets_data_dir, f'{dataset.name}.pt')
    if not osp.exists(triplets_data_path):
        print(f'Creating triplets for {dataset.name}')
        os.makedirs(triplets_data_dir)
        data = dataset[0]
        # Create triplet features
        x = singles_to_triplets(data.x, data.edge_index)
        # Create triplets labels
        y = singles_to_triplets(data.y.view(-1, 1), data.edge_index)
        # Get new split masks
        train_mask = get_triplets_mask(data.train_mask, data.edge_index, train=True)
        val_mask = get_triplets_mask(data.val_mask, data.edge_index)
        test_mask = get_triplets_mask(data.test_mask, data.edge_index)
        triplet_dataset = TripletDataset(
            name=dataset.name, x=x, y=y, num_classes=dataset.num_classes,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )
        print(f'Saving triplets dataset to: {triplets_data_dir}')
        torch.save(triplet_dataset, triplets_data_path)
    else:
        print(f'Loading triplets dataset from: {triplets_data_dir}')
        triplet_dataset = torch.load(triplets_data_path)

    return triplet_dataset


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train_mask, val_mask, test_mask, name, num_classes):
        self.x = x
        self.y = y
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
