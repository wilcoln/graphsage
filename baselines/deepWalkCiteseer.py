import json
import os.path as osp

import torch

from deepWalkModel import Deepwalk
from graphsage.datasets import Planetoid


def main():
    print("Main script")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_name = 'Citeseer'
    path = osp.join(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data'), dataset_name)
    dataset = Planetoid(path, dataset_name)

    data = dataset[0]

    with open('./deepWalk_config.json', 'r') as config_file:
        config = json.load(config_file)

    model = Deepwalk(edge_index=data.edge_index, embedding_dim=config['embedding_dim'],
                     walk_length=config['walk_length'], context_size=config['context_size'],
                     walks_per_node=config['walks_per_node'], p=1, q=1,
                     num_negative_samples=config['num_negative_samples'], sparse=False)

    loader = model.loader(batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    def train():

        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test_no_features():
        model.eval()
        z = model()
        acc = model.test_no_features(z[data.train_mask], data.y[data.train_mask],
                                     z[data.test_mask], data.y[data.test_mask], max_iter=5000)
        return acc

    @torch.no_grad()
    def test_features():
        model.eval()
        z = model()
        acc = model.test_features(z[data.train_mask], data.y[data.train_mask],
                                  z[data.test_mask], data.y[data.test_mask], data.x[data.train_mask],
                                  data.x[data.test_mask], max_iter=5000)
        return acc

    for epoch in range(1, config['num_epochs'] + 1):
        loss = train()
        F1_no_features = test_no_features()
        F1_features = test_features()
        if epoch % 1 == 0:
            print(f'Citeseer: Epoch: {epoch:02d}, Loss: {loss:.4f}')
            print(f'        No Features F1: {F1_no_features:.4f}')
            print(f'        With Features F1: {F1_features:.4f}')


if __name__ == '__main__':
    main()
