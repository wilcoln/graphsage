import os.path as osp

import torch
import torch_geometric.transforms as T

import experiments.fig3.settings as fig3_settings
from graphsage import settings
from graphsage.datasets import Reddit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from graphsage.datasets import Planetoid


if fig3_settings.DATASET == 'reddit':
    path = osp.join(settings.DATA_DIR, fig3_settings.DATASET.capitalize())
    dataset = Reddit(path)


if fig3_settings.DATASET in {'cora', 'citeseer', 'pubmed'}:
    dataset_name = fig3_settings.DATASET.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())


class RawFeaturesRunner:
    def __init__(self, noise_prop):
        self.noise_prop = noise_prop

    def run(self):
        data = dataset[0]

        # Add noise to the feature matrix
        data.x = (1 - self.noise_prop)*data.x + self.noise_prop*torch.randn_like(data.x)

        rawFeaturesClassifier = SGDClassifier(loss='log', max_iter=5, tol=None)

        rawFeaturesClassifier.fit(data.x[data.train_mask], data.y[data.train_mask])

        test_f1 = f1_score(data.y[data.test_mask], rawFeaturesClassifier.predict(data.x[data.test_mask]), average='micro')

        return {'test_f1': test_f1}


def get(noise_prop):
    return RawFeaturesRunner(noise_prop)
