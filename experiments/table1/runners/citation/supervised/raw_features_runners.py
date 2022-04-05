import os.path as osp
from types import SimpleNamespace

from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.trainers import RawFeaturesTrainerForNodeLevelTask


def get(dataset_name):
    dataset_name = dataset_name.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name)
    data = dataset[0]

    return RawFeaturesTrainerForNodeLevelTask(data)


cora = SimpleNamespace(get=lambda: get('cora'))
citeseer = SimpleNamespace(get=lambda: get('citeseer'))
pubmed = SimpleNamespace(get=lambda: get('pubmed'))
