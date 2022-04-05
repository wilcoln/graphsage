import os.path as osp

from graphsage import settings
from graphsage.datasets import Reddit
from graphsage.trainers import RawFeaturesTrainerForNodeLevelTask


def get():
    dataset_name = 'Reddit'
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Reddit(path)
    data = dataset[0]

    return RawFeaturesTrainerForNodeLevelTask(data)
