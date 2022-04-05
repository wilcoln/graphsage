import os.path as osp

from graphsage import settings
from graphsage.datasets import PPI
from graphsage.trainers import RawFeaturesTrainerForGraphLevelTask

device = settings.DEVICE

path = osp.join(settings.DATA_DIR, 'PPI')


def get():
    return RawFeaturesTrainerForGraphLevelTask(
        train_dataset=PPI(path, split='train'),
        test_dataset=PPI(path, split='test'),
    )
