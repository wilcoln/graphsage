import os.path as osp

from graphsage import settings
from graphsage.datasets import PPI
from graphsage.trainers import RandomTrainerForGraphLevelTask

device = settings.DEVICE

path = osp.join(settings.DATA_DIR, 'PPI')


def get():
    return RandomTrainerForGraphLevelTask(
        train_dataset=PPI(path, split='train'),
        test_dataset=PPI(path, split='test'),
    )
