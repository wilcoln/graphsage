import torch
import os.path as osp

from sklearn.multioutput import MultiOutputClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
import torch_geometric.transforms as T

def main():

    print("Main script")
    path = osp.join(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data'), 'PPI')
    dataset = PPI(path, transform=T.NormalizeFeatures())

    train_dataset = PPI(path, split='train')
    test_dataset = PPI(path, split='test')

    randomClassifier = MultiOutputClassifier(DummyClassifier())

    randomClassifier.fit(train_dataset.data.x, train_dataset.data.y)

    print('PPI', 'F1-score Random Classifier: ', f1_score(test_dataset.data.y, randomClassifier.predict(test_dataset.data.x), average='micro'))

if __name__ == '__main__':

    main()