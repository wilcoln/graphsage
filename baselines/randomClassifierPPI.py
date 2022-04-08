import os.path as osp

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from graphsage.datasets import PPI


def main():
    print("Main script")
    path = osp.join(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data'), 'PPI')

    train_dataset = PPI(path, split='train')
    test_dataset = PPI(path, split='test')

    randomClassifier = MultiOutputClassifier(DummyClassifier())

    randomClassifier.fit(train_dataset.data.x, train_dataset.data.y)

    print('PPI', 'F1-score Random Classifier: ',
          f1_score(test_dataset.data.y, randomClassifier.predict(test_dataset.data.x), average='micro'))


if __name__ == '__main__':
    main()
