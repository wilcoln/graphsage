import os.path as osp

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from graphsage.datasets import PPI


def main():
    print("Main script")
    path = osp.join(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data'), 'PPI')

    train_dataset = PPI(path, split='train')
    test_dataset = PPI(path, split='test')

    rawFeaturesClassifier = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=5, tol=None))

    rawFeaturesClassifier.fit(train_dataset.data.x, train_dataset.data.y)
    print('PPI', 'F1-score Raw Features Classifier: ',
          f1_score(test_dataset.data.y, rawFeaturesClassifier.predict(test_dataset.data.x), average='micro'))


if __name__ == '__main__':
    main()
