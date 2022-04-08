import os.path as osp

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from graphsage.datasets import Reddit


def main():
    print("Main script")
    path = osp.join(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data'), 'Reddit')
    dataset = Reddit(path)

    data = dataset[0]
    rawFeaturesClassifier = SGDClassifier(loss='log', max_iter=5, tol=None)

    rawFeaturesClassifier.fit(data.x[data.train_mask], data.y[data.train_mask])
    print('Reddit', 'F1-score Raw Features Classifier: ',
          f1_score(data.y[data.test_mask], rawFeaturesClassifier.predict(data.x[data.test_mask]), average='micro'))


if __name__ == '__main__':
    main()
