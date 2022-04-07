import os.path as osp

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from graphsage.datasets import Planetoid


def main():
    print("Main script")
    dataset_name = 'Cora'
    path = osp.join(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data'), dataset_name)
    dataset = Planetoid(path, dataset_name)

    data = dataset[0]
    randomClassifier = DummyClassifier()

    randomClassifier.fit(data.x[data.train_mask], data.y[data.train_mask])

    print('Cora', 'F1-score Random Classifier: ',
          f1_score(data.y[data.test_mask], randomClassifier.predict(data.x[data.test_mask]), average='micro'))


if __name__ == '__main__':
    main()
