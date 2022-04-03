from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from graphsage.trainers.base_trainers import BaseTrainer


class GraphLevelScikitLearnTrainer(BaseTrainer):
    def __init__(self, clf, train_dataset, test_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.clf = clf

    def test(self) -> dict:
        test_f1 = f1_score(
            self.test_dataset.data.y,
            self.clf.predict(self.test_dataset.data.x),
            average='micro'
        )
        return {'test_f1': test_f1}

    def train(self, *args, **kwargs) -> None:
        self.clf.fit(self.train_dataset.data.x, self.train_dataset.data.y)

    def run(self, *args, **kwargs) -> dict:
        self.train()
        return self.test()


class RawFeaturesTrainerForGraphLevelTask(GraphLevelScikitLearnTrainer):
    """
    Trainer for raw features.
    """
    def __init__(self, *args, **kwargs):
        clf = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=5, tol=None))
        super().__init__(clf, *args, **kwargs)


class RandomTrainerForGraphLevelTask(GraphLevelScikitLearnTrainer):
    """
    Trainer for ranom features.
    """
    def __init__(self, *args, **kwargs):
        clf = MultiOutputClassifier(DummyClassifier())
        super().__init__(clf, *args, **kwargs)
