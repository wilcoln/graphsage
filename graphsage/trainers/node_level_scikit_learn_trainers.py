from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from graphsage.trainers import BaseTrainer


class NodelLevelScikitLearnTrainer(BaseTrainer):
    def __init__(self, clf, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.clf = clf

    def test(self) -> dict:
        test_f1 = f1_score(
            self.data.y[self.data.test_mask],
            self.clf.predict(self.data.x[self.data.test_mask]),
            average='micro'
        )
        return {'test_f1': test_f1}

    def train(self, *args, **kwargs) -> None:
        self.clf.fit(self.data.x[self.data.train_mask], self.data.y[self.data.train_mask])

    def run(self, *args, **kwargs) -> dict:
        self.train()
        return self.test()


class RawFeaturesTrainerForNodeLevelTask(NodelLevelScikitLearnTrainer):
    """
    Trainer for raw features.
    """
    def __init__(self, *args, **kwargs):
        clf = SGDClassifier(loss='log', max_iter=5, tol=None)
        super().__init__(clf, *args, **kwargs)


class RandomTrainerForNodeLevelTask(NodelLevelScikitLearnTrainer):
    """
    Trainer for random features.
    """
    def __init__(self, *args, **kwargs):
        clf = DummyClassifier()
        super().__init__(clf, *args, **kwargs)