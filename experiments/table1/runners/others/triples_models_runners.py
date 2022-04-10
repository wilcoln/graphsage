import os.path as osp

import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

import experiments.fig3.settings as fig3_settings
import graphsage.models.triples
from graphsage import settings
from graphsage.datasets import Planetoid  # , Reddit, Flickr
from graphsage.datasets.triples import pyg_graph_to_triples
from graphsage.trainers.node_level_triples_models_trainers import UnsupervisedTriplesTorchModuleTrainer, \
    SupervisedTriplesTorchModuleTrainer


def get(dataset_name, training_mode, model_name):
    if training_mode == 'supervised':
        if 'mlp' in model_name:
            return TriplesMultiLayerPerceptronRunner(2, dataset_name)
        elif 'logreg' in model_name:
            return TriplesLogisticRegressionRunner(dataset_name)
    else:
        if 'mlp' in model_name:
            return UnsupervisedTriplesMultiLayerPerceptronRunner(2, dataset_name)
        raise NotImplementedError


class TriplesModelRunner:
    def __init__(self, dataset_name):

        if dataset_name in {'cora', 'citeseer', 'pubmed'}:
            dataset_name = dataset_name.capitalize()
            path = osp.join(settings.DATA_DIR, dataset_name)
            dataset = Planetoid(path, dataset_name)
        # elif dataset_name == 'reddit':
        #     path = osp.join(settings.DATA_DIR, dataset_name.capitalize())
        #     dataset = Reddit(path)
        # elif dataset_name == 'flickr':
        #     dataset_name = dataset_name.capitalize()
        #     path = osp.join(settings.DATA_DIR, dataset_name)
        #     dataset = Flickr(path)
        else:
            raise NotImplementedError

        self.dataset_name = dataset_name
        self.td = pyg_graph_to_triples(dataset)


class TriplesLogisticRegressionRunner(TriplesModelRunner):
    def run(self):
        self.td.y = self.td.y[:, 0]

        # Train a simple model on the dataset
        clf = SGDClassifier(loss='log', max_iter=5, tol=None)
        clf.fit(self.td.x[self.td.triple_train_mask], self.td.y[self.td.triple_train_mask])

        # Get labels for individual nodes
        y_test = self.td.y[self.td.test_mask]
        # Get predictions for individual nodes
        y_hat_test = clf.predict(self.td.x[self.td.test_mask])

        # Compute & return F1 score
        return {'test_f1': f1_score(y_hat_test, y_test, average='micro')}


class TriplesMultiLayerPerceptronRunner(TriplesModelRunner):
    def __init__(self, num_layers, dataset_name):
        super().__init__(dataset_name)
        self.num_layers = num_layers

    def run(self):
        # Train an mlp on the dataset
        model = graphsage.models.triples.TriplesMLP(
            in_channels=self.td.x.shape[1],
            num_layers=self.num_layers,
            hidden_channels=fig3_settings.HIDDEN_CHANNELS,
            out_channels=self.td.num_classes,
        ).to(settings.DEVICE)

        return SupervisedTriplesTorchModuleTrainer(
            dataset_name=self.dataset_name,
            model=model,
            data=self.td,
            num_epochs=settings.NUM_EPOCHS,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=fig3_settings.LEARNING_RATE, weight_decay=1e-1),
            device=settings.DEVICE,
        ).run()


class UnsupervisedTriplesMultiLayerPerceptronRunner(TriplesModelRunner):
    def __init__(self, num_layers, dataset_name):
        super().__init__(dataset_name)
        self.num_layers = num_layers

    def run(self):
        # Train an mlp on the dataset
        model = graphsage.models.triples.TriplesMLP(
            in_channels=self.td.x.shape[1],
            num_layers=self.num_layers,
            hidden_channels=fig3_settings.HIDDEN_CHANNELS,
        ).to(settings.DEVICE)

        return UnsupervisedTriplesTorchModuleTrainer(
            dataset_name=self.dataset_name,
            model=model,
            data=self.td,
            num_epochs=settings.NUM_EPOCHS,
            optimizer=torch.optim.Adam(model.parameters(), lr=fig3_settings.LEARNING_RATE, weight_decay=1e-1),
            device=settings.DEVICE,
        ).run()
