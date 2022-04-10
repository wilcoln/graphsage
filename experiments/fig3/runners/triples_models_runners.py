import os.path as osp
from types import SimpleNamespace

import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

import experiments.fig3.settings as fig3_settings
import graphsage.models.triples
from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.datasets import Reddit
from graphsage.datasets.triples import pyg_graph_to_triples, singles_to_triples
from graphsage.trainers.node_level_triples_models_trainers import SupervisedTriplesTorchModuleTrainer

if fig3_settings.DATASET == 'reddit':
    path = osp.join(settings.DATA_DIR, fig3_settings.DATASET.capitalize())
    dataset = Reddit(path)

if fig3_settings.DATASET in {'cora', 'citeseer', 'pubmed'}:
    dataset_name = fig3_settings.DATASET.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name)


def add_noise_and_convert_to_triples(dataset, noise_prop):
    data = dataset[0]
    # Add noise to the feature matrix
    data.x = (1 - noise_prop)*data.x + noise_prop*torch.randn_like(data.x)

    # Create the triples dataset
    td = pyg_graph_to_triples(dataset)
    td.x = singles_to_triples(data.x, data.edge_index)
    return td


class TriplesLogisticRegressionRunner:
    def __init__(self, noise_prop):
        self.noise_prop = noise_prop

    def run(self):
        # Create the noised triples dataset
        td = add_noise_and_convert_to_triples(dataset, self.noise_prop)
        td.y = td.y[:, 0]

        # Train a simple model on the dataset
        clf = SGDClassifier(loss='log', max_iter=5, tol=None)
        clf.fit(td.x[td.triple_train_mask], td.y[td.triple_train_mask])

        # Get labels for individual nodes
        y_test = td.y[td.test_mask]
        # Get predictions for individual nodes
        y_hat_test = clf.predict(td.x[td.test_mask])

        # Compute & return F1 score
        return {'test_f1': f1_score(y_hat_test, y_test, average='micro')}


class TriplesMultiLayerPerceptronRunner:
    def __init__(self, num_layers, noise_prop):
        self.noise_prop = noise_prop
        self.num_layers = num_layers

    def run(self):
        # Create the noised triples dataset
        td = add_noise_and_convert_to_triples(dataset, self.noise_prop)

        # Train an mlp on the dataset
        model = graphsage.models.triples.TriplesMLP(
            in_channels=td.x.shape[1],
            num_layers=self.num_layers,
            hidden_channels=fig3_settings.HIDDEN_CHANNELS,
            out_channels=td.num_classes,
        ).to(settings.DEVICE)

        return SupervisedTriplesTorchModuleTrainer(
            dataset_name=dataset_name,
            model=model,
            data=td,
            num_epochs=settings.NUM_EPOCHS,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=fig3_settings.LEARNING_RATE, weight_decay=1e-1),
            device=settings.DEVICE,
        ).run()


logreg = SimpleNamespace(get=lambda noise_prop: TriplesLogisticRegressionRunner(noise_prop))
mlp2 = SimpleNamespace(get=lambda noise_prop: TriplesMultiLayerPerceptronRunner(2, noise_prop))
