import os.path as osp
from types import SimpleNamespace

import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

import experiments.fig3.settings as fig3_settings
import triplets.models
from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.datasets import Reddit
from triplets.trainers import TripletMLPTrainer
from triplets.utils import pyg_graph_to_triplets, singles_to_triplets

if fig3_settings.DATASET == 'reddit':
    path = osp.join(settings.DATA_DIR, fig3_settings.DATASET.capitalize())
    dataset = Reddit(path)

if fig3_settings.DATASET in {'cora', 'citeseer', 'pubmed'}:
    dataset_name = fig3_settings.DATASET.capitalize()
    path = osp.join(settings.DATA_DIR, dataset_name)
    dataset = Planetoid(path, dataset_name)


def add_noise_and_convert_to_triplets(dataset, noise_prop):
    data = dataset[0]
    # Add noise to the feature matrix
    data.x = (1 - noise_prop)*data.x + noise_prop*torch.randn_like(data.x)

    # Create the triplets dataset
    td = pyg_graph_to_triplets(dataset)
    td.x = singles_to_triplets(data.x, data.edge_index)
    return td


class TripletsLogisticRegressionRunner:
    def __init__(self, noise_prop):
        self.noise_prop = noise_prop

    def run(self):
        # Create the noised triplets dataset
        td = add_noise_and_convert_to_triplets(dataset, self.noise_prop)

        # Train a simple model on the dataset
        clf = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=5, tol=None))
        clf.fit(td.x[td.train_mask], td.y[td.train_mask])

        # Get labels for individual nodes
        y_test = td.y[td.test_mask][:, 0]
        # Get predictions for individual nodes
        y_hat_test = clf.predict(td.x[td.test_mask])[:, 0]

        # Compute & return F1 score
        return {'test_f1': f1_score(y_hat_test, y_test, average='micro')}


class TripletsMultiLayerPerceptronRunner:
    def __init__(self, num_layers, noise_prop):
        self.noise_prop = noise_prop
        self.num_layers = num_layers

    def run(self):
        # Create the noised triplets dataset
        td = add_noise_and_convert_to_triplets(dataset, self.noise_prop)
        td.y = td.y[:, 0]

        # Train an mlp on the dataset
        model = triplets.models.MLP(
            in_channels=td.x.shape[1],
            num_layers=self.num_layers,
            hidden_channels=fig3_settings.HIDDEN_CHANNELS,
            out_channels=td.num_classes,
        ).to(settings.DEVICE)

        return TripletMLPTrainer(
            dataset_name=dataset_name,
            model=model,
            data=td,
            num_epochs=settings.NUM_EPOCHS,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=fig3_settings.LEARNING_RATE),
            device=settings.DEVICE,
        ).run()

logreg = SimpleNamespace(get=lambda noise_prop: TripletsLogisticRegressionRunner(noise_prop))
mlp2 = SimpleNamespace(get=lambda noise_prop: TripletsMultiLayerPerceptronRunner(2, noise_prop))
mlp3 = SimpleNamespace(get=lambda noise_prop: TripletsMultiLayerPerceptronRunner(3, noise_prop))
