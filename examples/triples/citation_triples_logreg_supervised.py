import os.path as osp

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.datasets.triples import pyg_graph_to_triples

device = settings.DEVICE
dataset_name = settings.args.dataset if settings.args.dataset is not None else 'cora'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Planetoid(path, dataset_name)

# Create the triples dataset
td = pyg_graph_to_triples(dataset)


# Train a triple model on the dataset
# region SGDClassifier
clf = SGDClassifier(loss='log', max_iter=5, tol=None)
td.y = td.y[:, 0]
clf.fit(td.x[td.train_mask], td.y[td.train_mask])

# Evaluation
# Get labels for individual nodes
y_val = td.y[td.val_mask]
y_test = td.y[td.test_mask]
# Get predictions for individual nodes
y_hat_val = clf.predict(td.x[td.val_mask])
y_hat_test = clf.predict(td.x[td.test_mask])

# Compute F1 score
eval_f1 = f1_score(y_val, y_hat_val, average='micro')
test_f1 = f1_score(y_test, y_hat_test, average='micro')
print('Val F1: {:.4f}, Test F1: {:.4f}'.format(eval_f1, test_f1))
# endregion
