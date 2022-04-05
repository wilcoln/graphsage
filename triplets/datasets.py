import os.path as osp

import torch

from graphsage import settings
from graphsage.datasets import Planetoid
from triplets.models import MLP
from triplets.trainers import TripletMLPTrainer
from triplets.utils import pyg_graph_to_triplets

device = settings.DEVICE
dataset_name = 'Cora'  # 'Cora'  # 'Citeseer', 'Cora', # 'PubMed'
path = osp.join(settings.DATA_DIR, dataset_name)
dataset = Planetoid(path, dataset_name)
# Create the triplets dataset
td = pyg_graph_to_triplets(dataset)


# Train a simple model on the dataset
# region Multi-output classifier
# clf = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=5, tol=None))
# clf.fit(td.x[td.train_mask], td.y[td.train_mask])
#
# # Evaluation
# # Get labels for individual nodes
# y_val = td.y[td.val_mask][:, 0]
# y_test = td.y[td.test_mask][:, 0]
# # Get predictions for individual nodes
# y_hat_val = clf.predict(td.x[td.val_mask])[:, 0]
# y_hat_test = clf.predict(td.x[td.test_mask])[:, 0]

# # Compute F1 score
# ic(f1_score(y_hat_val, y_val, average='micro'))
# ic(f1_score(y_hat_test, y_test, average='micro'))
# endregion

# region SGDClassifier
# clf = SGDClassifier(loss='log', max_iter=5, tol=None)
# td.y = td.y[:, 0]
# clf.fit(td.x[td.train_mask], td.y[td.train_mask])
#
# # Evaluation
# # Get labels for individual nodes
# y_val = td.y[td.val_mask]
# y_test = td.y[td.test_mask]
# # Get predictions for individual nodes
# y_hat_val = clf.predict(td.x[td.val_mask])
# y_hat_test = clf.predict(td.x[td.test_mask])

# # Compute F1 score
# ic(f1_score(y_hat_val, y_val, average='micro'))
# ic(f1_score(y_hat_test, y_test, average='micro'))
# endregion

# region MLP classifier
td.y = td.y[:, 0]
model = MLP(
    in_channels=td.x.shape[1],
    num_layers=2,
    hidden_channels=256,
    out_channels=td.num_classes
).to(device)

TripletMLPTrainer(
    dataset_name=dataset_name,
    model=model,
    data=td,
    num_epochs=settings.NUM_EPOCHS,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device=device,
).run()
# endregion
