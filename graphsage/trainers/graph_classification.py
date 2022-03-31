import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F

from graphsage import settings
from .base_trainers import BaseTrainer, SupervisedBaseTrainer


class SupervisedTrainerForGraphClassification(SupervisedBaseTrainer):
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 *args, **kwargs):
        super(SupervisedTrainerForGraphClassification, self).__init__(*args, **kwargs)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train(self, epoch):
        self.model.train()

        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(data.x, data.edge_index), data.y)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            self.optimizer.step()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval(self, loader):
        self.model.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y)
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            preds.append((out > 0).float().cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

    @torch.no_grad()
    def test(self):
        train_f1 = self.eval(self.train_loader)
        val_f1 = self.eval(self.val_loader)
        test_f1 = self.eval(self.test_loader)

        return {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        }


class UnsupervisedTrainerForGraphClassification(BaseTrainer):
    def __init__(self,
                 train_loader,
                 val_loader,
                 sampler,
                 test_loader,
                 *args, **kwargs):
        super(UnsupervisedTrainerForGraphClassification, self).__init__(*args, **kwargs)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_data_list = [data for data in train_loader.dataset]

        self.train_loader_list = []

        for curr_graph in self.train_data_list:
            _train_loader = sampler(curr_graph.edge_index, sizes=[25, 10], batch_size=settings.BATCH_SIZE, shuffle=True,
                                           num_nodes=curr_graph.num_nodes)
            self.train_loader_list.append(_train_loader)

    def train(self, epoch):
        self.model.train()

        total_loss = 0
        total_num_nodes = 0

        for i in tqdm(range(len(self.train_loader_list))):  # loop over all 20 train loaders
            # add up current train loaders # of nodes to total_num_nodes
            total_num_nodes += self.train_data_list[i].num_nodes
            # update the value for x and edge index for the current data
            x, edge_index = self.train_data_list[i].x.to(self.device), self.train_data_list[i].edge_index.to(
                self.device)
            for batch_size, n_id, adjs in self.train_loader_list[i]:  # train over the current graph
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]
                self.optimizer.zero_grad()  # set the gradients to zero

                out = self.model(x[n_id], adjs)
                out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

                pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
                neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
                loss = -pos_loss - neg_loss
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss) * out.size(0)

        return total_loss / total_num_nodes

    def _loader_to_embeddings_and_labels(self, model, loader):
        xs, ys = [], []
        for data in loader:
            ys.append(torch.argmax(data.y, dim=1))
            x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
            out = model.full_forward(x, edge_index)
            xs.append(out)
        return torch.cat(xs, dim=0).cpu(), torch.cat(ys, dim=0).cpu()


    @torch.no_grad()
    def test(self):
        self.model.eval()

        # Create classifier
        clf = SGDClassifier(loss="log", penalty="l2")

        # Train classifier on train data
        train_embeddings, train_labels = self._loader_to_embeddings_and_labels(self.model, self.train_loader)
        clf.fit(train_embeddings, train_labels)
        train_predictions = clf.predict(train_embeddings)
        train_f1 = f1_score(train_labels, train_predictions, average='micro')

        # Evaluate on validation set
        val_embeddings, val_labels = self._loader_to_embeddings_and_labels(self.model, self.val_loader)
        val_predictions = clf.predict(val_embeddings)
        val_f1 = f1_score(val_labels, val_predictions, average='micro')

        # Evaluate on validation set
        test_embeddings, test_labels = self._loader_to_embeddings_and_labels(self.model, self.test_loader)
        test_predictions = clf.predict(test_embeddings)
        test_f1 = f1_score(test_labels, test_predictions, average='micro')

        return {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        }


