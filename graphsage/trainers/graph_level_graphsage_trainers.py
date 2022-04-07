import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm

from .base_trainers import GraphSageBaseTrainer, SupervisedGraphSageBaseTrainer, dataloader_kwargs
from .. import settings


class SupervisedGraphSageTrainerForGraphLevelTask(SupervisedGraphSageBaseTrainer):
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 *args, **kwargs):
        super(SupervisedGraphSageTrainerForGraphLevelTask, self).__init__(*args, **kwargs)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train(self, epoch):
        self.model.train()

        pbar = tqdm(total=int(len(self.train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            y = batch.y
            y_hat = self.model(batch.x, batch.edge_index)
            loss = self.loss_fn(y_hat, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pbar.update(batch.num_graphs)

        pbar.close()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval(self, loader):
        self.model.eval()

        y, y_hat = [], []
        for data in loader:
            y.append(data.y)
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            y_hat.append((out > 0).float().cpu())

        y, y_hat = torch.cat(y, dim=0).numpy(), torch.cat(y_hat, dim=0).numpy()
        return f1_score(y, y_hat, average='micro')

    @torch.no_grad()
    def test(self):
        train_f1 = self.eval(self.train_loader) if not settings.NO_EVAL_TRAIN else None
        val_f1 = self.eval(self.val_loader)
        test_f1 = self.eval(self.test_loader)

        return {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        }


class UnsupervisedGraphSageTrainerForGraphLevelTask(GraphSageBaseTrainer):
    def __init__(self,
                 train_loader,
                 val_loader,
                 loader,
                 test_loader,
                 *args, **kwargs):
        super(UnsupervisedGraphSageTrainerForGraphLevelTask, self).__init__(*args, **kwargs)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_data_list = [data for data in train_loader.dataset]

        self.train_loader_list = []

        for curr_graph in self.train_data_list:
            _train_loader = loader(curr_graph, input_nodes=None, num_neighbors=[self.k1, self.k2], shuffle=True,
                                   **dataloader_kwargs)
            self.train_loader_list.append(_train_loader)

    def train(self, epoch):
        self.model.train()

        total_loss = 0
        total_num_nodes = 0

        for i in tqdm(range(len(self.train_loader_list))):  # loop over all 20 train loaders
            # add up current train loaders # of nodes to total_num_nodes
            total_num_nodes += self.train_data_list[i].num_nodes
            # update the value for x and edge index for the current data
            # train over the current graph
            for batch in self.train_loader_list[i]:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)[:batch.batch_size]
                loss = self.unsup_loss_fn(out, batch, self.train_data_list[i])

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * out.size(0)

        return total_loss / total_num_nodes

    def _loader_to_embeddings_and_labels(self, loader):
        xs, ys = [], []
        for data in loader:
            ys.append(data.y)
            xs.append(self.model(data.x.to(self.device), data.edge_index.to(self.device)))

        return torch.cat(xs, dim=0).cpu().numpy(), torch.cat(ys, dim=0).cpu().numpy()

    @torch.no_grad()
    def test(self):
        self.model.eval()

        # Create classifier
        clf = MultiOutputClassifier(SGDClassifier(loss="log", penalty="l2"))

        # Train classifier on train data
        train_embeddings, train_labels = self._loader_to_embeddings_and_labels(self.train_loader)
        clf.fit(train_embeddings, train_labels)

        # Evaluate on training set
        if not settings.NO_EVAL_TRAIN:
            train_predictions = clf.predict(train_embeddings)
            train_f1 = f1_score(train_labels, train_predictions, average='micro')
        else:
            train_f1 = None

        # Evaluate on validation set
        val_embeddings, val_labels = self._loader_to_embeddings_and_labels(self.val_loader)
        val_predictions = clf.predict(val_embeddings)
        val_f1 = f1_score(val_labels, val_predictions, average='micro')

        # Evaluate on validation set
        test_embeddings, test_labels = self._loader_to_embeddings_and_labels(self.test_loader)
        test_predictions = clf.predict(test_embeddings)
        test_f1 = f1_score(test_labels, test_predictions, average='micro')

        return {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        }
