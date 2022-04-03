import torch
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

from graphsage import settings
from graphsage.samplers import get_pos_neg_batches
from .base_trainers import GraphSageBaseTrainer, SupervisedGraphSageBaseTrainer


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
            y_hat = self.model(batch.x, batch.edge_index)[:batch.num_graphs]
            y = batch.y[:batch.num_graphs]
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

        y_hat = self.model.inference(loader)
        y_hat = (y_hat > 0).float()
        y = torch.cat([batch.y[:batch.num_graphs] for batch in loader]).to(self.device)
        return f1_score(y.cpu(), y_hat.cpu(), average='micro')

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

        loader_kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS}
        for curr_graph in self.train_data_list:
            _train_loader = loader(curr_graph, input_nodes=None, num_neighbors=[self.k1, self.k2], shuffle=True, **loader_kwargs)
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
                pos_batch, neg_batch = get_pos_neg_batches(batch, self.train_data_list[i])
                self.optimizer.zero_grad()

                out = self.model(batch.x, batch.edge_index)[:batch.batch_size]
                pos_out = self.model(pos_batch.x, pos_batch.edge_index)[:batch.batch_size]
                neg_out = self.model(neg_batch.x, neg_batch.edge_index)[:batch.batch_size]

                pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
                neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
                loss = -pos_loss - neg_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * out.size(0)

        return total_loss / total_num_nodes

    def _loader_to_embeddings_and_labels(self, model, loader):
        xs, ys = [], []
        for batch in loader:
            batch = batch.to(self.device)
            ys.append(torch.argmax(batch.y[:batch.num_graphs], dim=1))
            out = model(batch.x, batch.edge_index)[:batch.num_graphs]
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
