import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F


class SupervisedTrainerForGraphClassification:
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 train_loader,
                 val_loader,
                 test_loader,
                 num_epochs,
                 device):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.train_micro_f1s = []
        self.val_micro_f1s = []
        self.test_micro_f1s = []

    def train(self):
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

    def run(self):
        for epoch in range(1, self.num_epochs + 1):
            loss = self.train()
            val_f1 = self.eval(self.val_loader)
            test_f1 = self.eval(self.test_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
                  f'Test: {test_f1:.4f}')

            self.train_losses.append(loss)
            self.val_micro_f1s.append(val_f1)
            self.test_micro_f1s.append(test_f1)


class UnsupervisedTrainerForGraphClassification:
    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 val_loader,
                 test_loader,
                 num_epochs,
                 train_loader_list,
                 train_data_list,
                 device):

        self.model = model
        self.optimizer = optimizer
        self.train_loader_list = train_loader_list
        self.train_data_list = train_data_list
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.train_micro_f1s = []
        self.val_micro_f1s = []
        self.test_micro_f1s = []

    def train(self):
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

        return train_f1, val_f1, test_f1

    def run(self):
        for epoch in range(1, self.num_epochs + 1):
            loss = self.train()
            train_f1, val_f1, test_f1 = self.test()
            print('Epoch: {:03d}, Loss: {:.4f}, Train F1: {:.4f}, Val F1: {:.4f}, Test F1: {:.4f}'
                  .format(epoch, loss, train_f1, val_f1, test_f1))

            self.train_losses.append(loss)
            self.train_micro_f1s.append(train_f1)
            self.val_micro_f1s.append(val_f1)
            self.test_micro_f1s.append(test_f1)

