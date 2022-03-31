import torch
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from tqdm import tqdm


class SupervisedTrainerForNodeClassification:
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 data,
                 train_loader,
                 subgraph_loader,
                 num_epochs,
                 device):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data = data
        self.train_loader = train_loader
        self.subgraph_loader = subgraph_loader
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

    def train(self, epoch):
        self.model.train()

        pbar = tqdm(total=int(len(self.train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            y_hat = self.model(batch.x, batch.edge_index.to(self.device))[:batch.batch_size]
            loss = self.loss_fn(y_hat, y)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * batch.batch_size
            total_examples += batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        return total_loss / total_examples


    @torch.no_grad()
    def test(self):
        self.model.eval()
        y_hat = self.model.inference(self.data.x, self.subgraph_loader).argmax(dim=-1)
        y = self.data.y.to(y_hat.device)

        accs = [
            int((y_hat[mask] == y[mask]).sum()) / int(mask.sum())
            for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]
        ]
        return accs

    def run(self):
        for epoch in range(1, self.num_epochs + 1):
            loss = self.train(epoch)
            train_acc, val_acc, test_acc = self.test()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Test Acc: {test_acc:.4f}')

        # Save results
        self.train_losses.append(loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.test_accuracies.append(test_acc)


class UnsupervisedTrainerForNodeClassification:
    def __init__(self,
                 model,
                 optimizer,
                 data,
                 train_loader,
                 subgraph_loader,
                 num_epochs,
                 device):

        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.train_loader = train_loader
        self.subgraph_loader = subgraph_loader
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

    def train(self, epoch):
        self.model.train()
        print(f'Epoch: {epoch:02d}', end='\r')

        total_loss = 0
        for batch_size, n_id, adj_list in tqdm(self.train_loader):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adj_list = [adj.to(self.device) for adj in adj_list]
            self.optimizer.zero_grad()

            out = self.model(self.data.x[n_id].to(self.device), adj_list)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * out.size(0)

        return total_loss / self.data.num_nodes


    @torch.no_grad()
    def test(self):
        self.model.eval()
        out = self.model.inference(self.data.x, self.subgraph_loader).cpu()

        clf = SGDClassifier(loss="log", penalty="l2")
        clf.fit(out[self.data.train_mask], self.data.y[self.data.train_mask])

        # compute test and val accuracies score

        val_acc = clf.score(out[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = clf.score(out[self.data.test_mask], self.data.y[self.data.test_mask])

        # compute test and val f1 score
        pred = clf.predict(out[self.data.test_mask])
        test_f1 = f1_score(self.data.y[self.data.test_mask], pred, average='micro')
        pred = clf.predict(out[self.data.val_mask])
        val_f1 = f1_score(self.data.y[self.data.val_mask], pred, average='micro')

        return val_f1, val_acc, test_f1, test_acc

    def run(self):
        for epoch in range(1, self.num_epochs + 1):
            loss = self.train(epoch)
            val_f1, val_acc, test_f1, test_acc = self.test()
            # print epoch and results
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')

        # Save results
        self.train_losses.append(loss)
        self.val_accuracies.append(val_acc)
        self.test_accuracies.append(test_acc)
        self.val_micro_f1s.append(val_f1)
        self.test_micro_f1s.append(test_f1)

