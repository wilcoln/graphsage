import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

from .base_trainers import SupervisedGraphSageBaseTrainer, GraphSageBaseTrainer, dataloader_kwargs


class SupervisedGraphSageTrainerForNodeLevelTask(SupervisedGraphSageBaseTrainer):
    def __init__(self,
                 data,
                 loader,
                 *args, **kwargs):
        super(SupervisedGraphSageTrainerForNodeLevelTask, self).__init__(*args, **kwargs)

        self.data = data
        self.train_loader = loader(data, input_nodes=data.train_mask,
                                   num_neighbors=[self.k1, self.k2], shuffle=False, **dataloader_kwargs)

        self.val_loader = loader(data, input_nodes=data.val_mask, num_neighbors=[self.k1, self.k2], shuffle=False,
                                  **dataloader_kwargs)

        self.test_loader = loader(data, input_nodes=data.test_mask, num_neighbors=[self.k1, self.k2], shuffle=False,
                                  **dataloader_kwargs)

    def train(self, epoch):
        self.model.train()

        pbar = tqdm(total=int(len(self.train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = 0
        for batch in self.train_loader:
            # we slice with :batch.batch_size all the time to info about the actual nodes of the batch,
            # the rest being about the sampled neighbors
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            y = batch.y[:batch.batch_size].to(self.device)
            y_hat = self.model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = self.loss_fn(y_hat, y) + self.unsup_loss_fn(y_hat, batch, self.data)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval(self, loader):
        y_hat = self.model.inference(loader).argmax(dim=-1)
        y = torch.cat([batch.y[:batch.batch_size] for batch in loader]).to(self.device)
        acc = int((y_hat == y).sum()) / y.shape[0]
        micro_f1 = f1_score(y.cpu(), y_hat.cpu(), average='micro')
        return acc, micro_f1

    @torch.no_grad()
    def test(self):
        train_acc, train_f1 = self.eval(self.train_loader)
        val_acc, val_f1 = self.eval(self.val_loader)
        test_acc, test_f1 = self.eval(self.test_loader)

        return {
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_f1': test_f1
        }


class UnsupervisedGraphSageTrainerForNodeLevelTask(GraphSageBaseTrainer):
    def __init__(self,
                 data,
                 loader,
                 *args, **kwargs):
        super(UnsupervisedGraphSageTrainerForNodeLevelTask, self).__init__(*args, **kwargs)

        self.data = data

        self.train_loader = loader(data, input_nodes=data.train_mask,
                                   num_neighbors=[self.k1, self.k2], shuffle=False, **dataloader_kwargs)

        self.subgraph_loader = loader(self.data, input_nodes=None, num_neighbors=[self.k1, self.k2], shuffle=False,
                                      **dataloader_kwargs)

    def train(self, epoch):
        self.model.train()

        pbar = tqdm(total=int(len(self.train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = self.unsup_loss_fn(out, batch, self.data)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * out.size(0)
            pbar.update(batch.batch_size)
        pbar.close()

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self):
        out = self.model.inference(self.subgraph_loader).cpu()
        self.data.y = self.data.y.cpu()

        # Train downstream classifier on train split representations
        clf = SGDClassifier(loss="log", penalty="l2")
        clf.fit(out[self.data.train_mask], self.data.y[self.data.train_mask])

        # compute accuracies for each split
        train_acc = clf.score(out[self.data.train_mask], self.data.y[self.data.train_mask])
        val_acc = clf.score(out[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = clf.score(out[self.data.test_mask], self.data.y[self.data.test_mask])

        # compute f1 scores for each split
        pred = clf.predict(out[self.data.train_mask])
        train_f1 = f1_score(self.data.y[self.data.train_mask], pred, average='micro')
        pred = clf.predict(out[self.data.test_mask])
        test_f1 = f1_score(self.data.y[self.data.test_mask], pred, average='micro')
        pred = clf.predict(out[self.data.val_mask])
        val_f1 = f1_score(self.data.y[self.data.val_mask], pred, average='micro')

        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'val_f1': val_f1,
        }
