import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

from .base_trainers import SupervisedGraphSageBaseTrainer, GraphSageBaseTrainer, dataloader_kwargs
from .. import settings


class SupervisedGraphSageTrainerForNodeLevelTask(SupervisedGraphSageBaseTrainer):
    def __init__(self,
                 data,
                 loader,
                 *args, **kwargs):
        super(SupervisedGraphSageTrainerForNodeLevelTask, self).__init__(*args, **kwargs)

        self.data = data
        self.train_loader = loader(data, input_nodes=data.train_mask,
                                   num_neighbors=[self.k1, self.k2], shuffle=False, **dataloader_kwargs)

        self.val_loader = loader(data, input_nodes=data.val_mask,
                                 num_neighbors=[self.k1, self.k2], shuffle=False, **dataloader_kwargs)

        self.test_loader = loader(data, input_nodes=data.test_mask,
                                  num_neighbors=[self.k1, self.k2], shuffle=False, **dataloader_kwargs)

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
        train_acc, train_f1 = self.eval(self.train_loader) if not settings.NO_EVAL_TRAIN else (None, None)
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

        self.test_loader = loader(data, input_nodes=None, num_neighbors=[self.k1, self.k2], shuffle=False,
                                      **dataloader_kwargs)
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
        # Train downstream classifier on train split representations
        clf = SGDClassifier(loss="log", penalty="l2")
        train_out = self.model.inference(self.train_loader).cpu()
        self.data.y = self.data.y.cpu()
        clf.fit(train_out, self.data.y[self.data.train_mask])

        # compute f1 and accuracy on train split
        if not settings.NO_EVAL_TRAIN:
            train_acc = clf.score(train_out, self.data.y[self.data.train_mask])
            train_f1 = f1_score(self.data.y[self.data.train_mask],  clf.predict(train_out), average='micro')
        else:
            train_acc, train_f1 = None, None

        # compute f1 and accuracy on val split
        val_out = self.model.inference(self.val_loader).cpu()
        val_f1 = f1_score(self.data.y[self.data.val_mask], clf.predict(val_out), average='micro')
        val_acc = clf.score(val_out, self.data.y[self.data.val_mask])

        # compute f1 and accuracy on test split
        test_out = self.model.inference(self.test_loader).cpu()
        test_f1 = f1_score(self.data.y[self.data.test_mask], clf.predict(test_out), average='micro')
        test_acc = clf.score(test_out, self.data.y[self.data.test_mask])

        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'val_f1': val_f1,
        }
