import torch
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from tqdm import tqdm

from graphsage import settings
from .base_trainers import SupervisedBaseTrainer, BaseTrainer


class SupervisedTrainerForNodeClassification(SupervisedBaseTrainer):
    def __init__(self,
                 data,
                 loader,
                 *args, **kwargs):
        super(SupervisedTrainerForNodeClassification, self).__init__(*args, **kwargs)

        self.data = data
        kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS}
        self.train_loader = loader(data, input_nodes=data.train_mask,
                                   num_neighbors=[self.k1, self.k2], shuffle=False, **kwargs)

        self.subgraph_loader = loader(data, input_nodes=None, num_neighbors=[self.k1, self.k2], shuffle=False, **kwargs)

    def train(self, epoch):
        self.model.train()

        pbar = tqdm(total=int(len(self.train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            y = batch.y[:batch.batch_size].to(self.device)
            y_hat = self.model(batch.x.to(self.device), batch.edge_index.to(self.device))[:batch.batch_size]
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

        # compute accuracies for each class
        accs = [
            int((y_hat[mask] == y[mask]).sum()) / int(mask.sum())
            for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]
        ]

        # compute F1 scores for each class
        f1_scores = [
            f1_score(y[mask].cpu(), y_hat[mask].cpu(), average='micro')
            for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]
        ]
        return {
            'train_acc': accs[0],
            'val_acc': accs[1],
            'test_acc': accs[2],
            'train_f1': f1_scores[0],
            'val_f1': f1_scores[1],
            'test_f1': f1_scores[2]
        }


class UnsupervisedTrainerForNodeClassification(BaseTrainer):
    def __init__(self,
                 data,
                 sampler,
                 loader,
                 *args, **kwargs):
        super(UnsupervisedTrainerForNodeClassification, self).__init__(*args, **kwargs)

        self.data = data

        kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS}

        self.train_loader = sampler(self.data.edge_index, sizes=[25, 10], shuffle=True, num_nodes=data.num_nodes,
                                           **kwargs)

        self.subgraph_loader = loader(self.data, input_nodes=None, num_neighbors=[25, 10], shuffle=False, **kwargs)

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
        self.data.y = self.data.y.cpu()

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
