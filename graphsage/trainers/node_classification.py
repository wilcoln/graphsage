import torch
import torch.nn.functional as F

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from torch_cluster import random_walk
from torch_geometric.data import Data
from torch_sparse import SparseTensor
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

        self.val_loader = loader(data, input_nodes=data.val_mask, num_neighbors=[self.k1, self.k2], shuffle=False,
                                  **kwargs)

        self.test_loader = loader(data, input_nodes=data.test_mask, num_neighbors=[self.k1, self.k2], shuffle=False,
                                  **kwargs)

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
            loss = self.loss_fn(y_hat, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval(self, loader):
        self.model.eval()
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


def get_pos_neg_batches(batch, data):
    device = batch.x.device

    batch_edge_index = batch.edge_index
    batch_num_nodes = int(batch_edge_index.max()) + 1
    batch_edge_index = SparseTensor(
        row=batch_edge_index[0],
        col=batch_edge_index[1],
        sparse_sizes=(batch_num_nodes, batch_num_nodes)
    ).t()

    row, col, _ = batch_edge_index.coo()

    # For each node in `batch`, we sample a direct neighbor (as positive
    # example) and a random node (as negative example):
    pos_batch_n_id = random_walk(row, col, batch.n_id, walk_length=1, coalesced=False)[:, 1].cpu()

    neg_batch_n_id = torch.randint(0, data.num_nodes, (batch.n_id.numel(),), dtype=torch.long)

    pos_batch = Data(
        x=torch.index_select(data.x.cpu(), 0, pos_batch_n_id).to(device),
        n_id=pos_batch_n_id.to(device),
        edge_index=batch.edge_index.to(device),
    )
    neg_batch = Data(
        x=torch.index_select(data.x.cpu(), 0, neg_batch_n_id).to(device),
        n_id=neg_batch_n_id.to(device),
        edge_index=batch.edge_index.to(device),
    )

    return pos_batch, neg_batch


class UnsupervisedTrainerForNodeClassification(BaseTrainer):
    def __init__(self,
                 data,
                 loader,
                 *args, **kwargs):
        super(UnsupervisedTrainerForNodeClassification, self).__init__(*args, **kwargs)

        self.data = data

        kwargs = {'batch_size': settings.BATCH_SIZE, 'num_workers': settings.NUM_WORKERS}

        self.train_loader = loader(data, input_nodes=data.train_mask,
                                   num_neighbors=[self.k1, self.k2], shuffle=False, **kwargs)

        self.subgraph_loader = loader(self.data, input_nodes=None, num_neighbors=[self.k1, self.k2], shuffle=False,
                                      **kwargs)

    # def train(self, epoch):
    #     self.model.train()
    #     print(f'Epoch: {epoch:02d}', end='\r')
    #
    #     total_loss = 0
    #     for batch_size, n_id, adj_list in tqdm(self.train_loader):
    #         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
    #         adj_list = [adj.to(self.device) for adj in adj_list]
    #         self.optimizer.zero_grad()
    #
    #         out = self.model(self.data.x[n_id].to(self.device), adj_list)
    #         out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
    #
    #         pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    #         neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    #         loss = -pos_loss - neg_loss
    #         loss.backward()
    #         self.optimizer.step()
    #         total_loss += float(loss) * out.size(0)
    #
    #     return total_loss / self.data.num_nodes

    def train(self, epoch):
        self.model.train()

        pbar = tqdm(total=int(len(self.train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            pos_batch, neg_batch = get_pos_neg_batches(batch, self.data)
            self.optimizer.zero_grad()

            out = self.model(batch.x, batch.edge_index)[:batch.batch_size]
            pos_out = self.model(pos_batch.x, pos_batch.edge_index)[:batch.batch_size]
            neg_out = self.model(neg_batch.x, neg_batch.edge_index)[:batch.batch_size]

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self):
        self.model.eval()
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
