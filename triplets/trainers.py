import torch
import torch.nn.functional as F
from icecream import ic
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from graphsage.trainers.base_trainers import SupervisedTorchModuleBaseTrainer, dataloader_kwargs
from triplets.utils import mask2index


class TripletMLPTrainer(SupervisedTorchModuleBaseTrainer):
    def __init__(self, data, *args, **kwargs):
        super(TripletMLPTrainer, self).__init__(*args, **kwargs)
        self.data = data

        # Create loader objects
        self.triplet_train_loader = DataLoader(Subset(data, mask2index(data.triplet_train_mask)), shuffle=False,
                                               **dataloader_kwargs)
        self.train_loader = DataLoader(Subset(data, mask2index(data.train_mask)), shuffle=True, **dataloader_kwargs)
        self.val_loader = DataLoader(Subset(data, mask2index(data.val_mask)), shuffle=True, **dataloader_kwargs)
        self.test_loader = DataLoader(Subset(data, mask2index(data.test_mask)), shuffle=True, **dataloader_kwargs)

    def unsup_loss(self, out, data, batch_index, use_triplet_loss=True):
        # batch_offset = batch_index * self.triplet_train_loader.batch_size
        # triplet_indices = list(range(batch_offset, batch_offset + data.shape[0]))
        heads, tails = torch.split(data, data.shape[1]//2, dim=1)
        pos_data = torch.cat([tails, heads], dim=1)

        neg_data_indices = torch.randint(0, self.data.x.shape[0], (data.shape[0],))
        neg_data = self.data.x.index_select(0, neg_data_indices).to(self.device)

        pos_out = self.model(pos_data)
        neg_out = self.model(neg_data)

        if use_triplet_loss:
            return F.triplet_margin_loss(out, pos_out, neg_out)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        return -pos_loss - neg_loss

    def train(self, epoch) -> float:
        # train for one epoch
        pbar = tqdm(total=len(self.triplet_train_loader.dataset))
        pbar.set_description(f'Epoch {epoch:02d}')

        self.model.train()
        total_loss = 0
        for batch_index, (data, target) in enumerate(tqdm(self.triplet_train_loader)):
            # get range of indices for batch
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target) + self.unsup_loss(output, data, batch_index)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.update(len(data))
        pbar.close()

        return total_loss / len(self.triplet_train_loader.dataset)

    def eval(self, loader):
        self.model.eval()
        y_true = []
        y_pred = []
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        return f1_score(y_true, y_pred, average='micro')

    def test(self):
        train_f1 = self.eval(self.train_loader)
        val_f1 = self.eval(self.val_loader)
        test_f1 = self.eval(self.test_loader)

        return {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        }
