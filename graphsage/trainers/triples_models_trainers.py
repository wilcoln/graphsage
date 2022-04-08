import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from graphsage import settings
from graphsage.datasets.triples import mask2index
from graphsage.trainers.base_trainers import SupervisedTorchModuleBaseTrainer, dataloader_kwargs


class TriplesTorchModuleTrainer(SupervisedTorchModuleBaseTrainer):
    def __init__(self, data, *args, **kwargs):
        super(TriplesTorchModuleTrainer, self).__init__(*args, **kwargs)
        self.data = data

        # Create loader objects
        self.triple_train_loader = DataLoader(Subset(data, mask2index(data.triple_train_mask)), shuffle=True,
                                               **dataloader_kwargs)
        self.train_loader = DataLoader(Subset(data, mask2index(data.train_mask)), shuffle=True, **dataloader_kwargs)
        self.val_loader = DataLoader(Subset(data, mask2index(data.val_mask)), shuffle=True, **dataloader_kwargs)
        self.test_loader = DataLoader(Subset(data, mask2index(data.test_mask)), shuffle=True, **dataloader_kwargs)

    def unsup_loss(self, out, data, batch_index, use_triple_loss=True):
        loss = 0

        # TODO: Get positive samples
        # batch_offset = batch_index * self.triple_train_loader.batch_size
        # triple_indices = list(range(batch_offset, batch_offset + data.shape[0]))
        # pos_data = ...
        # Compute positive loss
        # pos_out = self.model(pos_data)
        # pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        # loss -= pos_loss

        # Get negative samples
        neg_data_indices = torch.randint(0, self.data.x.shape[0], (data.shape[0],))
        neg_data = self.data.x.index_select(0, neg_data_indices).to(self.device)
        # Compute negative loss
        neg_out = self.model(neg_data)
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss -= neg_loss

        # if use_triple_loss:
        #     return F.triple_margin_loss(out, pos_out, neg_out)

        return loss

    def train(self, epoch) -> float:
        # train for one epoch
        pbar = tqdm(total=len(self.triple_train_loader.dataset))
        pbar.set_description(f'Epoch {epoch:02d}')

        self.model.train()
        total_loss = 0
        for batch_index, (data, target) in enumerate(tqdm(self.triple_train_loader)):
            # get range of indices for batch
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            u_output, v_output = torch.split(output, output.shape[1] // 2, dim=1)
            u_target, v_target = torch.split(target, target.shape[1] // 2, dim=1)
            # squeeze targets
            u_target = u_target.squeeze(1)
            v_target = v_target.squeeze(1)

            # compute loss
            loss = self.loss_fn(u_output, u_target) + self.loss_fn(v_output, v_target)
            # Unsup positive loss
            loss -= F.logsigmoid((u_output * v_output).sum(-1)).mean()
            # Unsup negative loss
            u_perm = torch.randperm(u_output.shape[0])
            v_perm = torch.randperm(v_output.shape[0])
            loss -= F.logsigmoid(-(u_output[u_perm] * v_output[v_perm]).sum(-1)).mean()

            # Back propagate
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.update(len(data))
        pbar.close()

        return total_loss / len(self.triple_train_loader.dataset)

    def eval(self, loader):
        self.model.eval()
        y_true = []
        y_pred = []
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            u_output, v_output = torch.split(output, output.shape[1] // 2, dim=1)
            u_target, v_target = torch.split(target, target.shape[1] // 2, dim=1)
            # squeeze targets
            u_target = u_target.squeeze(1)
            v_target = v_target.squeeze(1)

            y_true.extend(u_target.cpu().numpy())
            y_pred.extend(torch.argmax(u_output, dim=1).cpu().numpy())

            y_true.extend(v_target.cpu().numpy())
            y_pred.extend(torch.argmax(v_output, dim=1).cpu().numpy())

        return f1_score(y_true, y_pred, average='micro')

    def test(self):
        train_f1 = self.eval(self.train_loader) if not settings.NO_EVAL_TRAIN else None
        val_f1 = self.eval(self.val_loader)
        test_f1 = self.eval(self.test_loader)

        return {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        }
