import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from graphsage import settings
from graphsage.trainers.base_trainers import SupervisedTorchModuleBaseTrainer


class GraphLevelTriplesTorchModuleTrainer(SupervisedTorchModuleBaseTrainer):
    def __init__(self, train_loader, val_loader, test_loader, *args, **kwargs):
        super(GraphLevelTriplesTorchModuleTrainer, self).__init__(*args, **kwargs)

        assert train_loader.batch_size == val_loader.batch_size == test_loader.batch_size == 1, \
            "batch size must be 1 for graph level triples models"

        # Create loader objects
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train(self, epoch) -> float:
        # train for one epoch
        pbar = tqdm(total=len(self.train_loader.dataset))
        pbar.set_description(f'Epoch {epoch:02d}')

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in tqdm(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            data, target = data.squeeze(), target.squeeze()
            self.optimizer.zero_grad()
            hidden, out = self.model(data)
            u_hidden, v_hidden = torch.split(hidden, hidden.shape[1] // 2, dim=1)

            # compute loss
            out = out.squeeze()
            loss = self.loss_fn(out, target)

            # Unsup positive loss
            loss -= F.logsigmoid((u_hidden * v_hidden).sum(-1)).mean()
            # Unsup negative loss
            u_perm = torch.randperm(u_hidden.shape[0])
            v_perm = torch.randperm(v_hidden.shape[0])
            loss -= F.logsigmoid(-(u_hidden[u_perm] * v_hidden[v_perm]).sum(-1)).mean()

            loss.backward()
            self.optimizer.step()

            out = out.squeeze()
            pred = out.argmax(dim=-1)
            correct += pred.eq(target).sum().item()
            total_loss += loss.item()
            total += 1

            total_loss += loss.item()
            pbar.update(len(data))
        pbar.close()

        return total_loss / len(self.train_loader.dataset)

    def eval(self, loader):
        self.model.eval()
        y_true = []
        y_pred = []
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            data, target = data.squeeze(), target.squeeze()
            _, out = self.model(data)
            out = out.squeeze()
            pred = out.argmax(dim=-1)
            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

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
