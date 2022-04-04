import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from graphsage.trainers.base_trainers import SupervisedTorchModuleBaseTrainer, dataloader_kwargs
from triplets.utils import mask2index


class TripletMLPTrainer(SupervisedTorchModuleBaseTrainer):
    def __init__(self, data, *args, **kwargs):
        super(TripletMLPTrainer, self).__init__(*args, **kwargs)
        # Create loader objects

        self.train_loader = DataLoader(Subset(data, mask2index(data.train_mask)), shuffle=True, **dataloader_kwargs)
        self.val_loader = DataLoader(Subset(data, mask2index(data.val_mask)), shuffle=True, **dataloader_kwargs)
        self.test_loader = DataLoader(Subset(data, mask2index(data.test_mask)), shuffle=True, **dataloader_kwargs)

    def train(self, epoch) -> float:
        # train for one epoch
        pbar = tqdm(total=len(self.train_loader.dataset))
        pbar.set_description(f'Epoch {epoch:02d}')

        self.model.train()
        total_loss = 0
        for bdata, target in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

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
            output = self.model(data)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        return f1_score(y_true, y_pred, average='micro')

    def test(self):
        val_f1 = self.eval(self.val_loader)
        test_f1 = self.eval(self.test_loader)

        return {
            'val_f1': val_f1,
            'test_f1': test_f1
        }
