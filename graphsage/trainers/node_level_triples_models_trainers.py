import torch
from sklearn.linear_model import SGDClassifier
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from graphsage import settings
from graphsage.datasets.triples import mask2index
from graphsage.trainers.base_trainers import TorchModuleBaseTrainer, SupervisedTorchModuleBaseTrainer, dataloader_kwargs


class SupervisedTriplesTorchModuleTrainer(SupervisedTorchModuleBaseTrainer):
    def __init__(self, data, *args, **kwargs):
        super(SupervisedTriplesTorchModuleTrainer, self).__init__(*args, **kwargs)
        self.data = data

        # Create loader objects
        self.triple_train_loader = DataLoader(Subset(data, mask2index(data.triple_train_mask)), shuffle=True,
                                               **dataloader_kwargs)
        self.train_loader = DataLoader(Subset(data, mask2index(data.train_mask)), shuffle=True, **dataloader_kwargs)
        self.val_loader = DataLoader(Subset(data, mask2index(data.val_mask)), shuffle=True, **dataloader_kwargs)
        self.test_loader = DataLoader(Subset(data, mask2index(data.test_mask)), shuffle=True, **dataloader_kwargs)

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


class UnsupervisedTriplesTorchModuleTrainer(TorchModuleBaseTrainer):
    def __init__(self, data, *args, **kwargs):
        super(UnsupervisedTriplesTorchModuleTrainer, self).__init__(*args, **kwargs)
        self.data = data

        # Create loader objects
        self.triple_train_loader = DataLoader(Subset(data, mask2index(data.triple_train_mask)), shuffle=True,
                                              **dataloader_kwargs)
        self.train_loader = DataLoader(Subset(data, mask2index(data.train_mask)), shuffle=True, **dataloader_kwargs)
        self.val_loader = DataLoader(Subset(data, mask2index(data.val_mask)), shuffle=True, **dataloader_kwargs)
        self.test_loader = DataLoader(Subset(data, mask2index(data.test_mask)), shuffle=True, **dataloader_kwargs)

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

            # compute loss
            loss = 0
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

    def get_embeddings_and_labels(self, loader):
        self.model.eval()
        xs = []
        ys = []
        for data, target in loader:
            data = data.to(self.device)
            output = self.model(data)
            u_output, _ = torch.split(output, output.shape[1] // 2, dim=1)
            u_target, _ = torch.split(target, target.shape[1] // 2, dim=1)
            xs.append(u_output)
            ys.append(u_target.squeeze(1)) # (1, 1) -> (1, )

        return torch.cat(xs, dim=0).cpu(), torch.cat(ys, dim=0).cpu()

    @torch.no_grad()
    def test(self):
        # Train downstream classifier on train split representations
        clf = SGDClassifier(loss="log", penalty="l2")
        train_out, train_labels = self.get_embeddings_and_labels(self.train_loader)
        clf.fit(train_out, train_labels)

        # compute f1 and accuracy on train split
        if not settings.NO_EVAL_TRAIN:
            train_acc = clf.score(train_out, train_labels)
            train_f1 = f1_score(train_labels,  clf.predict(train_out), average='micro')
        else:
            train_acc, train_f1 = None, None

        # compute f1 and accuracy on val split
        val_out, val_labels = self.get_embeddings_and_labels(self.val_loader)
        val_f1 = f1_score(val_labels, clf.predict(val_out), average='micro')
        val_acc = clf.score(val_out, val_labels)

        # compute f1 and accuracy on test split
        test_out, test_labels = self.get_embeddings_and_labels(self.test_loader)
        test_f1 = f1_score(test_labels, clf.predict(test_out), average='micro')
        test_acc = clf.score(test_out, test_labels)

        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'val_f1': val_f1,
        }
