import torch

from tqdm import tqdm


class Trainer:
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

