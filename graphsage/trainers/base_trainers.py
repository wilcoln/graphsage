class BaseTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 num_epochs,
                 device):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device

        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.train_micro_f1s = []
        self.val_micro_f1s = []
        self.test_micro_f1s = []


class SupervisedBaseTrainer(BaseTrainer):
    def __init__(self,
                 loss_fn,
                 *args, **kwargs):
        super(SupervisedBaseTrainer, self).__init__(*args, **kwargs)
        self.loss_fn = loss_fn
