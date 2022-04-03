import json
import os
import os.path as osp
import time
from datetime import datetime as dt

import pandas as pd
from matplotlib import pyplot as plt

from graphsage import settings


def capitalize(underscore_string):
    return ' '.join(w.capitalize() for w in underscore_string.split('_'))


class BaseTrainer:
    def __init__(self, dataset_name: str = None):
        self.dataset_name = dataset_name

    def train(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def test(self) -> dict:
        raise NotImplementedError
    
    def run(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class NNBaseTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 num_epochs,
                 device,
                 k1: int = 25,
                 k2: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.k1 = k1
        self.k2 = k2

        self.results = []

    def save_results(self):
        # Create dictionary with all the parameters
        folder_name_dict = {
            'dataset': self.dataset_name,
            'model': self.model.__class__.__name__,
            'num_epochs': self.num_epochs,
        }

        params_dict = {
            'dataset': self.dataset_name,
            'model': str(self.model),
            'optimizer': str(self.optimizer),
            'num_epochs': self.num_epochs,
            'device': self.device.type,
        }

        # Create folder name
        date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        folder_name = '_'.join([date] + [f'{k}={v}' for k, v in folder_name_dict.items()])
        folder_path = osp.join(settings.RESULTS_DIR, 'trainers', folder_name)

        # Create folder
        os.makedirs(folder_path)

        # Save figures
        df = pd.DataFrame(self.results)  # create dataframe
        df.index += 1  # shift index by 1, because epochs start at 1
        for i, col in enumerate(df.columns):
            df[col].plot(fig=plt.figure(i))
            col_name = capitalize(col)
            plt.title(col_name)
            plt.xlabel('Epoch')
            # plt.ylabel(col_name)

            plt.savefig(osp.join(folder_path, f'{col}.png'))
            plt.close()

        with open(osp.join(folder_path, 'params.json'), 'w') as f:
            json.dump(params_dict, f)

        with open(osp.join(folder_path, 'results.json'), 'w') as f:
            json.dump(self.results, f)

        # Print path to results
        print(f'Results saved to {folder_path}')

    def run(self, return_best_epoch_only=True):

        for epoch in range(1, self.num_epochs + 1):
            # Train & test
            start = time.time()
            loss = self.train(epoch)
            train_time = time.time() - start

            start = time.time()
            test_results = self.test()
            test_time = time.time() - start

            # Save epoch results
            epoch_results = {'loss': loss, **test_results, 'train_time': train_time, 'test_time': test_time}

            # print epoch and results
            current_results_str = ', '.join([f'{capitalize(k)}: {v:.4f}' for k, v in epoch_results.items()])
            print(f'Epoch: {epoch:02d}, {current_results_str}')

            # Save epoch results to list
            self.results.append(epoch_results)

        # Save results
        self.save_results()

        # Get best epoch result
        best = max(self.results, key=lambda x: max(x.get('val_acc', 0), x.get('val_f1', 0)))

        # Return best epoch results i.e. the one w/ the highest validation accuracy or f1 score
        if return_best_epoch_only:
            return best
        else:
            # Return best & all epoch results
            return best, self.results


class SupervisedBaseTrainer(NNBaseTrainer):
    def __init__(self,
                 loss_fn,
                 *args, **kwargs):
        super(SupervisedBaseTrainer, self).__init__(*args, **kwargs)
        self.loss_fn = loss_fn
