import os.path as osp

import torch
import torch_geometric.transforms as T

from graphsage import settings
from graphsage.datasets import Planetoid
from graphsage.models.supervised import GraphSAGE
from graphsage.samplers import UniformLoader
from graphsage.trainers import SupervisedTrainerForNodeClassification
import experiments.fig2b.settings as fig2b_settings

device = settings.DEVICE
dataset_name = 'Cora'
path = osp.join(settings.DATA_DIR, dataset_name)


class SampleSizeRunner:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def run(self):
        test_times = []
        test_f1s = []
        for _ in range(fig2b_settings.NUM_RUNS):
            dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

            # Already send node features/labels to GPU for faster access during sampling:
            data = dataset[0].to(device, 'x', 'y')

            model = GraphSAGE(
                in_channels=dataset.num_features,
                hidden_channels=fig2b_settings.BATCH_SIZE,
                out_channels=dataset.num_classes,
                num_layers=fig2b_settings.NUM_LAYERS,
                aggregator='mean',
            ).to(device)

            best_result, all_results = SupervisedTrainerForNodeClassification(
                dataset_name=dataset_name,
                model=model,
                data=data,
                k1=self.sample_size,
                k2=self.sample_size,
                loader=UniformLoader,
                num_epochs=settings.NUM_EPOCHS,
                loss_fn=torch.nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=fig2b_settings.LEARNING_RATE),
                device=device,
            ).run(return_best_epoch_only=False)

            test_times.append(sum(result['test_time'] for result in all_results) / len(all_results))
            test_f1s.append(best_result['test_f1'])

        # return dictionary with mean test time and best f1 score
        return {
            'test_time': sum(test_times) / len(test_times),
            'test_f1': sum(test_f1s) / len(test_f1s),
        }

