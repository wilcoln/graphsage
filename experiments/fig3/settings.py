from types import SimpleNamespace

import numpy as np

from graphsage.settings import args

DATASET = 'citation'

MODELS = [
    'graphsage_gcn',
    'graphsage_max_pool',
    'raw_features',
]

extensions = SimpleNamespace(
    MODELS=[
        'graphsage_mean',
        'graphsage_lstm',
        'graphsage_max',
        'graphsage_sum',
        'graphsage_bilstm',
        'graphsage_mean_pool',
        'triples_logreg',
        'triples_mlp',
    ]
)

FEATURE_NOISE_PROP = np.linspace(0, 1, 10)

LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 256
NUM_LAYERS = 2
BATCH_SIZE = 512
K1 = 25
K2 = 10

if not args.no_extensions:
    MODELS += extensions.MODELS

if args.dataset is not None:
    DATASET = args.dataset
    DATASET = 'cora' if DATASET == 'citation' else DATASET  # Cora is our default citation dataset

if args.batch_size is not None:
    BATCH_SIZE = args.batch_size

if args.ignore_aggregators:
    MODELS = [m for m in MODELS if not any(m.endswith(a) for a in args.ignore_aggregators)]
