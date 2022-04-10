from types import SimpleNamespace

from graphsage.settings import args

DATASETS = [
    'cora',
    'reddit',
    'ppi',
]

MODELS = [
    'random',
    'raw_features',
    'deep_walk',
    'deep_walk_plus_features',
    'graphsage_gcn',
    'graphsage_mean',
    'graphsage_lstm',
    'graphsage_max_pool',
]

extensions = SimpleNamespace(
    DATASETS=[
        'pubmed',
        'citeseer',
        'flickr',
        'mutag',
    ],
    MODELS=[
        'graphsage_max',
        'graphsage_sum',
        'graphsage_bilstm',
        'graphsage_mean_pool',
        'triples_logreg',
        'triples_mlp',
        'graphsaint_gcn',
        'graphsaint_mean',
        'graphsaint_max_pool',
        'shadow_gcn',
        'shadow_mean',
        'shadow_max_pool',
    ]
)

TRAINING_MODES = [
    'unsupervised',
    'supervised',
]

TRAINING_MODE_OBLIVIOUS_MODELS = [
    'random',
    'raw_features',
    'deep_walk',
    'deep_walk_plus_features',
]

HIDDEN_CHANNELS = 256
NUM_LAYERS = 2
SUPERVISED_LEARNING_RATE = 1e-3
UNSUPERVISED_LEARNING_RATE = 1e-5

if not args.no_extensions:
    DATASETS += extensions.DATASETS
    MODELS += extensions.MODELS

if args.ignore_datasets:
    try:
        DATASETS = [d for d in DATASETS if d not in args.ignore_datasets]
    except ValueError:
        pass

if args.ignore_aggregators:
    MODELS = [m for m in MODELS if not any(m.endswith(a) for a in args.ignore_aggregators)]
