from graphsage.settings import args


DATASETS = [
    'cora',
    'pubmed',
    'citeseer',
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
    'graphsage_max',
    'graphsage_bilstm',
    'graphsage_sum',
    'graphsage_mean_pool',
    'graphsage_max_pool',
]
TRAINING_MODES = [
    'unsupervised',
    'supervised',
]

GRAPH_OBLIVIOUS_MODELS = [
    'random',
    'raw_features',
    'deep_walk',
    'deep_walk_plus_features',
]

HIDDEN_CHANNELS = 256
NUM_LAYERS = 2
SUPERVISED_LEARNING_RATE = 1e-3
UNSUPERVISED_LEARNING_RATE = 1e-5

if args.ignore_datasets:
    try:
        DATASETS = [d for d in DATASETS if d not in args.ignore_datasets]
    except ValueError:
        pass
