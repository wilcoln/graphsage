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
]
TRAINING_MODES = [
    'unsupervised',
    'supervised',
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
