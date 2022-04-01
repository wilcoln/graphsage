from graphsage.settings import args

DATASET = 'reddit'
MODELS = [
    'graphsage_gcn',
    'graphsage_mean',
    'graphsage_max',
    'graphsage_lstm',
    'deepwalk',
]
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 256
NUM_LAYERS = 2
BATCH_SIZE = 512
K1 = 25
K2 = 10

if args.dataset is not None:
    DATASET = args.dataset
    DATASET = 'cora' if DATASET == 'citation' else DATASET  # Cora is our default citation dataset

