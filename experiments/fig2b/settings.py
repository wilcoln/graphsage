from graphsage.settings import args


NUM_LAYERS = 2
DATASET = 'citation'
MODEL = 'graphsage_mean'
SAMPLE_SIZES = [5, 10, 20, 30, 70]
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_WORKERS = 1


if args.dataset is not None:
    DATASET = args.dataset
    DATASET = 'cora' if DATASET == 'citation' else DATASET  # Cora is our default citation dataset
    assert DATASET in {'cora', 'citeseer', 'pubmed'}, 'Dataset must be one of cora, citeseer, pubmed'

if args.batch_size is not None:
    BATCH_SIZE = args.batch_size


