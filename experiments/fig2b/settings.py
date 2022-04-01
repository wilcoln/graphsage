from graphsage.settings import args


NUM_LAYERS = 2
DATASET = 'citation'
MODEL = 'graphsage_mean'
SAMPLE_SIZES = [5, 10, 20, 40, 70, 100]
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_RUNS = 5


if args.dataset is not None:
    DATASET = args.dataset
    DATASET = 'cora' if DATASET == 'citation' else DATASET  # Cora is our default citation dataset
    assert DATASET in {'cora', 'citeseer', 'pubmed'}, 'Dataset must be one of cora, citeseer, pubmed'

if args.num_runs is not None:
    NUM_RUNS = args.num_runs

if args.batch_size is not None:
    BATCH_SIZE = args.batch_size


