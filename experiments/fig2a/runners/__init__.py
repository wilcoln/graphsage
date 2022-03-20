from .reddit_graphsage_supervised import GraphSAGEMeanRunner, GraphSAGEPoolRunner

# Dict of all implemented runners
runners = {
    # model: runner
    'graphsage_mean': GraphSAGEMeanRunner,
    'graphsage_pool': GraphSAGEPoolRunner,
}


def get(model):
    """Returns the runner for the given dataset, model and training mode."""
    try:
        return runners[model]
    except KeyError:
        raise NotImplementedError
