from experiments.fig2a.runners import graphsage_runners


def get(model):
    """Returns the runner for the given dataset, model and training mode."""
    try:
        aggregator = model.split('_')[1]
        return graphsage_runners.get(aggregator)
    except KeyError:
        raise NotImplementedError
