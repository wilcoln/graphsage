from experiments.fig2a.runners import graphsage_runners

# Dict of all implemented runners
runners = {
    # model: runner
    'graphsage_mean': graphsage_runners.get('mean'),
    'graphsage_max': graphsage_runners.get('max'),
    'graphsage_gcn': graphsage_runners.get('gcn'),
    'graphsage_lstm': graphsage_runners.get('lstm'),
}


def get(model):
    """Returns the runner for the given dataset, model and training mode."""
    try:
        return runners[model]
    except KeyError:
        raise NotImplementedError
