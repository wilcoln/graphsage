from experiments.fig3.runners import graphsage_runners, raw_features_runner


def get(model, noise_prop):
    """Returns the runner for the given dataset, model and training mode."""
    if 'graphsage' in model:
        aggregator = model.split('_')[1]
        return graphsage_runners.get(aggregator, noise_prop)

    if model == 'raw_features':
        return raw_features_runner.get(noise_prop)
    else:
        raise NotImplementedError
