
# Dict of all implemented runners
runners = {
    # dataset, model, training_mode: runner
    ('citation', 'graphsage_mean', 'supervised'): None,
    ('citation', 'graphsage_mean', 'unsupervised'): None,
}


def get(dataset, model, training_mode):
    """Returns the runner for the given dataset, model and training mode."""
    try:
        return runners[dataset, model, training_mode]
    except KeyError:
        raise NotImplementedError

