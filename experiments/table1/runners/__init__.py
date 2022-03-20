from . import citation_graphsage_unsupervised, citation_graphsage_supervised

# Dict of all implemented runners
runners = {
    # dataset, model, training_mode: runner
    ('citation', 'graphsage_mean', 'supervised'): citation_graphsage_supervised,
    ('citation', 'graphsage_mean', 'unsupervised'): citation_graphsage_unsupervised,
}


def get(dataset, model, training_mode):
    """Returns the runner for the given dataset, model and training mode."""
    try:
        return runners[dataset, model, training_mode]
    except KeyError:
        raise NotImplementedError

