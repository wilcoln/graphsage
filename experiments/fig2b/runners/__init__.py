from .runners import SampleSizeRunner


def get(sample_size):
    """Returns the runner for the given dataset, model and training mode."""
    try:
        return SampleSizeRunner(sample_size)
    except KeyError:
        raise NotImplementedError
