from .citation_graphsage_supervised import SampleSizeRunner
# Dict of all implemented runners
from ..settings import SAMPLE_SIZES

runners = {sample_size: SampleSizeRunner(sample_size) for sample_size in SAMPLE_SIZES}


def get(sample_size):
    """Returns the runner for the given dataset, model and training mode."""
    try:
        return runners[sample_size]
    except KeyError:
        raise NotImplementedError
