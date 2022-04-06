import experiments.table1.runners.citation.supervised.graphsage_runners as sup_citation_graphsage_runners
import experiments.table1.runners.citation.supervised.random_runners as sup_citation_random_runners
import experiments.table1.runners.citation.supervised.raw_features_runners as sup_citation_raw_features_runners
import experiments.table1.runners.citation.unsupervised.graphsage_runners as unsup_citation_graphsage_runners
import experiments.table1.runners.ppi.supervised.graphsage_runners as sup_ppi_graphsage_runners
import experiments.table1.runners.ppi.supervised.random_runners as sup_ppi_random_runners
import experiments.table1.runners.ppi.supervised.raw_features_runners as sup_ppi_raw_features_runners
import experiments.table1.runners.ppi.unsupervised.graphsage_runners as unsup_ppi_graphsage_runners
import experiments.table1.runners.reddit.supervised.graphsage_runners as sup_reddit_graphsage_runners
import experiments.table1.runners.reddit.supervised.random_runners as sup_reddit_random_runners
import experiments.table1.runners.reddit.supervised.raw_features_runners as sup_reddit_raw_features_runners
import experiments.table1.runners.reddit.unsupervised.graphsage_runners as unsup_reddit_graphsage_runners
from experiments.table1.runners.others import triples_models_runners
import experiments.table1.settings as table1_settings

# Dict of all implemented runners
runners = {
    'cora': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.cora,
            'raw_features': sup_citation_raw_features_runners.cora,
            'random': sup_citation_random_runners.cora,
        },
        'unsupervised': {
            'graphsage': unsup_citation_graphsage_runners.cora,
        },
    },
    'citeseer': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.citeseer,
            'raw_features': sup_citation_raw_features_runners.citeseer,
            'random': sup_citation_random_runners.cora,
        },
        'unsupervised': {
            'graphsage': unsup_citation_graphsage_runners.citeseer,
        },
    },
    'pubmed': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.pubmed,
            'raw_features': sup_citation_raw_features_runners.pubmed,
            'random': sup_citation_random_runners.pubmed,
        },
        'unsupervised': {
            'graphsage': unsup_citation_graphsage_runners.pubmed,
        },
    },
    'ppi': {
        'supervised': {
            'graphsage': sup_ppi_graphsage_runners,
            'raw_features': sup_ppi_raw_features_runners,
            'random': sup_ppi_random_runners,
        },
        'unsupervised': {
            'graphsage': unsup_ppi_graphsage_runners,
        },
    },
    'reddit': {
        'supervised': {
            'graphsage': sup_reddit_graphsage_runners,
            'raw_features': sup_reddit_raw_features_runners,
            'random': sup_reddit_random_runners,
        },
        'unsupervised': {
            'graphsage': unsup_reddit_graphsage_runners,
        },
    },
}


def get(dataset, training_mode, model):
    """Returns the runner for the given dataset, model and training mode."""
    assert dataset in table1_settings.DATASETS, 'Unknown dataset: {}'.format(dataset)
    assert training_mode in table1_settings.TRAINING_MODES, 'Unknown training mode: {}'.format(training_mode)
    assert model in table1_settings.MODELS, 'Unknown model: {}'.format(model)
    try:
        if 'graphsage' in model:
            aggregator = model.split('_')[1]
            return runners[dataset][training_mode]['graphsage'].get(aggregator)
        if 'triples' in model:
            return triples_models_runners.get(dataset, training_mode, model)
        else:
            return runners[dataset][training_mode][model].get()
    except KeyError:
        raise NotImplementedError
