import experiments.table1.runners.citation.supervised.graphsage_runners as sup_citation_graphsage_runners
import experiments.table1.runners.citation.unsupervised.graphsage_runners as unsup_citation_graphsage_runners
import experiments.table1.runners.ppi.supervised.graphsage_runners as sup_ppi_graphsage_runners
import experiments.table1.runners.ppi.unsupervised.graphsage_runners as unsup_ppi_graphsage_runners
import experiments.table1.runners.reddit.supervised.graphsage_runners as sup_reddit_graphsage_runners
import experiments.table1.runners.reddit.unsupervised.graphsage_runners as unsup_reddit_graphsage_runners
import experiments.table1.settings as table1_settings

# Dict of all implemented runners
runners = {
    'cora': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.cora,
        },
        'unsupervised': {
            'graphsage': unsup_citation_graphsage_runners.cora,
        },
    },
    'citeseer': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.citeseer,
        },
        'unsupervised': {
            'graphsage': unsup_citation_graphsage_runners.citeseer,
        },
    },
    'pubmed': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.pubmed,
        },
        'unsupervised': {
            'graphsage': unsup_citation_graphsage_runners.pubmed,
        },
    },
    'ppi': {
        'supervised': {
            'graphsage': sup_ppi_graphsage_runners,
        },
        'unsupervised': {
            'graphsage': unsup_ppi_graphsage_runners,
        },
    },
    'reddit': {
        'supervised': {
            'graphsage': sup_reddit_graphsage_runners,
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
        else:
            return runners[dataset][training_mode][model]
    except KeyError:
        raise NotImplementedError
