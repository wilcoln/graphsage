import experiments.table1.settings as table1_settings

# Dict of all implemented runners
import experiments.table1.runners.citation.supervised.graphsage_runners as sup_citation_graphsage_runners
import experiments.table1.runners.citation.unsupervised.graphsage_runners as unsup_citation_graphsage_runners
import experiments.table1.runners.ppi.supervised.graphsage_runners as sup_ppi_graphsage_runners
import experiments.table1.runners.ppi.unsupervised.graphsage_runners as unsup_ppi_graphsage_runners
import experiments.table1.runners.reddit.supervised.graphsage_runners as sup_reddit_graphsage_runners
import experiments.table1.runners.reddit.unsupervised.graphsage_runners as unsup_reddit_graphsage_runners

runners = {
    # dataset, model, training_mode: runner
    ('citation', 'supervised', 'graphsage_mean'): sup_citation_graphsage_runners.get('mean'),
    ('citation', 'supervised', 'graphsage_gcn'): sup_citation_graphsage_runners.get('gcn'),
    ('citation', 'supervised', 'graphsage_lstm'): sup_citation_graphsage_runners.get('lstm'),
    ('citation', 'supervised', 'graphsage_max'): sup_citation_graphsage_runners.get('max'),
    ('citation', 'unsupervised', 'graphsage_mean'): unsup_citation_graphsage_runners.get('mean'),
    ('citation', 'unsupervised', 'graphsage_gcn'): unsup_citation_graphsage_runners.get('gcn'),
    ('citation', 'unsupervised', 'graphsage_lstm'): unsup_citation_graphsage_runners.get('lstm'),
    ('citation', 'unsupervised', 'graphsage_max'): unsup_citation_graphsage_runners.get('max'),

    ('ppi', 'supervised', 'graphsage_mean'): sup_ppi_graphsage_runners.get('mean'),
    ('ppi', 'supervised', 'graphsage_gcn'): sup_ppi_graphsage_runners.get('gcn'),
    ('ppi', 'supervised', 'graphsage_lstm'): sup_ppi_graphsage_runners.get('lstm'),
    ('ppi', 'supervised', 'graphsage_max'): sup_ppi_graphsage_runners.get('max'),
    ('ppi', 'unsupervised', 'graphsage_mean'): unsup_ppi_graphsage_runners.get('mean'),
    ('ppi', 'unsupervised', 'graphsage_gcn'): unsup_ppi_graphsage_runners.get('gcn'),
    ('ppi', 'unsupervised', 'graphsage_lstm'): unsup_ppi_graphsage_runners.get('lstm'),
    ('ppi', 'unsupervised', 'graphsage_max'): unsup_ppi_graphsage_runners.get('max'),

    # ('reddit', 'supervised', 'graphsage_mean'): sup_reddit_graphsage_runners.get('mean'),
    # ('reddit', 'supervised', 'graphsage_gcn'): sup_reddit_graphsage_runners.get('gcn'),
    # ('reddit', 'supervised', 'graphsage_lstm'): sup_reddit_graphsage_runners.get('lstm'),
    # ('reddit', 'supervised', 'graphsage_max'): sup_reddit_graphsage_runners.get('max'),
    # ('reddit', 'unsupervised', 'graphsage_mean'): unsup_reddit_graphsage_runners.get('mean'),
    # ('reddit', 'unsupervised', 'graphsage_gcn'): unsup_reddit_graphsage_runners.get('gcn'),
    # ('reddit', 'unsupervised', 'graphsage_lstm'): unsup_reddit_graphsage_runners.get('lstm'),
    # ('reddit', 'unsupervised', 'graphsage_max'): unsup_reddit_graphsage_runners.get('max'),
}


def get(dataset, training_mode, model):
    """Returns the runner for the given dataset, model and training mode."""
    assert dataset in table1_settings.DATASETS, 'Unknown dataset: {}'.format(dataset)
    assert training_mode in table1_settings.TRAINING_MODES, 'Unknown training mode: {}'.format(training_mode)
    assert model in table1_settings.MODELS, 'Unknown model: {}'.format(model)
    try:
        return runners[dataset, training_mode, model]
    except KeyError:
        raise NotImplementedError

