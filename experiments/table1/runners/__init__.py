import experiments.table1.runners.citation.supervised.graphsage_runners as sup_citation_graphsage_runners
import experiments.table1.runners.citation.supervised.graphsaint_runners as sup_citation_graphsaint_runners
import experiments.table1.runners.citation.supervised.shadow_runners as sup_citation_shadow_runners
import experiments.table1.runners.citation.supervised.random_runners as sup_citation_random_runners
import experiments.table1.runners.citation.supervised.raw_features_runners as sup_citation_raw_features_runners
import experiments.table1.runners.citation.unsupervised.graphsage_runners as unsup_citation_graphsage_runners
import experiments.table1.runners.ppi.supervised.graphsage_runners as sup_ppi_graphsage_runners
import experiments.table1.runners.ppi.supervised.random_runners as sup_ppi_random_runners
import experiments.table1.runners.ppi.supervised.raw_features_runners as sup_ppi_raw_features_runners
import experiments.table1.runners.ppi.unsupervised.graphsage_runners as unsup_ppi_graphsage_runners
import experiments.table1.runners.reddit.supervised.graphsage_runners as sup_reddit_graphsage_runners
import experiments.table1.runners.reddit.supervised.graphsaint_runners as sup_reddit_graphsaint_runners
import experiments.table1.runners.reddit.supervised.shadow_runners as sup_reddit_shadow_runners
import experiments.table1.runners.reddit.supervised.random_runners as sup_reddit_random_runners
import experiments.table1.runners.reddit.supervised.raw_features_runners as sup_reddit_raw_features_runners
import experiments.table1.runners.reddit.unsupervised.graphsage_runners as unsup_reddit_graphsage_runners
import experiments.table1.runners.flickr.supervised.graphsage_runners as sup_flickr_graphsage_runners
import experiments.table1.runners.flickr.unsupervised.graphsage_runners as unsup_flickr_graphsage_runners
import experiments.table1.runners.flickr.supervised.graphsaint_runners as sup_flickr_graphsaint_runners
# import experiments.table1.runners.flickr.unsupervised.graphsaint_runners as unsup_flickr_graphsaint_runners
import experiments.table1.runners.flickr.supervised.shadow_runners as sup_flickr_shadow_runners
# import experiments.table1.runners.flickr.unsupervised.shadow_runners as unsup_flickr_shadow_runners

# import experiments.table1.runners.mutag.supervised.graphsage_runners as sup_mutag_graphsage_runners
# import experiments.table1.runners.mutag.unsupervised.graphsage_runners as unsup_mutag_graphsage_runners
# import experiments.table1.runners.mutag.supervised.graphsaint_runners as sup_mutag_graphsaint_runners
# import experiments.table1.runners.mutag.unsupervised.graphsaint_runners as unsup_mutag_graphsaint_runners
# import experiments.table1.runners.mutag.supervised.shadow_runners as sup_mutag_shadow_runners
# import experiments.table1.runners.mutag.unsupervised.shadow_runners as unsup_mutag_shadow_runners

import experiments.table1.settings as table1_settings
from experiments.table1.runners.others import triples_models_runners

# Dict of all implemented runners
runners = {
    'cora': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.cora,
            'raw_features': sup_citation_raw_features_runners.cora,
            'random': sup_citation_random_runners.cora,
            'graphsaint': sup_citation_graphsaint_runners.cora,
            'shadow': sup_citation_shadow_runners.cora,
        },
        'unsupervised': {
            'graphsage': unsup_citation_graphsage_runners.cora,
        },
    },
    'citeseer': {
        'supervised': {
            'graphsage': sup_citation_graphsage_runners.citeseer,
            'raw_features': sup_citation_raw_features_runners.citeseer,
            'random': sup_citation_random_runners.citeseer,
            'graphsaint': sup_citation_graphsaint_runners.citeseer,
            'shadow': sup_citation_shadow_runners.citeseer,
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
            'graphsaint': sup_citation_graphsaint_runners.pubmed,
            'shadow': sup_citation_shadow_runners.pubmed,
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
            'graphsaint': sup_reddit_graphsaint_runners,
            'shadow': sup_reddit_shadow_runners,
        },
        'unsupervised': {
            'graphsage': unsup_reddit_graphsage_runners,
        },
    },
    'flickr': {
        'supervised': {
            'graphsage': sup_flickr_graphsage_runners,
            'graphsaint': sup_flickr_graphsaint_runners,
            'shadow': sup_flickr_shadow_runners,
        },
        'unsupervised': {
            'graphsage': unsup_flickr_graphsage_runners,
            # 'graphsaint': unsup_flickr_graphsaint_runners,
            # 'shadow': unsup_flickr_shadow_runners,
        },
    },
    # 'mutag': {
    #     'supervised': {
    #         'graphsage': sup_flickr_graphsage_runners,
    #         'graphsaint': sup_flickr_graphsaint_runners,
    #         'shadow': sup_flickr_shadow_runners,
    #     },
    #     'unsupervised': {
    #         'graphsage': unsup_mutag_graphsage_runners,
    #         'graphsaint': unsup_mutag_graphsaint_runners,
    #         'shadow': unsup_mutag_shadow_runners,
    #     },
    # },
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
        if 'graphsaint' in model:
            aggregator = model.split('_')[1]
            return runners[dataset][training_mode]['graphsaint'].get(aggregator)
        if 'shadow' in model:
            aggregator = model.split('_')[1]
            return runners[dataset][training_mode]['shadow'].get(aggregator)
        if 'triples' in model:
            return triples_models_runners.get(dataset, training_mode, model)
        else:
            return runners[dataset][training_mode][model].get()
    except KeyError:
        raise NotImplementedError
