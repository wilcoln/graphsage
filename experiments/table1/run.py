import os.path as osp
import json
from collections import defaultdict
from graphsage import settings as graphsage_settings
from experiments.table1 import settings as table1_settings
from experiments.table1 import runners

# Initialize the results dictionary
results = defaultdict(lambda: defaultdict(dict))

# Run experiments
for dataset in table1_settings.DATASETS:
    print(f'Running experiment for dataset {dataset}')
    for model in table1_settings.MODELS:
        print(f'Running experiment for model {model}')
        for training_mode in table1_settings.TRAINING_MODES:
            print(f'Running experiment for training mode {training_mode}')
            try:
                results[dataset][model][training_mode] = runners.get(dataset, model, training_mode).run()
            except NotImplementedError:
                print(f'Skipping {dataset}, {model}, {training_mode}')


# Add percentage f1 gain relative to raw features baseline
for dataset in table1_settings.DATASETS:
    for model in table1_settings.MODELS:
        for training_mode in table1_settings.TRAINING_MODES:
            try:
                results[dataset][model][training_mode]['percentage_f1_gain'] = \
                    (results[dataset][model][training_mode]['test_f1'] -
                     results[dataset]['raw_features'][training_mode]['test_f1']) / \
                    results[dataset]['raw_features'][training_mode]['test_f1']
            except:
                pass

# Save results
results_path = osp.join(graphsage_settings.RESULTS_DIR, f'table1.json')
with open(results_path, 'w') as f:
    json.dump(results, f)

# Print path to results
print(f'Results saved to {results_path}')




