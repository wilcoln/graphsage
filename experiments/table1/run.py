import json
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime as dt

from experiments.table1 import runners
from experiments.table1 import settings as table1_settings
from experiments.table1.latex import generate_latex_table
from graphsage import settings

# Initialize the results dictionary
results = defaultdict(lambda: defaultdict(dict))

# Evaluate only on validation and test set
settings.NO_EVAL_TRAIN = True

# Run experiments
for dataset in table1_settings.DATASETS:
    print(f'Running experiment for dataset {dataset}')
    for training_mode in table1_settings.TRAINING_MODES:
        print(f'Running experiment for training mode {training_mode}')
        for model in table1_settings.MODELS:
            print(f'Running experiment for model {model}')
            try:
                results[dataset][training_mode][model] = runners.get(dataset, training_mode, model).run()
            except NotImplementedError:
                print(f'Skipping {dataset}, {training_mode}, {model}')

# Post process results
for dataset in table1_settings.DATASETS:
    # Repeat performance across training mode for graph-oblivious models (as in the original paper)
    for model in set(table1_settings.GRAPH_OBLIVIOUS_MODELS) & set(table1_settings.MODELS):
        results[dataset]['unsupervised'][model] = results[dataset]['supervised'][model]
    # Add percentage f1 gain relative to raw features baseline
    for training_mode in table1_settings.TRAINING_MODES:
        for model in table1_settings.MODELS:
            try:
                results[dataset][training_mode][model]['percentage_f1_gain'] = \
                    (results[dataset][training_mode][model]['test_f1'] -
                     results[dataset][training_mode]['raw_features']['test_f1']) / \
                    results[dataset][training_mode]['raw_features']['test_f1']
            except:
                pass

# Create a timestamped and args-explicit named for the results folder
date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
folder_name = '_'.join([date] + [f'{k}={v}' for k, v in vars(settings.args).items() if v and not 
isinstance(v, list)])
results_path = osp.join(settings.RESULTS_DIR, 'table1', folder_name)

# Create the results folder
os.makedirs(results_path)

# Save results as a json file
json_path = osp.join(results_path, 'table1.json')
with open(json_path, 'w') as f:
    json.dump(results, f)

# Generate and save latex code for the table
table_path = osp.join(results_path, 'table1.tex')
with open(table_path, 'w') as f:
    latex_table = generate_latex_table(results)
    f.write(latex_table)
    print(latex_table)  # Print to console for convenience


# Print path to the results folder
print(f'Results saved to {results_path}')
