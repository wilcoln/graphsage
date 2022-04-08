import itertools
import json
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
from matplotlib import pyplot as plt

from experiments.fig3 import runners
from experiments.fig3 import settings as fig3_settings
from graphsage import settings

# Initialize the results dictionary
results = defaultdict(dict)

# Evaluate only on validation and test set
settings.NO_EVAL_TRAIN = True

# Run experiments
for model in fig3_settings.MODELS:
    print(f'Running experiment for model {model}')
    for noise_prop in fig3_settings.FEATURE_NOISE_PROP:
        print(f'Running experiment for noise proportion {noise_prop:.2f}')
        try:
            results[model][noise_prop] = runners.get(model, noise_prop).run()
        except NotImplementedError:
            print(f'Skipping {model} and {noise_prop:.2f}')

# Create a timestamped and args-explicit named for the results folder
date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
folder_name = '_'.join([date] + [f'{k}={v}' for k, v in vars(settings.args).items() if v and not 
isinstance(v, list)])
results_path = osp.join(settings.RESULTS_DIR, 'fig3', folder_name)

os.makedirs(results_path)

# Save results as a json file
json_path = osp.join(results_path, f'fig3.json')
with open(json_path, 'w') as f:
    json.dump(results, f)


# Use results to plot the figure
linestyle = itertools.cycle(['-', '--', '-.', ':'])
for model in results:
    y = np.array([results[model][noise_prop]['test_f1'] for noise_prop in fig3_settings.FEATURE_NOISE_PROP])
    plt.plot(fig3_settings.FEATURE_NOISE_PROP, y, linestyle=next(linestyle), label=model)


plt.xlabel('Feature Noise Proportion')
plt.ylabel('Micro F1')
plt.legend([model for model in fig3_settings.MODELS])


plt.tight_layout()  # otherwise the right y-label is slightly clipped

# Save the plot in the results directory
plt.savefig(osp.join(results_path, 'fig3.png'))

# Print path to the results directory
print(f'Results saved to {results_path}')

# Show the plot
if not settings.args.no_show:
    plt.show()

plt.close()
