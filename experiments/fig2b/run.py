import json
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
from matplotlib import pyplot as plt

from experiments.fig2b import runners
from experiments.fig2b import settings as fig2b_settings
from graphsage import settings

# Initialize the results dictionary
results = defaultdict(dict)

# Override global number of workers with the experiment setting
settings.NUM_WORKERS = fig2b_settings.NUM_WORKERS

# Evaluate only on validation and test set
settings.NO_EVAL_TRAIN = True

# Run experiments
for sample_size in fig2b_settings.SAMPLE_SIZES:
    print(f'Running experiment for sample_size {sample_size}')
    try:
        results[sample_size] = runners.get(sample_size).run()
    except NotImplementedError:
        print(f'Skipping {sample_size}')

# Create a timestamped and args-explicit named for the results folder
date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
folder_name = '_'.join([date] + [f'{k}={v}' for k, v in vars(settings.args).items() if v and not 
isinstance(v, list)])
results_path = osp.join(settings.RESULTS_DIR, 'fig2b', folder_name)

os.makedirs(results_path)

# Save results as a json file
json_path = osp.join(results_path, f'fig2b.json')
with open(json_path, 'w') as f:
    json.dump(results, f)

# Use results to plot the figure
xss = np.array(fig2b_settings.SAMPLE_SIZES)
data1 = np.array([results[ss]['test_f1'] for ss in xss])
data2 = np.array([results[ss]['test_time'] for ss in xss])

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Neighborhood sample size')
ax1.set_ylabel('Micro F1', color=color)
ax1.plot(xss, data1, color=color, marker='o', label='Micro F1')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Runtime', color=color)  # we already handled the x-label with ax1
ax2.plot(xss, data2, color=color, marker='o', linestyle='--', label='Runtime')
ax2.tick_params(axis='y', labelcolor=color)

fig.legend(loc='lower right')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Save the plot in the results directory
plt.savefig(osp.join(results_path, 'fig2b.png'))

# Print path to the results directory
print(f'Results saved to {results_path}')

# Show the plot
if not settings.args.no_show:
    plt.show()

plt.close()
