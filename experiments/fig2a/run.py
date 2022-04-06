import json
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
from matplotlib import pyplot as plt

from experiments.fig2a import runners
from experiments.fig2a import settings as fig2a_settings
from graphsage import settings

# Initialize the results dictionary
results = defaultdict(lambda: defaultdict(dict))

# Evaluate only on validation and test set
settings.NO_EVAL_TRAIN = True

# Run experiments
for model in fig2a_settings.MODELS:
    print(f'Running experiment for model {model}')
    try:
        results[model] = runners.get(model).run()
    except NotImplementedError:
        print(f'Skipping {model}')

# Create a timestamped and args-explicit named for the results folder
date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
folder_name = '_'.join([date] + [f'{k}={v}' for k, v in vars(settings.args).items() if v and not 
isinstance(v, list)])
results_path = osp.join(settings.RESULTS_DIR, 'fig2a', folder_name)
os.makedirs(results_path)

# Save results as a json file
json_path = osp.join(results_path, f'fig2a.json')
with open(json_path, 'w') as f:
    json.dump(results, f)

# Use results to plot train and test time for each model in a bar plot
x = results.keys()
_x = np.arange(len(results))
width = 0.3  # the width of the bars

fig, ax = plt.subplots()

label1 = 'Training (per batch)'
label2 = 'Inference (full test set)'
rects1 = ax.bar(_x - width / 2, [results[model]['train_time'] for model in results], width, label=label1)
rects2 = ax.bar(_x + width / 2, [results[model]['test_time'] for model in results], width, label=label2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
ax.set_xlabel('Models')
ax.set_title('Training (per batch) and Testing Time')
ax.set_xticks(_x)
ax.set_xticklabels(x)
ax.legend(loc='upper left')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Save the plot in the results directory
plt.savefig(osp.join(results_path, 'fig2a.png'))

# Print path to the results directory
print(f'Results saved to {results_path}')

# Show the plot
if not settings.args.no_show:
    plt.show()

plt.close()
