import json
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
from matplotlib import pyplot as plt

from experiments.fig2a import runners
from experiments.fig2a import settings as fig2a_settings
from graphsage import settings as graphsage_settings

# Initialize the results dictionary
results = defaultdict(lambda: defaultdict(dict))

# Run experiments
for model in fig2a_settings.MODELS:
    print(f'Running experiment for model {model}')
    try:
        results[model] = runners.get(model).run()
    except NotImplementedError:
        print(f'Skipping {model}')


# Create folder
date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
results_path = osp.join(graphsage_settings.RESULTS_DIR, 'fig2a', date)
os.makedirs(results_path)

# Save dictionary to json file
json_path = osp.join(results_path, f'fig2a.json')
with open(json_path, 'w') as f:
    json.dump(results, f)

# Use results to plot train and test time for each model in a bar plot
# Plot the results
x = results.keys()
_x = np.arange(len(results))
width = 0.3  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(_x - width / 2, [results[model]['train_time'] for model in results], width, label='Train time')
rects2 = ax.bar(_x + width / 2, [results[model]['test_time'] for model in results], width, label='Test time')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (seconds)')
plt.yscale('log')
ax.set_xlabel('Models')
ax.set_title('Training (per batch) and Testing Time')
ax.set_xticks(_x)
ax.set_xticklabels(x)
ax.legend(loc='upper left')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()


# Save the plot in results directory
plt.savefig(osp.join(results_path, 'fig2a.png'))

# Print path to results
print(f'Results saved to {results_path}')

# Show the plot
plt.show()

