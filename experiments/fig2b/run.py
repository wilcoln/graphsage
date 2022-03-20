import os.path as osp
import json
from collections import defaultdict

import numpy as np
from icecream import ic
from matplotlib import pyplot as plt

from graphsage import settings as graphsage_settings
from experiments.fig2b import settings as fig2b_settings
from experiments.fig2b import runners

# Initialize the results dictionary
results = defaultdict(dict)


# Run experiments
for sample_size in fig2b_settings.SAMPLE_SIZES:
    print(f'Running experiment for sample_size {sample_size}')
    try:
        results[sample_size] = runners.get(sample_size).run()
    except NotImplementedError:
        print(f'Skipping {sample_size}')

# Save results
# Save dictionary to json file
results_path = osp.join(graphsage_settings.RESULTS_DIR, f'fig2b.json')
with open(results_path, 'w') as f:
    json.dump(results, f)


xss = np.array(fig2b_settings.SAMPLE_SIZES)
data1 = np.array([results[ss]['test_f1'] for ss in xss])
data2 = np.array([results[ss]['test_time'] for ss in xss])

ic()
fig, ax1 = plt.subplots()  # TODO: Fix blocking caused by this line
ic()

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

fig.tight_layout()  # otherwise the right y-label is slightly clipped


# Save the plot in results directory
plt.savefig(osp.join(graphsage_settings.RESULTS_DIR, 'fig2b.png'))


# Show the plot
plt.show()