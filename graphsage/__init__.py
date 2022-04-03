import os
import os.path as osp

from graphsage.settings import DATA_DIR, RESULTS_DIR

for d in [DATA_DIR, RESULTS_DIR]:
    if not osp.exists(d):
        os.makedirs(d)
