import os
import os.path as osp
from graphsage.settings import CACHE_DIR, RESULTS_DIR

for d in [CACHE_DIR, RESULTS_DIR]:
    if not osp.exists(d):
        os.makedirs(d)
