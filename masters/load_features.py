import logging
import json
from datetime import datetime
import time

def load_features(fname, shas=False):
    # BRUUUH BLINDSPOT IS THIS INCORRECT
    """Load feature set. 

    Args:
        feature_set (str): The common prefix for the dataset. 
            (e.g., 'data/features/drebin' -> 'data/features/drebin-[X|Y|meta].json')

        shas (bool): Whether to include shas. In some versions of the dataset, 
            shas were included to double-check alignment - these are _not_ features 
            and _must_ be removed before training. 
    
    Returns:
        Tuple[List[Dict], List, List]: The features, labels, and timestamps 
            for the dataset. 

    """
    logging.info('Loading features...')
    with open('{}-X.json'.format(fname), 'r') as f:
        X = json.load(f)
    # if not shas:
    #     [o.pop('sha256') for o in X]

    logging.info('Loading labels...')
    with open('{}-y.json'.format(fname), 'rt') as f:
        y = json.load(f)
    if 'apg' not in fname:
        y = [o[0] for o in y]

    logging.info('Loading timestamps...')
    with open('{}-meta.json'.format(fname), 'rt') as f:
        t = json.load(f)
    t = [o['dex_date'] for o in t]
    if 'apg' not in fname:
        t = [datetime.strptime(o if isinstance(o, str) else time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(o)),
                               '%Y-%m-%dT%H:%M:%S') for o in t]
    else:
        t = [datetime.strptime(o if isinstance(o, str) else time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(o)),
                               '%Y-%m-%d %H:%M:%S') for o in t]
    return X, y, t
