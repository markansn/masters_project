from masters.load_features import *
import numpy as np

from sklearn.feature_extraction import DictVectorizer

X, y, t = load_features("/Users/sullivan/Downloads/apg-features/apg")

vec = DictVectorizer()
Xv = vec.fit_transform(X)
yv = np.asarray(y)
tv = np.asarray(t)

import IPython
IPython.embed()
