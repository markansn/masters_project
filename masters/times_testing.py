import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.append("..")
from sklearn.svm import LinearSVC
from sklearn import metrics as sklearn_metrics
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from tesseract import evaluation, temporal, metrics, mock, viz
import CommonModules as CM
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.feature_extraction import DictVectorizer
from masters.load_features import *
import IPython

big = []
small = []

small_sha = set()
big_sha = set()
with open('../features-bitbucket/drebin-parrot-v2-down-features-meta.json', 'r') as f:
    small = json.load(f)

with open('../apg-features/apg-meta.json', 'r') as f:
    big = json.load(f)


for item in small:
    small_sha.add(item["sha256"])
for item in big:
    big_sha.add(item["sha256"])

for item in big_sha:
    if item not in small_sha:
        print(item)