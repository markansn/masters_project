import datetime
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
from load_features import load_features
def main():




    X, Y, t = load_features("../droidapiminer_data/apg-droidapiminer")
    # X = []
    # Y = []

    # for i, item in enumerate(X1):
    #     if item != {}:
    #         X.append(item)
    #         Y.append(Y1[i])
    #         t.append(t1[i])
    for i in range(0, len(t)):
        item = t[i]
        print(item.year)
        t[i] = datetime.datetime(item.year, 1, 1)

    print(t)
    print(len(X))
    vec = DictVectorizer()
    x = vec.fit_transform(X)
    y = np.asarray(Y)
    tv = np.asarray(t)
    print(len(X))

    splits = temporal.time_aware_train_test_split(
        x, y, tv, train_size=24, test_size=1, granularity='month')
    Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # Perform a timeline evaluation
    # clf = GridSearchCV(LinearSVC(), Parameters, cv= 5, scoring= 'f1', n_jobs=4, verbose=2 )
    clf = LinearSVC(max_iter=10000, C=1)
    # clf = RandomForestClassifier()
    results = evaluation.fit_predict_update(clf, *splits)

    # View results
    metrics.print_metrics(results)

    plt = viz.plot_decay(results)
    plt.show()



    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


if __name__ == '__main__':
    main()
