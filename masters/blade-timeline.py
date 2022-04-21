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
import IPython
from sklearn.feature_extraction.text import TfidfTransformer
def main():



    times_dict = {'2011.1': {0}, '2011.6': {0}, '2011.11': {0}, '2012.4': {0}, '2011.12': {0}, '2011.7': {0}, '2013.5': {0, 1}, '2012.10': {0, 1}, '2011.5': {0}, '2011.2': {0}, '2011.4': {0}, '2012.6': {0}, '2013.9': {0, 1}, '2011.9': {0}, '2012.7': {0, 1}, '2012.9': {0, 1}, '2011.3': {0}, '2013.4': {0, 1}, '2013.10': {0, 1}, '2012.2': {0}, '2012.5': {0}, '2011.8': {0}, '2013.7': {0, 1}, '2012.11': {0, 1}, '2012.3': {0, 1}, '2013.3': {0, 1}, '2013.2': {0, 1}, '2012.12': {0, 1}, '2011.10': {0}, '2013.11': {0, 1}, '2013.8': {0, 1}, '2013.6': {0, 1}, '2012.1': {0}, '2012.8': {0, 1}, '2013.1': {0, 1}, '2013.12': {0, 1}, '2014.2': {1}, '2014.5': {1}, '2014.6': {1}, '2014.3': {1}, '2014.4': {1}, '2014.1': {1}}

    X1, Y1, t1 = load_features("../blade/AA/apg-autopsy")
    X = []
    Y = []
    t = []
    for i, item in enumerate(X1): #2010
        if len(item) > 0 and t1[i].year > 2010 and len(times_dict[str(t1[i].year) + "." + str(t1[i].month)]) == 2:
            X.append(item)
            Y.append(Y1[i])
            t.append(t1[i])

    print(len(X))
    times = {}
    for i, time in enumerate(t):
        a = str(time.year) + "." + str(time.month)
        if a in times:
            times[a].add(Y[i])
        else:
            times[a] = set()
            times[a].add(Y[i])


    print(str(times))
    print(len(times))
    # IPython.embed()
    vec = DictVectorizer()
    x = vec.fit_transform(X)
    y = np.asarray(Y)
    tv = np.asarray(t)


    splits = temporal.time_aware_train_test_split(
        x, y, tv, train_size=10, test_size=1, granularity='month')
    Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # Perform a timeline evaluation
    # clf = GridSearchCV(LinearSVC(), Parameters, cv= 5, scoring= 'f1', n_jobs=4, verbose=2 )
    clf = LinearSVC(max_iter=10000, C=1)
    # clf = RandomForestClassifier()
    results = evaluation.fit_predict_update(clf, *splits)

    # View results
    metrics.print_metrics(results)

    plt = viz.plot_decay(results)
    plt.savefig("test")



    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


if __name__ == '__main__':
    main()
