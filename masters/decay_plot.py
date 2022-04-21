import IPython
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import drebin_class_split
from masters.load_features import *
from tesseract import evaluation, temporal, metrics, viz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import time
def main(load_src, vec, clf, pltname, train_size):
    # Generate dummy predictors, labels and timestamps from Gaussians
    # X, Y, t = load_features("../features-bitbucket/drebin-parrot-v2-down-features")
    X, Y, t = load_features(load_src)
    # X, Y, t = load_apg("../apg-features/apg")

    times_dict = {'2011.1': {0}, '2011.6': {0}, '2011.11': {0}, '2012.4': {0}, '2011.12': {0}, '2011.7': {0}, '2013.5': {0, 1}, '2012.10': {0, 1}, '2011.5': {0}, '2011.2': {0}, '2011.4': {0}, '2012.6': {0}, '2013.9': {0, 1}, '2011.9': {0}, '2012.7': {0, 1}, '2012.9': {0, 1}, '2011.3': {0}, '2013.4': {0, 1}, '2013.10': {0, 1}, '2012.2': {0}, '2012.5': {0}, '2011.8': {0}, '2013.7': {0, 1}, '2012.11': {0, 1}, '2012.3': {0, 1}, '2013.3': {0, 1}, '2013.2': {0, 1}, '2012.12': {0, 1}, '2011.10': {0}, '2013.11': {0, 1}, '2013.8': {0, 1}, '2013.6': {0, 1}, '2012.1': {0}, '2012.8': {0, 1}, '2013.1': {0, 1}, '2013.12': {0, 1}, '2014.2': {1}, '2014.5': {1}, '2014.6': {1}, '2014.3': {1}, '2014.4': {1}, '2014.1': {1}}

    if load_src == "../blade/AA/apg-autopsy":
        X1 = X
        Y1 = Y
        t1 = t
        X = []
        Y = []
        t = []
        for i, item in enumerate(X1):  # 2010
            if len(item) > 0 and t1[i].year > 2010 and len(times_dict[str(t1[i].year) + "." + str(t1[i].month)]) == 2:
                X.append(item)
                Y.append(Y1[i])
                t.append(t1[i])

        print(len(X))
        times = {}
        for i, timey in enumerate(t):
            a = str(timey.year) + "." + str(timey.month)
            if a in times:
                times[a].add(Y[i])
            else:
                times[a] = set()
                times[a].add(Y[i])



    print(len(X))

    if str(vec) == str(TfidfVectorizer()) or str(vec) == str(CountVectorizer()):
        print("tfidf")
        X_all = []
        for item in X:
            X_all.append(str(item))
        X = X_all
    x = vec.fit_transform(X)
    y = np.asarray(Y)
    tv = np.asarray(t)
    # IPython.embed()

    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        x, y, tv, train_size=train_size, test_size=1, granularity='month')

    start = time.time()
    results = evaluation.fit_predict_update(clf, *splits)
    time_taken = time.time() - start

    # print(results)
    # # View results
    metrics.print_metrics(results)
    #
    # # View AUT(F1, 24 months) as a measure of robustness over time
    # print(metrics.aut(results, 'f1'))
    #
    #
    plt = viz.plot_decay(results)
    # plt.show()
    plt.savefig("../figs/" + pltname)

    return results, time_taken


if __name__ == '__main__':
    main()
