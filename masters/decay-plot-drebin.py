import IPython
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import drebin_class_split
from masters.load_features import *
from tesseract import evaluation, temporal, metrics, viz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree

from sklearn.feature_extraction import FeatureHasher
def main():
    # Generate dummy predictors, labels and timestamps from Gaussians
    # X, Y, t = load_features("../features-bitbucket/drebin-parrot-v2-down-features")
    X, Y, t = load_no_apg("../features-bitbucket/drebin-parrot-v2-down-features")
    # X, Y, t = load_apg("../apg-features/apg")

    # times = {}
    # times_count_pos = {}
    # times_count_neg = {}
    # for i, time in enumerate(t):
    #     a = str(time.year) + "." + str(time.month)
    #     if a in times:
    #         times[a].add(Y[i])
    #         if Y[i] == 1:
    #             times_count_neg[a] += 1
    #         else:
    #             times_count_pos[a] += 1
    #     else:
    #         times[a] = set()
    #         times[a].add(Y[i])
    #         if Y[i] == 1:
    #             times_count_neg[a] = 1
    #             times_count_pos[a] = 0
    #         else:
    #             times_count_pos[a] = 1
    #             times_count_neg[a] = 0


    # IPython.embed()

    # drebin_class_split.remove_class(X, "activities")
    print(len(X))
    # print(X)
    # vec = FeatureHasher()
    vec = DictVectorizer()
    # X_all = list(X.values())
    # vec = TfidfVectorizer()
    # X_all = []
    # for item in X:
    #     X_all.append(str(item))

    # drebin_class_split.remove_class(X, "api_permissions")
    # IPython.embed()
    x = vec.fit_transform(X)
    y = np.asarray(Y)
    tv = np.asarray(t)
    # IPython.embed()

    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        x, y, tv, train_size=12, test_size=1, granularity='month')
    # print some stuff out here
    # IPython.embed()
    # Perform a timeline evaluation
    # clf = LinearSVC(max_iter=10000, C=1)
    # clf = RandomForestClassifier(max_depth=10)
    # clf = KNeighborsClassifier()
    # clf = LinearSVC()
    clf = SGDClassifier(max_iter=10000)
    results = evaluation.fit_predict_update(clf, *splits)
    print(results)
    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


    plt = viz.plot_decay(results)
    plt.show()



if __name__ == '__main__':
    main()
