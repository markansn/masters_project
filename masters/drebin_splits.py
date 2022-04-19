import IPython
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import drebin_class_split
from masters.load_features import *
from tesseract import evaluation, temporal, metrics, viz
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree

from sklearn.feature_extraction import FeatureHasher
def main(removed, pltname):
    # Generate dummy predictors, labels and timestamps from Gaussians
    # X, Y, t = load_features("../features-bitbucket/drebin-parrot-v2-down-features")
    X, Y, t = load_no_apg("../features-bitbucket/drebin-parrot-v2-down-features")
    # X, Y, t = load_apg("../apg-features/apg")

    print(len(X))


    # vec = DictVectorizer()


    X = drebin_class_split.remove_class(X, removed)
    print(len(X))
    print(removed + " " + str(129728 - len(X)))
    # IPython.embed()
    # x = vec.fit_transform(X)
    # y = np.asarray(Y)
    # tv = np.asarray(t)
    # # IPython.embed()
    #
    # # Partition dataset
    # splits = temporal.time_aware_train_test_split(
    #     x, y, tv, train_size=12, test_size=1, granularity='month')
    #
    # clf = LinearSVC(max_iter=10000, C=1)
    #
    # results = evaluation.fit_predict_update(clf, *splits)
    # print(results)
    # metrics.print_metrics(results)
    #
    # # View AUT(F1, 24 months) as a measure of robustness over time
    # print(metrics.aut(results, 'f1'))
    #
    #
    # plt = viz.plot_decay(results)
    # plt.savefig("../figs/" + pltname)
    # # plt.show()
    #
    # with open("../data/" + pltname + "_" + ".txt", "w+") as f:
    #     f.write(str(results))
    # with open("../data/" + pltname + ".pickle", "wb+") as h:
    #     pickle.dump(results, h)



# "urls", "intents", "activities",
features = ["api_calls", "app_permissions", "api_permissions", "interesting_calls", "urls", "intents", "activities"]

for feature in features:
    main(feature, "drebin_split_" + feature)


# if __name__ == '__main__':
#     main()


