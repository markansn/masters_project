import sys
import matplotlib.pyplot as plt

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

def main():




    X, Y, t = load_features("../mamadroid/mamadroid")
    print(Y)
    vec = DictVectorizer()
    x = vec.fit_transform(X)
    print(x)
    y = np.asarray(Y)
    print(y)
    tv = np.asarray(t)

    # clf = RandomForestClassifier()
    IPython.embed()
    clf = LinearSVC(max_iter=10000)

    num_of_splits = 10
    skf = StratifiedKFold(n_splits=num_of_splits)
    lst_accu_stratified = []
    for train_index, test_index in skf.split(x, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        lst_accu_stratified.append(sklearn_metrics.f1_score(y_test, pred))
        print(sklearn_metrics.f1_score(y_test, pred))
        print(sklearn_metrics.precision_score(y_test, pred))
        print(sklearn_metrics.recall_score(y_test, pred))  # check args
        print("-----------")
        fpr, tpr, thresholds = sklearn_metrics.roc_curve(y_test, pred)
        roc_auc = sklearn_metrics.auc(fpr, tpr)
        display = sklearn_metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        plt.show()

    print(sum(lst_accu_stratified) / float(len(lst_accu_stratified)))





if __name__ == '__main__':
    main()
