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




    X, Y, t = load_features("../apg-features/apg")

    vec = DictVectorizer()
    x = vec.fit_transform(X)
    y = np.asarray(Y)
    tv = np.asarray(t)


    splits = temporal.time_aware_train_test_split(
        x, y, tv, train_size=12, test_size=1, granularity='month')
    # Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # Perform a timeline evaluation
    # clf = GridSearchCV(LinearSVC(), Parameters, cv= 5, scoring= 'f1', n_jobs=4, verbose=2 )
    clf = LinearSVC(max_iter=10000)
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
