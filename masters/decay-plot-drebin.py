import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from masters.load_features import load_features
from tesseract import evaluation, temporal, metrics, viz


def main():
    # Generate dummy predictors, labels and timestamps from Gaussians
    X, Y, t = load_features("../apg-features/apg")

    print(len(Y))
    vec = DictVectorizer()
    x = vec.fit_transform(X)
    y = np.asarray(Y)
    tv = np.asarray(t)

    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        x, y, tv, train_size=12, test_size=1, granularity='month')

    # Perform a timeline evaluation
    clf = LinearSVC(max_iter=10000, C=1)
    results = evaluation.fit_predict_update(clf, *splits)

    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))

    plt = viz.plot_decay(results)
    plt.show()


if __name__ == '__main__':
    main()
