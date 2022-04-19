import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from masters.load_features import load_features
from tesseract import temporal, evaluation, metrics,  viz
from tesseract.selection import UncertaintySamplingSelector


def main():
    X, Y, t = load_features("../apg-features/apg")

    vec = DictVectorizer()
    x = vec.fit_transform(X)
    y = np.asarray(Y)
    tv = np.asarray(t)

    splits = temporal.time_aware_train_test_split(
        x, y, tv, train_size=12, test_size=1, granularity='month')

    clf = LinearSVC(max_iter=10000, C=1)

    selector = UncertaintySamplingSelector('20%')
    results = evaluation.fit_predict_update(clf, *splits, selectors=[selector])

    metrics.print_metrics(results)

    print('Number of test objects selected each period:')
    print(results['selected'])

    print('Array indices for selected objects from first test period:')
    print(selector.selection_history[0])
    plt = viz.plot_decay(results)
    plt.show()

if __name__ == '__main__':
    main()
