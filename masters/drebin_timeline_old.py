from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from tesseract import evaluation, temporal, metrics, mock, viz


def main():
    # Generate dummy predictors, labels and timestamps from Gaussians

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'),
                           binary=True)

    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data")
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data")
    x = FeatureVectorizer.fit_transform(AllMalSamples + AllGoodSamples)

    X, y, t = mock.generate_binary_test_data(10, '2014', '2016')
    print(X)
    print(y)
    print(t)
    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        X, y, t, train_size=12, test_size=1, granularity='month')

    # Perform a timeline evaluation
    clf = LinearSVC()
    results = evaluation.fit_predict_update(clf, *splits)

    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


if __name__ == '__main__':
    main()
