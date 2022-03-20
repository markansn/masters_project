import sys
sys.path.append("..")
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from tesseract import evaluation, temporal, metrics, mock, viz
import CommonModules as CM
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
def main():
    times = {}
    with open("../examples/times.pickle", "rb") as a:
        times = pickle.load(a)

    # FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'),
    #                        binary=True)
    MalwareCorpus = '../drebin_data/badware_created'
    GoodwareCorpus = '../drebin_data/goodware_created'
    AllMalSamples = CM.IO.ListFiles(MalwareCorpus, ".data")
    AllGoodSamples = CM.IO.ListFiles(GoodwareCorpus, ".data")
    print(len(AllMalSamples))
    print(len(AllGoodSamples))
    # print(AllMalSamples)
    # print(str(AllMalSamples[0].split("/")[7].split(".")[0]))
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'),
                           binary=False)
    x = FeatureVectorizer.fit_transform(AllMalSamples + AllGoodSamples)
    # print(len(AllMalSamples))
    Mal_labels = np.ones(len(AllMalSamples))
    # print(AllMalSamples)
    Good_labels = np.empty(len(AllGoodSamples))
    Good_labels.fill(0)
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    labels = AllMalSamples + AllGoodSamples
    times_arr = []
    for item in labels:
        name = item.split("/")[7].split(".")[0]
        times_arr.append(times[name])
        print(name)

    print(times_arr)




    print("Label array - generated")

    print(x)


    # X, y_notused, t = mock.generate_binary_test_data(len(AllMalSamples) + len(AllGoodSamples), '2014', '2016')
    # print(X)
    # print(y)
    # print(t)



    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        x, y, np.array(times_arr), train_size=12, test_size=1, granularity='month')
    # Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # Perform a timeline evaluation
    # clf = GridSearchCV(LinearSVC(), Parameters, cv= 5, scoring= 'f1', n_jobs=4, verbose=2 )
    clf = LinearSVC()
    results = evaluation.fit_predict_update(clf, *splits)

    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


if __name__ == '__main__':
    main()
