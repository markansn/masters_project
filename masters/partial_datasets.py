import IPython
import numpy as np
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import drebin_class_split
from load_features import *
from tesseract import evaluation, temporal, metrics, viz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
import decay_plot
import json
import pickle

experiment = "datasetsizes"
# methods = ["../features-bitbucket/drebin-parrot-v2-down-features", "../mamadroid/mamadroid"]
# names = ["drebin", "mamadroid"]
methods = ["../mamadroid/mamadroid"]
names = ["mamadroid"]
# 0.1 0.9 0.7,
dataset_percentages = [0.5, 0.3]
for j, method in enumerate(methods):
    for percent in dataset_percentages:
        pltname = names[j] + "_" + experiment + "_" + str(percent).split(".")[1] + "output"
        X, Y, t = load_no_apg(method)


        #partial dataset code
        X_out = []
        Y_out = []
        t_out = []
        times_neg = {}
        times_pos = {}
        for i, time in enumerate(t):
            a = str(time.year) + "." + str(time.month)
            if Y[i] == 1:
                if a in times_neg:
                    times_neg[a].append([X[i], Y[i], t[i]])
                else:
                    times_neg[a] = [[X[i], Y[i], t[i]]]
            else:
                if a in times_pos:
                    times_pos[a].append([X[i] , Y[i], t[i]])
                else:
                    times_pos[a] = [[X[i], Y[i], t[i]]]

        for month in times_neg:
            num = len(times_neg[month])
            count = round(num * percent)
            for a in range(0, count):
                X_out.append(times_neg[month][a][0])
                Y_out.append(times_neg[month][a][1])
                t_out.append(times_neg[month][a][2])

        for month in times_pos:
            num = len(times_pos[month])
            count = round(num * percent)
            for a in range(0, count):
                X_out.append(times_pos[month][a][0])
                Y_out.append(times_pos[month][a][1])
                t_out.append(times_pos[month][a][2])


        print(len(X_out))



        vec = DictVectorizer()
        x = vec.fit_transform(X_out)
        y = np.asarray(Y_out)
        tv = np.asarray(t_out)

        # Partition dataset
        splits = temporal.time_aware_train_test_split(
            x, y, tv, train_size=12, test_size=1, granularity='month')

        clf = LinearSVC(max_iter=10000, C=1)

        results = evaluation.fit_predict_update(clf, *splits)
        print(results)
        metrics.print_metrics(results)

        # View AUT(F1, 24 months) as a measure of robustness over time
        print(metrics.aut(results, 'f1'))

        plt = viz.plot_decay(results)
        plt.savefig("../datasetsizes/" + pltname)
        # plt.show()

        with open("../datasetsizes/" + pltname + "_" + ".txt", "w+") as f:
            f.write(str(results))
        with open("../datasetsizes/" + pltname + ".pickle", "wb+") as h:
            pickle.dump(results, h)
        with open("../datasetsizes/" + pltname + "_times.txt", "w+") as l:
            l.write(str(len(X_out)))

