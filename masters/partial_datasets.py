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
methods = ["../mamadroid/mamadroid"]
names = ["mamadroid"]
# methods = ["../blade/AA/apg-autopsy"] "../features-bitbucket/drebin-parrot-v2-down-features",  "../mamadroid/mamadroid" "mamadroid" "drebin",
# names = ["blade"]
#
dataset_percentages = [0.1]
for j, method in enumerate(methods):
    for percent in dataset_percentages:
        pltname = names[j] + "_" + experiment + "_" + str(percent).split(".")[1] + "output"
        X, Y, t = load_features(method)

        if method == "../blade/AA/apg-autopsy":
            times_dict = {'2011.1': {0}, '2011.6': {0}, '2011.11': {0}, '2012.4': {0}, '2011.12': {0}, '2011.7': {0},
                          '2013.5': {0, 1}, '2012.10': {0, 1}, '2011.5': {0}, '2011.2': {0}, '2011.4': {0},
                          '2012.6': {0}, '2013.9': {0, 1}, '2011.9': {0}, '2012.7': {0, 1}, '2012.9': {0, 1},
                          '2011.3': {0}, '2013.4': {0, 1}, '2013.10': {0, 1}, '2012.2': {0}, '2012.5': {0},
                          '2011.8': {0}, '2013.7': {0, 1}, '2012.11': {0, 1}, '2012.3': {0, 1}, '2013.3': {0, 1},
                          '2013.2': {0, 1}, '2012.12': {0, 1}, '2011.10': {0}, '2013.11': {0, 1}, '2013.8': {0, 1},
                          '2013.6': {0, 1}, '2012.1': {0}, '2012.8': {0, 1}, '2013.1': {0, 1}, '2013.12': {0, 1},
                          '2014.2': {1}, '2014.5': {1}, '2014.6': {1}, '2014.3': {1}, '2014.4': {1}, '2014.1': {1}}

            X1 = X
            Y1 = Y
            t1 = t
            X = []
            Y = []
            t = []
            for i, item in enumerate(X1):  # 2010
                if len(item) > 0 and t1[i].year > 2010 and len(
                        times_dict[str(t1[i].year) + "." + str(t1[i].month)]) == 2:
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
            x, y, tv, train_size=10, test_size=1, granularity='month')

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

