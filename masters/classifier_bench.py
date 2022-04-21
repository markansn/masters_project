import IPython
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import drebin_class_split
from load_features import *
from tesseract import evaluation, temporal, metrics, viz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import decay_plot
import json
import pickle
name = "mamadroid"
experiment = "classifiers"

drebin = "../features-bitbucket/drebin-parrot-v2-down-features"
mamadroid = "../mamadroid/mamadroid"
blade = "../blade/AA/apg-autopsy"
types = ["../mamadroid/mamadroid", "../features-bitbucket/drebin-parrot-v2-down-features" ]
names = ["mamadroid", "drebin"]
# , tree.DecisionTreeClassifier(),
# KNeighborsClassifier(1), KNeighborsClassifier(5), KNeighborsClassifier(10), SGDClassifier(max_iter=10000), tree.DecisionTreeClassifier(), RandomForestClassifier() RandomForestClassifier() "drebin",  "../features-bitbucket/drebin-parrot-v2-down-features",
# classifiers = [] RandomForestClassifier() tree.DecisionTreeClassifier LinearSVC(max_iter=10000, C=1), KNeighborsClassifier(1), KNeighborsClassifier(5), LinearSVC(max_iter=10000, C=1)
classifiers = [RandomForestClassifier()]
results = []
for i, type in enumerate(types):
    print(type)
    for classifier in classifiers:
        print(classifier)
        vec = DictVectorizer()
        title = names[i] + "_" + experiment + "_" + str(classifier)
        out, time_taken = decay_plot.main(type, vec, classifier, title , 12) #set for drebin
        print(out)
        # dumping = {}
        # for item in out:
        #     if isinstance(item, str):
        #         dumping[item] = out[item]

        # IPython.embed()
        with open("../data/" + title + ".txt", "w+") as f:
            f.write(str(out))
        with open("../data/" + title + ".pickle", "wb+") as h:
            pickle.dump(out, h)
        with open("../data/" + title + "_time.txt", "w+") as l:
            l.write(str(time_taken))



print(results)