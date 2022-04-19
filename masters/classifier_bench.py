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


# ,
# classifiers = [] RandomForestClassifier() tree.DecisionTreeClassifier LinearSVC(max_iter=10000, C=1), KNeighborsClassifier(1), KNeighborsClassifier(5),
classifiers = [KNeighborsClassifier(10)]
results = []
for classifier in classifiers:
    vec = DictVectorizer()
    title = name + "_" + experiment + "_" + str(classifier)
    out, time_taken = decay_plot.main(mamadroid, vec, classifier, title , 12)
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