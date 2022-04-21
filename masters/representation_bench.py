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
from sklearn.feature_extraction.text import CountVectorizer
import json
import pickle
# name = "mamadroid"
experiment = "representations"

drebin = "../features-bitbucket/drebin-parrot-v2-down-features"
mamadroid = "../mamadroid/mamadroid"
#  CountVectorizer()
vectors = [FeatureHasher(), TfidfVectorizer(), DictVectorizer()]
# , "../features-bitbucket/drebin-parrot-v2-down-features", "drebin", "../mamadroid/mamadroid" "mamadroid"
# classifiers = [SGDClassifier(max_iter=10000), KNeighborsClassifier(1), KNeighborsClassifier(5), KNeighborsClassifier(10), LinearSVC(max_iter=10000, C=1)]
classifiers = [LinearSVC(max_iter=10000, C=1)]

results = []
# methods = ["../features-bitbucket/drebin-parrot-v2-down-features"]
# names = ["drebin"]
names = ["blade"]
methods = ["../blade/AA/apg-autopsy"]
for i, method in enumerate(methods):
    for vec in vectors:
        for classifier in classifiers:
            name = names[i] + "_" + experiment + "_" + str(classifier) + "_" + str(vec)
            out, time_taken = decay_plot.main(method, vec, classifier, name, 10)
            print(out)
            # dumping = {}
            # for item in out:
            #     if isinstance(item, str):
            #         dumping[item] = out[item]

            # IPython.embed()
            with open("../data/" + name + "_" + ".txt", "w+") as f:
                f.write(str(out))
            with open("../data/" + name + ".pickle", "wb+") as h:
                pickle.dump(out, h)
            with open("../data/" + name + "_" + "_time.txt", "w+") as l:
                l.write(str(time_taken))



print(results)