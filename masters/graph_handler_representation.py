import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tesseract import evaluation, temporal, metrics, viz
import pickle

line_kwargs = {'linewidth': 1, 'markersize': 5}


# def plot_f1(ax, l, data, alpha=1.0, neg=False, label=None, color='dodgerblue',
#             marker='o'):
#     if label is None:
#         label = 'F1 (gw)' if neg else 'F1 (mw)'
#     color = '#BCDEFE' if neg else color
#     # series = data['f1_n'] if neg else data['f1']
#     # print(data.index)
#     print(data[l])
#     ax.plot([i for i in range(0,len(data[l]))], data[l], label=label, alpha=alpha, marker=marker,
#             c=color, markeredgewidth=1, **line_kwargs)
#

# viz.set_style()
# features = ["api_calls", "app_permissions", "api_permissions", "interesting_calls", "urls", "intents", "activities"]
data = {}
# for feature in features:
#     f = open("../data/drebin_split_" + feature + ".pickle", "rb")
#     results = pickle.load(f)
#     f.close()
#     data[feature] = results['f1']

# fig, axes = plt.subplots(1, len(data))
# axes = axes if hasattr(axes, '__iter__') else (axes,)
#
# for l, ax in zip(data, axes):
#     print(l)
#     plot_f1(ax, l, data)
#
# viz.style_axes(axes, len(data["urls"]))
# fig.set_size_inches(4 * len(data), 4)
# plt.tight_layout()


#---
AUTs = {}
# plt.figure(0)
features = {}


for item in os.listdir("../data/"):
    print(item)
    if "representation" in item and ".pickle" in item:
        f = open("../data/" + item, "rb")
        results = pickle.load(f)
        f.close()
        details = item.split("_")
        out = results['f1']
        vec = details[4].split(".")[0]
        if details[0] not in features:
            features[details[0]] = {}
        features[details[0]][vec] = out
        if details[0] not in AUTs:
            AUTs[details[0]] = {}
        AUTs[details[0]][vec] = np.trapz(out) / (len(out) - 1)
        # plt.plot([i for i in range(0, len(out))], out, label=feature, markeredgewidth=1, markersize=3, marker='o')
        # plt.tight_layout()
        # plt.rcParams["font.family"] = "serif"
        # plt.ylabel("F1")
        # plt.xlabel("Testing Period")
        # plt.legend(loc="lower left", fancybox=True, fontsize='small', title="Removed element")
        # plt.grid(axis="y", which="both")
        # plt.xticks([i for i in range(0, len(out))])
        # plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

print(features)

# for classifier in features:
#     plt.clf()
#     for feature in features[classifier]:
#
#         out = features[classifier][feature]
#         print(out)
#         plt.plot([i for i in range(0, len(out))], out, label=feature, markeredgewidth=1, markersize=3, marker='o')
#         plt.tight_layout()
#         plt.rcParams["font.family"] = "serif"
#         plt.ylabel("F1")
#         plt.xlabel("Testing Period")
#         plt.legend(loc="upper right", fancybox=True, fontsize='small', title="Vectorizer")
#         plt.grid(axis="y", which="both", linewidth=1)
#         plt.xticks([i for i in range(0, len(out))])
#         plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#         plt.savefig(classifier + "_" + "representation")

print(AUTs)
plotdata1 = []
for classifier in AUTs:
    # plt.clf()


    plt.xticks([])
    plt.yticks([])
    plotdata1.append(["$\\bf{" + classifier + "}$", "$\\bf{AUT(F1, 24)}$"])
    avg1 = 0
    AUTs[classifier].sort()
    for a in AUTs[classifier]:
        plotdata1.append([a, AUTs[classifier][a]])
        avg1 += AUTs[classifier][a]

    plotdata1.append(["$\\bf{Average}$", str(avg1/3)])
    plt.tight_layout()
    plt.axis("off")
    plt.autoscale()
    plt.table(cellText=plotdata1, loc="best", bbox=None)
plt.savefig("_" + "representation" "_aut")













        # AUTs[feature] = np.trapz(out) / (len(out) - 1)
# plt.show()



# print(features)
# print(AUTs)

# plt.figure(1)
# plt.clf()
# plotdata1 = []
# plotdata2 = []
# manifest = ["app_permissions", "intents", "activities"]
# plt.xticks([])
# plt.yticks([])
# plotdata1.append(["$\\bf{Drebin\\ manifest\\ feature\\ set}$", "$\\bf{AUT(F1, 24)}$"])
# plotdata2.append(["$\\bf{Drebin\\ bytecode\\ feature\\ set}$", "$\\bf{AUT(F1, 24)}$"])
# avg1 = 0
# avg2 = 0
# for a in AUTs:
#     if a in manifest:
#         plotdata1.append([a, AUTs[a]])
#         avg1 += AUTs[a]
#     else:
#         plotdata2.append([a, AUTs[a]])
#         avg2 += AUTs[a]
#
# plotdata1.append(["$\\bf{Average}$", str(avg1/3)])
# plotdata2.append(["$\\bf{Average}$", str(avg1/4)])
# plt.tight_layout()
# plt.axis("off")
# plt.autoscale()
# plt.table(cellText=plotdata1, loc="best", bbox=None)
# plt.table(cellText=plotdata2, loc="center", bbox=None)
# plt.show()