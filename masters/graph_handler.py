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
features = ["api_calls", "app_permissions", "api_permissions", "interesting_calls", "urls", "intents", "activities"]
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
for feature in features:
    f = open("../data/drebin_split_" + feature + ".pickle", "rb")
    results = pickle.load(f)
    f.close()
    out = results['f1']
    # plt.plot([i for i in range(0, len(out))], out, label=feature, markeredgewidth=1, markersize=3, marker='o')
    # plt.tight_layout()
    # plt.rcParams["font.family"] = "serif"
    # plt.ylabel("F1")
    # plt.xlabel("Testing Period")
    # plt.legend(loc="lower left", fancybox=True, fontsize='small', title="Removed element")
    # plt.grid(axis="y", which="both")
    # plt.xticks([i for i in range(0, len(out))])
    # plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

    AUTs[feature] = np.trapz(out) / (len(out) - 1)
# plt.show()

# plt.figure(1)
# plt.clf()
plotdata = []
plt.xticks([])
plt.yticks([])
plotdata.append(["$\\bf{Drebin\\ feature\\ set}$", "$\\bf{AUT(F1, 24)}$"])
for a in AUTs:
    plotdata.append([a, AUTs[a]])
plt.tight_layout()
plt.axis("off")
plt.autoscale()
plt.table(cellText=plotdata, loc="center", bbox=None)
plt.show()