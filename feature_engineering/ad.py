import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


data_filename = 'data/ad.data'


# Convert sting to float
def convert_number(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


converters = {}
for i in range(1558):
    converters[i] = convert_number

converters[1558] = lambda x: 1 if x.strip() == "ad." else 0

ads = pd.read_csv(data_filename, header=None, converters=converters)


# data processing
ads.dropna(inplace=True)
X = ads.drop(1558, axis=1).values
y = ads[1558]

# select featrues
pca = PCA(n_components=5)
Xd = pca.fit_transform(X)

np.set_printoptions(precision=3, suppress=True)
print(pca.explained_variance_ratio_)

# 验证
clf = DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')
print(scores_reduced)

# plot
classes = set(y)
colors = ['red', 'green']

sns.set_style("whitegrid")
for cur_class, color in zip(classes, colors):
    mask = (y == cur_class).values
    plt.scatter(Xd[mask, 0], Xd[mask, 1], marker='o',
                color=color, label=int(cur_class))
plt.legend()
plt.show()
