import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from scipy.stats import pearsonr


adult = pd.read_csv('data/adult.data', header=None,
                    names['Age', 'Work-class',
                          'fnlwgt', 'Education',
                          'Education-Num', 'Marital-Status',
                          'Occupation',
                          'Relationship', 'Race', 'Sex',
                          'Capital-gain', 'Capital-loss',
                          'Hours-per-week', 'Native-Country',
                          'Earnings-Raw'])

# If work hours long than 40
adult['LongHours'] = adult['Hours-per-week'] > 40

sns.swarmplot(x='Education-Num', y='Hours-per-week', hue='Earnings-Raw', data=adult[::50])
# plt.savefig('./data/education_hours_50.png', dpi=300)


X = adult[['Age', 'Education-Num', 'Capital-gain', 'Capital-loss', 'Hours-per-week']].values
y = (adult['Earnings-Raw'] == ' >50K').values

transformer = SelectKBest(score_func=chi2, k=3)
Xt_chi2 = transformer.fit_transform(X, y)
print(transformer.scores_)
print(transformet.get_support())


##################################
# Pearsonr

# warpper function, allows us to use this for multivariate arrays
# like the one we have.
def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        # compute the Pearson correlation for this column only
        cur_score, cur_p = pearsonr(X[:, column], y)
        # Record both the score and p-value
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))


transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
Xt_pearson = transformer.fit_transform(X, y)
print(transformer.scores_)

# cross validation
clf = DecisionTreeClassifier(random_state=14)
scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring='accuracy')
scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring='accuracy')
print("Chi2 score: {:.3f}".format(scores_chi2.mean()))    # 0.829
print("Pearson score: {:.3f}".format(scores_pearson.mean()))   # 0.771
