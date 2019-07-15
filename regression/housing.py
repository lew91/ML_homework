import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import LinearRegresssionGD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


house_data = pd.read_csv('data/housing.data',header=None, sep='\s+')
house_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                       'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                       'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# plot selected attribute to find information
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(house_data[cols], size=2.5)
plt.tight_layout()

plt.savefig('./data/scatter.png', dpi=300)


# plot correlation map
cm = np.corrcoef(house_data[cols].values.T)
sns.set(font_scale=1.5)

hm = sns.heatmap(cm,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.tight_layout()
plt.savefig('./data/corr_mat.png', dpi=300)


################################
X = house_data[['RM']].values
y = house_data[['MEDV']].values

# standardize
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)


lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# cost function
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.tight_layout()


# plot function
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return None


lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.tight_layout()
plt.show()

# 斜率， 截距
print('Slope: %.3f' % lr.coef_[0])
print('Intercept: %.3f' % lr.intercept_)

# 预测 RM=5 时，房价为多少
num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))


##############################################

# estimating coefficient of regression model via sklearn
slr = LinearRegression()
slr.fit(X_std, y_std)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

lin_regplot(X_std, y_std, slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.tight_layout()


