import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

house_data = pd.read_csv('data/housing.data',header=None, sep='\s+')
house_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                       'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                       'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = house_data[['RM']].values
y = house_data[['MEDV']].values

ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         residual_threshold=5.0,
                         random_state=0)

ranacs.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)

