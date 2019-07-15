"""
Solving regression parameters with  gradient descent

Cost functions : $$J(w) = \frac{1}{2} \sum^n_{i=1} (y^{(i)} - \hat y^{(i)})^2$$
梯度：$$ \frac {\partial J}{\partial w_j}=-\sum^n_{i=1} (y^{(i)}-\hat y^{(i)})x_j^{(i)}$$
更新规则：$$w:=w-\eta\frac{\partial J}{\partial w} $$
"""


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta    # learning rate
        self.n_iter = n_iter    # 迭代次数

    def fit(self, X, y):
        self.coef_ = np.zeros(shape=(1, X.shape[1]))   # 代表被训练的系数，初始为0
        self.intercept_ = np.zeros(1)
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)    # 计算预测的y
            errors = y - output
            self.coef_ += self.eta * np.dot(errors.T, X)   # 更新系数
            self.intercept_ += self.eta * errors.sum()    # 更新bias， 相当于x取常数1
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.coef_.T) + self.intercept_

    def predict(self, X):
        return self.net_input(X)
