from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array


class MeanDiscrete(TransformerMixin):

    def fit(self, X, y=None):
        X = as_float_array(X)
        self.mean = X.mean(axis=0)
        return self

    def transform(self, X, y=None):
        X = as_float_array(X)
        assert X.shape[1] == self.mean.shape[0]
        return  X > self.mean

    
