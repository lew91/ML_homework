"""
Max entropy algorithm

$$H(P) = - \sum_{x} p(x) log p(x)$$, where the entropy  corresponded to
0<= H(P) <= log |X|
"""
import math
import copy

dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]


class MaxEntropy:
    def __init__(self,maxiter=1000, eps=0.005):
        self._samples = []
        self._Y = set()  # 去重后标签集合
        self._numXY = {}   # key为(x,y), value为出现次数
        self._N = 0    # 样本数
        self._Ep = []  # 样本分布的特征期望值
        self._xyID = {} # key为(x,y), value为index
        self._n = 0  # 特征值(x,y)的个数
        self._C = 0   # 最大特征数
        self._IDxy = {}
        self._w = []
        self._lastw = []
        self._maxiter=maxiter
        self._eps = eps

    def _Zx(self, X):
        """
        计算每个Z(x),规范化因子

        Z_w(x) =\sum_{y} exp(\sum^{n}_{i=1} w_i * f_i(x, y))
        """
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            zx += math.exp(ss)
        return zx

    def _model_pyx(self, y, X):
        """
        计算每个P(y|x)
        P_w(y|x) = exp(\sum^n_{i=1} w_i * f_i(x, y)) / z_w(x)
        """
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]
        pyx = math.exp(ss) / zx
        return pyx

    def _model_ep(self, index):
        """
        计算每个特征函数关于模型的期望 E(f_i) 
        """
        x, y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx / self._N
        return ep

    def _convergence(self):
        """
        判断收敛条件
        """
        for last, now in zip(self._lastw, self._w):
            if abs(last - now) >= self._eps:
                return False
        return True
        
    def fit(self, X):
        self._samples = copy.deepcopy(dataset)
        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._Y.add(y)
            for x in X:
                if (x, y) not in self._numXY:
                    self._numXY[(x, y)] = 1
                else:
                    self._numXY[(x, y)] += 1

        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample) -1 for sample in self._samples])
        self._w = [0] * self._n
        self._lastw = self._w[:]

        self._Ep = [0] * self._n
        for i, xy in enumerate(self._numXY):      # 计算特征值fi关于经验分布的期望
            self._Ep[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

        for _ in range(self._maxiter):
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)
                self._w[i] += math.log(self._Ep[i] /ep ) /self._C
            if self._convergence():
                break

    def predict(self, X):
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            pyx = math.exp(ss) / Z
            result[y] = pyx
        return result


if __name__ == '__main__':
    maxent = MaxEntropy()
    x = ['overcast', 'mild', 'high','False']
    maxent.fit(dataset)
    print('predict', maxent.predict(x))

