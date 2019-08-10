"""
三硬币模型
假设有3枚硬币，分别记作A，B，C. 这些硬币正面出现的概率论分别是 \pi, p 和q。
进行如下抛硬币试验：先抛硬币A， 根据其结果选出硬币B或硬币C， 正面选硬币B，方面选硬币C；
然后抛选出的硬币，抛硬币的结果，出现正面记作1，出现反面记作0；
独立重复n次试验，观测结果如下： 1，1，0，1，0，0，1，0，0，1
假设只能观测到抛硬币的结果，不能饿观测抛硬币的过程。问如何估计三硬币正面出现的概率。
"""


import math
import numpy as np


class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob

    def _e_step(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, self._X[i]) * math.pow((1-self.pro_B), 1 -self._X[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, self._X[i]) * math.pow((1-self.pro_C), 1 - self._X[i])
        return pro_1 / (pro_1 + pro_2)

    def fit(self, X):
        self._X = X
        cnt =  len(self._X)

        # m step 
        for j in range(cnt):
            _ = yield
            _p = [self._e_step(i) for i in range(cnt)]
            pro_A = 1 / cnt * sum(_p)
            pro_B = sum([_p[k] * self._X[k] for k in range(cnt)]) / sum([_p[k] for k in range(cnt)])
            pro_C = sum([(1- _p[k]) * self._X[k] for k in range(cnt)]) / sum([(1- _p[k]) for k in range(cnt)])

            print('{}/{} pro_a:{:.3f}, pro_b:{:.3f}, prob_c:{:.3f}'.format(j+1, cnt, pro_A, pro_B, pro_C))
            
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C


if __name__ == '__main__':
    data = [1,1,0,1,0,0,1,0,1,1]
    em = EM(prob=[0.5, 0.5, 0.5])
    f = em.fit(data)
    next(f)
    print("------------------")

    f.send(1)
    f.send(2)


