"""
Normal distribution test.

Q-Q plot:
For a standard normal distribution, the quantiles $q_{(j)}$ are defined by
the ralation
$$P[Z \leq q_{(j)}] = \int^{q_{(j)}}_{-\infty} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \text{d}z =
p_{(j)} = \frac{j- \frac{1}{2}}{n}$$
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


val = pd.DataFrame(np.random.randn(1000), columns=['values'])
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(3,1,1)
ax1.scatter(val.index, val.values)
plt.grid()

ax2 = fig.add_subplot(3,1,2)
val.hist(bins=30,alpha=0.5, ax=ax2)
val.plot(kind='kde', secondary_y=True, ax=ax2)
plt.grid()


###################
# 绘制Q-Q图
mean = val['values'].mean()
std = val['values'].std()

val.sort_values(by='values', inplace=True)
val_r = val.reset_index(drop=False)

val_r['p'] = (val_r.index - 0.5) / len(val_r)
val_r['q'] = (val_r['values'] - mean) / std

ax3 = fig.add_subplot(3, 1, 3)
ax3.scatter(val_r['q'], val_r['values'], alpha=.5)


plt.show()


#####################
# KS test ( Kolmogorov-Smirnov test)
from scipy import stats
scor = stats.kstest(val['values'], 'norm', (mean, std))
# scor = stats.kstest(val['values'], 'norm')

# scor = stats.normaltest(val['values'], axis=None)
print(scor)





