import numpy as np
import matplotlib.pyplot as plt
import random


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    梯度上升法

    Parameters
    --------------
    dataMatIn : 数据集
    classLabels : 数据标签

    Returns
    -------------
    weights.getA() : 权重数组（最优参数）
    weights_array : 每次更新的回归系数
    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.01   # 移动步长，学习速率
    maxCycles = 500
    weights = np.ones((n, 1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进随机的梯度上升法

    Parameters
    -----------------
    dataMatrix : 数据数组
    classLabels : 数据标签
    numIter : 迭代次数

    Returns
    -------------
    weights : 回归系数数组（最优参数）
    weights_array : 每次更新的回归系数
    """
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            weights_array = np.append(weights_array, weights, axis=0)
            del(dataIndex[randIndex])
    weights_array = weights_array.reshape(numIter * m, n)
    return weights, weights_array


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', alpha=0.5)
    ax.scatter(xcord2, ycord2, s=30, c='green', alpha=0.5)
    x = np.arange(-3.0, 3.0, 0.1)
    # w0 * x0 + w1 * x1 + w2 * x2 = 0
    # x0 = 1, x1 = x, x2 = y 
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plotWeight(weights_array1, weights_array2):
    fig, axes = plt.subplot(nrow=3, ncols=2, shares=False, sharey=False, figsize=(20,10))
    X1 = np.arange(0, len(weights_array1), 1)
    axes[0][0].plot(X1, weights_array1[:, 0])
    axes0_title_text = axes[0][0].set_title(u'改进的梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axes0_ylabel_text = axes[0][0].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axes0_title_text, size=20, weight='bold', color='black')
    plt.setp(axes0_ylabel_text, size=20, weight='bold', color='black')

    # 绘制w1与迭代次数的关系
    axes[1][0].plot(x1, weights_array1[:, 1])
    axes1_ylabel_text = axes[1][0].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axes1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axes[2][0].plot(x1, weights_array1[:, 2])
    axes2_title_text = axes[2][0].set_title(u'迭代次数', FontProperties=font)
    axes2_ylabel_text = axes[2][0].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axes2_title_text, size=20, weight='bold', color='black')
    plt.setp(axes2_ylabel_text, size=20, weight='bold', color='black')
    
    # x2坐标轴的范围
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axes[0][1].plot(x2, weights_array2[:, 0])
    axes0_title_text = axes[0][1].set_title(u'梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axes0_ylabel_text = axes[0][1].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axes0_title_text, size=20, weight='bold', color='black')
    plt.setp(axes0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axes[1][1].plot(x2, weights_array2[:, 1])
    axes1_ylabel_text = axes[1][1].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axes1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axes[2][1].plot(x2, weights_array2[:, 2])
    axes2_title_text = axes[2][1].set_title(u'迭代次数', FontProperties=font)
    axes2_ylabel_text = axes[2][1].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axes2_title_text, size=20, weight='bold', color='black')
    plt.setp(axes2_ylabel_text, size=20, weight='bold', color='black')
    
    plt.show()

    
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weight1, weight_array1 = gradAscent(dataMat, labelMat)
    weight2, weight_array2 = stocGradAscent1(np.array(dataMat), labelMat)

    plotWeight(weights_array1, weights_array2)
    
