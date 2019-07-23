import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)

    return dataMat


def distEclud(vecA, vecB):
    """
    计算欧式距离

    Parameters
    -------------
    vecA : 数据向量A
    vecB : 数据向量B

    Returns
    ----------
    两向量的欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    随机初始化k个质心（质心满足数据边界之内）

    Parameters
    ------------
    dataSet : 输入的数据集
    k : 选择k个质心

    Returns
    --------------
    centroids : 初始化得到的k个质心向量
    """
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))

    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)

    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    k-means聚类算法

    Parameters
    ----------
    dataSet : 用于聚类的数据集
    k : 选择k个质心
    distMeans : 距离计算方法
    createCent : 获取k个质心的方法

    Returns
    -----------
    centroids : k个聚类结果
    clusterAssment : 聚类误差
    """
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2))) # 一列存储簇索引值，第二列存储误差
    centroids = createCent(dataSet, k)

    clusterChanged = True

    while clusterChanged:
        clusterChanged = False

        for i in range(m):
            minDist = float('inf')
            minIndex = -1

            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            clusterAssment[i, :] = minIndex, minDist ** 2

    for cent in range(k):
        ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
        centroids[cent, :] = np.mean(ptsInClust, axis=0)

    return centroids, clusterAssment


def plotDataSet(filename):
    datMat = np.mat(loadDataSet(filename))

    myCentroids, clustAssing = kMeans(datMat, 4)
    clustAssing = clustAssing.tolist()
    myCentroids = myCentroids.tolist()
    xcord = [[], [], [], []]
    ycord = [[], [], [], []]
    datMat = datMat.tolist()
    m = len(clustAssing)
    for i in range(m):
        if int(clustAssing[i][0]) == 0:
            xcord[0].append(datMat[i][0])
            ycord[0].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 1:
            xcord[1].append(datMat[i][0])
            ycord[1].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 2:
            xcord[2].append(datMat[i][0])
            ycord[2].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 3:
            xcord[3].append(datMat[i][0])
            ycord[3].append(datMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=.5)
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='D', alpha=.5)
    ax.scatter(xcord[2], ycord[2], s=20, c='c', marker='>', alpha=.5)
    ax.scatter(xcord[3], ycord[3], s=20, c='k', marker='o', alpha=.5)

    ax.scatter(myCentroids[0][0], myCentroids[0][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[1][0], myCentroids[1][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[2][0], myCentroids[2][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[3][0], myCentroids[3][1], s=100, c='k', marker='+', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

