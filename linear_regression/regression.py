
from numpy import *


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1

    xArr = []
    yArr = []

    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))

    return xArr, yArr


def standRegres(xArr, yArr):
    """
    计算回归系数

    Parameters
    ---------------
    xArr : x数据集
    yArr : y数据集

    Returns
    -------------
    ws : 回归系数
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, connot do inverse")
        return

    ws = xTx.I * (xMat.T * yMat)
    return ws


def plotDataSet():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)

    xMat = mat(xArr)
    yMat = mat(yArr)

    xCopy = xMat.copy()
    xCopy.sort(0)

    yHat = xCopy * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=0.5)
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

    
