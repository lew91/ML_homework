
from numpy import *
import matplotlib.pyplot as plt


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


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


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


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 创建对角加权矩阵

    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / ( -2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)

    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)

    return yHat


def ridgeRegres(xMat, yMat, lam=0.2):
    """
    岭回归
    """
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam

    if linalg.det(denom) == 0 :
        print("This matrix is singular, cannot do inverse")
        return

    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, axis=0)
    yMat = yMat - yMean
    xMean = mean(xMat, axis=0)
    xVar = var(xMat, axis=0)
    xMat = (xMat - xMean) / xVar

    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))

    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T

    return wMat


def regularize(xMat, yMat):
    """
    数据标准化
    """
    xMean = mean(xMat, axis=0)
    yMean = mean(yMat, axis=0)

    xVar = var(xMat, axis=0)
    retxMat = (xMat - xMean) / xVar
    retyMat = yMat - yMean

    return retxMat, retyMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    前向逐步线性回归
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat, yMat = regularize(xMat, yMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        print(ws.T)
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)

                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest

        ws = wsMax.copy()
        returnMat[i, :] = ws.T

    return returnMat


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


def plot_lwlr():
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, k=0.003)

    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2,
               c='red')
    plt.show()


    
