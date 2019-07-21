from numpy import *


def loadSimpData():
    dataMat = matrix([[1., 2.1],
                      [1.5, 1.6],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def loadDataSet(filename):
    """
    自适应数据加载函数
    """
    numPeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []

    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numPeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    fr.close()

    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树分类器

    Parameters
    ------------
    dataMatrix : 数据矩阵
    dimen : 第dimen列，也就是第几个特征
    threshVal : 阈值
    threshIneq : 标志

    Returns
    -------------
    retArray : 分类结果
    """
    retArray = ones((shape(dataMatrix)[0], 1))

    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳单层决策树

    Parameters
    -------------
    dataArr : 数据矩阵
    classLabels : 数据标签
    D : 样本权重，每个样本额权重相等 1/n

    Returns
    ---------------
    bestStump : 最佳单层决策树信息
    minError : 最小误差
    bestClassEst : 最佳的分类结果
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T

    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = float('inf')

    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()

        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh inequal: %s,\
                # the weighted error is %.3f" % (i, threshVal, inequal,
                #                               weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    基于单层决策树的adabosst训练过程


    Parameters
    -------------
    dataArr : 数据矩阵
    classLabels : 数据标签
    numIt : 最大迭代次数

    Returns
    -----------
    weakClassArr : 存储单层决策树的list
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print("classEst", classEst.T)

        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()

        # aggClassEst += alpha * classEst
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggError.sum() / m
        # print("total error rate", errorRate)

        if errorRate == 0.0:
            break

    return weakClassArr


def adaClassify(dataToClass, classifierArr):
    """
    AdaBoost 分类函数

    Parameters
    ----------------
    dataToClass : 待分类样本
    classifierArr : 训练好的分类器

    Returns
    --------------
    分类结果
    """
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))

    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)

    return sign(aggClassEst)


