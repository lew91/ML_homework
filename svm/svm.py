import random
from numpy import *
from os import listdir
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def loadImage(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()
    return returnVect


class optStruct:
    """
    维护所有需要操作的值

    Parameters
    --------------------
    dataMatIn: 数据矩阵
    classLabels : 数据标签
    C : 松弛变量
    toler : 容错率
    kTup : 包含核函数信息的元组，第一个参数存放核函数类别，
    第二个参数存放必要的核函数需要用到的参数

    Returns
    -----------------
    None
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 根据矩阵行数初始化误差缓存矩阵，第一列为是否有效标志位，第二列 为实际的误差E的值
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))

        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def kernelTrans(X, A, kTup):
    """
    通过核函数将数据转换更高维空间

    Parameters
    -----------------------
    X : 数据矩阵
    A : 单个数据向量
    kTup : 包含核函数信息的元组

    Returns
    ------------------------
    K : 计算的核K
    """
    m, n = shape(X)
    K = mat(zeros((m, 1)))

    # 线性核函数只计算内积
    if kTup[0] == 'lin':
        K = X * A.T

    # 高斯核函数
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError("核函数无法识别")

    return K 


def calcEk(oS, k):
    """
    计算误差

    Parameters
    --------------
    oS : 数据结构
    k : 标号为k的数据

    Returns
    --------------
    Ek : 标号为k的数据误差
    """
    # fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b

    # 应用核函数
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJrand(i, m):
    """
    随机选择 alpha_j

    Parameters
    --------------
    i : alpha_i 的索引值
    m : alpha参数个数

    Returns
    --------------
    j : alpha_j 的索引值
    """
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    """
    内循环启发式

    Parameters
    ------------
    i : 标号为i的数据的索引值
    oS : 数据结构
    Ei : 标号为i的数据误差

    Returns
    --------------
    j : 标号为j的数据的索引值
    maxK : 标号为maxK的数据索引值
    Ej : 标号为j的数据误差
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0

    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]

    if(len(validEcacheList) > 1):
        for k in validEcacheList:
            if k == i:
                continue

            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek

        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)

    return j, Ej


def updataEk(oS, k):
    """
    计算Ek, 并更新误差缓存

    Parameters
    -----------------
    oS : 数据结构
    k : 标号为k的数据的索引值

    Returns
    -----------------
    None
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
    """
    修剪 alpha_j

    Parameters
    -------------------
    aj :  alpha_j 值
    H : alpha上限
    L : alpha下限

    Returns
    --------------
    aj : alpha_j 值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):
    """
    优化的SMO算法

    Parameters
    ------------------
    i : 标号为i的数据的索引值
    oS : 数据结构

    Returns
    ----------------
    1 : 有任意一对 alpha 值发生变化
    0 : 没有任意一对 alpha 值发生变化或变化太小
    """
    Ei = calcEk(oS, i)

    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
      ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)

        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print("L == H")
            return 0

        # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
        #    oS.X[j, :] * oS.X[j, :].T

        # 应用核函数版
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        
        if eta >= 0:
            print("eta >= 0")
            return 0

        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updataEk(oS, j)

        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j  变化太小")
            return 0

        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * \
            (alphaJold - oS.alphas[j])
        updataEk(oS, i)

        # b1 = oS.b - Ei - \
        #     oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
        #     oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[i, :].T
        # b2 = oS.b - Ej - \
        #     oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
        #     oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T

        # 应用核函数版
        b1 = oS.b - Ei - \
            oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - \
            oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        if(0 < oS.alphas[i] < oS.C):
            oS.b = b1
        elif(0 < oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    完整的线性SMO算法

    Parameters
    ---------------
    dataMatIn : 数据结构
    classLabels : 数据标签
    C : 松弛变量
    toler : 容错率
    maxIter : 最大迭代次数
    kTup : 包含核函数信息的元组

    Returns
    --------------------
    oS.b : SMO算法计算的b
    oS.alphas : SMO算法计算的alphas
    """
    # oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)

    iter = 0
    entrieSet = True
    alphaPairsChanged = 0

    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entrieSet)):
        alphaPairsChanged = 0

        # 遍历整个数据集
        if entrieSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 遍历非边界
        else:
            # 遍历不在边界0和C的alpha
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 遍历一次后改为非边界遍历
        if entrieSet:
            entrieSet = False
        elif(alphaPairsChanged == 0):
            entrieSet = True
        print("迭代次数:%d" % iter)

    return oS.b, oS.alphas


def testRbf(k1=1.3):
    """
    利用核函数进行分类的经向基测试函数

    Parameters
    -----------------
    k1 : 使用高斯核函数时表示到达率

    Returns
    ---------------
    None
    """
    # 加载训练集
    dataArr, labelArr = loadDataSet('testSetRBF.txt')

    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()

    # 获得支持向量
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("支持向量个数: %d" % shape(sVs)[0])

    m, n = shape(dataMat)
    errorCount = 0

    for i in range(m):
        # 计算各个点的核
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        # 根据支持向量的点计算超平面， 返回预测结果
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1

    print("训练集错误率: %.2f%%" % ((float(errorCount) / m) * 100))

    # 加载测试集
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')

    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)

    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1

    print("测试集错误率: %.2f%%" % ((float(errorCount) / m) * 100))

    # 绘制数据集
    showDataSet(dataArr, labelArr, alphas)


def testDigits(kTup=('rbf', 10)):
    # 加载训练集
    dataArr, labelArr = loadImage('trainingDigits')

    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()

    # 获得支持向量
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("支持向量个数: %d" % shape(sVs)[0])

    m, n = shape(dataMat)
    errorCount = 0

    for i in range(m):
        # 计算各个点的核
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        # 根据支持向量的点计算超平面， 返回预测结果
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1

    print("训练集错误率: %.2f%%" % ((float(errorCount) / m) * 100))

    # 加载测试集
    dataArr, labelArr = loadImage('testDigits')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)

    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1

    print("测试集错误率: %.2f%%" % ((float(errorCount) / m) * 100))


def calcWs(alphas, dataArr, classLabels):
    """
    计算w

    Parameters
    ------------------------
    alphas : alphas值
    dataArr : 数据矩阵
    classLabels : 数据标签

    Returns
    ---------------------
    w : 直线法向量
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))

    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def showClassifier(dataArr, classLabels, w, b):
    """
    分类结果可视化

    Parameters
    ---------------------
    dataMat : 数据矩阵
    classLabels : 数据标签
    w : 直线法向量
    b : 直线截距

    Returns
    -------------------
    None
    """
    data_plus = []
    data_minus = []

    for i in range(len(dataArr)):
        if classLabels[i] > 0:
            data_plus.append(dataArr[i])
        else:
            data_minus.append(dataArr[i])

    data_plus_np = array(data_plus)
    data_minus_np = array(data_minus)

    # 绘制数据散点图
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s=30, alpha=0.7)

    # 绘制直线
    x1 = max(dataArr)[0]
    x2 = min(dataArr)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])

    # 标记支持向量点
    for i, alpha in enumerate(alphas):
        if (abs(alpha) > 0):
            x, y = dataArr[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7,
                        linewidth=1.5, edgecolors='red')

    plt.show()


def showDataSet(dataArr, classLabels, alphas):
    """
    数据可视化

    Parameters
    ---------------
    dataMat : 数据矩阵
    labelMat : 数据标签

    Returns
    ----------------
    None
    """
    data_plus = []
    data_minus = []

    for i in range(len(dataArr)):
        if classLabels[i] > 0:
            data_plus.append(dataArr[i])
        else:
            data_minus.append(dataArr[i])

    data_plus_np = array(data_plus)
    data_minus_np = array(data_minus)

    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s=30, alpha=0.7)

    for i, alpha in enumerate(alphas):
        if (abs(alpha) > 0):
            x, y = dataArr[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7,
                        linewidth=1.5, edgecolors='red')

    plt.show()


if __name__ == "__main__":
    # dataArr, classLabels = loadDataSet('testSet.txt')
    # b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
    # w = calcWs(alphas, dataArr, classLabels)
    # showClassifier(dataArr, classLabels, w, b)

    testRbf()
