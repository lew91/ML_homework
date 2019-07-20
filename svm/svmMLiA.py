import random
from numpy import *
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


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    简化版SMO算法

    Parameters:
    --------------------------
    dataMatIn: 数据矩阵
    classLabels: 数据标签
    C: 松弛变量
    toler: 容错率
    maxIter: 最大迭代次数

    Returns:
    ----------------------
    b:
    alphas:
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter_num = 0

    while (iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # 计算误差Ei
            fxi = float(multiply(alphas, labelMat).T *  \
                        (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fxi - float(labelMat[i])
            # 优化alpha, 设定容错率
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
               ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个alpha_i 成对比优化的alpha_j
                j = selectJrand(i, m)
                # 计算误差Ej
                fxj = float(multiply(alphas, labelMat).T * \
                            (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 计算上下界
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if(L == H):
                    print("L == H")
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("J not moving enough")
                    continue

                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * \
                    dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[i, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * \
                    dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T

                if(0 < alphas[i] < C):
                    b = b1
                elif(0 < alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1

                print("iter: %d i:%d, pairs changed %d" % (iter_num, i, alphaPairsChanged))

        if(alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("iteration number: %d" % iter_num)
    return b, alphas


def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = array(alphas), array(dataMat), array(labelMat)

    # np.tile(labelMat.reshape(1, -1).T, (1, 2))将labelMat扩展为两列(将第1列复制得到第2列)
    # w = sum(alpha_i * yi * xi)
    w = dot((tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


def showClassifier(dataMat, w, b):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])

    data_plus_np = array(data_plus)
    data_minus_np = array(data_minus)

    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s=30, alpha=0.7)

    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]

    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * a2) / a2
    plt.plot([x1, x2], [y1, y2])

    for i, alpha in enumerate(alphas):
        if(abs(alpha) > 0):
           x, y = dataMat[i]
           plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifier(dataMat, w, b)




