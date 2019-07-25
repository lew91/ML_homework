from numpy import *
from numpy import linalg as la

def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def imgLoadData(filename):
    myl = []

    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)

    myMat = mat(myl)
    return myMat


def analyse_data(Sigma, loopNum=20):
    sig2 = Sigma ** 2
    SigmaSum = sum(sig2)

    for i in range(loopNum):
        SigmaI = sum(sig2[: i+1])
        print('主成分：%s, 方差占比: %s%%' % (format(i+1, '2.0f'), format(SigmaI / SigmaSum * 100, '4.2f')))


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1)
            else:
                print(0)
        print('')


def imgCompress(numSV=3, thresh=0.8):
    myMat = imgLoadData('0_5.txt')
    print("---------original matrix-----------")
    printMat(myMat, thresh)

    U, Sigma, VT = la.svd(myMat)

    analyse_data(Sigma, 20)
    SigRecon = mat(eye(numSV) * Sigma[: numSV])
    recomMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("--------recomstructed matrix using %d singular values ------" % numSV)
    printMat(recomMat, thresh)
    

def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]

        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item],dataMat[overLap, j])

        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def svdEst(dataMat, user, simMeas, item):
    """
    基于SVD的评分估计

    Parameters:
    --------------
    dataMat : 训练数据集
    user : 用户编号
    simMeas : 相似度计算方法
    item : 未评分的物品编号

    Returns:
    ----------------
    评分（0~5之间的值）
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品（物品的4个主要特征）
    xformedItems = dataMat.T * U[:,:4] * Sig4.I

    for j in range(n):
        userRating = dataMat[user, j]

        if userRating == 0 or j == item:
            continue

        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal



def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return ('you rated everything')

    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))

    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]




