import numpy as np


def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    stringArr =[line.strip().split(delim) for line in fr.readlines()]
    dataArr = [list(map(float, line)) for line in stringArr]
    return np.mat(dataArr)


def pca(dataMat, topNfeat=9999999):
    """
    PCA特征维度压缩函数

    Parameters
    -------------
    dataMat : 数据集数据
    topNfeat : 需要保留的特征维度

    Returns
    --------
    lowDDataMat : 压缩后的数据矩阵
    reconMat : 压缩后的数据矩阵反构出原始数据矩阵
    """
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    # 计算协方差矩阵，处以n-1是为了得到协方差的无偏估计
    # cov(x, 0) = cov(x)除数是n-1(n为样本个数)
    # cov(x, 1)除数是n
    covMat = np.cov(meanRemoved, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[: -(topNfeat+1): -1]

    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:, eigValInd]
    # 将去除均值后的矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects

    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    print(reconMat)

    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(datMat)[1]

    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal

    return datMat
