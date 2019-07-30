from math import log
import operator
import pickle

import treePlotter


def createDataSet():
    dataset = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataset, labels


def calcEntropy(dataSet):
    """
    计算数据集的经验熵
    Ent(D) = - SUM(kp *  log2 (kp))

    Parameters
    ----------
    dataset : 数据集

    Returns
    ----------
    shannonEnt : 经验熵
    """
    numEntries = len(dataSet)
    labelsCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelsCount.keys():
            labelsCount[currentLabel] = 0
        labelsCount[currentLabel] += 1
    shannonEnt = 0
    for key in labelsCount:
        prob = float(labelsCount[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, axis, val):
    """
    按照给定特征划分数据集

    Parameter:
    ----------
    dataSet : 待划分的数据集
    axis : 划分数据集的特征
    val : 需要返回的特征的值

    Returns
    ----------
    retDataSet : 划分的数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == val:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def getBestFeature(dataSet):
    """
    选择最优特征
    信息增益 gain(D, g) = Ent(D) - sum(|D_i|/ |D|) * Ent(D_i)

    Patameters
    --------
    dataSet: 数据集

    Returns
    ----------
    bestFeature : 信息增益最大的（最优）特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征存在featList这个列表中（列表生成式）
        featList = [vector[i] for vector in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0

        # 计算特征val对数据集dataSet的经验条件熵
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcEntropy(subDataSet)

        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    """
    统计classList中出现最多的元素（类标签）
    服务于递归第两个终止条件

    Parameters
    -----------
    classList : 类标签列表

    Returns
    -----------
    sortedClassCount[0][0] : 出现次数最多的元素（类标签）
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )
    return sortedclassCount[0][0]


def createTree(dataSet, labels):
    """
    创建决策树（ID3算法）

    Parameters
    -----------
    dataSet : 训练数据集
    labels : 分类属性标签

    Returns
    --------
    myTree : 决策树
    """
    classList = [vector[-1] for vector in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = getBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]  # 删除已经使用的特征标签
    valSet = set([vector[bestFeat] for vector in dataSet])

    for val in valSet:
        subLabels = labels[:]
        myTree[bestFeatLabel][val] = createTree(
            splitDataSet(dataSet, bestFeat, val), subLabels
        )

    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树分类

    Parameters
    ------------
    impurtTree : 已经生成的决策树
    featLabels : 存储选择的最优特征标签
    testVec : 测试数据列表，顺序对应最优特征标签

    Returns
    ------------
    classLabel : 分类结果
    """
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key, val in secondDict.items():
        if testVec[featIndex] == key:
            if isinstance(val, dict):
                classLabel = classify(val, featLabels, testVec)
            else:
                classLabel = val

    return classLabel


def storeTree(inputTree, filename):
    """
    存储决策树

    Parameters
    ------------
    imputTree : 已经生成的决策树
    filename : 决策树的存储文件

    Returns
    -----------
    None
    """
    with open(filename, "wb") as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    """
    读取决策树

    Parameter
    ---------
    filename : 决策树的存储文件名

    Returns
    -----------
    pickle.load(fr) : 决策树字典
    """
    fr = open(filename, "rb")
    return pickle.load(fr)


def readLense():
    attr_fname = "lenses.attr"
    data_fname = "lenses.data"

    attrDict = {}
    labels = []

    fin = open(attr_fname, "r")

    i = 0
    for line in fin.readlines():
        if len(line) == 0:
            continue
        wList = line.split()
        labels.append(wList[0])
        attrDict[i] = wList[1:]
        i += 1

    fin.close()

    dataSet = []
    fin = open(data_fname, "r")
    for line in fin.readlines():
        wList = line.split()
        vector = [attrDict[i][int(j) - 1] for i, j in enumerate(wList[1:])]
        dataSet.append(vector)
    fin.close()

    return dataSet, labels[:-1]


if __name__ == "__main__":
    dataSet, labels = readLense()
    myTree = createTree(dataSet, labels)
    print(myTree)
    treePlotter.createPlot(myTree)
