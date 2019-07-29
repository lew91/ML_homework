from math import log
from collections import defaultdict
import operator
import pickle


def calcEntropy(dataSet):
    numEntries = len(dataSet)
    labelsCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelsCount.keys():
            labelsCount[currentLabel] =0
        labelsCount[currentLabel] += 1
    shannonEnt = 1
    for key in labelsCount:
        prob = float(labelsCount[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt
        


def splitDataSet(dataSet, i, val):
    retDataSet = []
    for vector in dataSet:
        if vector[i] == val:
            retDataSet.append(vector[:i] + vector[i+1: ])
    return retDataSet


def getBestAttr(dataSet):
    mAttr = len(dataSet[0]) - 1
    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestAttr = -1

    for i in range(mAttr):
        valList = [vector[i] for vector in dataSet]
        valSet = set(valList)
        newEntropy = 0.0

        for val in valList:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcEntropy(subDataSet)

        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestAttr = i

    return bestAttr


def majorityCnt(classList):
    classDict = defaultdict(int)
    for item in classList:
        classDict[item] += 1
    sortedclassList = sorted(classDict.iteritems(), key=operator.itemgetter(1))
    return sortedclassList[-1][0]


def createTree(dataSet, labels):
    classList = [vector[-1] for vector in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestAttr = getBestAttr(dataSet)
    bestAttrLabel = labels[bestAttr]
    myTree = {bestAttrLabel:bestAttr}
    valSet = set([vector[bestAttr] for vector in dataSet])

    for val in valSet:
        subLabels = labels[:bestAttr] + labels[bestAttr+1: ]
        myTree[bestAttrLabel][val] = createTree(
            splitDataSet(dataSet, bestAttr, val), subLabels
        )

    return myTree


def classify(inputTree, attrLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    attrIndex = attrLabels.index(firstStr)

    for key, val in secondDict.items():
        if testVec[attrIndex] == key:
            if isinstance(val, dict):
                classLabel = classify(val, attrLabels, testVec)
            else:
                classLabel = val

    return classLabel


def storeTree(inputTree, filename):
    fw = open(filename, "w")
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename, 'r')
    return pickle.load(fr)


def readLense():
    attr_fname = 'lenses.attr'
    data_fname = 'lenses.data'

    attrDict = {}
    labels = []

    fin = open(attr_fname, 'r')

    i = 0
    for line in fin.readlines():
        if len(line) == 0:
            continue
        wList = line.split()
        labels.append(wList[0])
        attrDict[i] = wList[1: ]
        i += 1

    fin.close()

    dataSet = []
    fin = open(data_fname, 'r')
    for line in fin.readlines():
        wList = line.split()
        vector = [attrDict[i][int(j) -1] for i, j in enumerate(wList[1: ])]
        dataSet.append(vector)
    fin.close()

    return dataSet, labels[:-1]



