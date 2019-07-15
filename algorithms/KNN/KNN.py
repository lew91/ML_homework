#!/usr/bin/env python

from numpy import *
from os import listdir
import collections
import operator
import sys

def calEuclideanDist(inVector, dataSet):
    dataSetSize = dataSet.shpae[0]
    diffMat = tile(inVector, (dataSetSize, 1)) - dataSet
    sqDiiffMat = diffMat ** 2
    sqDist = sqDiiffMat.sum(exis=1)
    return sqDist ** 0.5

def calManhattanDist(inVector, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inVector, (dataSetSize, 1)) - dataSet
    absDiffMat = abs(diffMat)
    return absDiffMat

def calCosineMeasureDist(inVector, dataSet):
    sqDist = apply_along_axis(calCosineMeasure, len(dataSet.shape) -1, dataSet, inVector)
    return sqDist

def calCosineMeasure(x, y):
    ret = x.dot(y)
    return float(ret) / (sqrt(x.dot(x)) * sqrt(y.dot(y)))

def classify(testSet, dataSet, labels, k):
    if isinstance(testSet, (tuple, list)):
        for vector in testSet:
            classifyOneVec(vector, dataSet, labels, k)
    else:
            raise TypeError, 'invalid type of testSet in function [classify]'

def classifyOneVec(inVector, dataSet, labels, k, distFunc):
    distance = distFunc(inVector, dataSet)
    sortedDistIndexList = distance.argsort()
    classCount = collections.defaultdict(int)
    for i in sortedDistIndexList[:k]:
        classCount[labels[i]] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse =True)
    return sortedClassCount

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def file2matirx(fname):
    fin = open(fname, 'r')
    rawDataLines = fin.readline()
    fin.close()

    nLine = len(rawDataLines)
    rawData = rawDataLines[0]
    # calulate the num of attribute
    nAttr = len(rawData.split()) -1
    retMat = zeros((nLine, nAttr))
    retLabel = []
    for i, rawData in enumerate(rawDataLines):
        listFromData = rawData.strip().split()
        retMat[i,:] = listFromData[:-1]
        retLabel.append(int(listFromData[-1]))
    return retMat, retLabel

def KNNTest(dataSet, dataLabel, k, distFunc):
    testSetRatio =  0.1
    normMat, ranges, minVals = autoNorm(dataMat)
    m = dataMat.shape[0]
    nTestVec = int(m * testSetRatio)
    nError = 0
    for i in range(nTestVec):
        classifierResult = classifyOneVec(normMat[i, :], norMat[nTestVec:, :], dataLabel[nTestVec:], k, distFunc)
        if classiferResult != dataLabel[i] : nError +=1
        print("the classifier's result: %d, real result: %d"  % (classifierResult, dataLabel[i]))
    print ("\n the total number of errors: %d" % (nError))
    print ("\n the total error rate is: %d" % (float(nError)/float(nTestVec)))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = opem(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i*j] = int(lineStr[j])
    fr.close()
    return returnVect

def handwriingclassTest(k, distFunc):
    hwLabels = []
    train_fpath = './digits/trainingDigits'
    trainingFileList = listdir(train_fpath)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('%s/%s' % (train_fpath,fileNameStr))


    test_fpath = './digits/testDigits'
    testFileList = listdir(test_fpath)
    errorcount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('%s/%s' % (test_fpath,fileNameStr))
        classifierResult = classifyOneVec(
            vectorUnderTest[0], trainingMat, hwLabels, k, disFunc
        )
        print 'the classiferresult came back with: %d, the real answer is: %d' % (classifierResult, classNumStr)
        if classifierResult != classNumStr: errorcount +=1.0
    print '\nthe total number of errors: %d' % (errorcount)
    print '\nthe total error rate is: %f' % (errorcount/float(nTest))


distFuncDist = {
    'Euclidean': calEuclideanDist,
    'Manhattandist': calManhattanDist,
    'cosineMeasure': calCosineMeasureDist,
}

if __name__ == '__main__':
    fname = 'datingTestSet.txt'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    dataMat, dataLabel = file2matirx(fname)
    saveStdOud = sys.stdout
    k = 3
    for distFuncName, distFunc in distFuncDist.iteritems():
        print '\nuse %s...' % (distFuncName)
        fname = distFundName + '_result.log'
        fout = open(fname, 'w')
        sys.stdout = fout
        KNNTest(dataMat, dataLabel, k, distFunc)
        fout.close()
        sys.stdout = saveStdOud
