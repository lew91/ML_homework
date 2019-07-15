#!/usr/bin/env python

from numpy import *
from os import listdir
import operator

def createDataSet():
    group = array(
        [[1.0,1.1],[1.0,1.0],[0.,0.],[0.,0.1]]
    )
    labels = ['A', 'A', 'B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDiffMat ** 0.5
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]      
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.iteritems(), key=operator.itemgetter(1), reverse = True
    )
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    classLabelVector = []
    classLabels = {}
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] not in classLabels:
            classLabels[listFromLine[-1]] = len(classLabels)
        classLabelVector.append(classLabels[listFromLine[-1]])
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def showMatrix(matrix):
    fig = plt.figure()
    ax =  fig.add_subplot(111)
    ax.scatter(
        matrix[:,1], matrix[:,2], 15.0 * array(datingLabels), 15.0 * array(datingLabels)
    )
    plt.show()

def classfiyPearson():
    resultList = ['not at all', 'in small doses','in large doses']
    percentTats = float(raw_input("precentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent filer meils earned per year?"))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array(ffMiles, percentTats, iceCream)
    classiferResult = classify0(
        (inArr-minVals)/ranges, norMat, datingLabels, 3
    )
    print ('you will probably like this person: ', resultList[classiferResult -1])

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 *i +j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
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
        trainingMat[i,:] = img2vector('%s/%s' % (train_fpath, fileNameStr))

    test_fpath = './digits/testDigits'
    testFileList = listdir(test_fpath)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('%s/%s' % (test_fpath, fileNameStr))
        classifierResult = classify0(
            vectorUnderTest, trainingMat, hwLabels, 3
        )
        print ('the classiferResult came back with: %d , the real answer is ; %d '\
            % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print ('\nthe total number of error: %d' % (errorCount))
    print ('\nthe total error rate is: %f' % (errorCount/float(mTest)))


if __name__ == "__main__":
    handwritingClassTest()
