def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    # for trans in dataSet:
    #     fset = frozenset(trans)
    #     retDict.setdefault(fset, 0)
    #     retDict[fset] +=1

    for trans in dataSet:
        retDict[frozenset(trans)] = 1
        

    return retDict


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(' '* ind, self.name, ' ', self.count)

        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    """
    构建FP树

    Parameters
    ------------
    dataSet : 需要处理的数据集合
    minSup : 最少出现的次数（支持度）

    Returns
    -----------
    retTree : 树
    headerTable : 头指针表
    """
    headerTable = {}

    for trans in dataSet:
        for item in trans:
            # 由于dataSet里每个列表均为frozenset，所以每一个列表的值均为1，即dataSet[trans]=1
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    # for k in headerTable.keys():
    #     if headerTable[k] < minSup:
    #         del(headerTable[k])

    lessThanMinSup = list(filter(lambda k: headerTable[k] < minSup, headerTable.keys()))
    for k in lessThanMinSup:
        del(headerTable[k])
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k])
            
    freqItemSet = set(headerTable.keys())

    if len(freqItemSet) == 0:
        return None, None

    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('Null Set', 1, None)

    for tranSet, count in dataSet.items():
        localD = {}

        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]

        if len(localD) > 0:
            orderdItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            updateTree(orderdItems, retTree, headerTable, count)

    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)

    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)

        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]

        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    """
    寻找当前非空节点的前缀

    Parameters
    -----------
    leafNode : 当前选定的节点
    prefixPath : 当前节点的前缀

    Returns
    ---------
    None
    """
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)

        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    """
    返回条件模式基

    Parameters
    ----------
    basePat : 头指针列表中的元素
    treeNode : 树中的节点

    Returns
    ------------
    condPats : 返回条件模式基
    """
    condPats = {}

    while treeNode != None:
        prefixPath = []

        ascendTree(treeNode, prefixPath)

        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count

        treeNode = treeNode.nodeLink

    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    递归查找频繁项集

    Parameters
    ---------
    inTree :  初始创建的FP树
    headerTable : 头指针
    minSup : 最小支持度
    preFix : 前缀
    freqItemList : 条件树
    """
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]

    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPatBase = findPrefixPath(basePat, headerTable[basePat][1])
        myContTree, myHead = createTree(condPatBase, minSup)

        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            myContTree.disp(1)
            mineTree(myContTree, myHead, minSup, newFreqSet, freqItemList)

            
