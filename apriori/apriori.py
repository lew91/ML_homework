def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    创建数据集中所有单一元素组成的集合

    Parameters
    ------------
    dataSet : 需要处理的数据集

    Returns
    -----------
    单一元素组成的集合
    """
    C1 = []

    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """
    从C1生成L1

    Parameters
    --------------
    D : 原始数据集
    Ck : 上一步生成的单元素数据集
    minSupport : 最小支持度

    Returns
    --------------
    retList : 符合条件的元素
    supportData : 符合条件的元素及其支持度组成的字典
    """
    ssCnt = {}
    
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}

    for key in ssCnt:
        support = ssCnt[key] / numItems

        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support

    return retList, supportData


def aprioriGen(Lk, k):
    """
    组合向上合并

    Parameters
    ----------
    Lk : 频繁项集列表
    k : 项集元素个数

    Returns
    ------------
    retList : 符合条件的元素
    """
    retList = []
    lenLk = len(Lk)

    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()

            if L1 == L2:
                retList.append(Lk[i] | Lk[j])

    return retList


def apriori(dataSet, minSupport=0.5):
    """
    apriori 算法

    Parameters
    --------------
    dataSet : 原始数据集
    minSupport : 最小支持度

    Returns
    ------------
    L : 符合条件的元素
    supportData : 符合条件的元素及其支持率组成的字典
    """
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))

    L1, supportData = scanD(D, C1, 0.5)

    L = [L1]  # 将符合条件的元素转换为列表保存在L中，L会包含L1,L2,L3, ...
    k = 2

    # L[n]代表n+1元素集合，例如L[0]代表1个元素的集合
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)

        supportData.update(supK)
        L.append(Lk)
        k += 1

    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    """
    生成关联规则

   Parameters
    ------------
    L : 频繁项集列表
    supportData : 包含那些频繁项集支持数据字典
    minConf : 最小可信度阈值

    Returns
    -----------
    bigRulesList : 生成的规则列表
    """
    bigRuleList = []

    for i in range(1, len(L)):   # only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    return bigRuleList


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    生成候选规则集合，计算规则的可信度以及找到满足最小可信度的规则

    Parameters
    --------------
    freqSet : L中的某一个（i）频繁项集
    H : L中的某一个（i）频繁项集元素组成的列表
    supportData : 包含频繁项集支持数据的字典
    brl : 关联规则
    minConf : 最小可信度

    Returns
    ------------
    None
    """
    m = len(H[0])

    # 频繁项集元素数码大于单个集合的元素数
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)

        # 满足最小可信度要求的规则类标多于1，则递归来判断是否可以进一步组合这些规则
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    生成候选规则集合，，计算规则的可信度以及找到满足最小可信度要求的规则

    Parameters
    -------------
    freqSet : L中的某一个（i）频繁项集
    H : L中的某一个（i）频繁项集元素组成的列表
    supportData : 包含频繁项集支持数据的字典
    brl : 关联规则
    minconf : 最小可信度

    Returns
    -------------
    prunedH : 返回满足最小可信度要求的项列表
    """
    prunedH = []

    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]

        if  conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf', conf)

            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)

    return prunedH

