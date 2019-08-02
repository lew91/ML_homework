import numpy as np
import pandas as pd
from math import log
import operator


def create_data():
    datasets = [['青年','否','否','一般','否'],
                ['青年','否','否','好','否'],
                ['青年','是','否','好','是'],
                ['青年','是','是','一般','是'],
                ['青年','否','否','一般','否'],
                ['中年','否','否','一般','否'],
                ['中年','否','否','好','否'],
                ['中年','是','是','好','是'],
                ['中年','否','是','非常好','是'],
                ['中年','否','是','非常好','是'],
                ['中年','否','是','非常好','是'],
                ['中年','否','是','好','是'],
                ['中年','是','否','好','是'],
                ['中年','是','否','非常好','是'],
                ['中年','否','否','一般','否'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, labels


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)

    
class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    @staticmethod
    def calc_ent(X):
        N = len(X)
        label_cnt = {}
        for i in range(N):
            label = X[i][-1]
            if label not in label_cnt:
                label_cnt[label] = 0
            label_cnt[label] += 1
        ent = -sum([(p/N) * log(p/N, 2) for p in label_cnt.values()])
        return ent

    def cond_ent(self, X, axis=0):
        N = len(X)
        feature_sets = {}
        for i in range(N):
            feature = X[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(X[i])
        cond_ent = sum([(len(p)/N) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, X):
        cnt = len(X[0]) - 1
        ent = self.calc_ent(X)
        best_feature = []
        for c in range(cnt):
            c_info_gain = self.info_gain(ent, self.cond_ent(X, axis=c))
            best_feature.append((c, c_info_gain))
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def _split_dataset(self, X_, axis, val):
        retDataSet = []
        for featVec in X_:
            if featVec[axis] == val:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1 :])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def _major_ent(self, X):
        class_cnt = {}
        for vote in X:
            if vote not in class_cnt.keys():
                class_cnt[vote] = 0
            class_cnt[vote] += 1
        sortedClassCnt = sorted(
            class_cnt.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCnt[0][0]

    def _grow(self, X, y):
        classList = [vector[-1] for vector in X]
        if classList.count(classList[0]) == len(classList):
            return classList[0]

        if len(X[0]) == 1:
            return self._major_ent(X)

        max_feature, max_info_gain = self.info_gain_train(X)
        max_feature_name = y[max_feature]

        tree = {max_feature_name: {}}

        del y[max_feature]
        # TODO: grow the tree
        valSet = set([vector[max_feature] for vector in X])

        for val in valSet:
            sublabels = y[:]
            subdata = self._split_dataset(X, max_feature, val)
            tree[max_feature_name][val] = self._grow(subdata, sublabels)

        return tree

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True,
                        label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    # def fit(self, X, y):
    #     self._tree = self._grow(X, y)
    #     return self._tree


    # def predict(self,X):
    #     pass

    def fit(self, X):
        self._tree = self.train(X)
        return self._tree

    def predict(self, X_):
        return self._tree.predict(X_)
