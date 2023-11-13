# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 14:35
# @Author  : Yanjun Hao
# @Site    : 
# @File    : Logistic-Model-Trees.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 生成一个模拟的分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                           n_clusters_per_class=1, n_redundant=0,
                           random_state=42)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树模型
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# 在决策树的叶节点上拟合逻辑回归模型
logistic_models = {}
for leaf_node in np.unique(tree_model.apply(X_train)):
    # 找到落在叶节点的训练实例
    leaf_node_indexes = tree_model.apply(X_train) == leaf_node
    X_train_leaf = X_train[leaf_node_indexes]
    y_train_leaf = y_train[leaf_node_indexes]

    # 检查该叶节点是否有至少两个类别
    if len(np.unique(y_train_leaf)) > 1:
        # 拟合逻辑回归模型
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X_train_leaf, y_train_leaf)
        logistic_models[leaf_node] = lr
    else:
        # 只有一个类别，选择不在该叶子上拟合模型
        # 可以选择一个默认类别或者其他策略
        class_label = y_train_leaf[0]
        logistic_models[leaf_node] = class_label  # 存储默认类别


# 修改预测函数以处理只有一个类别的情况
def predict(X):
    leaf_nodes = tree_model.apply(X)
    preds = []
    for node in np.unique(leaf_nodes):
        if isinstance(logistic_models[node], LogisticRegression):
            # 使用逻辑回归模型预测
            node_preds = logistic_models[node].predict(X[leaf_nodes == node])
        else:
            # 使用默认类别
            node_preds = np.full(np.sum(leaf_nodes == node), logistic_models[node])
        preds.append(node_preds)
    return np.concatenate(preds)


# 使用自定义预测函数进行预测
y_pred = predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))

# 画出混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True)
plt.show()
