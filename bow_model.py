#coding: utf-8
#author: Ryan @ 解惑者学院
"""基于BOW文本特征的LR模型。

此部分代码仅为示例，还有很多工作可以做，建议自己尝试一下，包括不仅限于：

1. LogisticRegression有一个参数为C，尝试调整C的值，看看结果有什么变化，并思考为什么。

2. 创建字典时，设定的频次阈值为10，尝试设定其它的阈值，观察和思考结果及其影响。是否阈值越小越好，或者越大越好呢？这些都影响什么呢？

3. 打印出字典的内容时，你会发现有很多标点符号，这些对预测结果有影响么？如果有，试着分析影响到底多大；如果没有，那如果优化这部分呢？

4. 创建字典时，仅仅只使用了部分数据（拆分得到的训练集），为什么呢？可以使用所有的数据吗？

5. LogisticRegression有一个参数为penalty，其指定了是使用L1正则还是L2正则，试着改变一下参数，看看结果。并思考L1和L2的区别。

6. 文本转化为特征时，我们使用的是BOW，试着使用TF-IDF特征，观察实验结果是否有提升。尝试着分析结果改变的原因是什么。
"""
import sys
import os
import collections
import multiprocessing
import itertools
import functools
import operator
import array
import argparse

import numpy as np
import jieba
import jieba.posseg as posseg
import sklearn
import sklearn.linear_model as linear_model

def fetch_train_test(data_path, test_size=0.2):
    """读取数据，并拆分数据为训练集和测试集
    """
    y = list()
    text_list = list()
    for line in open(data_path, "r").xreadlines():
        label, text = line[:-1].split('\t', 1)
        text_list.append(list(jieba.cut(text)))
        y.append(int(label))
    return sklearn.model_selection.train_test_split(
                text_list, y, test_size=test_size, random_state=1028)


def build_dict(text_list, min_freq=5):
    """根据传入的文本列表，创建一个最小频次为min_freq的字典，并返回字典word -> wordid
    """
    freq_dict = collections.Counter(itertools.chain(*text_list))
    freq_list = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)#降序
    words, _ = zip(*filter(lambda wc: wc[1] >= min_freq, freq_list))
    return dict(zip(words, range(len(words))))


def text2vect(text_list, word2id):
    """将传入的文本转化为向量，返回向量大小为[n_samples, dict_size]
    """
    X = list()
    for text in text_list:
        vect = array.array('l', [0] * len(word2id))
        for word in text:
            if word not in word2id:
                continue
            vect[word2id[word]] = 1
        X.append(vect)
    return X


def evaluate(model, X, y):
    """评估数据集，并返回评估结果，包括：正确率、AUC值
    """
    accuracy = model.score(X, y)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    return accuracy, sklearn.metrics.auc(fpr, tpr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data_in/split/train.txt",
        help="Data file path",
    )
    return parser.parse_args()

if __name__ == "__main__":
    # for Unix-like system only
    # jieba.enable_parallel(multiprocessing.cpu_count())
    args = parse_args()

    # step 1. 将原始数据拆分成训练集和测试集
    X_train, X_test, y_train, y_test = fetch_train_test(args.data_path)

    # step 2. 创建字典
    word2id = build_dict(X_train, min_freq=10)

    # step 3. 抽取特征
    X_train = text2vect(X_train, word2id)
    X_test = text2vect(X_test, word2id)

    # step 4. 训练模型
    lr = linear_model.LogisticRegression(C=1)
    lr.fit(X_train, y_train)

    # step 5. 模型评估
    accuracy, auc = evaluate(lr, X_train, y_train)
    sys.stdout.write("训练集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("训练集AUC值：%.6f\n" % (auc))

    accuracy, auc = evaluate(lr, X_test, y_test)
    sys.stdout.write("测试集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("测试AUC值：%.6f\n" % (auc))


