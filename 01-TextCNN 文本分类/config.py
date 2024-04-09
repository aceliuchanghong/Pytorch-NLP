# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import argparse
import os.path


def parsers():
    """
    class_num（类别数）：表示分类任务中的类别数量。例如，如果你正在将文本分类到不同的类别中，class_num 将指示你有多少个类别。

    max_len（最大长度）：确定输入序列的最大长度。在自然语言处理任务中，通常会有固定长度的输入序列。max_len 指定了每个输入序列允许的最大长度。超过此长度的序列可能会被截断，而短于此长度的序列可能会被填充。

    embedding_num（嵌入维度）：指定词嵌入的维度。在自然语言处理任务中，单词通常被表示为嵌入空间中的密集向量。embedding_num 决定了这个嵌入空间的大小。例如，如果设置为 100，那么词汇表中的每个单词将被表示为一个 100 维的向量。

    batch_size（批量大小）：定义在更新模型参数之前要处理的样本数量。它通常用于训练神经网络。较大的批量大小可以加快训练速度，但需要更多的内存。

    epochs（轮数）：一个 epoch 是对整个训练数据集的一次完整遍历。epochs 指定了学习算法将对整个训练数据集进行多少次遍历。每个 epoch 包含对整个训练集的一次前向传播和一次反向传播。

    num_filters（卷积核数量）：确定卷积层输出的通道数（或滤波器数量）。在卷积神经网络中，卷积层通过应用一组卷积核来提取特征。num_filters 指定了卷积层输出的通道数，也可以理解为输出的特征图的数量。
    """
    parser = argparse.ArgumentParser(description="TextCNN model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("data", "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("data", "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("data", "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join("data", "class.txt"))
    parser.add_argument("--data_pkl", type=str, default=os.path.join("data", "dataset.pkl"))
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=38)
    parser.add_argument("--embedding_num", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--num_filters", type=int, default=2, help="卷积产生的通道数")
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args
