# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch.nn as nn
import torch


class Block(nn.Module):
    """
    Block 类在这个上下文中代表卷积神经网络（CNN）架构中的一个卷积块。
    在深度学习中，特别是在CNN中，通常将网络架构组织成块或模块，以便更好地抽象化、重用和更容易管理复杂的架构。
    """
    def __init__(self, kernel_s, embeddin_num, max_len, hidden_num):
        super().__init__()
        # shape [batch *  in_channel * max_len * emb_num]
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embeddin_num))
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))

    def forward(self, batch_emb):  # shape [batch *  in_channel * max_len * emb_num]
        c = self.cnn(batch_emb)
        a = self.act(c)
        a = a.squeeze(dim=-1)
        m = self.mxp(a)
        m = m.squeeze(dim=-1)
        return m


class TextCNNModel(nn.Module):
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]

        # 每个实例代表一个不同的块，具有特定的内核大小 (2、3 和 4)，这在文本CNN中是典型的，用于捕获不同的 n-gram 特征。
        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)

        self.emb_matrix = emb_matrix

        self.classifier = nn.Linear(hidden_num * 3, class_num)  # 2 * 3
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_idx):   # shape torch.Size([batch_size, 1, max_len])
        batch_emb = self.emb_matrix(batch_idx)   # shape torch.Size([batch_size, 1, max_len, embedding])
        b1_result = self.block1(batch_emb)  # shape torch.Size([batch_size, 2])
        b2_result = self.block2(batch_emb)  # shape torch.Size([batch_size, 2])
        b3_result = self.block3(batch_emb)  # shape torch.Size([batch_size, 2])

        # 拼接
        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)  # shape torch.Size([batch_size, 6])
        pre = self.classifier(feature)  # shape torch.Size([batch_size, class_num])

        return pre

