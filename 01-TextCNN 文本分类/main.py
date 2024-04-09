# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import read_data, built_corpus, TextDataset
from model import TextCNNModel
from config import parsers
import pickle as pkl
from sklearn.metrics import accuracy_score
import time
from test import test_data


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if os.path.exists(args.data_pkl):
        dataset = pkl.load(open(args.data_pkl, "rb"))
        word_2_index, words_embedding = dataset[0], dataset[1]
    else:
        word_2_index, words_embedding = built_corpus(train_text, args.embedding_num)

    train_dataset = TextDataset(train_text, train_label, word_2_index, args.max_len)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, args.max_len)
    dev_loader = DataLoader(dev_dataset, args.batch_size, shuffle=False)

    model = TextCNNModel(words_embedding, args.max_len, args.class_num, args.num_filters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = nn.CrossEntropyLoss()

    acc_max = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0, 0
        for batch_index, (batch_text, batch_label) in enumerate(train_loader):
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            pred = model(batch_text)

            loss = loss_fn(pred, batch_label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss
            count += 1

            # 打印内容
            if len(train_loader) - batch_index <= len(train_loader) % 1000 and count == len(train_loader) % 1000:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

            if batch_index % 1000 == 999:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

        # 是一个模型方法，用于将模型切换到评估模式。当调用时，模型会在推理（inference）阶段执行，而不是训练阶段。
        model.eval()
        # 评估模型在开发集（dev set）上的性能
        all_pred, all_true = [], []
        # with torch.no_grad() 这个语句块内的计算不会被PyTorch的自动求导（autograd）系统跟踪梯度，以节省内存和提高速度。
        with torch.no_grad():
            for batch_text, batch_label in dev_loader:
                batch_text = batch_text.to(device)
                batch_label = batch_label.to(device)
                pred = model(batch_text)

                # 对预测结果进行argmax操作，得到模型预测的类别
                pred = torch.argmax(pred, dim=1)
                # 将预测结果和实际标签从GPU移到CPU，并将它们转换为NumPy数组，然后转换为Python列表。
                pred = pred.cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()

                all_pred.extend(pred)
                all_true.extend(label)

        # 计算总体准确率，使用了一个叫做accuracy_score的函数，自于scikit-learn，用于计算分类任务的准确率。
        acc = accuracy_score(all_pred, all_true)
        print(f"dev acc:{acc:.4f}")

        if acc > acc_max:
            acc_max = acc
            torch.save(model.state_dict(), args.save_model_best)
            print(f"以保存最佳模型")
        # 打印分隔线
        print("*"*50)

    torch.save(model.state_dict(), args.save_model_last)

    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
    test_data()
