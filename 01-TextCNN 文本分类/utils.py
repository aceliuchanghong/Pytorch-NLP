import torch.nn as nn
from torch.utils.data import Dataset
import torch
import pickle as pkl
from config import parsers


def read_data(file):
    with open(file, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts, labels = [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            texts.append(text)
            labels.append(label)
    return texts, labels


def built_corpus(train_texts, embedding_num):
    """
    构建一个语料库（corpus），并将其保存为 pickle 文件。具体来说，函数接受两个参数：

    train_texts：一个包含训练文本的列表，每个文本是一个单词列表。
    embedding_num：嵌入向量的维度。
    函数首先创建一个空字典 word_2_index，其中包含两个特殊的键值对 <PAD> 和 <UNK>，分别代表填充标记和未知标记，并为它们分配索引 0 和 1

    在自然语言处理任务中，通常需要将文本数据转换为可以输入到神经网络模型中的数字表示。为了实现这一点，常见的做法是将文本中的单词映射到一个唯一的整数索引。这样做有几个原因：

    索引化： 将单词映射到整数索引可以方便地在模型中进行处理。神经网络模型通常只能处理数字输入，因此需要将文本转换为数字序列。

    嵌入层： 在神经网络模型中使用嵌入层时，需要将每个单词映射到一个唯一的整数索引，然后通过嵌入层获取对应的嵌入向量。

    填充和未知单词： 在处理文本序列时，通常会遇到长度不一致的情况。为了处理这种情况，常常会使用一个特殊的标记来表示填充（padding）以及一个特殊的标记来表示未知单词（unknown word）。这样可以保持所有文本序列的长度一致，并且能够处理模型未见过的单词。
    """
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    embedding = nn.Embedding(len(word_2_index), embedding_num)
    # 将项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来
    pkl.dump([word_2_index, embedding], open(parsers().data_pkl, "wb"))
    return word_2_index, embedding


class TextDataset(Dataset):
    def __init__(self, all_text, all_label, word_2_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_label[index])

        text_idx = [self.word_2_index.get(i, 1) for i in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))

        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)

        return text_idx, label

    def __len__(self):
        return len(self.all_text)
