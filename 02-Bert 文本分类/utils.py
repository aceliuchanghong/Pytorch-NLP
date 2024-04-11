# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
from config import parsers
# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。
# 所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码。
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch


def read_data(file):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签、句子的最大长度
    texts, labels, max_length = [], [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            max_length.append(len(text))
            texts.append(text)
            labels.append(label)
    # 根据不同的数据集返回不同的内容
    if os.path.split(file)[1] == "train.txt":
        max_len = max(max_length)
        return texts, labels, max_len
    return texts, labels,


class MyDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.all_text = texts
        self.all_label = labels
        self.max_len = max_length
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)

    def __getitem__(self, index):
        # 获取数据集中索引为 index 的文本数据，并截断至最大长度 max_len。
        text = self.all_text[index][:self.max_len]
        label = self.all_label[index]

        # 分词
        text_id = self.tokenizer.tokenize(text)
        # 在分词结果的开头加上 [CLS] 标记，表示序列的开头，这是 BERT 模型中的特殊标记之一。
        text_id = ["[CLS]"] + text_id

        # 编码  将分词后的文本转换为对应的词汇表中的 ID。
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)

        """
        mask = [1] * len(token_id)：首先，将整个文本的长度设为1（真实标记），因为现在只考虑文本中的真实标记，
        即除了填充标记和特殊标记以外的部分。
        + [0] * (self.max_len + 2 - len(token_id))：然后，将填充部分的长度（即最大长度减去真实标记的长度）设为0（填充标记），
        这样做是因为填充部分不是真实的标记，应该被屏蔽掉。
        
        我们假设有以下数据：

        文本： "Hello, how are you?"
        最大长度： 10
        首先，我们使用BERT分词器对文本进行分词，得到标记列表：
        token_ids=[101,7592,1010,2129,2024,2017,1029,102]
        
        其中：
        
        101 是 [CLS] 标记的ID
        102 是 [SEP] 标记的ID
        接下来，我们根据最大长度和标记列表的长度来生成掩码列表：
        真实标记部分的掩码：
            mask(real)=[1,1,1,1,1,1,1,1]
        填充标记部分的掩码:
            mask(pad)=[0,0,0]
        将这两部分拼接在一起，得到最终的掩码列表：   
        mask = [1,1,1,1,1,1,1,1,0,0,0]
        这样，我们就得到了针对该文本的掩码列表，用于标识哪些位置是真实的标记，哪些位置是填充的标记。 
        
        
        掩码（Mask）在自然语言处理中扮演着重要的角色，特别是在序列模型中。在BERT模型中，掩码的作用主要有两个方面：

        填充屏蔽（Padding Masking）：在输入序列中，通常会对长度不一致的句子进行填充操作，使它们的长度相同。
        但是在计算中，我们不希望填充部分的内容对结果产生影响。因此，通过掩码将填充部分的影响屏蔽掉，使模型不会考虑这些填充标记。
        
        语言模型的预训练：在BERT的预训练阶段，掩码还可以用于掩盖部分输入，让模型学会对缺失部分进行预测。
        这种机制被称为掩码语言模型（Masked Language Model, MLM）。在训练过程中，一些输入的标记将被随机掩码（例如用 [MASK] 标记替换），
        而模型需要预测这些掩码的位置对应的原始标记。这样的训练方式可以使模型更好地理解上下文和语言的语义。
        """
        # 掩码  创建一个掩码列表，用于标识哪些位置是真实的标记，哪些位置是填充的标记。这里的 [CLS] 和填充标记 [PAD] 都被视为真实标记。
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        # 编码后  将编码后的标记列表填充到最大长度 max_len，并加上 [CLS] 和 [SEP]（在这里没有显示，但通常在 BERT 模型中，在句子的最后会加上一个 [SEP] 标记）。
        token_ids = token_id + [0] * (self.max_len + 2 - len(token_id))

        # 填充到最大长度并加上特殊标记 [CLS] 和 [SEP] (上一句这这儿2句只能选一个)
        token_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(token_ids))
        token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]

        # str -》 int
        label = int(label)

        # 转化成tensor
        token_ids = torch.tensor(token_ids)
        mask = torch.tensor(mask)
        label = torch.tensor(label)

        return (token_ids, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)


if __name__ == "__main__":
    train_text, train_label, max_len = read_data("./data/train.txt")
    print(train_text[0], train_label[0])
    trainDataset = MyDataset(train_text, train_label, max_len)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    for batch_text, batch_label in trainDataloader:
        print(batch_text, batch_label)
