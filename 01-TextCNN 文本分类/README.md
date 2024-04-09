模型都未进行调参，未能使模型的准确率达到最高
# 项目名称：
使用 TextCNN 模型来对中文进行分类，即文本分类

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
TextCNN         
    |-- data                 数据集   
    |-- img                  相关图片            
    |-- model                保存的模型               
    |-- config.py            配置文件                                 
    |-- main.py              主函数                      
    |-- model.py             模型文件                     
    |-- predict.py           预测文件                         
    |-- requirement.txt      需要的安装包        
    |-- test.py              测试文件        
    |-- TextCNN.pdf          TextCNN 的论文        
    |-- utils.py             数据处理文件        
```

### TextCNN 模型结构图

![TextCNN 论文中模型](img/模型原型图.jpg)

![TextCNN 简化后模型](img/模型简化图.jpg)

输入一句话：I like this movie very much!，将其向量化，得到维度为5的矩阵，其 shape 为[1, 7, 5]。
```text
shape表示矩阵的维度大小。具体到[1, 7, 5]，它表示一个三维矩阵，其中：

第一个维度（1）表示样本的数量。在这个例子中，只有一个样本，因此为1。
第二个维度（7）表示句子的长度。"I like this movie very much!"有7个单词，所以是7。
第三个维度（5）表示词向量的维度。在这个例子中，每个单词被向量化为5维。

import numpy as np

# 词向量化
sentence = "I like this movie very much!"
words = sentence.split()
word_vectors = np.random.randn(len(words), 5)  # 假设词向量为5维的随机数

# 将向量化的句子转换为指定shape的矩阵
vectorized_sentence = np.expand_dims(word_vectors, axis=0)
vectorized_sentence = np.expand_dims(vectorized_sentence, axis=0)

print("Shape of vectorized sentence:", vectorized_sentence.shape)
print(vectorized_sentence)

```
将其送入模型，先经过3个卷积，卷积核大小分别为（2,5)，(3,5)，(4,5)。得到的 feature_map 的 shape 为 [1, 6]，[1, 5]，[1,4]。
```text
提到了三个不同大小的卷积核：(2, 5)，(3, 5)，(4, 5)。这表示卷积核的宽度是5，高度分别为2、3和4

卷积操作的结果是一个特征图（也称为feature map）。特征图的维度取决于卷积操作的参数设置以及输入数据的形状。在这个例子中，经过卷积操作后，得到的特征图的形状分别为[1, 6]，[1, 5]，[1, 4]。这表示每个卷积核产生的特征图的宽度会因为卷积核的不同而有所变化，而高度保持不变。
```
将得到的 feature_map 经过最大池化，得到 feature_map 的 shape 为 [1, 2]，[1, 2]，[1, 2]。
将池化后的 feature_map 进行拼接，得到的 shape 为 [1,6]，最后将其分为 2 分类。
**注：每个 shape 中的第一维度均为 batch_size。这里是以论文为主，所有为 1 ，实际不为 1。**

# 项目数据集
数据集使用THUCNews中的train.txt、test.txt、dev.txt，为十分类问题。
其中训练集一共有 180000 条，验证集一共有 10000 条，测试集一共有 10000 条。
其类别为 finance、realty、stocks、education、science、society、politics、sports、game、entertainment 这十个类别。


# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# 训练自己的数据集
train.txt、dev.txt、test.txt 的数据格式：文本\t标签（数字表示）

体验2D巅峰 倚天屠龙记十大创新概览\t8   
60年铁树开花形状似玉米芯(组图)\t5    

class.txt：标签类别（文本）

## 修改内容：
在配置文件中修改长度、类别数、预训练模型地址    
parser.add_argument("--class_num", type=int, default=10)   
parser.add_argument("--max_len", type=int, default=38)

# 博客地址
[CSDN TextCNN 文本分类](https://blog.csdn.net/qq_48764574/article/details/125757595)

[知乎 TextCNN 文本分类](https://zhuanlan.zhihu.com/p/641806406)

[微信公众号 TextCNN 文本分类](https://mp.weixin.qq.com/s?__biz=MzkxOTUzMDE0Nw==&mid=2247485013&idx=1&sn=082e076141f5af7a4e4e89cc9b3cff8a&chksm=c1a1f999f6d6708f346b7528b3c7666f5393ba3c0982c541620ca53bc1eada9caf423ca43ddc#rd)


