模型都未进行调参，未能使模型的准确率达到最高

# 项目名称：
使用 Bert 模型来对进行中文关系识别

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
Bert
    |--bert-base-chinese             bert模型权重文件
    |--data                          数据
    |--img                           存放模型相关图片 
    |--log                           日志文件
    |--model                         保存的模型
    |--config.py                     配置文件
    |--main.py                       主函数
    |--model.py                      模型文件
    |--predict.py                    预测文件
    |--requirement.txt               安装库文件
    |--split_data.py                 数据划分
    |--utils.py                      数据处理文件
```

# 模型介绍
使用了Bert模型来判断文本中实体与实体的关系

关于Bert模型的介绍，可以看[02-Bert文本分类/README.md](../02-Bert%20文本分类/README.md)

# 项目数据集
[数据集](https://github.com/buppt//raw/master/data/people-relation/train.txt)

朱时茂	陈佩斯	合作	《水与火的缠绵》《低头不见抬头见》《天剑群侠》小品陈佩斯与朱时茂1984年《吃面条》合作者：陈佩斯聽1985年《拍电影》合
女	卢润森	unknown	卢恬儿是现任香港南华体育会主席卢润森的千金，身为南华会太子女的卢恬儿是名门之后，身家丰厚，她长相
傅家俊	丁俊晖	好友	改写23年历史2010年10月29日，傅家俊1-5输给丁俊晖，这是联盟杯历史上首次出现了中国德比，丁俊晖傅家俊携手改写了

# 数据划分
`python split_data.py`
可以单独执行，也可以不单独执行
在该.py文件中，可以修改训练集、验证集、测试集中数据中数据占用比例
```
train_lines = lines[:len(lines) * 6 // 10]
val_lines = lines[len(lines) * 6 // 10:len(lines) * 8 // 10]
test_lines = lines[len(lines) * 8 // 10:]
```

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# 博客地址

[CSDN Bert 关系识别](https://blog.csdn.net/qq_48764574/article/details/132689289)

[知乎 Bert 关系识别](https://zhuanlan.zhihu.com/p/654396722)
