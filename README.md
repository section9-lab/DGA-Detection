软件版本
```
python v3.9
pytorch v1.12.0
```
```
├── date/
│   ├── alexa_top_10w.txt
│   └── dga.txt
└── models/
│   └── lstm.py
├── main.py
├── README.md

data/：数据集
models/：模型定义
main.py：主文件，训练和测试程序的入口
```
embedding层（嵌入层）：它把我们的稀疏矩阵，通过一些线性变换，变成了一个密集矩阵

lstm层是训练模型的核心，将从样本中学习特征

dropout层是为了训练的神经网络过拟合，随机断开一定比例的神经元连接

dense层是为了将学习到的特征映射到样本空间

activation激活层将权值转换成二分类结果
