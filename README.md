<center> <font size=6>Common Ditributions </font></center>
[TOC]

# 0. Introdution

## 0.1 distribution.md
[distribution.md](distributions.md)列出了常见的分布和函数的公式介绍以及绘图说明。目录如下：

![](f1.png)

其中Sigmoid函数和Hinge函数是在ML中常见的损失函数,relu是激活函数，可以通过调整相关的中心位置以及曲线坡度控制函数的形态

Poisson分布是离散事件的概率分布，描述单位时间内随机事件发生的次数的概率分布,事件的发生相互独立且同分布(二项分布)。

Normal分布、Laplace分布以及Student-T分布都有着类似的函数形态，因此将三者放在一栏中，其中laplace分布可以看作是一种对称的指数分布，Student-T的分布形态和正态分布类似，通过控制自由度的大小，可以控制其与标准正态分布拟合程度，自由度越高,越接近正态分布

Gamma分布描述的是n个事件共同发生的时间的问题，而指数分布(Exponential Distribution)描述的是随机事件发生一次的时间的分布。从这可以看出，指数分布和泊松分布有着很大的关联：$Pr(T_i > t+s|T_{i-1}=s)=Pr(N(t+s)-N(s)=0)=e^{-\lambda t}$泊松分布任意两次事件发生的时间间隔就是指数分布。

卡方分布描述的是n个独立同分布(iid)的标准正态分布的随机变量的平方和的分布,n在卡方分布中的自由度,n越高，方差越大（分布越分散），F分布是两个独立的卡方分布的比值

Beta分布是定义在(0,1)上的分布，通常被用作二项分布概率的分布的先验。

## 0.2 penalty function

[penalty_function.md](penalty_function.md)是满足两侧高中间低且波谷偏向一侧波峰的函数。

## 0.3 fake.py

这是生成虚假的交易数据的脚本，主要实现两个功能：
- generateNoise: 从原始交易中抹去部分交易，或者构造两个地址间不存在的交易 
- generateFake: 构造虚假的地址，并透过这些地址向重要的节点(supportAddress)存取以提高自己的重要性, 该函数主要是生成这些交易数据

另外还有一些函数可能需要解释:
- generateFakeAddress:生成虚假的地址
- readAddress：从文件中读取地址列表，可以将重要的地址保存在文件中，一行一个
- Transaction类：存储一次交易信息的类


# 1. Dependencies
enviroment:
> python matlab(octave)

python package:

>numpy  scipy   matplotlib

为了能完美显示，建议使用Mathjax解析数学公式部分

# 2. Others
推荐使用atom或者vs code浏览编辑，因为加入了一些latex公式，所以请使用拓展的markdown解释器（KaTex、TOC支持），推荐[Markdown-preview-enhance插件](https://github.com/shd101wyy/markdown-preview-enhanced),atom和vs code都可以下载使用
