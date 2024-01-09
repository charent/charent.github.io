---
title: 深度学习反向传播推导
# description: 
date: 2024-01-09
# slug: 
# image: 
categories:
    - 机器学习
    - 深度学习
    - 反向传播
---

# 反向传播推导
深度学习神经网络的前向传播大家都很清楚，对我来说，反向传播一直都是一知半解。今天重新复习一下。

我们以`sigmoid`函数作为一个计算图节点为例（加法乘法太简单了）。首先，`sigmoid函数`的定义如下：
$$
    y = \frac{1}{1 + e^{-x}} 
$$
对$y$求$x$的偏导数：
$$ \begin {aligned}
    \frac {\partial y}{ \partial x} &= \frac {-1}{(1 + e^{-x})^{2}} · (-e^{-x})  \\
    &= (\frac {1}{1 + e^{-x}}) ^2 · (e^{-x}) \\
    &= \frac {1}{1 + e^{-x}} · \frac {e^{-x}}{1 + e^{-x}} \\
    &= y · \frac {1 + e^{-x} - 1}{1 + e^{-x}} \\
    &= y · (1 - \frac {1}{1 + e^{-x}}) \\
    &= y · (1 - y) 
\end {aligned}
$$
假设`sigomid`节点前向传播的输出是$y$，反向传播到`sigomid`节点的数值是$L$，根据链式求导法则，则该节点对$x$的梯度为 $ \frac {\partial L}{ \partial y} ·  \frac {\partial y}{ \partial x} = \frac {\partial L}{ \partial y} · y · (1 - y) $ 

到这里可以写代码了：

```python
import math
class sigmoid:
    def __init__(self):
        self.y = None
    
    def forward(self, x: float) -> float:
        # sigmoid func
        y = 1 / (1 + math.exp( -x ))
        self.y = y

        return y
    
    def backward(self, out: float) -> float:
        dx = out * self.y * (1.0 - self.y)

        return dx

```
