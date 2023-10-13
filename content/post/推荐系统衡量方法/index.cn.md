---
title: 推荐系统衡量方法
# description: 
date: 2019-12-26
# slug: 
# image: 
categories:
    - 机器学习
    - 深度学习
    - 推荐系统
---

# AB Test：
A/B 测试是为Web或App界面或流程制作两个（A/B）或多个（A/B/n）版本，在同一时间维度，分别让组成成分相同（相似）的访客群组（目标人群）随机的访问这些版本，收集各群组的用户体验数据和业务数据，最后分析、评估出最好版本，正式采用。

# 点击通过率（Click-through Rate，CTR）: 
一般指网络广告的点击到达率，即该广告的实际点击次数除以广告的展现量，即clicks/views。反映了网页上某一内容的受关注程度，常常用来衡量广告的吸引程度

# ROC曲线
接受者操作特性曲线（receiver operating characteristic curve，简称ROC曲线）。曲线的横坐标为假阳性率（False Positive Rate, FPR）

```math
FPR=\frac {FP} {(FP+TN)}
```

N是真实负样本的个数，
FP是N个负样本中被分类器预测为正样本的个数。
纵坐标为真阳性率（True Positive Rate, TPR）

```math
TPR=\frac {TP}{P}=\frac {TP} {(TP+FN)}
```

其中，P是真实正样本的个数，TP是P个正样本中被分类器预测为正样本的个数。
# AUC （ROC曲线下方的面积大小）：
AUC（Area Under Curve）被定义为ROC曲线下与坐标轴围成的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。AUC越接近1.0，检测方法真实性越高;等于0.5时，则真实性最低，无应用价值。我们往往使用AUC值作为模型的评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好


