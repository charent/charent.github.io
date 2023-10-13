---
title: 准确率、召回率、F1分数、灵敏度、特异度
# description: 
date: 2018-09-12
# slug: 
# image: 
categories:
    - 机器学习
---

#### 预测-真实值定义
""|预测值=1|预测值=0 
---|:--:|---:
真实值=1|True Positive(TP)|False Negative(FN)
真实值=0|Positive (FP)|True Negative(TN)

#### 真假阳性定义：
1. 真阳性`True Positive`，$TP$：样本的真实类别是正例，并且模型预测的结果也是正例
2. 真阴性`True Negative`，$TN$：样本的真实类别是负例，并且模型将其预测成为负例
3. 假阳性`False Positive`，$FP$：样本的真实类别是负例，但是模型将其预测成为正例
4. 假阴性`False Negative`，$FN$：样本的真实类别是正例，但是模型将其预测成为负例

#### 计算
1. 准确度：
   $$ Accuracy = \frac {TP+TN} {TP+TN+FN+TN} $$
2. 正确率:
   $$ Precision = \frac {TP}  {TP + FP)} $$ 
3. 真阳性率(True Positive Rate，TPR)，灵敏度(Sensitivity)，召回率: 
   $$ Recall = \frac {TP}  {TP + FN} $$
4. 真阴性率(True Negative Rate，TNR)，特异度:
   $$ Specificity = \frac {TN} {TN + FP} $$
5. 假阴性率(False Negatice Rate，FNR)，漏诊率( = 1 - 灵敏度) : 
   $$ \frac {FN} {TP + FN} = 1 - TPR $$
6. 假阳性率(False Positice Rate，FPR)，误诊率( = 1 - 特异度) ：
   $$ \frac {FP} {FP + TN} = 1 - TNR $$
7. F1分数：
    $$ F1_{score} = \frac {2 * TP} { 2 * TP + FP + FN} $$ 