---
title: Pytorch大数据集num_workers大于1内存缓慢增涨
description: pytorch加载大数据集（930万条）设置num_workers=8，内存缓缓增涨，最后OOM
date: 2023-10-09
# slug: markdown test
# image: helena-hertz-wWZzXlDpMog-unsplash.jpg
categories:
    - 深度学习
    - Pytorch
---
### 问题
在用`dataloadaer`加载数据集训练模型时，参数如下：
```python 
 train_dataloader = DataLoader(
        train_dataset,          # 大小为900多万
        batch_size=batch_size,  
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        num_workers=8,  
    )
```
在训练过程中CPU内存缓慢增涨，训练一个epoch还没结束，60G内存就OOM了。 

### 解决
搜索后发现，`num_workers >= 2`时，会把已经加载的batch数据缓存到CPU内存中，对于小数据集内存占用不明显，但是大数据集内存占用很可观。
设置`pin_memory=False`和`num_workers=0`后问题解决，但是num_workers=0会导致数据加载较慢，一个epoch会多消耗约30分钟，目前还没有特别好的解决方法。
