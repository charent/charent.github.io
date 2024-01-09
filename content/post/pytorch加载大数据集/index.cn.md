---
title: pytorch加载大数据集
description: 加载一个很大的数据集，全部加载到内存的话会导致内存爆炸（OOM），该怎么办？
date: 2023-11-05

categories:
    - 深度学习
    - pytorch
    - 数据集
---

# 问题背景
假设我们有一个很大的数据集（GB级别），需要加载到内存中，但是全部加载的话会导致内存爆炸。同时我们还要保证可以在训练的不同的epoch打乱数据，所以先打乱数据保存到文件，再一行一行（一个样本一个样本）加载进来行不通，而且一行一行加载会导致非常大的IO，速度还会变慢。

# 解决办法
1. 先把数据集转换为`parquet`文件，这中个格式的好处是可以分块读取，有效降低磁盘IO。
2. 在`torch`数据集类`dataset`中对`parquet`文件循环，一次加载`N`条数据到内存缓冲区中，对缓存中的N条数据执行打乱（shuffle）操作即可
3. 为了实现每个`epoch`都可以循环区数据集中的数据，可以使用`python`的`yield`特性实现迭代。

# 代码示例
```python
from typing import Union

from torch.utils.data import Dataset
from torch import LongTensor
from transformers import PreTrainedTokenizerFast
import pyarrow.parquet as pq
from numpy import array, int64
from numpy.random import shuffle

class MyDataset(Dataset):

    def __init__(self, 
                parquet_file: str,
                tokenizer_dir: str,
                keep_in_memory: bool=False,
                max_seq_len: int=512,
                buffer_size: int=40960,
            ) -> None:
        '''
        keep_in_memory: 是否将parquet文件转换为pandas.DataFrame格式存放到内存, 
            False将使用迭代生成器(迭代生成器不支持打乱数据)，减少大数据集内存占用
        '''
        super().__init__()

        self.keep_in_memory = keep_in_memory
        self.max_seq_len = max_seq_len

        # 使用pyarrow.parquet读取，to_pandas、for遍历速度更快
        parquet_table = pq.read_table(parquet_file)

        # 获取数据集长度
        self.length = parquet_table.num_rows

        # 缓冲区大小不能超过数据长度
        self.buffer_size = self.length if buffer_size > self.length else buffer_size

        if keep_in_memory:
            # 转化为pandas放到内存中
            self.data = parquet_table.to_pandas()  
        else:
            self.data = parquet_table

        # 初始化tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

        # 在这里初始化generator
        self.sample_generator = self.item_generator()
    
    def item_generator(self,) -> tuple:
        '''
        一条数据的生成器，防止大数据集OOM
        '''
                
        parquet_table = self.data

        # 生成器是死循环，不用退出，训练结束（epoch结束）会停止调用next()
        buffer_list = []
        while True:

            for prompt, response in zip(parquet_table['prompt'], parquet_table['response']):
                
                # 缓存数据不够，添加数据
                if len(buffer_list) < self.buffer_size:
                    buffer_list.append( (prompt.as_py(), response.as_py()) )
                    continue
                
                # 执行到这里，缓存区够了，打乱数据
                shuffle(buffer_list)
                for p, r in buffer_list:
                    # 在这里迭代
                    yield  p, r

                # 迭代完成，清空缓存区
                buffer_list = []
    
    def __getitem__(self, index):
        '''
        返回一条样本
        '''
        if self.keep_in_memory:
            data = self.data
            prompt, response = data.iloc[index].prompt, data.iloc[index].response
        else:
            prompt, response = next(self.sample_generator)

        max_seq_len = self.max_seq_len - 5 # len('[EOS]') = 5
        # add an eos token note that end of resopnse, using in generate.
        return f"{prompt[0: max_seq_len]}[EOS]", f"{response[0: max_seq_len]}[EOS]"

    def collate_fn(self, data: list[list]) -> dict:
        '''
        合并一个批次数据返回
        '''
        tokenizer = self.tokenizer

        prompt = tokenizer([item[0] for item in data], padding=True, return_token_type_ids=False)
        response = tokenizer([item[1] for item in data], padding=True, return_token_type_ids=False)

        input_ids = array(prompt.input_ids, dtype=int64)
        input_mask = array(prompt.attention_mask, dtype=int64)
        target_ids = array(response.input_ids, dtype=int64)

        ret = {
            'input_ids': LongTensor(input_ids),
            'input_mask': LongTensor(input_mask),
            'target_ids': LongTensor(target_ids),
        }
        return ret
    
    def __len__(self) -> int:
        return self.length

```


