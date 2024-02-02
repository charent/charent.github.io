---
title: Mini hash文档去重
# description: 这是一个副标题
date: 2024-01-17
# slug: markdown test
# image: 汇总图.png
categories:
    - 数据处理
---

# 基于mini hash的文档去重

## 实现代码

```python

from datasketch import MinHash, MinHashLSH
from collections import defaultdict

# 结束标点符号
END_PUN = set(".。!！）)》}】?？\"”")

# 保留中文和英文、下划线，不要标点符号
NON_CHAR = re.compile("[^[\u4E00-\u9FA5|A-Za-z_0-9]")

def _get_doc_mini_hash(doc: list[str] | str, num_perm: int) -> MinHash:
    '''
    获取一段文本的mini hash
    '''
    mini_hash = MinHash(num_perm=num_perm)
    for s in doc:
        mini_hash.update(s.encode('utf-8'))
    return mini_hash

class DropDatasetDuplicate:

    def __init__(self,  threshold: float=0.85, num_perm: int=256) -> None:
        '''
        获取一个数据集中所有重复（相似的超过threshold）的index，输入为：list[str]，一个str元素为一段文本(doc)
        如输入： [a, b, c, d, c, d, e] 返回：{4, 5} (后面两个 c, d 的index)
        '''
        self.similar_index_cluster = defaultdict(set)
        self.data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) 
        self.num_perm = num_perm

    def add_doc(self, index: object, doc: str,) -> set[int]:
        '''
        添加文档，
        index： 文档的索引
        doc: 文档本身
        '''

        # 只保留中文和英文、下划线，不要标点符号
        doc = ''.join(NON_CHAR.split(doc))
        # doc = [''.join(t) for t in list(ngrams(doc, 3))]

        doc_hash = _get_doc_mini_hash(doc, self.num_perm)
        close_duplicates = self.data_lsh.query(doc_hash)

        self.data_lsh.insert(index, doc_hash)

        # 所有相似的doc在similar_index_cluster中的key都是最早出现的idx
        # 如：data中索引inndex 2, 7, 8, 9, 10, 12 是相似的，则在similar_index_cluster中表现为 {2: {8, 9, 10, 12}}
        if len(close_duplicates) > 0:
            min_idx= min(close_duplicates)
            self.similar_index_cluster[min_idx].add(index)
    
    def get_duplicate_indexs(self):
        '''
        返回所有的重复文档索引
        '''
        similar_index_cluster = self.similar_index_cluster
        need_to_remove_idx = set()
        
        for key_idx in similar_index_cluster.keys():
            need_to_remove_idx |= similar_index_cluster[key_idx]

        return need_to_remove_idx

```

## 使用方法
单进程速度非常慢，需要多进程处理（待补充）

```python

# 先顺序遍历获取哪些行是重复的
for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):
    row_index += 1

    doc = f"{prompt.as_py()}{response.as_py()}"
    drop_dataset_duplicate.add_doc(index=row_index, doc=doc)

row_index = -1
need_to_drop_indexs = drop_dataset_duplicate.get_duplicate_indexs()
```