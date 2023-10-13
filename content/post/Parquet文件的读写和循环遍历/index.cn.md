---
title: Parquet文件的读写和循环遍历
description: 
date: 2023-08-21
# slug: markdown test
# image: helena-hertz-wWZzXlDpMog-unsplash.jpg
categories:
    - 大数据
---
### Parquet文件介绍
>Parquet 是 Hadoop 生态圈中主流的列式存储格式，最早是由 Twitter 和 Cloudera 合作开发，2015 年 5 月从 Apache 孵化器里毕业成为 Apache 顶级项目。 

优点：
1. 数据压缩比高，文件大小较小，适合网络传输;
2. 读写方便， python中`pandas`支持直接读写，`FastParquet`和`pyarrow`则提供更多的自定义操作。
3. I/O操作次数少，减少磁盘的使用率。
4. 内存占用少，适合处理大数据集。


### 读Parquet文件
1. `FastParquet`读及遍历
```python
from fastparquet import ParquetFile

pf = ParquetFile('./example.parquet')

# 大数据集to_pandas会占用大量的内存
# df = pf.to_pandas()

# 查看行数
print(pf.count())

for pf_chunk in pf:
    for rows in pf_chunk.iter_row_groups():
        for prompt, response in zip(rows['prompt'], rows['response']):
            pass
```
2. `pyarrow`读及遍历
```python
import pyarrow.parquet as pq

pt =  pq.read_table('./example.parquet')

# 大数据集to_pandas会占用大量的内存
# df = pt.to_pandas()

# 查看行数
print(pt.num_rows)

for prompt, response in zip(pt['prompt'], pt['response']):
    prompt, response = prompt.as_py(), response.as_py()
```

