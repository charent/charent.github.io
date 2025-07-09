---
title: Milvus Lite向量数据库多进程并发支持（只读）
date: 2025-07-09
# slug: markdown test
# image: 1.png
categories:
    - 向量检索
    - 多进程
---
# Milvus Lite向量数据库多进程并发支持

最近在用 Milvus Lite 数据库做数据表结构的检索和召回，要检索的文档数据量不大（千级别），本着能少部署一个服务就少部署一个服务的原则，就没有采用 Milvus Standalone 做部署。  
为了利用多进程提高并发量，在每个进程启动时用如下方法读取向量数据库文件：  
```python
from pymilvus import MilvusClient

db_file_path = '/path/to/milvus_demo.db'
client = MilvusClient(db_file_path)
```
但是启动时报错了：
```bash
Open /path/to/milvus_demo.db failed, the file has been opened by another program
2025-07-09 21:24:12,429 [ERROR][_create_connection]: Failed to create new connection using: milvus_demo.db (milvus_client.py:916)

....

ConnectionConfigException: <ConnectionConfigException: (code=1, message=Open local milvus failed)>
```

原来是不允多个进程读取同一个db个文件，但是我读取向量数据的需求是**只读**，**不涉及写向量数据库**，不存在锁的问题。这就好办了，利用linux的共享内存`/dev/shm`。进程启动时先把向量数据库db文件复制一份到共享内存，再读取共享内存那份数据就可以了。但是注意要做好清理工作，否则会造成内存泄漏（`/dev/shm`目录下一堆db文件）。

代码实现如下：
```python
import os
import shutil

from pymilvus import MilvusClient


class Demo:
    # 省略其他初始化代码
    def setup(slef):
        pid = os.getpid()

        origin_vector_db_path = "/path/to"
        vector_db_name = "milvus_demo.db"
        self.db_file_path = f"/dev/shm/pid-{pid}-{vector_db_name}"
        self.db_lock_file_path = f"/dev/shm/.pid-{pid}-{vector_db_name}.lock"

        logger.info(
            f"using temp vector db file: `{self.db_file_path}` for multiple process,"
            f" we will remove it when service is stop."
        )

        # 移动 db 文件到 /dev/
        shutil.copy(f"{origin_vector_db_path}/{vector_db_name}", self.db_file_path)

        # 多进程加载
        self.client = MilvusClient(self.db_file_path)

    def clean_up(self):
        """
        析构函数，在进程终止时删除共享目录下的 db file
        """
        try:
            self.client.close()
        except Exception as _:
            pass

        if os.path.exists(self.db_file_path):
            os.remove(self.db_file_path)
            logger.info(f"removed `{self.db_file_path}`")

        if os.path.exists(self.db_lock_file_path):
            os.remove(self.db_lock_file_path)
            logger.info(f"removed `{self.db_lock_file_path}`")

```

大功告成！