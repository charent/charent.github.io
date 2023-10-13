---
title: Linux常用命令
# description: 这是一个副标题
date: 2019-02-17

categories:
    - Linux
---
# 生成当前目录下所有文件的md5值
```bash
find ./ -type f -print0 | xargs -0 md5sum > ./md5.txt
```

