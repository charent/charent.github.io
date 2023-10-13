---
title: win10下Linux子系统导入导出
# description: 这是一个副标题
date: 2020-03-26

categories:
    - Linux
    - Windows
---

```shell
wsl.exe --list --running
#导出
wsl --export Ubuntu-18.04 D:\backup\ubuntu1804.tar
#导入
wsl.exe --import Ubuntu-18.04 D:\ubuntu\Ubuntu-18.04 D:\backup\ubuntu1804.tar
#运行
wsl --distribution Ubuntu-18.04
#删除
wsl.exe --unregister Ubuntu-18.04
```
