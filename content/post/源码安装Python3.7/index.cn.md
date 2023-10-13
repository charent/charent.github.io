---
title: 源码安装Python3.7
# description: 这是一个副标题
date: 2020-05-25

categories:
    - Python
---

1. python官网下载对应系统的安装源码包
2. 安装依赖
```bash
sudo apt-get install gcc
sudo apt-get install make
sudo apt-get install zlib*
sudo apt-get install libssl-dev
sudo apt-get install openssl
sudo apt-get install libffi-dev
sudo apt-get install sqlite3
sudo apt-get install libsqlite3-dev
sudo apt-get install libbz2-dev
```
3. 安装python37
```bash
wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz
#解压
tar -zxvf Python-3.7.7.tgz -C /usr/lib/python
cd /usr/lib/python/Python3.7.7
sudo gedit ./Modules/Setup
#编辑Setup这个文件
# Socket module helper for SSL support; you must comment out the other
# socket line above, and possibly edit the SSL variable:
#SSL=/usr/local/ssl
_ssl _ssl.c \
-DUSE_SSL -I$(SSL)/include -I$(SSL)/include/openssl \
-L$(SSL)/lib -lssl -lcrypto
#去掉上面四行前面的#号

#配置,加上--with-ssl，否则pip3用不了；
#加上--enable-optimizations性能大能提升10%,但是可能无法编译
sudo ./configure --with-ssl  --enable-loadable-sqlite-extensions
sudo make
sudo make install
```
