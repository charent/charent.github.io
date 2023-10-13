---
title: Flask高并发处理
# description: 这是一个副标题
date: 2020-03-23

categories:
    - API高并发
---

# 通过设置app.run()的参数，来达到多线程的效果，具体参数：
```python
# 1.threaded : 多线程支持，默认为False，即不开启多线程;
app.run(threaded=True)
# 2.processes：进程数量，默认为1.
app.run(processes=True)
#ps：多进程或多线程只能选择一个，不能同时开启
```
# 使用genvent做协程，解决高并发：
```python
from genvent.wsgi import  WSGIServer
from genvent import monkey
 
monkey.patch_all()
app = Flask(__name__)
app.config.from_object(config)
api = Api(app)
 
db = DBInfo()
# db_old = DBInfo_old()

```
# 通过uvcorn(with genvent)的形式来对app进行包装，来启动服务：
``` shell
# 启动命令
gunicorn -c gun.py thread_explore:app
```
其中`gun.py`是`gunicorn`的配置文件
`thread_explore`是服务的主程序
app是flask的app
`gun.py`的具体内容：
```python
import os 
import gevent.monkey
gevent.monkey.patch_all()
import multiprocessing

# 服务地址（adderes:port） 
bind = 127.0.0.1;5000 
# 启动进程数量
workers = multiprocessing.cpu_count() * 2 +1
worker_class = 'gevent'
threads = 20
preload_app = True
reload = True
x_forwarded_for_header = 'X_FORWARDED-FOR'
```