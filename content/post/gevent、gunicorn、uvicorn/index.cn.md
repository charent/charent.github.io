---
title: gevent、gunicorn、uvicorn介绍
# description: 这是一个副标题
date: 2020-03-24

categories:
    - API高并发
---

# gevent
gevent是python的一个并发框架，以微线程greenlet为核心，使用了epoll事件监听机制以及诸多其他优化而变得高效。而且其中有个monkey类，将现有基于Python线程直接转化为greenlet(类似于打patch)
当一个greenlet遇到IO操作时，比如访问网络/睡眠等待，就自动切换到其他的greenlet，等到IO操作完成，再在适当的时候切换回来继续执行。由于IO操作非常耗时，经常使程序处于等待状态，有了gevent为我们自动切换协程，就保证总有greenlet在运行，而不是等待IO。同时也因为只有一个线程在执行，会极大的减少上下文切换的成本。

# gunicorn
Gunicorn是一个unix上被广泛使用的高性能的Python WSGI UNIX HTTP Server。和大多数的web框架兼容，并具有实现简单，轻量级，高性能等特点.

# uvicorn
uvicorn 是一个基于 asyncio 开发的一个轻量级高效的 web 服务器框架。仅支持 python 3.5.3 以上版本。