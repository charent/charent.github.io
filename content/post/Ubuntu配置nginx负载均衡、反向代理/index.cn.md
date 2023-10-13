---
title: Ubuntu配置nginx负载均衡、反向代理
# description: 这是一个副标题
date: 2020-03-26

categories:
    - 负载均衡
    - API高并发
---
# 负载均衡、反向代理配置
反向代理指以代理服务器来接受Internet上的连接请求，然后将请求转发给内部网络上的服务器，并将从服务器上得到的结果返回给Internet上请求连接到客户端。
##  安装依赖
```bash
#安装依赖
sudo apt-get install libpcre3 libpcre3-dev
apt-get install zlib1g-dev
apt-get install openssl

#安装nginx
sudo apt-get install nginx

#启动服务
sudo /etc/init.d/nginx start

#编辑配置文件
cd /etc/nginx/sites-available/
vi default
```
负载均衡将前端的请求根据服务器的负载情况分发给不同的服务器。

## 配置文件示例
``` conf
#default 文件内容
#多服务器负载均衡配置
upstream nlp_api {
	#ip_hash每个请求按照访问ip的hash结果分配
	#这样每个访客固定访问一个后端服务器，
	#可以解决session的问题
	#ip_hash;

	#后面的权重是指定轮询几率
	#weight和访问比率成正比，用于后端服务器性能不均的情况
	#weight越大，负载的权重就越大,默认1
	server 127.0.0.1:8094 weight=5;
	server 127.0.0.1:8095 weight=5;
}

server {
	listen 80;
	server_name localhost;
 
	location / {
		#不允许跳转
		proxy_redirect off;
		
		#请求头设置
		proxy_set_header Host $host;
		proxy_set_header X-Real-IP $remote_addr;
		proxy_set_header REMOTE-HOST $remote_addr;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
		
		#从监听端口进来的请求将由upstream做分发
		proxy_pass http://nlp_api;
	}
}
```
## 重启服务
```bash
#检查配置文件是否有问题
sudo nginx -t

#最后重新加载配置
sudo /etc/init.d/nginx reload
#或者重启服务
sudo /etc/init.d/nginx restart
```

