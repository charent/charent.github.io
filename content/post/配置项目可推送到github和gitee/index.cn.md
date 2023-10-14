---
title: 配置项目可推送到github和gitee
# description: 这是一个副标题
date: 2021-01-03

categories:
    - Git
---
# 创建两个密钥
```bash
# Linux
cd ~/.ssh

# Windows
cd C:/Users/Dream/.ssh/

# 生成gitee密钥
# 创建gitee的ssh key文件名输入id_rsa.gitee

ssh-keygen -t rsa -C '你的邮箱@163.com'

# 生成github密钥
# 创建github的ssh key文件名输入id_rsa.github
ssh-keygen -t rsa -C '你的邮箱@163.com'
```
最后在.ssh目录下得到如下4个文件：
- id_rsa.gitee
- id_rsa.gitee.pub
- id_rsa.github
- id_rsa.github.pub

# 把公钥（id_rsa.*.pub）复制到Gitee、Github
这一步比较简单，去自己的git页面的设置找就可以了

# 配置密钥使用config
在·目录下创建config文件，添加以下内容到文件中
```conf
# gitee
Host gitee.com
HostName gitee.com
User 你的用户名
PreferredAuthentications publickey

# Linux：IdentityFile ~/.ssh/id_rsa.gitee
IdentityFile C:\Users\XXX\.ssh\id_rsa.gitee

# github
Host github.com
HostName github.com
User 你的用户名
PreferredAuthentications publickey

# Linux: IdentityFile ~/.ssh/id_rsa.github
IdentityFile C:\Users\XXX\.ssh\id_rsa.github
```
# 测试密钥是否配置成功
```bash
# gitee测试
ssh -T git@gitee.com
# 返回以下信息表示配置成功
Hi XXX! You've successfully authenticated, 
but GITEE.COM does not provide shell access.


# github测试
ssh -T git@github.com
# 返回以下信息表示配置成功
Hi XXX! You've successfully authenticated, 
but GitHub does not provide shell access.
```

# 配置一个项目可以推送两个仓库
在项目的`.git/config`文件修改`url = git@github.com:你的用户名/你的项目仓库.git`，如果你要推送到多个git仓库，比如要推送到`gitee`和`gitub`，则配置`[remote "github"]`和`[remote "gitee"]`
```conf
[core]
    repositoryformatversion = 0
    filemode = false
    bare = false
    logallrefupdates = true
    symlinks = false
    ignorecase = true
    
# 修改GitHub仓库地址为ssh推送
[remote "github"]
    url = git@github.com:你的用户名/你的项目仓库.git.git
    fetch = +refs/heads/*:refs/remotes/github/*
    
# 修改Gitee仓库地址为ssh推送
[remote "gitee"]
    url = git@gitee.com:你的用户名/你的项目仓库.git
    fetch = +refs/heads/*:refs/remotes/gitee/*
[branch "main"]
    remote = gitee
    merge = refs/heads/main

```




