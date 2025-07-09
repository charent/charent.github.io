# charent-chen 的个人博客项目

- 不定期更新，回复随缘
- 一些博客是自己以前的笔记，技术可能比较老了，也一起般过来吧

## 项目使用方法：
### 安装hugo：
- 从<https://github.com/gohugoio/hugo/releases>下载适合自己操纵系统的hugo版本，解压后将hugo可执行文件（如hugo.exe）所在的目录添加到环境变量中。
### 下载博客主题：
- 下载<https://github.com/CaiJimmy/hugo-theme-stack>主题并解压到themes目录下  
- 或者:
```bash
mkdir themes & cd themes
git clone --depth 1 https://github.com/CaiJimmy/hugo-theme-stack
```
### 撰写博客
- 在目录/content/post/新建文件夹名称为博客标题，参考/content/post/模板/index.cn.md文件，撰写自己的markdown博客。

## 本地运行调试
- 启动hugo服务：
```bash
# windows 
run.bat

# 或者
hugo server --theme=hugo-theme-stack --buildDrafts --bind=127.0.0.1 --baseURL=http://127.0.0.1 --port=8123

# linux
./run.sh
```

- 预览  
    在浏览器输入地址：<http://127.0.0.1:8123/>，预览并调整自己的博客内容

## 发布到githu.io
```bash

# windows
# 该脚本将生成静态文件到docs文件夹（需要在GitHub仓库中设置为docs）
build.bat

# 推送到github

git add .

git commit -m "your commit message"

git push

```