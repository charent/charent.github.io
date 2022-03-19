---
title: Markdown模板
description: 这是一个副标题
date: 2018-09-09
# slug: markdown test
image: helena-hertz-wWZzXlDpMog-unsplash.jpg
categories:
    - 未定义分类
tags:
    - 未定义tag
---

## 标题效果

# 这是一级标题
## 这是二级标题
### 这是三级标题
#### 这是四级标题
##### 这是五级标题
###### 这是六级标题

# 字体效果
**这是加粗的文字** 

*这是倾斜的文字*` 

***这是斜体加粗的文字*** 

~~这是加删除线的文字~~

# 段落控制
&emsp;&emsp;这是首行缩进  
换行：
```
<br/> 或者 空格 + 空格 + 回车
```


>这是引用的内容
>>这是引用的内容
>>>>>这是引用的内容

# 分割线
---
----
***
*****

# 超链接

[简书](http://jianshu.com) 

[百度](http://baidu.com)

# 列表
- 列表内容
+ 列表内容
* 列表内容 
   * 缩进列表，敲3个空格

# 表格
注意：- + * 跟内容之间都要有一个空格

表头|表头|表头
---|:--:|---:
内容|内容|内容
内容|内容|内容

第二行分割表头和内容。
- 有一个就行，为了对齐，多加了几个
文字默认居左
-两边加：表示文字居中
-右边加：表示文字居右
注：原生的语法两边都要用 | 包起来。此处省略


# LaTeX 公式
$ 表示行内公式(\$之间不能有空格)： 

质能守恒方程可以用一个很简洁的方程式 $E=mc^2$ 来表达。  

\$$ 表示整行公式：

$$\sum_{i=1}^n a_i=0$$

$$f(x_1,x_x,\ldots,x_n) = x_1^2 + x_2^2 + \cdots + x_n^2 $$

$$\sum^{j-1}_{k=0}{\widehat{\gamma}_{kj} z_k}$$

# 代码

`单行代码内容` 

``` javascript
    function fun(){
         echo "代码块";
    }
    fun();
```

流程图：
```flow
st=>start: 开始
op=>operation: My Operation
cond=>condition: Yes or No?
e=>end
st->op->cond
cond(yes)->e
cond(no)->op
```


## 图片
图片alt就是显示在图片下面的文字，相当于对图片内容的解释。
图片title是图片的标题，当鼠标移到图片上时显示的内容。title可加可不加

![Photo by Luca Bravo on Unsplash](luca-bravo-alS7ewQ41M8-unsplash.jpg) 

![Photo by Helena Hertz on Unsplash](helena-hertz-wWZzXlDpMog-unsplash.jpg)  ![Photo by Hudai Gayiran on Unsplash](hudai-gayiran-3Od_VKcDEAA-unsplash.jpg)

```markdown
![Photo by Florian Klauer on Unsplash](florian-klauer-nptLmg6jqDo-unsplash.jpg)  ![Photo by Luca Bravo on Unsplash](luca-bravo-alS7ewQ41M8-unsplash.jpg) 

![Photo by Helena Hertz on Unsplash](helena-hertz-wWZzXlDpMog-unsplash.jpg)  ![Photo by Hudai Gayiran on Unsplash](hudai-gayiran-3Od_VKcDEAA-unsplash.jpg)
```

相册语法来自 [Typlog](https://typlog.com/)