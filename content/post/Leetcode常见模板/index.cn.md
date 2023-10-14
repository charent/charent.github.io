---
title: Leetcode常见模板
# description: 这是一个副标题
date: 2020-03-28

categories:
    - Leetcode
    - C++
    - Python
---

# C++ 字符串操作
```cpp
 //substrt的参数是从哪个位置开始，截取多少个字符，不是截取到哪个位置结束
 ans = s.substr(i, j - i + 1);
 //ans = s.substr(i, j); //错误

 //字符串的插入
string str1, str2="abc";
str1.push_back('c');

str2.push_back(str2);//错误
str2.append(str2);//正确

```
# C++输入处理
```cpp
 /*
输入：
    1    
  2 1 2  
3 4 2 1 3
保存到数组：
0 0 1 0 0 
0 2 1 2 0
3 4 2 1 3
*/

int cols = 2 * n - 1;
vector<vector<int>> nums(n, vector<int>(cols, 0));
string str;
//吸收换行
getchar();
int a;
stringstream ss; //将字符转换为int

for (int i = 0; i < n; ++i){
    getline(cin, str);
    ss.clear();
    ss<<str;
    int j =  cols / 2 - i;

    while (true){
        ss>>a;
        if (ss.fail()) break;
        nums[i][j] = a;
        ++j;
    }
}


stringstream ss;
string str;

int ans, tmp;

//输入行数一定，空格隔开
while ( getline(cin, str) )
{
    ss.clear();
    ss << str;
    ans = 0;
    while ( ss >> tmp )
    {
        ans += tmp;
    }
    cout << ans << endl;
}

//输入行数不一定
stringstream ss;
string str;
int ans, tmp;
while ( getline(cin, str) )
{
    ss.clear();
    ss << str;
    ans = 0;
    while ( ss >> tmp )
    {
        ans += tmp;
    }
    cout << ans << endl;
}

//数据用逗号隔开
vector<string> str;

while (getline(cin, line))
{
    istringstream ss(line);
    //stringstream ss(line); //也行
    str.clear();
    while ( getline(ss, tmp, ','))
    {
        //int tmp_int = atoi(tmp.c_str());
        //stringstream sss(tmp);
        //sss>> tmp_int;
        str.push_back(tmp);
    }
}
```
# const用法
```cpp
//const 修饰指针变量有以下三种情况。
//A: const 修饰指针指向的内容，则内容为不可变量。
//B: const 修饰指针，则指针为不可变量。
//C: const 修饰指针和指针指向的内容，则指针和指针指向的内容都为不可变量。

//C++ const
//指针常量： 指针本身是常量
TYPE* const pContent;

//向常量的指针：指针所指向的内容是常量
const TYPE *pContent;

//修饰引用： 常量引用，常用于形参类型，即避免了拷贝，
//又避免了函数对值的修改；表示函数内引用所指的内容不能改
void function4(const int& Var);

//const 修饰类成员函数，其目的是防止成员函数修改被调用对象的值，
Type func_name() const;


//则指针指向的内容 8 不可改变。简称左定值，因为 const 位于 * 号的左边。
const int *p = 8;

int a = 8;
int* const p = &a;
*p = 9; // 正确
int  b = 7;
p = &b; // 错误

//指针和内容都不可变
const int * const  p = &a;
```
# cpp排序
``` cpp
//降序
sort(vec.begin(), vec.end(), greater<int>());

//升序
sort(vec.begin(), vec.end(), less<int>());
```

# cpp二维数组初始化
```cpp
//二维vector初始化，二维数据初始化
vector<vector<int> > ary(row, vector<int>(col, 0));
```
## cpp map使用
```cpp
string intToRoman(int num) {
    string str = "";
    
    map<int, string> change_map{
        {1, "I"},
        {4, "IV"},
        {5, "V"},
    };
    map<int, string>::reverse_iterator iter;
    for (iter = change_map.rbegin(); iter != change_map.rend(); iter++)
    {
        while(num >= iter->first)
        {
            str.append(iter->second);
            num -= iter->first;
        }
    }
```
# cpp 堆操作
```cpp
//堆
#include <algorithm> 
/*
STL 堆操作
（1）make_heap()构造堆
void make_heap(first_pointer,end_pointer,compare_function);
默认比较函数是(<)，即最大堆。
函数的作用是将[begin,end)内的元素处理成堆的结构

（2）push_heap()添加元素到堆
void push_heap(first_pointer,end_pointer,compare_function);
新添加一个元素在末尾，然后重新调整堆序。该算法必须是在一个已经满足堆序的条件下。
先在vector的末尾添加元素，再调用push_heap

（3）pop_heap()从堆中移出元素
void pop_heap(first_pointer,end_pointer,compare_function);
把堆顶元素取出来，放到了数组或者是vector的末尾。
要取走，则可以使用底部容器（vector）提供的pop_back()函数。
先调用pop_heap再从vector中pop_back元素

（4）sort_heap()对整个堆排序
排序之后的元素就不再是一个合法的堆了。
*/

vector<int> min={10,30,22,6,15,9};
//建立小顶堆
make_heap(min.begin(), min.end(), greater<int>());//6 10 9 30 15 22

min.push_back(20);
//该算法前提：必须在堆的条件下
push_heap(min.begin(),min.end(), greater<int>());
//6 10 9 30 15 22 20   仍为小顶堆

//9 10 20 30 15 22 6  不为小顶堆 
//这个pop_heap操作后，实际上是把堆顶元素放到了末尾
pop_heap(min.begin(),min.end(), greater<int>());
min.pop_back();//这才彻底在底层vector数据容器中删除
//9 10 20 30 15 22  仍为小顶堆


//堆排序  保持greater，小顶堆，得到的是降序
sort_heap(min.begin(),min.end(), greater<int>());
//试了用less，结果杂乱无章
//30 22 20 15 10 9 注意结果是降序的
//其实是调用了很多次pop_heap(...,greater..)，
//每一次都把小顶堆堆顶的元素往末尾放，没放一次end迭代器减1
```

# cpp图的深度优先遍历
```cpp
/*
用vivited[]数组记录哪些节点被访问过
从0节点开始访问，依次访问未访问的相邻节点
*/
void dfs(int** G, int v)
{
    visited[v] = 1
    vistt(v)
    for (p in neighbour(G, v)))
    {
        if (!visited[p]) dfs(G, v);
    }
}
```

# cpp图的广度优先遍历
```cpp
/*
用vivited[]数组记录哪些节点被访问过
用队列que来存放每一层的顶点
*/
void bfs(int** G, v)
{
    que.push(v)
    while (!que.empty())
    {
        s = que.top();que.pop();
        for (s int neighbour(G, s))
        {
            if (!visited[s]) 
            {
                visited[s] = true;
               que.push()
            }
        }
    }
}
```
