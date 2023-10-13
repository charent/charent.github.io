---
title: CPP二叉树非递归遍历
# description: 这是一个副标题
date: 2020-10-27

categories:
    - C++
    - Leetcode
---
# 前序遍历
```cpp
//前序遍历
 /*
   a
  / \
  b  c
a b c
*/
vector<int> preorder(TreeNode* root)
{
    stack<TreeNode*> s;
    TreeNode* p = root;
    vector<int> ans;
    while (p || !s.empty())
    {
        //左子树走到底
        while (p)
        {
            //访问根节点
            ans.push_back(p->val);
            s.push(p);
            p= p->left;
        }
        //走完左子树，该走右子树了
        if (!s.empty())
        {
            p = s.top();
            s.pop();
            //遍历右子树
            p = p->right;
        }
    }
    return ans;
}

```

# 中序遍历
```cpp

//中序遍历
 /*
   a
  / \
  b  c
b a c
*/
vector<int> preorder(TreeNode* root)
{
    stack<TreeNode*> s;
    TreeNode* p = root;
    vector<int> ans;
    while (p || !s.empty())
    {
        //左子树走到底
        while (p)
        {
            s.push(p);
            p = p->left;
        }
        //走完左子树，访问
        if (!s.empty())
        {
            p = s.top();
            s.pop();
            //访问
            ans.push_back(p->val);
            
            //遍历右子树
            p = p->right;
        }
    }
    
    return ans;
}

```

# 后序遍历
```cpp 

//后序遍历
 /*
   a
  / \
  b  c
b c a

*/
vector<int> preorder(TreeNode* root)
{
    stack<TreeNode*> s;
    TreeNode* p = root;
    TreeNode* last_visit = nullptr;//最后一个访问的节点,初始化为null
    vector<int> ans;
    while (p || !s.empty())
    {
        if (p)
        {
            //p不为空，走到左子树的最末尾
            s.push(p);
            p = p->left;
        }
        else //左子树已经走到底
        {
            p = s.top(); //取栈顶元素，判断该怎么走
            if (p->right && p->right != last_visit)
            {
                //右子树存在，且不是上次访问的节点
                p = p->right;
            }
            else //右子树已经为空或者已经访问
            {
                //访问根节点
                ans.push_back(p->val);
                last_visit = p;
                s.pop(); //访问过的节点出栈
                p = nullptr;//设置p为空，让循环再次取出栈顶元素
            }
        }
    }
    
    return ans;
}

```