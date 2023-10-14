---
title: "并查集"
date: 2023-03-02 
categories:
    - Leetcode
---

# 并查集实现

```python
def find(parent: list, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node

def find(parent: list, node: int) -> int:
    if parent[node] != node:
         # find的过程中，将每个节点的父节点都设置为最后的节点
         parent[node] = find(parent, parent[node])
         
     return parent[node]

def union(parent: list, node1: int, node2: int) -> None:
    parent[find(parent, node1)] = find(parent, node2)
```

