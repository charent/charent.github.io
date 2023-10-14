---
title: "Prime算法"
date: 2023-05-17 
categories:
    - Leetcode
---

# prime模板
```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)

        def get_distance(a: list[int], b: list[int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        int_max = 2 ** 31
        ans = 0

        graph = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dist = get_distance(points[i], points[j])
                graph[i][j] = dist
                graph[j][i] = dist
        
        low_cost = [int_max for _ in range(n)]
        vec = [0 for _ in range(n)]

        vec[0] = 1 # 从顶点0开始
     
        for i in range(1, n):
            low_cost[i] = graph[0][i] # 顶点0到i的距离
        
        for _ in range(1, n):
            # 找出利vec中为离所有1的点最近的点
            min_idx = -1 
            min_cost = int_max

            for j in range(n):
                # vec[j] = 0没有加入最小生成树合集的节点j
                if vec[j] == 0 and low_cost[j] < min_cost:
                    min_idx = j
                    min_cost = low_cost[j]
            
            vec[min_idx] = 1
            ans += min_cost
          

            # 更新vec中的所有lowcost
            for j in range(n):
                # 初始化时low_cost[j]为0到j的距离，加入新的节点min_idx后，最小生成树到未加入节点的距离可能会变短
                # vec[j] = 0没有加入最小生成树合集的节点j，更新j到新加入节点min_idx的最小距离
                if vec[j] == 0 and graph[min_idx][j] < low_cost[j]:
                    low_cost[j] = graph[min_idx][j]
            # print(vec, low_cost)
    
        return ans 


```