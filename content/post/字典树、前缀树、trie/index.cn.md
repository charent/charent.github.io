---
title: "字典树、前缀树、trie"
date: 2022-11-02 
categories:
    - Leetcode
---
# 前缀树字典形式
```python
构建前缀树
for word in dictionary:
    cur = trie
    for ch in word:
        if ch not in cur:
            cur[ch] = {}
        cur = cur[ch]

    cur['#'] = {} # word的结束
    
# 查找
 for i, word in enumerate(sentence):
    cur = trie
    for j, ch in enumerate(word):
        if '#' in cur:
            find = True
            break
        if ch not in cur:
            break
        cur = cur[ch]

```

# 前缀树class形式
```cpp
class Trie {
    class TrieNode {
        boolean end;
        TrieNode[] tns = new TrieNode[26];
    }

    TrieNode root;
    public Trie() {
        root = new TrieNode();
    }

    public void insert(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) p.tns[u] = new TrieNode();
            p = p.tns[u]; 
        }
        p.end = true;
    }

    public boolean search(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) return false;
            p = p.tns[u]; 
        }
        return p.end;
    }

    public boolean startsWith(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) return false;
            p = p.tns[u]; 
        }
        return true;
    }
}
```
