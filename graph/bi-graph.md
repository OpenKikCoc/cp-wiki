## 定义

二分图，又称二部图，英文名叫 Bipartite graph。

二分图是什么？节点由两个集合组成，且两个集合内部没有边的图。

换言之，存在一种方案，将节点划分成满足以上性质的两个集合。

![](graph/images/bi-graph.svg)

## 性质

- 如果两个集合中的点分别染成黑色和白色，可以发现二分图中的每一条边都一定是连接一个黑色点和一个白色点。

-   > [!TIP] question **二分图不存在长度为奇数的环**
    > 
    > 因为每一条边都是从一个集合走到另一个集合，只有走偶数次才可能回到同一个集合。

## 判定

如何判定一个图是不是二分图呢？

换言之，我们需要知道是否可以将图中的顶点分成两个满足条件的集合。

显然，直接枚举答案集合的话实在是太慢了，我们需要更高效的方法。

考虑上文提到的性质，我们可以使用 [DFS（图论）](graph/dfs.md) 或者 [BFS](graph/bfs.md) 来遍历这张图。如果发现了奇环，那么就不是二分图，否则是。

> [!NOTE] **[AcWing 860. 染色法判定二分图](https://www.acwing.com/problem/content/862/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100010, M = 200010;

int n, m;
int h[N], e[M], ne[M], idx;
int color[N];

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

bool dfs(int u, int c) {
    color[u] = c;

    for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        if (!color[j]) {
            if (!dfs(j, 3 - c)) return false;
        } else if (color[j] == c)
            return false;
    }

    return true;
}

int main() {
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);

    while (m--) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a);
    }

    bool flag = true;
    for (int i = 1; i <= n; i++)
        if (!color[i]) {
            if (!dfs(i, 1)) {
                flag = false;
                break;
            }
        }

    if (flag)
        puts("Yes");
    else
        puts("No");

    return 0;
}
```

##### **Python**

```python
"""
1. 什么是二分图？
2. 可以把所有的点 划分为两个集合，集合A和B中的点互不相连。
3. 一个图是二分图，当且仅当图中不存在奇数环（一个环中的边数是奇数）
   ​	用反证法很好证明。（1-2-1-2-1）起点和终点是会连在一起的.

染色法：判别一个图是不是二分图（一个简单的DFS）
- O(n+m)
- 一条边的两个点一定是属于两个不同的集合
- 由于二分图中没有奇数环，所以染色过程一定没有矛盾（反证法可以证明）

"""


# Python用DFS容易爆栈，建议用BFS再做一遍
def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def dfs(u, c):
    color[u] = c
    i = h[u]
    while i != -1:
        j = ev[i]
        if color[j] == 0:
            if not dfs(j, (3 - c)):
                return False
        elif color[j] == c:
            return False
        i = ne[i]
    return True


if __name__ == '__main__':
    N = 100010
    M = 2 * N
    h = [-1] * N
    ev = [0] * M
    ne = [0] * M
    idx = 0
    color = [0] * N

    n, m = map(int, input().split())
    for _ in range(m):
        a, b = map(int, input().split())
        add_edge(a, b)
        add_edge(b, a)
    flag = True
    for i in range(1, n + 1):
        if color[i] == 0:
            if not dfs(i, 1):
                flag = False
                break
    if flag:
        print("Yes")
    else:
        print("No")


# BFS
def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def bfs(i, u):
    import queue
    q = queue.Queue()
    q.put([i, u])

    color[i] = u
    while not q.empty():
        u, c = q.get()
        i = h[u]
        while i != -1:
            j = ev[i]
            if not color[j]:
                color[j] = 3 - c
                q.put([j, 3 - c])
            elif color[j] == c:
                return False
            i = ne[i]
    return True


if __name__ == '__main__':
    N = 100010
    M = 2 * N
    h = [-1] * N
    ev = [0] * M
    ne = [0] * M
    idx = 0
    color = [0] * N

    n, m = map(int, input().split())
    for _ in range(m):
        a, b = map(int, input().split())
        add_edge(a, b)
        add_edge(b, a)

    flag = True
    for i in range(1, n + 1):
        if not color[i]:
            if not bfs(i, 1):
                flag = False
                break
    if flag:
        print("Yes")
    else:
        print("No")
```

<!-- tabs:end -->
</details>

<br>

* * *


## 应用

### 二分图最大匹配

详见 [二分图最大匹配](graph/graph-matching/bigraph-match.md) 页面。

### 二分图最大权匹配

详见 [二分图最大权匹配](graph/graph-matching/bigraph-weight-match.md) 页面。

### 一般图最大匹配

详见 [一般图最大匹配](graph/graph-matching/general-match.md) 页面。

### 一般图最大权匹配

详见 [一般图最大权匹配](graph/graph-matching/general-weight-match.md) 页面。

## 习题

> [!NOTE] **[Luogu [NOIP2010 提高组] 关押罪犯](https://www.luogu.com.cn/problem/P1525)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分 + 二分图

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 2e4 + 10, M = 2e5 + 10;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
int color[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

bool dfs(int u, int c, int m) {
    color[u] = c;
    for (int i = h[u]; ~i; i = ne[i]) {
        // ATTENTION
        if (w[i] <= m)
            continue;
        
        int j = e[i];
        if (color[j] == c)
            return false;
        if (!color[j] && !dfs(j, 3 - c, m))
            return false;
    }
    return true;
}

bool check(int m) {
    memset(color, 0, sizeof color);
    for (int i = 1; i <= n; ++ i )
        if (!color[i])
            if (!dfs(i, 1, m))
                return false;
    return true;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    
    while (m -- ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    
    int l = 0, r = 1e9;
    while (l < r) {
        int m = l + r >> 1;
        if (check(m))
            r = m;
        else
            l = m + 1;
    }
    cout << l << endl;
    
    return 0;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 染色法判二分 确定无环

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> g;
    vector<int> color;

    bool dfs(int u, int c) {
        color[u] = c;
        for (auto v : g[u])
            if (color[v] != -1) {
                if (color[v] == c)
                    return false;
            } else if (!dfs(v, c ^ 1))
                return false;
        return true;
    }

    bool isBipartite(vector<vector<int>>& graph) {
        g = graph;
        color = vector<int>(g.size(), -1);

        for (int i = 0; i < g.size(); ++ i )
            if (color[i] == -1)
                if (!dfs(i, 0))
                    return false;
        return true;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *