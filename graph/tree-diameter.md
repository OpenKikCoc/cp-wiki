> [!TIP] **区别**
>
> 两种方式求树直径：
>
> 1. dfs：在树上随机选一个点，对其进行一次 dfs 求出距离它最远的一个点 A , 然后再从 A dfs 找到一个离 A 点距离最远的 B 点， AB 之间的路径即为树的直径
>
>    **优点**：可以记录直径的起点 / 中点 / 完整路径
>
>    **缺点**：不可处理负权边
>
> 2. dp：标准树 dp ，枚举每一个点作为路径中层高最低的点即可
>
>    **优点**：可以处理负边权的树
>
>    **缺点**：只能求一个树的直径的长度，其他的求不出

图中所有最短路径的最大值即为「直径」，可以用两次 DFS 或者树形 DP 的方法在 O(n) 时间求出树的直径。

前置知识：[树基础](./tree-basic.md)。

## 例题

> [!NOTE] **[SPOJ PT07Z, Longest path in a tree](https://www.spoj.com/problems/PT07Z/)**
> 
> 给定一棵 $n$ 个节点的树，求其直径的长度。$1\leq n\leq 10^4$。

## 做法 1. 两次 DFS

首先对任意一个结点做 DFS 求出最远的结点，然后以这个结点为根结点再做 DFS 到达另一个最远结点。第一次 DFS 到达的结点可以证明一定是这个图的直径的一端，第二次 DFS 就会达到另一端。下面来证明这个定理。

但是在证明定义之前，先证明一个引理：

引理：在一个连通无向无环图中，$x$、$y$ 和 $z$ 是三个不同的结点。当 $x$ 到 $y$ 的最短路与 $y$ 到 $z$ 的最短路不重合时，$x$ 到 $z$ 的最短路就是这两条最短路的拼接。

证明：假设 $x$ 到 $z$ 有一条不经过 $y$ 的更短路 $\delta(x,z)$，则该路与 $\delta(x,y)$、$\delta(y,z)$ 形成一个环，与前提矛盾。

定理：在一个连通无向无环图中，以任意结点出发所能到达的最远结点，一定是该图直径的端点之一。

证明：假设这条直径是 $\delta(s,t)$。分两种情况：

- 当出发结点 $y$ 在 $\delta(s,t)$ 时，假设到达的最远结点 $z$ 不是 $s,t$ 中的任一个。这时将 $\delta(y,z)$ 与不与之重合的 $\delta(y,s)$ 拼接（也可以假设不与之重合的是直径的另一个方向），可以得到一条更长的直径，与前提矛盾。
-   当出发结点 $y$ 不在 $\delta(s,t)$ 上时，分两种情况：
    - 当 $y$ 到达的最远结点 $z$ 横穿 $\delta(s,t)$ 时，记与之相交的结点为 $x$。此时有 $\delta(y,z)=\delta(y,x)+\delta(x,z)$。而此时 $\delta(y,z)>\delta(y,t)$，故可得 $\delta(x,z)>\delta(x,t)$。由 1 的结论可知该假设不成立。
    - 当 $y$ 到达的最远结点 $z$ 与 $\delta(s,t)$ 不相交时，定义从 $y$ 开始到 $t$ 结束的简单路径上，第一个同时也存在于简单路径 $\delta(s,t)$ 上的结点为 $x$，最后一个存在简单路径 $\delta(y, z)$ 上的节点为 $x'$。如下图。

那么我们可以列出一些式子如下：

$$
\begin{array}{rcl}
\delta(y, z)&=&\delta(y, x') + \delta(x', z)\\
\delta(y, t)&=&\delta(y, x') + \delta(x', x) + \delta(x, t)\\
\delta(s, t)&=&\delta(s, x) + \delta(x, t)
\end{array}
$$

那么根据假设，有 $\delta(y, z) \ge \delta(y, t) \Longrightarrow \delta(x', x) + \delta(x, t) \ge \delta(x', z)$。既然这样子的话，那么 $\delta(x, z) \ge \delta(x, t)$，和 $\delta(s, t)$ 对应着直径这一前提不符，故 $y$ 的最远节点 $z$ 不可能在 $s$ 到 $t$ 这个直径对应的路外面。

![当 y 不在 s-t 上，且 z 也不在的情况](./images/tree-diameter.svg)

因此定理成立。

```cpp
const int N = 10000 + 10;

int n, c, d[N];
vector<int> E[N];

void dfs(int u, int fa) {
    for (int v : E[u]) {
        if (v == fa) continue;
        d[v] = d[u] + 1;
        if (d[v] > d[c]) c = v;
        dfs(v, u);
    }
}

int main() {
    scanf("%d", &n);
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        E[u].push_back(v), E[v].push_back(u);
    }
    dfs(1, 0);
    d[c] = 0, dfs(c, 0);
    printf("%d\n", d[c]);
    return 0;
}
```

## 做法 2. 树形 DP

我们记录当 $1$ 为树的根时，每个节点作为子树的根向下，所能延伸的最远距离 $d_1$，和次远距离 $d_2$，那么直径就是所有 $d_1 + d_2$ 的最大值。

```cpp
const int N = 10000 + 10;

int n, d = 0;
int d1[N], d2[N];
vector<int> E[N];

void dfs(int u, int fa) {
    d1[u] = d2[u] = 0;
    for (int v : E[u]) {
        if (v == fa) continue;
        dfs(v, u);
        int t = d1[v] + 1;
        if (t > d1[u])
            d2[u] = d1[u], d1[u] = t;
        else if (t > d2[u])
            d2[u] = t;
    }
    d = max(d, d1[u] + d2[u]);
}

int main() {
    scanf("%d", &n);
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        E[u].push_back(v), E[v].push_back(u);
    }
    dfs(1, 0);
    printf("%d\n", d);
    return 0;
}
```

## 习题

- [CodeChef, Diameter of Tree](https://www.codechef.com/problems/DTREE)
- [Educational Codeforces Round 35, Problem F, Tree Destruction](https://codeforces.com/contest/911/problem/F)
- [ZOJ 3820, Building Fire Stations](https://vjudge.net/problem/ZOJ-3820)
- [CEOI2019/CodeForces 1192B. Dynamic Diameter](https://codeforces.com/contest/1192/problem/B)
- [IPSC 2019 网络赛，Lightning Routing I](https://nanti.jisuanke.com/t/41398)

## 习题

> [!NOTE] **[Luogu [NOIP2007 提高组] 树网的核](https://www.luogu.com.cn/problem/P1099)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典树问题
> 
> 树直径原理 + 枚举优化 + 二分
> 
> 反复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PII = pair<int, int>;
const int N = 5e5 + 10, M = N << 1;

int n, s;
int h[N], e[M], w[M], ne[M], idx;
int q[N], dist[N], pre[N];
vector<PII> path;
bool st[N];

// ----------------- helper func -----------------

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int get_max() {
    int t = 1;
    for (int i = 1; i <= n; ++ i )
        if (dist[t] < dist[i])
            t = i;
    return t;
}

// ----------------- basic func -----------------

void bfs(int start) {
    memset(dist, 0x3f, sizeof dist);
    memset(pre, -1, sizeof pre);
    dist[start] = 0;
    int hh = 0, tt = -1;
    q[ ++ tt] = start;
    
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[t] + w[i]) {
                pre[j] = t;
                dist[j] = dist[t] + w[i];
                q[ ++ tt] = j;
            }
        }
    }
}

int bfs_max_dist(int start) {
    int res = 0;
    int hh = 0, tt = -1;
    q[ ++ tt] = start;
    while (hh <= tt) {
        int t = q[hh ++ ];
        res = max(res, dist[t]);
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (!st[j]) {
                st[j] = true;
                dist[j] = dist[t] + w[i];
                q[ ++ tt] = j;
            }
        }
    }
    return res;
}

bool check(int m) {
    // 3. 找分别与 u / v 距离不超过 mid 的且最远的节点
    //    分别作为 p / q
    int u = 0, v = path.size() - 1;
    while (u + 1 < path.size() && path[u + 1].second <= m)
        u ++ ;
    while (v - 1 >= 0 && path.back().second - path[v - 1].second <= m)
        v -- ;
    if (u > v)
        return true;
    // 4. p 和 q 之间距离不超过 mid
    if (path[v].second - path[u].second > s)
        return false;
    
    memset(st, 0, sizeof st);
    memset(dist, 0, sizeof dist);
    for (auto p : path)
        st[p.first] = true;
    
    // 5. p 和 q 之间所有点到其他所有点的距离不超过 mid
    for (int i = u; i <= v; ++ i )
        if (bfs_max_dist(path[i].first) > m)
            return false;
    return true;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> s;
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    
    // 1. 先找直径
    bfs(1);
    int u = get_max();
    bfs(u);
    int v = get_max();
    while (v != -1) {
        path.push_back({v, dist[v]});
        v = pre[v];
    }
    reverse(path.begin(), path.end());
    
    // 2. 二分偏心距
    int l = 0, r = 2e9;
    while (l < r) {
        int mid = (LL) l + r >> 1;
        if (check(mid))
            r = mid;
        else
            l = mid + 1;
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

> [!NOTE] **[Luogu [APIO2010]巡逻](https://www.luogu.com.cn/problem/P3629)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **对比两种(dfs/dp)求直径的方式**
> 
> 同时使用两种方式
> 
> **取反计算的trick思维**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PII = pair<int, int>;
const int N = 1e5 + 10, M = N << 1;

int n, k;
int h[N], e[M], w[M], ne[M], idx;
// ---------- bfs 求直径(可得完整路径) ----------
int q[N], dist[N], pre[N];
vector<PII> path;
unordered_set<LL> S;
// ---------- dfs 求带负权的直径 ----------
int f[N], res;
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int get_max() {
    int t = 1;
    for (int i = 1; i <= n; ++ i )
        if (dist[t] < dist[i])
            t = i;
    return t;
}

void bfs(int start) {
    memset(dist, 0x3f, sizeof dist);
    memset(pre, -1, sizeof pre);
    dist[start] = 0;
    int hh = 0, tt = -1;
    q[ ++ tt] = start;
    
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[t] + w[i]) {
                pre[j] = t;
                dist[j] = dist[t] + w[i];
                q[ ++ tt] = j;
            }
        }
    }
}

void dfs(int u) {
    st[u] = true;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (st[j])
            continue;
        int cost = w[i];
        if (S.count((LL)u * N + j) || S.count((LL)j * N + u))
            cost = -cost;
        dfs(j);
        // ATTENTION 多叉树求直径
        res = max(res, f[u] + f[j] + cost);
        f[u] = max(f[u], f[j] + cost);
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> k;
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b, 1), add(b, a, 1);
    }
    
    bfs(1);
    int u = get_max();
    bfs(u);
    int v = get_max();
    while (v != -1) {
        path.push_back({v, dist[v]});
        {
            S.insert((LL)v * N + pre[v]);
        }
        v = pre[v];
    }
    reverse(path.begin(), path.end());
    
    if (k == 1) {
        //   (n - 1) * 2 - (l1 - 1)
        // 其中 l1 = path.size() - 1
        cout << (n - 1) * 2 - (path.size() - 1 - 1) << endl;
    } else {
        memset(f, 0, sizeof f);
        res = 0;
        dfs(1);
        //   (n - 1) * 2 - (l1 - 1) - (l2 - 1)
        // 其中 l1 = path.size() - 1, l2 = res
        cout << (n - 1) * 2 - (res - 1) - (path.size() - 1 - 1) << endl;
    }
    
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

> [!NOTE] **[2246. 相邻字符不同的最长路径](https://leetcode.cn/problems/longest-path-with-different-adjacent-characters/)**
>
> 题意: TODO

> [!TIP] **思路**
>
> 显然相同字符不能连边，最终会形成森林

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```c++
class Solution {
public:
    const static int N = 1e5 + 10, M = N << 1;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n, d;
    bool st[N];
    int d1[N], d2[N];
    
    void dfs(int u, int fa) {
        st[u] = true;
        d1[u] = d2[u] = 1;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            dfs(j, u);
            if (d1[j] + 1 >= d1[u])
                d2[u] = d1[u], d1[u] = d1[j] + 1;
            else if (d1[j] + 1 > d2[u])
                d2[u] = d1[j] + 1;
        }
        // cout << "... u = " << u << " " << d1[u] << " " << d2[u] << endl;
        d = max(d, d1[u] + d2[u] - 1);
    }
    
    int longestPath(vector<int>& parent, string s) {
        init();
        
        n = parent.size();
        for (int i = 1; i < n; ++ i ) {
            int a = parent[i], b = i;
            if (s[a] != s[b])
                add(a, b), add(b, a);
        }
        
        int res = 0;
        memset(st, 0, sizeof st);
        for (int i = 0; i < n; ++ i )
            if (!st[i]) {
                d = 0;
                dfs(i, -1);
                res = max(res, d);
                // cout << " i = " << i << " d = " << d << endl;
            }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        g = defaultdict(list)
        for i, u in enumerate(parent):
            g[u].append(i)

        @lru_cache(None)        
        def dfs(u):
            d, res = [0, 0], 0
            for j in g[u]:
                if s[j] != s[u]:
                    cur = dfs(j)
                    res = max(res, cur[1])
                    if d[0] <= cur[0]:
                        d[0], d[1] = cur[0], d[0]
                    elif d[1] < cur[0]:
                        d[1] = cur[0]
            return [d[0] + 1, max(res, d[0] + d[1] + 1)]
        
        ans = 0
        for i in range(len(parent)):
            ans = max(ans, max(dfs(i)))
        return ans
```

##### **Python-2**

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        def dfs(u):             
            nonlocal ans
            d1 = d2 = 1         # 最大、次大路径长度的节点个数
            for v in graph[u]:   
                d = dfs(v) + 1
                if s[v] == s[u]:    
                    continue 
                if d >= d1:          # 更新最大值、次大值
                    d1, d2 = d, d1
                elif d > d2:
                    d2 = d 
            
            ans = max(ans, d1 + d2 - 1)  
            return d1          # 返回值: 节点u到后代节点的最长路径上边的节点个数
        
        n = len(parent)                 # 节点数目
        graph = [[] for _ in range(n)]  
        for v in range(1, n):           # 遍历节点（跳过根结点）
            graph[parent[v]].append(v)
        
        ans = 0
        dfs(0)
        return ans   
```

##### **Python-3**

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        N = 100010; M = N << 1
        h, ev, ne, idx = [-1] * N, [0] * M, [0] * M, 0

        def add_edge(a, b):
            nonlocal idx
            ev[idx] = b; ne[idx] = h[a]; h[a] = idx; idx += 1

        def dfs(u, fa):
            nonlocal res
            d1 = d2 = 1      
            i = h[u]
            while i != -1:
                j = ev[i]
                i = ne[i]
                if j == fa:
                    continue
                d = dfs(j, u) + 1
                if s[j] == s[u]:
                    continue
                if d >= d1:          # 更新最大值、次大值
                    d1, d2 = d, d1
                elif d > d2:
                    d2 = d 
            res = max(res, d1 + d2 - 1)
            return d1

        n = len(parent)
        a, b = 0, 0
        for i in range(1, n):
            a, b = parent[i], i
            add_edge(a, b)
            add_edge(b, a)

        res = 0
        dfs(0, -1)
        return res
```



<!-- tabs:end -->
</details>

<br>