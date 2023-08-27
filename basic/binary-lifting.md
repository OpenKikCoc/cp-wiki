倍增法（英语：binary lifting），顾名思义就是翻倍。它能够使线性的处理转化为对数级的处理，大大地优化时间复杂度。

这个方法在很多算法中均有应用，其中最常用的是 RMQ 问题和求 [LCA（最近公共祖先）](graph/lca.md)。

## RMQ 问题

参见：[RMQ 专题](topic/rmq.md)

RMQ 是 Range Maximum/Minimum Query 的缩写，表示区间最大（最小）值。使用倍增思想解决 RMQ 问题的方法是 [ST 表](ds/sparse-table.md)。

## 树上倍增求 LCA

参见：[最近公共祖先](graph/lca.md)

## 例题

### 题 1

> [!NOTE] **例题**
> 
> 如何用尽可能少的砝码称量出 $[0,31]$ 之间的所有重量？（只能在天平的一端放砝码）

> [!TIP] **解题思路**
> 
> 答案是使用 1 2 4 8 16 这五个砝码，可以称量出 $[0,31]$ 之间的所有重量。同样，如果要称量 $[0,127]$ 之间的所有重量，可以使用 1 2 4 8 16 32 64 这七个砝码。每次我们都选择 2 的整次幂作砝码的重量，就可以使用极少的砝码个数量出任意我们所需要的重量。
> 
> 为什么说是极少呢？因为如果我们要量出 $[0,1023]$ 之间的所有重量，只需要 9 个砝码，需要量出 $[0,1048575]$ 之间的所有重量，只需要 19 个。如果我们的目标重量翻倍，砝码个数只需要增加 1。这叫“对数级”的增长速度，因为砝码的所需个数与目标重量的范围的对数成正比。

### 题 2

> [!NOTE] **例题**
> 
> 给出一个长度为 $n$ 的环和一个常数 $k$，每次会从第 $i$ 个点跳到第 $(i+k)\bmod n+1$ 个点，总共跳了 $m$ 次。每个点都有一个权值，记为 $a_i$，求 $m$ 次跳跃的起点的权值之和对 $10^9+7$ 取模的结果。
> 
> 数据范围：$1\leq n\leq 10^6$，$1\leq m\leq 10^{18}$，$1\leq k\leq n$，$0\le a_i\le 10^9$。

> [!TIP] **解题思路**
> 
> 这里显然不能暴力模拟跳 $m$ 次。因为 $m$ 最大可到 $10^{18}$ 级别，如果暴力模拟的话，时间承受不住。
> 
> 所以就需要进行一些预处理，提前整合一些信息，以便于在查询的时候更快得出结果。如果记录下来每一个可能的跳跃次数的结果的话，不论是时间还是空间都难以承受。
> 
> 在这题上，就是我们预处理出从每个点开始跳 1、2、4、8 等等步之后的结果（所处点和点权和），然后如果要跳 13 步，只需要跳 1+4+8 步就好了。也就是说先在起始点跳 1 步，然后再在跳了之后的终点跳 4 步，再接着跳 8 步，同时统计一下预先处理好的点权和，就可以知道跳 13 步的点权和了。
> 
> 对于每一个点开始的 $2^i$ 步，记录一个 `go[i][x]` 表示第 $x$ 个点跳 $2^i$ 步之后的终点，而 `sum[i][x]` 表示第 $x$ 个点跳 $2^i$ 步之后能获得的点权和。预处理的时候，开两重循环，对于跳 $2^i$ 步的信息，我们可以看作是先跳了 $2^{i-1}$ 步，再跳 $2^{i-1}$ 步，因为显然有 $2^{i-1}+2^{i-1}=2^i$。即我们有 `sum[i][x] = sum[i-1][x]+sum[i-1][go[i-1][x]]`，且 `go[i][x] = go[i-1][go[i-1][x]]`。
> 
> 当然还有一些实现细节需要注意。为了保证统计的时候不重不漏，我们一般预处理出“左闭右开”的点权和。亦即，对于跳 1 步的情况，我们只记录该点的点权和；对于跳 2 步的情况，我们只记录该点及其下一个点的点权和。相当于总是不将终点的点权和计入 sum。这样在预处理的时候，只需要将两部分的点权和直接相加就可以了，不需要担心第一段的终点和第二段的起点会被重复计算。
> 
> 这题的 $m\leq 10^{18}$，虽然看似恐怖，但是实际上只需要预处理出 $65$ 以内的 $i$，就可以轻松解决，比起暴力枚举快了很多。用行话讲，这个做法的 `时间复杂度` 是预处理 $\Theta(n\log m)$，查询每次 $\Theta(\log m)$。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
TODO@binacs
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

## 习题

### 基础倍增

> [!NOTE] **[Luogu 跑路](https://www.luogu.com.cn/problem/P1613)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 基础倍增 + floyd

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 55, M = 65;

int n, m;
int f[N][N][M], dis[N][N];

int main() {
    cin >> n >> m;
    
    memset(dis, 0x3f, sizeof dis);
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        f[a][b][0] = 1;
        dis[a][b] = 1;
    }
    
    for (int d = 1; d < M; ++ d )
        for (int k = 1; k <= n; ++ k )
            for (int i = 1; i <= n; ++ i )
                for (int j = 1; j <= n; ++ j )
                    if (f[i][k][d - 1] && f[k][j][d - 1])
                        f[i][j][d] = 1, dis[i][j] = 1;
    
    for (int k = 1; k <= n; ++ k )
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= n; ++ j )
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);
    cout << dis[1][n] << endl;
    
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

> [!NOTE] **[LeetCode 1483. 树节点的第 K 个祖先](https://leetcode-cn.com/problems/kth-ancestor-of-a-tree-node/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> LCA 简单版
> 
> 倍增即可 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ bfs 标准**

```cpp
class TreeAncestor {
public:
    const static int N = 5e4 + 10, M = N << 1;

    int n, m;
    int h[N], e[M], ne[M], idx;
    int depth[N], fa[N][17];
    int q[N];

    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }

    void bfs(int root) {
        memset(depth, 0x3f, sizeof depth);
        depth[0] = 0, depth[root] = 1;
        int hh = 0, tt = -1;
        q[ ++ tt] = root;
        while (hh <= tt) {
            int t = q[hh ++ ];
            for (int i = h[t]; ~i; i = ne[i]) {
                int j = e[i];
                if (depth[j] > depth[t] + 1) {
                    depth[j] = depth[t] + 1;
                    q[ ++ tt] = j;

                    fa[j][0] = t;
                    for (int k = 1; k <= 16; ++ k )
                        fa[j][k] = fa[fa[j][k - 1]][k - 1];
                }
            }
        }
    }

    // int lca(int a, int b) {
    //     if (depth[a] < depth[b])
    //         swap(a, b);
    //     for (int k = 16; k >= 0; -- k )
    //         if (depth[fa[a][k]] >= depth[b])
    //             a = fa[a][k];
    //     if (a == b)
    //         return a;
    //     for (int k = 16; k >= 0; -- k )
    //         if (fa[a][k] != fa[b][k])
    //             a = fa[a][k], b = fa[b][k];
    //     return fa[a][0];
    // }

    TreeAncestor(int n, vector<int>& parent) {
        memset(h, -1, sizeof h); idx = 0;
        for (int i = 1; i < n; ++ i )
            add(parent[i] + 1, i + 1);
        bfs(1);
    }
    
    int getKthAncestor(int node, int k) {
        int x = node + 1;
        for (int i = 16; i >= 0; -- i )
            if (k >> i & 1)
                x = fa[x][i];
        return x - 1;
    }
};
```

##### **C++ dfs**

```cpp
vector<int> v[200001];
int d[100001][19];

class TreeAncestor {
public:
    void dfs(int x, int fa) {
        d[x][0] = fa;
        for (int j = 1; j <= 18; j ++)
            d[x][j] = d[d[x][j - 1]][j - 1];
        for (int i = 0; i < (int )v[x].size(); i ++) {
            if (v[x][i] == fa) continue;
            dfs(v[x][i], x);
        }
    }
    TreeAncestor(int n, vector<int>& parent) {
        for (int i = 1; i <= n; i ++) v[i].clear();
        for (int i = 0; i < n; i ++) {
            if (i == 0) continue;
            v[i + 1].push_back(parent[i] + 1);
            v[parent[i] + 1].push_back(i + 1);
        }
        dfs(1, 0);
    }
    
    int getKthAncestor(int node, int k) {
        int x = node + 1;
        for (int i = 18; i >= 0; i --)
            if ((k >> i) & 1) x = d[x][i];
        return x - 1;
    }
};
// ZhuolinYang
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2836. 在传球游戏中最大化函数值](https://leetcode.cn/problems/maximize-value-of-function-in-a-ball-passing-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准倍增
> 
> 数据范围敏感度 要能想到使用倍增

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using LL = long long;
const static int N = 1e5 + 10, M = 35;

// 倍增预处理
// 每个节点的 i 的第 2^j 个祖先节点，以及从 i 到第 2^j 祖先的节点编号之和 (不包含第 2^j 的祖先)
// [放在 class 外防止 TLE]
int f[N][M];
LL g[N][M];    

class Solution {
public:
    vector<int> r;
    int n;

    long long getMaxFunctionValue(vector<int>& receiver, long long k) {
        this->r = receiver;
        this->n = r.size();
        
        k ++ ;  // ATTENTION: 恰好传 k 次
        
        {
            // 标准倍增
            for (int j = 0; j < M; ++ j )
                for (int i = 0; i < n; ++ i )
                    if (j == 0) {
                        f[i][0] = r[i], g[i][0] = i;
                    } else {
                        f[i][j] = f[f[i][j - 1]][j - 1];
                        g[i][j] = g[i][j - 1] + g[f[i][j - 1]][j - 1];
                    }
        }
            
        LL res = 0;
        for (int i = 0; i < n; ++ i ) {
            LL t = 0;
            for (int j = M - 1, p = i; j >= 0; -- j )
                if (k >> j & 1) {
                    t += g[p][j];
                    p = f[p][j];
                }
            res = max(res, t);
        }
        return res;
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