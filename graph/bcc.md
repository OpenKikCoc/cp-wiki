## 简介

在阅读下列内容之前，请务必了解 [图论相关概念](./concept.md) 部分。

相关阅读：[割点和桥](./cut.md)

## 定义

割点和桥更严谨的定义参见 [图论相关概念](./concept.md)。

在一张连通的无向图中，对于两个点 $u$ 和 $v$，如果无论删去哪条边（只能删去一条）都不能使它们不连通，我们就说 $u$ 和 $v$  **边双连通**。

在一张连通的无向图中，对于两个点 $u$ 和 $v$，如果无论删去哪个点（只能删去一个，且不能删 $u$ 和 $v$ 自己）都不能使它们不连通，我们就说 $u$ 和 $v$  **点双连通**。

边双连通具有传递性，即，若 $x,y$ 边双连通，$y,z$ 边双连通，则 $x,z$ 边双连通。

点双连通 **不** 具有传递性，反例如下图，$A,B$ 点双连通，$B,C$ 点双连通，而 $A,C$  **不** 点双连通。

![bcc-counterexample.png](./images/bcc-0.svg)

## DFS

对于一张连通的无向图，我们可以从任意一点开始 DFS，得到原图的一棵生成树（以开始 DFS 的那个点为根），这棵生成树上的边称作 **树边**，不在生成树上的边称作 **非树边**。

由于 DFS 的性质，我们可以保证所有非树边连接的两个点在生成树上都满足其中一个是另一个的祖先。

DFS 的代码如下：

```cpp
// C++ Version
void DFS(int p) {
    visited[p] = true;
    for (int to : edge[p])
        if (!visited[to]) DFS(to);
}
```

```python
# Python Version
def DFS(p):
    visited[p] = True
    for to in edge[p]:
        if visited[to] == False:
            DFS(to)
```

## DFS 找桥并判断边双连通

首先，对原图进行 DFS。

![bcc-1.png](./images/bcc-1.svg)

如上图所示，黑色与绿色边为树边，红色边为非树边。每一条非树边连接的两个点都对应了树上的一条简单路径，我们说这条非树边 **覆盖** 了这条树上路径上所有的边。绿色的树边 **至少** 被一条非树边覆盖，黑色的树边不被 **任何** 非树边覆盖。

我们如何判断一条边是不是桥呢？显然，非树边和绿色的树边一定不是桥，黑色的树边一定是桥。

如何用算法去实现以上过程呢？首先有一个比较暴力的做法，对于每一条非树边，都逐个地将它覆盖的每一条树边置成绿色，这样的时间复杂度为 $O(nm)$。

怎么优化呢？可以用差分。对于每一条非树边，在其树上深度较小的点处打上 `-1` 标记，在其树上深度较大的点处打上 `+1` 标记。然后 $O(n)$ 求出每个点的子树内部的标记之和。对于一个点 $u$，其子树内部的标记之和等于覆盖了 $u$ 和 $u$ 的父亲之间的树边的非树边数量。若这个值非 $0$，则 $u$ 和 $u$ 的父亲之间的树边不是桥，否则是桥。

用以上的方法 $O(n+m)$ 求出每条边分别是否是桥后，两个点是边双连通的，当且仅当它们的树上路径中 **不** 包含桥。

## DFS 找割点并判断点双连通

![bcc-2.png](./images/bcc-2.svg)

如上图所示，黑色边为树边，红色边为非树边。每一条非树边连接的两个点都对应了树上的一条简单路径。

考虑一张新图，新图中的每一个点对应原图中的每一条树边（在上图中用蓝色点表示）。对于原图中的每一条非树边，将这条非树边对应的树上简单路径中的所有边在新图中对应的蓝点连成一个连通块（这在上图中也用蓝色的边体现出来了）。

这样，一个点不是割点，当且仅当与其相连的所有边在新图中对应的蓝点都属于同一个连通块。两个点点双连通，当且仅当它们在原图的树上路径中的所有边在新图中对应的蓝点都属于同一个连通块。

蓝点间的连通关系可以用与求边双连通时用到的差分类似的方法维护，时间复杂度 $O(n+m)$。

## 习题

> [!NOTE] **[AcWing 395. 冗余路径](https://www.acwing.com/problem/content/397/)**
> 
> 题意: 
> 
> 边双联通分量

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 5010, M = 20010;

int n, m;
int h[N], e[M], ne[M], idx;
int dfn[N], low[N], timestamp;
int stk[N], top;
int id[N], dcc_cnt;
bool is_bridge[M];
int d[N];   // 度数

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void tarjan(int u, int from) {
    dfn[u] = low[u] = ++ timestamp;
    stk[ ++ top] = u;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!dfn[j]) {
            tarjan(j, i);
            low[u] = min(low[u], low[j]);
            if (dfn[u] < low[j])
                is_bridge[i] = is_bridge[i ^ 1] = true;
        } else if (i != (from ^ 1)) low[u] = min(low[u], dfn[j]);
    }
    if (dfn[u] == low[u]) {
        ++ dcc_cnt;
        int y;
        do {
            y = stk[top -- ];
            id[y] = dcc_cnt;
        } while (y != u);
    }
}

int main() {
    cin >> n >> m;
    memset(h, -1, sizeof h);
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    tarjan(1, -1);
    for (int i = 0; i < idx; ++ i )
        if (is_bridge[i])
            d[id[e[i]]] ++ ;
    int cnt = 0;
    for (int i = 1; i <= dcc_cnt; ++ i )
        if (d[i] == 1)
            ++ cnt;
    cout << (cnt + 1) / 2 << endl;  // cnt/2取上界
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1183. 电力](https://www.acwing.com/problem/content/1185/)**
> 
> 题意: 
> 
> 点双联通分量

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int  N = 10010, M = 30010;

int n, m;
int h[N], e[M], ne[M], idx;
int dfn[N], low[N], timestamp;
int root, ans;  // 当前根 去除某个点得到的最大联通量个数

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void tarjan(int u) {
    dfn[u] = low[u] = ++ timestamp;
    // 去除当前节点后联通量个数
    int cnt = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!dfn[j]) {
            tarjan(j);
            low[u] = min(low[u], low[j]);
            if (low[j] >= dfn[u]) ++ cnt;
        } else low[u] = min(low[u], dfn[j]);
    }
    if (u != root) ++ cnt;
    ans = max(ans, cnt);
}

int main() {
    while (cin >> n >> m, n || m) {
        memset(h, -1, sizeof h);
        memset(dfn, 0, sizeof dfn);
        idx = timestamp = 0;
        
        while (m -- ) {
            int a, b;
            cin >> a >> b;
            add(a, b), add(b, a);
        }
        ans = 0;
        int cnt = 0;    // 联通量个数
        for (root = 0; root < n; ++ root )
            if (!dfn[root]) {
                ++ cnt;
                tarjan(root);
            }
        cout << ans + cnt - 1 << endl;
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 396. 矿场搭建](https://www.acwing.com/problem/content/398/)**
> 
> 题意: 
> 
> 求具体的点双联通分量

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using ULL = unsigned long long;

const int N = 1010, M = 1010;

int n, m;
int h[N], e[M], ne[M], idx;
int dfn[N], low[N], timestamp;
int stk[N], top;
int dcc_cnt;
vector<int> dcc[N];
bool cut[N];
int root;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void tarjan(int u) {
    dfn[u] = low[u] = ++ timestamp;
    stk[ ++ top] = u;
    if (u == root && h[u] == -1) {
        ++ dcc_cnt ;
        dcc[dcc_cnt].push_back(u);
        return;
    }
    
    int cnt = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!dfn[j]) {
            tarjan(j);
            low[u] = min(low[u], low[j]);
            if (low[j] >= dfn[u]) {
                ++ cnt ;
                if (u != root || cnt > 1) cut[u] = true;
                ++ dcc_cnt;
                int y;
                do {
                    y = stk[top -- ];
                    dcc[dcc_cnt].push_back(y);
                } while (y != j);
                dcc[dcc_cnt].push_back(u);
            }
        } else low[u] = min(low[u], dfn[j]);
    }
}

int main() {
    int T = 1;
    while (cin >> m, m) {
        for (int i = 1; i <= dcc_cnt; ++ i ) dcc[i].clear();
        idx = n = timestamp = top = dcc_cnt = 0;
        memset(h, -1, sizeof h);
        memset(dfn, 0, sizeof dfn);
        memset(cut, 0, sizeof cut);
        
        while (m -- ) {
            int a, b;
            cin >> a >> b;
            n = max(n, a), n = max(n, b);
            add(a, b), add(b, a);
        }
        for (root = 1; root <= n; ++ root )
            if (!dfn[root])
                tarjan(root);
        
        int res = 0;
        ULL num = 1;
        for (int i = 1; i <= dcc_cnt; ++ i ) {
            int cnt = 0;
            for (int j = 0; j < dcc[i].size(); ++ j )
                if (cut[dcc[i][j]])
                    ++ cnt ;    // 割点个数
            // 1. 割点数大于1 则无论哪个被堵 连通性依旧 略
            // 2. 割点数为0 则若点数量大于1:任意建两个 点数量为1:只能建一个
            // 3. 割点数为1 割点及分量内任意一点各建一个
            if (cnt == 0) {
                // 至少需要两个
                if (dcc[i].size() > 1) res += 2, num *= dcc[i].size() * (dcc[i].size() - 1) / 2;
                // 需要一个
                else ++ res;
            } else if (cnt == 1) ++ res, num *= dcc[i].size() - 1;
        }
        cout << "Case " << T++ << ": " << res << " " << num << endl;
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *