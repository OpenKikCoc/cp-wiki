
## 定义

（还记得这些定义吗？在阅读下列内容之前，请务必了解 [图论相关概念](./concept.md) 中的基础部分。）

- 路径
- 最短路
- 有向图中的最短路、无向图中的最短路
- 单源最短路、每对结点之间的最短路

## 性质

对于边权为正的图，任意两个结点之间的最短路，不会经过重复的结点。

对于边权为正的图，任意两个结点之间的最短路，不会经过重复的边。

对于边权为正的图，任意两个结点之间的最短路，任意一条的结点数不会超过 $n$，边数不会超过 $n-1$。

## 记号

为了方便叙述，这里先给出下文将会用到的一些记号的含义。

- $n$ 为图上点的数目，$m$ 为图上边的数目；
- $s$ 为最短路的源点；
- $D(u)$ 为 $s$ 点到 $u$ 点的 **实际** 最短路长度；
- $dis(u)$ 为 $s$ 点到 $u$ 点的 **估计** 最短路长度。任何时候都有 $dis(u) \geq D(u)$。特别地，当最短路算法终止时，应有 $dis(u)=D(u)$。
- $w(u,v)$ 为 $(u,v)$ 这一条边的边权。

## Floyd 算法

是用来求任意两个结点之间的最短路的。

复杂度比较高，但是常数小，容易实现。（我会说只有三个 `for` 吗？）

适用于任何图，不管有向无向，边权正负，但是最短路必须存在。（不能有个负环）

### 实现

我们定义一个数组 `f[k][x][y]`，表示只允许经过结点 $1$ 到 $k$（也就是说，在子图 $V'={1, 2, \ldots, k}$ 中的路径，注意，$x$ 与 $y$ 不一定在这个子图中），结点 $x$ 到结点 $y$ 的最短路长度。

很显然，`f[n][x][y]` 就是结点 $x$ 到结点 $y$ 的最短路长度（因为 $V'={1, 2, \ldots, n}$ 即为 $V$ 本身，其表示的最短路径就是所求路径）。

接下来考虑如何求出 `f` 数组的值。

`f[0][x][y]`：$x$ 与 $y$ 的边权，或者 $0$，或者 $+\infty$（`f[0][x][y]` 什么时候应该是 $+\infty$？当 $x$ 与 $y$ 间有直接相连的边的时候，为它们的边权；当 $x = y$ 的时候为零，因为到本身的距离为零；当 $x$ 与 $y$ 没有直接相连的边的时候，为 $+\infty$）。

`f[k][x][y] = min(f[k-1][x][y], f[k-1][x][k]+f[k-1][k][y])`（`f[k-1][x][y]`，为不经过 $k$ 点的最短路径，而 `f[k-1][x][k]+f[k-1][k][y]`，为经过了 $k$ 点的最短路）。

上面两行都显然是对的，所以说这个做法空间是 $O(N^3)$，我们需要依次增加问题规模（$k$ 从 $1$ 到 $n$），判断任意两点在当前问题规模下的最短路。

```cpp
// C++ Version
for (k = 1; k <= n; k++) {
    for (x = 1; x <= n; x++) {
        for (y = 1; y <= n; y++) {
            f[k][x][y] = min(f[k - 1][x][y], f[k - 1][x][k] + f[k - 1][k][y]);
        }
    }
}
```

```python
# Python Version
for k in range(1, n + 1):
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            f[k][x][y] = min(f[k - 1][x][y], f[k - 1][x][k] + f[k - 1][k][y])
```

因为第一维对结果无影响，我们可以发现数组的第一维是可以省略的，于是可以直接改成 `f[x][y] = min(f[x][y], f[x][k]+f[k][y])`。

> [!NOTE] **证明第一维对结果无影响**
> 
> 我们注意到如果放在一个给定第一维 `k` 二维数组中，`f[x][k]` 与 `f[k][y]` 在某一行和某一列。而 `f[x][y]` 则是该行和该列的交叉点上的元素。
> 
> 现在我们需要证明将 `f[k][x][y]` 直接在原地更改也不会更改它的结果：我们注意到 `f[k][x][y]` 的涵义是第一维为 `k-1` 这一行和这一列的所有元素的最小值，包含了 `f[k-1][x][y]`，那么我在原地进行更改也不会改变最小值的值，因为如果将该三维矩阵压缩为二维，则所求结果 `f[x][y]` 一开始即为原 `f[k-1][x][y]` 的值，最后依然会成为该行和该列的最小值。
> 
> 故可以压缩。

```cpp
// C++ Version
for (k = 1; k <= n; k++) {
    for (x = 1; x <= n; x++) {
        for (y = 1; y <= n; y++) { f[x][y] = min(f[x][y], f[x][k] + f[k][y]); }
    }
}
```

```python
# Python Version
for k in range(1, n + 1):
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            f[x][y] = min(f[x][y], f[x][k] + f[k][y])
```

综上时间复杂度是 $O(N^3)$，空间复杂度是 $O(N^2)$。

### 应用

> [!NOTE] question **给一个正权无向图，找一个最小权值和的环。**
> 
> 首先这一定是一个简单环。
> 
> 想一想这个环是怎么构成的。
> 
> 考虑环上编号最大的结点 u。
> 
> `f[u-1][x][y]` 和 (u,x), (u,y）共同构成了环。
> 
> 在 Floyd 的过程中枚举 u，计算这个和的最小值即可。
> 
> 时间复杂度为 $O(n^3)$。

> [!TIP]+question **已知一个有向图中任意两点之间是否有连边，要求判断任意两点是否连通。**
> 
> 该问题即是求 **图的传递闭包**。
> 
> 我们只需要按照 Floyd 的过程，逐个加入点判断一下。
> 
> 只是此时的边的边权变为 $1/0$，而取 $\min$ 变成了 **或** 运算。
> 
> 再进一步用 bitset 优化，复杂度可以到 $O(\frac{n^3}{w})$。

```cpp
// std::bitset<SIZE> f[SIZE];
for (k = 1; k <= n; k++)
    for (i = 1; i <= n; i++)
        if (f[i][k]) f[i] = f[i] | f[k];
```

* * *

## Bellman-Ford 算法

Bellman-Ford 算法是一种基于松弛（relax）操作的最短路算法，可以求出有负权的图的最短路，并可以对最短路不存在的情况进行判断。

在国内 OI 界，你可能听说过的“SPFA”，就是 Bellman-Ford 算法的一种实现。

### 流程

先介绍 Bellman-Ford 算法要用到的松弛操作（Djikstra 算法也会用到松弛操作）。

对于边 $(u,v)$，松弛操作对应下面的式子：$dis(v) = \min(dis(v), dis(u) + w(u, v))$。

这么做的含义是显然的：我们尝试用 $S \to u \to v$（其中 $S \to u$ 的路径取最短路）这条路径去更新 $v$ 点最短路的长度，如果这条路径更优，就进行更新。

Bellman-Ford 算法所做的，就是不断尝试对图上每一条边进行松弛。我们每进行一轮循环，就对图上所有的边都尝试进行一次松弛操作，当一次循环中没有成功的松弛操作时，算法停止。

每次循环是 $O(m)$ 的，那么最多会循环多少次呢？

在最短路存在的情况下，由于一次松弛操作会使最短路的边数至少 $+1$，而最短路的边数最多为 $n-1$，因此整个算法最多执行 $n-1$ 轮松弛操作。故总时间复杂度为 $O(nm)$。

但还有一种情况，如果从 $S$ 点出发，抵达一个负环时，松弛操作会无休止地进行下去。注意到前面的论证中已经说明了，对于最短路存在的图，松弛操作最多只会执行 $n-1$ 轮，因此如果第 $n$ 轮循环时仍然存在能松弛的边，说明从 $S$ 点出发，能够抵达一个负环。

> [!WARNING] **负环判断中存在的常见误区**
> 
> 需要注意的是，以 $S$ 点为源点跑 Bellman-Ford 算法时，如果没有给出存在负环的结果，只能说明从 $S$ 点出发不能抵达一个负环，而不能说明图上不存在负环。
> 
> 因此如果需要判断整个图上是否存在负环，最严谨的做法是建立一个超级源点，向图上每个节点连一条权值为 0 的边，然后以超级源点为起点执行 Bellman-Ford 算法。

### 代码实现


```cpp
// C++ Version
struct edge {
    int v, w;
};
vector<edge> e[maxn];
int dis[maxn];
bool bellmanford(int n, int s) {
    memset(dis, 63, sizeof(dis));
    dis[s] = 0;
    bool flag;
    for (int i = 1; i <= n; i++) {
        flag = false;
        for (int u = 1; u <= n; u++) {
            for (auto ed : e[u]) {
                int v = ed.v, w = ed.w;
                if (dis[v] > dis[u] + w) {
                    dis[v] = dis[u] + w;
                    flag = true;
                }
            }
        }
        // 没有可以松弛的边时就停止算法
        if (!flag) break;
    }
    // 第 n 轮循环仍然可以松弛时说明 s 点可以抵达一个负环
    return flag;
}
```

```python
    # Python Version
    class Edge:
        v = 0
        w = 0
    
    e = [[Edge() for i in range(maxn)] for j in range(maxn)]
    dis = [63] * maxn
    
    def bellmanford(n, s):
        dis[s] = 0
        for i in range(1, n + 1):
            flag = False
            for u in range(1, n + 1):
                for ed in e[u]:
                    v = ed.v; w = ed.w
                    if dis[v] > dis[u] + w:
                        flag = True
            # 没有可以松弛的边时就停止算法
            if flag == False:
                break
        # 第 n 轮循环仍然可以松弛时说明 s 点可以抵达一个负环
        return flag
```

### 队列优化：SPFA

即 Shortest Path Faster Algorithm。

很多时候我们并不需要那么多无用的松弛操作。

很显然，只有上一次被松弛的结点，所连接的边，才有可能引起下一次的松弛操作。

那么我们用队列来维护“哪些结点可能会引起松弛操作”，就能只访问必要的边了。

SPFA 也可以用于判断 $s$ 点是否能抵达一个负环，只需记录最短路经过了多少条边，当经过了至少 $n$ 条边时，说明 $s$ 点可以抵达一个负环。


```cpp
// C++ Version
struct edge {
    int v, w;
};
vector<edge> e[maxn];
int dis[maxn], cnt[maxn], vis[maxn];
queue<int> q;
bool spfa(int n, int s) {
    memset(dis, 63, sizeof(dis));
    dis[s] = 0, vis[s] = 1;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop(), vis[u] = 0;
        for (auto ed : e[u]) {
            int v = ed.v, w = ed.w;
            if (dis[v] > dis[u] + w) {
                dis[v] = dis[u] + w;
                cnt[v] = cnt[u] + 1;  // 记录最短路经过的边数
                if (cnt[v] >= n) return false;
                // 在不经过负环的情况下，最短路至多经过 n - 1 条边
                // 因此如果经过了多于 n 条边，一定说明经过了负环
                if (!vis[v]) q.push(v), vis[v] = 1;
            }
        }
    }
    return true;
}
```

```python
    # Python Version
    class Edge:
        v = 0
        w = 0
    
    e = [[Edge() for i in range(maxn)] for j in range(maxn)]
    dis = [63] * maxn; cnt = [] * maxn; vis = [] * maxn
    
    q = []
    def spfa(n, s):
        dis[s] = 0; vis[s] = 1
        q.append(s)
        while len(q) != 0:
            u = q[0]
            q.pop(); vis[u] = 0
            for ed in e[u]:
                if dis[v] > dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1 # 记录最短路经过的边数
                    if cnt[v] >= n:
                        return False
                    # 在不经过负环的情况下，最短路至多经过 n - 1 条边
                    # 因此如果经过了多于 n 条边，一定说明经过了负环
                    if vis[v] == True:
                        q.append(v)
                        vis[v] = True
```

虽然在大多数情况下 SPFA 跑得很快，但其最坏情况下的时间复杂度为 $O(nm)$，将其卡到这个复杂度也是不难的，所以考试时要谨慎使用（在没有负权边时最好使用 Dijkstra 算法，在有负权边且题目中的图没有特殊性质时，若 SPFA 是标算的一部分，题目不应当给出 Bellman-Ford 算法无法通过的数据范围）。

> [!NOTE] **Bellman-Ford 的其他优化**
> 
> 除了队列优化（SPFA）之外，Bellman-Ford 还有其他形式的优化，这些优化在部分图上效果明显，但在某些特殊图上，最坏复杂度可能达到指数级。
> 
> - 堆优化：将队列换成堆，与 Dijkstra 的区别是允许一个点多次入队。在有负权边的图可能被卡成指数级复杂度。
> 
> - 栈优化：将队列换成栈（即将原来的 BFS 过程变成 DFS），在寻找负环时可能具有更高效率，但最坏时间复杂度仍然为指数级。
> 
> - LLL 优化：将普通队列换成双端队列，每次将入队结点距离和队内距离平均值比较，如果更大则插入至队尾，否则插入队首。
> 
> - SLF 优化：将普通队列换成双端队列，每次将入队结点距离和队首比较，如果更大则插入至队尾，否则插入队首。
> 
> - D´Esopo-Pape 算法：将普通队列换成双端队列，如果一个节点之前没有入队，则将其插入队尾，否则插入队首。
> 
> 更多优化以及针对这些优化的 Hack 方法，可以看 [fstqwq 在知乎上的回答](https://www.zhihu.com/question/292283275/answer/484871888)。

## Dijkstra 算法

Dijkstra（/ˈdikstrɑ/或/ˈdɛikstrɑ/）算法由荷兰计算机科学家 E. W. Dijkstra 于 1956 年发现，1959 年公开发表。是一种求解 **非负权图** 上单源最短路径的算法。

### 流程

将结点分成两个集合：已确定最短路长度的点集（记为 $S$ 集合）的和未确定最短路长度的点集（记为 $T$ 集合）。一开始所有的点都属于 $T$ 集合。

初始化 $dis(s)=0$，其他点的 $dis$ 均为 $+\infty$。

然后重复这些操作：

1. 从 $T$ 集合中，选取一个最短路长度最小的结点，移到 $S$ 集合中。
2. 对那些刚刚被加入 $S$ 集合的结点的所有出边执行松弛操作。

直到 $T$ 集合为空，算法结束。

### 时间复杂度

有多种方法来维护 1 操作中最短路长度最小的结点，不同的实现导致了 Dijkstra 算法时间复杂度上的差异。

- 暴力：不使用任何数据结构进行维护，每次 2 操作执行完毕后，直接在 $T$ 集合中暴力寻找最短路长度最小的结点。2 操作总时间复杂度为 $O(m)$，1 操作总时间复杂度为 $O(n^2)$，全过程的时间复杂度为 $O(n^2 + m) = O(n^2)$。
- 二叉堆：每成功松弛一条边 $(u,v)$，就将 $v$ 插入二叉堆中（如果 $v$ 已经在二叉堆中，直接修改相应元素的权值即可），1 操作直接取堆顶结点即可。共计 $O(m)$ 次二叉堆上的插入（修改）操作，$O(n)$ 次删除堆顶操作，而插入（修改）和删除的时间复杂度均为 $O(\log n)$，时间复杂度为 $O((n+m) \log n) = O(m \log n)$。
- 优先队列：和二叉堆类似，但使用优先队列时，如果同一个点的最短路被更新多次，因为先前更新时插入的元素不能被删除，也不能被修改，只能留在优先队列中，故优先队列内的元素个数是 $O(m)$ 的，时间复杂度为 $O(m \log m)$。
- Fibonacci 堆：和前面二者类似，但 Fibonacci 堆插入的时间复杂度为 $O(1)$，故时间复杂度为 $O(n \log n + m) = O(n \log n)$，时间复杂度最优。但因为 Fibonacci 堆较二叉堆不易实现，效率优势也不够大[^1]，算法竞赛中较少使用。
- 线段树：和二叉堆原理类似，不过将每次成功松弛后插入二叉堆的操作改为在线段树上执行单点修改，而 1 操作则是线段树上的全局查询最小值。时间复杂度为 $O(m \log n)$。

在稀疏图中，$m = O(n)$，使用二叉堆实现的 Dijkstra 算法较 Bellman-Ford 算法具有较大的效率优势；而在稠密图中，$m = O(n^2)$，这时候使用暴力做法较二叉堆实现更优。

### 正确性证明

下面用数学归纳法证明，在 **所有边权值非负** 的前提下，Dijkstra 算法的正确性[^2]。

简单来说，我们要证明的，就是在执行 1 操作时，取出的结点 $u$ 最短路均已经被确定，即满足 $D(u) = dis(u)$。

初始时 $S = \varnothing$，假设成立。

接下来用反证法。

设 $u$ 点为算法中第一个在加入 $S$ 集合时不满足 $D(u) = dis(u)$ 的点。因为 $s$ 点一定满足 $D(u)=dis(u)=0$，且它一定是第一个加入 $S$ 集合的点，因此将 $u$ 加入 $S$ 集合前，$S \neq \varnothing$，如果不存在 $s$ 到 $u$ 的路径，则 $D(u) = dis(u) = +\infty$，与假设矛盾。

于是一定存在路径 $s \to x \to y \to u$，其中 $y$ 为 $s \to u$ 路径上第一个属于 $T$ 集合的点，而 $x$ 为 $y$ 的前驱结点（显然 $x \in S$）。需要注意的是，可能存在 $s = x$ 或 $y = u$ 的情况，即 $s \to x$ 或 $y \to u$ 可能是空路径。

因为在 $u$ 结点之前加入的结点都满足 $D(u) = dis(u)$，所以在 $x$ 点加入到 $S$ 集合时，有 $D(x) = dis(x)$，此时边 $(x,y)$ 会被松弛，从而可以证明，将 $u$ 加入到 $S$ 时，一定有 $D(y)=dis(y)$。

下面证明 $D(u) = dis(u)$ 成立。在路径 $s \to x \to y \to u$ 中，因为图上所有边边权非负，因此 $D(y) \leq D(u)$。从而 $dis(y) \leq D(y) \leq D(u)\leq dis(u)$。但是因为 $u$ 结点在 1 过程中被取出 $T$ 集合时，$y$ 结点还没有被取出 $T$ 集合，因此此时有 $dis(u)\leq dis(y)$，从而得到 $dis(y) = D(y) = D(u) = dis(u)$，这与 $D(u)\neq dis(u)$ 的假设矛盾，故假设不成立。

因此我们证明了，1 操作每次取出的点，其最短路均已经被确定。命题得证。

注意到证明过程中的关键不等式 $D(y) \leq D(u)$ 是在图上所有边边权非负的情况下得出的。当图上存在负权边时，这一不等式不再成立，Dijkstra 算法的正确性将无法得到保证，算法可能会给出错误的结果。

### 代码实现

这里同时给出 $O(n^2)$ 的暴力做法实现和 $O(m \log m)$ 的优先队列做法实现。

> [!NOTE] **暴力实现**
```cpp
// C++ Version
struct edge {
    int v, w;
};
vector<edge> e[maxn];
int dis[maxn], vis[maxn];
void dijkstra(int n, int s) {
    memset(dis, 63, sizeof(dis));
    dis[s] = 0;
    for (int i = 1; i <= n; i++) {
        int u = 0, mind = 0x3f3f3f3f;
        for (int j = 1; j <= n; j++)
            if (!vis[j] && dis[j] < mind) u = j, mind = dis[j];
        vis[u] = true;
        for (auto ed : e[u]) {
            int v = ed.v, w = ed.w;
            if (dis[v] > dis[u] + w) dis[v] = dis[u] + w;
        }
    }
}
```

```python
    # Python Version
    class Edge:
        v = 0
        w = 0
    e = [[Edge() for i in range(maxn)] for j in range(maxn)]
    dis = [63] * maxn; vis = [] * maxn
    def dijkstra(n, s):
        dis[s] = 0
        for i in range(1, n + 1):
            u = 0; mind = 0x3f3f3f3f
            for j in range(1, n + 1):
                if vis[j] == False and dis[v] < mind:
                    u = j; mind = dis[j]
            vis[u] = True
            for ed in e[u]:
                v = ed.v; w = ed.w
                if dis[v] > dis[u] + w:
                    dis[v] = dis[u] + w
```

> [!NOTE] **优先队列实现**
```cpp
struct edge {
    int v, w;
};
struct node {
    int dis, u;
    bool operator>(const node& a) const { return dis > a.dis; }
};
vector<edge> e[maxn];
int dis[maxn], vis[maxn];
priority_queue<node, vector<node>, greater<node> > q;
void dijkstra(int n, int s) {
    memset(dis, 63, sizeof(dis));
    dis[s] = 0;
    q.push({0, s});
    while (!q.empty()) {
        int u = q.top().u;
        q.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        for (auto ed : e[u]) {
            int v = ed.v, w = ed.w;
            if (dis[v] > dis[u] + w) {
                dis[v] = dis[u] + w;
                q.push({dis[v], v});
            }
        }
    }
}
```

## Johnson 全源最短路径算法

Johnson 和 Floyd 一样，是一种能求出无负环图上任意两点间最短路径的算法。该算法在 1977 年由 Donald B. Johnson 提出。

任意两点间的最短路可以通过枚举起点，跑 $n$ 次 Bellman-Ford 算法解决，时间复杂度是 $O(n^2m)$ 的，也可以直接用 Floyd 算法解决，时间复杂度为 $O(n^3)$。

注意到堆优化的 Dijkstra 算法求单源最短路径的时间复杂度比 Bellman-Ford 更优，如果枚举起点，跑 $n$ 次 Dijkstra 算法，就可以在 $O(nm\log m)$（取决于 Dijkstra 算法的实现）的时间复杂度内解决本问题，比上述跑 $n$ 次 Bellman-Ford 算法的时间复杂度更优秀，在稀疏图上也比 Floyd 算法的时间复杂度更加优秀。

但 Dijkstra 算法不能正确求解带负权边的最短路，因此我们需要对原图上的边进行预处理，确保所有边的边权均非负。

一种容易想到的方法是给所有边的边权同时加上一个正数 $x$，从而让所有边的边权均非负。如果新图上起点到终点的最短路经过了 $k$ 条边，则将最短路减去 $kx$ 即可得到实际最短路。

但这样的方法是错误的。考虑下图：

![](./images/shortest-path1.svg)

$1 \to 2$ 的最短路为 $1 \to 5 \to 3 \to 2$，长度为 $−2$。

但假如我们把每条边的边权加上 $5$ 呢？

![](./images/shortest-path2.svg)

新图上 $1 \to 2$ 的最短路为 $1 \to 4 \to 2$，已经不是实际的最短路了。

Johnson 算法则通过另外一种方法来给每条边重新标注边权。

我们新建一个虚拟节点（在这里我们就设它的编号为 $0$）。从这个点向其他所有点连一条边权为 $0$ 的边。

接下来用 Bellman-Ford 算法求出从 $0$ 号点到其他所有点的最短路，记为 $h_i$。

假如存在一条从 $u$ 点到 $v$ 点，边权为 $w$ 的边，则我们将该边的边权重新设置为 $w+h_u-h_v$。

接下来以每个点为起点，跑 $n$ 轮 Dijkstra 算法即可求出任意两点间的最短路了。

一开始的 Bellman-Ford 算法并不是时间上的瓶颈，若使用 `priority_queue` 实现 Dijkstra 算法，该算法的时间复杂度是 $O(nm\log m)$。

### 正确性证明

为什么这样重新标注边权的方式是正确的呢？

在讨论这个问题之前，我们先讨论一个物理概念——势能。

诸如重力势能，电势能这样的势能都有一个特点，势能的变化量只和起点和终点的相对位置有关，而与起点到终点所走的路径无关。

势能还有一个特点，势能的绝对值往往取决于设置的零势能点，但无论将零势能点设置在哪里，两点间势能的差值是一定的。

接下来回到正题。

在重新标记后的图上，从 $s$ 点到 $t$ 点的一条路径 $s \to p_1 \to p_2 \to \dots \to p_k \to t$ 的长度表达式如下：

$(w(s,p_1)+h_s-h_{p_1})+(w(p_1,p_2)+h_{p_1}-h_{p_2})+ \dots +(w(p_k,t)+h_{p_k}-h_t)$

化简后得到：

$w(s,p_1)+w(p_1,p_2)+ \dots +w(p_k,t)+h_s-h_t$

无论我们从 $s$ 到 $t$ 走的是哪一条路径，$h_s-h_t$ 的值是不变的，这正与势能的性质相吻合！

为了方便，下面我们就把 $h_i$ 称为 $i$ 点的势能。

上面的新图中 $s \to t$ 的最短路的长度表达式由两部分组成，前面的边权和为原图中 $s \to t$ 的最短路，后面则是两点间的势能差。因为两点间势能的差为定值，因此原图上 $s \to t$ 的最短路与新图上 $s \to t$ 的最短路相对应。

到这里我们的正确性证明已经解决了一半——我们证明了重新标注边权后图上的最短路径仍然是原来的最短路径。接下来我们需要证明新图中所有边的边权非负，因为在非负权图上，Dijkstra 算法能够保证得出正确的结果。

根据三角形不等式，图上任意一边 $(u,v)$ 上两点满足：$h_v \leq h_u + w(u,v)$。这条边重新标记后的边权为 $w'(u,v)=w(u,v)+h_u-h_v \geq 0$。这样我们证明了新图上的边权均非负。

这样，我们就证明了 Johnson 算法的正确性。

* * *

## 不同方法的比较

| 最短路算法    | Floyd      | Bellman-Ford | Dijkstra     | Johnson       |
| -------- | ---------- | ------------ | ------------ | ------------- |
| 最短路类型    | 每对结点之间的最短路 | 单源最短路        | 单源最短路        | 每对结点之间的最短路    |
| 作用于      | 任意图        | 任意图          | 非负权图         | 任意图           |
| 能否检测负环？  | 能          | 能            | 不能           | 能             |
| 推荐作用图的大小 | 小          | 中/小          | 大/中          | 大/中           |
| 时间复杂度    | $O(N^3)$   | $O(NM)$      | $O(M\log M)$ | $O(NM\log M)$ |

注：表中的 Dijkstra 算法在计算复杂度时均用 `priority_queue` 实现。

## 输出方案

开一个 `pre` 数组，在更新距离的时候记录下来后面的点是如何转移过去的，算法结束前再递归地输出路径即可。

比如 Floyd 就要记录 `pre[i][j] = k;`，Bellman-Ford 和 Dijkstra 一般记录 `pre[v] = u`。

## 参考资料与注释

[^1]: [Worst case of fibonacci heap - Wikipedia](https://en.wikipedia.org/wiki/Fibonacci_heap#Worst_case)

[^2]: 《算法导论（第 3 版中译本）》，机械工业出版社，2013 年，第 384 - 385 页。

## 习题

> [!NOTE] **[AcWing 849. Dijkstra求最短路 I](https://www.acwing.com/problem/content/851/)**
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

const int N = 510;

int n, m;
int g[N][N];
int dist[N];
bool st[N];

int dijkstra() {
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    for (int i = 0; i < n - 1; i++) {
        int t = -1;
        for (int j = 1; j <= n; j++)
            if (!st[j] && (t == -1 || dist[t] > dist[j])) t = j;

        for (int j = 1; j <= n; j++) dist[j] = min(dist[j], dist[t] + g[t][j]);

        st[t] = true;
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

int main() {
    scanf("%d%d", &n, &m);

    memset(g, 0x3f, sizeof g);
    while (m--) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);

        g[a][b] = min(g[a][b], c);
    }

    printf("%d\n", dijkstra());

    return 0;
}
```

##### **Python**

```python
"""
> 迪杰斯特拉算法的核心在于：每次用当前d最小的节点去更新别的节点
- 【朴素Dijkstra算法】
  - 每次暴力循环找距离最近的点
  - 先定义d[1]=0, d[i]=float('inf')
  - for循环遍历:
    - 集合s：当前已经确定最短路的点
    - 循环n次： for i ：n，迭代找到不在s中的距离最近的点t，把t也加入到s中（st[i]=True）
  - 然后用这个点开始更新一下它到其他点的距离
> 朴素Dijkstra算法适用于稠密图，图用邻接矩阵来存储；

- 【堆优化版的Dijkstra算法】
  - 用【堆】维护所有点到起点的距离
  - 不用每次O(N)去找最小的d节点，而是用一个堆维护，这样复杂度可以降到O(mlogn)
  - 基本思路：
    - 首先初始化起点的距离为0，其余点的距离为无穷
    - 将起点加入到优先队列，优先队列维护最小值
    - 根据堆顶元素的权值和它能到达的点，去更新其他点的距离，然后将更新的点加入到队列中
"""


# 朴素dijkstra写法，适合稠密图===>用邻接矩阵存储图
# 时间复杂度：O(n*n)
# 只能处理边权为正数的问题，没有用堆优化：每次都暴力循环找距离最近的点。

def dijkstra():
    d[1] = 0
    # 迭代n次，每一次迭代都是确定当前还没有确定最短路点中的最短的那个


for i in range(n):
    t = -1  # t=-1表示本次循环中的最短路的点还没有确定
    for j in range(1, n + 1):
        # 如果当前这个点 还没有确定最短路，并且t==-1或者d[t]>d[j]：表示当前的t不是最短的
        if not st[j] and (t == -1 or d[t] > d[j]):
            t = j
    st[t] = True  # 把t加入到已经处理完的集合里
    # 用t来更新其他点的距离
    # 注意：邻接矩阵遍历出边的方式和邻接表遍历出边的方式不一样。
    # 邻接矩阵直接for循环遍历1～n个点，更新距离即可。
    for j in range(1, n + 1):
        d[j] = min(d[j], d[t] + g[t][j])
if d[n] == float('inf'):
    print('-1')
else:
    print(d[n])

# 稀疏图：用邻接表，堆优化的dijkstra算法 
# Dijkstra+heap优化 ： O(mlogn)
# 用堆维护所有点到起点的距离，适用于稀疏图===>用邻接表存储,只能处理边权为正数的问题
# 堆可以直接用STL中的heapq（python里的包）

N = 100010
M = 2 * N
h = [-1] * N
ev = [0] * M
ne = [0] * M
w = [0] * M
idx = 0
st = [False] * N
d = [float('inf')] * N  # 存储所有点到1号点的距离


def add_edge(a, b, c):
    global idx
    ev[idx] = b
    w[idx] = c
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def dijkstra():
    d[1] = 0
    import heapq
    q = []  # 初始化小根堆；维护距离的时候，需要知道节点编号，所以堆q里存储的是一个pair
    heapq.heappush(q, [0, 1])  # 距离为0，节点编号是1
    while q:
        dist, ver = heapq.heappop(q)
        if st[ver]: continue
        st[ver] = True
        i = h[ver]
        while i != -1:
            j = ev[i]
            if d[j] > dist + w[i]:
                d[j] = dist + w[i]
                heapq.heappush(q, [d[j], j])
            i = ne[i]
    if d[n] == float('inf'):
        return -1
    else:
        return d[n]


if __name__ == '__main__':
    n, m = map(int, input().split())
    for _ in range(m):
        a, b, c = map(int, input().split())
        add_edge(a, b, c)
    print(dijkstra())
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 850. Dijkstra求最短路 II](https://www.acwing.com/problem/content/852/)**
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
#include <queue>

using namespace std;

typedef pair<int, int> PII;

const int N = 1e6 + 10;

int n, m;
int h[N], w[N], e[N], ne[N], idx;
int dist[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int dijkstra() {
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});

    while (heap.size()) {
        auto t = heap.top();
        heap.pop();

        int ver = t.second, distance = t.first;

        if (st[ver]) continue;
        st[ver] = true;

        for (int i = h[ver]; i != -1; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[ver] + w[i]) {
                dist[j] = dist[ver] + w[i];
                heap.push({dist[j], j});
            }
        }
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

int main() {
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);
    while (m--) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    cout << dijkstra() << endl;

    return 0;
}
```

##### **Python**

```python
"""
> 迪杰斯特拉算法的核心在于：**每次用当前d最小的节点去更新别的节点**

- 朴素Dijkstra算法
  - 每次暴力循环找距离最近的点
  - 先定义d[1]=0, d[i]=float('inf')
  - for循环遍历:
    - 集合s：当前已经确定最短路的点
    - 循环n次： for i ：n，迭代找到不在s中的距离最近的点t，把t也加入到s中（st[i]=True）
  - 然后用这个点开始更新一下它到其他点的距离

> 朴素Dijkstra算法适用于稠密图，图用邻接矩阵来存储；

- 堆优化版的Dijkstra算法
  - 用【堆】维护所有点到起点的距离
  - 不用每次O(N)去找最小的d节点，而是用一个堆维护，这样复杂度可以降到O(mlogn)
  - 基本思路：
    - 首先初始化起点的距离为0，其余点的距离为无穷
    - 将起点加入到优先队列，优先队列维护最小值
    - 根据堆顶元素的权值和它能到达的点，去更新其他点的距离，然后将更新的点加入到队列中
"""


# 朴素dijkstra写法，适合稠密图===>用邻接矩阵存储图
# 时间复杂度：O(n*n)
# 只能处理边权为正数的问题，没有用堆优化：每次都暴力循环找距离最近的点。

def dijkstra():
    d[1] = 0
    # 迭代n次，每一次迭代都是确定当前还没有确定最短路点中的最短的那个


for i in range(n):
    t = -1  # t=-1表示本次循环中的最短路的点还没有确定
    for j in range(1, n + 1):
        # 如果当前这个点 还没有确定最短路，并且t==-1或者d[t]>d[j]：表示当前的t不是最短的
        if not st[j] and (t == -1 or d[t] > d[j]):
            t = j
    st[t] = True  # 把t加入到已经处理完的集合里
    # 用t来更新其他点的距离
    # 注意：邻接矩阵遍历出边的方式和邻接表遍历出边的方式不一样。
    # 邻接矩阵直接for循环遍历1～n个点，更新距离即可。
    for j in range(1, n + 1):
        d[j] = min(d[j], d[t] + g[t][j])
if d[n] == float('inf'):
    print('-1')
else:
    print(d[n])

# 稀疏图：用邻接表，堆优化的dijkstra算法 
# Dijkstra+heap优化 ： O(mlogn)
# 用堆维护所有点到起点的距离，适用于稀疏图===>用邻接表存储,只能处理边权为正数的问题
# 堆可以直接用STL中的heapq（python里的包）

N = 100010
M = 2 * N
h = [-1] * N
ev = [0] * M
ne = [0] * M
w = [0] * M
idx = 0
st = [False] * N
d = [float('inf')] * N  # 存储所有点到1号点的距离


def add_edge(a, b, c):
    global idx
    ev[idx] = b
    w[idx] = c
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def dijkstra():
    d[1] = 0
    import heapq
    q = []  # 初始化小根堆；维护距离的时候，需要知道节点编号，所以堆q里存储的是一个pair
    heapq.heappush(q, [0, 1])  # 距离为0，节点编号是1
    while q:
        dist, ver = heapq.heappop(q)
        if st[ver]: continue
        st[ver] = True
        i = h[ver]
        while i != -1:
            j = ev[i]
            if d[j] > dist + w[i]:
                d[j] = dist + w[i]
                heapq.heappush(q, [d[j], j])
            i = ne[i]
    if d[n] == float('inf'):
        return -1
    else:
        return d[n]


if __name__ == '__main__':
    n, m = map(int, input().split())
    for _ in range(m):
        a, b, c = map(int, input().split())
        add_edge(a, b, c)
    print(dijkstra())
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 853. 有边数限制的最短路](https://www.acwing.com/problem/content/855/)**
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

const int N = 510, M = 10010;

struct Edge {
    int a, b, c;
} edges[M];

int n, m, k;
int dist[N];
int last[N];

void bellman_ford() {
    memset(dist, 0x3f, sizeof dist);

    dist[1] = 0;
    for (int i = 0; i < k; i++) {
        memcpy(last, dist, sizeof dist);
        for (int j = 0; j < m; j++) {
            auto e = edges[j];
            dist[e.b] = min(dist[e.b], last[e.a] + e.c);
        }
    }
}

int main() {
    scanf("%d%d%d", &n, &m, &k);

    for (int i = 0; i < m; i++) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        edges[i] = {a, b, c};
    }

    bellman_ford();

    if (dist[n] > 0x3f3f3f3f / 2)
        puts("impossible");
    else
        printf("%d\n", dist[n]);

    return 0;
}
```

##### **Python**

```python
"""
- Bellman-Ford算法
  - 与迪杰斯特拉算法最大的不同是每次都是从源点s重新出发进行“松弛”更新操作，而Dijkstra算法则是从源点出发向外扩逐个处理相邻的节点，不会去重复处理节点。（这里也可以看出Dijkstra算法效率更高）
  - 本算法可以处理负权边的问题，并且可以限定边数!!!（特点）
  - 核心操作：
    - 迭代k次(for k次)：迭代k次，表示经过1号点走过不超过k条边的最短距离；如果迭代n次的时候还有更新，说明存在负环
    - for 所有边： 遍历所有边m：a b w;d[b]=min(d[b],d[a]+w[i])
    - 在遍历了n次之后保证了d[b]<=d[a]+w[i]（三角不等式）
  - 在处理之前，需要对边进行备份：由于有k条边数的限制，所以需要有备份；并且还可能发生“串联”，保证每次更新的时候 都只用上一次迭代的结果，就可以不发生串联。
  - 有边数限制的题（从1号点到n号点最多不经过k条边的最短路）只能BF算法；只要没有负环，就可以用 SPFA，绝大多数的最短路问题不存在负环。
  - 每次循环只会迭代更新一条边。

> BF算法的限制很少，不需要用邻接矩阵或者邻接表存储，可以直接用结构体存储图。
"""


# BF算法的存图方式很简单，开一个结构体数组就可以了==> 在python中直接用一个list，每个下标里存储三个数。
# 算法流程：循环n次，每次循环所有边，d[b]=min(d[b],d[a]+w) ===>更新的过程叫：松弛操作 
# 循环n次后，所有边都满足d[b]<=d[a]+w（三角不等式）
##如果有负权回路，那最短路不一定会存在。
# BF算法可以求出来 是否存在最短路。（但时间复杂度高，所以一般用SPFA算法）

def bellman_ford():
    d[1] = 0
    # 遍历k次
    for i in range(k):
        backup = d[:]  # 注意：很容易忘了要备份距离！
        # 遍历m条边
        for j in range(m):
            a = q[j][0]
            b = q[j][1]
            w = q[j][2]
            if d[b] > backup[a] + w:
                d[b] = backup[a] + w
    if d[n] == float("inf"):
        print("impossible")
    else:
        print(d[n])


if __name__ == '__main__':
    N = 510
    M = 10010
    d = [float('inf')] * N
    backup = [float('inf')] * N
    q = []

    n, m, k = map(int, input().split())
    for _ in range(m):
        q.append(list(map(int, input().split())))
    bellman_ford()
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 851. spfa求最短路](https://www.acwing.com/problem/content/853/)**
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
#include <queue>

using namespace std;

const int N = 100010;

int n, m;
int h[N], w[N], e[N], ne[N], idx;
int dist[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int spfa() {
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    queue<int> q;
    q.push(1);
    st[1] = true;

    while (q.size()) {
        int t = q.front();
        q.pop();

        st[t] = false;

        for (int i = h[t]; i != -1; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[t] + w[i]) {
                dist[j] = dist[t] + w[i];
                if (!st[j]) {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    return dist[n];
}

int main() {
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);

    while (m--) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    int t = spfa();

    if (t == 0x3f3f3f3f)
        puts("impossible");
    else
        printf("%d\n", t);

    return 0;
}
```

##### **Python**

```python
"""
- SPFA算法
  - 由BF算法优化来，在BF算法中，遍历所有边的时候 d[b]=min(d[a]+c) ： 这个表达式只有d[a]变小的时候 d[b]的值才会变小。所有SPFA将d[a]变小的点加入到队列，并更新他们所连接的边，可以省去无用的迭代。
  - 用宽搜进行优化上述步骤。
  - SPFA算法很推荐，正权图也可以用；但是可能会被卡时间，如果被卡时间了，就换。
  - 核心操作：
    - 先将节点1入队
    - 将d[a]变小的节点放入队列中，取出队头，把队头删掉后，用t更新以t为起点的出边
    - 更新成功后，把点b加入队列，判断，如果队列有b 则不加入
  - 和Dijkstra堆优化版本算法的不同在于：SPFA可能会多次遍历一个点，可能会存在重复处理节点的操作；导致代码上 会有st[j]=True 然后pop出来之后又会将st[j]=False

> SPFA算法用邻接表存储图；正权图也可以优先用SPFA，如果被卡时间了，再换迪杰斯特拉算法。
"""


def spfa():
    from collections import deque
    d[1] = 0
    q = deque()
    q.append(1)
    st[1] = True
    while q:
        t = q.popleft()
        st[t] = False
        i = h[t]
        while i != -1:
            j = ev[i]
            if d[j] > d[t] + w[i]:
                d[j] = d[t] + w[i]
                if not st[j]:
                    q.append(j)
                    st[j] = True
            i = ne[i]
    if d[n] == float('inf'):
        print("impossible")
    else:
        print(d[n])


def add_edge(a, b, c):
    global idx
    ev[idx] = b
    w[idx] = c
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


if __name__ == '__main__':
    N = 100010
    M = 2 * N
    h = [-1] * N
    ev = [0] * M
    ne = [0] * M
    w = [0] * M
    idx = 0

    d = [float("inf")] * N  # 存储所有点到1号点的距离
    st = [False] * N  # 存储的是距离变短，导致后面的点会被更新的点

    n, m = map(int, input().split())
    for _ in range(m):
        a, b, c = map(int, input().split())
        add_edge(a, b, c)

    spfa()
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 852. spfa判断负环](https://www.acwing.com/problem/content/854/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 2010, M = 10010;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
int dist[N], cnt[N];

int q[N * N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

bool spfa() {
    int hh = 0, tt = -1;
    for (int i = 1; i <= n; ++ i )
        q[ ++ tt ] = i, st[i] = true;
        
    while (hh <= tt) {
        int t = q[hh ++ ];
        st[t] = false;
        
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[t] + w[i]) {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                
                if (cnt[j] >= n) return true;
                if (!st[j]) q[ ++ tt ] = j, st[j] = true;
            }
        }
    }
    return false;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    while (m -- ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c);
    }
    
    if (spfa()) cout << "Yes" << endl;
    else cout << "No" << endl;
    
    return 0;
}
```

##### **Python**

```python
# 应用抽屉原理，判断是否存在负环: 统计当前每个点的最短路中所包含的边数，如果某点的最短路所包含的边数大于等于n，则也说明存在环
# 用d[x]表示1～x的最短距离，cnt[x]：表示当前最短路边的数量

# 统计当前每个点的最短路中所包含的边数，如果某点的最短路所包含的边数大于等于n，则也说明存在环

# 注意：需要从每个点都出发一次，才能完全确定此图中是否有环；
# cnt[j]=n: 表示编号为j的节点 是第n个加入到路径的节点。
def spfa():
    from collections import deque
    q = deque()
    d[1] = 0
    # 题意是不是存在负环，而这个负环可能从一号点到不了
    # 所以 把所有的点都放到队列里。
    for i in range(1, n + 1):
        q.append(i)
        st[i] = True
    while q:
        t = q.popleft()
        st[t] = False
        i = h[t]
        while i != -1:
            j = ev[i]
            if d[j] > d[t] + w[i]:
                d[j] = d[t] + w[i]
                cnt[j] = cnt[t] + 1
                if cnt[j] >= n:
                    return True

                if not st[j]:
                    q.append(j)
                    st[j] = True
            i = ne[i]
    else:
        return False


def add_edge(a, b, c):
    global idx
    ev[idx] = b
    w[idx] = c
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


if __name__ == '__main__':
    N = 100010
    M = 2 * N
    h = [-1] * N
    ev = [0] * M
    ne = [0] * M
    w = [0] * M
    idx = 0

    d = [0] * N  # 注意！！这里不能用float("inf")，因为在比较d距离时，d[i]+w还是==float("inf")，因此不能更新距离；d[x] 表示 当前x点到1号点的最短距离。
    st = [False] * N
    cnt = [0] * N  # 当前最短路的边的个数

    n, m = map(int, input().split())
    for _ in range(m):
        a, b, c = map(int, input().split())
        add_edge(a, b, c)

    res = spfa()
    if res:
        print("Yes")
    else:
        print("No")
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 854. Floyd求最短路](https://www.acwing.com/problem/content/856/)**
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

const int N = 210, INF = 1e9;

int n, m, Q;
int d[N][N];

void floyd() {
    for (int k = 1; k <= n; k++)
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
}

int main() {
    scanf("%d%d%d", &n, &m, &Q);

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            if (i == j)
                d[i][j] = 0;
            else
                d[i][j] = INF;

    while (m--) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        d[a][b] = min(d[a][b], c);
    }

    floyd();

    while (Q--) {
        int a, b;
        scanf("%d%d", &a, &b);

        int t = d[a][b];
        if (t > INF / 2)
            puts("impossible");
        else
            printf("%d\n", t);
    }

    return 0;
}
```

##### **Python**

```python
"""
Floyd算法

- 用邻接图存储图
- 标准的佛洛依德算法，三重循环。循环结束之后，d[i][j]存储的是 **点i--->点j** 的最短距离
- 需要注意：循环顺序不能变：第一层枚举中间点，第二层和第三层枚举起点和终点：
  - 初始化d
  - 用k,i,j 去更新d
- 可以处理负权，但是不能存在负回路。
"""

# f[i, j, k]表示从i走到j的路径上除i和j点外只经过1到k的点的所有路径的最短距离。
# 那么f[i, j, k] = min(f[i, j, k - 1), f[i, k, k - 1] + f[k, j, k - 1]。
# 因此在计算第k层的f[i, j]的时候必须先将第k - 1层的所有状态计算出来，所以需要把k放在最外层。

# 邻接矩阵存图；
if __name__ == '__main__':
    N = 210
    n, m, q = map(int, input().split())
    g = [[float('inf')] * (n + 1) for _ in range(n + 1)]  # g[][]存储图的邻接矩阵
    d = [[0] * N for _ in range(N)]

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                d[i][j] = 0
            else:
                d[i][j] = float('inf')

    for _ in range(m):
        a, b, w = map(int, input().split())
        d[a][b] = min(d[a][b], w)  # 存在重边和自回路

    # floyd算法核心
    # 基于动态规划实现的：d[k,i,j]:从i点出发，只经过1-k这些中间点 到j的最短距离。
    # 注意：一定要先循环k
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    for _ in range(q):
        a, b = map(int, input().split())
        if d[a][b] == float('inf'):
            print('impossible')
        else:
            print(d[a][b])
```

<!-- tabs:end -->
</details>

<br>

* * *

### dijkstra 进阶

> [!NOTE] **[AcWing 920. 最优乘车](https://www.acwing.com/problem/content/922/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **学习STL处理输入流**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;

const int N = 510, M = N * N, INF = 0x3f3f3f3f;

int m, n;
int h[N], e[M], w[M], ne[M], idx;
int d[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dijkstra() {
    memset(d, 0x3f, sizeof d);
    memset(st, 0, sizeof st);
    
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});
    d[1] = 0;
    
    while (heap.size()) {
        auto [dis, ver] = heap.top(); heap.pop();
        if (st[ver]) continue;
        st[ver] = true;
        
        for (int i = h[ver]; ~i; i = ne[i]) {
            int j = e[i];
            if (d[j] > dis + w[i]) {
                d[j] = dis + w[i];
                heap.push({d[j], j});
            }
        }
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> m >> n;
    
    string line;
    getline(cin, line);  // \r
    while (m -- ) {
        getline(cin, line);
        stringstream ss(line);
        int p, cnt;
        vector<int> stop;
        while (ss >> p) stop.push_back(p);
        cnt = stop.size();
        
        for (int i = 0; i < cnt; ++ i )
            for (int j = i + 1; j < cnt; ++ j )
                add(stop[i], stop[j], 1);
    }
    
    dijkstra();
    
    if (d[n] == INF) cout << "NO" << endl;
    else cout << max(d[n] - 1, 0) << endl;
    
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

> [!NOTE] **[AcWing 903. 昂贵的聘礼](https://www.acwing.com/problem/content/description/905/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> WA 好几次因为没有在 dijkstra 内部初始化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;

const int N = 110, INF = 0x3f3f3f3f;

int n, m;
int w[N][N], l[N];
int d[N];
bool st[N];

int dijkstra(int down, int up) {
    memset(d, 0x3f, sizeof d);
    memset(st, 0, sizeof st);
    
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    d[0] = 0;
    heap.push({0, 0});
    while (!heap.empty()) {
        auto [dis, ver] = heap.top(); heap.pop();
        if (st[ver]) continue;
        st[ver] = true;
        
        for (int v = 1; v <= n; ++ v )
            if (l[v] >= down && l[v] <= up && d[v] > dis + w[ver][v]) {
                d[v] = dis + w[ver][v];
                heap.push({d[v], v});
            }
    }
    return d[1];
}

int main() {
    cin >> m >> n;
    
    memset(w, 0x3f, sizeof w);
    for (int i = 1; i <= n; ++ i ) w[i][i] = 0;
    
    for (int i = 1; i <= n; ++ i ) {
        int price, cnt;
        cin >> price >> l[i] >> cnt;
        w[0][i] = min(price, w[0][i]);
        while (cnt -- ) {
            int id, cost;
            cin >> id >> cost;
            w[id][i] = min(w[id][i], cost);
        }
    }
    
    int res = INF;
    for (int i = l[1] - m; i <= l[1]; ++ i )
        res = min(res, dijkstra(i, i + m));
        
    cout << res << endl;
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

### 单源最短路综合应用

> [!NOTE] **[AcWing 1135. 新年好](https://www.acwing.com/problem/content/1137/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 从 5 个亲戚起点各自跑最短路 然后 dfs 所有先后顺序排列组合

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>

using namespace std;

typedef pair<int, int> PII;

const int N = 50010, M = 200010, INF = 0x3f3f3f3f;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
int q[N], dist[6][N];
int source[6];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

void dijkstra(int start, int dist[]) {
    memset(dist, 0x3f, N * 4);
    dist[start] = 0;
    memset(st, 0, sizeof st);

    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, start});

    while (heap.size()) {
        auto t = heap.top();
        heap.pop();

        int ver = t.second;
        if (st[ver]) continue;
        st[ver] = true;

        for (int i = h[ver]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[ver] + w[i]) {
                dist[j] = dist[ver] + w[i];
                heap.push({dist[j], j});
            }
        }
    }
}

int dfs(int u, int start, int distance) {
    if (u > 5) return distance;

    int res = INF;
    for (int i = 1; i <= 5; i++)
        if (!st[i]) {
            int next = source[i];
            st[i] = true;
            res = min(res, dfs(u + 1, i, distance + dist[start][next]));
            st[i] = false;
        }

    return res;
}

int main() {
    scanf("%d%d", &n, &m);
    source[0] = 1;
    for (int i = 1; i <= 5; i++) scanf("%d", &source[i]);

    memset(h, -1, sizeof h);
    while (m--) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c), add(b, a, c);
    }

    for (int i = 0; i < 6; i++) dijkstra(source[i], dist[i]);

    memset(st, 0, sizeof st);
    printf("%d\n", dfs(1, 0, 0));

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

> [!NOTE] **[AcWing 340. 通信线路](https://www.acwing.com/problem/content/342/)**
> 
> 题意: 使线路上的最长边最小

> [!TIP] **思路**
> 
> 1. 最大值最小 可以用二分做
> 
>  ====> 如果线路上有大于它的，就需要消耗掉一条免费的，只要最后消耗的小于k就行了。
> 
> 2. 分层最短路 TODO
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <deque>
#include <iostream>

using namespace std;

const int N = 1010, M = 20010;

int n, m, k;
int h[N], e[M], w[M], ne[M], idx;
int dist[N];
deque<int> q;
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

bool check(int bound) {
    memset(dist, 0x3f, sizeof dist);
    memset(st, 0, sizeof st);

    q.push_back(1);
    dist[1] = 0;

    while (q.size()) {
        int t = q.front();
        q.pop_front();

        if (st[t]) continue;
        st[t] = true;

        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i], x = w[i] > bound;
            if (dist[j] > dist[t] + x) {
                dist[j] = dist[t] + x;
                if (!x)
                    q.push_front(j);
                else
                    q.push_back(j);
            }
        }
    }

    return dist[n] <= k;
}

int main() {
    cin >> n >> m >> k;
    memset(h, -1, sizeof h);
    while (m--) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }

    int l = 0, r = 1e6 + 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (check(mid))
            r = mid;
        else
            l = mid + 1;
    }

    if (r == 1e6 + 1)
        cout << -1 << endl;
    else
        cout << r << endl;

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

> [!NOTE] **[AcWing 342. 道路与航线](https://www.acwing.com/problem/content/344/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 团内部dijkstra 团与团直接topo

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;

const int N = 25010, M = 150010, INF = 0x3f3f3f3f;

int n, r, p, s;
int id[N];  //
int bcnt, din[N];
vector<int> block[N];
queue<int> q;

int h[N], e[M], w[M], ne[M], idx;
int d[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u, int bid) {
    id[u] = bid, block[bid].push_back(u);
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!id[j])
            dfs(j, bid);
    }
}

void dijkstra(int bid) {
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    
    for (auto u : block[bid])
        heap.push({d[u], u});
        
    while (heap.size()) {
        auto [dis, ver] = heap.top(); heap.pop();
        if (st[ver]) continue;
        st[ver] = true;
        
        for (int i = h[ver]; ~i; i = ne[i]) {
            int j = e[i];
            if (id[j] != id[ver] && -- din[id[j]] == 0) q.push(id[j]);
            if (d[j] > dis + w[i]) {
                d[j] = dis + w[i];
                if (id[j] == id[ver]) heap.push({d[j], j});
            }
        }
    }
}

void toposort() {
    memset(d, 0x3f, sizeof d);
    memset(st, 0, sizeof st);
    d[s] = 0;
    
    for (int i = 1; i <= bcnt; ++ i )
        if (!din[i])
            q.push(i);

    while (q.size()) {
        int t = q.front(); q.pop();
        dijkstra(t);
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> r >> p >> s;
    
    int a, b, c;
    while (r -- ) {
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    
    for (int i = 1; i <= n; ++ i )
        if (!id[i]) {
            bcnt ++ ;
            dfs(i, bcnt);
        }
    
    
    while (p -- ) {
        cin >> a >> b >> c;
        din[id[b]] ++ ;
        add(a, b, c);
    }
    
    toposort();
    
    for (int i = 1; i <= n; ++ i )
        if (d[i] > INF / 2) cout << "NO PATH" << endl;
        else cout << d[i] << endl;
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

> [!NOTE] **[AcWing 341. 最优贸易](https://www.acwing.com/problem/content/343/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 先求出：
> 
> - 从 1 走到 i 的过程中，买入水晶球的最低价格 dmin[i]；
> 
> - 从 i 走到 n 的过程中，卖出水晶球的最高价格 dmax[i]；
> 
> 然后枚举每个城市作为买卖的中间城市，求出 dmax[i] - dmin[i] 的最大值即可。
> 
> ==> 因此 需要反向边
> 
> **可能有环 所以只能 spfa**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010, M = 2000010, INF = 0x3f3f3f3f;

int n, m;
int w[N];
int hs[N], ht[N], e[M], ne[M], idx;
int dmin[N], dmax[N];
bool st[N];

int q[N];

void add(int h[], int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void spfa(int h[], int d[], int type) {
    int hh = 0, tt = 0;                 // ATTENTION 1 : 循环队列
    if (type == 0) {
        memset(d, 0x3f, sizeof dmin);   // ATTENTION 2 : 不能用 sizeof d    可以用 memset(d, 0x3f, N * 4)
        q[tt ++ ] = 1;
        d[1] = w[1];
    } else {
        memset(d, 0xcf, sizeof dmax);
        q[tt ++ ] = n;
        d[n] = w[n];
    }
    
    while (hh != tt) {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        
        st[t] = false;
        
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (type == 0 && d[j] > min(d[t], w[j]) ||
                type == 1 && d[j] < max(d[t], w[j])
            ) {
                if (type == 0) d[j] = min(d[t], w[j]);
                else d[j] = max(d[t], w[j]);
                
                if (!st[j]) {
                    q[tt ++ ] = j;
                    if (tt == N) tt = 0;
                    st[j] = true;
                }
            }
        }
    }
}

int main() {
    memset(hs, -1, sizeof hs);
    memset(ht, -1, sizeof ht);
    
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) cin >> w[i];
    
    while (m -- ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(hs, a, b), add(ht, b, a);
        if (c == 2) add(hs, b, a), add(ht, a, b);
    }
    
    spfa(hs, dmin, 0);
    spfa(ht, dmax, 1);
    
    int res = 0;
    for (int i = 1; i <= n; ++ i )
        res = max(res, dmax[i] - dmin[i]);
    cout << res << endl;
    
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


### 单元最短路扩展(计数)

> [!NOTE] **[AcWing 1134. 最短路计数](https://www.acwing.com/problem/content/1136/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 最短路有多少个
> 
> **数学推导**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;

const int N = 100010, M = 400010, mod = 100003;

int n, m;
int h[N], e[M], ne[M], idx;
int d[N], cnt[N];
bool st[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dijkstra() {
    memset(d, 0x3f, sizeof d);
    memset(cnt, 0, sizeof cnt);
    memset(st, 0, sizeof st);
    
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});
    d[1] = 0, cnt[1] = 1;
    
    while (heap.size()) {
        auto [dis, ver] = heap.top(); heap.pop();
        if (st[ver]) continue;
        st[ver] = true;
        
        for (int i = h[ver]; ~i; i = ne[i]) {
            int j = e[i];
            if (d[j] > dis + 1) {
                d[j] = dis + 1;
                cnt[j] = cnt[ver];
                heap.push({d[j], j});
            } else if (d[j] == dis + 1) {
                cnt[j] = (cnt[j] + cnt[ver]) % mod;
            }
        }
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    dijkstra();
    
    for (int i = 1; i <= n; ++ i )
        cout << cnt[i] << endl;
        
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

> [!NOTE] **[AcWing 383. 观光](https://www.acwing.com/problem/content/385/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 最短路 + (最短路+1) 个数统计
> 
> 注意写法
> 
> **后面周赛遇到过变形 加强理解**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using TIII = tuple<int, int, int>;

const int N = 1010, M = 20010;

int n, m, s, t;
int h[N], e[M], w[M], ne[M], idx;
int d[N][2], cnt[N][2];
bool st[N][2];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int dijkstra() {
    memset(d, 0x3f, sizeof d);
    memset(st, 0, sizeof st);
    memset(cnt, 0, sizeof cnt);
    
    priority_queue<TIII, vector<TIII>, greater<TIII>> heap;
    heap.push({0, s, 0});
    d[s][0] = 0, cnt[s][0] = 1;
    
    while (heap.size()) {
        auto [dis, ver, type] = heap.top(); heap.pop();
        if (st[ver][type]) continue;
        st[ver][type] = true;
        
        for (int i = h[ver]; ~i; i = ne[i]) {
            int j = e[i];
            if (d[j][0] > dis + w[i]) {
                d[j][1] = d[j][0], cnt[j][1] = cnt[j][0];
                heap.push({d[j][1], j, 1});
                d[j][0] = dis + w[i], cnt[j][0] = cnt[ver][type];
                heap.push({d[j][0], j, 0});
            } else if (d[j][0] == dis + w[i])
                cnt[j][0] += cnt[ver][type];
            else if (d[j][1] > dis + w[i]) {
                d[j][1] = dis + w[i], cnt[j][1] = cnt[ver][type];
                heap.push({d[j][1], j, 1});
            } else if (d[j][1] == dis + w[i])
                cnt[j][1] += cnt[ver][type];
        }
    }
    
    int res = cnt[t][0];
    if (d[t][0] + 1 == d[t][1]) res += cnt[t][1];
    return res;
}

int main() {
    int T;
    cin >> T;
    while (T -- ) {
        memset(h, -1, sizeof h); idx = 0;
        
        cin >> n >> m;
        
        while (m -- ) {
            int a, b, c;
            cin >> a >> b >> c;
            add(a, b, c);
        }
        
        cin >> s >> t;
        cout << dijkstra() << endl;
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

### floyd

> [!NOTE] **[AcWing 343. 排序](https://www.acwing.com/problem/content/345/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 传递闭包**

```cpp
// 传递闭包 O(mn3)
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 26;

int n, m;
bool g[N][N], d[N][N];
bool st[N];

void floyd() {
    memcpy(d, g, sizeof d);

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) d[i][j] |= d[i][k] && d[k][j];
}

int check() {
    for (int i = 0; i < n; i++)
        if (d[i][i]) return 2;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (!d[i][j] && !d[j][i]) return 0;

    return 1;
}

char get_min() {
    for (int i = 0; i < n; i++)
        if (!st[i]) {
            bool flag = true;
            for (int j = 0; j < n; j++)
                if (!st[j] && d[j][i]) {
                    flag = false;
                    break;
                }
            if (flag) {
                st[i] = true;
                return 'A' + i;
            }
        }
}

int main() {
    while (cin >> n >> m, n || m) {
        memset(g, 0, sizeof g);
        int type = 0, t;
        for (int i = 1; i <= m; i++) {
            char str[5];
            cin >> str;
            int a = str[0] - 'A', b = str[2] - 'A';

            if (!type) {
                g[a][b] = 1;
                floyd();
                type = check();
                if (type) t = i;
            }
        }

        if (!type)
            puts("Sorted sequence cannot be determined.");
        else if (type == 2)
            printf("Inconsistency found after %d relations.\n", t);
        else {
            memset(st, 0, sizeof st);
            printf("Sorted sequence determined after %d relations: ", t);
            for (int i = 0; i < n; i++) printf("%c", get_min());
            printf(".\n");
        }
    }

    return 0;
}
```

##### **C++ 增量算法**

```cpp
// 增量算法 O(mn2)
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 26;

int n, m;
bool d[N][N];
bool st[N];

int check() {
    for (int i = 0; i < n; i++)
        if (d[i][i]) return 2;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (!d[i][j] && !d[j][i]) return 0;

    return 1;
}

char get_min() {
    for (int i = 0; i < n; i++)
        if (!st[i]) {
            bool flag = true;
            for (int j = 0; j < n; j++)
                if (!st[j] && d[j][i]) {
                    flag = false;
                    break;
                }
            if (flag) {
                st[i] = true;
                return 'A' + i;
            }
        }
}

int main() {
    while (cin >> n >> m, n || m) {
        memset(d, 0, sizeof d);

        int type = 0, t;
        for (int i = 1; i <= m; i++) {
            char str[5];
            cin >> str;
            int a = str[0] - 'A', b = str[2] - 'A';

            if (!type) {
                d[a][b] = 1;
                for (int x = 0; x < n; x++) {
                    if (d[x][a]) d[x][b] = 1;
                    if (d[b][x]) d[a][x] = 1;
                    for (int y = 0; y < n; y++)
                        if (d[x][a] && d[b][y]) d[x][y] = 1;
                }
                type = check();
                if (type) t = i;
            }
        }

        if (!type)
            puts("Sorted sequence cannot be determined.");
        else if (type == 2)
            printf("Inconsistency found after %d relations.\n", t);
        else {
            memset(st, 0, sizeof st);
            printf("Sorted sequence determined after %d relations: ", t);
            for (int i = 0; i < n; i++) printf("%c", get_min());
            printf(".\n");
        }
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

> [!NOTE] **[AcWing 344. 观光之旅](https://www.acwing.com/problem/content/346/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模版 **无向图最小环**
> 
> > 对于有向图直接 floyd 即可
> > 直接一遍floyd；然后求自己到自己的最短距离，注意初始化 inf

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

floyd是典型的插点算法，每次插入点k，为此，在点k被[插入前]可计算i-j-k这个环
即此时中间节点为：1~k-1，即我们已经算出了任意i<->j的最短道路，中间经过的节点可以为 (1,2,3,…,k-1)
我们只需枚举所有以k为环中最大节点的环即可。

设一个环中的最大结点为k(编号最大), 与他相连的两个点为i, j, 这个环的最短长度
为g[i][k]+g[k][j]+i到j的路径中所有结点编号都小于k的最短路径长度。

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 110, INF = 0x3f3f3f3f;

int n, m;
int d[N][N], g[N][N];
int pos[N][N];
int path[N], cnt;

void get_path(int i, int j) {
    if (pos[i][j] == 0) return;

    int k = pos[i][j];
    get_path(i, k);
    path[cnt++] = k;
    get_path(k, j);
}

int main() {
    cin >> n >> m;

    memset(g, 0x3f, sizeof g);
    for (int i = 1; i <= n; i++) g[i][i] = 0;

    while (m--) {
        int a, b, c;
        cin >> a >> b >> c;
        g[a][b] = g[b][a] = min(g[a][b], c);
    }

    int res = INF;
    memcpy(d, g, sizeof d);
    // 至少包含三个点的环经过的点的最大编号是k
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i < k; i++)
            for (int j = i + 1; j < k; j++)
                if ((long long)d[i][j] + g[j][k] + g[k][i] < res) {
                    res = d[i][j] + g[j][k] + g[k][i];
                    cnt = 0;
                    path[cnt++] = k;
                    path[cnt++] = i;
                    get_path(i, j);
                    path[cnt++] = j;
                }
        // 学习如何记录路径
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                if (d[i][j] > d[i][k] + d[k][j]) {
                    d[i][j] = d[i][k] + d[k][j];
                    pos[i][j] = k;
                }
    }

    if (res == INF)
        puts("No solution.");
    else {
        for (int i = 0; i < cnt; i++) cout << path[i] << ' ';
        cout << endl;
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

> [!NOTE] **[AcWing 345. 牛站](https://www.acwing.com/problem/content/347/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 从 s 到 e 恰好 k 条边的最短路:
> 
> 矩阵快速幂

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>

using namespace std;

const int N = 210;

int k, n, m, S, E;
int g[N][N];
int res[N][N];

void mul(int c[][N], int a[][N], int b[][N]) {
    static int temp[N][N];
    memset(temp, 0x3f, sizeof temp);
    for (int k = 1; k <= n; k++)
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                temp[i][j] = min(temp[i][j], a[i][k] + b[k][j]);
    memcpy(c, temp, sizeof temp);
}

void qmi() {
    memset(res, 0x3f, sizeof res);
    for (int i = 1; i <= n; i++) res[i][i] = 0;

    while (k) {
        if (k & 1) mul(res, res, g);  // res = res * g
        mul(g, g, g);                 // g = g * g
        k >>= 1;
    }
}

int main() {
    cin >> k >> m >> S >> E;

    memset(g, 0x3f, sizeof g);
    map<int, int> ids;
    if (!ids.count(S)) ids[S] = ++n;
    if (!ids.count(E)) ids[E] = ++n;
    S = ids[S], E = ids[E];

    while (m--) {
        int a, b, c;
        cin >> c >> a >> b;
        if (!ids.count(a)) ids[a] = ++n;
        if (!ids.count(b)) ids[b] = ++n;
        a = ids[a], b = ids[b];

        g[a][b] = g[b][a] = min(g[a][b], c);
    }

    qmi();

    cout << res[S][E] << endl;

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

### spfa

> [!NOTE] **[AcWing 361. 观光奶牛](https://www.acwing.com/problem/content/363/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推导 二分 转化为负环判定

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const double eps = 1e-6;
const int N = 1010, M = 5010;

int n, m;
int wf[N];
int h[N], e[M], w[M], ne[M], idx;
double dist[N];
int cnt[N];

int q[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

bool check(double mid) {
    memset(dist, 0, sizeof dist);
    memset(cnt, 0, sizeof cnt);
    memset(st, 0, sizeof st);
    
    int hh = 0, tt = 0;
    for (int i = 1; i <= n; ++ i )
        q[tt ++ ] = i, st[i] = true;
    
    while (hh != tt) {
        int t = q[hh ++ ];
        if (hh == N) hh = 0;
        st[t] = false;
        
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] < dist[t] + wf[t] - mid * w[i]) {
                dist[j] = dist[t] + wf[t] - mid * w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= n) return true;
                if (!st[j]) {
                    q[tt ++ ] = j;
                    if (tt == N) tt = 0;
                    st[j] = true;
                }
            }
        }
    }
    return false;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) cin >> wf[i];
    
    while (m -- ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c);
    }
    
    double l = 0, r = 1e6;
    while (r - l > eps) {
        double mid = (l + r) / 2;
        if (check(mid)) l = mid;
        else r = mid;
    }
    printf("%.2lf\n", l);
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

> [!NOTE] **[AcWing 1165. 单词环](https://www.acwing.com/problem/content/1167/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> STL超时 学习加边方法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 700, M = 100010;

int n;
int h[N], e[M], w[M], ne[M], idx;
double dist[N];
int q[N], cnt[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

bool check(double mid) {
    memset(st, 0, sizeof st);
    memset(cnt, 0, sizeof cnt);

    int hh = 0, tt = 0;
    for(int i = 0; i < 676; ++i) {
        q[tt++] = i;
        st[i] = true;
    }

    int count = 0;
    while(hh != tt) {
        int t = q[hh++];
        if(hh == N) hh = 0;
        st[t] = false;

        for(int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] < dist[t] + w[i] - mid) {
                dist[j] = dist[t] + w[i] - mid;
                cnt[j] = cnt[t] + 1;
                if(++count > 10000) return true; // 经验上的trick
                if(cnt[j] >= N) return true;
                if(!st[j]) {
                    q[tt++] = j;
                    if(tt == N) tt = 0;
                    st[j] = true;
                }
            }
        }
    }
    return false;
}

int main() {
    string str;
    while(cin >> n, n) {
        memset(h, -1, sizeof h);
        for(int i = 0; i < n; ++i) {
            cin >> str;
            int len = str.size();
            if(len > 1) {
                int l = (str[0]-'a')*26 + str[1]-'a', r = (str[len-2]-'a')*26 + str[len-1]-'a';
                add(l, r, len);
            }
        }
        if(!check(0)) cout << "No solution" << endl;
        else {
            double l = 0, r = 1000;
            while(r-l > 1e-4) {
                double mid = l + (r-l)/2;
                if(check(mid)) l = mid;
                else r = mid;
            }
            cout << l << endl;
        }
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

### 拆点最短路

> [!NOTE] **[LeetCode 1928. 规定时间内到达终点的最小花费](https://leetcode-cn.com/problems/minimum-cost-to-reach-destination-in-time/)**
> 
> [biweekly 56](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-07-10_Biweekly-56)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 无向连通图（有无自环无所谓），二维最短路问题
> 
> ===> 拆点，每个点在每个时间下的状态单独作为一个点
> 
> 对于本题共计 1e6 个点，直接拆点跑最短路即可
> 
> **对于二维费用的最短路进行拆点再跑最短路**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 拆点最短路**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 1010, M = N << 1;
    const int INF = 0x3f3f3f3f;
    
    vector<int> c;
    int n, mt;
    int h[N], e[M], w[M], ne[M], idx;
    bool st[N][N];
    int d[N][N];
    
    void init() {
        memset(h, -1, sizeof h);
        memset(st, 0, sizeof st);//
        memset(d, 0x3f, sizeof d);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int minCost(int maxTime, vector<vector<int>>& edges, vector<int>& passingFees) {
        init();
        
        this->c = passingFees;
        this->n = c.size();
        this->mt = maxTime;
        for (auto & eg : edges) {
            int a = eg[0], b = eg[1], c = eg[2];
            add(a, b, c), add(b, a, c);
        }
        
        d[0][0] = c[0];
        queue<PII> q;
        q.push({0, 0});
        
        while (q.size()) {
            auto [x, y] = q.front(); q.pop();
            st[x][y] = false;
            
            for (int i = h[x]; ~i; i = ne[i]) {
                int nx = e[i], ny = y + w[i];
                if (ny > mt)
                    continue;
                if (d[nx][ny] > d[x][y] + c[nx]) {
                    d[nx][ny] = d[x][y] + c[nx];
                    if (!st[nx][ny])
                        st[nx][ny] = true, q.push({nx, ny});
                }
            }
        }
        
        int res = INF;
        for (int i = 0; i <= mt; ++ i )
            res = min(res, d[n - 1][i]);
        return res == INF ? -1 : res;
    }
};
```

##### **C++ 拆点dp 学习**

```cpp
class Solution {
    const int INF = 0x3f3f3f3f;

public:
    int minCost(int T, vector<vector<int>>& edges, vector<int> a) {
        int n = a.size();
        std::vector<std::vector<std::pair<int, int>>> E(n);
        for (int i = 0; i < (int)edges.size(); ++i) {
            int u = edges[i][0], v = edges[i][1], w = edges[i][2];
            E[u].emplace_back(v, w);
            E[v].emplace_back(u, w);
        }
        std::vector<std::vector<int>> f(T + 1, std::vector<int>(n, INF));
        f[0][0] = a[0];
        int ans = INF;
        for (int i = 1; i <= T; ++i) {
            for (int u = 0; u < n; ++u) {
                for (auto [v, w] : E[u]) {
                    if (i >= w) {
                        f[i][u] = std::min(f[i][u], f[i - w][v] + a[u]);
                    }
                }
            }
            ans = std::min(ans, f[i][n - 1]);
        }
        return ans == INF ? -1 : ans;
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

### 次短路等

> [!NOTE] **[AcWing 2045. 到达目的地的第二短时间](https://leetcode-cn.com/problems/second-minimum-time-to-reach-destination/)**
> 
> [weekly-263](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-10-17_Weekly-263)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准次短路：对于 n 较大的情况使用 astar 复杂度较高易超时，所以还是用标准次短路写法较好
> 
> 经典求次短路
> 
> $2 <= n <= 10^4$
> 
> $n - 1 <= edges.length <= min(2 \* 104, n \* (n - 1) / 2)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // using LL = long long;
    using PII = pair<int, int>;
    using PIII = tuple<int, int, int>;
    const static int N = 1e4 + 10, M = 4e4 + 10; // M = 2e4 * 2
    
    int fix_time(int x, int l) {
        int d = x / ch;
        // red
        if (d & 1)
            return (d + 1) * ch + l;
        return x + l;
    }
    
    int n, ch;
    int h[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int dist[N], t_dist[N];
    // bool st[N];
    int dijkstra() {
        memset(dist, 0x3f, sizeof dist);
        memset(t_dist, 0x3f, sizeof t_dist);
        // memset(st, 0, sizeof st);
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        heap.push({0, 1});  // ATTENTION from 1 to n
        dist[1] = 0;    // from 1 to n
        
        while (heap.size()) {
            auto [dis, ver] = heap.top();
            heap.pop();
            // if (st[ver])
            //     continue;
            // st[ver] = true;
            
            // 剪枝
            // ATTENTION 必须是 <   因为等于也可能会更新
            //
            // 在这里只需要求次短 所以我们直接使用 t_dist[ver]
            // 如果所求为倒数第k短 我们可以使用dist[][]
            // 随后记录当前是第x短 通过dist[ver][x]<dis来进行剪枝判断
            if (t_dist[ver] < dis)
                continue;
            
            for (int i = h[ver]; ~i; i = ne[i]) {
                int j = e[i];
                int new_dis = fix_time(dis, w[i]);
                if (dist[j] > new_dis) {
                    t_dist[j] = dist[j];
                    dist[j] = new_dis;
                    heap.push({dist[j], j});
                    heap.push({t_dist[j], j});
                } else if (dist[j] != new_dis && t_dist[j] > new_dis) {
                    // dist[j] != new_dis 严格小于
                    t_dist[j] = new_dis;
                    heap.push({t_dist[j], j});
                }
            }
        }
        return t_dist[n];
    }
    
    int secondMinimum(int n, vector<vector<int>>& edges, int time, int change) {
        this->n = n, this->ch = change;
        init();
        
        for (auto & e : edges)
            add(e[0], e[1], time), add(e[1], e[0], time);
        
        return dijkstra();;
    }
};
```

##### **C++ A* TLE**

```cpp
// TLE 52 / 76
class Solution {
public:
    // using LL = long long;
    using PII = pair<int, int>;
    using PIII = tuple<int, int, int>;
    const static int N = 1e4 + 10, M = 4e4 + 10; // M = 2e4 * 2
    
    int fix_time(int x, int l) {
        int d = x / ch;
        // red
        if (d & 1)
            return (d + 1) * ch + l;
        return x + l;
    }
    
    int n, ch;
    int h[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int dist[N];
    bool st[N];
    void dijkstra() {
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        heap.push({0, 1});  // ATTENTION from 1 to n
        memset(dist, 0x3f, sizeof dist);
        memset(st, 0, sizeof st);
        dist[1] = 0;    // from 1 to n
        
        while (heap.size()) {
            auto [dis, ver] = heap.top();
            heap.pop();
            if (st[ver])
                continue;
            st[ver] = true;
            
            for (int i = h[ver]; ~i; i = ne[i]) {
                int j = e[i];
                int new_dis = fix_time(dis, w[i]);
                if (dist[j] > new_dis) {
                    dist[j] = new_dis;
                    heap.push({dist[j], j});
                }
            }
        }
    }
    
    int cnt[N];
    int astar(int k) {
        // 估价-真实值-点
        priority_queue<PIII, vector<PIII>, greater<PIII>> heap;
        heap.push({dist[1], 0, 1});
        memset(cnt, 0, sizeof cnt);
        
        while (heap.size()) {
            auto [_, dis, ver] = heap.top();
            heap.pop();
            
            cnt[ver] ++ ;
            // ATTENTION: 重要 dis 还必须是大于最短值 否则会wa 23/76
            // 因为题目要求 【严格大于】
            // dis > dist[n]
            if (cnt[n] >= k && dis > dist[n])
                return dis;
            
            for (int i = h[ver]; ~i; i = ne[i]) {
                int j = e[i];
                int new_dis = fix_time(dis, w[i]);
                if (cnt[j] < k) // ATTENTION TLE 35/76
                    heap.push({new_dis + dist[j], new_dis, j}); // ATTENTION
            }
        }
        return -1;
    }
    
    int secondMinimum(int n, vector<vector<int>>& edges, int time, int change) {
        this->n = n, this->ch = change;
        init();
        
        for (auto & e : edges)
            add(e[0], e[1], time), add(e[1], e[0], time);
        
        dijkstra();
        
        // 次短路
        return astar(2);
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

### Trick 最短路 思维题

> [!NOTE] **[AcWing 1386. 卡米洛特](https://www.acwing.com/problem/content/1388/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 核心：枚举
>
> 枚举汇合点、骑士接国王 枚举接国王的骑士 ==> 复杂度过高
>
> 考虑：
>
> 对于枚举的每一个汇合点，国王两种走法。
>
> 1. 自己走过去
>
> 2. 选一名骑士接（额外步数 dist_min 最少的骑士）
>
> `换了枚举思路 降低复杂度`
>
> 参见 https://www.acwing.com/solution/content/32282/
>
> `dist_sum 所有骑士到汇合点[i, j]的最短距离和`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
using PIII = tuple<int, int, int>;

const int N = 31, M = N * N, INF = 0x3f3f3f3f;

int n, m;
PII king;
int dist_sum[N][N], dist_min[N][N], dist[N][N][2];
// struct Node {
//     int x, y, z;
// }q[M];
PIII q[M];
bool st[N][N][2];
int dx[] = {-2, -1, 1, 2, 2, 1, -1, -2};
int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};

void spfa(int sx, int sy) {
    memset(dist, 0x3f, sizeof dist);
    
    int hh = 0, tt = 0;
    q[tt ++ ] = {sx, sy, 0};
    dist[sx][sy][0] = 0;
    st[sx][sy][0] = true;
    
    while (hh != tt) {
        auto [x, y, z] = q[hh ++ ];
        if (hh == M) hh = 0;
        st[x][y][z] = false;    // 出队
        
        for (int i = 0; i < 8; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 1 || nx > n || ny < 1 || ny > m) continue;
            if (dist[nx][ny][z] > dist[x][y][z] + 1) {
                dist[nx][ny][z] = dist[x][y][z] + 1;
                if (!st[nx][ny][z]) {
                    q[tt ++ ] = {nx, ny, z};
                    if (tt == M) tt = 0;
                    st[nx][ny][z] = true;
                }
            }
            
            if (!z) {
                int d = dist[x][y][z] + max(abs(king.first - x), abs(king.second - y));
                if (dist[x][y][1] > d) {
                    dist[x][y][1] = d;
                    if (!st[x][y][1]) {
                        q[tt ++ ] = {x, y, 1};
                        if (tt == M) tt = 0;
                        st[x][y][1] = true;
                    }
                }
            }
        }
    }
    
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            if (dist[i][j][0] == INF)
                dist_sum[i][j] = INF;
            else
                dist_sum[i][j] += dist[i][j][0];
    
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            dist_min[i][j] = min(dist_min[i][j], dist[i][j][1] - dist[i][j][0]);    // [1] - [0] 即可
}

int main() {
    cin >> n >> m;
    
    int x; char y;
    cin >> y >> x;
    y = y - 'A' + 1;
    king = {x, y};
    
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            dist_min[i][j] = max(abs(i - x), abs(j - y));
    while (cin >> y >> x) {
        y = y - 'A' + 1;
        spfa(x, y);
    }
    
    int res = INF;
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            res = min(res, dist_sum[i][j] + dist_min[i][j]);
    cout << res << endl;
    
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