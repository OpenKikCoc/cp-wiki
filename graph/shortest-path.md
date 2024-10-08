# 坑点

> [!NOTE] **注意事项**
> 
> 在函数参数重传了 `int sth[]` 数组的不要在数组内部 `memset(sth, 0x3f, sizeof sth)`
> 
> 因为 `sizeof` 拿到的不是真实大小
> 
> => 如果是 int 用 `sizeof 4 * N`

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

### 一般 dijkstra

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

> [!NOTE] **[LeetCode 743. 网络延迟时间](https://leetcode.cn/problems/network-delay-time/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 110, M = 6010;
    const int INF = 2e9;
    using PII = pair<int, int>;
    int n;
    int h[N], e[M], w[M], ne[M], idx;
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }

    int dijkstra(int s) {
        vector<int> d(n + 1, INF);
        vector<bool> st(n + 1);
        d[s] = 0;
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        heap.push({0, s});
        while (heap.size()) {
            auto [dis, ver] = heap.top();
            heap.pop();
            if (st[ver])
                continue;
            st[ver] = true;

            for (int i = h[ver]; ~i; i = ne[i] ) {
                int j = e[i], v = w[i];
                if (d[j] > dis + v) {
                    d[j] = dis + v;
                    heap.push({dis + v, j});
                }
            }
        }

        int maxv = 0;
        for (int i = 1; i <= n; ++ i )
            maxv = max(maxv, d[i]);
        return maxv > INF / 2 ? -1 : maxv;
    }

    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        memset(h, -1, sizeof h);
        for (auto & e : times)
            add(e[0], e[1], e[2]);
        this->n = n;
        return dijkstra(k);
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

> [!NOTE] **[LeetCode 1514. 概率最大的路径](https://leetcode.cn/problems/path-with-maximum-probability/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    double maxProbability(int n, vector<vector<int>>& edges,
                          vector<double>& succProb, int start, int end) {
        vector<vector<pair<int, double>>> g(n);
        int m = edges.size();
        for (int i = 0; i < m; i++) {
            g[edges[i][0]].push_back({edges[i][1], succProb[i]});
            g[edges[i][1]].push_back({edges[i][0], succProb[i]});
        }
        vector<double> d(n, 0);
        d[start] = 1.0;
        priority_queue<pair<double, int>> pq;
        pq.push({1.0, start});
        while (!pq.empty()) {
            auto u = pq.top();
            pq.pop();
            int v = u.second;
            double p = u.first;
            if (v == end) break;
            for (auto x : g[v]) {
                int y = x.first;
                double pp = x.second;
                if (pp * p > d[y]) {
                    d[y] = pp * p;
                    pq.push({d[y], y});
                }
            }
        }
        return d[end];
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

> [!NOTE] **[LeetCode 3123. 最短路径中的边](https://leetcode.cn/problems/find-edges-in-shortest-paths/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准求【最短路径边】的算法
> 
> -   两次 dijkstra 校验每条边左右两侧到起始点的距离和
> 
> -   一次 dijkstra + 逆序路径长度判断 (较麻烦 略)
> 
> `sizeof 4 * N`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 5e4 + 10, M = 1e5 + 10, INF = 0x3f3f3f3f;
    
    int h[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int d1[N], d2[N];
    bool st[N];
    void dijkstra(int dist[], int s) {
        memset(dist, 0x3f, sizeof 4 * N);
        memset(st, 0, sizeof st);
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        dist[s] = 0; heap.push({0, s});
        
        while (heap.size()) {
            auto [d, u] = heap.top(); heap.pop();
            if (st[u])
                continue;
            st[u] = true;
            
            for (int i = h[u]; ~i; i = ne[i]) {
                int j = e[i], c = w[i];
                if (dist[j] > d + c)
                    heap.push({dist[j] = d + c, j});
            }
        }
    }
    
    vector<bool> findAnswer(int n, vector<vector<int>>& edges) {
        init();
        for (auto & e : edges) {
            int a = e[0], b = e[1], c = e[2];
            add(a, b, c), add(b, a, c);
        }
        
        dijkstra(d1, 0);
        dijkstra(d2, n - 1);
        
        int tot = d1[n - 1];
        vector<bool> res;
        for (auto & e : edges) {
            int a = e[0], b = e[1], c = e[2];
            if (d1[a] + d2[b] + c == tot || d1[b] + d2[a] + c == tot)
                res.push_back(true);
            else
                res.push_back(false);
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

> [!NOTE] **[LeetCode 1786. 从第一个节点出发到最后一个节点的受限路径数](https://leetcode.cn/problems/number-of-restricted-paths-from-first-to-last-node/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 注意 dijkstra auto 不能写引用 会出问题 // todo
> - 因为必然可达，所以可以直接 sort

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 20010, M = 100010, MOD = 1e9 + 7, INF = 0x3f3f3f3f;
    using PII = pair<int, int>;
    
    // gragh
    int h[N], e[M], w[M], ne[M], idx;
    
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    // val
    int n;
    int dist[N], f[N];
    bool st[N];
    
    void init() {
        memset(h, -1, sizeof h);
        memset(st, 0, sizeof st);
        idx = 0;
    }
    
    void dijkstra(int s) {
        memset(dist, 0x3f, sizeof dist);
        memset(st, 0, sizeof st);
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        heap.push({0, s});
        dist[s] = 0;
        while (!heap.empty()) {
            auto [d, u] = heap.top(); heap.pop();
            if (st[u]) continue;
            st[u] = true;
            for (int i = h[u]; ~i; i = ne[i]) {
                int j = e[i];
                if (dist[j] > d + w[i]) {
                    dist[j] = d + w[i];
                    heap.push({dist[j], j});
                }
            }
        }
    }
    
    void dfs(int u) {
        if (f[u] != -1) return;
        
        f[u] = 0;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[u] > dist[j]) {
                dfs(j);
                f[u] = (f[u] + f[j]) % MOD;
            }
        }
    }
    
    int countRestrictedPaths(int n, vector<vector<int>>& edges) {
        init();
        this->n = n;
        for (auto & e : edges)
            add(e[0], e[1], e[2]), add(e[1], e[0], e[2]);
        
        dijkstra(n);
        
        memset(f, -1, sizeof f);
        memset(st, 0, sizeof st);
        f[n] = 1;
        dfs(1);
        return f[1];
    }
};
```

##### **C++ 直接sort**

```cpp
using PII = pair<int, int>;
#define x first
#define y second

class Solution {
public:
    const int INF = 0x3f3f3f3f, MOD = 1e9 + 7;
    vector<vector<PII>> g;
    vector<int> f, dist;
    vector<bool> st;
    
    int countRestrictedPaths(int n, vector<vector<int>>& edges) {
        g.resize(n + 1), f.resize(n + 1), dist.resize(n + 1, INF), st.resize(n + 1);
        for(auto& e: edges) {
            int a = e[0], b = e[1], c = e[2];
            g[a].push_back({b, c});
            g[b].push_back({a, c});
        }
        queue<int> q;
        q.push(n);
        dist[n] = 0;
        while (q.size()) {
            auto t = q.front();
            q.pop();
            st[t] = false;
            
            for (auto& v: g[t]) {
                int j = v.x, w = v.y;
                if (dist[j] > dist[t] + w) {
                    dist[j] = dist[t] + w;
                    if (!st[j]) {
                        q.push(j);
                        st[j] = true;
                    }
                }
            }
        }
        
        vector<PII> vs;
        for (int i = 1; i <= n; i ++ ) vs.push_back({dist[i], i});
        sort(vs.begin(), vs.end());
        
        f[n] = 1;
        for (auto& v: vs) {
            int d = v.x, u = v.y;
            for (auto p: g[u]) {
                int j = p.x;
                if (d > dist[j])
                    f[u] = (f[u] + f[j]) % MOD;
            }
        }
        return f[1];
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

> [!NOTE] **[LeetCode LCP 35. 电动车游城市](https://leetcode.cn/problems/DFPeFJ/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 踩坑。。比赛按照传统写法 TLE
> 
> **其实充电的情况可以同样认为是边拓展，这样写会减少一些 `多次入队` 的情况**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 110, M = N << 2;
int h[N], e[M], w[M], ne[M], idx;

void add(int a, int b, int c) {
   e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int dist[N][N];
bool st[N][N];

class Solution {
public:
    using TIII = tuple<int, int, int>;
    
    int n, start, end, cnt;
    vector<int> ch;
    
    void dijkstra() {
        memset(dist, 0x3f, sizeof dist);
        memset(st, 0, sizeof st);
        
        priority_queue<TIII, vector<TIII>, greater<TIII>> heap;
        
        dist[start][0] = 0;
        heap.push({dist[start][0], start, 0});
        
        while (heap.size()) {
            auto [dis, ver, oil] = heap.top();
            heap.pop();
            if (st[ver][oil])
                continue;
            st[ver][oil] = true;
            
            for (int i = oil + 1; i <= cnt; ++ i ) {
                if (dist[ver][i] > dis + ch[ver] * (i - oil)) {
                    dist[ver][i] = dis + ch[ver] * (i - oil);
                    heap.push({dist[ver][i], ver, i});
                }
            }
            
            for (int i = h[ver]; ~i; i = ne[i]) {
                int j = e[i];
                if (w[i] > oil)
                    continue;
                if (dist[j][oil - w[i]] > dis + w[i]) {
                    dist[j][oil - w[i]] = dis + w[i];
                    heap.push({dist[j][oil - w[i]], j, oil - w[i]});
                }
            }
        }
    }
    
    int electricCarPlan(vector<vector<int>>& paths, int cnt, int start, int end, vector<int>& charge) {
        this->n = charge.size(), this->start = start, this->end = end, this->cnt = cnt, this->ch = charge;
        memset(h, -1, sizeof h);
        idx = 0;
        
        for (auto & e : paths) {
            int u = e[0], v = e[1], w = e[2];
            add(u, v, w), add(v, u, w);
        }
        
        dijkstra();
        
        int res = INT_MAX;
        for (int i = 0; i <= cnt; ++ i )
            res = min(res, dist[end][i]);
        
        return res;
    }
};
```

##### **C++ TLE**

```cpp
// TLE
// 63 / 63 个通过测试用例

const int N = 110, M = N << 2;
int h[N], e[M], w[M], ne[M], idx;

void add(int a, int b, int c) {
   e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int dist[N][N];
bool st[N][N];

class Solution {
public:
    using TIII = tuple<int, int, int>;
    
    int n, start, end, cnt;
    vector<int> ch;
    
    void dijkstra() {
        memset(dist, 0x3f, sizeof dist);
        memset(st, 0, sizeof st);
        
        priority_queue<TIII, vector<TIII>, greater<TIII>> heap;
        
        for (int i = 0; i <= cnt; ++ i ) {
            dist[start][i] = ch[start] * i;
            heap.push({dist[start][i], start, i});
        }
        
        while (heap.size()) {
            auto [dis, ver, oil] = heap.top();
            heap.pop();
            if (st[ver][oil])
                continue;
            st[ver][oil] = true;
            
            
            for (int i = h[ver]; ~i; i = ne[i]) {
                int j = e[i];
                if (w[i] > oil)
                    continue;
                int rest = oil - w[i];
                for (int k = rest; k <= cnt; ++ k ) {
                    int add = k - rest;
                    if (dist[j][k] > dis + w[i] + add * ch[j]) {
                        dist[j][k] = dis + w[i] + add * ch[j];
                        heap.push({dist[j][k], j, k});
                    }
                }
            }
        }
    }
    
    int electricCarPlan(vector<vector<int>>& paths, int cnt, int start, int end, vector<int>& charge) {
        this->n = charge.size(), this->start = start, this->end = end, this->cnt = cnt, this->ch = charge;
        memset(h, -1, sizeof h);
        idx = 0;
        
        for (auto & e : paths) {
            int u = e[0], v = e[1], w = e[2];
            add(u, v, w), add(v, u, w);
        }
        
        dijkstra();
        
        int res = INT_MAX;
        for (int i = 0; i <= cnt; ++ i )
            res = min(res, dist[end][i]);
        
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

> [!NOTE] **[LeetCode 2203 得到要求路径的最小带权子图](https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典图论：**枚举中间点**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    using PLL = pair<LL, LL>;
    const static int N = 1e5 + 10, M = 2e5 + 10;
    
    int h[N], rh[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        memset(rh, -1, sizeof rh);
        idx = 0;
    }
    void add(int h[], int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n;
    LL d[3][N];
    
    void dijkstra(int src, int h[], LL d[]) {
        for (int i = 0; i < n; ++ i )
            d[i] = 1e18;
        bool st[N];
        memset(st, 0, sizeof st);
        
        priority_queue<PLL, vector<PLL>, greater<PLL>> heap;
        heap.push({0, src}); d[src] = 0;
        while (heap.size()) {
            auto [dis, u] = heap.top(); heap.pop();
            if (st[u])
                continue;
            st[u] = true;
            for (int i = h[u]; ~i; i = ne[i]) {
                int j = e[i], c = w[i];
                if (d[j] > d[u] + c) {
                    d[j] = d[u] + c;
                    heap.push({d[j], j});
                }
            }
        }
    }
    
    long long minimumWeight(int n, vector<vector<int>>& edges, int src1, int src2, int dest) {
        init();
        for (auto & e : edges)
            add(h, e[0], e[1], e[2]), add(rh, e[1], e[0], e[2]);
        this->n = n;
        
        dijkstra(src1, h, d[0]);
        dijkstra(src2, h, d[1]);
        dijkstra(dest, rh, d[2]);
        
        LL res = 1e18;
        for (int i = 0; i < n; ++ i )
            res = min(res, d[0][i] + d[1][i] + d[2][i]);
        if (res >= 1e18)
            return -1;
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

> [!NOTE] **[Codeforces Missile Silos](http://codeforces.com/problemset/problem/144/D)**
> 
> 题意: 
> 
> 给定一张连通的无向图，求其中有多少秘密基地。
> 
> 所谓一个秘密基地，就是距离 $s$ 号点的最短距离恰好等于 $k$ 的位置，**这个位置可以在一个结点上，也可以在一条边的中间**。

> [!TIP] **思路**
> 
> ATTENTION **计算推理细节**
> 
> 条件写错WA 反复思考判断条件的推理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Missile Silos
// Contest: Codeforces - Codeforces Round #103 (Div. 2)
// URL: https://codeforces.com/problemset/problem/144/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PLI = pair<LL, int>;
const static int N = 1e5 + 10, M = 2e5 + 10;

int h[N], e[M], w[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int n, m, s, l;
bool st[N];
LL dist[N];

void dijkstra() {
    memset(st, 0, sizeof st);
    for (int i = 0; i < N; ++i)
        dist[i] = 1e17;

    priority_queue<PLI, vector<PLI>, greater<PLI>> heap;
    dist[s] = 0;
    heap.push({0ll, s});
    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (st[u])
            continue;
        st[u] = true;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] > d + w[i]) {
                dist[j] = d + w[i];
                heap.push({dist[j], j});
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n >> m >> s;
    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    cin >> l;

    dijkstra();

    int res = 0;
    {
        // 1. 判定点
        for (int i = 1; i <= n; ++i)
            if (dist[i] == l)
                res++;
    }
    {
        // 2. 判定边
        // ATTENTION 判定规则
        LL cnt = 0;
        for (int i = 1; i <= n; ++i)
            for (int _ = h[i]; ~_; _ = ne[_]) {
                int j = e[_], c = w[_];
                LL di = dist[i], dj = dist[j];
                if (di < l && dj < l && di + dj + c == 2ll * l)
                    // 中间的点（唯一）
                    cnt++;
                else if (di + dj + c > 2ll * l) {
                    // ATTENTION why?
                    // NOT: di < l && dj > l, but just concern di
                    //
                    // 必须分开算两个 否则判断条件会比较复杂
                    // i 侧 ATTENTION 不能加上对dj的限制
                    if (di < l)
                        cnt++;
                    // j 侧
                    if (dj < l)
                        cnt++;
                }
            }
        res += cnt / 2;
    }
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

> [!NOTE] **[LeetCode 882. 细分图中的可到达结点](https://leetcode.cn/problems/reachable-nodes-in-subdivided-graph/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 无向图边权非负 显然可以 dijkstra 跑最短路
> 
> 随后扫一遍边即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 3e3 + 10, M = 2e4 + 10;

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
    void dijkstra(int s) {
        memset(dist, 0x3f, sizeof dist);
        memset(st, 0, sizeof st);
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        heap.push({0, s});
        dist[s] = 0;
        while (heap.size()) {
            auto [d, u] = heap.top(); heap.pop();
            if (st[u])
                continue;
            st[u] = true;
            for (int i = h[u]; ~i; i = ne[i]) {
                int j = e[i];
                if (dist[j] > d + w[i] + 1) {
                    dist[j] = d + w[i] + 1;
                    heap.push({dist[j], j});
                }
            }
        }
    }

    int reachableNodes(vector<vector<int>>& edges, int maxMoves, int n) {
        init();
        for (auto & e : edges) {
            int a = e[0], b = e[1], c = e[2];
            add(a, b, c), add(b, a, c);
        }
        dijkstra(0);

        int res = 0;
        for (auto & e : edges) {
            int a = e[0], b = e[1], c = e[2];
            if (dist[a] >= maxMoves && dist[b] >= maxMoves)
                continue;
            int add = 0;
            if (dist[a] < maxMoves) {
                add += maxMoves - dist[a];
            }
            if (dist[b] < maxMoves) {
                add += maxMoves - dist[b];
            }
            // cout << " a = " << a << " b = " << b << " da = " << dist[a] << " db = " << dist[b] << " add = " << min(add, c) << endl;
            res += min(add, c);
        }
        for (int i = 0; i < n; ++ i )
            if (dist[i] <= maxMoves)
                res ++ ;

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

> [!NOTE] **[LeetCode 505. 迷宫 II](https://leetcode.cn/problems/the-maze-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO 解决最后一个 case TLE
> 
> 更有充分的正确性证明的代码？

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 110, INF = 0x3f3f3f3f;

    struct Node {
        int x, y, r, d;
        bool operator< (const Node & t) const { // ATTENTION 第二个const不可少
            return d < t.d;
        }
    };

    vector<vector<int>> g;
    int n, m;

    int d[N][N][4];
    int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};

    bool willStop(int x, int y, int r) {
        int nx = x + dx[r], ny = y + dy[r];
        if (nx < 0 || nx >= n || ny < 0 || ny >= m || g[nx][ny])
            return true;
        return false;
    }

    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        this->g = maze;
        n = maze.size(), m = maze[0].size();
        int sx = start[0], sy = start[1];
        int ex = destination[0], ey = destination[1];

        memset(d, 0x3f, sizeof d);
        priority_queue<Node> pq;
        for (int i = 0; i < 4; ++ i ) {
            d[sx][sy][i] = 0;
            pq.push({sx, sy, i, 0});
        }
            
        while (pq.size()) {
            auto u = pq.top(); pq.pop();
            bool canChange = willStop(u.x, u.y, u.r); // 能否转弯
            for (int i = 0; i < 4; ++ i ) {
                if (i != u.r && !canChange)
                    continue;
                int nx = u.x + dx[i], ny = u.y + dy[i];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                    continue;
                if (maze[nx][ny] == 1)
                    continue;
                if (d[nx][ny][i] > u.d + 1) {
                    d[nx][ny][i] = u.d + 1;
                    pq.push({nx, ny, i, u.d + 1});
                }
            }
        }

        int res = INF;
        for (int i = 0; i < 4; ++ i )
            if (willStop(ex, ey, i))
                res = min(res, d[ex][ey][i]);
        return res >= INF / 2 ? -1 : res;
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

> [!NOTE] **[LeetCode 2577. 在网格图中访问一个格子的最少时间](https://leetcode.cn/problems/minimum-time-to-visit-a-cell-in-a-grid/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 特殊逻辑的 dijkstra 即可
> 
> **重点在于对题意特征的简化，并得到结论：先排除无解; 再分析 $mod2$ 的特征**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 显然 需要按格子的值的顺序排序（思考 应该可以排序后使用并查集维护联通性质）
    // 唯一麻烦的点在于 【不能停留在格子上】
    //   考虑如果点周围有合法点，则可以来回两次移动（不太好维护 直接换成一个队列存储可行位置？）==> 【ATTENTION dijkstra】
    //
    // 思考：啥时候无解？当且仅当第一步没法走的时候无解 否则其他情况都可以来回踱步来最终走到目的地
    // 思考：来回踱步有啥特征？对于每一个位置 处于它的时刻 mod2 总是不变的
    //      => 具体来说，对于 [x, y]
    //          x + y 的奇偶性相关
    
    using PII = pair<int, int>;
    using TIII = tuple<int, int, int>;
    
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    
    vector<vector<int>> g;
    int n, m;
    
    vector<vector<int>> st;
    
    int minimumTime(vector<vector<int>>& grid) {
        if (grid[1][0] > 1 && grid[0][1] > 1)
            return -1;
        // 后续能够保证一定有解
        
        this->g = grid, this->n = g.size(), this->m = g[0].size();
        // 预处理一下 每个位置所要求的时间
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (((i + j) & 1) ^ (g[i][j] & 1)) {    // ATTENTION &1 别写岔了
                    g[i][j] ++ ;
                }
        
        // 考虑 bfs 存储已经到达的位置和到达该位置的最早时间
        st = vector<vector<int>>(n, vector<int>(m, 1e9));
        
        priority_queue<TIII, vector<TIII>, greater<TIII>> q;
        q.push({0, 0, 0});
        while (!q.empty()) {
            auto [t, x, y] = q.top(); q.pop();
            if (st[x][y] <= 1e8)
                continue;
            st[x][y] = t;
            
            for (int i = 0; i < 4; ++ i ) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                    continue;
                
                // 如果这个点现在到不了 也可以提前加入但是要追加时间
                int nt = max(t + 1, g[nx][ny]);
                // if (nt > st[nx][ny])
                //     continue;
                
                q.push({nt, nx, ny});
            }
        }
        
        return st[n - 1][m - 1];
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

> [!NOTE] **[LeetCode 2699. 修改图中的边权](https://leetcode.cn/problems/modify-graph-edge-weights/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 理解细节
> 
> TODO: 反复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 100 个点，最多 1e4 边
    // 思考：修改负权边只会让最短距离变小，如果原图最短距离已经很小显然无解
    // 【联通图 则一定有办法】
    using PII = pair<int, int>;
    const static int N = 110, INF = 0x3f3f3f3f;

    int n, s, t;
    vector<vector<PII>> es;

    vector<vector<int>> modifiedGraphEdges(int n, vector<vector<int>>& edges, int source, int destination, int target) {
        this->n = n, this->s = source, this->t = destination;
        // 预处理边 记录对应的 idx
        this->es = vector<vector<PII>>(n);
        for (int i = 0; i < edges.size(); ++ i ) {
            auto & e = edges[i];
            int a = e[0], b = e[1];
            es[a].push_back({b, i}), es[b].push_back({a, i});
        }

        static int d[2][N];
        memset(d, 0x3f, sizeof d);
        d[0][s] = d[1][s] = 0;
        static bool st[N];
        int diff;

        auto dijkstra = [&](int k) {
            memset(st, 0, sizeof st);
            for (;;) {
                // 朴素 dijkstra
                // 找到当前最短路的点 并更新其邻居的距离
                int x = -1;
                for (int i = 0; i < n; ++ i )
                    if (!st[i] && (x < 0 || d[k][i] < d[k][x]))
                        x = i;
                if (x == t)
                    break;
                st[x] = true;
                for (auto [y, idx] : es[x]) {
                    int w = edges[idx][2];
                    if (w == -1)
                        w = 1;  // -1 改成 1
                    
                    // 如果是第二次 dijkstra
                    if (k == 1 && edges[idx][2] == -1) {
                        // 则在第二次修改 w
                        // 【理解细节】
                        //  对于一个可修改的边，假定将其修改为 W 那么 s->x->y->t 由三部分组成
                        //      1. s->x 的最短路     => d[1][x]
                        //      2. x->y             => W
                        //      3. y->t 的最短路     => d[0][t] - d[0][y]
                        // 【ATTENTION】这个式子只有当前路径会是最短路时才会成立
                        //      但【不在最短路上也不会对最短路产生影响】故可以非常简单的不做判断
                        //      在此前提下，如果要让三部分的和为 target，则新边长即为下式：
                        int tw = diff + d[0][y] - d[1][x];
                        if (tw > w)
                            edges[idx][2] = w = tw;  // ATTENTION
                    }
                    d[k][y] = min(d[k][y], d[k][x] + w);
                }
            }
        };

        dijkstra(0);
        if (d[0][t] > target)   // -1 全改 1 时最短路也超长
            return {};
        diff = target - d[0][t];

        dijkstra(1);
        if (d[1][t] < target)   // ATTENTION 最短路最长也就这样了
            return {};
        
        for (auto & e : edges)
            if (e[2] == -1)
                e[2] = 1;
        return edges;
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

### bellmanford

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

> [!NOTE] **[LeetCode 787. K 站中转内最便宜的航班](https://leetcode.cn/problems/cheapest-flights-within-k-stops/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准 bellman ford

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int INF = 1e8;

    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        vector<int> d(n, INF);
        d[src] = 0;
        k ++ ;
        while (k -- ) {
            auto cur = d;
            for (auto & e : flights) {
                int a = e[0], b = e[1], c = e[2];
                cur[b] = min(cur[b], d[a] + c);
            }
            d = cur;
        }
        return d[dst] == INF ? -1 : d[dst];
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

> [!NOTE] **[Codeforces Jzzhu and Cities](https://codeforces.com/problemset/problem/449/B)**
> 
> 题意: 
> 
> $n$ 个点，$m$ 条带权边的无向图，另外还有 $k$ 条特殊边，每条边连接 $1$ 和 $i$ 。
> 
> 问最多可以删除这 $k$ 条边中的多少条，使得每个点到 $1$ 的最短距离不变。

> [!TIP] **思路**
> 
> 思路：存图后判断最短路是否唯一。
> 
> 判断最短路是否唯一的方法：
> 
> - 若特殊边长度大于 $dis[v]$，删除。
> 
> - 若特殊边长度等于 $dis[v]$，若到 $v$ 的最短路不止一条，删除，同时最短路数量 $-1$。
> 
> - 在求最短路时更新 $v$ 的最短路数量。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Jzzhu and Cities
// Contest: Codeforces - Codeforces Round #257 (Div. 1)
// URL: https://codeforces.com/problemset/problem/449/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const static int N = 1e5 + 10, M = 8e5 + 10;

int h[N], e[M], w[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

// cnt 最短路计数
int dist[N], cnt[N];
bool st[N];
void dijkstra(int s) {
    memset(st, 0, sizeof st);
    memset(cnt, 0, sizeof cnt);
    memset(dist, 0x3f, sizeof dist);
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, s});
    dist[s] = 0;
    while (heap.size()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (st[u])
            continue;
        st[u] = true;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i], c = w[i];
            if (dist[j] > d + c) {
                dist[j] = d + c;
                cnt[j] = 1;  // ATTENTION
                heap.push({dist[j], j});
            } else if (dist[j] == d + c)
                cnt[j]++;
        }
    }
}

int n, m, k;
PII es[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n >> m >> k;
    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    for (int i = 0; i < k; ++i) {
        cin >> es[i].first >> es[i].second;
        add(1, es[i].first, es[i].second), add(es[i].first, 1, es[i].second);
    }

    dijkstra(1);

    int res = 0;
    for (int i = 0; i < k; ++i) {
        int b = es[i].first, c = es[i].second;
        if (dist[b] < c)
            res++;
        else if (dist[b] == c && cnt[b] > 1)  // ATTENTION
            res++, cnt[b]--;
    }
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


### spfa

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

> [!NOTE] **[AcWing 361. 观光奶牛](https://www.acwing.com/problem/content/363/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推导 二分 转化为负环判定
> 
> SPFA

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


### floyd


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

> [!NOTE] **[LeetCode 1761. 一个图中连通三元组的最小度数](https://leetcode.cn/problems/minimum-degree-of-a-connected-trio-in-a-graph/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 无向图最小环 floyd 模板题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
bool st[1000000];

class Solution {
public:
    int minTrioDegree(int n, vector<vector<int>>& edges) {
        memset(st, 0, sizeof st);
        vector<int> d(n + 1);
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            if (a > b) swap(a, b);
            d[a] ++ , d[b] ++ ;
            st[a * 1000 + b] = true;
        }
        
        int res = INT_MAX;
        for (int i = 1; i <= n; ++ i )
            for (int j = i + 1; j <= n; ++ j )
                if (st[i * 1000 + j])
                    for (int k = j + 1; k <= n; ++ k )
                        if (st[i * 1000 + k] && st[j * 1000 + k])
                            res = min(res, d[i] + d[j] + d[k] - 6);
        if (res == INT_MAX) res = -1;
        return res;
    }
};
```

##### **C++ 老标准**

```cpp
class Solution {
public:
    int f[501][501];
    int deg[501];
    
    int minTrioDegree(int n, vector<vector<int>>& edges) {
        int m = edges.size();
        for (int i = 1; i <= n; i ++) deg[i] = 0;
        int ans = 0x3f3f3f3f;
        for (int i = 1; i <= n; i ++)
            for (int j = i + 1; j <= n; j ++)
                f[i][j] = false;
        for (int i = 0; i < m; i ++) {
            int x = edges[i][0], y = edges[i][1];
            if (x > y) swap(x, y);
            ++ deg[x]; ++ deg[y];
            f[x][y] = true;
        }
        for (int i = 1; i <= n; i ++) {
            for (int j = i + 1; j <= n; j ++) {
                if (f[i][j] == false) continue;
                for (int k = j + 1; k <= n; k ++) {
                    if (f[i][k] == false || f[j][k] == false) continue;
                    ans = min(ans, deg[i] + deg[j] + deg[k] - 6);
                }
            }
        }
        if (ans == 0x3f3f3f3f) return -1;
        return ans;
    }
};
```

##### **C++ 新标准**

```cpp
class Solution {
public:
    const static int N = 410, INF = 0x3f3f3f3f;

    int din[N];
    bool g[N][N];

    int minTrioDegree(int n, vector<vector<int>>& edges) {
        memset(din, 0, sizeof din);
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            // if (a > b)
            //     swap(a, b);
            din[a] ++ , din[b] ++ ;
            g[a][b] = g[b][a] = true;
        }

        int res = INF;
        for (int k = 1; k <= n; ++ k )
            for (int i = 1; i < k; ++ i )
                for (int j = 1; j < i; ++ j ) {
                    if (!g[i][j] || !g[j][k] || !g[i][k])
                        continue;
                    res = min(res, din[i] + din[j] + din[k] - 6);
                }
        if (res == INF)
            return -1;
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

> [!NOTE] **[Codeforces Greg and Graph](http://codeforces.com/problemset/problem/295/B)**
> 
> 题意: 
> 
> 逆序还原，每次加一个点

> [!TIP] **思路**
> 
> **深刻理解 floyd 本质**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Greg and Graph
// Contest: Codeforces - Codeforces Round #179 (Div. 1)
// URL: https://codeforces.com/problemset/problem/295/B
// Memory Limit: 256 MB
// Time Limit: 3000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 510;
const static LL INF = 1e18;

int n;
LL g[N][N];
int vs[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            cin >> g[i][j];

    for (int i = n; i >= 1; --i)
        cin >> vs[i];

    vector<LL> res;
    for (int i = 1; i <= n; ++i) {
        // ATTENTION 这里更新矩阵范围必须到 n
        // 深刻理解 floyd
        for (int j = 1; j <= n; ++j)
            for (int k = 1; k <= n; ++k) {
                int a = vs[j], b = vs[k], c = vs[i];
                if (g[a][b] > g[a][c] + g[c][b])
                    g[a][b] = g[a][c] + g[c][b];
            }

        LL c = 0;
        for (int j = 1; j <= i; ++j)
            for (int k = 1; k <= i; ++k) {
                int a = vs[j], b = vs[k];
                c += g[a][b];
            }
        res.push_back(c);
    }
    for (int i = n - 1; i >= 0; --i)
        cout << res[i] << ' ';
    cout << endl;

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

> [!NOTE] **[LeetCode 2642. 设计可以求最短路径的图类](https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准的 floyd 思想：每次增加一条边，求任意点间的最短路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Graph {
public:
    const static int N = 110, M = 2e4 + 10, INF = 0x3f3f3f3f; // 有向图
    
    int n;
    int d[N][N];
    
    Graph(int n, vector<vector<int>>& edges) {
        this->n = n;
        memset(d, 0x3f, sizeof d);
        for (int i = 0; i < n; ++ i )
            d[i][i] = 0;
        for (auto & e : edges)
            d[e[0]][e[1]] = e[2];
        for (int k = 0; k < n; ++ k )
            for (int i = 0; i < n; ++ i )
                for (int j = 0; j < n; ++ j )
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
    }
    
    void addEdge(vector<int> edge) {
        int a = edge[0], b = edge[1], c = edge[2];
        if (d[a][b] <= c)
            return;
        // floyd 标准的 dp 思想
        d[a][b] = c;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < n; ++ j )
                d[i][j] = min(d[i][j], d[i][a] + d[a][b] + d[b][j]);
    }
    
    int shortestPath(int node1, int node2) {
        return d[node1][node2] < INF / 2 ? d[node1][node2] : -1;
    }
};

/**
 * Your Graph object will be instantiated and called as such:
 * Graph* obj = new Graph(n, edges);
 * obj->addEdge(edge);
 * int param_2 = obj->shortestPath(node1,node2);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 拆点最短路 -> TODO 放到graph子章节

> [!NOTE] **[LeetCode 1928. 规定时间内到达终点的最小花费](https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/)** [TAG]
> 
> [biweekly 56](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-07-10_Biweekly-56)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 经典问题：无向连通图（有无自环无所谓），二维最短路问题
> 
>   ===> 拆点，每个点在每个时间下的状态单独作为一个点
> 
>   对于本题共计 1e6 个点，直接拆点跑最短路即可
> 
>   **对于二维费用的最短路进行拆点再跑最短路**
> 
> - 也可以 dp 拆状态然后递推
> 
>   若按照时间的升序转移，由于图中边权均为正，从当前时间出发是不可能转移到过去的时间上的，从而保证状态无后效性

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

> [!NOTE] **[AcWing 2045. 到达目的地的第二短时间](https://leetcode.cn/problems/second-minimum-time-to-reach-destination/)**
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
> $n - 1 <= edges.length <= min(2 e 10 ^ 4, n * (n - 1) / 2)$

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

> [!NOTE] **[LeetCode 建信03. 地铁路线规划](https://leetcode.cn/contest/ccbft-2021fall/problems/zQTFs4/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典最短路变形

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 10010, M = 20010;
    
    int n;
    int h[N], e[M], w[M], ne[M], idx;
    int belong[N];
    int st, ed;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
        
        memset(belong, -1, sizeof belong);
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    // TLE vector
    vector<int> res, path;
    bool vis[N];
    int ans;
    //         当前点    次数     当前线路
    void dfs(int u, int cost, int last) {
        // 剪枝
        if (cost > ans)
            return;
        if (u == ed) {
            if (cost < ans || (cost == ans && path < res)) {
                ans = cost;
                res = path;
            }
            return;
        }
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i], next = belong[i];
            if (!vis[j]) {
                vis[j] = true;
                path.push_back(j);
                dfs(j, cost + (next != last), next);
                path.pop_back();
                vis[j] = false;
            }
        }
    }
    
    vector<int> metroRouteDesignI(vector<vector<int>>& lines, int start, int end) {
        init();
        this->n = lines.size(), this->st = start, this->ed = end;
        for (int i = 0; i < n; ++ i ) {
            auto line = lines[i];
            int len = line.size();
            for (int j = 1; j < len; ++ j ) {
                belong[idx] = i;
                add(line[j - 1], line[j]);
                belong[idx] = i;
                add(line[j], line[j - 1]);
            }
        }
        
        memset(vis, 0, sizeof vis);
        this->ans = INT_MAX;
        this->res = this->path = vector<int>();
        
        // 起点可以选择不同的线路
        for (int i = h[st]; ~i; i = ne[i]) {
            int next = belong[i];
            vis[st] = true;
            path.push_back(st);
            dfs(st, 0, next);
            path.pop_back();
            vis[st] = false;
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