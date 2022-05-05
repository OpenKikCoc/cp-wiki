## 定义

最近公共祖先简称 LCA（Lowest Common Ancestor）。两个节点的最近公共祖先，就是这两个点的公共祖先里面，离根最远的那个。
为了方便，我们记某点集 $S={v_1,v_2,\ldots,v_n}$ 的最近公共祖先为 $\text{LCA}(v_1,v_2,\ldots,v_n)$ 或 $\text{LCA}(S)$。

## 性质

> 本节 **性质** 部分内容翻译自 [wcipeg](http://wcipeg.com/wiki/Lowest_common_ancestor)，并做过修改。

1. $\text{LCA}({u})=u$；
2. $u$ 是 $v$ 的祖先，当且仅当 $\text{LCA}(u,v)=u$；
3. 如果 $u$ 不为 $v$ 的祖先并且 $v$ 不为 $u$ 的祖先，那么 $u,v$ 分别处于 $\text{LCA}(u,v)$ 的两棵不同子树中；
4. 前序遍历中，$\text{LCA}(S)$ 出现在所有 $S$ 中元素之前，后序遍历中 $\text{LCA}(S)$ 则出现在所有 $S$ 中元素之后；
5. 两点集并的最近公共祖先为两点集分别的最近公共祖先的最近公共祖先，即 $\text{LCA}(A\cup B)=\text{LCA}(\text{LCA}(A), \text{LCA}(B))$；
6. 两点的最近公共祖先必定处在树上两点间的最短路上；
7. $d(u,v)=h(u)+h(v)-2h(\text{LCA}(u,v))$，其中 $d$ 是树上两点间的距离，$h$ 代表某点到树根的距离。

## 求法

### 朴素算法

可以每次找深度比较大的那个点，让它向上跳。显然在树上，这两个点最后一定会相遇，相遇的位置就是想要求的 LCA。
或者先向上调整深度较大的点，令他们深度相同，然后再共同向上跳转，最后也一定会相遇。

朴素算法预处理时需要 dfs 整棵树，时间复杂度为 $O(n)$，单次查询时间复杂度为 $\Theta(n)$。但由于随机树高为 $O(\log n)$，所以朴素算法在随机树上的单次查询时间复杂度为 $O(\log n)$。

### 倍增算法

倍增算法是最经典的 LCA 求法，他是朴素算法的改进算法。通过预处理 $\text{fa}_{x,i}$ 数组，游标可以快速移动，大幅减少了游标跳转次数。$\text{fa}_{x,i}$ 表示点 $x$ 的第 $2^i$ 个祖先。$\text{fa}_{x,i}$ 数组可以通过 dfs 预处理出来。

现在我们看看如何优化这些跳转：
在调整游标的第一阶段中，我们要将 $u,v$ 两点跳转到同一深度。我们可以计算出 $u,v$ 两点的深度之差，设其为 $y$。通过将 $y$ 进行二进制拆分，我们将 $y$ 次游标跳转优化为「$y$ 的二进制表示所含 `1` 的个数」次游标跳转。
在第二阶段中，我们从最大的 $i$ 开始循环尝试，一直尝试到 $0$（包括 $0$），如果 $\text{fa}_{u,i}\not=\text{fa}_{v,i}$，则 $u\gets\text{fa}_{u,i},v\gets\text{fa}_{v,i}$，那么最后的 LCA 为 $\text{fa}_{u,0}$。

倍增算法的预处理时间复杂度为 $O(n \log n)$，单次查询时间复杂度为 $O(\log n)$。
另外倍增算法可以通过交换 `fa` 数组的两维使较小维放在前面。这样可以减少 cache miss 次数，提高程序效率。

> [!NOTE] 例题
> 
> [HDU 2586 How far away?](http://acm.hdu.edu.cn/showproblem.php?pid=2586)
> 
> 树上最短路查询。原题为多组数据，以下代码为针对单组数据的情况编写的。

可先求出 LCA，再结合性质 $7$ 进行解答。也可以直接在求 LCA 时求出结果。


```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#define MXN 50007
using namespace std;
std::vector<int> v[MXN];
std::vector<int> w[MXN];

int fa[MXN][31], cost[MXN][31], dep[MXN];
int n, m;
int a, b, c;

// dfs，用来为 lca 算法做准备。接受两个参数：dfs 起始节点和它的父亲节点。
void dfs(int root, int fno) {
    // 初始化：第 2^0 = 1 个祖先就是它的父亲节点，dep 也比父亲节点多 1。
    fa[root][0] = fno;
    dep[root] = dep[fa[root][0]] + 1;
    // 初始化：其他的祖先节点：第 2^i 的祖先节点是第 2^(i-1) 的祖先节点的第
    // 2^(i-1) 的祖先节点。
    for (int i = 1; i < 31; ++i) {
        fa[root][i] = fa[fa[root][i - 1]][i - 1];
        cost[root][i] = cost[fa[root][i - 1]][i - 1] + cost[root][i - 1];
    }
    // 遍历子节点来进行 dfs。
    int sz = v[root].size();
    for (int i = 0; i < sz; ++i) {
        if (v[root][i] == fno) continue;
        cost[v[root][i]][0] = w[root][i];
        dfs(v[root][i], root);
    }
}

// lca。用倍增算法算取 x 和 y 的 lca 节点。
int lca(int x, int y) {
    // 令 y 比 x 深。
    if (dep[x] > dep[y]) swap(x, y);
    // 令 y 和 x 在一个深度。
    int tmp = dep[y] - dep[x], ans = 0;
    for (int j = 0; tmp; ++j, tmp >>= 1)
        if (tmp & 1) ans += cost[y][j], y = fa[y][j];
    // 如果这个时候 y = x，那么 x，y 就都是它们自己的祖先。
    if (y == x) return ans;
    // 不然的话，找到第一个不是它们祖先的两个点。
    for (int j = 30; j >= 0 && y != x; --j) {
        if (fa[x][j] != fa[y][j]) {
            ans += cost[x][j] + cost[y][j];
            x = fa[x][j];
            y = fa[y][j];
        }
    }
    // 返回结果。
    ans += cost[x][0] + cost[y][0];
    return ans;
}

int main() {
    // 初始化表示祖先的数组 fa，代价 cost 和深度 dep。
    memset(fa, 0, sizeof(fa));
    memset(cost, 0, sizeof(cost));
    memset(dep, 0, sizeof(dep));
    // 读入树：节点数一共有 n 个。
    scanf("%d", &n);
    for (int i = 1; i < n; ++i) {
        scanf("%d %d %d", &a, &b, &c);
        ++a, ++b;
        v[a].push_back(b);
        v[b].push_back(a);
        w[a].push_back(c);
        w[b].push_back(c);
    }
    // 为了计算 lca 而使用 dfs。
    dfs(1, 0);
    // 查询 m 次，每一次查找两个节点的 lca 点。
    scanf("%d", &m);
    for (int i = 0; i < m; ++i) {
        scanf("%d %d", &a, &b);
        ++a, ++b;
        printf("%d\n", lca(a, b));
    }
    return 0;
}
```

### Tarjan 算法

`Tarjan 算法` 是一种 `离线算法`，需要使用 `并查集` 记录某个结点的祖先结点。做法如下：

1. 首先接受输入（邻接链表）、查询（存储在另一个邻接链表内）。查询边其实是虚拟加上去的边，为了方便，每次输入查询边的时候，将这个边及其反向边都加入到 `queryEdge` 数组里。
2. 然后对其进行一次 DFS 遍历，同时使用 `visited` 数组进行记录某个结点是否被访问过、`parent` 记录当前结点的父亲结点。
3. 其中涉及到了 `回溯思想`，我们每次遍历到某个结点的时候，认为这个结点的根结点就是它本身。让以这个结点为根节点的 DFS 全部遍历完毕了以后，再将 `这个结点的根节点` 设置为 `这个结点的父一级结点`。
4. 回溯的时候，如果以该节点为起点，`queryEdge` 查询边的另一个结点也恰好访问过了，则直接更新查询边的 LCA 结果。
5. 最后输出结果。

Tarjan 算法需要初始化并查集，所以预处理的时间复杂度为 $O(n)$，Tarjan 算法处理所有 $m$ 次询问的时间复杂度为 $O(n + m)$。但是 Tarjan 算法的常数比倍增算法大。

需要注意的是，Tarjan 算法中使用的并查集性质比较特殊，在仅使用路径压缩优化的情况下，单次调用 `find()` 函数的时间复杂度为均摊 $O(1)$，而不是 $O(\log n)$。具体可以见 [并查集部分的引用：A Linear-Time Algorithm for a Special Case of Disjoint Set Union](ds/dsu.md#references)。


```cpp
#include <algorithm>
#include <iostream>
using namespace std;

class Edge {
public:
    int toVertex, fromVertex;
    int next;
    int LCA;
    Edge() : toVertex(-1), fromVertex(-1), next(-1), LCA(-1){};
    Edge(int u, int v, int n) : fromVertex(u), toVertex(v), next(n), LCA(-1){};
};

const int MAX = 100;
int head[MAX], queryHead[MAX];
Edge edge[MAX], queryEdge[MAX];
int parent[MAX], visited[MAX];
int vertexCount, edgeCount, queryCount;

void init() {
    for (int i = 0; i <= vertexCount; i++) { parent[i] = i; }
}

int find(int x) {
    if (parent[x] == x) {
        return x;
    } else {
        return find(parent[x]);
    }
}

void tarjan(int u) {
    parent[u] = u;
    visited[u] = 1;

    for (int i = head[u]; i != -1; i = edge[i].next) {
        Edge& e = edge[i];
        if (!visited[e.toVertex]) {
            tarjan(e.toVertex);
            parent[e.toVertex] = u;
        }
    }

    for (int i = queryHead[u]; i != -1; i = queryEdge[i].next) {
        Edge& e = queryEdge[i];
        if (visited[e.toVertex]) {
            queryEdge[i ^ 1].LCA = e.LCA = find(e.toVertex);
        }
    }
}

int main() {
    memset(head, 0xff, sizeof(head));
    memset(queryHead, 0xff, sizeof(queryHead));

    cin >> vertexCount >> edgeCount >> queryCount;
    int count = 0;
    for (int i = 0; i < edgeCount; i++) {
        int start = 0, end = 0;
        cin >> start >> end;

        edge[count] = Edge(start, end, head[start]);
        head[start] = count;
        count++;

        edge[count] = Edge(end, start, head[end]);
        head[end] = count;
        count++;
    }

    count = 0;
    for (int i = 0; i < queryCount; i++) {
        int start = 0, end = 0;
        cin >> start >> end;

        queryEdge[count] = Edge(start, end, queryHead[start]);
        queryHead[start] = count;
        count++;

        queryEdge[count] = Edge(end, start, queryHead[end]);
        queryHead[end] = count;
        count++;
    }

    init();
    tarjan(1);

    for (int i = 0; i < queryCount; i++) {
        Edge& e = queryEdge[i * 2];
        cout << **(" << e.fromVertex << **," << e.toVertex << **)** << e.LCA
             << endl;
    }

    return 0;
}
```

### 用欧拉序列转化为 RMQ 问题

对一棵树进行 DFS，无论是第一次访问还是回溯，每次到达一个结点时都将编号记录下来，可以得到一个长度为 $2n-1$ 的序列，这个序列被称作这棵树的欧拉序列。

在下文中，把结点 $u$ 在欧拉序列中第一次出现的位置编号记为 $pos(u)$（也称作节点 $u$ 的欧拉序），把欧拉序列本身记作 $E[1..2n-1]$。

有了欧拉序列，LCA 问题可以在线性时间内转化为 RMQ 问题，即 $pos(LCA(u, v))=\min\{pos(k)|k\in E[pos(u)..pos(v)]\}$。

这个等式不难理解：从 $u$ 走到 $v$ 的过程中一定会经过 $LCA(u,v)$，但不会经过 $LCA(u,v)$ 的祖先。因此，从 $u$ 走到 $v$ 的过程中经过的欧拉序最小的结点就是 $LCA(u, v)$。

用 DFS 计算欧拉序列的时间复杂度是 $O(n)$，且欧拉序列的长度也是 $O(n)$，所以 LCA 问题可以在 $O(n)$ 的时间内转化成等规模的 RMQ 问题。


```cpp
int dfn[N << 1], dep[N << 1], dfntot = 0;
void dfs(int t, int depth) {
    dfn[++dfntot] = t;
    pos[t] = dfntot;
    dep[dfntot] = depth;
    for (int i = head[t]; i; i = side[i].next) {
        dfs(side[i].to, t, depth + 1);
        dfn[++dfntot] = t;
        dep[dfntot] = depth;
    }
}
void st_preprocess() {
    lg[0] = -1;  // 预处理 lg 代替库函数 log2 来优化常数
    for (int i = 1; i <= (N << 1); ++i) lg[i] = lg[i >> 1] + 1;
    for (int i = 1; i <= (N << 1) - 1; ++i) st[0][i] = dfn[i];
    for (int i = 1; i <= lg[(N << 1) - 1]; ++i)
        for (int j = 1; j + (1 << i) - 1 <= ((N << 1) - 1); ++j)
            st[i][j] = dep[st[i - 1][j]] < dep[st[i - 1][j + (1 << i - 1)]
                            ? st[i - 1][j]
                            : st[i - 1][j + (1 << i - 1)];
}
```

当我们需要查询某点对 $(u, v)$ 的 LCA 时，查询区间 $[\min\{pos[u], pos[v]\}, \max\{pos[u], pos[v]\}]$ 上最小值的所代表的节点即可。

若使用 ST 表来解决 RMQ 问题，那么该算法不支持在线修改，预处理的时间复杂度为 $O(n\log n)$，每次查询 LCA 的时间复杂度为 $O(1)$。

### 树链剖分

LCA 为两个游标跳转到同一条重链上时深度较小的那个游标所指向的点。

树链剖分的预处理时间复杂度为 $O(n)$，单次查询的时间复杂度为 $O(\log n)$，并且常数较小。

### [动态树](ds/lct.md)

设连续两次 [access](ds/lct.md#access) 操作的点分别为 `u` 和 `v`，则第二次 [access](ds/lct.md#access) 操作返回的点即为 `u` 和 `v` 的 LCA.

在无 link 和 cut 等操作的情况下，使用 link cut tree 单次查询的时间复杂度为 $O(\log n)$。

### 标准 RMQ

前面讲到了借助欧拉序将 LCA 问题转化为 RMQ 问题，其瓶颈在于 RMQ。如果能做到 $O(n) \sim O(1)$ 求解 RMQ，那么也就能做到 $O(n) \sim O(1)$ 求解 LCA。

注意到欧拉序满足相邻两数之差为 1 或者 -1，所以可以使用 $O(n) \sim O(1)$ 的 [加减 1RMQ](topic/rmq.md#1rmq) 来做。

时间复杂度 $O(n) \sim O(1)$，空间复杂度 $O(n)$，支持在线查询，常数较大。

#### 例题 [Luogu P3379【模板】最近公共祖先（LCA）](https://www.luogu.com.cn/problem/P3379)



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

## 习题

- [祖孙询问](https://loj.ac/problem/10135)
- [货车运输](https://loj.ac/problem/2610)
- [点的距离](https://loj.ac/problem/10130)

### 一般 LCA

> [!NOTE] **[AcWing 1172. 祖孙询问](https://www.acwing.com/problem/content/1174/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 最近公共祖先
> 
> **倍增LCA**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 40010, M = N * 2;

int n, m;
int h[N], e[M], ne[M], idx;
int depth[N], fa[N][16];
int q[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void bfs(int root) {
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[root] = 1;
    int hh = 0, tt = 0;
    q[0] = root;
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (depth[j] > depth[t] + 1) {
                depth[j] = depth[t] + 1;
                q[ ++ tt] = j;
                
                fa[j][0] = t;
                for (int k = 1; k <= 15; ++ k )
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
            }
        }
    }
}

int lca(int a, int b) {
    if (depth[a] < depth[b])
        swap(a, b);
    for (int k = 15; k >= 0; -- k )
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b)
        return a;
    for (int k = 15; k >= 0; -- k )
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n;
    int root = 0;
    
    for (int i = 0; i < n; ++ i ) {
        int a, b;
        cin >> a >> b;
        if (b == -1)
            root = a;
        else
            add(a, b), add(b, a);
    }
    
    bfs(root);
    
    cin >> m;
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        int p = lca(a, b);
        if (p == a)
            cout << 1 << endl;
        else if (p == b)
            cout << 2 << endl;
        else
            cout << 0 << endl;
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

> [!NOTE] **[AcWing 1171. 距离](https://www.acwing.com/problem/content/1173/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> tarjan

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const int N = 10010, M = 20010;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
int dist[N];
int p[N];
int res[M];
int st[N];
vector<PII> query[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int find(int x) {
    if (p[x] != x)
        p[x] = find(p[x]);
    return p[x];
}

void dfs(int u, int fa) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dist[j] = dist[u] + w[i];
        dfs(j, u);
    }
}

void tarjan(int u) {
    st[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!st[j]) {
            tarjan(j);
            p[j] = u;
        }
    }
    
    for (auto [oth, id] : query[u])
        if (st[oth] == 2) {
            int anc = find(oth);
            res[id] = dist[u] + dist[oth] - dist[anc] * 2;
        }
    
    st[u] = 2;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    
    for (int i = 0; i < m; ++ i ) {
        int a, b;
        cin >> a >> b;
        if (a != b)
            query[a].push_back({b, i}), query[b].push_back({a, i});
    }
    
    for (int i = 1; i <= n; ++ i )
        p[i] = i;
    
    dfs(1, -1);
    tarjan(1);
    
    for (int i = 0; i < m; ++ i )
        cout << res[i] << endl;
    
    return 0;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

* * *

> [!NOTE] **[AcWing 356. 次小生成树](https://www.acwing.com/problem/content/description/358/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **次小生成树 倍增LCA优化**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 100010, M = 300010, INF = 0x3f3f3f3f;

int n, m;
struct Edge {
    int a, b, w;
    bool used;
    bool operator< (const Edge & t) const {
        return w < t.w;
    }
}edge[M];
int p[N];
int h[N], e[M], w[M], ne[M], idx;
int depth[N], fa[N][17], d1[N][17], d2[N][17];
int q[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int find(int x) {
    if (p[x] != x)
        p[x] = find(p[x]);
    return p[x];
}

LL kruskal() {
    for (int i = 1; i <= n; ++ i )
        p[i] = i;
    sort(edge, edge + m);
    
    LL res = 0;
    for (int i = 0; i < m; ++ i ) {
        int a = find(edge[i].a), b = find(edge[i].b), w = edge[i].w;
        if (a != b) {
            p[a] = b;
            res += w;
            edge[i].used = true;
        }
    }
    return res;
}

void build() {
    memset(h, -1, sizeof h);

    for (int i = 0; i < m; ++ i )
        if (edge[i].used) {
            int a = edge[i].a, b = edge[i].b, w = edge[i].w;
            add(a, b, w), add(b, a, w);
        }
}

void bfs() {
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[1] = 1;
    int hh = 0, tt = 0;
    q[0] = 1;   // ATTENTION
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (depth[j] > depth[t] + 1) {
                depth[j] = depth[t] + 1;
                q[ ++ tt] = j;
                
                fa[j][0] = t;
                d1[j][0] = w[i], d2[j][0] = -INF;
                for (int k = 1; k <= 16; ++ k ) {
                    int anc = fa[j][k - 1];
                    fa[j][k] = fa[anc][k - 1];
                    
                    int distance[4] = {d1[j][k - 1], d2[j][k - 1], d1[anc][k - 1], d2[anc][k - 1]};
                    d1[j][k] = d2[j][k] = -INF;
                    for (int u = 0; u < 4; ++ u ) {
                        int d = distance[u];
                        if (d > d1[j][k])
                            d2[j][k] = d1[j][k], d1[j][k] = d;
                        else if (d != d1[j][k] && d > d2[j][k])
                            d2[j][k] = d;
                    }
                }
            }
        }
    }
}

int lca(int a, int b, int w) {
    static int distance[N * 2];
    int cnt = 0;
    
    if (depth[a] < depth[b])
        swap(a, b);
    for (int k = 16; k >= 0; -- k )
        if (depth[fa[a][k]] >= depth[b]) {
            distance[cnt ++ ] = d1[a][k];
            distance[cnt ++ ] = d2[a][k];
            a = fa[a][k];
        }
    if (a != b) {
        for (int k = 16; k >= 0; -- k )
            if (fa[a][k] != fa[b][k]) {
                distance[cnt ++ ] = d1[a][k];
                distance[cnt ++ ] = d2[a][k];
                distance[cnt ++ ] = d1[b][k];
                distance[cnt ++ ] = d2[b][k];
                a = fa[a][k], b = fa[b][k];
            }
        distance[cnt ++ ] = d1[a][0];
        distance[cnt ++ ] = d1[b][0];
    }
    
    int dist1 = -INF, dist2 = -INF;
    for (int i = 0; i < cnt; ++ i ) {
        int d = distance[i];
        if (d > dist1)
            dist2 = dist1, dist1 = d;
        else if (d != dist1 && d > dist2)
            dist2 = d;
    }
    if (w > dist1)
        return w - dist1;
    if (w > dist2)
        return w - dist2;
    return INF;
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < m; ++ i ) {
        int a, b, c;
        cin >> a >> b >> c;
        edge[i] = {a, b, c};
    }
    
    LL sum = kruskal();
    build();
    bfs();
    
    LL res = 1e18;
    for (int i = 0; i < m; ++ i )
        if (!edge[i].used) {
            int a = edge[i].a, b = edge[i].b, w = edge[i].w;
            res = min(res, sum + lca(a, b, w));
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

> [!NOTE] **[Luogu [JLOI2009]二叉树问题](https://www.luogu.com.cn/problem/P3884)**
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
#include <bits/stdc++.h>
using namespace std;

const int N = 110, M = 8;

int n;
int h[N], e[N], ne[N], idx;
int depth[N], cnt[N], max_depth, max_cnt;
int fa[N][M], q[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void bfs(int root) {
    memset(depth, 0x3f, sizeof depth);
    memset(cnt, 0, sizeof cnt);
    
    depth[0] = 0, depth[root] = 1;
    int hh = 0, tt = -1;
    q[ ++ tt] = root;
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (depth[j] > depth[t] + 1) {
                depth[j] = depth[t] + 1;
                cnt[depth[j]] ++ ;
                
                q[ ++ tt] = j;
                fa[j][0] = t;
                for (int k = 1; k < M; ++ k )
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
            }
        }
    }
}

int lca(int a, int b) {
    if (depth[a] < depth[b])
        swap(a, b);
    for (int k = M - 1; k >= 0; -- k )
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b)
        return a;
    for (int k = M - 1; k >= 0; -- k )
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n;
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b);
    }
    
    bfs(1);
    
    for (int i = 1; i <= n; ++ i )
        max_depth = max(max_depth, depth[i]);
    for (int i = 1; i <= max_depth; ++ i )
        max_cnt = max(max_cnt, cnt[i]);
    
    int u, v;
    cin >> u >> v;
    
    int pa = lca(u, v);
    
    cout << max_depth << endl;
    cout << max_cnt << endl;
    cout << (depth[u] - depth[pa]) * 2 + (depth[v] - depth[pa]) << endl;
    
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

> [!NOTE] **[Luogu [USACO19DEC]Milk Visits S](https://www.luogu.com.cn/problem/P5836)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准LCA
> 
> **有 O(n) 的并查集做法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10, M = N << 1, K = 18;

int n, m;
char c[N];
int h[N], e[M], ne[M], idx;

int depth[N], fa[N][K], q[N];
int H[N], G[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void bfs(int root) {
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[root] = 1;
    H[root] = c[root] == 'H', G[root] = c[root] == 'G';
    int hh = 0, tt = -1;
    q[ ++ tt] = root;
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (depth[j] > depth[t] + 1) {
                depth[j] = depth[t] + 1;
                H[j] = H[t] + (c[j] == 'H'), G[j] = G[t] + (c[j] == 'G');
                q[ ++ tt] = j;
                fa[j][0] = t;
                for (int k = 1; k < K; ++ k )
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
            }
        }
    }
}

int lca(int a, int b) {
    if (depth[a] < depth[b])
        swap(a, b);
    for (int k = K - 1; k >= 0; -- k )
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b)
        return a;
    for (int k = K - 1; k >= 0; -- k )
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int main() {
    memset(h, -1, sizeof h);
    cin >> n >> m;
    cin >> (c + 1);
    
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    
    bfs(1);
    
    while (m -- ) {
        int a, b;
        char w;
        cin >> a >> b >> w;
        int p = lca(a, b);    
        if (w == 'H' && H[a] + H[b] - H[p] * 2 + (c[p] == 'H') > 0 ||
            w == 'G' && G[a] + G[b] - G[p] * 2 + (c[p] == 'G') > 0)
            cout << 1;
        else
            cout << 0;
    }
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


### 综合应用

> [!NOTE] **[AcWing 352. 闇の連鎖](https://www.acwing.com/problem/content/description/354/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 切两刀 需要为边增权
> 
> **树差分**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 100010, M = N * 2;

int n, m;
int h[N], e[M], ne[M], idx;
int depth[N], fa[N][17];
int d[N];
int q[N];
int ans;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void bfs() {
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[1] = 1;
    int hh = 0, tt = 0;
    q[0] = 1;
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

int lca(int a, int b) {
    if (depth[a] < depth[b]) swap(a, b);
    for (int k = 16; k >= 0; -- k )
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b) return a;
    for (int k = 16; k >= 0; -- k )
        if (fa[a][k] != fa[b][k]) {
            a = fa[a][k];
            b = fa[b][k];
        }
    return fa[a][0];
}

int dfs(int u, int father) {
    int res = d[u];
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != father) {
            int s = dfs(j, u);
            if (s == 0) ans += m;
            else if (s == 1) ans ++ ;
            res += s;
        }
    }
    return res;
}

int main() {
    cin >> n >> m;
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    bfs();
    
    for (int i = 0; i < m; ++ i ) {
        int a, b;
        cin >> a >> b;
        int p = lca(a, b);
        d[a] ++ , d[b] ++ , d[p] -= 2;
    }
    dfs(1, -1);
    cout << ans << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu 仓鼠找sugar](https://www.luogu.com.cn/problem/P3398)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 判断树上两路径相交的条件
> 
> 若两路径相交，则必有其中一条路径的LCA在另一条路径上
> 
> **LCA + 高度较低的LCA 与 高度较高的两点之一 的LCA相同即可**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10, M = N << 1, K = 18;

int n, m;
int h[N], e[M], ne[M], idx;
int depth[N], fa[N][K], q[N];

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
                for (int k = 1; k < K; ++ k )
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
            }
        }
    }
}

int lca(int a, int b) {
    if (depth[a] < depth[b])
        swap(a, b);
    for (int k = K - 1; k >= 0; -- k )
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b)
        return a;
    for (int k = K - 1; k >= 0; -- k )
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    
    bfs(1);
    
    while (m -- ) {
        int a, b, c, d;
        cin >> a >> b >> c >> d;
        int x = lca(a, b), y = lca(c, d);
        bool f;
        if (depth[x] < depth[y]) {
            f = (lca(a, y) == y || lca(b, y) == y);
        } else {
            f = (lca(c, x) == x || lca(d, x) == x);
        }
        cout << (f ? "Y" : "N") << endl;
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

> [!NOTE] **[Luogu [AHOI2008]紧急集合 / 聚会](https://www.luogu.com.cn/problem/P4281)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **LCA + 推导**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5e5 + 10, M = N << 1, K = 19;

int n, m;
int h[N], e[M], ne[M], idx;
int depth[N], fa[N][K], q[N];

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
                for (int k = 1; k < K; ++ k )
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
            }
        }
    }
}

int lca(int a, int b) {
    if (depth[a] < depth[b])
        swap(a, b);
    for (int k = K - 1; k >= 0; -- k )
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b)
        return a;
    for (int k = K - 1; k >= 0; -- k )
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    bfs(1);
    
    while (m -- ) {
        int a, b, c;
        cin >> a >> b >> c;
        
        int pab = lca(a, b), pac = lca(a, c), pbc = lca(b, c);
        int p = lca(pab, pbc);
        
        // 容易推理得出 要去的点一定在
        // [较低的lca & 较高的lca] 的路径上
        // 进一步推导 应该在较低的lca上
        if (p != pab) {
            int t = depth[pab] + depth[c] - depth[p] * 2;
            int d = depth[a] + depth[b] - depth[pab] * 2 + t;
            cout << pab << ' ' << d << endl;
        } else if (p != pac) {
            int t = depth[pac] + depth[b] - depth[p] * 2;
            int d = depth[a] + depth[c] - depth[pac] * 2 + t;
            cout << pac << ' ' << d << endl;
        } else {
            int t = depth[pbc] + depth[a] - depth[p] * 2;
            int d = depth[b] + depth[c] - depth[pbc] * 2 + t;
            cout << pbc << ' ' << d << endl;
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

> [!NOTE] **[Luogu [CSP-S2019] 树的重心](https://www.luogu.com.cn/problem/P5666)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dfs类LCA 经典倍增优化题
> 
> 经典 换根 + 倍增
> 
> 断断续续写了两天独立A掉 好题
> 
> Luogu 和网络上各类题解及其代码都特别麻烦。。。
> 
> **本质上就是枚举删哪条边，同时利用倍增快速找到重心，以及利用换根降低计算复杂度。**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 3e5 + 10, M = N << 1, K = 20; // 2^19

int t, n;
int h[N], e[M], ne[M], idx;
int son[N][K], sz[N], pa[N], w[N][2];
LL res;

void init() {
    memset(h, -1, sizeof h);
    idx = 0;
    res = 0;
    memset(sz, 0, sizeof sz);   // sz[0] = 0;
    memset(son, 0, sizeof son);
    memset(w, 0, sizeof w);
    memset(pa, 0, sizeof pa);
}

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void update_st(int u) {
    for (int k = 1; k < K; ++ k )
        son[u][k] = son[son[u][k - 1]][k - 1];
}

void calc(int u) {
    int x = u;
    for (int k = K - 1; k >= 0; -- k )
        if (son[x][k] && sz[son[x][k]] * 2 >= sz[u])
            x = son[x][k];
    // 如果某子树大小刚好是根树的一半 则子树及子树的父节点都是重心
    if (sz[x] * 2 == sz[u])
        res += (LL)pa[x];
    res += (LL)x;
}

void dfs_d(int u, int fa) {
    sz[u] = 1, pa[u] = fa;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa) {
            dfs_d(j, u);
            
            sz[u] += sz[j];
            // 考虑断开某条边后 换根后需要对是否是重链做判读 预处理
            // 重儿子和次重儿子
            if (sz[j] >= sz[w[u][0]])
                w[u][1] = w[u][0], w[u][0] = j;
            else if (sz[j] >= sz[w[u][1]])
                w[u][1] = j;
        }
    }
    
    son[u][0] = w[u][0];
    update_st(u);
}

void dfs_u(int u, int fa) {
    // t1: 在换根过程中 sz[u] 会变，使用临时变量 t1 记录
    // t2: 换根过程中 u 的父节点会作为 u 的儿子，会修改 pa[fa]，故使用临时变量记录
    int t1 = sz[u], t2 = pa[fa];
    pa[fa] = u;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa) {
            // 1. 计算 j 作为根的树
            calc(j);
 
            // 2. 计算 u 作为根的树
            //   2.1 先在向下的方向找个最大的
            son[u][0] = (j == w[u][0] ? w[u][1] : w[u][0]);
            //   2.2 向上的方向继续更新
            if (sz[son[u][0]] < n - t1)
                son[u][0] = pa[u];
            update_st(u);
            sz[u] = n - sz[j];
            calc(u);

            // 3. 递归
            dfs_u(j, u);
        }
    }
    sz[u] = t1, pa[fa] = t2;
    son[u][0] = w[u][0];
    update_st(u);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    
    cin >> t;
    while (t -- ) {
        init();
        
        cin >> n;
        for (int i = 0; i < n - 1; ++ i ) {
            int a, b;
            cin >> a >> b;
            add(a, b), add(b, a);
        }
        
        // 0 instead of -1
        dfs_d(1, 0);
        dfs_u(1, 0);
        
        cout << res << endl;
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

> [!NOTE] **[LeetCode 1257. 最小公共区域](https://leetcode-cn.com/problems/smallest-common-region/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> LCA思想 实现TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    unordered_map<string, vector<string>> mp;
    unordered_map<string, bool> in;
    string dfs(string& t, string& s1, string& s2) {
        if (t == s1 || t == s2) return t;
        if (mp[t].empty()) return "";
        string res;
        int cnt = 0;
        for (auto& s : mp[t]) {
            string ret = dfs(s, s1, s2);
            if (ret != "") res = ret, ++cnt;
        }
        if (cnt == 1)
            return res;
        else if (cnt == 2)
            return t;
        return "";
    }
    string findSmallestRegion(vector<vector<string>>& regions, string region1,
                              string region2) {
        vector<string> out;
        for (auto& regs : regions) {
            int sz = regs.size();
            out.push_back(regs[0]);
            for (int i = 1; i < sz; ++i) {
                mp[regs[0]].push_back(regs[i]);
                in[regs[i]] = true;
            }
        }
        vector<string> ve;
        for (auto t : out)
            if (in[t] == false) ve.push_back(t);
        string res;
        for (auto& t : ve) {
            res = dfs(t, region1, region2);
            if (res != "") return res;
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

> [!NOTE] **[Codeforces A and B and Lecture Rooms](http://codeforces.com/problemset/problem/519/E)**
> 
> 题意: 
> 
> 题目的意思就是每次询问点 $x,y$ ，要你求其中离它距离一样的点。

> [!TIP] **思路**
> 
> 分情况讨论即可，需要 LCA 优化找点的过程
> 
> 具体的：需要找到中间点 (LCA)，并排除两个点所在的子树，随后计算大小 (dfs 统计树大小)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: E. A and B and Lecture Rooms
// Contest: Codeforces - Codeforces Round #294 (Div. 2)
// URL: https://codeforces.com/problemset/problem/519/E
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10, M = 2e5 + 10;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int depth[N], fa[N][18];
int q[N];
void bfs(int root) {
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[root] = 1;
    int hh = 0, tt = -1;
    q[++tt] = root;
    while (hh <= tt) {
        int t = q[hh++];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (depth[j] > depth[t] + 1) {
                depth[j] = depth[t] + 1;
                q[++tt] = j;

                fa[j][0] = t;
                for (int k = 1; k < 18; ++k)
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
            }
        }
    }
}
int lca(int a, int b) {
    if (depth[a] < depth[b])
        swap(a, b);
    for (int k = 17; k >= 0; --k)
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if (a == b)
        return a;
    for (int k = 17; k >= 0; --k)
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int sz[N];
void dfs(int u, int fa) {
    sz[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs(j, u);
        sz[u] += sz[j];
    }
}

// 由 x 向上爬到 dep 的深度
int climpup(int x, int dep) {
    for (int k = 17; k >= 0; --k)
        if (depth[fa[x][k]] >= dep)
            x = fa[x][k];
    return x;
}

int n, m;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n;
    for (int i = 0; i < n - 1; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    bfs(1);
    dfs(1, -1);

    cin >> m;
    while (m--) {
        int x, y;
        cin >> x >> y;
        int pa = lca(x, y);
        int dis = depth[x] + depth[y] - depth[pa] * 2;
        // cout << "dis = " << dis << endl;
        if (dis & 1)
            cout << 0 << endl;
        else {
            // 分情况讨论

            // find more
            // 1. find the middle node
            // 2. expand the nodes
            int dx = depth[x] - depth[pa], dy = depth[y] - depth[pa];

            if (dx == dy) {
                if (x == y)
                    cout << n << endl;
                else {
                    int res = n;
                    {
                        int dep = depth[x] - dis / 2 + 1;
                        int node = climpup(x, dep);
                        res -= sz[node];
                    }
                    {
                        int dep = depth[y] - dis / 2 + 1;
                        int node = climpup(y, dep);
                        res -= sz[node];
                    }
                    cout << res << endl;
                }
            } else {
                // dx != dy
                if (dx < dy) {
                    int dep = depth[y] - dis / 2 + 1;
                    int node = climpup(y, dep);
                    cout << sz[fa[node][0]] - sz[node] << endl;
                } else {
                    // dx > dy
                    int dep = depth[x] - dis / 2 + 1;
                    int node = climpup(x, dep);
                    cout << sz[fa[node][0]] - sz[node] << endl;
                }
            }
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
