## 点分治

点分治适合处理大规模的树上路径信息问题。

> [!NOTE] **例题 [luogu P3806【模板】点分治 1](https://www.luogu.com.cn/problem/P3806)**
> 
> 给定一棵有 $n$ 个点的带边权树，$m$ 次询问，每次询问给出 $k$，询问树上距离为 $k$ 的点对是否存在。
> 
> $n\le 10000,m\le 100,k\le 10000000$

我们先随意选择一个节点作为根节点 $\mathit{rt}$，所有完全位于其子树中的路径可以分为两种，一种是经过当前根节点的路径，一种是不经过当前根节点的路径。对于经过当前根节点的路径，又可以分为两种，一种是以根节点为一个端点的路径，另一种是两个端点都不为根节点的路径。而后者又可以由两条属于前者链合并得到。所以，对于枚举的根节点 $rt$，我们先计算在其子树中且经过该节点的路径对答案的贡献，再递归其子树对不经过该节点的路径进行求解。

在本题中，对于经过根节点 $\mathit{rt}$ 的路径，我们先枚举其所有子节点 $\mathit{ch}$，以 $\mathit{ch}$ 为根计算 $\mathit{ch}$ 子树中所有节点到 $\mathit{rt}$ 的距离。记节点 $i$ 到当前根节点 $rt$ 的距离为 $\mathit{dist}_i$，$\mathit{tf}_{d}$ 表示之前处理过的子树中是否存在一个节点 $v$ 使得 $\mathit{dist}_v=d$。若一个询问的 $k$ 满足 $tf_{k-\mathit{dist}_i}=true$，则存在一条长度为 $k$ 的路径。在计算完 $\mathit{ch}$ 子树中所连的边能否成为答案后，我们将这些新的距离加入 $\mathit{tf}$ 数组中。

注意在清空 $\mathit{tf}$ 数组的时候不能直接用 `memset`，而应将之前占用过的 $\mathit{tf}$ 位置加入一个队列中，进行清空，这样才能保证时间复杂度。

点分治过程中，每一层的所有递归过程合计对每个点处理一次，假设共递归 $h$ 层，则总时间复杂度为 $O(hn)$。

若我们 **每次选择子树的重心作为根节点**，可以保证递归层数最少，时间复杂度为 $O(n\log n)$。

请注意在重新选择根节点之后一定要重新计算子树的大小，否则一点看似微小的改动就可能会使时间复杂度错误或正确性难以保证。

代码：


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

> [!NOTE] **例题 [luogu  P4178 Tree](https://www.luogu.com.cn/problem/P4178)**
> 
> 给定一棵有 $n$ 个点的带权树，给出 $k$，询问树上距离小于等于 $k$ 的点对数量。
> 
> $n\le 40000,k\le 20000,w_i\le 1000$

由于这里查询的是树上距离为 $[0,k]$ 的点对数量，所以我们用线段树来支持维护和查询。


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

## 边分治

与上面的点分治类似，我们选取一条边，把树尽量均匀地分成两部分（使边连接的两个子树的 $\mathit{size}$ 尽量接近）。然后递归处理左右子树，统计信息。

ちょっとまって，这不行吧……

考虑一个菊花图

![菊花图](./images/tree-divide1.svg)

我们发现当一个点下有多个 $size$ 接近的儿子时，应用边分治的时间复杂度是无法接受的。

如果这个图是个二叉树，就可以避免上面菊花图中应用边分治的弊端。因此我们考虑把一个多叉树转化成二叉树。

显然，我们只需像线段树那样建树就可以了。就像这样

![建树](./images/tree-divide2.svg)

新建出来的点根据题目要求给予恰当的信息即可。例如：统计路径长度时，将原边边权赋为 $1$, 将新建的边边权赋为 $0$ 即可。

分析复杂度，发现最多会增加 $O(n)$ 个点，则总复杂度为 $O(n\log n)$

几乎所有点分治的题边分都能做（常数上有差距，但是不卡），所以就不放例题了。

## 点分树

点分树是通过更改原树形态使树的层数变为稳定 $\log n$ 的一种重构树。

常用于解决与树原形态无关的带修改问题。

### 算法分析

我们通过点分治每次找重心的方式来对原树进行重构。

将每次找到的重心与上一层的重心缔结父子关系，这样就可以形成一棵 $\log n$ 层的树。

由于树是 $\log n$ 层的，很多原来并不对劲的暴力在点分树上均有正确的复杂度。

### 代码实现

有一个小技巧：每次用递归上一层的总大小 $\mathit{tot}$ 减去上一层的点的重儿子大小，得到的就是这一层的总大小。这样求重心就只需一次 DFS 了

```cpp
#include <bits/stdc++.h>

using namespace std;

typedef vector<int>::iterator IT;

struct Edge {
    int to, nxt, val;

    Edge() {}
    Edge(int to, int nxt, int val) : to(to), nxt(nxt), val(val) {}
} e[300010];
int head[150010], cnt;

void addedge(int u, int v, int val) {
    e[++cnt] = Edge(v, head[u], val);
    head[u] = cnt;
}

int siz[150010], son[150010];
bool vis[150010];

int tot, lasttot;
int maxp, root;

void getG(int now, int fa) {
    siz[now] = 1;
    son[now] = 0;
    for (int i = head[now]; i; i = e[i].nxt) {
        int vs = e[i].to;
        if (vs == fa || vis[vs]) continue;
        getG(vs, now);
        siz[now] += siz[vs];
        son[now] = max(son[now], siz[vs]);
    }
    son[now] = max(son[now], tot - siz[now]);
    if (son[now] < maxp) {
        maxp = son[now];
        root = now;
    }
}

struct Node {
    int fa;
    vector<int> anc;
    vector<int> child;
} nd[150010];

int build(int now, int ntot) {
    tot = ntot;
    maxp = 0x7f7f7f7f;
    getG(now, 0);
    int g = root;
    vis[g] = 1;
    for (int i = head[g]; i; i = e[i].nxt) {
        int vs = e[i].to;
        if (vis[vs]) continue;
        int tmp = build(vs, ntot - son[vs]);
        nd[tmp].fa = now;
        nd[now].child.push_back(tmp);
    }
    return g;
}

int virtroot;

int main() {
    int n;
    cin >> n;
    for (int i = 1; i < n; i++) {
        int u, v, val;
        cin >> u >> v >> val;
        addedge(u, v, val);
        addedge(v, u, val);
    }
    virtroot = build(1, n);
}
```

## 习题

### 点分治

#### 递归求重心

> [!NOTE] **[Codeforces Ciel the Commander](http://codeforces.com/problemset/problem/321/C)**
> 
> 题意: 
> 
> 给出一棵有 $n$ 个点的无根树，要求构造出一种方案，使得树上的每一个结点的权值都满足 $1 \leq v_i \leq 26$ 
> 
> 且对于每一对拥有相同权值 $x$ 的点，在它们的简单路径上至少有一个权值大于 $x$ 的点 (字符小于)。

> [!TIP] **思路**
> 
> 原先想的是找中心，然后bfs拓展一层一层赋值 ==> WA
> 
> 实际上是找重心 **本题重在推导 随后套点分治模版即可**
> 
> - 显然最多只有一个点标记 $A$，因为没有比 $A$ 更大的字符了。如果存在两个 $A$ ，没办法在他们中间放更大的
> 
> - 考虑在哪放 $A$ 最优：如果我们在点 $x$ 位置放置了 $A$ ，那么对于点 $x$ 的所有子树都不能放 $A$ ，问题变成用 $B$ 到 $Z$ 处理点 $x$ 的每棵子树 ==> 子问题
> 
> - 子问题 ==> 树的重心 ==> 拆分整棵树，然后对每棵子树递归求重心即可
> 
> - 递归总层数为 $log(n)$ 显然不超过 $26$ 个字符可以表示的范围，故必然有解

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: CF321C Ciel the Commander
// Contest: Luogu
// URL: https://www.luogu.com.cn/problem/CF321C
// Memory Limit: 250 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 原先想的是找中心，然后bfs拓展一层一层赋值 ==> WA
// 实际上是找重心，

const static int N = 1e5 + 10, M = 2e5 + 10;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n;
int sz[N], son[N], root, tot;	// ATTENTION tot需重置 直接用n就WA
bool st[N];  // 点分治的全局标记

// 求重心
void dfs(int u, int fa) {
    son[u] = 0, sz[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa || st[j])  // 增加对全局标记的判断
            continue;
        dfs(j, u);
        son[u] = max(son[u], sz[j]);
        sz[u] += sz[j];
    }
    son[u] = max(son[u], tot - sz[u]);
    if (root == -1 || son[u] < son[root])
        root = u;
}

char res[N];

void divide(int u, char c) {
    res[u] = c;
    st[u] = true;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (st[j])
            continue;
        root = -1, tot = sz[j];
        dfs(j, -1);           // 找子树重心
        divide(root, c + 1);  // 标记并递归处理
    }
}

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

    memset(st, 0, sizeof st);
    root = -1, tot = n;  // tot 重要！在dfs过程中会用于计算上面的son
    dfs(1, -1);
    divide(root, 'A');

    // logn <= 26 故必然有合法答案
    for (int i = 1; i <= n; ++i)
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
