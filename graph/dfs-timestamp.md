## Intro

> [!NOTE] **DFS 序经典性质**
>
> TODO

> [!NOTE] **DFS 序七个经典问题**
>
> dfs 序是树在dfs先序遍历时的序列，将树形结构转化成序列问题处理。
>
> dfs 有一个很好的性质：一棵子树所在的位置处于一个连续区间中。
>
> ps: $deep[x]$ 为 $x$ 的深度，$tsl[x]$ 为 dfs 序中 $x$ 的位置，$tsr[x]$ 为 dfs 序中 $x$ 子树的结束位置
>
> 1.   **点修改，子树和查询**
>
>      在 dfs 序中，子树处于一个连续区间中。所以这题可以转化为：点修改，区间查询。用树状数组或线段树即可。
>
> 2.   **树链修改，单点查询**
>
>      将一条树链 $x,y$ 上的所有点的权值加 $v$。这个问题可以等价为：
>
>      1）$x$ 到根节点的链上所有节点权值加 $v$
>
>      2）$y$ 到根节点的链上所有节点权值加 $v$
>
>      3）$lca(x,y)$ 到根节点的链上所有节点权值和减 $v$
>
>      4）$fa(lca(x,y))$ 到根节点的链上所有节点权值和减 $v$
>
>      上面四个操作可以归结为：节点 $x$ 到根节点链上所有节点的权值加减 $v$。修改节点 $x$ 权值，当且仅当 $y$ 是 $x$ 的祖先节点时，$x$ 对 $y$ 的值有贡献。
>
>      所以节点 $y$ 的权值可以转化为节点 $y$ 的子树节点贡献和。从贡献和的角度想：这就是点修改，区间和查询问题。
>
>      修改树链 $x,y$ 等价于 $$add(tsl[x],v),add(tsl[y],v),\\ add(tsl[lca(x,y)],-v),add(tsl[fa(lca(x,y))],-v)$$
>
>      查询：$getsum(tsr[x])-getsum(tsl[x]-1)$
>
>      用树状数组或线段树即可。　
>
> 3.   **树链修改，子树和查询**
>
>      树链修改部分同上一问题。下面考虑子树和查询问题：前一问是从贡献的角度想，子树和同理。
>
>      对于节点 $y$，考虑其子节点 $x$ 的贡献：$$w[x]*(deep[x]-deep[y]+1) \\= w[x]*(deep[x]+1)-w[x]*deep[y]$$
>
>      所以节点 $y$ 的子树和为：
>
>      $$\sum_{i=tsl[y]}^{tsr[y]}w[i]*(deep[i]+1)-deep[y]*\sum_{i=tsl[y]}^{tsr[y]}w[i]$$
>
>      所以用两个树状数组或线段树即可：
>
>      - 第一个维护 $\sum_{i=tsl[y]}^{tsr[y]}w[i]*(deep[i]+1)$: 支持操作单点修改，区间和查询。（这也就是问题2）
>
>      - 第二个维护 $\sum_{i=tsl[y]}^{tsr[y]}w[i]$: 支持操作单点修改，区间查询。（这其实也是问题2）
>
> 4.   **单点更新，树链和查询**
>
>      树链和查询与树链修改类似，树链和 $(x,y)$ 等于下面四个部分和相加：
>
>      1）$x$ 到根节点的链上所有节点权值加。
>
>      2）$y$ 到根节点的链上所有节点权值加。
>
>      3）$lca(x,y)$ 到根节点的链上所有节点权值和的 -1 倍。
>
>      4）$fa(lca(x,y))$ 到根节点的链上所有节点权值和的 -1 倍。
>
>      所以问题转化为：查询点 $x$ 到根节点的链上的所有节点权值和。
>
>      修改节点 $x$ 权值，当且仅当 $y$ 是 $x$ 的子孙节点时，$x$ 对 $y$ 的值有贡献。
>
>      差分前缀和，$y$ 的权值等于 dfs 中 $[1,tsl[y]]$ 的区间和。
>
>      单点修改：$add(tsl[x],v),add(tsr[x]+1,-v)$;
>
> 5.   **子树修改，单点查询**
>
>      修改节点 $x$ 的子树权值，当且仅当 $y$ 是 $x$ 的子孙节点时（或 $y$ 等于 $x$），$x$ 对 $y$ 的值有贡献。
>
>      所以从贡献的角度考虑，$y$ 的权值和为：子树所有节点的权值和（即区间和问题）
>
>      然后子树修改变成区间修改：$add(tsl[x],v),add(tsr[x]+1,-v)$;
>
>      这就是点修改，区间查询问题了。用树状数组或线段树即可。
>
> 6.   **子树修改，子树和查询**
>
>      题目等价与区间修改，区间查询问题。用树状数组或线段树即可。
>
> 7.   **子树修改，树链查询**
>
>      树链查询同上，等价为根节点到 $y$ 节点的链上所有节点和问题。
>
>      修改节点 $x$ 的子树权值，当且仅当 $y$ 是 $x$ 的子孙节点时（或 $y$ 等于 $x$），$x$ 对 $y$ 的值有贡献。
>
>      $x$ 对根节点到 $y$ 节点的链上所有节点和的贡献为：$$w[x]*(deep[y]-deep[x]+1)\\=w[x]*deep[y]-w[x]*(1-deep[x])$$
>
>      同问题三，用两个树状数组或线段树即可。

## 习题

> [!NOTE] **[LeetCode 2322. 从树中删除边的最小分数](https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然可以固定根并处理每个子树的异或值
> 
> 1e3 数据量接受枚举三部分的根，考虑计算三个部分的数值
> 
> 唯一的点在于处理 `其中一部分是另一部分的子树` 的情况
> 
> - LCA
> 
> - **DFS 序**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ DFS序**

```cpp
class Solution {
public:
    const static int N = 1010, M = 2010;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n;
    vector<int> vals;
    int x[N], fa[N], tsl[N], tsr[N], timestamp;
    void dfs(int u, int pa) {
        fa[u] = pa, tsl[u] = ++ timestamp;
        x[u] = vals[u];
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            dfs(j, u);
            x[u] ^= x[j];
        }
        tsr[u] = timestamp;
    }
    
    pair<int, int> get(int a, int b) {
        if (fa[a] == b)
            return {b, a};
        return {a, b};
    }
    
    bool is_subtree(int a, int b) {
        return tsl[a] >= tsl[b] && tsr[a] <= tsr[b];
    }
    
    int minimumScore(vector<int>& nums, vector<vector<int>>& edges) {
        this->vals = nums, this->n = vals.size();
        init();
        for (auto & e : edges)
            add(e[0], e[1]), add(e[1], e[0]);
        
        memset(fa, 0, sizeof fa);
        memset(x, 0, sizeof x);
        this->timestamp = 0;
        dfs(0, -1);
        
        int res = INT_MAX;
        for (auto & e1 : edges)
            for (auto & e2 : edges) {
                auto [u1, v1] = get(e1[0], e1[1]);
                auto [u2, v2] = get(e2[0], e2[1]);
                if (u1 == u2 && v1 == v2)
                    continue;
                int a = x[v1], b = x[v2], c = 0;
                
                if (is_subtree(v1, v2) || is_subtree(v2, v1)) {
                    if (is_subtree(v2, v1))
                        c = x[0] ^ a, a ^= b;
                    else
                        c = x[0] ^ b, b ^= a;
                } else {
                    c = x[0] ^ a ^ b;
                }
                
                res = min(res, max({a, b, c}) - min({a, b, c}));
            }
        
        return res;
    }
};
```

##### **C++ LCA**

```cpp
class Solution {
public:
    const static int N = 1010, M = 2010;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n;
    vector<int> vals;
    int x[N];
    void dfs(int u, int pa) {
        x[u] = vals[u - 1];
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            dfs(j, u);
            x[u] ^= x[j];
        }
    }
    int depth[N], fa[N][11], q[N];
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
                    for (int k = 1; k <= 10; ++ k )
                        fa[j][k] = fa[fa[j][k - 1]][k - 1];
                }
            }
        }
    }
    int lca(int a, int b) {
        if (depth[a] < depth[b])
            swap(a, b);
        for (int k = 10; k >= 0; -- k )
            if (depth[fa[a][k]] >= depth[b])
                a = fa[a][k];
        if (a == b)
            return a;
        for (int k = 10; k >= 0; -- k )
            if (fa[a][k] != fa[b][k])
                a = fa[a][k], b = fa[b][k];
        return fa[a][0];
    }
    
    pair<int, int> get(int a, int b) {
        if (fa[a][0] == b)
            return {b, a};
        return {a, b};
    }
    
    int minimumScore(vector<int>& nums, vector<vector<int>>& edges) {
        this->vals = nums, this->n = vals.size();
        init();
        for (auto & e : edges)
            add(e[0] + 1, e[1] + 1), add(e[1] + 1, e[0] + 1);
        memset(fa, 0, sizeof fa);
        memset(x, 0, sizeof x);
        
        // 0->1 映射 方便LCA的fa数组哨兵节点
        dfs(1, 0);
        bfs(1);
        
        int res = INT_MAX;
        for (auto & e1 : edges)
            for (auto & e2 : edges) {
                auto [u1, v1] = get(e1[0] + 1, e1[1] + 1);
                auto [u2, v2] = get(e2[0] + 1, e2[1] + 1);
                if (u1 == u2 && v1 == v2)
                    continue;
                int a = x[v1], b = x[v2], c = 0;
                int t = lca(v1, v2);
                if (t == v1 || t == v2) {
                    if (t == v1)
                        c = x[1] ^ a, a ^= b;
                    else
                        c = x[1] ^ b, b ^= a;
                } else {
                    c = x[1] ^ a ^ b;
                }
                
                // cout << " v1 = " << v1-1 << " a = " << a << " v2 = " << v2-1 << " b = " << b << " c = " << c << endl;
                res = min(res, max({a, b, c}) - min({a, b, c}));
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

> [!NOTE] **[Codeforces Propagating tree](http://codeforces.com/problemset/problem/383/C)**
> 
> 题意: 
> 
> 一颗树，每个点都有一个权值，你需要完成这两种操作：
> 
> 1. `u val` 表示给 $u$ 节点的权值增加 $val$
> 
> 2. `u` 表示查询 $u$ 节点的权值
> 
> 但是这不是普通的橡树，它是神橡树。所以它还有个神奇的性质：
> 
> 当某个节点的权值增加 $val$ 时，它的子节点权值都增加 $-val$ ，它子节点的子节点权值增加 $-(-val)$... 如此一直进行到树的底部。

> [!TIP] **思路**
> 
> 容易想到树差分。因为有递归修改的性质，点差分显然是不够用的。
> 
> 同时易想到 BIT 维护差分数组，进而达到 $log(n)$ 查询节点权值的目的。
> 
> 问题在于：无法直接用点编号作为 BIT 元素，应该如何维护？
> 
> **DFS序，将序列作为节点维护，整颗树的修改即变为区间修改**
> 
> **此时可以分别用两个 BIT 维护 `奇数/偶数` 层的差分数值变化，当然也可以用技巧只用一个差分数组**
> 
> TODO: **DFS序相关的一系列题目**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Propagating tree
// Contest: Codeforces - Codeforces Round #225 (Div. 1)
// URL: https://codeforces.com/problemset/problem/383/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 2e5 + 10, M = 4e5 + 10;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int tr[N];
int lowbit(int x) { return x & -x; }
void addv(int x, int v) {
    for (int i = x; i < N; i += lowbit(i))
        tr[i] += v;
}
int sum(int x) {
    int ret = 0;
    for (int i = x; i; i -= lowbit(i))
        ret += tr[i];
    return ret;
}

int pa[N], dep[N], tsl[N], tsr[N], timestamp = 0;
void dfs(int u, int fa) {
    tsl[u] = ++timestamp;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dep[j] = dep[u] + 1;
        dfs(j, u);
    }
    tsr[u] = timestamp;
}

int n, m;
int a[N], lazy[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n >> m;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];
    for (int i = 0; i < n - 1; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    dep[1] = 0;
    dfs(1, -1);

    while (m--) {
        int type, x, val;
        cin >> type >> x;
        if (type == 1) {
            cin >> val;
            if (dep[x] & 1)
                addv(tsl[x], val), addv(tsr[x] + 1, -val);
            else  // 偶数层值反过来算
                addv(tsl[x], -val), addv(tsr[x] + 1, val);
        } else {
            // ATTENTION: 为什么可以直接 sum(tsl) ==> 细节 思考 理解
            int t = sum(tsl[x]);
            if (dep[x] & 1)
                cout << a[x] + t << '\n';
            else
                cout << a[x] - t << '\n';
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


> [!NOTE] **[Codeforces Water Tree](http://codeforces.com/problemset/problem/343/D)**
> 
> 题意: 
> 
> 给出一棵以 $1$ 为根节点的 $n$ 个节点的有根树。每个点有一个权值，初始为 $0$。
> 
> $m$ 次操作。操作有 $3$ 种：
> 
> - 将点 $u$ 和其子树上的所有节点的权值改为 $1$。
> 
> - 将点 $u$ 到 $1$ 的路径上的所有节点的权值改为 $0$。
> 
> - 询问点 $u$ 的权值。

> [!TIP] **思路**
> 
> 珂朵莉树维护 DFS 序
> 
> **加强对树剖的熟悉程度 深入理解维护流程**
> 
> 注意 tid 与 ts 的计算细节，在 update 函数中写错 DEBUG 花了很久...
> 
> TODO: **树剖相关概念 & 重链相关**
> 
> TODO: **DFS序将树抽象为单点 整理**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Water Tree
// Contest: Codeforces - Codeforces Round #200 (Div. 1)
// URL: https://codeforces.com/problemset/problem/343/D
// Memory Limit: 256 MB
// Time Limit: 4000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 5e5 + 10, M = 1e6 + 10;

// -------------------- graph --------------------
int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }
// -------------------- odt --------------------
struct Node_t {
    int l, r;
    mutable int v;
    inline bool operator<(const Node_t& o) const {
        return l < o.l;  // 按 l 升序排列
    }
};
set<Node_t> odt;
auto split(int x) {
    auto it = odt.lower_bound({x, 0, 0});  // 找到大于等于x的第一个
    if (it != odt.end() && it->l == x)
        return it;
    // 否则x一定被前一段包含，向前移找到该段
    it--;
    auto [l, r, v] = *it;
    odt.erase(it);
    odt.insert({l, x - 1, v});
    return odt.insert({x, r, v}).first;  // ATTENTION 返回迭代器
}
void merge(set<Node_t>::iterator it) {
    if (it == odt.end() || it == odt.begin())
        return;
    auto lit = prev(it);
    auto [ll, lr, lv] = *lit;
    auto [rl, rr, rv] = *it;
    if (lv == rv) {
        odt.erase(lit), odt.erase(it), odt.insert({ll, rr, lv});
        // ... 其他操作
    }
}
void assign(int l, int r, int v) {
    auto itr = split(r + 1), itl = split(l);  // 顺序不能颠倒
    // 清除一系列节点
    odt.erase(itl, itr);
    odt.insert({l, r, v});
    // 维护区间 【视情况而定】
    merge(odt.lower_bound({l, 0, 0})), merge(itr);
}
// -------------------- DFS 序 --------------------
// ATTENTION: 为什么要记录 重链？
int sz[N], p1[N], pa[N];  // 分别表示子树大小，以及重儿子是谁
void dfs_1(int u, int fa) {
    sz[u] = 1, p1[u] = -1, pa[u] = -1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs_1(j, u);
        sz[u] += sz[j];
        pa[j] = u;
        if (p1[u] == -1 || sz[p1[u]] < sz[j])
            p1[u] = j;
    }
}
int tid[N], ts[N], timestamp = 0;
void dfs_2(int u, int fa, int id) {  // ATTENTION DFS序
    tid[u] = id;
    ts[u] = ++timestamp;
    if (p1[u] == -1)
        return;
    dfs_2(p1[u], u, id);
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa || j == p1[u])
            continue;
        dfs_2(j, u, j);  // ATTENTION id changed
    }
}
// -------------------- logic --------------------
void update(int x, int id) {
    do {
        // id的起始到x
        assign(ts[id], ts[x], 0);
        // 注意 x = pa[id] 而不是 x = pa[x], TLE 查错很久很久...
        x = pa[id], id = tid[x];
    } while (x != -1);
}
int sum(int x) {
    auto it = split(x);
    return it->v;
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

    dfs_1(1, -1);
    dfs_2(1, -1, 1);

    odt.insert({0, N, 0});

    cin >> m;
    while (m--) {
        int a, b;
        cin >> a >> b;
        if (a == 1) {
            // l = ts[b], r = ts[b] + sz[b] - 1
            // l r 本质是 DFS 序
            assign(ts[b], ts[b] + sz[b] - 1, 1);
        } else if (a == 2) {
            update(b, tid[b]);
        } else {
            cout << sum(ts[b]) << '\n';
        }
    }

    return 0;
}
```

##### **C++ 错误做法TLE**

```cpp
// Problem: D. Water Tree
// Contest: Codeforces - Codeforces Round #200 (Div. 1)
// URL: https://codeforces.com/problemset/problem/343/D
// Memory Limit: 256 MB
// Time Limit: 4000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 5e5 + 10, M = 1e6 + 10;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int pre[N], q[N];
bool st[N];
void bfs(int root) {
    memset(st, 0, sizeof st);
    memset(pre, 0, sizeof pre);
    int hh = 0, tt = -1;
    q[++tt] = root, st[root] = true;
    while (hh <= tt) {
        int t = q[hh++];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (!st[j]) {
                pre[j] = t;
                q[++tt] = j;
                st[j] = true;
            }
        }
    }
}

int n, m;
bool state[N];
int stk[N], top, cnt;
void track(int x) {
    top = 0, cnt = 0;
    do {
        stk[top++] = x;
        if (state[x])
            cnt++;
        x = pre[x];
    } while (x);
}
void Set(int x) { state[x] = true; }
void Reset(int x) {
    track(x);
    if (cnt == 0)
        return;
    stk[top] = -1;
    for (int i = top - 1; i >= 0; --i) {
        x = stk[i];
        if (state[x]) {
            for (int j = h[x]; ~j; j = ne[j]) {
                int k = e[j];
                if (k != stk[i + 1])
                    state[k] = 1;
            }
            state[x] = 0;
        }
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

    bfs(1);

    cin >> m;
    while (m--) {
        int a, b;
        cin >> a >> b;
        if (a == 1) {
            Set(b);
        } else if (a == 2) {
            Reset(b);
        } else {
            track(b);
            cout << (cnt ? 1 : 0) << '\n';
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

> [!NOTE] **[Codeforces Little Girl and Problem on Trees](http://codeforces.com/problemset/problem/276/E)**
> 
> 题意: 
> 
> 给一棵树，初始点权都为 $0$，保证除了 $1$ 号节点以外的节点度数不超过 $2$。
> 
> 换句话说，这棵树就是 $1$ 号节点下面挂了若干条链形成的菊花图（下文提到根节点下挂的链不包含根节点）。有两种操作：
> 
> $(0,v,x,d)$ ：把距离 $v$ 号点距离不超过 $d$ 的点点权加 $x$ 。
> 
> $(1,v)$ ：查询 $v$ 号点的点权。

> [!TIP] **思路**
> 
> 容易想到 dfs 计算时间戳。把树上的修改转移到序列上的区间修改，又想到了在时间戳上建立 BIT 。
> 
> 仔细分析，显然对于某个点 $u$ 需要分别 `向上 / 向下` 延展 $d$ 的距离。
> 
> - 向下延展较好解决，相应的 DFS 序区间统一加就可以
> 
> - 向上延展时，需要考虑跨过了根节点进而影响其他链的情况，所以向上需要分两个子区间单独处理（靠下部分统一加，上面部分通过根节点统一维护所有链即可）
> 
> 故需要两个 BIT ，一个维护 DFS 序对应的节点值，另一个维护根节点起始向下指定层的增加量（需要维护每一个链）
> 
> **认真思考：对于 BIT 维护 DFS 序的节点差分来说，使用时直接求 $sum$ 即可，不需要考虑区间减**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: E. Little Girl and Problem on Trees
// Contest: Codeforces - Codeforces Round #169 (Div. 2)
// URL: https://codeforces.com/problemset/problem/276/E
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

int tr1[N], tr2[N];
int lowbit(int x) { return x & -x; }
void add(int tr[], int x, int c) {
    for (int i = x; i < N; i += lowbit(i))
        tr[i] += c;
}
int sum(int tr[], int x) {
    int ret = 0;
    for (int i = x; i; i -= lowbit(i))
        ret += tr[i];
    return ret;
}

int tsl[N], tsr[N], timestamp = 0;
int dep[N];
void dfs(int u, int fa) {
    tsl[u] = ++timestamp;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dep[j] = dep[u] + 1;
        dfs(j, u);
    }
    tsr[u] = timestamp;
}

int n, q;

void print() {
    for (int i = 1; i <= n; ++i) {
        // int a = sum(tr1, tsr[i]) - sum(tr1, tsl[i] - 1),
        int a = sum(tr1, tsl[i]), b = sum(tr2, dep[i]);
        cout << " i = " << i << " a = " << a << " b = " << b
             << " sum = " << a + b << endl;
    }
    cout << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n >> q;
    for (int i = 0; i < n - 1; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    // 获取 DFS 序
    dep[1] = 1;  // 因为tr2需要用到，所以下标以1起始
    dfs(1, -1);

    // 树上默认全0 故差分也是0
    memset(tr1, 0, sizeof tr1), memset(tr2, 0, sizeof tr2);

    while (q--) {
        int type, v, x, d;
        cin >> type;
        if (type == 0) {
            cin >> v >> x >> d;
            if (v == 1) {
                add(tr2, 1, x), add(tr2, 1 + d + 1, -x);
            } else if (dep[v] > d + 1) {
                add(tr1, tsl[v] - d, x);
                add(tr1, min(tsl[v] + d + 1, tsr[v] + 1), -x);
            } else {
                // ps: 因为本题子树都是单链，所以可以直接获取上部分的位置
                int dis = d + 1 - dep[v];  // 注意还要加1
                // ps: 上半部分以根为中心统一加
                add(tr2, 1, x), add(tr2, 1 + dis + 1, -x);
                // ps: 下半部分一起加
                // ATTEINTION
                int l = max(tsl[v] - d, tsl[v] - (dep[v] - (dis + 1)) + 1);
                int r = min(tsl[v] + d, tsr[v]);
                if (l <= r) {
                    add(tr1, l, x);
                    add(tr1, r + 1, -x);
                }
            }
            // print();
        } else {
            cin >> v;
            // ATTENTION: 本来维护的就是差分，所以直接
            // 求 sum(tr1, tsl[v]) 就可以了，不需要考虑区间减
            int a = sum(tr1, tsl[v]), b = sum(tr2, dep[v]);
            cout << a + b << endl;
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

> [!NOTE] **[Codeforces Tree Requests](http://codeforces.com/problemset/problem/570/D)**
> 
> 题意: 
> 
> 一棵以 $1$ 为根的树，每个节点上都有 $1$ 个字母，有 $m$ 个询问。
> 
> 每次询问 $v$ 对应的子树中，深度为 $h$ 的这层节点的字母，能否打乱重排组成回文串。
> 
> 根的深度为 $1$，每个点的深度为到根的距离。

> [!TIP] **思路**
> 
> 题目要求指定根节点的子树，显然可以通过 DFS 序的构造实现
> 
> 显然需要先按层分，层内压入所有节点的 DFS 序及其奇偶性质，利用异或前缀和判断即可
> 
> 注意边界细节：`x = get(d, tsl[v]) ^ get(d, tsr[v]);` 
> 
> 特别注意左区间写 `tsl[v]` 因为它就是取不到的位置！

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Tree Requests
// Contest: Codeforces - Codeforces Round #316 (Div. 2)
// URL: https://codeforces.com/problemset/problem/570/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const static int N = 5e5 + 10, M = 1e6 + 10;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n, m;
char cs[N];

int tsl[N], tsr[N], tsc[N], dep[N], timestamp = 0;
vector<PII> depth[N], s[N];
void dfs(int u, int d) {
    tsl[u] = ++timestamp, dep[u] = d;
    int x = 1 << (cs[u] - 'a');
    depth[d].push_back({tsl[u], x});
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        dfs(j, d + 1);
    }
    tsr[u] = timestamp;
}

int get(int d, int x) {
    auto& xs = s[d];
    PII t = {x, 1e9};
    auto it = lower_bound(xs.begin(), xs.end(), t);
    it--;
    return (*it).second;
}

void print(int i) {
    cout << " s i = " << i << endl;
    for (auto [x, y] : s[i])
        cout << x << " " << y << endl;
    cout << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n >> m;
    for (int i = 2; i <= n; ++i) {
        int p;
        cin >> p;
        add(p, i);
    }

    cin >> (cs + 1);
    dfs(1, 1);

    for (int i = 1; i < N; ++i) {
        if (depth[i].empty())
            break;
        auto& dep = depth[i];
        sort(dep.begin(), dep.end());  // 需按id排序
        s[i].push_back({0, 0});
        for (auto& [x, y] : dep)
            s[i].push_back({x, y ^ s[i].back().second});
        // print(i);
    }

    while (m--) {
        int v, h;
        cin >> v >> h;
        int d = h;  // ATTENTION 注意题意并非 dep[v]+h-1
        if (s[d].empty())
            // cout << "No" << endl;
            cout << "Yes" << endl;  // ATTENTION WA 15
        else {
            // cout << " id = " << tsl[v] << ' ' << tsr[v] << endl;
            int x = get(d, tsl[v]) ^ get(d, tsr[v]);
            // 查询某一层的字符状态，bit位为1表示奇数个
            int cnt = 0;
            for (int i = 0; i < 26; ++i)
                if (x >> i & 1)
                    cnt++;
            cout << (cnt <= 1 ? "Yes" : "No") << '\n';
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
