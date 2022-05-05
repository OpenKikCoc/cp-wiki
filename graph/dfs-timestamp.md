## 习题


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

* * *
