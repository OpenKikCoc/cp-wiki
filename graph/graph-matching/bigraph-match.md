
为了描述方便将两个集合分成左和右两个部分，所有匹配边都是横跨左右两个集合，可以假想成男女配对。

假设图有 $n$ 个顶点，$m$ 条边。

## 增广路算法 Augmenting Path Algorithm

因为增广路长度为奇数，路径起始点非左即右，所以我们先考虑从左边的未匹配点找增广路。
注意到因为交错路的关系，增广路上的第奇数条边都是非匹配边，第偶数条边都是匹配边，于是左到右都是非匹配边，右到左都是匹配边。
于是我们给二分图 **定向**，问题转换成，有向图中从给定起点找一条简单路径走到某个未匹配点，此问题等价给定起始点 $s$ 能否走到终点 $t$。
那么只要从起始点开始 DFS 遍历直到找到某个未匹配点，$O(m)$。
未找到增广路时，我们拓展的路也称为 **交错树**。

因为要枚举 $n$ 个点，总复杂度为 $O(nm)$。

### 代码

```cpp
struct augment_path {
    vector<vector<int> > g;
    vector<int> pa;  // 匹配
    vector<int> pb;
    vector<int> vis;  // 访问
    int n, m;         // 两个点集中的顶点数量
    int dfn;          // 时间戳记
    int res;          // 匹配数

    augment_path(int _n, int _m) : n(_n), m(_m) {
        assert(0 <= n && 0 <= m);
        pa = vector<int>(n, -1);
        pb = vector<int>(m, -1);
        vis = vector<int>(n);
        g.resize(n);
        res = 0;
        dfn = 0;
    }

    void add(int from, int to) {
        assert(0 <= from && from < n && 0 <= to && to < m);
        g[from].push_back(to);
    }

    bool dfs(int v) {
        vis[v] = dfn;
        for (int u : g[v]) {
            if (pb[u] == -1) {
                pb[u] = v;
                pa[v] = u;
                return true;
            }
        }
        for (int u : g[v]) {
            if (vis[pb[u]] != dfn && dfs(pb[u])) {
                pa[v] = u;
                pb[u] = v;
                return true;
            }
        }
        return false;
    }

    int solve() {
        while (true) {
            dfn++;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (pa[i] == -1 && dfs(i)) { cnt++; }
            }
            if (cnt == 0) { break; }
            res += cnt;
        }
        return res;
    }
};
```

## 转为网络最大流模型

二分图最大匹配可以转换成网络流模型。

将源点连上左边所有点，右边所有点连上汇点，容量皆为 $1$。原来的每条边从左往右连边，容量也皆为 $1$，最大流即最大匹配。

如果使用 [Dinic 算法](graph/flow/max-flow.md#dinic) 求该网络的最大流，可在 $O(\sqrt{n}m)$ 求出。

Dinic 算法分成两部分，第一部分用 $O(m)$ 时间 BFS 建立网络流，第二步是 $O(nm)$ 时间 DFS 进行增广。

但因为容量为 $1$，所以实际时间复杂度为 $O(m)$。

接下来前 $O(\sqrt{n})$ 轮，复杂度为 $O(\sqrt{n}m)$。$O(\sqrt{n})$ 轮以后，每条增广路径长度至少 $\sqrt{n}$，而这样的路径不超过 $\sqrt{n}$，所以此时最多只需要跑 $\sqrt{n}$ 轮，整体复杂度为 $O(\sqrt{n}m)$。

代码可以参考 [Dinic 算法](graph/flow/max-flow.md#dinic) 的参考实现，这里不再给出。

## 补充

### 二分图最大独立集

选最多的点，满足两两之间没有边相连。

二分图中，最大独立集 =$n$- 最大匹配。

### 二分图最小点覆盖

选最少的点，满足每条边至少有一个端点被选，不难发现补集是独立集。

二分图中，最小点覆盖 =$n$- 最大独立集。

## 习题

> [!NOTE] **[UOJ #78. 二分图最大匹配](https://uoj.ac/problem/78) **
> 
> 模板题
    
```cpp
#include <bits/stdc++.h>
using namespace std;

struct augment_path {
    vector<vector<int> > g;
    vector<int> pa;  // 匹配
    vector<int> pb;
    vector<int> vis;  // 访问
    int n, m;         // 顶点和边的数量
    int dfn;          // 时间戳记
    int res;          // 匹配数

    augment_path(int _n, int _m) : n(_n), m(_m) {
        assert(0 <= n && 0 <= m);
        pa = vector<int>(n, -1);
        pb = vector<int>(m, -1);
        vis = vector<int>(n);
        g.resize(n);
        res = 0;
        dfn = 0;
    }

    void add(int from, int to) {
        assert(0 <= from && from < n && 0 <= to && to < m);
        g[from].push_back(to);
    }

    bool dfs(int v) {
        vis[v] = dfn;
        for (int u : g[v]) {
            if (pb[u] == -1) {
                pb[u] = v;
                pa[v] = u;
                return true;
            }
        }
        for (int u : g[v]) {
            if (vis[pb[u]] != dfn && dfs(pb[u])) {
                pa[v] = u;
                pb[u] = v;
                return true;
            }
        }
        return false;
    }

    int solve() {
        while (true) {
            dfn++;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (pa[i] == -1 && dfs(i)) { cnt++; }
            }
            if (cnt == 0) { break; }
            res += cnt;
        }
        return res;
    }
};

int main() {
    int n, m, e;
    cin >> n >> m >> e;
    augment_path solver(n, m);
    int u, v;
    for (int i = 0; i < e; i++) {
        cin >> u >> v;
        u--, v--;
        solver.add(u, v);
    }
    cout << solver.solve() << "\n";
    for (int i = 0; i < n; i++) { cout << solver.pa[i] + 1 << ' '; }
    cout << **\n ";
}
```

> [!NOTE] **[AcWing 861. 二分图的最大匹配](https://www.acwing.com/problem/content/863/)**
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

const int N = 510, M = 100010;

int n1, n2, m;
int h[N], e[M], ne[M], idx;
int match[N];
bool st[N];

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

bool find(int x) {
    for (int i = h[x]; i != -1; i = ne[i]) {
        int j = e[i];
        if (!st[j]) {
            st[j] = true;
            if (match[j] == 0 || find(match[j])) {
                match[j] = x;
                return true;
            }
        }
    }

    return false;
}

int main() {
    scanf("%d%d%d", &n1, &n2, &m);

    memset(h, -1, sizeof h);

    while (m--) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);
    }

    int res = 0;
    for (int i = 1; i <= n1; i++) {
        memset(st, false, sizeof st);
        if (find(i)) res++;
    }

    printf("%d\n", res);

    return 0;
}

```

##### **Python**

```python
"""
    /**
     * 要了解匈牙利算法必须先理解下面的概念：
     *     匹配：在图论中，一个「匹配」是一个边的集合，其中任意两条边都没有公共顶点。
     *     最大匹配：一个图所有匹配中，所含匹配边数最多的匹配，称为这个图的最大匹配。
     *
     * 下面是一些补充概念：
     *     完美匹配：如果一个图的某个匹配中，所有的顶点都是匹配点，那么它就是一个完美匹配。
     *     交替路：从一个未匹配点出发，依次经过非匹配边、匹配边、非匹配边…形成的路径叫交替路。
     *     增广路：从一个未匹配点出发，走交替路，如果途径另一个未匹配点（出发的点不算），则这条交替 路称为增广路（agumenting path）。
     *
     * 匈牙利算法思路：
     *     每个点从另一个集合里挑对象，没冲突的话就先安排上，要是冲突了就用增广路径重新匹配。重复上述思路，
     *     直到所有的点都找到对象，或者找不到对象也找不到增广路。
     */

算法流程：
如果你想找的妹子已经有了男朋友，
你就去问问她男朋友，你有没有备胎，把这个让给我好吧

多么真实而实用的算法

TIP: 因为你要去问的都是男孩子，所以存边的时候，都是由男孩子指向女孩子

"""


def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def find(x):
    i = h[x]
    # 遍历自己的连接（自己喜欢的女孩）
    while i != -1:
        j = ev[i]
        # 如果在这一轮模拟匹配中,这个点还没有被用过
        if not st[j]:
            # 预定这个点
            st[j] = True
            # 如果j没有对应的点，或者j之前的点还可以连接其他点。配对成功,更新match
            if match[j] == 0 or find(match[j]):
                match[j] = x
                return True
        i = ne[i]
    return False


if __name__ == '__main__':
    N = 510
    M = 100010  # 注意：边的范围
    h = [-1] * N
    ev = [0] * M
    ne = [0] * M
    idx = 0
    match = [0] * N  # 右边对应的点，match[j]=a,表示右边的点j的现有配对点是a
    st = [False] * N  # 表示 本次轮匹配中 有没有检查过该点
    res = 0  # 匹配的数量

    n1, n2, m = map(int, input().split())
    for _ in range(m):
        a, b = map(int, input().split())
        # 枚举左边集合，只需要存从左边指向右边就可以了
        add_edge(a, b)

    for i in range(1, n1 + 1):
        st = [False] * N  # clear flag
        if find(i):
            res += 1
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[P1640 [SCOI2010]连续攻击游戏](https://www.luogu.com.cn/problem/P1640) **
> 
> None

> [!NOTE] **[Codeforces 1139E - Maximize Mex](https://codeforces.com/problemset/problem/1139/E) **
> 
> None
