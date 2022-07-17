
并查集，Kruskal 重构树的思维方式是很类似的，他们都能用于处理与连通性有关的问题。本文通过例题讲解的方式给大家介绍并查集思想的应用。

## A

> [!NOTE] **A**
> 
> 有 $n$ 个点，初始时均为孤立点。
> 
> 接下来有 $m$ 次加边操作，第 $i$ 次操作在 $a_i$ 和 $b_i$ 之间加一条无向边。设 $L(i,j)$ 表示结点 $i$ 和 $j$ 最早在第 $L(i,j)$ 次操作后连通。
> 
> 在 $m$ 次操作完后，你要求出 $\sum_{i=1}^n\sum_{j=i+1}^nL(i,j)$ 的值。

这是基础并查集的应用，并查集记录一下子树的大小。考虑统计每次操作的贡献。如果第 $i$ 次操作 $a_i$ 和 $b_i$ 分属于两个不同子树，就将这两个子树合并，并将两者子树大小的乘积乘上 $i$ 累加到答案里。时间复杂度 $O(n\alpha(n))$。

## B

> [!NOTE] **B**
> 
> 有 $n$ 个点，初始时均为孤立点。
> 
> 接下来有 $m$ 次加边操作，第 $i$ 次操作在 $a_i$ 和 $b_i$ 之间加一条无向边。
> 
> 接下来有 $q$ 次询问，第 $i$ 次询问 $u_i$ 和 $v_i$ 最早在第几次操作后连通。

考虑在并查集合并的时候记录「并查集生成树」，也就是说如果第 $i$ 次操作 $a_i$ 和 $b_i$ 分属于两个不同子树，那么把 $(a_i,b_i)$ 这条边纳入生成树中。边权是 $i$。那么查询就是询问 $u$ 到 $v$ 路径上边权的最大值，可以使用树上倍增或者树链剖分的方法维护。时间复杂度 $O(n\log n)$。

另外一个方法是维护 Kruskal 重构树，其本质与并查集生成树是相同的。复杂度亦相同。

## C

> [!NOTE] **C**
> 
> 有 $n$ 个点，初始时均为孤立点。
> 
> 接下来有 $m$ 次加边操作，第 $i$ 次操作在 $a_i$ 和 $b_i$ 之间加一条无向边。
> 
> 接下来有 $q$ 次询问，第 $i$ 次询问第 $x_i$ 个点在第 $t_i$ 次操作后所在连通块的大小。

离线算法：考虑将询问按 $t_i$ 从小到大排序。在加边的过程中顺便处理询问即可。时间复杂度 $O(q\log q+(n+q)\alpha(n))$。

在线算法：本题的在线算法只能使用 Kruskal 重构树。Kruskal 重构树与并查集的区别是：第 $i$ 次操作 $a_i$ 和 $b_i$ 分属于两个不同子树，那么 Kruskal 会新建一个结点 $u$，然后让 $a_i$ 所在子树的根和 $b_i$ 所在子树的根分别连向 $u$，作为 $u$ 的两个儿子。不妨设 $u$ 的点权是 $i$。对于初始的 $n$ 个点，点权为 $0$。

对于询问，我们只需要求出 $x_i$ 在重构树中最大的一个连通块使得连通中的点权最大值不超过 $t_i$，询问的答案就是这个连通块中点权为 $0$ 的结点个数，即叶子结点个数。

由于我们操作的编号是递增的，因此重构树上父结点的点权总是大于子结点的点权。这意味着我们可以在重构树上从 $x_i$ 到根结点的路径上倍增找到点权最大的不超过 $t_i$ 的结点。这样我们就求出了答案。时间复杂度 $O(n\log n)$。

## D

> [!NOTE] **D**
> 
> 给一个长度为 $n$ 的 01 序列 $a_1,\ldots,a_n$，一开始全是 $0$，接下来进行 $m$ 次操作：
> 
> - 令 $a_x=1$；
> 
> - 求 $a_x,a_{x+1},\ldots,a_n$ 中左数第一个为 $0$ 的位置。

建立一个并查集，$f_i$ 表示 $a_i,a_{i+1},\ldots,a_n$ 中第一个 $0$ 的位置。初始时 $f_i=i$。

对于一次 $a_x=1$ 的操作，如果 $a_x$ 原本就等于 $1$，就不管。否则我们令 $f_x=f_{x+1}$。

时间复杂度 $O(n\log n)$，如果要使用按秩合并的话实现会较为麻烦，不过仍然可行。也就是说时间复杂度或为 $O(n\alpha(n))$。

## E

> [!NOTE] **E**
> 
> 给出三个长度为 $n$ 的正整数序列 $a$，$b$，$c$。枚举 $1\le i\le j\le n$，求 $a_i\cdot b_j\cdot \min_{i\le k\le j}c_k$ 的最大值。

本题同样有许多做法，这里我们重点讲解并查集思路。按权值从大到小考虑 $c_k$。相当于我们在 $k$ 上加入一个点，然后将 $k-1$ 和 $k+1$ 位置上的点所在的连通块与之合并（如果这两个位置上有点的话）。连通块上记录 $a$ 的最大值和 $b$ 的最大值，即可在合并的时候更新答案。时间复杂度 $O(n\log n)$。

## F

> [!NOTE] **F**
> 
> 给出一棵 $n$ 个点的树，接下来有 $m$ 次操作：
> 
> - 加一条从 $a_i$ 到 $b_i$ 的边。
> 
> - 询问两个点 $u_i$ 和 $v_i$ 之间是否有至少两条边不相交的路径。

询问可以转化为：求 $u_i$ 和 $v_i$ 是否在同一个简单环上。按照双连通分量缩点的想法，每次我们在 $a_i$ 和 $b_i$ 间加一条边，就可以把 $a_i$ 到 $b_i$ 树上路径的点缩到一起。如果两条边 $(a_i,b_i)$ 和 $(a_j,b_j)$ 对应的树上路径有交，那么这两条边就会被缩到一起。

换言之，加边操作可以理解为，将 $a_i$ 到 $b_i$ 树上路径的边覆盖一次。而询问就转化为了：判断 $u_i$ 到 $v_i$ 路径上是否存在未被覆盖的边。如果不存在，那么 $u_i$ 和 $v_i$ 就属于同一个双连通分量，也就属于同一个简单环。

考虑使用并查集维护。给树定根，设 $f_i$ 表示 $i$ 到根的路径中第一个未被覆盖的边。那么每次加边操作，我们就暴力跳并查集。覆盖了一条边后，将这条边对应结点的 $f$ 与父节点合并。这样，每条边至多被覆盖一次，总复杂度 $O(n\log n)$。使用按秩合并的并查集同样可以做到 $O(n\alpha(n))$。

本题的维护方式类似于 D 的树上版本。

## G

> [!NOTE] **G**
> 
> 无向图 $G$ 有 $n$ 个点，初始时均为孤立点（即没有边）。
> 
> 接下来有 $m$ 次加边操作，第 $i$ 次操作在 $a_i$ 和 $b_i$ 之间加一条无向边。
> 
> 每次操作后，你均需要求出图中桥的个数。
> 
> 桥的定义为：对于一条 $G$ 中的边 $(x,y)$，如果删掉它会使得连通块数量增加，则 $(x,y)$ 被称作桥。
> 
> 强制在线。

本题考察对并查集性质的理解。考虑用并查集维护连通情况。对于边双树，考虑维护有根树，设 $p_i$ 表示结点 $i$ 的父亲。也就是不带路径压缩的并查集。

如果第 $i$ 次操作 $a_i$ 和 $b_i$ 属于同一个连通块，那么我们就需要将边双树上 $a_i$ 到 $b_i$ 路径上的点缩起来。这可以用并查集维护。每次缩点，边双连通分量的个数减少 $1$，最多减少 $n-1$ 次，因此缩点部分的并查集复杂度是 $O(n\alpha(n))$。

为了缩点，我们要先求出 $a_i$ 和 $b_i$ 在边双树上的 LCA。对此我们可以维护一个标记数组。然后从 $a_i$ 和 $b_i$ 开始轮流沿着祖先一个一个往上跳，并标记沿途经过的点。一但跳到了某个之前就被标记过的点，那么这个点就是 $a_i$ 和 $b_i$ 的 LCA。这个算法的复杂度与 $a_i$ 到 $b_i$ 的路径长度是线性相关的，可以接受。

如果 $a_i$ 和 $b_i$ 分属于两个不同连通块，那么我们将这两个连通块合并，并且桥的数量加 $1$。此时我们需要将两个点所在的边双树连起来，也就是加一条 $a_i$ 到 $b_i$ 的边。因此我们需要将其中一棵树重新定根，然后接到另一棵树上。这里运用启发式合并的思想：我们把结点数更小的重新定根。这样的总复杂度是 $O(n\log n)$ 的。

综上，该算法的总复杂度是 $O(n\log n+m\log n)$ 的。

## 小结

并查集与 Kruskal 重构树有许多共通点，而并查集的优化（按秩合并）正是启发式合并思想的应用。因此灵活运用并查集可以方便地处理许多与连通性有关的图论问题。


## 习题

### 一般联通性质

> [!NOTE] **[LeetCode 1562. 查找大小为 M 的最新分组](https://leetcode-cn.com/problems/find-latest-group-of-size-m/)**
> 
> 题意: 
> 
> 长度为 n 起始全 0 的串，每次会有一位变成 1（因此共n次操作），求满足存在一个连续长为 m 的最后一个操作的次数。

> [!TIP] **思路**
> 
> - 自己做法：考虑维护 起点->连续1个数
> 
>   这个做法其实和题解区 O(n) 解法的思路有一点点像：
> 
>   题解区考虑记录 每一个长度区间 => 有多少个
> 
> - 并查集

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findLatestStep(vector<int>& arr, int m) {
        int n = arr.size(), step = 0, res = -1;
        // 如果直接记录个数 显然 o(n^2)会超时
        // 维护int 起始位置 连续个数
        map<int, int> mp;
        map<int, int> vis;
        for (auto v : arr) {
            ++step;
            auto t = mp.lower_bound(v);
            int start = v, cnt = 1;
            if (!mp.empty() && t != mp.begin()) {
                auto [ts, tcnt] = *t;
                auto s = --t;
                ++t;
                auto [ss, scnt] = *s;
                if (scnt) {
                    if (ss + scnt == v) {
                        mp.erase(s);
                        --vis[scnt];
                        start = ss, cnt = scnt + 1;
                    }
                }
            }
            if (!mp.empty() && t != mp.end()) {
                auto [ts, tcnt] = *t;
                if (tcnt) {
                    if (start + cnt == ts) {
                        mp.erase(t);
                        --vis[tcnt];
                        cnt += tcnt;
                    }
                }
            }
            ++vis[cnt];
            if (vis[m]) res = step;
            mp[start] = cnt;
        }
        return res;
    }
};
```

##### **C++ 并查集**

```cpp
class Solution {
public:
    static const int maxn = 100005;
    int par[maxn], sz[maxn], b[maxn];
    int find(int x) { return x == par[x] ? x : par[x] = find(par[x]); }
    int cnt[maxn];
    void merge(int x, int y) {
        int a = find(x), b = find(y);
        if (a != b) {
            cnt[sz[a]]--;
            cnt[sz[b]]--;
            par[a] = b;
            sz[b] += sz[a];
            cnt[sz[b]]++;
        }
    }

    int findLatestStep(vector<int>& a, int m) {
        int n = a.size();
        for (int i = 1; i <= n; ++i) par[i] = i, sz[i] = 1;
        int ret = 0, tp = 1;
        for (int i : a) {
            b[i] = 1;
            cnt[1]++;
            if (b[i - 1]) { merge(i, i - 1); }
            if (b[i + 1]) { merge(i, i + 1); }
            if (cnt[m]) ret = tp;
            ++tp;
        }
        if (!ret) return -1;
        return ret;
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

### 离线并查集

> [!NOTE] **[LeetCode 1697. 检查边长度限制的路径是否存在](https://leetcode-cn.com/problems/checking-existence-of-edge-length-limited-paths/)** [TAG]
> 
> [Weekly-220](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2020-12-20_Weekly-220)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 离线并查集
> 
> 还有 LCA 优化的做法，虚拟比赛的时候想的是 MST + LCA
> 
> TODO 补上这部分代码

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 100010;

struct Node {
    int a, b, c, d;
    bool operator< (const Node& t) const {
        return c < t.c;
    }
}e[N], q[N];

class Solution {
public:
    vector<int> p;

    int find(int x) {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>>& ee, vector<vector<int>>& qq) {
        int m = ee.size(), k = qq.size();
        for (int i = 0; i < m; ++ i )
            e[i] = {ee[i][0], ee[i][1], ee[i][2]};
        for (int i = 0; i < k; ++ i )
            q[i] = {qq[i][0], qq[i][1], qq[i][2], i};
        sort(e, e + m), sort(q, q + k);
        p.resize(n);
        for (int i = 0; i < n; ++ i ) p[i] = i;
        vector<bool> res(k);
        for (int i = 0, j = 0; i < k; ++ i ) {
            while (j < m && e[j].c < q[i].c) {
                int a = e[j].a, b = e[j].b;
                p[find(a)] = find(b);
                ++ j ;
            }
            res[q[i].d] = find(q[i].a) == find(q[i].b);
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

> [!NOTE] **[LeetCode 803. 打砖块](https://leetcode.cn/problems/bricks-falling-when-hit/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 离线并查集 逆序操作处理击打的砖块
> 
> **经典 重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 离线并查集 逆序操作处理击打的砖块
    const static int N = 4e4 + 10;
    
    int p[N], sz[N];
    void init() {
        for (int i = 0; i < N; ++ i )
            p[i] = i, sz[i] = 1;
    }
    int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }

    int n, m;
    int get(int x, int y) {
        return x * m + y;
    }

    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};

    vector<int> hitBricks(vector<vector<int>>& grid, vector<vector<int>>& hits) {
        this->n = grid.size(), this->m = grid[0].size();
        int S = n * m;  // 顶作为虚拟原点
        init();

        // 先将要击打的砖块移除，后面离线加回
        vector<bool> st;
        for (auto & p : hits) {
            int x = p[0], y = p[1];
            if (grid[x][y]) {
                st.push_back(true);
                grid[x][y] = 0;
            } else
                st.push_back(false);
        }

        // 移除后，处理现有图形
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (grid[i][j]) {
                    int a = get(i, j);
                    if (!i) {
                        // 特殊处理最顶层的砖块
                        if (find(S) != find(a)) {
                            sz[find(S)] += sz[find(a)]; // 先操作sz
                            p[find(a)] = find(S);
                        }
                    }
                    for (int k = 0; k < 4; ++ k ) {
                        int x = i + dx[k], y = j + dy[k];
                        if (x < 0 || x >= n || y < 0 || y >= m || grid[x][y] == 0)
                            continue;
                        int b = get(x, y);
                        if (find(a) != find(b)) {
                            sz[find(b)] += sz[find(a)];
                            p[find(a)] = find(b);
                        }
                    }
                }
        
        // 逆序加回，通过联通量大小变化获取掉落砖块数量
        vector<int> res(hits.size());
        int last = sz[find(S)];
        for (int i = hits.size() - 1; i >= 0; -- i )
            if (st[i]) {
                // 加回
                int x = hits[i][0], y = hits[i][1];
                grid[x][y] = 1;
                int a = get(x, y);
                if (!x) {
                    // 特殊处理顶部的砖块
                    if (find(S) != find(a)) {
                        sz[find(S)] += sz[find(a)];
                        p[find(a)] = find(S);
                    }
                }
                for (int k = 0; k < 4; ++ k ) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m || grid[nx][ny] == 0)
                        continue;
                    int b = get(nx, ny);
                    if (find(a) != find(b)) {
                        sz[find(b)] += sz[find(a)];
                        p[find(a)] = find(b);
                    }
                }

                // 变化的量（就是下面连接的部分的大小） 加上这个节点本身
                res[i] = max(0, sz[find(S)] - last - 1);
                last = sz[find(S)];
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

### 联通性质进阶: 线段染色 / 快速找右侧的一个

> [!NOTE] **[LeetCode 1851. 包含每个查询的最小区间](https://leetcode-cn.com/problems/minimum-interval-to-include-each-query/)** [TAG]
> 
> [weekly-239](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-05-02_Weekly-239)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 并查集 经典模型 疯狂的馒头
> 
> 注意细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 并查集 经典模型 疯狂的馒头
    vector<int> xs, p, w;
    
    int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }
    
    int get(int x) {
        return lower_bound(xs.begin(), xs.end(), x) - xs.begin();
    }
    
    vector<int> minInterval(vector<vector<int>>& segs, vector<int>& queries) {
        for (auto & s : segs)
            xs.push_back(s[0]), xs.push_back(s[1]);
        for (auto x : queries)
            xs.push_back(x);
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());
        
        int n = xs.size();
        p.resize(n + 1), w.resize(n + 1, -1);
        for (int i = 0; i < n + 1; ++ i )
            p[i] = i;
        
        // 按区间长度排序 优先染短区间【细节】
        sort(segs.begin(), segs.end(), [](vector<int> & a, vector<int> & b) {
            return a[1] - a[0] < b[1] - b[0];
        });
        
        for (auto & s : segs) {
            int l = get(s[0]), r = get(s[1]), len = s[1] - s[0] + 1;
            while (find(l) <= r) {
                l = find(l);
                w[l] = len;
                p[l] = l + 1;
            }
        }
        
        vector<int> res;
        for (auto x : queries)
            res.push_back(w[get(x)]);
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

> [!NOTE] **[Codeforces Vessels](http://codeforces.com/problemset/problem/371/D)**
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

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *
