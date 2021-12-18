
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

> [!NOTE] **[AcWing 1252. 搭配购买](https://www.acwing.com/problem/content/1254/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 并查集结合 01 背包

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 10010;

int n, m, vol;
int v[N], w[N];
int p[N];
int f[N];

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int main() {
    cin >> n >> m >> vol;
    for (int i = 1; i <= n; ++ i ) p[i] = i;
    for (int i = 1; i <= n; ++ i ) cin >> v[i] >> w[i];
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        int pa = find(a), pb = find(b);
        if (pa != pb) {
            v[pb] += v[pa];
            w[pb] += w[pa];
            p[pa] = pb;
        }
    }
    // 01背包
    for (int i = 1; i <= n; ++ i )
        if (p[i] == i)  // 选择根的技巧
            for (int j = vol; j >= v[i]; -- j )
                f[j] = max(f[j], f[j - v[i]] + w[i]);
    cout << f[vol] << endl;
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

> [!NOTE] **[AcWing 237. 程序自动分析](https://www.acwing.com/problem/content/239/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **先处理相等条件**的思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 1000010;

struct Query{
    int x, y, e;
}query[N];

int n, m;
int p[N];
unordered_map<int, int> S;  // 数据太大 在此离散化
// 因为分析知询问顺序对结果无影响 故先考虑相等元素

int get(int x) {
    if (S.count(x) == 0) S[x] = ++ n ;
    return S[x];
}

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int main() {
    int t;
    cin >> t;
    while (t -- ) {
        //memset(p, 0, sizeof p);
        n = 0;
        S.clear();
        
        cin >> m;
        for (int i = 0; i < m; ++ i ) {
            int x, y, e;
            cin >> x >> y >> e;
            query[i] = {get(x), get(y), e};
        }
        
        for (int i = 1; i <= n; ++ i ) p[i] = i;
        
        // 相等条件 此时不可能产生矛盾
        for (int i = 0; i < m; ++ i )
            if (query[i].e == 1) {
                int pa = find(query[i].x), pb = find(query[i].y);
                p[pa] = pb;
            }
        // 不等条件
        bool has_conflict = false;
        for (int i = 0; i < m; ++ i )
            if (query[i].e == 0) {
                int pa = find(query[i].x), pb = find(query[i].y);
                if (pa == pb) {
                    has_conflict = true;
                    break;
                    // 无需考虑不相等的情况 因为总有值可以满足
                }
            }
        if (has_conflict) cout << "NO" << endl;
        else cout << "YES" << endl;
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

> [!NOTE] **[AcWing 239. 奇偶游戏](https://www.acwing.com/problem/content/241/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数据范围暗示应用离散化
> 
> 以及前缀和
> 
> S[L] S[R] 奇偶个数  ====>  S[R] S[L-1] 奇偶性是否相同

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 20010;

int n, m;
int p[N], d[N];
unordered_map<int, int> S;

int get(int x) {
    if (S.count(x) == 0) S[x] = ++ n ;
    return S[x];
}

int find(int x) {
    if (p[x] != x) {
        int root = find(p[x]);
        // 0 同类 1 不同类
        //d[x] += d[p[x]];
        d[x] ^= d[p[x]];
        p[x] = root;
    }
    return p[x];
}

int main() {
    cin >> n >> m;
    n = 0;
    for (int i = 0; i < N; ++ i ) p[i] = i;
    
    int res = m;
    for (int i = 1; i <= m; ++ i ) {
        int a, b;
        string type;
        cin >> a >> b >> type;
        a = get(a - 1), b = get(b);
        
        int t = 0;
        if (type == "odd") t = 1;
        
        int pa = find(a), pb = find(b);
        if (pa == pb) {
            // 同一个集合内：意味着其相对关系已知
            //
            if ((d[a] ^ d[b]) != t) {
            //if (((d[a] + d[b]) % 2 + 2) % 2 != t) {
                res = i - 1;
                break;
            }
        } else {
            p[pa] = pb;
            d[pa] = d[a] ^ d[b] ^ t;
        }
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

> [!NOTE] **[AcWing 238. 银河英雄传说](https://www.acwing.com/problem/content/240/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 边带权
> 
> **在之前的一些题目中可以直接使用 `a = find(a)` 这样覆盖原本的 a ，本题不可行，因为原来的 a, b 仍有需要**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 30010;

int p[N], d[N], sz[N];

int find(int x) {
    if (p[x] != x) {
        int root = find(p[x]);
        d[x] += d[p[x]];
        p[x] = root;
    }
    return p[x];
}

int main() {
    
    for (int i = 0; i < N; ++ i ) p[i] = i, d[i] = 0, sz[i] = 1;
    
    int t;
    cin >> t;
    
    char op[2];
    int a, b;
    while (t -- ) {
        cin >> op >> a >> b;
        int pa = find(a), pb = find(b);
        if (op[0] == 'M') {
            d[pa] = sz[pb];
            sz[pb] += sz[pa];
            p[pa] = pb;
        } else {
            if (pa != pb) cout << -1 << endl;
            else cout << max(0, abs(d[a] - d[b]) - 1) << endl;
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

### 离线并查集

> [!NOTE] **[LeetCode 1697. 检查边长度限制的路径是否存在](https://leetcode-cn.com/problems/checking-existence-of-edge-length-limited-paths/)**
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

### 并查集线段染色

> [!NOTE] **[LeetCode 1851. 包含每个查询的最小区间](https://leetcode-cn.com/problems/minimum-interval-to-include-each-query/)**
> 
> [weekly-239](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-05-02_Weekly-239)
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