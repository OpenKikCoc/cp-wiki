
![](images/disjoint-set.svg)

并查集是一种树形的数据结构，顾名思义，它用于处理一些不交集的 **合并** 及 **查询** 问题。
它支持两种操作：

- 查找（Find）：确定某个元素处于哪个子集；
- 合并（Union）：将两个子集合并成一个集合。

> [!WARNING]
> 
> 并查集不支持集合的分离，但是并查集在经过修改后可以支持集合中单个元素的删除操作（详见 UVA11987 Almost Union-Find）。使用动态开点线段树还可以实现可持久化并查集。

## 初始化


## 查找

通俗地讲一个故事：几个家族进行宴会，但是家族普遍长寿，所以人数众多。由于长时间的分离以及年龄的增长，这些人逐渐忘掉了自己的亲人，只记得自己的爸爸是谁了，而最长者（称为「祖先」）的父亲已经去世，他只知道自己是祖先。为了确定自己是哪个家族，他们想出了一个办法，只要问自己的爸爸是不是祖先，一层一层的向上问，直到问到祖先。如果要判断两人是否在同一家族，只要看两人的祖先是不是同一人就可以了。

在这样的思想下，并查集的查找算法诞生了。

![](images/disjoint-set-find.svg)

显然这样最终会返回 $x$ 的祖先。

### 路径压缩

这样的确可以达成目的，但是显然效率实在太低。为什么呢？因为我们使用了太多没用的信息，我的祖先是谁与我父亲是谁没什么关系，这样一层一层找太浪费时间，不如我直接当祖先的儿子，问一次就可以出结果了。甚至祖先是谁都无所谓，只要这个人可以代表我们家族就能得到想要的效果。**把在路径上的每个节点都直接连接到根上**，这就是路径压缩。

![](images/disjoint-set-compress.svg)

## 合并

宴会上，一个家族的祖先突然对另一个家族说：我们两个家族交情这么好，不如合成一家好了。另一个家族也欣然接受了。  
我们之前说过，并不在意祖先究竟是谁，所以只要其中一个祖先变成另一个祖先的儿子就可以了。

![](images/disjoint-set-merge.svg)

### 启发式合并（按秩合并）

一个祖先突然抖了个机灵：「你们家族人比较少，搬家到我们家族里比较方便，我们要是搬过去的话太费事了。」

由于需要我们支持的只有集合的合并、查询操作，当我们需要将两个集合合二为一时，无论将哪一个集合连接到另一个集合的下面，都能得到正确的结果。但不同的连接方法存在时间复杂度的差异。具体来说，如果我们将一棵点数与深度都较小的集合树连接到一棵更大的集合树下，显然相比于另一种连接方案，接下来执行查找操作的用时更小（也会带来更优的最坏时间复杂度）。

当然，我们不总能遇到恰好如上所述的集合————点数与深度都更小。鉴于点数与深度这两个特征都很容易维护，我们常常从中择一，作为估价函数。而无论选择哪一个，时间复杂度都为 $O (m\alpha(m,n))$，具体的证明可参见 References 中引用的论文。

在算法竞赛的实际代码中，即便不使用启发式合并，代码也往往能够在规定时间内完成任务。在 Tarjan 的论文[1]中，证明了不使用启发式合并、只使用路径压缩的最坏时间复杂度是 $O (m \log n)$。在姚期智的论文[2]中，证明了不使用启发式合并、只使用路径压缩，在平均情况下，时间复杂度依然是 $O (m\alpha(m,n))$。

如果只使用启发式合并，而不使用路径压缩，时间复杂度为 $O(m\log n)$。由于路径压缩单次合并可能造成大量修改，有时路径压缩并不适合使用。例如，在可持久化并查集、线段树分治 + 并查集中，一般使用只启发式合并的并查集。

C++ 的参考实现，其选择点数作为估价函数：

```cpp
// C++ Version
std::vector<int> size(N, 1);  // 记录并初始化子树的大小为 1
void unionSet(int x, int y) {
    int xx = find(x), yy = find(y);
    if (xx == yy) return;
    if (size[xx] > size[yy])  // 保证小的合到大的里
        swap(xx, yy);
    fa[xx] = yy;
    size[yy] += size[xx];
}
```

Python 的参考实现：

```python
# Python Version
size = [1] * N # 记录并初始化子树的大小为 1
def unionSet(x, y):
    xx = find(x); yy = find(y)
    if xx == yy:
        return
    if size[xx] > size[yy]: # 保证小的合到大的里
        xx, yy = yy, xx
    fa[xx] = yy
    size[yy] = size[yy] + size[xx]
```

## 时间复杂度及空间复杂度

### 时间复杂度

同时使用路径压缩和启发式合并之后，并查集的每个操作平均时间仅为 $O(\alpha(n))$，其中 $\alpha$ 为阿克曼函数的反函数，其增长极其缓慢，也就是说其单次操作的平均运行时间可以认为是一个很小的常数。

[Ackermann 函数](https://en.wikipedia.org/wiki/Ackermann_function)  $A(m, n)$ 的定义是这样的：

$A(m, n) = \begin{cases}n+1&\text{if }m=0\\A(m-1,1)&\text{if }m>0\text{ and }n=0\\A(m-1,A(m,n-1))&\text{otherwise}\end{cases}$

而反 Ackermann 函数 $\alpha(n)$ 的定义是阿克曼函数的反函数，即为最大的整数 $m$ 使得 $A(m, m) \leqslant n$。

时间复杂度的证明 [在这个页面中](./dsu-complexity.md)。

### 空间复杂度

显然为 $O(n)$。

## 带权并查集

我们还可以在并查集的边上定义某种权值、以及这种权值在路径压缩时产生的运算，从而解决更多的问题。比如对于经典的「NOI2001」食物链，我们可以在边权上维护模 3 意义下的加法群。

## 经典题目

[「NOI2015」程序自动分析](https://uoj.ac/problem/127)

[「JSOI2008」星球大战](https://www.luogu.com.cn/problem/P1197)

[「NOI2001」食物链](https://www.luogu.com.cn/problem/P2024)

[「NOI2002」银河英雄传说](https://www.luogu.com.cn/problem/P1196)

[UVA11987 Almost Union-Find](https://www.luogu.com.cn/problem/UVA11987)

## 其他应用

[最小生成树算法](graph/mst.md) 中的 Kruskal 和 [最近公共祖先](graph/lca.md) 中的 Tarjan 算法是基于并查集的算法。

相关专题见 [并查集应用](topic/dsu-app.md)。

## 参考资料与拓展阅读

- [1]Tarjan, R. E., & Van Leeuwen, J. (1984). Worst-case analysis of set union algorithms. Journal of the ACM (JACM), 31(2), 245-281.[ResearchGate PDF](https://www.researchgate.net/profile/Jan_Van_Leeuwen2/publication/220430653_Worst-case_Analysis_of_Set_Union_Algorithms/links/0a85e53cd28bfdf5eb000000/Worst-case-Analysis-of-Set-Union-Algorithms.pdf)
- [2]Yao, A. C. (1985). On the expected performance of path compression algorithms.[SIAM Journal on Computing, 14(1), 129-133.](https://epubs.siam.org/doi/abs/10.1137/0214010?journalCode=smjcat)
- [3][知乎回答：是否在并查集中真的有二分路径压缩优化？](<https://www.zhihu.com/question/28410263/answer/40966441>)
- [4]Gabow, H. N., & Tarjan, R. E. (1985). A Linear-Time Algorithm for a Special Case of Disjoint Set Union. JOURNAL OF COMPUTER AND SYSTEM SCIENCES, 30, 209-221.[CORE PDF](https://core.ac.uk/download/pdf/82125836.pdf)


## 习题

### 并查集初步

> [!NOTE] **[AcWing 836. 合并集合](https://www.acwing.com/problem/content/838/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int p[N], n, m;

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) p[i] = i;
    
    char op[2];
    int a, b;
    while (m -- ) {
        cin >> op >> a >> b;
        a = find(a), b = find(b);
        if (op[0] == 'M') {
            p[a] = b;
        } else {
            if (a == b) cout << "Yes" << endl;
            else cout << "No" << endl;
        }
    }
    return 0;
}
```

##### **Python**

```python
"""
I. 概念
1. 并查集：主要用于解决一些 **元素分组** 的问题，它管理一系列不相交的集合，并支持两种操作：
   - 合并：把两个不相交的集合合并成一个集合
   - 查询：查询两个元素是否在同一集合中
2. 基本原理：每个集合用一棵树来表示；树根的编号就是整个集合的编号 父节点，p[x]表示x的父节点

II. 理解模版代码
1. 如何判断树根？ if p[x]==x
2. 如何求x的集合编号： while p(x)!=x:x=p[x]
3. 如何合并两个集合：px是x的集合编号，py是y的集合编号：p[x]=y

"""

N = 100010
p = [0] * N


# 查询：用递归的写法实现对元素的查询：一层一层访问父节点，直到祖宗节点（根节点的标志就是父节点是本身
# 要判断两个元素是否属于同一集合，只需要看它们的根节点是否相同即可

def find(x):  # 递归回溯的条件是 p[x] == x, 即找到祖宗节点
    if p[x] != x:  # x的父节点不是祖宗节点
        p[x] = find(p[x])  # 找寻父节点的祖宗节点
    return p[x]


if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):  # 初始化每个元素的祖宗节点为本身
        p[i] = i
    for _ in range(m):
        oper = input().split()
        if oper[0] == 'M':
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b): continue

            # 合并：先找到两个集合的代表元素，然后将前者的父节点设为后者即可
            p[find(a)] = find(b)  # 使a的祖宗节点变成b的祖宗节点
            # 踩坑：p[find(a)] 是把b的祖宗节点 肤质给 找到的a的祖宗节点！错误写法： p[a] = find(b) !xxx 非常错！ 踩坑好几次了！！！
        else:
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b):
                print("Yes")
            else:
                print("No")
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 684. 冗余连接](https://leetcode-cn.com/problems/redundant-connection/)**
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
    vector<int> p;
    int find(int x) {
        if (p[x] != x) return p[x] = find(p[x]);
        return p[x];
    }
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        p.resize(n + 1);
        for (int i = 1; i <= n; ++ i ) p[i] = i;

        for (auto & e : edges) {
            int a = find(e[0]), b = find(e[1]);
            if (a != b) p[a] = p[b];
            else return e;
        }
        return {};
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


> [!NOTE] **[LeetCode 685. 冗余连接 II](https://leetcode-cn.com/problems/redundant-connection-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> arjan找环 理清楚几种情况
> 
> 优雅的实现思路
> 
> 1. 出现了环 （有一条边连向了根结点）删掉一边即可
> 2. 无环 但某个点入度为二（某个一子树节点连向了另一子树节点）去除两个入度的任意一个都可
> 3. 环且某个点入度为二（某个节点连向了其祖先节点）只能删掉12情况的公共边

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ tarjan**

```cpp
class Solution {
public:
    int n;
    vector<bool> st1, st2, st, in_k, in_c;
    vector<vector<int>> g;
    stack<int> stk;

    bool dfs(int u) {
        st[u] = true;
        stk.push(u), in_k[u] = true;

        for (int x : g[u]) {
            if (!st[x]) {
                if (dfs(x))
                    return true;
            } else if (in_k[x]) {
                while (stk.top() != x) {
                    in_c[stk.top()] = true;
                    stk.pop();
                }
                in_c[x] = true;
                return true;
            }
        }

        stk.pop(), in_k[u] = false;
        return false;
    }

    void work1(vector<vector<int>> & edges) {
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            g[a].push_back(b);
        }
        for (int i = 1; i <= n; ++ i )
            if (!st[i] && dfs(i))
                break;
        for (int i = 0; i < n; ++ i ) {
            int a = edges[i][0], b = edges[i][1];
            if (in_c[a] && in_c[b])
                st1[i] = true;
        }
    }

    void work2(vector<vector<int>> & edges) {
        vector<int> p(n + 1, -1);
        for (int i = 0; i < n; ++ i ) {
            int a = edges[i][0], b = edges[i][1];
            if (p[b] != -1) {
                st2[p[b]] = st2[i] = true;
                break;
            } else
                p[b] = i;
        }
    }

    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
        n = edges.size();
        st1 = st2 = st = in_k = in_c = vector<bool>(n + 1);
        g.resize(n + 1);
        work1(edges);
        work2(edges);

        for (int i = n - 1; i >= 0; -- i )
            if (st1[i] && st2[i])
                return edges[i];
        for (int i = n - 1; i >= 0; -- i )
            if (st1[i] || st2[i])
                return edges[i];
        return {};
    }
};
```

##### **C++ 旧代码 并查集**

```cpp
struct UnionFind {
    vector<int> fa;
    UnionFind(int n) {
        fa.resize(n);
        for (int i = 0; i < n; ++ i ) fa[i] = i;
    }
    int find(int x) {
        if (fa[x] == x) return x;
        return fa[x] = find(fa[x]);
    }
    void merge(int u, int v) {
        fa[find(u)] = find(v);
    }
};

class Solution {
public:
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        UnionFind uf = UnionFind(n+1);
        vector<int> pa(n + 1);
        for (int i = 1; i <= n; ++ i ) pa[i] = i;
        int conflict = -1, cycle = -1;
        for (int i = 0; i < n; ++ i ) {
            int u = edges[i][0], v = edges[i][1];
            if (pa[v] != v) conflict = i;       // v作为后继已经有了前序
            else {
                pa[v] = u;
                if (uf.find(u) == uf.find(v)) cycle = i;      // 成环
                else uf.merge(u, v);
            }
        }
        if (conflict < 0) return edges[cycle];
        else {
            if (cycle < 0) return edges[conflict];
            else return vector<int>{pa[edges[conflict][1]], edges[conflict][1]};
        }
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

> [!NOTE] **[LeetCode 1319. 连通网络的操作次数](https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> n 个节点已有连接 connections , 检查至少还需要拆多少个绳子
> 
> 一开始做法是检查有多少个节点完全没有绳子连接 返回个数 但这样没有考虑到已经用绳子连接的部分可能不联通 有绳子不联通的统一需要加绳子
> 
> ==> 并查集 检查联通量个数即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int find(int x, vector<int>& fa) {
        return x == fa[x] ? x : fa[x] = find(fa[x], fa);
    }
    int makeConnected(int n, vector<vector<int>>& connections) {
        int cn = connections.size();
        if (n > cn + 1) return -1;

        vector<int> fa(n);
        for (int i = 0; i < n; ++i) fa[i] = i;
        int part = n;
        for (auto v : connections) {
            int f0 = find(v[0], fa), f1 = find(v[1], fa);
            if (f0 != f1) {
                --part;
                fa[f0] = f1;
            }
        }
        return part - 1;
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

> [!NOTE] **[LeetCode 1627. 带阈值的图连通性](https://leetcode-cn.com/problems/graph-connectivity-with-threshold/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 枚举因子，然后将其所有的倍数与这一因子之间连边。
> 
> **一开始在想怎么找某个数的因子，重要的是先找因子再向后连接**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
    int fa[100005], sz[100005];
    int Find(int x) {
        if (fa[x] == x) return x;
        return fa[x] = Find(fa[x]);
    }
    void Merge(int x, int y) {
        int fx = Find(x), fy = Find(y);
        if (fx == fy) return;
        if (sz[fx] > sz[fy])
            fa[fy] = fx, sz[fx] += sz[fy];
        else
            fa[fx] = fy, sz[fy] += sz[fx];
    }

public:
    vector<bool> areConnected(int n, int threshold,
                              vector<vector<int>>& queries) {
        for (int i = 1; i <= n; ++i) fa[i] = i, sz[i] = 1;
        for (int i = threshold + 1; i <= n; ++i)
            for (int j = i; j <= n; j += i) Merge(i, j);
        vector<bool> res;
        for (auto& q : queries) res.push_back(Find(q[0]) == Find(q[1]));
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

### 带权并查集

> [!NOTE] **[AcWing 240. 食物链](https://www.acwing.com/problem/content/242/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

使用 0/1/2 表示同类、吃、被吃的关系

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 50010;

int n, m;
int p[N], d[N];

int find(int x) {
    if (p[x] != x) {
        int root = find(p[x]);
        d[x] += d[p[x]];
        p[x] = root;
    }
    return p[x];
}

int main() {
    cin >> n >> m;
    
    for (int i = 1; i <= n; ++ i ) p[i] = i;
    
    int res = 0;
    int t, x, y;
    while (m -- ) {
        cin >> t >> x >> y;
        if (x > n || y > n) ++ res ;
        else {
            int px = find(x), py = find(y);
            if (t == 1) {
                if (px == py && (d[x] - d[y]) % 3) ++ res ;
                else if (px != py) {
                    p[px] = py;
                    d[px] = d[y] - d[x];
                }
            } else {
                if (px == py && (d[x] - d[y] - 1) % 3) ++ res ;
                else if (px != py) {
                    p[px] = py;
                    d[px] = d[y] + 1 - d[x];
                }
            }
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

> [!NOTE] **[LeetCode 399. 除法求值](https://leetcode-cn.com/problems/evaluate-division/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 并查集 OR floyd

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ DSU**

```cpp
// 带权并查集
class Solution {
public:
    unordered_map<string, string> p;  //记录祖先节点
    unordered_map<string, double> d;  //到祖先节点的距离

    string find (string x) {
        if (p[x] != x) {
            string t = find(p[x]);
            d[x] *= d[p[x]];  //p[x] / root
            p[x] = t;
        }
        return p[x];
    }

    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        //初始化所有节点
        for (int i = 0; i < equations.size(); i ++ ) {
            string a = equations[i][0], b = equations[i][1];
            p[a] = a, p[b] = b;
            d[a] = 1, d[b] = 1;
        }
        //集合合并
        for (int i = 0; i < equations.size(); i ++ ) {
            string a = equations[i][0], b = equations[i][1];
            // ra -> b
            string ra = find(a); //找到a的祖先节点
            //更新ra节点
            p[ra] = b;
            d[ra] = values[i] / d[a];
        }
        vector<double> res;
        for (auto q : queries) {
            string a = q[0], b = q[1];
            //如果a、b处于同一集合会自动算出d[a]和d[b]
            if (!p.count(a) || !p.count(b) || find(a) != find(b))
                res.push_back(-1.0);
            else
                res.push_back(d[a] / d[b]);
        }
        return res;
    }
};
```

##### **C++ floyd**

```cpp
// floyd 连通性
class Solution {
public:
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        unordered_set<string> vers;
        unordered_map<string, unordered_map<string, double>> d;
        for (int i = 0; i < equations.size(); ++ i ) {
            auto a = equations[i][0], b = equations[i][1];
            auto c = values[i];
            d[a][b] = c, d[b][a] = 1 / c;
            vers.insert(a), vers.insert(b);
        }
        for (auto k : vers)
            for (auto i : vers)
                for (auto j : vers)
                    if (d[i][k] && d[k][j])
                        d[i][j] = d[i][k] * d[k][j];
        vector<double> res;
        for (auto q : queries) {
            auto a = q[0], b = q[1];
            if (d[a][b]) res.push_back(d[a][b]);
            else res.push_back(-1);
        }
        return res;
    }
};
```

##### **Python**

```python
"""
题解：建立有向图 + BFS。我们可以按照给定的关系去建立一个有向图，如果当前a / b = 2.0，那么w[a][b] = 2.0,w[b][a] = 0.5，询问假设为x / y，如果我们能从x出发通过BFS到达y，那么路径上的乘积就是我们需要的答案。如果待查询的字符串没有遇到过或者无法遍历到，就返回-1。

"""

class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        vertex = set()
        path = collections.defaultdict(dict)
        for (a, b), v in zip(equations, values):
            path[a][b] = v
            path[b][a] = 1/v
            # 把记录过的点加入到vertex里去
            vertex.add(a)
            vertex.add(b)

        # 把所有的点之间的距离都求出来，后面遍历就行了
        for k in vertex:
            for i in vertex:
                for j in vertex:
                    # i/k * j/k = i/k, 得到i ~ k的距离
                    if k in path[i].keys() and j in path[k].keys():
                        path[i][j] = path[i][k] * path[k][j]

        res = []
        for q in queries:
            up = q[0]
            down = q[1]
            # up/down 存在，返回结果
            if down in path[up].keys():
                res.append(path[up][down])
            else:
                res.append(-1)

        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

### 启发式合并

> [!NOTE] **[AcWing 2154. 梦幻布丁](https://www.acwing.com/problem/content/2156/)**
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

const int N = 100010, M = 1000010;  // 布丁数量 颜色

int n, m;
int h[M], e[N], ne[N], idx;         // 每个颜色 都有哪下标的布丁
int color[N], sz[M], p[M];          // 布丁颜色 颜色集合大小 颜色映射
int res;                            // 性质：随着颜色变化 段数只会减少不会增加

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    sz[a] ++ ;
}

void merge(int & x, int & y) {
    if (x == y)
        return;
    // 启发式合并 保证x较小且合入较大的y
    if (sz[x] > sz[y])
        swap(x, y);
    for (int i = h[x]; ~i; i = ne[i]) {
        int j = e[i];
        // 维护 res ===> 性质决定了只需要在此时减一下来维护
        res -= (color[j - 1] == y) + (color[j + 1] == y);
    }
    for (int i = h[x]; ~i; i = ne[i]) {
        int j = e[i];
        color[j] = y;
        // 头插法 直接把短的这一部分加入到新的的头部
        if (ne[i] == -1) {
            ne[i] = h[y], h[y] = h[x];
            break;
        }
    }
    h[x] = -1;
    sz[y] += sz[x], sz[x] = 0;
}

int main() {
    cin >> n >> m;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; ++ i ) {
        cin >> color[i];
        if (color[i] != color[i - 1])
            res ++ ;
        add(color[i], i);
    }
    
    // 初始化颜色与单链表的映射
    for (int i = 0; i < M; ++ i )
        p[i] = i;
    
    while (m -- ) {
        int op;
        cin >> op;
        if (op == 2)
            cout << res << endl;
        else {
            int x, y;
            cin >> x >> y;
            merge(p[x], p[y]);
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

> [!NOTE] **[AcWing 3189. Lomsat gelral](https://www.acwing.com/problem/content/3191/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 更多请参阅 **树上并查集**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 100010, M = N << 1;

int n;
int h[N], e[M], ne[M], idx;
int color[N], cnt[N], sz[N], son[N];
LL res[N], sum;
int mx;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 求重儿子
int dfs_son(int u, int pa) {
    sz[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == pa)
            continue;
        sz[u] += dfs_son(j, u);
        // 更新重儿子
        // 注意 son[u] 默认是0 大小也是0 所以可以这么写 trick
        if (sz[j] > sz[son[u]])
            son[u] = j;
    }
    return sz[u];
}

// sign = 1 表示累加， sign = -1 表示清除  pson 为重儿子
void update(int u, int pa, int sign, int pson) {
    int c = color[u];
    cnt[c] += sign; // cnt[c] ++ or cnt[c] -- 
    if (cnt[c] > mx)
        mx = cnt[c], sum = c;
    else if (cnt[c] == mx)
        sum += c;
    
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == pa || j == pson)
            continue;
        update(j, u, sign, pson);
    }
}

// op 表示儿子类型
void dfs(int u, int pa, int op) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == pa || j == son[u])
            continue;
        dfs(j, u, 0);
    }
    
    // 最后访问重儿子 以保留其信息
    if (son[u])
        dfs(son[u], u, 1);
    update(u, pa, 1, son[u]);
    
    res[u] = sum;
    
    if (!op)
        update(u, pa, -1, 0), sum = mx = 0;
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> color[i];
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    
    dfs_son(1, -1);
    dfs(1, -1, 1);
    
    for (int i = 1; i <= n; ++ i )
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

### 并查集与反集

> [!NOTE] **[Luogu [BOI2003]团伙](https://www.luogu.com.cn/problem/P1892)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 并查集 与 反集

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 关于并查集与反集

const int N = 1010;

int n, m;
int p[N << 1];  // 前 N 表示原集合，后 N 表示反集

void init() {
    for (int i = 1; i < N << 1; ++ i )
        p[i] = i;
}

int find(int x) {
    if (p[x] != x)
        p[x] = find(p[x]);
    return p[x];
}

int main() {
    init();
    
    cin >> n >> m;
    
    while (m -- ) {
        char op[2];
        int a, b;
        cin >> op >> a >> b;
        if (op[0] == 'F')
            p[find(a)] = find(b);
        else {
            // 反集合并
            //
            // 本质上 find(a + n) 记录了 a 的敌人的集合的根
            // 所以 find(a +  n) 可以理解为是一个范围 [1, n] 的数
            // 本操作将敌人与敌人合并
            p[find(a + n)] = find(b);
            p[find(b + n)] = find(a);
        }
    }
    
    int res = 0;
    for (int i = 1; i <= n; ++ i )
        if (find(i) == i)
            res ++ ;
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

### 并查集与图

> [!NOTE] **[AcWing 837. 连通块中点的数量](https://www.acwing.com/problem/content/839/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int n, m;
int p[N], sz[N];

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) p[i] = i, sz[i] = 1;
    
    char op[3];
    int a, b;
    while (m -- ) {
        cin >> op;
        if (op[0] == 'C') {
            cin >> a >> b;
            a = find(a), b = find(b);
            if (a != b) {
                // sz 相加要放在前面，不然先修改父节点就没法算了
                sz[b] += sz[a];
                p[a] = b;
            }
        } else if (op[0] == 'Q' && op[1] == '1') {
            cin >> a >> b;
            a = find(a), b = find(b);
            cout << (a == b ? "Yes" : "No") << endl;
        } else {
            cin >> a;
            cout << sz[find(a)] << endl;
        }
    }
    return 0;
}
```

##### **Python**

```python
N = 100010
p = [0] * N
size = [1] * N  # 初始化 根节点的值为该树的节点数1


def find(x):
    if p[x] != x:
        p[x] = find(p[x])
    return p[x]


if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):  # 初始化每个元素的树，让根节点等于自己。
        p[i] = i

    for _ in range(m):
        oper = input().split()
        if oper[0] == 'C':
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b): continue  # 特判：两个点在一个集合中就跳过合并
            size[find(b)] += size[find(a)]  # 更新新树的size，注意！！需要先将size合并才能p[a]=b; p[a]=b操作之后a的根就是b了，会出错
            p[find(a)] = find(b)  # a插入b中
        elif oper[0] == 'Q1':
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b):
                print("Yes")
            else:
                print("No")
        else:
            a = int(oper[1])
            print(size[find(a)])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1359. 城堡](https://www.acwing.com/problem/content/1361/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可以 flood fill 也可以并查集 并查集较短所以这里用并查集
> 
> **典型 并查集在图中的应用**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 55, M = N * N;

int n, m;
int g[N][N];
int p[M], sz[M];

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int main() {
    cin >> m >> n;
    for (int i = 0; i < n; ++ i )
        for (int j = 0; j < m; ++ j )
            cin >> g[i][j];
            
    for (int i = 0; i < n * m; ++ i )
        p[i] = i, sz[i] = 1;
    
    int dx[2] = {-1, 0}, dy[2] = {0, 1}, dw[2] = {2, 4};
    int cnt = n * m, max_area = 1;
    for (int x = 0; x < n; ++ x )
        for (int y = 0; y < m; ++ y )
            for (int u = 0; u < 2; ++ u )
                if (!(g[x][y] & dw[u])) {
                    int nx = x + dx[u], ny = y + dy[u];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                    int a = x * m + y, b = nx * m + ny;
                    a = find(a), b = find(b);
                    if (a != b) {
                        cnt -- ;
                        sz[b] += sz[a];
                        p[a] = b;
                        max_area = max(max_area, sz[b]);
                    }
                }
    cout << cnt << endl << max_area << endl;
    
    max_area = 0;
    int rx, ry, rw;
    for (int y = 0; y < m; ++ y )
        for (int x = n - 1; x >= 0; -- x )
            for (int u = 0; u < 2; ++ u )
                if (g[x][y] & dw[u]) {
                    int nx = x + dx[u], ny = y + dy[u];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                    int a = x * m + y, b = nx * m + ny;
                    a = find(a), b = find(b);
                    if (a != b) {
                        int area = sz[a] + sz[b];
                        if (area > max_area) {
                            max_area = area;
                            rx = x, ry = y, rw = u;
                        }
                    }
                }
    cout << max_area << endl;
    cout << rx + 1 << ' ' << ry + 1 << ' ' << (rw ? 'E' : 'N') << endl;
    
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

> [!NOTE] **[LeetCode 1361. 验证二叉树](https://leetcode-cn.com/problems/validate-binary-tree-nodes/)**
> 
> 题意:
> 
> 二叉树上有 n 个节点，按从 0 到 n - 1 编号，其中节点 i 的两个子节点分别是 leftChild[i] 和 rightChild[i]。
> 
> 只有 所有 节点能够形成且 只 形成 一颗 有效的二叉树时，返回 true；否则返回 false

> [!TIP] **思路**
> 
> 并查集
> 
> 题解有很多按度数算是错误的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int f[10005];
    int find(int x) {
        if (f[x] == x) return x;
        return f[x] = find(f[x]);
    }
    void merge(int i, int j) {
        int x = find(i), y = find(j);
        if (x != y) f[i] = j;
    }
    bool judge(int i, int j) { return find(i) == find(j); }
    bool validateBinaryTreeNodes(int n, vector<int>& leftChild,
                                 vector<int>& rightChild) {
        unordered_map<int, int> l, r, fa;
        for (int i = 1; i <= n; ++i) f[i] = i;
        int v, cnt = 0;
        for (int i = 0; i < n; ++i) {
            v = leftChild[i];
            if (v != -1) {
                ++v;
                if (l[i + 1] || fa[v]) return false;
                if (judge(i + 1, v)) return false;
                l[i + 1] = v;
                fa[v] = i + 1;
                merge(i + 1, v);
                ++cnt;
            }
        }
        for (int i = 0; i < n; ++i) {
            v = rightChild[i];
            if (v != -1) {
                ++v;
                if (r[i + 1] || fa[v]) return false;
                if (judge(i + 1, v)) return false;
                r[i + 1] = v;
                fa[v] = i + 1;
                ++cnt;
            }
        }
        return cnt == n - 1;
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

> [!NOTE] **[LeetCode 1579. 保证图可完全遍历](https://leetcode-cn.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
struct DSU {
    int f[100005];
    int N;
    void init(int n) {
        N = n;
        for (int i = 1; i <= n; ++i) f[i] = i;
    }
    int find(int x) {
        if (x == f[x]) return x;
        return f[x] = find(f[x]);
    }
    bool merge(int x, int y) {
        int fx = find(x), fy = find(y);
        if (fx == fy) return false;
        f[fx] = fy;
        --N;
        return true;
    }
} a, b;

class Solution {
public:
    // 先尽可能的保留公共边，并删除冗余的公共边，然后再删除单独的冗余边
    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        a.init(n);
        b.init(n);
        int res = 0;
        for (auto& e : edges)
            if (e[0] == 3) {
                bool f1 = a.merge(e[1], e[2]);
                bool f2 = b.merge(e[1], e[2]);
                if (!f1 && !f2) ++res;  // 对于AB来说都是冗余边
            }
        for (auto& e : edges) {
            if (e[0] == 1 && !a.merge(e[1], e[2])) ++res;
            if (e[0] == 2 && !b.merge(e[1], e[2])) ++res;
        }
        if (a.N > 1 || b.N > 1) return -1;  // 不连通
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

> [!NOTE] **[LeetCode 1970. 你能穿过矩阵的最后一天](https://leetcode-cn.com/problems/last-day-where-you-can-still-cross/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 显然二分即可
> 
>   直接申请 2e4 * 2e4 的静态数组显然会爆 换 vector 即可
> 
> - 并查集做法
> 
>   因为最后一天一定是全部都是水域，因此考虑倒序且无需预处理的往前推进并还原陆地

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 二分**

```cpp
class Solution {
public:
    const int INF = 0x3f3f3f3f;
    
    int n, m, mid;
    vector<vector<int>> g;
    vector<vector<bool>> st;
    
    int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0}, dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    
    bool dfs(int x, int y) {
        if (y == m)
            return true;
        st[x][y] = true;
        for (int i = 0; i < 8; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx > 0 && nx <= n && ny > 0 && ny <= m && g[nx][ny] <= mid && !st[nx][ny]) {
                if (dfs(nx, ny))
                    return true;
            }
        }
        return false;
    }
    
    bool check() {
        st = vector<vector<bool>>(n + 1, vector<bool>(m + 1, false));
        for (int i = 1; i <= n; ++ i )
            if (g[i][1] <= mid && dfs(i, 1))
                return false;
        return true;
    }
    
    int latestDayToCross(int row, int col, vector<vector<int>>& cells) {
        this->n = row; this->m = col;
        this->g = vector<vector<int>>(n + 1, vector<int>(m + 1, INF));
        int sz = cells.size();
        for (int i = 0; i < sz; ++ i )
            g[cells[i][0]][cells[i][1]] = i + 1;
        int l = 0, r = n * m;
        while (l < r) {
            mid = l + r >> 1;
            if (check())
                l = mid + 1;
            else
                r = mid;
        }
        return l - 1;
    }
};
```

##### **C++ 并查集**

```cpp
// 逆序并查集
class Solution {
public:
    int find(vector<int>& f, int x) {
        if (f[x] == x) { return x; }
        int nf = find(f, f[x]);
        f[x] = nf;
        return nf;
    }

    void merge(vector<int>& f, int x, int y) {
        int fx = find(f, x);
        int fy = find(f, y);
        f[fx] = fy;
    }

    int latestDayToCross(int row, int col, vector<vector<int>>& cells) {
        auto idx = [=](int i, int j) { return i * col + j; };
        auto check = [=](int i, int j) { return i >= 0 && i < row && j >= 0 && j < col; };
        int tot = (row * col);
        vector<int> f(tot + 2);
        for (int i = 0; i < tot + 2; i++) f[i] = i;

        // tot 和 tot + 1 分别代表上下界
        vector<vector<int>> state(row, vector<int>(col, 1));
        vector<pair<int, int>> directions{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int day = cells.size() - 1; day >= 0; day--) {
            int i = cells[day][0] - 1, j = cells[day][1] - 1;

            if (i == 0) { merge(f, tot, idx(0, j)); }
            if (i == row - 1) { merge(f, tot + 1, idx(row - 1, j)); }

            for (auto [di, dj] : directions) {
                int ni = i + di, nj = j + dj;
                if (check(ni, nj) && state[ni][nj] == 0)
                    merge(f, idx(i, j), idx(ni, nj));
            }
            state[i][j] = 0;

            if (find(f, tot) == find(f, tot + 1)) return day;
        }
        return 0;
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

### 综合应用

> [!NOTE] **[Luogu [NOI2015] 程序自动分析](https://www.luogu.com.cn/problem/P1955)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 并查集 + 离散化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e6 + 10;

struct Query {
    int x, y, e;
} qs[N];

int n, m;
int p[N];
unordered_map<int, int> S;

int get(int x) {
    if (S.count(x) == 0)
        S[x] = ++ n ;
    return S[x];
}

int find(int x) {
    if (p[x] != x)
        p[x] = find(p[x]);
    return p[x];
}

int main() {
    int t;
    cin >> t;
    while (t -- ) {
        n = 0;
        S.clear();
        
        cin >> m;
        for (int i = 0; i < m; ++ i ) {
            int x, y, e;
            cin >> x >> y >> e;
            qs[i] = {get(x), get(y), e};
        }
        
        for (int i = 1; i <= n; ++ i )
            p[i] = i;
        
        for (int i = 0; i < m; ++ i )
            if (qs[i].e == 1)
                p[find(qs[i].x)] = p[find(qs[i].y)];
        
        bool has_conflict = false;
        for (int i = 0; i < m; ++ i )
            if (qs[i].e == 0) {
                int pa = find(qs[i].x), pb = find(qs[i].y);
                if (pa == pb) {
                    has_conflict = true;
                    break;
                }
            }
        cout << (has_conflict ? "NO" : "YES") << endl;
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

> [!NOTE] **[LeetCode 1202. 交换字符串中的元素](https://leetcode-cn.com/problems/smallest-string-with-swaps/)** [TAG]
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
    vector<int> p;
    int find(int x) {
        if (p[x] != x) return p[x] = find(p[x]);
        return p[x];
    }
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
        int n = s.size();
        p.resize(n);
        for (int i = 0; i < n; ++ i ) p[i] = i;
        for (auto e : pairs) p[find(e[0])] = find(e[1]);
        
        vector<vector<char>> v(n);
        for (int i = 0; i < n; ++ i )
            v[find(i)].push_back(s[i]);
        for (int i = 0; i < n; ++ i ) {
            sort(v[i].begin(), v[i].end());
            reverse(v[i].begin(), v[i].end());
        }
        
        string res;
        for (int i = 0; i < n; ++ i ) {
            res.push_back(v[find(i)].back());
            v[find(i)].pop_back();
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

> [!NOTE] **[LeetCode 1722. 执行交换操作后的最小汉明距离](https://leetcode-cn.com/problems/minimize-hamming-distance-after-swap-operations/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 前几天的每日一题类似 [Weekly-155-C](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2019-09-22_Weekly-155)
> 
> 并查集即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    
    vector<int> p;
    void init() {
        p = vector<int>(n);
        for (int i = 0; i < n; ++ i ) p[i] = i;
    }
    int find(int x) {
        if (p[x] != x) return p[x] = find(p[x]);
        return p[x];
    }
    
    int minimumHammingDistance(vector<int>& sr, vector<int>& tr, vector<vector<int>>& as) {
        n = sr.size();
        
        init();
        
        for (auto & e : as) 
            p[find(e[0])] = find(e[1]);
        
        vector<vector<int>> v1(n), v2(n);
        for (int i = 0; i < n; ++ i ) {
            v1[find(i)].push_back(sr[i]);
            v2[find(i)].push_back(tr[i]);
        }
        
        int cnt = 0;
        for (int i = 0; i < n; ++ i )
            if (find(i) == i) {
                sort(v1[i].begin(), v1[i].end());
                sort(v2[i].begin(), v2[i].end());
                int sz = v1[i].size();
                for (int j = 0, k = 0; j < sz && k < sz; ) {
                    if (v1[i][j] == v2[i][k]) {
                        ++ cnt;
                        ++ j, ++ k;
                    } else if (v1[i][j] < v2[i][k]) ++ j;
                    else ++ k;
                }
            }
        return n - cnt;
    }
};
```

##### **C++ STL**

```cpp
class Solution {
public:
    vector<int> p;

    int find(int x) {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    int minimumHammingDistance(vector<int>& a, vector<int>& b, vector<vector<int>>& as) {
        int n = a.size();
        for (int i = 0; i < n; i ++ ) p.push_back(i);
        for (auto& t: as) p[find(t[0])] = find(t[1]);

        vector<unordered_multiset<int>> hash(n);
        for (int i = 0; i < n; i ++ )
            hash[find(i)].insert(a[i]);
        int res = 0;
        for (int i = 0; i < n; i ++ ) {
            auto& h = hash[find(i)];
            if (h.count(b[i])) h.erase(h.find(b[i]));
            else res ++ ;
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

> [!NOTE] **[LeetCode 1998. 数组的最大公因数排序](https://leetcode-cn.com/problems/gcd-sort-of-an-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 很好的综合练习题，考验熟练度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 100010;
    
    // 并查集
    int p[N];
    int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }
    void merge(int a, int b) {
        int pa = find(a), pb = find(b);
        if (pa != pb)
            p[pa] = pb;
    }
    
    // 线性筛
    int primes[N], cnt;
    bool st[N];
    
    void get_primes() {
        for (int i = 2; i < N; ++ i ) {
            if (!st[i])
                primes[cnt ++ ] = i;
            for (int j = 0; primes[j] <= (N - 1) / i; ++ j ) {
                st[primes[j] * i] = true;
                if (i % primes[j] == 0)
                    break;
            }
        }
    }
    
    // init
    void init() {
        for (int i = 0; i < N; ++ i )
            p[i] = i;
        
        memset(st, 0, sizeof st);
        cnt = 0;
        get_primes();
    }
    
    bool gcdSort(vector<int>& nums) {
        init();
        
        int n = nums.size();
        auto sorted_nums = nums;
        sort(sorted_nums.begin(), sorted_nums.end());
        
        for (int i = 0; i < n; ++ i ) {
            int x = nums[i];
            for (int j = 0; j < cnt && primes[j] <= x / primes[j]; ++ j ) {
                int y = primes[j];
                if (x % y == 0) {
                    merge(nums[i], y);
                    while (x % y == 0)
                        x /= y;
                }
            }
            if (x > 1)
                merge(nums[i], x);
        }
        
        // trick 直接与排序后的数组比对
        for (int i = 0; i < n; ++ i )
            if (find(nums[i]) != find(sorted_nums[i]))
                return false;
        return true;
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

### 并查集的抽象进阶

> [!NOTE] **[AcWing 1417. 塑造区域](https://www.acwing.com/problem/content/1419/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **经典 并查集**
> 
> 先后染色统计问题 + 扫描线
> 
> 分解之后 求每个区域不同颜色各自的面积和
> 
> 有一个超经典的题 【疯狂的馒头】
> 
> 想当然的会有线段树做法 但复杂度过高 有巧妙的 并查集做法
> 
> 并查集经典应用
> 
> **理解透彻**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

#define x first
#define y second

using PII = pair<int, int>;
const int N = 1010, M = 25010;

int A, B, n;
struct Rect {
    int c;
    PII a, b;
}rect[N];
int ans[M];
int p[10010];

// 找到 x 开始向右至 n 第一个未染色的节点
int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int draw(int a, int b) {
    int res = 0;
    for (int i = find(a); i <= b; i = find(i)) {
        p[i] = i + 1;
        res ++ ;
    }
    return res;
}

void get_range(int a, int b) {
    for (int i = 1; i <= B + 1; ++ i ) p[i] = i;
    // 逆序 方便使用并查集draw的思路
    for (int i = n - 1; i >= 0; -- i ) {
        auto & r = rect[i];
        if (r.a.x <= a && r.b.x >= b)
            ans[r.c] += draw(r.a.y + 1, r.b.y) * (b - a);   // +1
    }
    ans[1] += draw(1, B) * (b - a);
}

int main() {
    cin >> A >> B >> n;
    vector<int> xs;
    for (int i = 0; i < n; ++ i ) {
        int x1, y1, x2, y2, c;
        cin >> x1 >> y1 >> x2 >> y2 >> c;
        rect[i] = {c, {x1, y1}, {x2, y2}};
        xs.push_back(x1), xs.push_back(x2);
    }
    xs.push_back(0), xs.push_back(A);
    
    sort(xs.begin(), xs.end());
    for (int i = 0; i + 1 < xs.size(); ++ i )
        if (xs[i] != xs[i + 1])
            get_range(xs[i], xs[i + 1]);
    
    for (int i = 1; i < M; ++ i )
        if (ans[i])
            cout << i << ' ' << ans[i] << endl;
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
