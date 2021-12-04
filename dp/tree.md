树形 DP，即在树上进行的 DP。由于树固有的递归性质，树形 DP 一般都是递归进行的。

## 基础

> [!NOTE] **例题 [洛谷 P1352 没有上司的舞会](https://www.luogu.com.cn/problem/P1352)**
> 
> 某大学有 $n$ 个职员，编号为 $1 \sim N$。他们之间有从属关系，也就是说他们的关系就像一棵以校长为根的树，父结点就是子结点的直接上司。现在有个周年庆宴会，宴会每邀请来一个职员都会增加一定的快乐指数 $a_i$，但是呢，如果某个职员的上司来参加舞会了，那么这个职员就无论如何也不肯来参加舞会了。所以，请你编程计算，邀请哪些职员可以使快乐指数最大，求最大的快乐指数。

我们可以定义 $f(i,0/1)$ 代表以 $i$ 为根的子树的最优解（第二维的值为 0 代表 $i$ 不参加舞会的情况，1 代表 $i$ 参加舞会的情况）。

显然，我们可以推出下面两个状态转移方程（其中下面的 $x$ 都是 $i$ 的儿子）：

- $f(i,0) = \sum\max \{f(x,1),f(x,0)\}$（上司不参加舞会时，下属可以参加，也可以不参加）
- $f(i,1) = \sum{f(x,0)} + a_i$（上司参加舞会时，下属都不会参加）

我们可以通过 DFS，在返回上一层时更新当前结点的最优解。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 6010;

int n;
int h[N], e[N], ne[N], idx;
int happy[N];
int f[N][2];
bool has_fa[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u) {
    f[u][1] = happy[u];

    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        dfs(j);

        f[u][1] += f[j][0];
        f[u][0] += max(f[j][0], f[j][1]);
    }
}

int main() {
    scanf("%d", &n);

    for (int i = 1; i <= n; i ++ ) scanf("%d", &happy[i]);

    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i ++ ) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(b, a);
        has_fa[a] = true;
    }

    int root = 1;
    while (has_fa[root]) root ++ ;

    dfs(root);

    printf("%d\n", max(f[root][0], f[root][1]));

    return 0;
}
```

##### **Python**

```python
# 树是特殊的图，所以树用邻接表来存储（链式前向星）
# 状态表示（有两个状态）：f[u][0] 从以u为根的所有子树中选择，并且不能选u的方案数；属性：最大值
#           f[u][1] 从以u为根的所有子树中选择，且选u的方案数；属性：最大值
# 状态计算：f[u][0] = max(f[si][0], f[si][1])  （si是u的子树们，每个子树都是独立的）
#           f[u][1] = max(f[si][0])
# 求树形dp的时候，是需要从根节点往下求。
# 这道题对python不友好，用dfs会爆栈，所以需要用下面的语句来限制，可以尝试用BFS来维护一个队列

import sys

limit = 10000
sys.setrecursionlimit(limit)


def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def dfs(u):
    f[u][1] = happy[u]  # 如果选择当前节点
    i = h[u]  # 开始遍历他的子树
    while i != -1:
        j = ev[i]
        dfs(j)
        f[u][0] += max(f[j][0], f[j][1])
        f[u][1] += f[j][0]
        i = ne[i]


if __name__ == '__main__':
    N = 6010
    h = [-1] * N
    ev = [0] * N
    ne = [0] * N
    idx = 0
    happy = [0]
    has_father = [False] * N  # 用来确定根结点，dfs需要从根节点往下走

    f = [[0, 0] for i in range(N)]  # 每一个点 都有两个状态表示
    n = int(input())
    for _ in range(n):
        happy.append(int(input()))
    for i in range(1, n):
        a, b = map(int, input().split())
        add_edge(b, a)
        has_father[a] = True
    root = 1
    while has_father[root]:
        root += 1
    dfs(root)
    print(max(f[root][0], f[root][1]))

```

<!-- tabs:end -->
</details>

<br>

* * *

## 树上背包

树上的背包问题，简单来说就是背包问题与树形 DP 的结合。

> [!NOTE] **例题 [洛谷 P2014 CTSC1997 选课](https://www.luogu.com.cn/problem/P2014)**
> 
> 现在有 $n$ 门课程，第 $i$ 门课程的学分为 $a_i$，每门课程有零门或一门先修课，有先修课的课程需要先学完其先修课，才能学习该课程。
> 
> 一位学生要学习 $m$ 门课程，求其能获得的最多学分数。
> 
> $n,m \leq 300$

每门课最多只有一门先修课的特点，与有根树中一个点最多只有一个父亲结点的特点类似。

因此可以想到根据这一性质建树，从而所有课程组成了一个森林的结构。为了方便起见，我们可以新增一门 $0$ 学分的课程（设这个课程的编号为 $0$），作为所有无先修课课程的先修课，这样我们就将森林变成了一棵以 $0$ 号课程为根的树。

我们设 $f(u,i,j)$ 表示以 $u$ 号点为根的子树中，已经遍历了 $u$ 号点的前 $i$ 棵子树，选了 $j$ 门课程的最大学分。

转移的过程结合了树形 DP 和背包 DP 的特点，我们枚举 $u$ 点的每个子结点 $v$，同时枚举以 $v$ 为根的子树选了几门课程，将子树的结果合并到 $u$ 上。

记点 $x$ 的儿子个数为 $s_x$，以 $x$ 为根的子树大小为 $\textit{siz_x}$，很容易写出下面的转移方程：

$$
f(u,i,j)=\max_{v,k \leq j,k \leq \textit{siz_v}} f(u,i-1,j-k)+f(v,s_v,k)
$$

注意上面转移方程中的几个限制条件，这些限制条件确保了一些无意义的状态不会被访问到。

$f$ 的第二维可以很轻松地用滚动数组的方式省略掉，注意这时需要倒序枚举 $j$ 的值。

我们可以证明，该做法的时间复杂度为 $O(nm)$。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> v, w;
vector<vector<int>> f, es;

int N, V;

void dfs(int x) {
    // 点 x 必选 故初始化包含 x 的价值
    for (int i = v[x]; i <= V; ++i) f[x][i] = w[x];
    for (auto& y : es[x]) {
        dfs(y);
        // j 范围   小于 v[x] 无法选择节点x
        for (int j = V; j >= v[x]; --j)
            // 分给子树y的空间不能大于 j-v[x] 否则无法选择物品x
            for (int k = 0; k <= j - v[x]; ++k)
                f[x][j] = max(f[x][j], f[x][j - k] + f[y][k]);
    }
}

int main() {
    int p, root;
    cin >> N >> V;
    v.resize(N + 1);
    w.resize(N + 1);
    f.resize(N + 1, vector<int>(V + 1));
    es.resize(N + 1);

    for (int i = 1; i <= N; ++i) {
        cin >> v[i] >> w[i] >> p;
        if (p == -1)
            root = i;
        else
            es[p].push_back(i);
    }
    dfs(root);
    cout << f[root][V] << endl;
}
```

##### **Python**

```python
# 用递归的思路考虑，框架是树形dp的模板
# 状态表示：f[u,j]:表示所有以u为根的子树中选，体积不超过j的方案；属性：Max
# 状态转移：用新的划分方式：以体积来进行划分!!!（从 每个子树选or不选，优化成 体积为0-m的划分）k个子树，选or不选，时间复杂度2**k 
# 每棵子树都分为0-m种情况，分别表示：体积是0,...m的价值是多少（一共有m+1类，求每一类里取最大价值就可以）
# 这道题转化为了：一共有n颗子树，每个子树内 可以选择用多大的体积，问总体积不超过j的最大价值是多少。
# 总结：可以把每个子树看成一个物品组，每个组内部有m+1个物品，第0个物品表示体积是0，价值f[0],第m个类表示体积是m,价值是f[m]。每个物品组只能选一个出来。
# dp分析精髓： 用某一个数表示一类；这里就是用不同的体积表示一大类，

# 分组背包问题：一定要记住：先循环组数，再循环体积，再循环决策（选某个组的哪个物品）

# 如果会爆栈的话，可以加入以下代码段
# import sys
# limit = 10000
# sys.setrecursionlimit(limit)

N = 110
h = [-1] * N
ev = [0] * N
ne = [0] * N
idx = 0
v = [0] * N
w = [0] * N
f = [[0] * N for _ in range(N)]


def dfs1(u):
    global m
    i = h[u]
    while i != -1:  # 1 go thru every sub-tree
        son = ev[i]
        dfs(ev[i])
        # group knapsack template
        # 1 travel group == every sub-tree
        # 2 travel volumn of knapsack
        for j in range(m - v[u], -1, -1):  # 必须要加上根节点,所有为根节点留出空间
            # 3 travel options 枚举当前用到的背包容量
            for k in range(0, j + 1):
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])
        i = ne[i]
    # 将当前点root加进来
    for i in range(m, v[u] - 1, -1):
        f[u][i] = f[u][i - v[u]] + w[u]
    # 当容量小于根节点v[root],则必然放不下，最大值都是0
    for i in range(v[u]):
        f[u][i] = 0


def dfs(u):
    global m  # 递归时 m需要随着变化，所以需要加global关键字
    for i in range(v[u], m + 1):  # 点u必须选，才能选子节点，所以初始化f[u][v[u]~m] = w[u]
        f[u][i] = w[u]
    i = h[u]
    while i != -1:
        son = ev[i]
        dfs(ev[i])

        for j in range(m, v[u] - 1, -1):  # j的范围为v[x]~m, 小于v[x]无法选择以x为子树的物品
            for k in range(0, j - v[u] + 1):  # 分给子树y的空间不能大于j-v[x],不然都无法选根物品x
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])
        i = ne[i]  # 踩坑：一定要记得写啊！！！


def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


if __name__ == '__main__':
    n, m = map(int, input().split())
    root = 0
    for i in range(1, n + 1):
        v[i], w[i], p = map(int, input().split())
        if p == -1:
            root = i
        else:
            add_edge(p, i)
    dfs(root)
    print(f[root][m])  # f[i][m] 前i个物品，体积<=m，的最大价值
```

<!-- tabs:end -->
</details>

<br>

## 换根 DP

树形 DP 中的换根 DP 问题又被称为二次扫描，通常不会指定根结点，并且根结点的变化会对一些值，例如子结点深度和、点权和等产生影响。

通常需要两次 DFS，第一次 DFS 预处理诸如深度，点权和之类的信息，在第二次 DFS 开始运行换根动态规划。

接下来以一些例题来带大家熟悉这个内容。

> [!NOTE] **例题 [[POI2008]STA-Station](https://www.luogu.com.cn/problem/P3478)**
> 
> 给定一个 $n$ 个点的树，请求出一个结点，使得以这个结点为根时，所有结点的深度之和最大。

不妨令 $u$ 为当前结点，$v$ 为当前结点的子结点。首先需要用 $s_i$ 来表示以 $i$ 为根的子树中的结点个数，并且有 $s_u=1+\sum s_v$。显然需要一次 DFS 来计算所有的 $s_i$，这次的 DFS 就是预处理，我们得到了以某个结点为根时其子树中的结点总数。

考虑状态转移，这里就是体现＂换根＂的地方了。令 $f_u$ 为以 $u$ 为根时，所有结点的深度之和。

$f_v\leftarrow f_u$ 可以体现换根，即以 $u$ 为根转移到以 $v$ 为根。显然在换根的转移过程中，以 $v$ 为根或以 $u$ 为根会导致其子树中的结点的深度产生改变。具体表现为：

- 所有在 $v$ 的子树上的结点深度都减少了一，那么总深度和就减少了 $s_v$；

- 所有不在 $v$ 的子树上的结点深度都增加了一，那么总深度和就增加了 $n-s_v$；

根据这两个条件就可以推出状态转移方程 $f_v = f_u - s_v + n - s_v=f_u + n - 2 \times s_v$。

于是在第二次 DFS 遍历整棵树并状态转移 $f_v=f_u + n - 2 \times s_v$，那么就能求出以每个结点为根时的深度和了。最后只需要遍历一次所有根结点深度和就可以求出答案。

TODO@binacs

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

### 基础

- [HDU 2196 Computer](http://acm.hdu.edu.cn/showproblem.php?pid=2196)

- [POJ 1463 Strategic game](http://poj.org/problem?id=1463)

- [\[POI2014\]FAR-FarmCraft](https://www.luogu.com.cn/problem/P3574)

### 树上DP

- [「CTSC1997」选课](https://www.luogu.com.cn/problem/P2014)

- [「JSOI2018」潜入行动](https://loj.ac/problem/2546)

- [「SDOI2017」苹果树](https://loj.ac/problem/2268)

### 换根

- [POJ 3585 Accumulation Degree](http://poj.org/problem?id=3585)

- [\[POI2008\]STA-Station](https://www.luogu.com.cn/problem/P3478)

- [\[USACO10MAR\]Great Cow Gathering G](https://www.luogu.com.cn/problem/P2986)

- [CodeForce 708C Centroids](http://codeforces.com/problemset/problem/708/C)
