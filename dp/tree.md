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

记点 $x$ 的儿子个数为 $s_x$ ，以 $x$ 为根的子树大小为 $siz_x$ ，很容易写出下面的转移方程：

$$
f(u,i,j)=\max_{v,k \leq j,k \leq siz_v} f(u,i-1,j-k)+f(v,s_v,k)
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


> [!NOTE] **[AcWing 1072. 树的最长路径](https://www.acwing.com/problem/content/description/1074/)**
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
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 10010, M = N * 2;

int n;
int h[N], e[M], w[M], ne[M], idx;
int ans;

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int dfs(int u, int father) {
    int dist = 0;  // 表示从当前点往下走的最大长度
    int d1 = 0, d2 = 0;

    for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        if (j == father) continue;
        int d = dfs(j, u) + w[i];
        dist = max(dist, d);

        if (d >= d1)
            d2 = d1, d1 = d;
        else if (d > d2)
            d2 = d;
    }

    ans = max(ans, d1 + d2);

    return dist;
}

int main() {
    cin >> n;

    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i++) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }

    dfs(1, -1);

    cout << ans << endl;

    return 0;
}
```

##### **Python**

```python
# 也被称为 一般情况下的 树的直径；
# （还有一种树的直径是：没有边权的 树的直径：1. 任取一点作为起点，找到距离改点最远的一个点u （DFS，BFS） 2. 再找到距离u最远的距离v.那么u和v之间的路径就是一条直径（DFS，BFS））
# 证明：图论的证明：第一步中找到的u一定是某一条直径的一个端点。反证法：如果A的找到的u不是某条直径的端点的话，找到另外一条直径，分情况讨论：1 该直径和Au相交；2. 该直径和Au不相交
# 扩展后的做法：就是每条边都有自己的边权，不再都是1；那就需要用树形dp的方法：想办法把所有的路径枚举一遍，在其中找到边权最大的路径即可。
# 任意选一个点为根节点，然后把所有直径进行分类：在每条直径上 找到高度最高的点，把这条直径的值放到这个高度最高的点上。
# 以每个点把所有直径进行分类，每个点代表的类是：所有路径最高的点 就是这个点的所有路径（每条路径上都有一个唯一的最高点）
# 问题转移为：当固定完一个点后，如何求挂到这个点上的长度的最大值呢？
# 首先 先求出这个点上 所有子节点往下走的最大长度，1）挂在这个点下的最大值 2）经过这个点然后找到最长 和 第二长的 长度 相加 就是经过这个点的最大值

# 状态表示：（集合）：
# 状态转移：

N = 10010
M = 2 * N
h = [-1] * N
ne = [0] * M
ev = [0] * M
w = [0] * M
idx = 0


def dfs(u, father):  # father表示当前点的father节点，避免子树往上走（子树方向只能向下走）
    global ans
    dist = 0  # 表示 从当前点往下走的最大长度
    d1, d2 = 0, 0  # 如果是负数，就是不存在，那就是0

    i = h[u]  # 开始遍历当前点的所有子节点
    while i != -1:
        j = ev[i]
        if j == father:
            i = ne[i]  # 踩坑：不要忘了 把i往后移动！
            continue  # 不能从子节点 走到 father节点上去
        d = dfs(j, u) + w[i]  # 返回值 就是j点 往下走的最大长度
        dist = max(dist, d)

        if d >= d1:
            d2 = d1
            d1 = d
        elif d > d2:
            d2 = d
        i = ne[i]
    ans = max(ans, d1 + d2)
    return dist


def add_edge(a, b, c):
    global idx
    ev[idx] = b
    w[idx] = c
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


if __name__ == '__main__':
    n = int(input())
    for _ in range(n - 1):
        a, b, c = map(int, input().split())
        add_edge(a, b, c)
        add_edge(b, a, c)
    ans = 0
    dfs(1, -1)  # 任取一个点作为起点（其实就是作为根节点），根节点没有father节点，所以传入-1即可
    print(ans)
```

<!-- tabs:end -->
</details>

<br>

* * *


> [!NOTE] **[AcWing 1075. 数字转换](https://www.acwing.com/problem/content/description/1077/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 相当于求树中的最长路径
> 
> 问题在于如何快速求出 5e4范围内的数的约数和 ====> 筛法
> 
> 对于数据规模较大的情况 要想到筛法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 50010, M = N;

int n;
int h[N], e[M], w[M], ne[M], idx;
int sum[N];
bool st[N];
int ans;

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int dfs(int u) {
    st[u] = true;

    int dist = 0;
    int d1 = 0, d2 = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!st[j]) {
            int d = dfs(j);
            dist = max(dist, d);
            if (d >= d1)
                d2 = d1, d1 = d;
            else if (d > d2)
                d2 = d;
        }
    }

    ans = max(ans, d1 + d2);

    return dist + 1;
}

int main() {
    cin >> n;
    memset(h, -1, sizeof h);

    for (int i = 1; i <= n; i++)
        for (int j = 2; j <= n / i; j++) sum[i * j] += i;

    for (int i = 2; i <= n; i++)
        if (sum[i] < i) add(sum[i], i);

    for (int i = 1; i <= n; i++)
        if (!st[i]) dfs(i);

    cout << ans << endl;

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

### 树上DP

- [「CTSC1997」选课](https://www.luogu.com.cn/problem/P2014)

- [「JSOI2018」潜入行动](https://loj.ac/problem/2546)

- [「SDOI2017」苹果树](https://loj.ac/problem/2268)


> [!NOTE] **[AcWing 1074. 二叉苹果树](https://www.acwing.com/problem/content/description/1076/)**
> 
> 题意: 必须从根节点开始 一个路径

> [!TIP] **思路**
> 
> 分组背包 每个边的值当做从父节点到该点的值
> 
> 标准树dp写法
> 
> 先递归子树，再更新当前

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 110, M = N * 2;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
int f[N][N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

void dfs(int u, int father) {
    for (int i = h[u]; ~i; i = ne[i]) {
        if (e[i] == father) continue;
        dfs(e[i], u);
        for (int j = m; j; j--)
            for (int k = 0; k + 1 <= j; k++)
                f[u][j] = max(f[u][j], f[u][j - k - 1] + f[e[i]][k] + w[i]);
    }
}

int main() {
    cin >> n >> m;
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i++) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c), add(b, a, c);
    }

    dfs(1, -1);

    printf("%d\n", f[1][m]);

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


> [!NOTE] **[Luogu [ZJOI2006]三色二叉树](https://www.luogu.com.cn/problem/P2585)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典树dp流程
> 
> 细节处理起来有点麻烦

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
const int N = 500050;
char s[N];
int f[N][3], g[N][3], cnt;

int res = 1;

void dfs(int x) {
    if (s[x] == '0') {  //叶节点
        g[x][0] = f[x][0] = 1;
        return;
    }
    dfs(++cnt);
    // 左儿子编号为x+1
    if (s[x] == '1') {  //一个儿子
        f[x][0] = max(f[x + 1][1], f[x + 1][2]) + 1;
        f[x][1] = max(f[x + 1][0], f[x + 1][2]);
        f[x][2] = max(f[x + 1][0], f[x + 1][1]);

        // 上方代码完全是复制一遍到下面
        g[x][0] = min(g[x + 1][1], g[x + 1][2]) + 1;
        g[x][1] = min(g[x + 1][0], g[x + 1][2]);
        g[x][2] = min(g[x + 1][0], g[x + 1][1]);
    } else {
        // 右儿子编号为k
        int k = ++cnt;
        dfs(k);
        f[x][0] = max(f[x + 1][1] + f[k][2], f[x + 1][2] + f[k][1]) + 1;
        f[x][1] = max(f[x + 1][0] + f[k][2], f[x + 1][2] + f[k][0]);
        f[x][2] = max(f[x + 1][0] + f[k][1], f[x + 1][1] + f[k][0]);

        g[x][0] = min(g[x + 1][1] + g[k][2], g[x + 1][2] + g[k][1]) + 1;
        g[x][1] = min(g[x + 1][0] + g[k][2], g[x + 1][2] + g[k][0]);
        g[x][2] = min(g[x + 1][0] + g[k][1], g[x + 1][1] + g[k][0]);
    }
    res = max(res, f[x][0]);
}
int main() {
    scanf("%s", s + 1);
    dfs(++cnt);
    cout << res << ' ' << min(g[1][0], min(g[1][1], g[1][2])) << endl;

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

> [!NOTE] **[Luogu 有线电视网](https://www.luogu.com.cn/problem/P1273)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典的树形分组dp变形
> 
> 【状态设计 转移】
> 
> 反复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 显然树上分组背包 考虑如何进行状态设计
//
// f[i][j] 表示以i为根满足j个客户需求的收入
// 不必考虑当前收入是否为负 只要最后取正值即可 ==> 思考
//
// 【核心在于定义和分组时遍历的实现】

const int N = 3010, M = N;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
int in[N];
int f[N][N], sz[N];

void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dp(int u) {
    // user
    if (u > n - m) {
        f[u][1] = in[u];
        sz[u] = 1;
        return;
    }
    
    sz[u] = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int son = e[i];
        dp(son);

        sz[u] += sz[son];
        // 中继本身不占用名额 故 j >= 0
        for (int j = sz[u]; j >= 0; -- j )
            // 分给多余sz[son]的空间也没用 所以 k <= sz[son]
            for (int k = 0; k <= sz[son]; ++ k )
                if (j - k >= 0)
                    f[u][j] = max(f[u][j], f[u][j - k] + f[son][k] - w[i]);
    }
}

int main() {
    init();
    cin >> n >> m;

    for (int i = 1; i <= n - m; ++ i ) {
        int k, a, c;
        cin >> k;
        while (k -- ) {
            cin >> a >> c;
            add(i, a, c);
        }
    }
    
    for (int i = n - m + 1; i <= n; ++ i )
        cin >> in[i];

    memset(f, 0xcf, sizeof f);  // -INF
    for (int i = 1; i <= n; ++ i )
        f[i][0] = 0;
    dp(1);
    for (int i = m; i >= 1; -- i )
        if (f[1][i] >= 0) {
            cout << i << endl;
            break;
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

### 换根

- [POJ 3585 Accumulation Degree](http://poj.org/problem?id=3585)

- [\[POI2008\]STA-Station](https://www.luogu.com.cn/problem/P3478)

- [\[USACO10MAR\]Great Cow Gathering G](https://www.luogu.com.cn/problem/P2986)

- [CodeForce 708C Centroids](http://codeforces.com/problemset/problem/708/C)

> [!NOTE] **[AcWing 1073. 树的中心](https://www.acwing.com/problem/content/description/1075/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 两次树dp
> 
> 1. 根据子树更新当前节点
> 
> 2. 根据当前节点更新当前并递归更新子树
> 
> 记录最长和次长的思想

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 10010, M = N * 2, INF = 0x3f3f3f3f;

int n;
int h[N], e[M], w[M], ne[M], idx;
int d1[N], d2[N], p1[N], up[N];
bool is_leaf[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int dfs_d(int u, int father) {
    d1[u] = d2[u] = -INF;   // 考虑负权边
    for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        if (j == father) continue;
        int d = dfs_d(j, u) + w[i];
        if (d >= d1[u]) {
            d2[u] = d1[u], d1[u] = d;
            p1[u] = j;
        } else if (d > d2[u])
            d2[u] = d;
    }

    if (d1[u] == -INF) {
        d1[u] = d2[u] = 0;
        is_leaf[u] = true;
    }

    return d1[u];
}

void dfs_u(int u, int father) {
    for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        if (j == father) continue;

        if (p1[u] == j)
            up[j] = max(up[u], d2[u]) + w[i];
        else
            up[j] = max(up[u], d1[u]) + w[i];

        dfs_u(j, u);
    }
}

int main() {
    cin >> n;
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i++) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }

    dfs_d(1, -1);
    dfs_u(1, -1);

    int res = d1[1];
    for (int i = 2; i <= n; i++)
        if (is_leaf[i])
            res = min(res, up[i]);
        else
            res = min(res, max(d1[i], up[i]));

    printf("%d\n", res);

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

> [!NOTE] **[Luogu 医院设置](https://www.luogu.com.cn/problem/P1364)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典换根

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 110;

int n, s;
int w[N];
int sz[N], sum[N], up[N];
int h[N], e[N], ne[N], idx;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs_d(int u, int fa) {
    sz[u] = w[u], sum[u] = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs_d(j, u);
        sz[u] += sz[j];
        sum[u] += sum[j] + sz[j];
    }
}

void dfs_u(int u, int fa) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        // 上面部分的距离
        // up[u] + (s - sz[j]) - sz[j]
        //              up             + w[u] +   [sum of other son]    +  [size of other son]
        // up[j] = up[u] + (s - sz[u]) + w[u] + sum[u] - sum[j] - sz[j] + sz[u] - w[u] - sz[j];
        up[j] = up[u] + s + sum[u] - sum[j] - 2 * sz[j];
        dfs_u(j, u);
    }
}

int main() {
    s = idx = 0;
    memset(h, -1, sizeof h);
    
    cin >> n;
    for (int i = 1; i <= n; ++ i ) {
        int l, r;
        cin >> w[i] >> l >> r;
        if (l)
            add(i, l);
        if (r)
            add(i, r);
        s += w[i];
    }
    
    dfs_d(1, -1);
    dfs_u(1, -1);

    // cout << "s = " << s << endl;
    // cout << "sz[1] = " << sz[1] << endl;


    // for (int i = 1; i <= n; ++ i )
    //     cout << i << " sz  = " << sz[i] << endl;
    // for (int i = 1; i <= n; ++ i )
    //     cout << i << " sum = " << sum[i] << endl;
    // for (int i = 1; i <= n; ++ i )
    //     cout << i << " up  = " << up[i] << endl;

    int res = 2e9;
    for (int i = 1; i <= n; ++ i )
        res = min(res, sum[i] + up[i]);
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

> [!NOTE] **[Luogu 会议](https://www.luogu.com.cn/problem/P1395)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 同【医院设置】但更简单
> 
> 因为本题只需计算节点个数即可 每个节点不会有多个权

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5e4 + 10, M = N << 1;

int n;
int h[N], e[M], w[M], ne[M], idx;
int down[N], up[N], sz[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs_d(int u, int fa) {
    down[u] = 0, sz[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa) {
            dfs_d(j, u);
            down[u] += down[j] + sz[j];
            sz[u] += sz[j];
        }
    }
}

void dfs_u(int u, int fa) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa) {
            // ATTENTION 容易漏掉
            // (down[u] - down[j] - sz[j]) 是 u 向其他子节点伸展的部分
            up[j] = up[u] + (down[u] - down[j] - sz[j]) + n - sz[j];
            dfs_u(j, u);
        }
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n;
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b, 1), add(b, a, 1);
    }
    
    dfs_d(1, -1);
    dfs_u(1, -1);   // up[1] = 0

    // for (int i = 1; i <= n; ++ i )
    //     cout << i << ' ' << down[i] << ' ' << up[i] << endl;
    
    int res = INT_MAX, p = 0;
    for (int i = 1; i <= n; ++ i )
        if (down[i] + up[i] < res)
            res = down[i] + up[i], p = i;
    cout << p << ' ' << res << endl;
    
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

> [!NOTE] **[LeetCode 310. 最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 枚举每个节点作为根节点的树高度
> 
> 本质就是 dp 求树中心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> g;
    vector<int> d1, d2, p1, up;
    void dfs_d(int u, int fa) {
        for (auto v : g[u]) if (v != fa) {
            dfs_d(v, u);
            int d = d1[v] + 1;
            if (d >= d1[u]) d2[u] = d1[u], d1[u] = d, p1[u] = v;
            else if (d > d2[u]) d2[u] = d;
        }
    }
    void dfs_u(int u, int fa) {
        for (auto v : g[u]) if (v != fa) {
            if (p1[u] == v) up[v] = max(up[u], d2[u]) + 1;
            else up[v] = max(up[u], d1[u]) + 1;
            dfs_u(v, u);
        }
    }
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        g.resize(n);
        d1 = d2 = p1 = up = vector<int>(n);
        for (auto & e : edges) g[e[0]].push_back(e[1]), g[e[1]].push_back(e[0]);

        dfs_d(0, -1);
        dfs_u(0, -1);

        int mind = n + 1;
        for (int i = 0; i < n; ++i) mind = min(mind, max(up[i], d1[i]));
        vector<int> res;
        for (int i = 0; i < n; ++i)
            if (max(up[i], d1[i]) == mind) res.push_back(i);
        return res;
    }
};
```

##### **C++ 基于拓扑**

很久以前的代码:

1. 从叶子结点开始，每一轮删除所有叶子结点。
2. 删除后，会出现新的叶子结点，此时再删除。
3. 重复以上过程直到剩余 1 个或 2 个结点，此时这 1 个或 2 个结点就是答案。

```cpp
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        vector<int> ans;
        vector<int> degree(n);  // 初始化度为0
        vector<vector<int>> G(n);
        queue<int> q;
        if (n == 1) {
            ans.push_back(0);
            return ans;
        }
        for (auto e : edges) {
            G[e[0]].push_back(e[1]);
            G[e[1]].push_back(e[0]);
            ++ degree[e[0]] ;
            ++ degree[e[1]] ;
        }
        for (int i = 0; i < n; ++ i ) 
            if (degree[i] == 1)
                q.push(i);
        int left = n;
        while (left > 2) {
            int len = q.size();
            left -= len;
            while (len -- ) {
                int u = q.front(); q.pop();
                for (auto v : G[u]) {
                    if (degree[v] > 0) -- degree[v] ;
                    if (degree[v] == 1) q.push(v);
                }
            }
        }
        while (!q.empty()) {
            ans.push_back(q.front());
            q.pop();
        }
        return ans;
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


### 进阶：树DP状态设计

> [!NOTE] **[AcWing 323. 战略游戏](https://www.acwing.com/problem/content/description/325/)**
> 
> 题意: 树上放，监控边

> [!TIP] **思路**
> 与leetcode看守题不同之处在于：这个是看边 leetcode是看节点
> 
> https://leetcode-cn.com/problems/binary-tree-cameras/
> 
> - f[i] 没有放置也没有被覆盖的不考虑
> - f[i][0] i节点没有放置 覆盖全部边
> - f[i][1] i节点放置 覆盖全部边

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 1510;

int n;
int h[N], e[N], ne[N], idx;
int f[N][2];
bool st[N];

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

void dfs(int u) {
    f[u][0] = 0, f[u][1] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        dfs(j);
        f[u][0] += f[j][1];
        f[u][1] += min(f[j][0], f[j][1]);
    }
}

int main() {
    while (cin >> n) {
        memset(h, -1, sizeof h);
        idx = 0;

        memset(st, 0, sizeof st);
        for (int i = 0; i < n; i++) {
            int id, cnt;
            scanf("%d:(%d)", &id, &cnt);
            while (cnt--) {
                int ver;
                cin >> ver;
                add(id, ver);
                st[ver] = true;
            }
        }

        int root = 0;
        while (st[root]) root++;
        dfs(root);

        printf("%d\n", min(f[root][0], f[root][1]));
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

> [!NOTE] **[AcWing 1077. 皇宫看守](https://www.acwing.com/problem/content/description/1079/)**
> 
> 题意: 树上放，监控点 思想

> [!TIP] **思路**
> 
> 
> - f[i][0]  // 第i个结点的父结点被选
> - f[i][1]  // 第i个结点有一个子节点被选
> - f[i][2]  // 第i个节点本身被选

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 1510;

int n;
int h[N], w[N], e[N], ne[N], idx;
int f[N][3];
bool st[N];

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

void dfs(int u) {
    f[u][2] = w[u];

    int sum = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        dfs(j);
        f[u][0] += min(f[j][1], f[j][2]);
        f[u][2] += min(min(f[j][0], f[j][1]), f[j][2]);
        sum += min(f[j][1], f[j][2]);
    }

    f[u][1] = 1e9;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        f[u][1] = min(f[u][1], sum - min(f[j][1], f[j][2]) + f[j][2]);
    }
}

int main() {
    cin >> n;

    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i++) {
        int id, cost, cnt;
        cin >> id >> cost >> cnt;
        w[id] = cost;
        while (cnt--) {
            int ver;
            cin >> ver;
            add(id, ver);
            st[ver] = true;
        }
    }

    int root = 1;
    while (st[root]) root++;

    dfs(root);

    cout << min(f[root][1], f[root][2]) << endl;

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