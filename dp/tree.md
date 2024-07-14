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

- 树是特殊的图，所以树用邻接表来存储（链式前向星）

- 求树形dp的时候，是需要从根节点向下递归求解。

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
# 这道题对python不友好，用dfs会爆栈，所以需要用下面的语句来限制，可以尝试用BFS来维护一个队列

import sys
limit = 10000
sys.setrecursionlimit(limit)

N = 6010
h = [-1] * N
ev = [0] * N
ne = [0] * N
idx = 0
happy = [0]
has_father = [False] * N
f = [[0, 0] for _ in range(N)]  

def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def dfs(u):
    f[u][1] = happy[u]  # 如果选择当前节点，开始遍历它的子树
    i = h[u]  
    while i != -1:
        j = ev[i]
        dfs(j)
        f[u][0] += max(f[j][0], f[j][1])
        f[u][1] += f[j][0]
        i = ne[i]

if __name__ == '__main__':
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
# 如果会爆栈的话，可以加入以下代码段
# import sys
# limit = 10000
# sys.setrecursionlimit(limit)

N = 110
h, ev, ne, idx = [-1] * N, [0] * N, [0] * N, 0
v, w = [0] * N, [0] * N
f = [[0] * N for _ in range(N)]


def dfs1(u):
    global m
    i = h[u]
    while i != -1:
        son = ev[i]
        dfs(ev[i])
        # 1 travel group 
        # 2 travel volumn of knapsack
        for j in range(m - v[u], -1, -1):  # 必须要加上根节点,所有为根节点留出空间
            # 3 travel options 
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
    global m  
    for i in range(v[u], m + 1):  #点u必须选才能选子节点，所以初始化f[u][v[u]~m] = w[u]
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
    print(f[root][m])  
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
> 每条边都有权值，也就是本题。
>
> **树形dp**：想办法把所有的路径枚举一遍，在其中找到边权最大的路径即可。
>
> 1. 状态表示：以 $u$ 为根节点的所有路径的集合；属性：最长路径，也就是以 $u$ 为根节点，求最大子节点路径和第二大子节点路径的最大和。
> 2. 状态转移：枚举 $u$ 的子节点 $j$, 返回最长路径。求得以 $u$ 为根节点的第一长子路径 $d1$, 和第二长子路径$d2$, $res = max(d1 + d2)$. 递归返回只返回第一长路径

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
N = 10010
M = 2 * N
h, ne, ev, w = [-1] * N, [0] * M, [0] * M, [0] * M
idx = 0

def dfs(u, father): #father表示当前点的father节点，避免子树往上（只能向下走，避免上下死循环） 
    global res
    d1, d2 = 0, 0  # 如果是负数，就是不存在，那就是0
    i = h[u]  # 开始遍历当前点的所有子节点
    while i != -1:
        j = ev[i]
        if j == father: # 不能从子节点走到 father节点上去
            i = ne[i]  
            continue  
        d = dfs(j, u) + w[i]  # 递归的返回值是j点往下走的最大长度
        if d >= d1:
            d2 = d1
            d1 = d
        elif d > d2:
            d2 = d
        i = ne[i]
    res = max(res, d1 + d2)
    return d1

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
    res = 0
    dfs(1, -1)  # 任取一个点作为作为根节点，作为根节点时，是没有father节点，所以传入-1即可
    print(res)
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
>
> 1. 题目定义如果x的约数之和y小于x,则x和y可以相互转换；x的约数是固定的，所以x的约数之和也是固定的
>
> 2. 如果x可以跟约数之和y相互转换，可看成从y连一条有向边指向x
>
> 3. y可以指向很多x，而x只能指向y,原因看第二点
>
> 4. 按照上述步骤处理完所有的x之后，会形成一片森林
>
> 5. 例如n==8时，形成两棵树：
>                a、1-->(2,3,5,7)
>                   3-->4; 7-->8
>                b、6
> 6. 形成森林后，枚举每颗树，就能求到ans

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
N = 50010
ssum = [0] * N
st = [False] * N  
h, e, ne = [-1] * N, [0] * N, [0] * N
idx = 0

def add(a, b):
    global idx
    e[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1

# 返回这个树从根节点到叶节点的最长路径
# 方法：记录树从根节点到叶节点的最长路径和次长路径
def dfs(u):
    global ans
    i = h[u]
    # 记录最长路径和次长路径
    d1 = d2 = 0
    while i != -1:
        j = e[i]
        d = dfs(j) + 1
        if d >= d1:
            d2 = d1
            d1 = d
        elif d > d2:
            d2 = d
        i = ne[i]
    ans = max(ans, d1 + d2)
    return d1

if __name__ == '__main__':
    n = int(input())
    for i in range(1, n + 1):
    # 因为定义的约数之和不包含x本身，所以在筛选时，倍数从2开始
        j = 2
        while j <= n // i:
            ssum[j * i] += i
            j += 1

    for i in range(2, n + 1):
        # 约数之和y小于x，加一条边,ssum[i]作为父节点
        if ssum[i] < i:
            add(ssum[i], i)
            st[i] = True

    # 利用每颗树的直径更新答案
    ans = 0
    # 遍历每棵树，求ans
    for i in range(1, n + 1):
        if not st[i]:
            dfs(i)
    print(ans)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces Eternal Victory](http://codeforces.com/problemset/problem/61/D)**
> 
> 题意: 
> 
> 一颗树，从 $1$ 起始游遍所有节点问最少总距离

> [!TIP] **思路**
> 
> 显然让最长的链直走一次即可，其他都是两次

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Eternal Victory
// Contest: Codeforces - Codeforces Beta Round #57 (Div. 2)
// URL: https://codeforces.com/problemset/problem/61/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10, M = 2e5 + 10;

int h[N], e[M], w[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int n;
LL d1[N], d2[N];  // 本题要求下d2可不要

void dfs(int u, int fa) {
    d1[u] = d2[u] = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs(j, u);
        LL t = d1[j] + w[i];
        if (t > d1[u])
            d2[u] = d1[u], d1[u] = t;
        else if (t > d2[u])
            d2[u] = t;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n;

    LL s = 0;
    for (int i = 0; i < n - 1; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
        s += c;
    }

    dfs(1, -1);

    // FIX: 如果是起点任意，当然可以这样
    // cout << s * 2ll - res << endl;
    // ATTENTION: 必须从 1 起始，那么有一个最长链走一次即可
    cout << s * 2ll - d1[1] << endl;

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
> 标准树dp写法：先递归子树，再更新当前
>
> -------------------------------------------------------------------------------------
>
> 完全二叉树， 有依赖的背包问题题型简化版
>
> 物品组：u 为根节点的整颗子树；体积（容量）：可以留下的树枝数量；价值：树枝上的苹果数量，边的权重。
>
> 1. 状态表示：$f[u][j]$，表示所有以 $u$ 为树根的子树中选，选 $j$ 条边的价值；属性：$max$
> 2. 状态转移：每一棵子树看成一组背包，若需要选择该子树, 则根节点 $u$ 到子树的边一定得用上，因此能用上的总边数（体积）一定减 $1$；总共可以选择 $j$ 条边时，当前子树分配的最大边树是 $j-1$, 对于任意一棵子树有：$f[u][j] = max(f[u][j], f[u][j - 1 - k] + f[son][k] + w[i])$
> 3. 注意，在写有依赖的背包问题的时候，一定要注意循环顺序！1. 先枚举物品组，2. 枚举体积，3.枚举决策。这是因为我们做了一个空间的优化，所以顺利不能错。



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

##### **Python-链式前向星**

```python
N = 110
M = N * 2
f = [[0] * N for _ in range(N)]
h, ev, ne, w = [-1] * N, [0] * M, [0] * M, [0] * M
idx = 0

def dfs(u, father):
    i = h[u]  # 先枚举物品组
    while i != -1:
        if ev[i] == father: 
            i = ne[i]
            continue
        dfs(ev[i], u)
        for j in range(m, -1, -1): # 枚举体积
            for k in range(j):  # 枚举决策，每组里选哪个物品
                f[u][j] = max(f[u][j], f[u][j - k - 1] + f[ev[i]][k] + w[i])
        i = ne[i] 
            
def add_edge(a, b, c):
    global idx
    ev[idx] = b 
    w[idx] = c
    ne[idx] = h[a]
    h[a] = idx
    idx += 1
    
if __name__ == '__main__':
    n, m = map(int, input().split())
    for _ in range(n - 1):
        a, b, c = map(int, input().split())
        add_edge(a, b, c)
        add_edge(b, a, c)
    dfs(1, -1)  #-1 表示没有父节点
    print(f[1][m])
```

##### **Python-字典**

```python
import collections

def dfs(u, fa):
    for v, w in g[u]:
        if v == fa:
            continue
        dfs(v, u)
        for j in range(q, 0, -1):
            k = 0
            while k + 1 <= j:
                f[u][j] = max(f[u][j], f[u][j - k - 1] + f[v][k] + w);
                k += 1

if __name__ == '__main__':
    n, q = map(int, input().split())
    g = collections.defaultdict(list)
    f = [[0] * (q + 1)  for _ in range(n + 1)]
    for i in range(n - 1):
        u, v, w = map(int, input().split())
        g[u].append([v, w])
        g[v].append([u, w])
    dfs(1, -1)
    print(f[1][q])
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

> [!NOTE] **[LeetCode 1372. 二叉树中的最长交错路径](https://leetcode.cn/problems/longest-zigzag-path-in-a-binary-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 只记录优雅写法
>
> 记录当前节点是左/右孩子，即记录当前路径的方向。搜索其孩子时，根据上一条路径方向判断。
>
> 如果当前路径方向相反，路径+1；如果路径相同，路径长度置为1

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxAns;
    /* 0 => left, 1 => right */
    void dfs(TreeNode* o, bool dir, int len) {
        maxAns = max(maxAns, len);
        if (!dir) {		// left		
            if (o->left) dfs(o->left, 1, len + 1);
            if (o->right) dfs(o->right, 0, 1);
        } else {
            if (o->right) dfs(o->right, 0, len + 1);
            if (o->left) dfs(o->left, 1, 1);
        }
    } 

    int longestZigZag(TreeNode* root) {
        if (!root) return 0;
        maxAns = 0;
        dfs(root, 0, 0);
        dfs(root, 1, 0);
        return maxAns;
    }
};
```

##### **Python**

```python
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.res = 0

        def dfs(u, dir, dis):
            if not u:return
            self.res = max(self.res, dis)
            # 当前节点是其父节点的右孩子
            if dir == 1:
                dfs(u.left, 0, dis + 1) # 搜索它的左孩子
                dfs(u.right, 1, 1)  # 搜索它的右孩子
            # 当前节点是其父节点的左孩子
            else:
                dfs(u.left, 0, 1)
                dfs(u.right, 1, dis +1)

        dfs(root.right, 1, 1)
        dfs(root.left, 0, 1) 
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1373. 二叉搜索子树的最大键值和](https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/)** [TAG]
> 
> 题意: 
> 
> 求满足二叉搜索树性质的最大节点和

> [!TIP] **思路**
> 
> 记录优雅写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    struct Node {
        int sum, lo, hi;
        Node(int sum, int lo, int hi) : sum(sum), lo(lo), hi(hi) {}
    };
    int res;
    Node dfs(TreeNode* root) {
        Node l = Node(0, root->val, INT_MIN), r = Node(0, INT_MAX, root->val);
        if (root->left) l = dfs(root->left);
        if (root->right) r = dfs(root->right);

        if (root->val > l.hi && root->val < r.lo) {
            res = max(res, root->val + l.sum + r.sum);
            return Node(l.sum + r.sum + root->val, l.lo, r.hi);
        }
        return Node(INT_MIN, INT_MIN, INT_MAX);
    }

    int maxSumBST(TreeNode* root) {
        res = 0;
        dfs(root);
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

> [!NOTE] **[LeetCode LCP 34. 二叉树染色](https://leetcode.cn/problems/er-cha-shu-ran-se-UGC/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准树 dp 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    using LL = long long;
    int k;
    LL res;
    
    vector<LL> dfs(TreeNode * root) {
        if (!root)
            return vector<LL>(k + 1, 0);
        
        auto l = dfs(root->left);
        auto r = dfs(root->right);
        
        // ret[0]为不选当前
        vector<LL> ret(k + 1, 0);
        
        ret[0] = l[k] + r[k];
        
        for (int i = 1; i <= k; ++ i )
            for (int j = 0; j < i; ++ j )
                ret[i] = max(ret[i], (LL)root->val + l[j] + r[i - j - 1]);
        for (int i = 1; i <= k; ++ i )
            ret[i] = max(ret[i - 1], ret[i]);
        
        res = max(res, ret[k]);
        
        return ret;
    }
    
    int maxValue(TreeNode* root, int k) {
        this->k = k, this->res = 0;
        dfs(root);
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

> [!NOTE] **[Codeforces C. Valera and Elections](https://codeforces.com/problemset/problem/369/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **从未见过的树形DP**
> 
> 思路 实现 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Valera and Elections
// Contest: Codeforces - Codeforces Round #216 (Div. 2)
// URL: http://codeforces.com/problemset/problem/369/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 思维题
// 显然是树形结构 考虑类似树形DP的办法检查子树中维修所有树需要统计多少个点
// 在遍历子树时更新 技巧 思路

const int N = 100010, M = 200010;

int n;
int h[N], e[M], w[M], ne[M], idx;
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int dfs(int u, int fa) {
    int ret = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa) {
            int t = dfs(j, u);
            // 该路有问题 且子树没有选取的点 则选取该点
            if (w[i] == 2 && t == 0) {
                t = 1;
                st[j] = true;
            }
            ret += t;
        }
    }
    return ret;
}

int main() {
    memset(h, -1, sizeof h);

    cin >> n;
    for (int i = 0; i < n - 1; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c);
        add(b, a, c);
    }

    int s = dfs(1, 1);
    cout << s << endl;
    for (int i = 1; i <= n; ++i)
        if (st[i])
            cout << i << ' ';
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

> [!NOTE] **[Codeforces D. Valid Sets](https://codeforces.com/problemset/problem/486/D)**
> 
> 题意: 
> 
> 给出一棵树，树上有点权，求这棵树的满足最大点权与最小点权之差小于 d 的连通子图的个数。

> [!TIP] **思路**
> 
> 思维题 树形dp 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Valid Sets
// Contest: Codeforces - Codeforces Round #277 (Div. 2)
// URL: https://codeforces.com/problemset/problem/486/D
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 思维 动态规划 树形dp 【重复做】
// 首先考虑去重问题：
//     定义一个点 u 比一个点 v 好，
//     (为确保唯一)
//     当且仅当 a[u] > a[v] 或者 a[u]=a[v] 且 u < v
// f[u] 为子树 u 中 u 为最好节点且节点差不超过 d 的所有方案

using LL = long long;
const int N = 2010, M = 4010, MOD = 1e9 + 7;

int d, n;
int h[N], e[M], ne[M], idx;
LL a[N], f[N];

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

bool better(int rt, int t) { return a[rt] > a[t] || a[rt] == a[t] && rt < t; }

void dfs(int u, int fa, int root) {
    f[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        // 思维 重要实现
        if (!better(root, j) || a[root] - a[j] > d)
            continue;
        dfs(j, u, root);
        (f[u] *= (f[j] + 1)) %= MOD;
    }
}

int main() {
    memset(h, -1, sizeof h);

    cin >> d >> n;

    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    for (int i = 1; i < n; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    LL res = 0;
    for (int i = 1; i <= n; ++i)
        dfs(i, -1, i), (res += f[i]) %= MOD;
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

> [!NOTE] **[Codeforces Zero Tree](http://codeforces.com/problemset/problem/274/B)**
> 
> 题意: 
> 
> 给出一棵 $n$ 个点带点权的树，每次操作可以对一个联通子图中的点全部加 $1$，或者全部减 $1$，**且每次操作必须包含点 $1$**
> 
> 问最少通过多少次操作可以让整棵树每个点的权值变为 $0$。

> [!TIP] **思路**
> 
> **对于每个点，只需关注它的加或减的次数最大的那个儿子即可**
> 
> **思维 trick 反复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Zero Tree
// Contest: Codeforces - Codeforces Round #168 (Div. 1)
// URL: https://codeforces.com/problemset/problem/274/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10, M = N << 1;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n;
LL w[N], up[N], dn[N];
// up 变为0加的次数,
// dn 减变为0的次数

void dfs(int u, int pa) {
    dn[u] = up[u] = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == pa)
            continue;
        dfs(j, u);
        up[u] = max(up[u], up[j]), dn[u] = max(dn[u], dn[j]);
    }
    w[u] += up[u] - dn[u];  // 更新u, 因为子树至少要执行这些操作
    if (w[u] > 0)
        dn[u] += w[u];
    else
        up[u] += -w[u];
    return;
}

int main() {
    init();

    cin >> n;
    for (int i = 0; i < n - 1; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    for (int i = 1; i <= n; ++i)
        cin >> w[i];

    dfs(1, -1);

    cout << up[1] + dn[1] << endl;

    return 0;
}
```

##### **C++ WA**

问题在于，可能同时修改【不在同一路径】下的非链的子树

下面的写法可能会导致多计算大量的操作

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10, M = N << 1;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n;
LL w[N], res;

LL dfs(int u, int pa) {
    LL c = 0;  // 记录所有子节点对当前值以及更上层的节点的影响
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == pa)
            continue;
        LL t = dfs(j, u);
        c += t;
        w[u] += t;
    }
    res += abs(w[u]);
    return c - w[u];
}

int main() {
    init();

    cin >> n;
    for (int i = 0; i < n - 1; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    for (int i = 1; i <= n; ++i)
        cin >> w[i];

    res = 0;
    dfs(1, -1);
    
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

> [!NOTE] **[LeetCode 2867. 统计树中的合法路径数目](https://leetcode.cn/problems/count-valid-paths-in-a-tree/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 树形 DP **边遍历边计算**
> 
> 加强敏感度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 初看似乎是 lca 应用, 麻烦的点在于复杂度无法接受枚举两个端点
    // 转换思路 & 反向思考: 假设某个点的编号是质数[数据范围不超过1e4个质数]，那么恰好经过当前点有多少条可行的路径？
    //  =>  dfs 遍历显然可以解 问题在于如何减少重复的双向遍历? => [标记已经被遍历过的点即可]
    //      ====> 【加强此类"边遍历边计算"的敏感度】
    using LL = long long;
    const static int N = 1e5 + 10, M = 2e5 + 10;
    
    // -------------------- graph --------------------
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    // -------------------- primes --------------------
    int ps[N], tot;
    bool st[N];
    void get_primes() {
        memset(st, 0, sizeof st);
        // ATTENTION set 1 因为 1 会被用到
        st[1] = true;
        tot = 0;
        for (int i = 2; i < N; ++ i ) {
            if (!st[i])
                ps[tot ++ ] = i;
            for (int j = 0; ps[j] <= (N - 1) / i; ++ j ) {
                st[ps[j] * i] = true;
                if (i % ps[j] == 0)
                    break;
            }
        }
    }
    
    // -------------------- dfs --------------------
    LL res;
    int f[N][2];    // 以 i 为最高点，向子树延伸，分别包含 0/1 个质数节点的方案数
    
    void dfs(int u, int fa) {
        f[u][!st[u]] = 1;    // st[u] = false 是质数
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            dfs(j, u);

            // ATTENTION 边遍历边累加 [此时 f[u] 是此前子树的状态和]
            res += (LL)f[j][0] * f[u][1] + (LL)f[j][1] * f[u][0];
            if (!st[u]) {    // 质数
                f[u][1] += f[j][0];
            } else {        // 非质数
                f[u][0] += f[j][0];
                f[u][1] += f[j][1];
            }
        }
    }
    
    long long countPaths(int n, vector<vector<int>>& edges) {
        init();
        get_primes();
        
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            add(a, b), add(b, a);
        }
        
        res = 0;
        memset(f, 0, sizeof f);
        dfs(1, -1);
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

> [!NOTE] **[LeetCode 2872. 可以被 K 整除连通块的最大数目](https://leetcode.cn/problems/maximum-number-of-k-divisible-components/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典树形 DP
> 
> 一开始在想是否有多种构造方案，实际上可以证明对于 "最大数目" 的情况方案是固定的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 思考: 如果某个边可以割，那割完之后的左右两侧一定需要上k整数倍，这样两侧才能独自一体或继续被割
    //   题目保证values之和可以被k整除 则一定有解
    // 1. dfs 求以某个节点为根的联通块数量及模数 返回根节点的数据即可
    // 2. 问题在于 不同的根结果是否不同？ ==> 结论是不会 【重点是证明】
    
    const static int N = 3e4 + 10, M = 6e4 + 10;
    
    // ---------------------- graph ----------------------
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    // ---------------------- dfs ----------------------
    int f[N], g[N]; // 数量 模数
    
    void dfs(int u, int fa) {
        f[u] = 0, g[u] = vs[u] % k;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            dfs(j, u);
            f[u] += f[j], g[u] = (g[u] + g[j]) % k;
        }
        if (!g[u])
            f[u] ++ ;
    }
    
    int n, k;
    vector<int> vs;
    
    int maxKDivisibleComponents(int n, vector<vector<int>>& edges, vector<int>& values, int k) {
        init();
        this->n = n, this->k = k, this->vs = values;
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            add(a, b), add(b, a);
        }
        
        dfs(0, -1);
        
        return f[0];
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
"""
求每个点到其他点的最长距离是多少，再在这些最长距离里选一个最小值。
怎么求每个点到其他点的最长距离：
1. 从当前节点往下走：上一题的 dist
2. 从当前节点往上走：其实就是求一个点的父节点不走该节点的最长路径。一个子节点j的向上最长路径就是，它的父节点u的最长向上路径和最长向下路径取最大值，
  如果向下最长路径经过了j, 就改成次长的向下路径。
3. d1[u]: 节点u向下走的最长路径的长度
  d2[u]: 节点u向下走的次长路径的长度
  p1[u]: 节点u向下走的最长路径是从哪个节点走过来的
  p2[u]: 节点u向下走的次长路径是从哪个节点走过来的
  up[u]：节点u向上走的最长路径的长度
"""
    
N = 10010
M = 2 * N
h, ev, w, ne = [-1] * M, [0] * M, [0] * M, [0] * M
d1, d2 = [0] * N, [0] * N
up, p = [0] * N, [0] * N
idx = 0


def add_edge(a, b, c):
    global idx
    ev[idx] = b
    w[idx] = c
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def dfs_down(u, father):
    d1[u], d2[u] = float('-inf'), float('-inf')
    i = h[u]
    while i != -1:
        j = ev[i]
        k = w[i]
        i = ne[i]
        if j == father:
            continue
        d = dfs_down(j, u) + k
        if d >= d1[u]:
            d2[u], d1[u], p[u] = d1[u], d, j
        elif d >= d2[u]:
            d2[u] = d

    if d1[u] == float('-inf'):
        d1[u], d2[u] = 0, 0
    return d1[u]


def dfs_up(u, father):
    i = h[u]
    while i != -1:
        j = ev[i]
        k = w[i]
        i = ne[i]
        if j == father:
            continue
        if p[u] == j:
            up[j] = max(up[u], d2[u]) + k
        else:
            up[j] = max(up[u], d1[u]) + k
        dfs_up(j, u)


if __name__ == '__main__':
    n = int(input())
    for i in range(n - 1):
        a, b, c = map(int, input().split())
        add_edge(a, b, c)
        add_edge(b, a, c)

    # 从任意一点开始找向下走的最长路径
    dfs_down(1, -1)
    
    # 从任意一点开始找向上走的最长路径
    dfs_up(1, -1)

    res = float('inf')
    for i in range(1, n + 1):
        res = min(res, max(up[i], d1[i]))

    print(res)
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

> [!NOTE] **[LeetCode 310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/)**
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
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        g = collections.defaultdict(list)
        d1, d2, p, up = [0] * n, [0] * n, [0] * n, [0] * n

        for e in edges:
            g[e[0]].append(e[1])
            g[e[1]].append(e[0])
            
        def dfs_down(u, fa):
            for v in g[u]:
                if v != fa:
                    dfs_down(v, u)
                    d = d1[v] + 1
                    if d >= d1[u]:
                        d2[u], d1[u], p[u] = d1[u], d, v
                    elif d > d2[u]:
                        d2[u] = d

        def dfs_up(u, fa):
            for v in g[u]:
                if v != fa:
                    if p[u] == v:
                        up[v] = max(up[u], d2[u]) + 1
                    else:
                        up[v] = max(up[u], d1[u]) + 1
                    dfs_up(v, u)
        
        dfs_down(0, -1)
        dfs_up(0, -1)

        mind = n + 1
        for i in range(n):
            mind = min(mind, max(up[i], d1[i]))
        res = []
        for i in range(n):
            maxv = max(up[i], d1[i])
            if maxv == mind:
                res.append(i)
        return res

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces Choosing Capital for Treeland](http://codeforces.com/problemset/problem/219/D)**
> 
> 题意: 
> 
> 单向边建立双向图，权 1/0 ，找到到各个点的距离总和最小的距离

> [!TIP] **思路**
> 
> 经典换根 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Choosing Capital for Treeland
// Contest: Codeforces - Codeforces Round #135 (Div. 2)
// URL: https://codeforces.com/problemset/problem/219/D
// Memory Limit: 256 MB
// Time Limit: 3000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 2e5 + 10, M = N << 1;

int h[N], e[M], w[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int n;
int fd[N], fu[N];

void dfs_d(int u, int fa) {
    fd[u] = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs_d(j, u);
        fd[u] += fd[j] + w[i];
    }
}

void dfs_u(int u, int fa) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        fu[j] = fu[u] + (w[i] != 1) + (fd[u] - fd[j] - w[i]);
        dfs_u(j, u);
    }
}

int main() {
    init();

    cin >> n;
    for (int i = 0; i < n - 1; ++i) {
        int s, t;
        cin >> s >> t;
        add(s, t, 0), add(t, s, 1);
    }

    dfs_d(1, -1);
    fu[1] = 0;
    dfs_u(1, -1);

    int res = 1e9;
    vector<int> xs;
    for (int i = 1; i <= n; ++i) {
        // cout << " i = " << i << " fd[i] = " << fd[i] << " fu[i] = " << fu[i]
        // << endl;
        int c = fd[i] + fu[i];
        if (c < res)
            res = c, xs = {i};
        else if (c == res)
            xs.push_back(i);
    }
    cout << res << endl;
    for (auto x : xs)
        cout << x << ' ';
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

> [!NOTE] **[Codeforces Lucky Tree](http://codeforces.com/problemset/problem/109/C)**
> 
> 题意: 
> 
> 有多少个三元组满足从一点出发到其他两点，路径上必然有幸运边

> [!TIP] **思路**
> 
> 经典换根，重点在状态定义与转移
> 
> 将路径有幸运边转化为有多少个点可以通过幸运边达到 ==> **分别向下/向上** ==> **换根DP**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Lucky Tree
// Contest: Codeforces - Codeforces Beta Round #84 (Div. 1 Only)
// URL: https://codeforces.com/problemset/problem/109/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10, M = 2e5 + 10;

int h[N], e[M], w[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int n;
LL fd[N], fu[N], sz[N];

void dfs_d(int u, int fa) {
    fd[u] = 0, sz[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i], c = w[i];
        if (j == fa)
            continue;
        dfs_d(j, u);
        sz[u] += sz[j];
        if (c == 1)
            fd[u] += sz[j];
        else
            fd[u] += fd[j];
    }
}

void dfs_u(int u, int fa) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i], c = w[i];
        if (j == fa)
            continue;
        if (c == 1)
            fu[j] += sz[1] - sz[j];  // 较显然
        else
            //       继承 + 其他子节点
            fu[j] += fu[u] + fd[u] - fd[j];
        dfs_u(j, u);
    }
}

int main() {
    init();

    cin >> n;
    for (int i = 0; i < n - 1; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        string s = to_string(c);
        bool flag = true;
        for (auto& ch : s)
            if (ch != '4' && ch != '7') {
                flag = false;
                break;
            }
        add(a, b, flag), add(b, a, flag);
    }
    dfs_d(1, -1);
    dfs_u(1, -1);

    LL res = 0;
    for (int i = 1; i <= n; ++i) {
        LL t = fd[i] + fu[i];
        res += t * (t - 1);  // 因为顺序不同也是不同，故不需要除2
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

> [!NOTE] **[Codeforces Book of Evil](http://codeforces.com/problemset/problem/337/D)**
> 
> 题意: 
> 
> 有一棵树有 $n$ 个节点，其中有 $m$ 个节点发现了怪物。
> 
> 已知树上有一本魔法书，魔法书可以让到其距离小于等于 $d$ 的点出现怪物，求魔法书所在点有几种可能。

> [!TIP] **思路**
> 
> 重在细节 **初始化-inf来避免复杂条件判断**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Book of Evil
// Contest: Codeforces - Codeforces Round #196 (Div. 2)
// URL: https://codeforces.com/problemset/problem/337/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10, M = 2e5 + 10, INF = 0x3f3f3f3f;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n, m, d;
int a[N];
bool st[N];

int fd[N], fu[N];
int pson[N], fd2[N];

void dfs_d(int u, int fa) {
    if (st[u])
        fd[u] = fd2[u] = 0;
    pson[u] = -1;

    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs_d(j, u);
        int t = fd[j] + 1;
        if (t + 1 > fd[u]) {
            pson[u] = j;
            fd2[u] = fd[u];
            fd[u] = t;
        } else if (t + 1 > fd2[u])
            fd2[u] = t;
    }
}

void dfs_u(int u, int fa) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        if (j == pson[u])
            fu[j] = max(fu[u], fd2[u]) + 1;
        else
            fu[j] = max(fu[u], fd[u]) + 1;
        dfs_u(j, u);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m >> d;

    memset(st, 0, sizeof st);
    for (int i = 0; i < m; ++i) {
        int t;
        cin >> t;
        st[t] = true;
    }

    init();
    for (int i = 0; i < n - 1; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    memset(fd, 0xcf, sizeof fd);  // -inf
    memset(fd2, 0xcf, sizeof fd2);
    memset(fu, 0xcf, sizeof fu);
    dfs_d(1, -1);
    dfs_u(1, -1);

    int res = 0;
    for (int i = 1; i <= n; ++i) {
        // cout << " i = " << i << " fd = " << fd[i] << " fu = " << fu << endl;
        if (max(fd[i], fu[i]) <= d)
            res++;
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

> [!NOTE] **[LeetCode 2538. 最大价值和与最小价值和的差值](https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准换根 dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 求 每一个点为根时的 最大路径与最小路径
    // 因为价值都是正数 显然最小路径就是根节点本身 求最大路径要 dp
    using LL = long long;
    const static int N = 1e5 + 10, M = 2e5 + 10;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    vector<int> p;
    LL d1[N], d2[N], up[N];
    int p1[N];
    
    void dfs_d(int u, int fa) {
        d1[u] = d2[u] = 0;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            dfs_d(j, u);
            int v = d1[j];
            if (v > d1[u]) {
                d2[u] = d1[u], d1[u] = v;
                p1[u] = j;
            } else if (v > d2[u])
                d2[u] = v;
        }
        
        d1[u] += p[u], d2[u] += p[u];
    }
    void dfs_u(int u, int fa) {
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            
            if (p1[u] == j)
                up[j] = max(up[u], d2[u]) + p[j];
            else
                up[j] = max(up[u], d1[u]) + p[j];
            
            dfs_u(j, u);
        }
    }
    
    long long maxOutput(int n, vector<vector<int>>& edges, vector<int>& price) {
        init();
        for (auto & e : edges)
            add(e[0], e[1]), add(e[1], e[0]);
        
        this->p = price;
        memset(p1, -1, sizeof p1);
        
        dfs_d(0, -1);
        dfs_u(0, -1);
        
        LL res = 0;
        for (int i = 0; i < n; ++ i )
            res = max(res, max(d1[i], up[i]) - p[i]);
        
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

> [!NOTE] **[LeetCode 2581. 统计可能的树根数目](https://leetcode.cn/problems/count-number-of-possible-root-nodes/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然经典树换根 DP
> 
> 需要加快速度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 数据范围 1e5 显然不能枚举哪些 guesses 为 true
    // 换个思路：
    //      考虑枚举根节点，则题意要求转化为【以某个节点为根的情况下 能满足 guesses 猜测 >= k 的情况】
    // 显然树dp + 换根
    
    const static int N = 1e5 + 10, M = 2e5 + 10;
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n, k;
    unordered_set<int> gs[N];
    int f[N], g[N];
    
    void dfs_d(int u, int pa) {
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            dfs_d(j, u);
            f[u] += f[j] + (gs[u].find(j) != gs[u].end());    // 如果 u 是 j 的父节点是一个已有的猜测
        }
    }
    void dfs_u(int u, int pa) {
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            int other = f[u] - f[j] - (gs[u].find(j) != gs[u].end());   // ATTENTION 理清思路：需要计算这一部分
            g[j] = g[u] + (gs[j].find(u) != gs[j].end()) + other;       // 如果 j 是 u 的父节点是一个已有的猜测
            dfs_u(j, u);
        }
    }
    
    int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k) {
        init();
        for (auto & e : edges)
            add(e[0], e[1]), add(e[1], e[0]);
        
        this->n = edges.size() + 1, this->k = k;
        for (auto & g : guesses)
            gs[g[0]].insert(g[1]);
        
        memset(f, 0, sizeof f), memset(g, 0, sizeof g);
        dfs_d(0, -1);
        dfs_u(0, -1);
        
        int res = 0;
        for (int i = 0; i < n; ++ i )
            if (f[i] + g[i] >= k)
                res ++ ;
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

> [!NOTE] **[LeetCode 2603. 收集树中金币](https://leetcode.cn/problems/collect-coins-in-a-tree/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO more details

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 【可以选择从任意一个节点出发】
    // 对题意做等价转换：
    //      - 收集所有的金币：一颗最小子树 访问子树可以在扩展两个节点距离的情况下收集所有金币
    //      - 需要返回原点：子树的最长路走一次 其他路径都要走两次
    // 考虑枚举树根 则容易得到对应的子树【】 也就能得到相应的开销
    // ATTENTION 距离为 2 意味着可以在每一个点向下暴力枚举 => ?
    const static int N = 3e4 + 10, M = 6e4 + 10;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    vector<int> cs;
    int n;
    
    // ATTENTION 难点在于想到状态定义：【定义距离恰好为1的情况的路程】
    int f[N], f1[N], f2[N];     // 以 i 为根的子树，收集 [所有金币/距离恰好为1的金币/距离2或以上的金币] 的路程
    int g[N], g1[N], g2[N];     // 向上的路程
    
    void dfs_d(int u, int pa) {
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            
            dfs_d(j, u);
            
            // j 需要走过去再回来
            f[u] += f[j] + (f2[j] > 0 ? 2 : 0);
            
            // 距离恰好为 1 的情况下统计
            f1[u] += cs[j];
            
            // 2 个或者以上的情况下统计
            f2[u] += f1[j] + f2[j];
        }
    }
    
    void dfs_u(int u, int pa) {
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            
            // 以 j 为根向上:
            //          上面的部分                +      兄弟节点的部分
            g[j] = (g[u] + (g2[u] > 0 ? 2 : 0)) + (f[u] - f[j] - (f2[j] > 0 ? 2 : 0));
            
            // 距离恰好为 1 的情况下:
            //          上面的部分            +   兄弟节点的部分?
            g1[j] = (pa != -1 ? cs[pa] : 0) + (f1[u] - cs[j]);
            
            // 距离大于等于 2 的情况下
            //          上面的部分  + 其他兄弟节点?
            g2[j] = g1[u] + g2[u] + (f2[u] - (f1[j] + f2[j]));
            
            dfs_u(j, u);
        }
    }
    
    int collectTheCoins(vector<int>& coins, vector<vector<int>>& edges) {
        this->cs = coins;
        this->n = cs.size();
        
        init();
        for (auto & e : edges)
            add(e[0], e[1]), add(e[1], e[0]);
        
        // cleanup
        memset(f, 0, sizeof f);
        memset(g, 0, sizeof g);
        
        dfs_d(0, -1);
        dfs_u(0, -1);

        int res = 2e9;
        for (int i = 0; i < n; ++ i )
            res = min(res, f[i] + g[i] + (g2[i] > 0 ? g[i] + 2 : 0));
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

> [!NOTE] **[LeetCode 2858. 可以到达每一个节点的最少边反转次数](https://leetcode.cn/problems/minimum-edge-reversals-so-every-node-is-reachable/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 将方向抽象为边权即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 显然是树形DP 考虑状态设计与转移
    const static int N = 1e5 + 10, M = 2e5 + 10;
    
    int h[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int f[N], g[N];
    
    void dfs_d(int u, int fa) {
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            dfs_d(j, u);
            f[u] += f[j] + (w[i] == 1);
        }
    }
    void dfs_u(int u, int fa) {
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            //    向上的部分 + 本条边        +  父节点的其他子树
            g[j] = g[u] + (w[i ^ 1] == 1) + (f[u] - f[j] - (w[i] == 1));
            dfs_u(j, u);
        }
    }
    
    vector<int> minEdgeReversals(int n, vector<vector<int>>& edges) {
        init();
        
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            add(a, b, 0);
            add(b, a, 1);   // ATTENTION 反转的边代价计为 1 则后续 DP 时即为统计某节点为根向外 1 的边的数量
        }
        
        memset(f, 0, sizeof f);
        memset(g, 0, sizeof g);
        dfs_d(0, -1);
        dfs_u(0, -1);
        
        // for (int i = 0; i < n; ++ i )
        //     cout << " i = " << i << ' ' << f[i] << ' ' << g[i] << endl;
        // cout << endl;
        
        vector<int> res;
        for (int i = 0; i < n; ++ i )
            res.push_back(f[i] + g[i]);
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

### 进阶：树DP状态设计

> [!NOTE] **[AcWing 323. 战略游戏](https://www.acwing.com/problem/content/description/325/)**
> 
> 题意: 树上放，监控边

> [!TIP] **思路**
> 与leetcode看守题不同之处在于：这个是看边 leetcode是看节点
> 
> https://leetcode.cn/problems/binary-tree-cameras/
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

> [!NOTE] **[LeetCode 968. 监控二叉树](https://leetcode.cn/problems/binary-tree-cameras)**
> 
> 题意: 同 [AcWing 323. 战略游戏](https://www.acwing.com/problem/content/description/325/)

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    using TIII = tuple<int, int, int>;

    unordered_map<TreeNode*, TIII> hash;

    void dfs(TreeNode * u) {
        int a = 0, b = 1e9, c = 1;
        int s = 0;
        for (auto x : {u->left, u->right})
            if (x) {
                dfs(x);
                auto [aa, bb, cc] = hash[x];
                a += min(bb, cc);
                c += min({aa, bb, cc});
                s += min(bb, cc);
            }
        
        for (auto x : {u->left, u->right})
            if (x) {
                auto [aa, bb, cc] = hash[x];
                b = min(b, s - min(bb, cc) + cc);
            }
        hash[u] = {a, b, c};
    }

    int minCameraCover(TreeNode* root) {
        dfs(root);
        auto [a, b, c] = hash[root];
        return min(b, c);
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

> [!NOTE] **[Luogu P4516 [JSOI2018] 潜入行动](https://www.luogu.com.cn/problem/P4516)** [TAG]
> 
> 题意: 
> 
> 树的节点上放，可以监控**除了本节点**以外的其他相邻点
> 
> 求监控所有点的方案数

> [!TIP] **思路**
> 
> 类似但有别于上一题 [AcWing 1077. 皇宫看守](https://www.acwing.com/problem/content/description/1079/)
> 
> **TODO: 思考细节**
> 
> 考虑分情况讨论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10, M = 110, MOD = 1e9 + 7;

int h[N], e[N << 1], ne[N << 1], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n, k;

int sz[N];
// 以i为根共放了j个装置，其中i有没有放置，有没有被监听到 的所有方案数
int f[N][M][2][2];  // 必须用 int 并在中间转LL
void modadd(int& a, LL b) { a = (a % MOD + b % MOD) % MOD; }

void dfs(int u, int fa) {
    sz[u] = 1;
    f[u][0][0][0] = f[u][1][1][0] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs(j, u);

        static int t[M][2][2];
        memcpy(t, f[u], sizeof t), memset(f[u], 0, sizeof f[u]);
        for (int x = 0; x <= min(sz[u], k); ++x)
            for (int y = 0; y <= min(sz[j], k - x); ++y) {
                // u 没被监听 => j 没放装置
                modadd(f[u][x + y][0][0], (LL)t[x][0][0] * f[j][y][0][1]);
                // u 没放但被监听 => 分情况
                modadd(f[u][x + y][0][1],
                       (LL)t[x][0][1] * ((LL)f[j][y][0][1] + f[j][y][1][1]) +
                           (LL)t[x][0][0] * f[j][y][1][1]);
                // u 没被监听但放了 => j 不能放装置
                modadd(f[u][x + y][1][0],
                       (LL)t[x][1][0] * (f[j][y][0][0] + f[j][y][0][1]));
                // u 放了也被监听 => 分情况
                modadd(f[u][x + y][1][1],
                       (LL)t[x][1][0] * ((LL)f[j][y][1][0] + f[j][y][1][1]) +
                           (LL)t[x][1][1] * ((LL)f[j][y][0][0] + f[j][y][0][1] +
                                             f[j][y][1][0] + f[j][y][1][1]));
            }

        // sz 修改必须放最后
        sz[u] += sz[j];
    }
}

int main() {
    init();

    cin >> n >> k;
    for (int i = 1; i < n; ++i) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    dfs(1, -1);
    cout << f[1][k][0][1] + f[1][k][1][1] << endl;

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

> [!NOTE] **[Codeforces Appleman and Tree](http://codeforces.com/problemset/problem/461/B)**
> 
> 题意: 
> 
> 给出一棵以 $0$ 为根的树，除根之外有些点是黑色，有些点是白色。
> 
> 求有多少种划分方案数，使得将树划分成若干个连通块并且每个连通块有且仅有一个黑点，对 $10^9+7$ 取模。

> [!TIP] **思路**
> 
> 重点在于 **梳理状态定义与转移**
> 
> 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Appleman and Tree
// Contest: Codeforces - Codeforces Round #263 (Div. 1)
// URL: https://codeforces.com/problemset/problem/461/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10, M = N << 1, MOD = 1e9 + 7;

int h[N], e[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n;
int color[N];
LL f[N][2];
// f[x][1] 表示当前根为x 在只有一个黑点的联通块里的方案数
// f[x][0] 表示当前根为x 当前在没有黑点的联通块里的方案数

void dfs(int u, int fa) {
    f[u][color[u]] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs(j, u);
        // ATTENTION
        // f[u][1] = f[u][1] * (f[j][0] + f[j][1]) + f[u][0] * f[j][1]
        // f[u][0] = f[u][0] * (f[j][0] + f[j][1])

        // 计算顺序有依赖，所以先算f[u][1]
        LL t = (f[j][0] + f[j][1]) % MOD;
        f[u][1] = (f[u][1] * t % MOD + f[u][0] * f[j][1] % MOD) % MOD;
        f[u][0] = f[u][0] * t % MOD;
    }
}

int main() {
    init();

    cin >> n;
    for (int i = 0, j; i < n - 1; ++i) {
        cin >> j;
        add(j, i + 1), add(i + 1, j);
    }

    for (int i = 0; i < n; ++i)
        cin >> color[i];

    memset(f, 0, sizeof f);
    dfs(0, -1);

    cout << f[0][1] << endl;

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

### 进阶: 边/点 思维

> [!NOTE] **[Luogu P3177 [HAOI2015] 树上染色](https://www.luogu.com.cn/problem/P3177)**
> 
> 题意: 
> 
> 有一棵点数为 $n$ 的树，树边有边权。给你一个在 $0 \sim n$ 之内的正整数 $k$ ，你要在这棵树中选择 $k$ 个点，将其染成黑色，并将其他 的 $n-k$ 个点染成白色。将所有点染色后，你会获得黑点两两之间的距离加上白点两两之间的距离的和的受益。问受益最大值是多少。

> [!TIP] **思路**
> 
> 受益即为距离和
> 
> **转换思想**: 求多个点对间的距离 => 求当前树边在多少点对路径上 => 对于当前树边来说两侧同色的节点各有多少个
> 
> **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 2010, M = N << 1;

int h[N], e[M], w[M], ne[M], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int n, k;

int sz[N];
LL f[N][N];  // 以i为根 黑色共有j个的受益总和

void dfs(int u, int fa) {
    sz[u] = 1;
    // 当前点黑不黑都是0
    f[u][0] = f[u][1] = 0;

    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa)
            continue;
        dfs(j, u);
        sz[u] += sz[j];

        // 枚举u黑色数目
        for (int x = min(k, sz[u]); x >= 0; --x)
            // 枚举子树黑色数目 注意顺序case
            for (int y = 0; y <= min(x, sz[j]); ++y)
                // ATTENTION: when f[u][x - y] != -1
                if (~f[u][x - y]) {
                    // 从u指向j的边的贡献
                    LL black = y * (k - y),
                       white = (sz[j] - y) * ((n - sz[j]) - (k - y));
                    // !!! w[i] 而不是 w[j] 写错了排查很久
                    LL t = (black + white) * w[i];
                    f[u][x] = max(f[u][x], f[u][x - y] + f[j][y] + t);
                }
    }
}

int main() {
    init();

    cin >> n >> k;
    for (int i = 1; i < n; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }

    memset(f, -1, sizeof f);
    dfs(1, -1);

    cout << f[1][k] << endl;

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

### 思维转化

> [!NOTE] **[LeetCode 2458. 移除子树后的二叉树高度](https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries/)** [TAG]
> 
> 题意: 
> 
> 一串询问，问每次删除某个完整子树后的树高（每次询问后都会恢复原树）

> [!TIP] **思路**
> 
> - 思维 考虑重链 => 考虑删除当前重链上的某个点之后 答案应当是多少 => 一定是从重链的非重子节点处获取所有可能答案
> 
> - 子树里的所有点是 DFS 序里的一个连续区间 => DFS 序 => 转变为删除连续区间后求全局最值的问题 【敏感度】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 思维 重链**

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    const static int N = 1e5 + 10;
    
    int n;
    int height[N];
    int tr[N][2], son[N];
    
    int t[N];
    
    void dfs(TreeNode * u) {
        if (!u)
            return;
        
        n ++ ;
        dfs(u->left), dfs(u->right);
        
        int x = u->val, l = 0, r = 0;
        if (u->left)
            tr[x][0] = u->left->val, l = height[u->left->val];
        if (u->right)
            tr[x][1] = u->right->val, r = height[u->right->val];
        
        if (l > r)
            son[x] = 0;
        else if (l < r)
            son[x] = 1;
        height[x] = max(l, r) + 1;
    }
    
    vector<int> treeQueries(TreeNode* root, vector<int>& queries) {
        memset(tr, 0, sizeof tr), memset(son, -1, sizeof son);
        n = 0;
        dfs(root);
        
        for (int i = 1; i <= n; ++ i )
            t[i] = height[root->val];
        
        // 核心思想: 我们只需要关注改变某个点之后一定会影响全局的部分即可
        // -> 对于有冗余备份的部分 其删除不影响全局 => 故只需要关心一条链 不需要 bfs
        // 
        // 对于该链上发生变化的节点 新值即为每个节点另一个方向的高度 + 当前深度
        for (int u = root->val, path = 0, maxd = 0, k; u && son[u] != -1; u = tr[u][k] ) {
            path ++ ;
            // 【思维】对于当前位置处 如果删掉了 tr[u][k] 则新的高度为 tr[u][k^1] + path
            k = son[u];
            maxd = max(maxd, height[tr[u][k ^ 1]] + path);
            t[tr[u][k]] = maxd;
        }
        
        vector<int> res;
        for (auto x : queries)
            res.push_back(t[x] - 1);
        return res;
    }
};
```

##### **C++ DFS 序**

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    const static int N = 1e5 + 10;
    
    int n, ts;
    int l[N], r[N];
    int d[N];
    
    void dfs(TreeNode * u, int dep) {
        if (!u)
            return;
        
        n ++ ;
        int x = u->val;
        
        l[x] = ++ ts;
        // ATTENTION 不是使用 x 而是使用 ts 来标记深度
        // 方便后面前后缀分别统计
        d[ts] = dep;
        dfs(u->left, dep + 1), dfs(u->right, dep + 1);
        r[x] = ts;
    }
    
    int f[N], g[N];
    
    vector<int> treeQueries(TreeNode* root, vector<int>& queries) {
        n = 0, ts = 0;
        dfs(root, 0);
        
        memset(f, 0, sizeof f), memset(g, 0, sizeof g);
        for (int i = 1; i <= n; ++ i )
            f[i] = max(f[i - 1], d[i]);
        for (int i = n; i >= 1; -- i )
            g[i] = max(g[i + 1], d[i]);
        
        vector<int> res;
        for (auto q : queries)
            res.push_back(max(f[l[q] - 1], g[r[q] + 1]));
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