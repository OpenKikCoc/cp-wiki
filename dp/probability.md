概率 DP 用于解决概率问题与期望问题，建议先对 [概率 & 期望](math/expectation.md) 的内容有一定了解。一般情况下，解决概率问题需要顺序循环，而解决期望问题使用逆序循环，如果定义的状态转移方程存在后效性问题，还需要用到 [高斯消元](math/linear-algebra/matrix.md) 来优化。概率 DP 也会结合其他知识进行考察，例如 [状态压缩](dp/state.md)，树上进行 DP 转移等。

## DP 求概率

这类题目采用顺推，也就是从初始状态推向结果。同一般的 DP 类似的，难点依然是对状态转移方程的刻画，只是这类题目经过了概率论知识的包装。

> [!NOTE] **例题 [Codeforces 148 D Bag of mice](https://codeforces.com/problemset/problem/148/D)**
> 
> 题目大意：袋子里有 w 只白鼠和 b 只黑鼠，公主和龙轮流从袋子里抓老鼠。谁先抓到白色老鼠谁就赢，如果袋子里没有老鼠了并且没有谁抓到白色老鼠，那么算龙赢。公主每次抓一只老鼠，龙每次抓完一只老鼠之后会有一只老鼠跑出来。每次抓的老鼠和跑出来的老鼠都是随机的。公主先抓。问公主赢的概率。

设 $f_{i,j}$ 为轮到公主时袋子里有 $i$ 只白鼠，$j$ 只黑鼠，公主赢的概率。初始化边界，$f_{0,j}=0$ 因为没有白鼠了算龙赢，$f_{i,0}=1$ 因为抓一只就是白鼠，公主赢。
考虑 $f_{i,j}$ 的转移：

- 公主抓到一只白鼠，公主赢了。概率为 $\frac{i}{i+j}$；
- 公主抓到一只黑鼠，龙抓到一只白鼠，龙赢了。概率为 $\frac{j}{i+j}\cdot \frac{i}{i+j-1}$；
- 公主抓到一只黑鼠，龙抓到一只黑鼠，跑出来一只黑鼠，转移到 $f_{i,j-3}$。概率为 $\frac{j}{i+j}\cdot\frac{j-1}{i+j-1}\cdot\frac{j-2}{i+j-2}$；
- 公主抓到一只黑鼠，龙抓到一只黑鼠，跑出来一只白鼠，转移到 $f_{i-1,j-2}$。概率为 $\frac{j}{i+j}\cdot\frac{j-1}{i+j-1}\cdot\frac{i}{i+j-2}$；

考虑公主赢的概率，第二种情况不参与计算。并且要保证后两种情况合法，所以还要判断 $i,j$ 的大小，满足第三种情况至少要有 3 只黑鼠，满足第四种情况要有 1 只白鼠和 2 只黑鼠。


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

### 习题

- [CodeForces 148 D Bag of mice](https://codeforces.com/problemset/problem/148/D)
- [POJ3071 Football](http://poj.org/problem?id=3071)
- [CodeForces 768 D Jon and Orbs](https://codeforces.com/problemset/problem/768/D)

## DP 求期望

> [!NOTE] **例题 [POJ2096 Collecting Bugs](http://poj.org/problem?id=2096)**
> 
> 题目大意：一个软件有 $s$ 个子系统，会产生 $n$ 种 bug。某人一天发现一个 bug，这个 bug 属于某种 bug 分类，也属于某个子系统。每个 bug 属于某个子系统的概率是 $\frac{1}{s}$，属于某种 bug 分类的概率是 $\frac{1}{n}$。求发现 $n$ 种 bug，且 $s$ 个子系统都找到 bug 的期望天数。

令 $f_{i,j}$ 为已经找到 $i$ 种 bug 分类，$j$ 个子系统的 bug，达到目标状态的期望天数。这里的目标状态是找到 $n$ 种 bug 分类，$s$ 个子系统的 bug。那么就有 $f_{n,s}=0$，因为已经达到了目标状态，不需要用更多的天数去发现 bug 了，于是就以目标状态为起点开始递推，答案是 $f_{0,0}$。

考虑 $f_{i,j}$ 的状态转移：

- $f_{i,j}$，发现一个 bug 属于已经发现的 $i$ 种 bug 分类，$j$ 个子系统，概率为 $p_1=\frac{i}{n}\cdot\frac{j}{s}$
- $f_{i,j+1}$，发现一个 bug 属于已经发现的 $i$ 种 bug 分类，不属于已经发现的子系统，概率为 $p_2=\frac{i}{n}\cdot(1-\frac{j}{s})$
- $f_{i+1,j}$，发现一个 bug 不属于已经发现 bug 分类，属于 $j$ 个子系统，概率为 $p_3=(1-\frac{i}{n})\cdot\frac{j}{s}$
- $f_{i+1,j+1}$，发现一个 bug 不属于已经发现 bug 分类，不属于已经发现的子系统，概率为 $p_4=(1-\frac{i}{n})\cdot(1-\frac{j}{s})$

再根据期望的线性性质，就可以得到状态转移方程：

$$
\begin{aligned}
f_{i,j} &= p_1\cdot f_{i,j}+p_2\cdot f_{i,j+1}+p_3\cdot f_{i+1,j}+p_4\cdot f_{i+1,j+1} + 1\\
&= \frac{p_2\cdot f_{i,j+1}+p_3\cdot f_{i+1,j}+p_4\cdot f_{i+1,j+1}+1}{1-p_1}
\end{aligned}
$$


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

> [!NOTE] **例题 [「NOIP2016」换教室](http://uoj.ac/problem/262)**
> 
> 题目大意：牛牛要上 $n$ 个时间段的课，第 $i$ 个时间段在 $c_i$ 号教室，可以申请换到 $d_i$ 号教室，申请成功的概率为 $p_i$，至多可以申请 $m$ 节课进行交换。第 $i$ 个时间段的课上完后要走到第 $i+1$ 个时间段的教室，给出一张图 $v$ 个教室 $e$ 条路，移动会消耗体力，申请哪几门课程可以使他因在教室间移动耗费的体力值的总和的期望值最小，也就是求出最小的期望路程和。

对于这个无向连通图，先用 Floyd 求出最短路，为后续的状态转移带来便利。以移动一步为一个阶段（从第 $i$ 个时间段到达第 $i+1$ 个时间段就是移动了一步），那么每一步就有 $p_i$ 的概率到 $d_i$，不过在所有的 $d_i$ 中只能选 $m$ 个，有 $1-p_i$ 的概率到 $c_i$，求出在 $n$ 个阶段走完后的最小期望路程和。
定义 $f_{i,j,0/1}$ 为在第 $i$ 个时间段，连同这一个时间段已经用了 $j$ 次换教室的机会，在这个时间段换（1）或者不换（0）教室的最小期望路程和，那么答案就是 $max \{f_{n,i,0},f_{n,i,1}\} ,i\in[0,m]$。注意边界 $f_{1,0,0}=f_{1,1,1}=0$。

考虑 $f_{i,j,0/1}$ 的状态转移：

- 如果这一阶段不换，即 $f_{i,j,0}$。可能是由上一次不换的状态转移来的，那么就是 $f_{i-1,j,0}+w_{c_{i-1},c_{i}}$, 也有可能是由上一次交换的状态转移来的，这里结合条件概率和全概率的知识分析可以得到 $f_{i-1,j,1}+w_{d_{i-1},c_{i}}\cdot p_{i-1}+w_{c_{i-1},c_{i}}\cdot (1-p_{i-1})$，状态转移方程就有

$$
\begin{aligned}
f_{i,j,0}=min(f_{i-1,j,0}+w_{c_{i-1},c_{i}},f_{i-1,j,1}+w_{d_{i-1},c_{i}}\cdot p_{i-1}+w_{c_{i-1},c_{i}}\cdot (1-p_{i-1}))
\end{aligned}
$$

- 如果这一阶段交换，即 $f_{i,j,1}$。类似地，可能由上一次不换的状态转移来，也可能由上一次交换的状态转移来。那么遇到不换的就乘上 $(1-p_i)$，遇到交换的就乘上 $p_i$，将所有会出现的情况都枚举一遍出进行计算就好了。这里不再赘述各种转移情况，相信通过上一种阶段例子，这里的状态转移应该能够很容易写出来。


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

比较这两个问题可以发现，DP 求期望题目在对具体是求一个值或是最优化问题上会对方程得到转移方式有一些影响，但无论是 DP 求概率还是 DP 求期望，总是离不开概率知识和列出、化简计算公式的步骤，在写状态转移方程时需要思考的细节也类似。

### 习题

- [POJ2096 Collecting Bugs](http://poj.org/problem?id=2096)
- [HDU3853 LOOPS](http://acm.hdu.edu.cn/showproblem.php?pid=3853)
- [HDU4035 Maze](http://acm.hdu.edu.cn/showproblem.php?pid=4035)
- [「NOIP2016」换教室](http://uoj.ac/problem/262)
- [「SCOI2008」奖励关](https://www.luogu.com.cn/problem/P2473)

## 有后效性 DP

> [!NOTE] **[CodeForces 24 D Broken robot](https://codeforces.com/problemset/problem/24/D)**
    题目大意：给出一个 $n*m$ 的矩阵区域，一个机器人初始在第 $x$ 行第 $y$ 列，每一步机器人会等概率地选择停在原地，左移一步，右移一步，下移一步，如果机器人在边界则不会往区域外移动，问机器人到达最后一行的期望步数。

在 $m=1$ 时每次有 $\frac{1}{2}$ 的概率不动，有 $\frac{1}{2}$ 的概率向下移动一格，答案为 $2\cdot (n-x)$。
设 $f_{i,j}$ 为机器人机器人从第 i 行第 j 列出发到达第 $n$ 行的期望步数，最终状态为 $f_{n,j}=0$。
由于机器人会等概率地选择停在原地，左移一步，右移一步，下移一步，考虑 $f_{i,j}$ 的状态转移：

- $f_{i,1}=\frac{1}{3}\cdot(f_{i+1,1}+f_{i,2}+f_{i,1})+1$
- $f_{i,j}=\frac{1}{4}\cdot(f_{i,j}+f_{i,j-1}+f_{i,j+1}+f_{i+1,j})+1$
- $f_{i,m}=\frac{1}{3}\cdot(f_{i,m}+f_{i,m-1}+f_{i+1,m})+1$

在行之间由于只能向下移动，是满足无后效性的。在列之间可以左右移动，在移动过程中可能产生环，不满足无后效性。
将方程变换后可以得到：

- $2f_{i,1}-f_{i,2}=3+f_{i+1,1}$
- $3f_{i,j}-f_{i,j-1}-f_{i,j+1}=4+f_{i+1,j}$
- $2f_{i,m}-f_{i,m-1}=3+f_{i+1,m}$

由于是逆序的递推，所以每一个 $f_{i+1,j}$ 是已知的。
由于有 $m$ 列，所以右边相当于是一个 $m$ 行的列向量，那么左边就是 $m$ 行 $m$ 列的矩阵。使用增广矩阵，就变成了 m 行 m+1 列的矩阵，然后进行 [高斯消元](math/linear-algebra/matrix.md) 即可解出答案。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 1e3 + 10;

double a[maxn][maxn], f[maxn];
int n, m;

void solve(int x) {
    memset(a, 0, sizeof a);
    for (int i = 1; i <= m; i++) {
        if (i == 1) {
            a[i][i] = 2;
            a[i][i + 1] = -1;
            a[i][m + 1] = 3 + f[i];
            continue;
        } else if (i == m) {
            a[i][i] = 2;
            a[i][i - 1] = -1;
            a[i][m + 1] = 3 + f[i];
            continue;
        }
        a[i][i] = 3;
        a[i][i + 1] = -1;
        a[i][i - 1] = -1;
        a[i][m + 1] = 4 + f[i];
    }

    for (int i = 1; i < m; i++) {
        double p = a[i + 1][i] / a[i][i];
        a[i + 1][i] = 0;
        a[i + 1][i + 1] -= a[i][i + 1] * p;
        a[i + 1][m + 1] -= a[i][m + 1] * p;
    }

    f[m] = a[m][m + 1] / a[m][m];
    for (int i = m - 1; i >= 1; i--)
        f[i] = (a[i][m + 1] - f[i + 1] * a[i][i + 1]) / a[i][i];
}

int main() {
    scanf("%d %d", &n, &m);
    int st, ed;
    scanf("%d %d", &st, &ed);
    if (m == 1) {
        printf("%.10f\n", 2.0 * (n - st));
        return 0;
    }
    for (int i = n - 1; i >= st; i--) { solve(i); }
    printf("%.10f\n", f[ed]);
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

### 习题

- [CodeForce 24 D Broken robot](https://codeforces.com/problemset/problem/24/D)
- [HDU Time Travel](http://acm.hdu.edu.cn/showproblem.php?pid=4418)
- [「HNOI2013」游走](https://loj.ac/problem/2383)

## 参考文献

[kuangbin 概率 DP 总结](https://www.cnblogs.com/kuangbin/archive/2012/10/02/2710606.html)


## 习题

> [!NOTE] **[LeetCode 1377. T 秒后青蛙的位置](https://leetcode.cn/problems/frog-position-after-t-seconds/)** [TAG]
> 
> 题意: 
> 
> 青蛙从根开始跳 求 t 秒后在 target 点的概率

> [!TIP] **思路**
> 
> 大量细节
> 
> TODO **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
    double pr[105];  // 表示跳到每个节点的概率,初始值pr[1] = 1.0
    bool vis[105];   // dfs过程会使用到，用于记录该节点有没有被遍历过。
    map<int, vector<int>> mp;             // 记录边的连接信息
    void dfs(int cur, int t) {
        if (t <= 0) return;               // 如果时间到了，那就退出
        int to_count = 0;
        for (auto next : mp[cur])
            if (!vis[next]) to_count++;   // 首先观察青蛙能去的地方
        if (to_count == 0) return;        // 如果已经没有地方能去了，那就退出
        double p = pr[cur] / to_count;    // 跳往每个地方都是均匀分布的，概率均匀
        for (auto next : mp[cur]) {
            if (!vis[next]) {
                vis[next] = true;
                pr[cur] -= p;
                pr[next] += p;            // 开始跳之前，将概率转移
                dfs(next, t - 1);         // 这样青蛙就可以安心跳过去了
                vis[next] = false;
            }
        }
    }

public:
    double frogPosition(int n, vector<vector<int>>& edges, int t, int target) {
        // 题目的数据量很小，才100个节点，本来担心暴力dfs模拟的话会不会超时。
        // 但是青蛙不会走回头路，这样就可以去掉很多很多的情况，对于dfs来说，应该不会超时的，然后开始干！
        for (int i = 0; i < n; i++)
            pr[i] = 0, vis[i] = false;
        for (auto edge : edges) {
            mp[edge[0]].push_back(edge[1]);
            mp[edge[1]].push_back(edge[0]);  //因为是无向图
        }
        pr[1] = 1, vis[1] = true;  //初始化表示在当前1节点。
        dfs(1, t);
        return pr[target];
    }
};

// 作者：wu-bin-cong
```

##### **C++ dfs better**

```cpp
class Solution {
public:
    double f[105][55];
    vector<int> G[105];
    void dfs(int cur, int fa, int curt) {
        int sz = G[cur].size();
        if (fa) --sz;
        if (sz == 0) {
            f[cur][curt] += f[cur][curt - 1];
            return;
        }
        for (int x : G[cur]) {
            if (x == fa) continue;
            f[x][curt] = 1.0 * f[cur][curt - 1] * (1.0 / sz);
            dfs(x, cur, curt);
        }
    }
    double frogPosition(int n, vector<vector<int>>& edges, int t, int target) {
        for (auto e : edges) {
            G[e[0]].push_back(e[1]);
            G[e[1]].push_back(e[0]);
        }
        f[1][0] = 1;
        for (int i = 1; i <= t; ++i) { dfs(1, 0, i); }
        return f[target][t];
    }
};
```

##### **C++ bfs better**

```cpp
class Solution {
public:
    double frogPosition(int n, vector<vector<int>>& edges, int t, int target) {
        vector<vector<int>> tree(n + 1);

        for (const auto &e : edges) {
            tree[e[0]].push_back(e[1]);
            tree[e[1]].push_back(e[0]);
        }

        vector<pair<int, double>> dis(n + 1, {INT_MAX, 0});
        queue<int> q;

        q.push(1);
        dis[1].first = 0;
        dis[1].second = 1;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            if (dis[u].first >= t)
                continue;

            for (int v : tree[u])
                if (dis[v].first > dis[u].first + 1) {
                    dis[v].first = dis[u].first + 1;
                    dis[v].second = dis[u].second / (tree[u].size() - (int)(u != 1));
                    q.push(v);
                }
        }

        if (dis[target].first < t) {
            if (target == 1 && tree[1].size() > 0
               || tree[target].size() > 1)
                return 0;
        }

        return dis[target].second;
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

> [!NOTE] **[Codeforces A. Little Pony and Expected Maximum](https://codeforces.com/problemset/problem/453/A)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 概率与期望
> 
> 重复做 期望dp
> 
> [Luogu](https://www.luogu.com.cn/problem/solution/CF453A)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

https://www.luogu.com.cn/blog/da32s1da/solution-cf453a

如果投 n 次，最大点数是 k ，那么情况共有 k^n − (k−1)^n 种。

若 n 次投掷的点数都在 1 到 k 内，共有 k^n 种情况。

若 n 次投掷的点数都在 1 到 k−1 内，共有 (k−1)^n 种情况。

两数相减可得最大值是 k 的情况

所以期望 = $\sum_{i=1}^m i * (i ^ n − (i − 1) ^ n)$

也即 $\sum_{i=1}^m i * ((i / m) ^ n − ((i - 1) / m) ^ n)$

```cpp
// Problem: A. Little Pony and Expected Maximum
// Contest: Codeforces - Codeforces Round #259 (Div. 1)
// URL: https://codeforces.com/problemset/problem/453/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

int main() {
    // ATTENTION use double here for convenient
    // cause `pow` need double
    double n, m;
    cin >> m >> n;

    double res = 0;
    for (int i = 1; i <= m; ++i)
        res += (double)i * (pow(i / m, n) - pow((i - 1) / m, n));
    printf("%.12lf\n", res);

    return 0;
}
```

##### **Python**

```python
// Problem: A. Little Pony and Expected Maximum
// Contest: Codeforces - Codeforces Round #259 (Div. 1)
// URL: https://codeforces.com/problemset/problem/453/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

int main() {
    // ATTENTION use double here for convenient
    // cause `pow` need double
    double n, m;
    cin >> m >> n;

    double res = 0;
    for (int i = 1; i <= m; ++i)
        res += (double)i * (pow(i / m, n) - pow((i - 1) / m, n));
    printf("%.12lf\n", res);

    return 0;
}
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces Bad Luck Island](http://codeforces.com/problemset/problem/540/D)** [TAG]
> 
> 题意: 
> 
> 在孤岛上有三种人，分别有 $r,s,p$ 个， 每两个人相遇的概率相等，相遇时 $r$ 吃 $s$，$s$ 吃 $p$，$p$ 吃 $r$，分别求最后剩下一种种族的概率。

> [!TIP] **思路**
> 
> 显然当其他俩都为 0 个时可以累计当前物种存活的概率。
> 
> 重点在于计算递推过程
> 
> 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Bad Luck Island
// Contest: Codeforces - Codeforces Round #301 (Div. 2)
// URL: https://codeforces.com/problemset/problem/540/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 110;

int r, s, p;
double f[N][N][N];

int main() {
    cin >> r >> s >> p;

    double fr = 0, fs = 0, fp = 0;
    f[r][s][p] = 1.0;
    for (int i = r; i >= 0; --i)
        for (int j = s; j >= 0; --j)
            for (int k = p; k >= 0; --k) {
                // 总的可能方案
                double tot = i * j + j * k + i * k;
                if (i && j)
                    f[i][j - 1][k] += f[i][j][k] * i * j / tot;
                if (j && k)
                    f[i][j][k - 1] += f[i][j][k] * j * k / tot;
                if (i && k)
                    f[i - 1][j][k] += f[i][j][k] * i * k / tot;
                if (i && !j && !k)
                    fr += f[i][j][k];
                if (!i && j && !k)
                    fs += f[i][j][k];
                if (!i && !j && k)
                    fp += f[i][j][k];
            }
    printf("%.12f %.12f %.12f\n", fr, fs, fp);

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

> [!NOTE] **[Codeforces Bag of mice](http://codeforces.com/problemset/problem/148/D)** [TAG]
> 
> 题意: 
> 
> 袋子里有 w 只白鼠和 b 只黑鼠 ，A和B轮流从袋子里抓，谁先抓到白色谁就赢。
> 
> A每次随机抓一只，B每次随机抓完一只之后会有另一只随机老鼠跑出来。
> 
> 如果两个人都没有抓到白色则B赢。A先抓，问A赢的概率。

> [!TIP] **思路**
> 
> 分情况讨论
> 
> TODO: 明确为什么在转移时不需要关心先后手
> 
> TODO: 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Bag of mice
// Contest: Codeforces - Codeforces Round #105 (Div. 2)
// URL: https://codeforces.com/problemset/problem/148/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// 谁先抓到白色谁赢 或者最后为空龙赢
// ATTENTION 分先后手

const static int N = 1010;

int w, b;
double f[N][N];  // f[i][j] 剩下 i 个白，j 个黑时公主赢的概率

int main() {
    cin >> w >> b;

    // init
    // 全为白必胜 有一个黑胜率i/(i+1)
    for (int i = 1; i <= w; ++i)
        f[i][0] = 1.0, f[i][1] = 1.0 * i / (i + 1);

    for (int i = 1; i <= w; ++i)
        for (int j = 2; j <= b; ++j) {
            // 1. 先手白兔
            f[i][j] = 1.0 * i / (i + j);
            // 2. 先手黑 后手白
            f[i][j] += 0;
            // 3. 先手黑 后手黑 跑一个白
            f[i][j] += 1.0 * j / (i + j) * (j - 1) / (i + j - 1) * i /
                       (i + j - 2) * f[i - 1][j - 2];
            // 4. 先手黑 后手黑 跑一个黑
            if (j ^ 2)  // j > 2
                f[i][j] += 1.0 * j / (i + j) * (j - 1) / (i + j - 1) * (j - 2) /
                           (i + j - 2) * f[i][j - 3];
        }
    printf("%.9lf\n", f[w][b]);

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

> [!NOTE] **[LeetCode 837. 新 21 点](https://leetcode.cn/problems/new-21-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有正反两种思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 正向**

```cpp
class Solution {
public:
    const static int N = 1e4 + 10;

    // f[i] 表示得到 i 点分数的概率
    double f[N], s[N];

    double new21Game(int n, int k, int maxPts) {
        if (k == 0)
            return 1;
        
        memset(f, 0, sizeof f), memset(s, 0, sizeof s);
        f[0] = s[0] = 1;

        for (int i = 1; i <= n; ++ i ) {
            int l = max(0, i - maxPts), r = min(k - 1, i - 1);
            if (l <= r) {
                if (l == 0)
                    f[i] = s[r] / maxPts;
                else
                    f[i] = (s[r] - s[l - 1]) / maxPts;
            }
            s[i] = s[i - 1] + f[i];
        }
        return s[n] - s[k - 1];
    }
};
```

##### **C++ 反向**

```cpp
class Solution {
public:
    const static int N = 2e4 + 10;

    // f[i] 表示当前位于 i 的局面，获胜的概率
    // 所谓获胜即为得到 [k, min(n, k + maxPts - 1)] 的情况
    double f[N];

    double new21Game(int n, int k, int maxPts) {
        if (k == 0)
            return 1;
        
        memset(f, 0, sizeof f);
        for (int i = k; i <= n && i < k + maxPts; ++ i )
            f[i] = 1;

        // 从 k-1 开始往前算
        // 1. 计算 k-1
        f[k - 1] = 0;
        for (int i = 1; i <= maxPts; ++ i )
            f[k - 1] += f[k - 1 + i] / (double)maxPts;
        
        // 2. 计算前面的部分
        for (int i = k - 2; i >= 0; -- i )
            // 基于 f[i + 1] 做修订，即得到 f[i]
            // 修订即 (f[i+1]-f[i+maxPts+1])/maxPts
            f[i] = f[i + 1] + (f[i + 1] - f[i + maxPts + 1]) / (double)maxPts;
        
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

> [!NOTE] **[LeetCode 808. 分汤](https://leetcode.cn/problems/soup-servings/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 细节推导，缩小范围，分情况讨论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 510;

    double f[N][N];

    double g(int x, int y) {
        return f[max(0, x)][max(0, y)];
    }

    double soupServings(int n) {
        n = (n + 24) / 25;  // 先约减一下方便操作
        if (n >= 500)       // 从期望上看，>= 500 时无限趋近于 1
            return 1;
        
        for (int i = 0; i <= n; ++ i )
            for (int j = 0; j <= n; ++ j ) {
                if (!i && !j)
                    f[i][j] = 0.5;
                else if (i && !j)
                    f[i][j] = 0;
                else if (!i && j)
                    f[i][j] = 1;
                else
                    f[i][j] = (g(i - 4, j) + g(i - 3, j - 1) + g(i - 2, j - 2) + g(i - 1, j - 3)) / 4.0;
            }
        return f[n][n];
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

> [!NOTE] **[LeetCode 1467. 两个盒子中球的颜色数相同的概率](https://leetcode.cn/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp + 组合数
> 
> [题解](https://leetcode.cn/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/solution/cdong-tai-gui-hua-bi-sai-de-shi-hou-bei-fan-yi-ken/)
> 
> 重复做 todo

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    double getProbability(vector<int>& balls) {
        // 颜色数和球的数量
        const int k = balls.size();
        const int n = accumulate(balls.begin(), balls.end(), 0) / 2;
        // 预处理阶乘
        vector<double> fact;
        fact.push_back(1.0);
        for (int i = 1; i <= 2 * n; ++i) { fact.push_back(fact[i - 1] * i); }
        // 总的排列方法数
        double total = fact[2 * n];
        for (auto ball : balls) { total /= fact[ball]; }
        // 动态规划
        vector<vector<double>> dp(2 * n + 1, vector<double>(2 * k + 1, 0.0));
        dp[0][k] = 1.0;
        int num = 0;
        for (int i = 0; i < k; ++i) {
            vector<vector<double>> next(2 * n + 1,
                                        vector<double>(2 * k + 1, 0.0));
            for (int j = 0; j <= balls[i]; ++j) {
                int trans = 0;
                trans = j == 0 ? -1 : trans;
                trans = j == balls[i] ? 1 : trans;
                for (int front = 0; front <= 2 * n; ++front)
                    for (int color = 0; color <= 2 * k; ++color) {
                        if (dp[front][color] == 0) continue;
                        double ways = dp[front][color];
                        ways *= fact[front + j] / (fact[front] * fact[j]);
                        ways *= fact[num - front + balls[i] - j] /
                                (fact[num - front] * fact[balls[i] - j]);
                        next[front + j][color + trans] += ways;
                    }
            }
            swap(dp, next);
            num += balls[i];
        }
        return dp[n][k] / total;
    }
};

// mskadr
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *
