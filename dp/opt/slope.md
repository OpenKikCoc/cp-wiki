> [!NOTE] **ATTENTION**
> 
> 重点在于：根据其形式进行转换和推断，并应用斜率优化
> 
> 随后在查询以及维护时，结合其他各类数据结构

> [!NOTE] **ATTENTION**
> 
> 维护凸包过程中，假定先后有 $a, b, c$ 三个点，其中 $a, b$ 是已在凸包栈中的点
> 
> 则: 既可以求 $c-a, b-a$ 叉积，也可以求 $c-b, b-a$ 叉积，效果是一样的

## 例题

> [!NOTE] **[「HNOI2008」玩具装箱](https://loj.ac/problem/10188)**
> 
> 有 $n$ 个玩具，第 $i$ 个玩具价值为 $c_i$。要求将这 $n$ 个玩具排成一排，分成若干段。对于一段 $[l,r]$，它的代价为 $(r-l+\sum_{i=L}^R c_i-L)^2$。求分段的最小代价。
> 
> $1\le n\le 5\times 10^4,1\le L,0\le c_i\le 10^7$。

令 $f_i$ 表示前 $i$ 个物品，分若干段的最小代价。

状态转移方程：$f_i=\min_{j<i}\{f_j+(pre_i-pre_j+i-j-1-L)^2\}$。

其中 $pre_i$ 表示前 $i$ 个数的和，即 $\sum_{j=1}^i c_j$。

简化状态转移方程式：令 $s_i=pre_i+i,L'=L+1$，则 $f_i=\min_{j<i}\{f_j+(s_i-s_j-L')^2\}$。

将与 $j$ 无关的移到外面，我们得到

$$
f_i - (s_i-L')^2=\min_{j<i}\{f_j+s_j^2 + 2s_j(L'-s_i) \} 
$$

考虑一次函数的斜截式 $y=kx+b$，将其移项得到 $b=y-kx$。我们将与 $j$ 有关的信息表示为 $y$ 的形式，把同时与 $i,j$ 有关的信息表示为 $kx$，把要最小化的信息（与 $i$ 有关的信息）表示为 $b$，也就是截距。具体地，设

$$
\begin{aligned}
x_j&=s_j\\
y_j&=f_j+s_j^2\\
k_i&=-2(L'-s_i)\\
b_i&=f_i-(s_i-L')^2\\
\end{aligned}
$$

则转移方程就写作 $b_i = \min_{j<i}\{ y_j-k_ix_j \}$。我们把 $(x_j,y_j)$ 看作二维平面上的点，则 $k_i$ 表示直线斜率，$b_i$ 表示一条过 $(x_j,y_j)$ 的斜率为 $k_i$ 的直线的截距。问题转化为了，选择合适的 $j$（$1\le j<i$），最小化直线的截距。

![slope_optimization](images/optimization.svg)

如图，我们将这个斜率为 $k_i$ 的直线从下往上平移，直到有一个点 $(x_p,y_p)$ 在这条直线上，则有 $b_i=y_p-k_ix_p$，这时 $b_i$ 取到最小值。算完 $f_i$，我们就把 $(x_i,y_i)$ 这个点加入点集中，以做为新的 DP 决策。那么，我们该如何维护点集？

容易发现，可能让 $b_i$ 取到最小值的点一定在下凸壳上。因此在寻找 $p$ 的时候我们不需要枚举所有 $i-1$ 个点，只需要考虑凸包上的点。而在本题中 $k_i$ 随 $i$ 的增加而递增，因此我们可以单调队列维护凸包。

具体地，设 $K(a,b)$ 表示过 $(x_a,y_a)$ 和 $(x_b,y_b)$ 的直线的斜率。考虑队列 $q_l,q_{l+1},\ldots,q_r$，维护的是下凸壳上的点。也就是说，对于 $l<i<r$，始终有 $K(q_{i-1},q_i) < K(q_i,q_{i+1})$ 成立。

我们维护一个指针 $e$ 来计算 $b_i$ 最小值。我们需要找到一个 $K(q_{e-1},q_e)\le k_i< K(q_e,q_{e+1})$ 的 $e$（特别地，当 $e=l$ 或者 $e=r$ 时要特别判断），这时就有 $p=q_e$，即 $q_e$ 是 $i$ 的最优决策点。由于 $k_i$ 是单调递减的，因此 $e$ 的移动次数是均摊 $O(1)$ 的。

在插入一个点 $(x_i,y_i)$ 时，我们要判断是否 $K(q_{r-1},q_r)<K(q_r,i)$，如果不等式不成立就将 $q_r$ 弹出，直到等式满足。然后将 $i$ 插入到 $q$ 队尾。

这样我们就将 DP 的复杂度优化到了 $O(n)$。

概括一下上述斜率优化模板题的算法：

1. 将初始状态入队。
2. 每次使用一条和 $i$ 相关的直线 $f(i)$ 去切维护的凸包，找到最优决策，更新 $dp_i$。
3. 加入状态 $dp_i$。如果一个状态（即凸包上的一个点）在 $dp_i$ 加入后不再是凸包上的点，需要在 $dp_i$ 加入前将其剔除。

接下来我们介绍斜率优化的进阶应用，将斜率优化与二分/分治/数据结构等结合，来维护性质不那么好（缺少一些单调性性质）的 DP 方程。

## 二分/CDQ/平衡树优化 DP

当我们在 $i$ 这个点寻找最优决策时，会使用一个和 $i$ 相关的直线 $f(i)$ 去切我们维护的凸包。切到的点即为最优决策。

在上述例题中，直线的斜率随 $i$ 单调变化，但是对于有些问题，斜率并不是单调的。这时我们需要维护凸包上的每一个节点，然后每次用当前的直线去切这个凸包。这个过程可以使用二分解决，因为凸包上相邻两个点的斜率是有单调性的。

> [!NOTE] **玩具装箱 改**
> 
> 有 $n$ 个玩具，第 $i$ 个玩具价值为 $c_i$。要求将这 $n$ 个玩具排成一排，分成若干段。对于一段 $[l,r]$，它的代价为 $(r-l+\sum_{i=L}^R c_i-L)^2$。求分段的最小代价。
> 
> $1\le n\le 5\times 10^4,1\le L,-10^7\le c_i\le 10^7$。

本题与「玩具装箱」问题唯一的区别是，玩具的价值可以为负。延续之前的思路，令 $f_i$ 表示前 $i$ 个物品，分若干段的最小代价。

状态转移方程：$f_i=\min_{j<i}\{f_j+(pre_i-pre_j+i-j-1-L)^2\}$。

其中 $pre_i = \sum_{j=1}^i c_j$。

将方程做相同的变换

$$
f_i - (s_i-L')^2=\min_{j<i}\{f_j+s_j^2 + 2s_j(L'-s_i) \} 
$$

然而这时有两个条件不成立了：

1. 直线的斜率不再单调；
2. 每次加入的决策点的横坐标不再单调。

仍然考虑凸壳的维护。

在寻找最优决策点，也就是用直线切凸壳的时候，我们将单调队列找队首改为：凸壳上二分。我们二分出斜率最接近直线斜率的那条凸壳边，就可以找到最优决策。

在加入决策点，也就是凸壳上加一个点的时候，我们有两种方法维护：

1. 直接用平衡树维护凸壳。那么寻找决策点的二分操作就转化为在平衡树上二分，插入决策点就转化为在平衡树上插入一个结点，并删除若干个被踢出凸壳的点。此方法思路简洁但实现繁琐。
2. 考虑 CDQ 分治。

$\text{CDQ}(l,r)$ 代表计算 $f_i,i\in [l,r]$。考虑 $\text{CDQ}(1,n)$：

- 我们先调用 $\text{CDQ}(1,mid)$ 算出 $f_i,i\in[1,mid]$。然后我们对 $[1,mid]$ 这个区间内的决策点建凸壳，然后使用这个凸壳去更新 $f_i,i\in [mid+1,n]$。这时我们决策点集是固定的，不像之前那样边计算 DP 值边加入决策点，那么我们就可以把 $i \in [mid+1,n]$ 的 $f_i$ 先按照直线的斜率 $k_i$ 排序，然后就可以使用单调队列来计算 DP 值了。当然，也可以在静态凸壳上二分计算 DP 值。

- 对于 $[mid+1,n]$ 中的每个点，如果它的最优决策的位置是在 $[1,mid]$ 这个区间，在这一步操作中他就会被更新成最优答案。当执行完这一步操作时，我们发现 $[1,mid]$ 中的所有点已经发挥了全部的作用，凸壳中他们存不存在已经不影响之后的答案更新。因此我们可以直接舍弃这个区间的决策点，并使用 $\text{CDQ}(mid+1,n)$ 解决右区间剩下的问题。

时间复杂度 $n\log^2 n$。

对比「玩具装箱」和「玩家装箱 改」，可以总结出以下两点：

- 二分/CDQ/平衡树等能够优化 DP 方程的计算，于一定程度上降低复杂度，但不能改变这个方程本身。
- DP 方程的性质会取决于数据的特征，但 DP 方程本身取决于题目中的数学模型。

## 小结

斜率优化 DP 需要灵活运用，其宗旨是将最优化问题转化为二维平面上与凸包有关的截距最值问题。遇到性质不太好的方程，有时需要辅以数据结构来加以解决，届时还请就题而论。

## 习题

- [「SDOI2016」征途](https://loj.ac/problem/2035)
- [「ZJOI2007」仓库建设](https://loj.ac/problem/10189)
- [「APIO2010」特别行动队](https://loj.ac/problem/10190)
- [「JSOI2011」柠檬](https://www.luogu.com.cn/problem/P5504)
- [「Codeforces 311B」Cats Transport](http://codeforces.com/problemset/problem/311/B)
- [「NOI2007」货币兑换](https://loj.ac/problem/2353)
- [「NOI2019」回家路线](https://loj.ac/problem/3156) => 【有点毒瘤 跳过等待更好的标准做法】
- [「NOI2016」国王饮水记](https://uoj.ac/problem/223)
- [「NOI2014」购票](https://uoj.ac/problem/7)

### 一维

> [!NOTE] **[AcWing 300. 任务安排1](https://www.acwing.com/problem/content/302/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性dp似乎就可以
> 
> **重要思想：分段对后续有影响的直接累加到本段计算**
> 
> $f_i = \min_{j=0}^{i-1}\{f_j + sumT_i * (sumC_i - sumC_j) + S * (sumC_n - sumC_j) \}$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 5010;

int n, s;
int sc[N], st[N];
int f[N];

int main() {
    scanf("%d%d", &n, &s);
    for (int i = 1; i <= n; i++) {
        scanf("%d%d", &st[i], &sc[i]);
        st[i] += st[i - 1];
        sc[i] += sc[i - 1];
    }

    memset(f, 0x3f, sizeof f);
    f[0] = 0;

    for (int i = 1; i <= n; i++)
        for (int j = 0; j < i; j++)
            f[i] =
                min(f[i], f[j] + (sc[i] - sc[j]) * st[i] + s * (sc[n] - sc[j]));

    printf("%d\n", f[n]);

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

> [!NOTE] **[AcWing 301. 任务安排2](https://www.acwing.com/problem/content/303/)**
> 
> 题意: 比上题数据范围更大

> [!TIP] **思路**
> 
> 斜率优化（凸包） 依据方程转移得新表达式
> 
> 对于从前至后的每一个点 i ，找到它前方的某固定斜率的最低 j ，随着节点加入凸包形成的斜率逐步增加。
> 
> 单调队列维护凸包，同时因为斜率单调递增，可以从队头删除。
> 
> $f_i = \min_{j=0}^{i-1}\{f_j + sumT_i * (sumC_i - sumC_j) + S * (sumC_n - sumC_j) \}$
> 
> => $f_i = f_j + sumT_i * sumC_i - sumT_i * sumC_j + S * sumC_n - S * sumC_j$
> 
> => $f_j = (sumT_i + S) * sumC_j + f_i - sumT_i * sumC_i - S * sumC_n$
> 
> 令 $y = f_j$, $x = sumC_j$
> 
> => $y = (sumT_i + S) * x + f_i - sumT_i * sumC_i - S * sumC_n$
> 
> - 对于当前位置 $i$，**斜率固定**的情况下要使得**截距最小**，显然找到第一个大于 $sumT_i + S$ 的斜率即可
> 
> - 又因为 $sumT_i$ 伴随着 $i$ 递增，故可以直接**头部弹出不符合要求的部分，因为以后也用不到**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 300010;

int n, s;
LL c[N], t[N];
LL f[N];
int q[N];

int main() {
    scanf("%d%d", &n, &s);
    for (int i = 1; i <= n; i++) {
        scanf("%lld%lld", &t[i], &c[i]);
        t[i] += t[i - 1];
        c[i] += c[i - 1];
    }

    int hh = 0, tt = 0;
    q[0] = 0;

    for (int i = 1; i <= n; i++) {
        // head < tail 至少两个元素时
        while (hh < tt && (f[q[hh + 1]] - f[q[hh]]) <=
                              (t[i] + s) * (c[q[hh + 1]] - c[q[hh]]))
            hh++;
        int j = q[hh];
        f[i] = f[j] - (t[i] + s) * c[j] + t[i] * c[i] + s * c[n];
        while (hh < tt &&
               (__int128)(f[q[tt]] - f[q[tt - 1]]) * (c[i] - c[q[tt - 1]]) >=
                   (__int128)(f[i] - f[q[tt - 1]]) * (c[q[tt]] - c[q[tt - 1]]))
            tt--;
        q[++tt] = i;
    }

    printf("%lld\n", f[n]);

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

> [!NOTE] **[AcWing 302. 任务安排3](https://www.acwing.com/problem/content/304/)**
> 
> 题意: 比上题出现负数

> [!TIP] **思路**
> 
> 单调队列维护凸包；但因为斜率并非单调递增，所以需要二分查找 j 。
> 
> 同理得 $f_j = (sumT_i + S) * sumC_j + f_i - sumT_i * sumC_i - S * sumC_n$
> 
> - 对于当前位置 $i$，**斜率固定**的情况下要使得**截距最小**，显然找到第一个大于 $sumT_i + S$ 的斜率即可
> 
> - 又因为 $sumT_i$ 并不伴随着 $i$ 递增，故**不能从头部弹出任何节点**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 300010;

int n, s;
LL t[N], c[N];
LL f[N];
int q[N];

int main() {
    scanf("%d%d", &n, &s);
    for (int i = 1; i <= n; i++) {
        scanf("%lld%lld", &t[i], &c[i]);
        t[i] += t[i - 1];
        c[i] += c[i - 1];
    }

    int hh = 0, tt = 0;
    q[0] = 0;

    for (int i = 1; i <= n; i++) {
        int l = hh, r = tt;
        while (l < r) {
            int mid = l + r >> 1;
            if (f[q[mid + 1]] - f[q[mid]] >
                (t[i] + s) * (c[q[mid + 1]] - c[q[mid]]))
                r = mid;
            else
                l = mid + 1;
        }

        int j = q[r];
        f[i] = f[j] - (t[i] + s) * c[j] + t[i] * c[i] + s * c[n];
        while (hh < tt &&
               (double)(f[q[tt]] - f[q[tt - 1]]) * (c[i] - c[q[tt - 1]]) >=
                   (double)(f[i] - f[q[tt - 1]]) * (c[q[tt]] - c[q[tt - 1]]))
            tt--;
        q[++tt] = i;
    }

    printf("%lld\n", f[n]);

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

> [!NOTE] **[AcWing 303. 运输小猫](https://www.acwing.com/problem/content/305/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有点复杂 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 100010, M = 100010, P = 110;

int n, m, p;
LL d[N], t[N], a[N], s[N];
LL f[P][M];
int q[M];

LL get_y(int k, int j) { return f[j - 1][k] + s[k]; }

int main() {
    scanf("%d%d%d", &n, &m, &p);

    for (int i = 2; i <= n; i++) {
        scanf("%lld", &d[i]);
        d[i] += d[i - 1];
    }

    for (int i = 1; i <= m; i++) {
        int h;
        scanf("%d%lld", &h, &t[i]);
        a[i] = t[i] - d[h];
    }

    sort(a + 1, a + m + 1);

    for (int i = 1; i <= m; i++) s[i] = s[i - 1] + a[i];

    memset(f, 0x3f, sizeof f);
    for (int i = 0; i <= p; i++) f[i][0] = 0;

    for (int j = 1; j <= p; j++) {
        int hh = 0, tt = 0;
        q[0] = 0;

        for (int i = 1; i <= m; i++) {
            while (hh < tt && (get_y(q[hh + 1], j) - get_y(q[hh], j)) <=
                                  a[i] * (q[hh + 1] - q[hh]))
                hh++;
            int k = q[hh];
            f[j][i] = f[j - 1][k] - a[i] * k + s[k] + a[i] * i - s[i];
            while (hh < tt &&
                   (get_y(q[tt], j) - get_y(q[tt - 1], j)) * (i - q[tt]) >=
                       (get_y(i, j) - get_y(q[tt], j)) * (q[tt] - q[tt - 1]))
                tt--;
            q[++tt] = i;
        }
    }

    printf("%lld\n", f[p][m]);

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

> [!NOTE] **[LeetCode 1776. 车队 II](https://leetcode-cn.com/problems/car-fleet-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 合并：后车撞前车，后车消失即可
> 
> **graham 维护凸包 (下凸壳)** 即可
> 
> 与此前 dp 的斜率优化略有不同:
> 
> - 此前是 `固定一个斜率求该斜率下的截距最值`
> 
> - 本题是 `固定一个点求限定条件下斜率的最值`
> 
> **也是单调栈的思想**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using PDD = pair<double, double>;
#define x first
#define y second

class Solution {
public:
    double cross(double x1, double y1, double x2, double y2) {
        return x1 * y2 - x2 * y1;
    }
    
    double area(PDD a, PDD b, PDD c) {
        return cross(b.x - a.x, b.y - a.y, c.x - a.x, c.y - a.y);
    }
    
    vector<double> getCollisionTimes(vector<vector<int>>& cars) {
        int n = cars.size();
        vector<PDD> stk(n + 1);
        vector<double> res(n);
        int top = 0;
        for (int i = n - 1; i >= 0; -- i ) {
            auto & c = cars[i];
            PDD p(c[0], c[1]);
            while (top >= 2 && area(p, stk[top], stk[top - 1]) <= 0) top -- ;
            if (!top) res[i] = -1;
            else {
                auto & q = stk[top];
                // <= 不能相遇
                if (p.y <= q.y) res[i] = -1;
                else res[i] = (q.x - p.x) / (p.y - q.y);
            }
            stk[ ++ top] = p;
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

> [!NOTE] **[Codeforces Kalila and Dimna in the Logging Industry](http://codeforces.com/problemset/problem/319/C)**
> 
> 题意: 
> 
> - 给定 $n$ 棵树，高度为 $a_1$ 到 $a_n$ ，权值为 $b_1$ 到 $b_n$，当一颗树的 $a$ 为 $0$ 时就说它被砍倒了
> 
> - 有一个电锯，每一次砍伐树会让一棵树的高度 $-1$
> 
> - 砍伐的费用为当前砍倒的编号的最大的树的 $b$【**不必前面连续的都被砍倒**】，第一次砍伐不需要费用
> 
> - 保证 $a_i<a_{i+1},b_i>b_{i+1},a_1=1,b_n=0$，求砍倒所有树的最小费用。
> 
> - 数据范围：$1\leqslant n\leqslant 10^5$。


> [!TIP] **思路**
> 
> 很显然，我们第一次砍的一定是第一棵树，且砍倒最后一棵树以后就不需要任何费用了，这样题意就转化为如何花费最小的费用砍倒第 $n$ 棵树。
> 
> 有一个很显然的贪心思想，我们砍树一定从前往后砍（**中间可以跳过一些树**），因为我们砍这棵树一定无法更新砍树的费用。
> 
> 设 $f_i$ 为砍倒第 $i$ 棵树的费用，那么可以列出转移方程 $f_i=\min_{j=1}^{i-1}\{f_j+b_j\cdot a_i\}$。
> 
> 原始 $O(n^2)$ 必然需要优化。
> 
> 考虑斜率优化：设 $j,k$ 两个 $i$ 的决策点满足 $k<j<i$，且 $j$ 比 $k$ 更优，那么有：
> 
> $$
> f_j+b_j\cdot a_i\leqslant f_k+b_k\cdot a_i
> $$
> 
> 变一下形式：
> 
> $$
> f_j+b_j\cdot a_i\leqslant f_k+b_k\cdot a_i
> $$
> $$
> \Rightarrow f_j-f_k\leqslant a_i\cdot(b_k-b_j)
> $$
> $$
> \Rightarrow \frac{f_j-f_k}{b_k-b_j}\leqslant a_i
> $$
> $$
> \Rightarrow -\frac{f_j-f_k}{b_j-b_k}\leqslant a_i
> $$
> $$
> \Rightarrow\frac{f_j-f_k}{b_j-b_k}\geqslant -a_i
> $$
> 
> 化成斜率式之后，由于 $a$ 是递增的，因此用单调队列维护一个上凸壳就好了

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Kalila and Dimna in the Logging Industry
// Contest: Codeforces - Codeforces Round #189 (Div. 1)
// URL: https://codeforces.com/problemset/problem/319/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10;
const static LL INF = 1e18;

int n;
// ai up, bi down
// a1=1, bn=0
LL a[N], b[N];
LL f[N];
int q[N], hh, tt;

inline LL x(int p) { return b[p]; }
inline LL y(int p) { return f[p]; }
inline double slope(int a, int b) {
    if (x(a) == x(b))
        return y(a) > y(b) ? INF : -INF;
    return 1.0 * (y(a) - y(b)) / (x(a) - x(b));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];
    for (int i = 1; i <= n; ++i)
        cin >> b[i];

    for (int i = 1; i <= n; ++i)
        f[i] = INF;

    hh = 0, tt = -1;
    q[++tt] = 0;
    f[1] = b[1];
    for (int i = 1; i <= n; ++i) {
        while (hh < tt && slope(q[hh + 1], q[hh]) >= -a[i])
            hh++;
        // f[i] = min(f[j] + bj*ai) ==> 1 <= j <= i-1
        f[i] = f[q[hh]] + b[q[hh]] * a[i];
        while (hh < tt && slope(q[tt - 1], i) >= slope(q[tt], q[tt - 1]))
            tt--;
        q[++tt] = i;
    }
    cout << f[n] << endl;

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

> [!NOTE] **[Luogu P2900 [USACO08MAR]Land Acquisition G](https://www.luogu.com.cn/problem/P2900)**
> 
> 题意: 
> 
> 考虑购买 $N$ 块长方形的土地
> 
> 可以选择并购一组土地，并购的价格为这些土地中最大的长乘以最大的宽。比如 FJ 并购一块 $3 \times 5$ 和一块 $5 \times 3$ 的土地，他只需要支付 $5 \times 5=25$ 元， 比单买合算。
> 
> 求购买所有土地所需的最小费用

> [!TIP] **思路**
> 
> - 细节: 注意 sorting 规则, 必须 `while`
> 
> - `long long`
> 
> 首先考虑排序预处理，最终得到 [w 递增, h 递减] 的序列
> 
> 进一步考虑最后一块并购了哪一段: $f[i] = min{f[j] + w[i] * h[j + 1]}$
> 
> => $f[j] = - w[i] * h[j + 1] + f[i]$
> 
> 要想 $f[i]$ 最小则截距最小，则需找到第一个大于 $-w[i]$ 的位置
> 
> 又因为 $w[i]$ 伴随 $i$ 增加，故每次从队头直接弹出不合法部分即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PLL = pair<LL, LL>;
const static int N = 5e4 + 10;

int n;
PLL xs[N];
LL f[N];
int q[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> xs[i].first >> xs[i].second;  // w, h
    {
        sort(xs + 1, xs + n + 1);
        int p = 1;
        // ATTENTION sort rule
        for (int i = 1; i <= n; ++i) {
            // ATTENTION 不能一个简单的 if-condition, 必须是 while 不断更新
            while (p && xs[p].second <= xs[i].second)
                p--;
            xs[++p] = xs[i];
        }
        n = p;
        // 剩下的都是  [w 递增, h 递减] 的序列
    }

    memset(f, 0x3f, sizeof f);
    f[0] = 0;

    // f[i] = min{f[j] + xs[i].first * xs[j + 1].second}
    // f[j] = -xs[i].first * xs[j + 1].second + f[i]
    // y = kx + b
    // f[i] 最小 => 即截距最小 => 下凸壳，斜率(<0且)越来越大
    //     类似 y = 1/x

    int hh = 0, tt = 0;
    q[0] = 0;
    for (int i = 1; i <= n; ++i) {
        // 去除较小的斜率 o1 找目标 j
        // while dy >= k * dx
        while (hh < tt && (f[q[hh + 1]] - f[q[hh]]) <=
                           -xs[i].first * (xs[q[hh + 1] + 1].second - xs[q[hh] + 1].second))
            hh++;
        int j = q[hh];
        f[i] = f[j] + xs[i].first * xs[j + 1].second;
        // while dy / dx <= dy / dx
        while (hh < tt && (f[q[tt]] - f[q[tt - 1]]) * (xs[i + 1].second - xs[q[tt] + 1].second) <=
                          (f[i] - f[q[tt]]) * (xs[q[tt] + 1].second - xs[q[tt - 1] + 1].second))
            tt--;
        q[++tt] = i;
    }

    cout << f[n] << endl;

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

> [!NOTE] **[Luogu P3195 [HNOI2008]玩具装箱](https://www.luogu.com.cn/problem/P3195)**
> 
> 题意: 
> 
> 有编号为 $1 \cdots n$ 的 $n$ 件玩具，第 $i$ 件玩具经过压缩后的一维长度为 $C_i$。
> 
> - 在同一个一维容器中的玩具编号是连续的。
> 
> - 如果将第 $i$ 件玩具到第 $j$ 个玩具放到一个容器中，那么容器的长度将为 $x=j-i+\sum\limits_{k=i}^{j}C_k$。
> 
> 制作容器的费用与容器的长度有关，如果容器长度为 $x$，其制作费用为 $(x-L)^2$。其中 $L$ 是一个常量。
> 
> 不关心容器的数目，他可以制作出任意长度的容器，甚至超过 $L$。但希望所有容器的总费用最小。求最小费用。

> [!TIP] **思路**
> 
> 巨坑: 必须要写 `slope` 使用除法，否则乘法会因为数值溢出
> 
> 或者写 `__int128`
> 
> $f[i] = min{f[j] + (s[i] - s[j] - 1 - L)^2}$
> 
> => $f[i] = min{f[j] + (s[i] - (s[j] + L + 1))^2}$
> 
> 令 $t(j) = s[j] + L + 1$
> 
> => $f[i] = f[j] + (s[i] - t(j))^2$
> 
> => $f[j] + t(j)^2 = 2*s[i]*t(j) + f[i] - s[i]^2$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PLL = pair<LL, LL>;
const static int N = 5e4 + 10;

LL n, L;
LL c[N], s[N];

int q[N];
LL f[N];

LL t(int v) { return s[v] + L + 1; }

LL x(int v) { return t(v); }

LL y(int v) { return f[v] + t(v) * t(v); }

double slope(int a, int b) { return (y(b) - y(a)) / (x(b) - x(a)); }

int main() {
    cin >> n >> L;

    for (int i = 1; i <= n; ++i)
        cin >> c[i];

    s[0] = 0;
    for (int i = 1; i <= n; ++i)
        s[i] = s[i - 1] + c[i] + 1;

    memset(f, 0x3f, sizeof f);
    f[0] = 0;

    // f[i] = min{f[j] + (s[i] - s[j] - 1 - L)^2}
    // f[i] = min{f[j] + (s[i] - (s[j] + L + 1))^2}
    // 令 t(j) = s[j] + L + 1
    // f[i] = f[j] + (s[i] - t(j))^2
    // f[j] + t(j)^2 = 2*s[i]*t(j) + f[i] - s[i]^2
    // f[i] 越小 => 截距越小 => s[i]是递增的
    // 下凸壳 形似 y=x^2
    int hh = 0, tt = 0;
    q[0] = 0;

    for (int i = 1; i <= n; ++i) {
        while (hh < tt && slope(q[hh], q[hh + 1]) <= 2ll * s[i])
            hh++;
        int j = q[hh];
        f[i] = f[j] + (s[i] - s[j] - 1 - L) * (s[i] - s[j] - 1 - L);
        // while (hh < tt &&
        //        (__int128)(y(q[tt]) - y(q[tt - 1])) * (x(i) - x(q[tt - 1])) >=
        //            (__int128)(y(i) - y(q[tt - 1])) * (x(q[tt]) - x(q[tt - 1])))
        while (hh < tt && slope(q[tt - 1], q[tt]) >= slope(i, q[tt - 1]))
            tt--;
        q[++tt] = i;
    }
    cout << f[n] << endl;

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

> [!NOTE] **[Luogu P3628 [APIO2010] 特别行动队](https://www.luogu.com.cn/problem/P3628)**
> 
> 题意: 
> 
> 有一支由 $n$ 名预备役士兵组成的部队，士兵从 $1$ 到 $n$ 编号，你要将他们拆分成若干特别行动队调入战场。出于默契的考虑，同一支特别行动队中队员的编号**应该连续**，即为形如 $(i, i + 1, \cdots i + k)$的序列。所有的队员都应该属于且仅属于一支特别行动队。
> 
> 编号为 $i$ 的士兵的初始战斗力为 $x_i$ ，一支特别行动队的初始战斗力 $X$ 为队内士兵初始战斗力之和，即 $X = x_i + x_{i+1} + \cdots + x_{i+k}$。
> 
> 对于一支初始战斗力为 $X$ 的特别行动队，其修正战斗力 $X'= aX^2+bX+c$，其中 $a,~b,~c$ 是已知的系数（$a < 0$）。 作为部队统帅，现在你要为这支部队进行编队，使得所有特别行动队的修正战斗力之和最大。试求出这个最大和。

> [!TIP] **思路**
> 
> 二元方程转换即可，注意右侧只留一元的部分，其余全部丢左边
> 
> 令 $t = s[i] - s[j]$
> 
> => $f[i] = max{f[j] + a*t^2 + b*t + c}$
> 
> => $f[i] = f[j] + a*(s[i]^2-2*s[i]*s[j]+s[j]^2) + b*(s[i]-s[j]) + c$
> 
> => $f[i] = f[j] + a*s[i]^2 + b*s[i] - (b+2*a*s[i])*s[j] + a*s[j]^2 + c$
> 
> => $f[j] + a*s[j]^2 = (2*a*s[i]+b)*s[j] - (a*s[i]+b)*s[i] - c + f[i]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e6 + 10;

LL n, a, b, c;
LL s[N];

int q[N];
LL f[N];

LL x(int i) { return s[i]; }

LL y(int i) { return f[i] + a * s[i] * s[i]; }

double slope(int a, int b) { return double(y(b) - y(a)) / (x(b) - x(a)); }

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> a >> b >> c;
    for (int i = 1; i <= n; ++i)
        cin >> s[i], s[i] += s[i - 1];

    // t = s[i] - s[j]
    // f[i] = max{f[j] + a*t^2 + b*t + c}
    // f[i] = f[j] + a*(s[i]^2-2*s[i]*s[j]+s[j]^2) + b*(s[i]-s[j]) + c
    // f[i] = f[j] + a*s[i]^2 + b*s[i] - (b+2*a*s[i])*s[j] + a*s[j]^2 + c
    // f[j] + a*s[j]^2 = (2*a*s[i]+b)*s[j] - (a*s[i]+b)*s[i] - c + f[i]
    // 要求 f[i] 最大，则截距最大 => 又因a是负数 s[i]正数 => 斜率越来越小 (上凸)

    memset(f, 0, sizeof f);

    int hh = 0, tt = 0;
    q[0] = 0;
    for (int i = 1; i <= n; ++i) {
        while (hh < tt && slope(q[hh], q[hh + 1]) >= 2ll * a * s[i] + b)
            hh++;
        int j = q[hh];
        LL t = s[i] - s[j];
        f[i] = f[j] + a * t * t + b * t + c;
        while (hh < tt && slope(q[tt - 1], q[tt]) <= slope(q[tt - 1], i))
            tt--;
        q[++tt] = i;
    }
    cout << f[n] << endl;

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

### 二维

> [!NOTE] **[Luogu P3648 [APIO2014] 序列分割](https://www.luogu.com.cn/problem/P3648)**
> 
> 题意: 
> 
> 一个关于长度为 $n$ 的非负整数序列。需要把序列分成 $k + 1$ 个非空的块。为了得到 $k + 1$ 块，你需要重复下面的操作 $k$ 次：
> 
> 选择一个有超过一个元素的块（初始时你只有一块，即整个序列）
> 
> 选择两个相邻元素把这个块从中间分开，得到两个非空的块。
> 
> 每次操作后你将获得那两个新产生的块的元素和的乘积的分数。你想要最大化最后的总得分。

> [!TIP] **思路**
> 
> 重要推论: **答案与切的顺序无关**
> 
> 有了推论基础才能写出状态转移方程 $f[i][k] = max(f[j][k-1] + (s[i]-s[j]) * s[j])$
> 
> 这类题有可能会 **卡精度 卡常**
> 
> $f[i][k] = max{f[j][k-1] + (s[i]-s[j]) * s[j]}$
> 
> 滚动数组压掉一维 => $f[i] = f[j] + (s[i]-s[j])*s[j]$
> 
> $f[j] - s[j]^2 = -s[i]*s[j] + f[i]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// ATTENTION: 重要推论

// 1. slope return 1e18
// 2. O2 优化 否则 10/11/12 TLE
// 3. scanf + inline 否则还是 T

using LL = long long;
const static int N = 1e5 + 10, M = 210;

int n, k;
LL s[N];

int q[N];
LL f[N], g[N];
int pre[N][M];

inline LL x(int i) { return s[i]; }

inline LL y(int i) {
    return g[i] - s[i] * s[i];
}  // ATTENTION g[i] instead of f[i]

inline double slope(int a, int b) {
    // 需要特判
    if (x(b) == x(a))
        return 1e18;  // 必须正无穷
                      // 否则wa最后一个点【但是本地跑最后一个点可以过...】
                      // 根据讨论区应该是卡精度了 建议 long double
    return double(y(b) - y(a)) / (x(b) - x(a));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    // cin >> n >> k;
    scanf("%d%d", &n, &k);
    s[0] = 0;
    for (int i = 1; i <= n; ++i)
        // cin >> s[i], s[i] += s[i - 1];
        scanf("%lld", &s[i]), s[i] += s[i - 1];

    // f[i][k] = max{f[j][k-1] + (s[i]-s[j]) * s[j]}
    // 滚动数组压掉一维 => f[i] = f[j] + (s[i]-s[j])*s[j]
    // f[j] - s[j]^2 = -s[i]*s[j] + f[i]
    // f[i] 最大 => 截距最大 => s[i]递增(斜率递减) => 上凸

    memset(f, 0, sizeof f), memset(g, 0, sizeof g);
    memset(pre, -1, sizeof pre);

    for (int _ = 1; _ <= k; ++_) {  // ATTENTION k次
        int hh = 0, tt = 0;
        q[0] = 0;
        for (int i = 1; i <= n; ++i) {
            while (hh < tt && slope(q[hh], q[hh + 1]) >= -s[i])
                hh++;
            int j = q[hh];
            f[i] = g[j] + (s[i] - s[j]) * s[j];
            pre[i][_] = j;  // ATTENTION
            while (hh < tt && slope(q[tt - 1], q[tt]) <= slope(q[tt - 1], i))
                tt--;
            q[++tt] = i;
        }
        memcpy(g, f, sizeof f);
    }
    /*
    cout << f[n] << endl;
    for (int j = k, i = n; j; i = pre[i][j], --j)
        cout << pre[i][j] << ' ';
    cout << endl;
    */
    printf("%lld\n", f[n]);
    for (int j = k, i = n; j; i = pre[i][j], --j)
        printf("%d ", pre[i][j]);
    printf("\n");

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

### 特殊公式推导

> [!NOTE] **[Luogu P4360 [CEOI2004] 锯木厂选址](https://www.luogu.com.cn/problem/P4360)**
> 
> 题意: 
> 
> 从山顶上到山底下沿着一条直线种植了$n$棵老树。当地的政府决定把他们砍下来。为了不浪费任何一棵木材，树被砍倒后要运送到锯木厂。 
> 
> 木材只能朝山下运。山脚下有一个锯木厂。另外两个锯木厂将新修建在山路上。你必须决定在哪里修建这两个锯木厂，使得运输的费用总和最小。假定运输每公斤木材每米需要一分钱。 
> 
> 你的任务是编写一个程序，从输入文件中读入树的个数和他们的重量与位置，计算最小运输费用。

> [!TIP] **思路**
> 
> 重点在于只开两个厂
> 
> - 枚举最高的工厂位置，则中间工厂的位置显然可以推导求最优 => 计算很麻烦，推翻
> 
> - 枚举中间工厂的位置，则左侧工厂的位置可以推导求最优 => 需使用 `全部运到山脚的总消耗` 去减优化的消耗
> 
> 重点在于对该推导过程进行斜率优化
> 
> **ATTENTION: 与标准斜率优化表达式不同, 必须通过两个值比较来推导 ==> 其实也可以转化为标准表达式, 【只是把其中一项 `d[j]*s[j]` 移动到左侧】**
> 
> TODO: **重复做 抽象出更具个人风格的流程**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 2e4 + 10;

int n;
LL w[N], d[N];
LL s[N];

int q[N];

double slope(int a, int b) {
    return double(d[b] * s[b] - d[a] * s[a]) / (s[b] - s[a]);
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> w[i] >> d[i];

    for (int i = n; i; --i)
        d[i] += d[i + 1];
    LL tot = 0;
    for (int i = 1; i <= n; ++i)
        s[i] = s[i - 1] + w[i], tot += d[i] * w[i];

    // 正向计算显然非常不好算，考虑逆向，用所有树运下山的总费用，减去中间建厂的费用减少量
    // s[j] 是正向的重量前缀和
    // 【ATTENTION: 中间的厂作为 i ，最上面的作为 j 这样好转移】
    // f[i] = min{tot - d[j]*s[j] - d[i]*(s[i]-s[j])}
    //          tot - 最左侧的减少量 - 中间的减少量
    // f[i] = -d[j]*s[j] - d[i]*s[i] + d[i]*s[j] + tot
    // f[i] = (d[i]-d[j])*s[j] - d[i]*s[i] + tot
    // f[i] 最小显然 （d[i]-d[j])*s[j] 最小
    // 【ATTENTION: 与传统斜率优化不同】
    //
    // 另一思路:
    // d[j]*s[j] = d[i]*s[j] - d[i]*s[i] + tot - f[i]
    // 则 斜率 d[i]
    // 也可以

    int hh = 0, tt = 0;
    q[0] = 0;
    LL res = 2e9;
    for (int i = 1; i <= n; ++i) {
        // ATTENTION 对 >= d[i] 的推导
        while (hh < tt && slope(q[hh], q[hh + 1]) >= d[i])
            hh++;
        int j = q[hh];
        res = min(res, tot - d[j] * s[j] - d[i] * (s[i] - s[j]));
        while (hh < tt && slope(q[tt - 1], q[tt]) <= slope(q[tt - 1], i))
            tt--;
        q[++tt] = i;
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

### 非单调栈维护

> [!NOTE] **[Luogu P4027 [NOI2007] 货币兑换](https://www.luogu.com.cn/problem/P4027)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO
> 
> splay / cdq

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

> [!NOTE] **[Luogu P2305 [NOI2014] 购票](https://www.luogu.com.cn/problem/P2305)**
> 
> 题意: 
> 
> 以 $1$ 为根共计 $n$ 个点的有根树，已知树边长。
> 
> 从 $v$ 前往 $1$ 的方法为：选择 $v$ 的一个祖先 $a$，支付购票的费用，乘坐交通工具到达 $a$。再选择城市 $a$ 的一个祖先 $b$，支付费用并到达 $b$。以此类推，直至到达 $1$。
> 
> 对于任意一个 $v$，我们会给出一个交通工具的距离限制 $l_v$。对于 $v$ 的祖先 A，只有当它们之间所有道路的总长度不超过 $l_v$  时，从 $v$ 才可以通过一次购票到达 A，否则不能通过一次购票到达。  
> 
> 对于每个 $v$，我们还会给出两个非负整数 $p_v,q_v$  作为票价参数。若城市 $v$ 到城市 A 所有道路的总长度为 $d$，那么从城市 $v$ 到城市 A 购买的票价为 $dp_v+q_v$。
> 
> 求用于购票的最少总资金

> [!TIP] **思路**
> 
> 树上斜率优化
> 
> TODO

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