> [!NOTE] **注意**
> 
> 单调队列在实际应用中往往需要根据推导对形式进行转换，转换后的代码实现逻辑会与转换前有非常大的不同
> 
> 所以 **一定要梳理清楚形式转换和维护的逻辑**
> 
> 典型的代码实现与暴力实现差异极大的形如:
> 
> - 单调队列优化多重背包
> 
> - 推导转化单调推列维护 (比如 [Luogu P3089 [USACO13NOV]Pogo-Cow S(https://www.luogu.com.cn/problem/P3089))

前置知识: [单调队列](ds/monotonous-queue.md) 及 [单调栈](ds/monotonous-stack.md) 部分。

> [!NOTE] **例题[CF372C Watching Fireworks is Fun](http://codeforces.com/problemset/problem/372/C)**
> 
> 题目大意：城镇中有 $n$ 个位置，有 $m$ 个烟花要放。第 $i$ 个烟花放出的时间记为 $t_i$，放出的位置记为 $a_i$。如果烟花放出的时候，你处在位置 $x$，那么你将收获 $b_i-|a_i-x|$ 点快乐值。
> 
> 初始你可在任意位置，你每个单位时间可以移动不大于 $d$ 个单位距离。现在你需要最大化你能获得的快乐值。

设 $f_{i,j}$ 表示在放第 $i$ 个烟花时，你的位置在 $j$ 所能获得的最大快乐值。

写出 **状态转移方程**：$f_{i,j}=\max\{f_{i-1,k}+b_i-|a_i-j|\}$

这里的 $k$ 是有范围的，$j-(t_{i}-t_{i-1})\times d\le k\le j+(t_{i}-t_{i-1})\times d$。

我们尝试将状态转移方程进行变形：

由于 $\max$ 里出现了一个确定的常量 $b_i$，我们可以将它提到外面去。

$f_{i,j}=\max\{f_{i-1,k}+b_i-|a_i-j|\}=\max\{f_{i-1,k}-|a_i-j|\}+b_i$

如果确定了 $i$ 和 $j$ 的值，那么 $|a_i-j|$ 的值也是确定的，也可以将这一部分提到外面去。

最后，式子变成了这个样子：$f_{i,j}=\max\{f_{i-1,k}-|a_i-j|\}+b_i=\max\{f_{i-1,k}\}-|a_i-j|+b_i$

看到这一熟悉的形式，我们想到了什么？**单调队列优化**。由于最终式子中的 $\max$ 只和上一状态中连续的一段的最大值有关，所以我们在计算一个新的 $i$ 的状态值时候只需将原来的 $f_{i-1}$ 构造成一个单调队列，并维护单调队列，使得其能在均摊 $O(1)$ 的时间复杂度内计算出 $\max\{f_{i-1,k}\}$ 的值，从而根据公式计算出 $f_{i,j}$ 的值。

总的时间复杂度为 $O(nm)$。

> [!TIP]+ 参考代码

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

讲完了，让我们归纳一下单调队列优化动态规划问题的基本形态：当前状态的所有值可以从上一个状态的某个连续的段的值得到，要对这个连续的段进行 RMQ 操作，相邻状态的段的左右区间满足非降的关系。

## 单调队列优化多重背包

> [!NOTE] **问题描述**
> 
> 你有 $n$ 个物品，每个物品重量为 $w_i$，价值为 $v_i$，数量为 $k_i$。你有一个承重上限为 $m$ 的背包，现在要求你在不超过重量上限的情况下选取价值和尽可能大的物品放入背包。求最大价值。

不了解背包 DP 的请先阅读 [背包 DP](knapsack.md)。设 $f_{i,j}$ 表示前 $i$ 个物品装入承重为 $j$ 的背包的最大价值，朴素的转移方程为

$$
f_{i,j}=\max_{k=0}^{k_i}(f_{i-1,j-k\times w_i}+v_i\times k)
$$

时间复杂度 $O(m\sum k_i)$。

考虑优化 $f_i$ 的转移。为方便表述，设 $g_{x,y}=f_{i,x\times w_i+y},g'_{x,y}=f_{i-1,x\times w_i+y}$，则转移方程可以表示为：

$$
g_{x,y}=\max_{k=0}^{k_i}(g'_{x-k,y}+v_i\times k)
$$

设 $G_{x,y}=g'_{x,y}-v_i\times x$。则方程可以表示为：

$$
g_{x,y}=\max_{k=0}^{k_i}(G_{x-k,y})+v_i\times x
$$

这样就转化为一个经典的单调队列优化形式了。$G_{x,y}$ 可以 $O(1)$ 计算，因此对于固定的 $y$，我们可以在 $O\left( \left\lfloor \dfrac{W}{w_i} \right\rfloor \right)$ 的时间内计算出 $g_{x,y}$。因此求出所有 $g_{x,y}$ 的复杂度为 $O\left( \left\lfloor \dfrac{W}{w_i} \right\rfloor \right)\times O(w_i)=O(W)$。这样转移的总复杂度就降为 $O(nW)$。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// AcWing 6. 多重背包问题 III
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 20010;

int n, m;
int f[N], g[N], q[N];

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; i ++ ) {
        int v, w, s;
        cin >> v >> w >> s;
        // 实际上并不需要二维的dp数组
        // 可以重复利用dp数组来保存上一轮的信息
        memcpy(g, f, sizeof f);
        // 取模范围 0~v 【共 v 个单调队列，每个队列队首维护取模为 j 的最大值下标（递减队列）】
        for (int j = 0; j < v; j ++ ) {
            int hh = 0, tt = -1;
            // 下面for循环 单调队列 本质求最大
            for (int k = j; k <= m; k += v) {
                // 维护队列元素个数 不能继续入队 弹出队头
                // （去除当前 k 下的不合法最值）
                // 本题 队列元素个数为 s+1, 0~s 个物品
                if (hh <= tt && q[hh] < k - s * v) hh ++ ;
                // 维护单调性，尾值 <= 当前元素
                while (hh <= tt && g[q[tt]] - (q[tt] - j) / v * w <= g[k] - (k - j) / v * w) tt -- ;
                // 【每次入队元素对应的值是 f[i-1][j+kv]-kw，这里公式的 k 对应代码中的 (x-j)/v】
                // 1.
                q[ ++ tt] = k;
                f[k] = g[q[hh]] + (k - q[hh]) / v * w;
                // 2. 如果入队写最后需要取max
                // if(head <= tail) f[k] = max(f[k], pre[q[head]] + (k-q[head])/v*w);
                // q[++tail] = k;
            }
        }
    }

    cout << f[m] << endl;

    return 0;
}
```

##### **Python**

```python
# 状态表示：f[i,j] 所有只能从前i个物品，体积为j的选法的集合
# 优化：完全背包问题：可以优化成所有前缀的最大值；多重背包问题：求滑动窗口内的最大值

if __name__ == "__main__":
    n, m = map(int, input().split())
    N = n + 1
    M = m + 1

    v = [0] * N
    w = [0] * N
    s = [0] * N

    for i in range(1, N):
        a, b, c = map(int, input().split())
        v[i] = a
        w[i] = b
        s[i] = c

    f = [[0] * M for i in range(N)]
    q = [0] * 20010

    for i in range(1, N):
        for j in range(v[i]):
            hh = 0
            tt = -1
            for k in range((m - j) // v[i] + 1):
                while hh <= tt and k - q[hh] > s[i]:
                    hh += 1
                while hh <= tt and f[i - 1][j + q[tt] * v[i]] - q[tt] * w[i] < f[i - 1][j + k * v[i]] - k * w[i]:
                    tt -= 1

                tt += 1
                q[tt] = k

                f[i][j + k * v[i]] = f[i - 1][j + q[hh] * v[i]] - q[hh] * w[i] + k * w[i]

    print(f[n][m])
```

<!-- tabs:end -->
</details>

<br>

* * *


## 习题

[「Luogu P1886」滑动窗口](https://loj.ac/problem/10175)

[「NOI2005」瑰丽华尔兹](https://www.luogu.com.cn/problem/P2254)

[「SCOI2010」股票交易](https://loj.ac/problem/10183)

### 单调队列

> [!NOTE] **[AcWing 1087. 修剪草坪](https://www.acwing.com/problem/content/1089/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 记忆
> 
> 变种 方程变形转化为求窗口内最值 进而使用单调队列

<details>
<summary>详细代码</summary>

$f[i]$ 表示前 $i$ 头牛 符合条件的最大

对于使用 $i$ 牛的情况：

$f[i] = max(f[i-j-1] + s[i] - s[i-j]);  1 <= j <= k$ 因为题目要求不超过k都可

对于不适用i牛的情况： $f[i] = f[i-1]$

<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n, m;
LL s[N];
LL f[N];
int q[N];

LL g(int i) {
    if (!i) return 0;
    return f[i - 1] - s[i];
}

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++) {
        scanf("%lld", &s[i]);
        s[i] += s[i - 1];
    }

    int hh = 0, tt = 0;
    for (int i = 1; i <= n; i++) {
        if (q[hh] < i - m) hh++;
        f[i] = max(f[i - 1], g(q[hh]) + s[i]);
        while (hh <= tt && g(q[tt]) <= g(i)) tt--;
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

> [!NOTE] **[AcWing 1088. 旅行问题](https://www.acwing.com/problem/content/1090/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 问题转化为求区间最值 以及思考入队顺序

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

const int N = 2e6 + 10;

int n;
int oil[N], dist[N];
LL s[N];
int q[N];
bool ans[N];

int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) {
        scanf("%d%d", &oil[i], &dist[i]);
        s[i] = s[i + n] = oil[i] - dist[i];
    }
    for (int i = 1; i <= n * 2; i++) s[i] += s[i - 1];

    int hh = 0, tt = 0;
    q[0] = n * 2 + 1;
    for (int i = n * 2; i >= 0; i--) {
        if (q[hh] > i + n) hh++;
        if (i < n) {
            if (s[i] <= s[q[hh]]) ans[i + 1] = true;
        }
        while (hh <= tt && s[q[tt]] >= s[i]) tt--;
        q[++tt] = i;
    }

    dist[0] = dist[n];
    for (int i = 1; i <= n; i++) s[i] = s[i + n] = oil[i] - dist[i - 1];
    for (int i = 1; i <= n * 2; i++) s[i] += s[i - 1];

    hh = 0, tt = 0;
    q[0] = 0;
    for (int i = 1; i <= n * 2; i++) {
        if (q[hh] < i - n) hh++;
        if (i > n) {
            if (s[i] >= s[q[hh]]) ans[i - n] = true;
        }
        while (hh <= tt && s[q[tt]] <= s[i]) tt--;
        q[++tt] = i;
    }

    for (int i = 1; i <= n; i++)
        if (ans[i])
            puts("TAK");
        else
            puts("NIE");

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

> [!NOTE] **[AcWing 1090. 绿色通道](https://www.acwing.com/problem/content/1092/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 注意窗口和间隔定义

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 50010, INF = 1e9;

int n, m;
int w[N];
int f[N], q[N];

bool check(int k) {
    f[0] = 0;
    int hh = 0, tt = 0;
    for (int i = 1; i <= n; i++) {
        if (hh <= tt && q[hh] < i - k - 1) hh++;
        f[i] = f[q[hh]] + w[i];
        while (hh <= tt && f[q[tt]] >= f[i]) tt--;
        q[++tt] = i;
    }

    int res = INF;
    for (int i = n - k; i <= n; i++) res = min(res, f[i]);

    return res <= m;
}

int main() {
    scanf("%d%d", &n, &m);

    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);

    int l = 0, r = n;
    while (l < r) {
        int mid = l + r >> 1;
        if (check(mid))
            r = mid;
        else
            l = mid + 1;
    }

    printf("%d\n", r);

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

> [!NOTE] **[AcWing 1091. 理想的正方形](https://www.acwing.com/problem/content/1093/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **二维窗口**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 1010, INF = 1e9;

int n, m, k;
int w[N][N];
int row_min[N][N], row_max[N][N];
int q[N];

void get_min(int a[], int b[], int tot) {
    int hh = 0, tt = -1;
    for (int i = 1; i <= tot; i++) {
        if (hh <= tt && q[hh] <= i - k) hh++;
        while (hh <= tt && a[q[tt]] >= a[i]) tt--;
        q[++tt] = i;
        b[i] = a[q[hh]];
    }
}

void get_max(int a[], int b[], int tot) {
    int hh = 0, tt = -1;
    for (int i = 1; i <= tot; i++) {
        if (hh <= tt && q[hh] <= i - k) hh++;
        while (hh <= tt && a[q[tt]] <= a[i]) tt--;
        q[++tt] = i;
        b[i] = a[q[hh]];
    }
}

int main() {
    scanf("%d%d%d", &n, &m, &k);

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) scanf("%d", &w[i][j]);

    for (int i = 1; i <= n; i++) {
        get_min(w[i], row_min[i], m);
        get_max(w[i], row_max[i], m);
    }

    int res = INF;
    int a[N], b[N], c[N];
    for (int i = k; i <= m; i++) {
        for (int j = 1; j <= n; j++) a[j] = row_min[j][i];
        get_min(a, b, n);

        for (int j = 1; j <= n; j++) a[j] = row_max[j][i];
        get_max(a, c, n);

        for (int j = k; j <= n; j++) res = min(res, c[j] - b[j]);
    }

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

> [!NOTE] **[LeetCode 1687. 从仓库到码头运输箱子](https://leetcode-cn.com/problems/delivering-boxes-from-storage-to-ports/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 单调队列优化dp 注意细节
> 
> **重点在于如何通过公式转化并得到单调性质**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    vector<int> s;
    
    // (l, r] 有多少个不同段
    int cost(int l, int r) {
        // l + 1 是分界点
        if (s[l] != s[l + 1]) return s[r] - s[l];
        return s[r] - s[l] + 1;
    }
    
    int boxDelivering(vector<vector<int>>& boxes, int portsCount, int maxBoxes, int maxWeight) {
        n = boxes.size();
        s.resize(n + 2);
        for (int i = 1; i <= n; ++ i ) {
            s[i] = s[i - 1];
            // 是分段的起始
            if (i == 1 || boxes[i - 1][0] != boxes[i - 2][0]) ++ s[i] ;
        }
        
        vector<int> f(n + 1);
        deque<int> q;
        q.push_back(0);
        for (int i = 1, j = 1, w = 0; i <= n; ++ i ) {
            w += boxes[i - 1][1];
            while (w > maxWeight || i - j + 1 > maxBoxes) {
                w -= boxes[j - 1][1];
                ++ j ;
            }
            while (q.front() < j - 1)
                q.pop_front();
            
            int k = q.front();
            f[i] = f[k] + cost(k, i) + 1;
            
            // (i, i + 1] 与 (q.back(), i] 无论右区间取哪里都是一样的 所以直接用 i+1
            while (!q.empty() && f[q.back()] >= f[i] + cost(i, i + 1) - cost(q.back(), i + 1))
                q.pop_back();
            q.push_back(i);
        }
        return f[n];
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

> [!NOTE] **[LeetCode 1696. 跳跃游戏 VI](https://leetcode-cn.com/problems/jump-game-vi/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单单调队列优化dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int maxResult(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> f(n + 1);
        vector<int> q(n + 1);
        int hh = 0, tt = -1;
        f[0] = 0;
        for (int i = 1; i <= n; ++ i ) {
            if (hh <= tt && q[hh] < i - k) ++ hh;
            f[i] = f[q[hh]] + nums[i - 1];
            while (hh <= tt && f[q[tt]] <= f[i]) -- tt;
            q[ ++ tt] = i;
        }
        return f[n];
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

> [!NOTE] **[Luogu P4852 yyf hates choukapai](https://www.luogu.com.cn/problem/P4852)**
> 
> 题意: 
> 
> 在抽每张卡时欧气值都是固定的，第 $i$ 张卡的欧气值为 $a_i$ ，而在连抽时，欧气值等于第一张卡的欧气值。
> 
> “每次抽卡的欧气之和”指每次单抽的欧气之和加上每次连抽的欧气之和，一次连抽的欧气不加权，只计算一次
> 
> yyf想 $c$ 连抽（连续抽 $c$ 张卡） $n$ 次，单抽 $m$ 次，因为一直单抽太累，**yyf不想连续单抽超过 $d$ 次（可以连续单抽恰好 $d$ 次）**。
> 
> 共有 $c*n+m$ 张卡，抽卡的顺序不能改变，每张卡都必须且只能抽一次，只能改变哪几张卡连抽、哪几张卡单抽。那么yyf每次抽卡的欧气之和最多能达到多少，又如何实现呢？

> [!TIP] **思路**
> 
> 代码实现形式发生转换
> 
> 注意维护队头和队尾的细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// ATTENTION 对于前面i张牌，一旦连抽的次数确定，则单抽次数确定

const static int N = 200010, M = 41;

int n, m, c, d, tot;

int a[N], s[N];
int f[N][M], pre[N][M];

int q[N];

int get(int k, int j) { return f[k][j - 1] + a[k + 1] - s[k + c]; }
void print(int i, int j) {
    if (!j)
        return;
    print(pre[i][j], j - 1);
    cout << pre[i][j] + 1 << ' ';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m >> c >> d;

    tot = c * n + m;

    for (int i = 1; i <= tot; ++i) {
        cin >> a[i];
        s[i] = s[i - 1] + a[i];
    }

    memset(f, 0xcf, sizeof f);
    for (int i = 1; i <= d; ++i)
        f[i][0] = s[i];
    f[0][0] = 0;

    // f[i][j] = max{f[k][j - 1] + a[k+1] + s[i] - s[k + c]}
    // 其中 max(0,i-c-d) <= k <= i-c
    // 显然先枚举第二维，滚动第一维
    for (int j = 1; j <= n; ++j) {  // j = 0 时已被初始化
        int hh = 0, tt = -1;

        for (int i = j * c; i <= tot; ++i) {
            while (hh <= tt && q[hh] < i - c - d)
                hh++;
            // ATTENTION 注意维护队尾的操作 放在取值之前
            while (hh <= tt && get(q[tt], j) <= get(i - c, j))
                tt--;
            q[++tt] = i - c;
            if (get(q[hh], j) + s[i] > f[i][j])
                f[i][j] = get(q[hh], j) + s[i], pre[i][j] = q[hh];
        }
    }

    cout << f[tot][n] << endl;
    print(tot, n);
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

### 单调队列 转化形式

> [!NOTE] **[Luogu P3089 [USACO13NOV]Pogo-Cow S](https://www.luogu.com.cn/problem/P3089)**
> 
> 题意: 
> 
> $Betty$ 在一条直线的 $N(1\leq N\leq 1000)$ 个目标点上跳跃，目标点i在目标点 $x(i)$，得分为 $p(i)$。$Betty$ 可选择<strong>任意一个目标点上</strong>，<strong>只朝一个方向跳跃</strong>，且每次跳跃距离<strong>成不下降序列</strong>。
> 
> 每跳到一个目标点，$Betty$ 可以拿到该点的得分，求最大总得分。

> [!TIP] **思路**
> 
> 推导和形式转换

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// https://www.luogu.com.cn/blog/gzw2005/luogup3089-usaco13novpogo-di-niu-pogo-cow

using PII = pair<int, int>;
const static int N = 1010, INF = 0x3f3f3f3f;

#define x first
#define s second

int n;
PII xs[N];

int f[N][N];  // f[i][j] 表示当前位置 i 上一个位置 j 的最大得分

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> xs[i].x >> xs[i].s;
    // 按位置从左到右排序
    sort(xs + 1, xs + n + 1);

    // f[i][j] = max{f[j][k] + s[i]}
    //  => f[i][j] = max{f[j][k]} + s[i]
    // 【其中 i,j,k 的位置有约束，需满足  x(j) - x(k) <= x(i) - x(j)】
    // 考虑: f[i-1][j] = max{f[j][k]} + s[i-1]
    // 则有【注意: 非严格等于, 可以取到的范围比f[i-1][j]更大】:
    //  => f[i][j] = max{f[i-1][j] - s[i-1] + s[i], 以及更大的一个部分}
    // 为什么会范围更大?
    //   => 因为 x(i) > x(i-1)，而上式对应 x(j) - x(k) <= x(i-1) - x(j)

    int res = 0;

    for (int j = 1; j <= n; ++j) {
        f[j][j] = xs[j].s;  // 边界
        int maxv = -INF;
        for (int i = j + 1, k = j; i <= n; ++i) {
            while (k >= 1 && xs[j].x - xs[k].x <= xs[i].x - xs[j].x)
                maxv = max(maxv, f[j][k]), k--;
            // ATTENTION 这种写法不需要再 -xs[i-1].s
            f[i][j] = maxv + xs[i].s;
            res = max(res, f[i][j]);
        }
        res = max(res, f[j][j]);
    }

    for (int j = n; j >= 1; --j) {
        f[j][j] = xs[j].s;
        int maxv = -INF;
        for (int i = j - 1, k = j; i >= 1; --i) {
            while (k <= n && xs[k].x - xs[j].x <= xs[j].x - xs[i].x)
                maxv = max(maxv, f[j][k]), k++;
            f[i][j] = maxv + xs[i].s;
            res = max(res, f[i][j]);
        }
        res = max(res, f[j][j]);
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


> [!NOTE] **[Luogu P5665 [CSP-S2019] 划分](https://www.luogu.com.cn/problem/P5665)** [TAG]
> 
> 题意: 
> 
> 长度为 $n$ 的序列分为任意段，要求每段的数值和都大于等于前面的一段，每段有其消耗为数值和的平方
> 
> 求最小总消耗

> [!TIP] **思路**
> 
> 状态转移约束上类似上一题 [Luogu P3089 [USACO13NOV]Pogo-Cow S](https://www.luogu.com.cn/problem/P3089)
> 
> **较裸的状态转移**
> 
> 考虑到划分的段是递增的，设 $dp[i][j]$ 为划分到第 $i$ 个的前驱是 $j$，转移：
> 
> $$
> dp[i][j]=dp[j][k]+(sum[i]-sum[j])^2
> $$
> 
> 判一下递增，暴力转移时间复杂度 $O(n^3)$
> 
> **单调队列优化**
> 
> 可以固定 $j$，发现在移动 $i$ 的过程中，$k$ 也在移动，满足一个单调性
> 
> 故可以维护一个 $dp[j][k]$ 的最小值，时间复杂度 $O(n^2)$
> 
> 仍然不能通过本题的数据范围
> 
> **结合贪心优化**
> 
> 注意到：在合法的情况下，$dp[i][j]\leq dp[i][j-1]$
> 
> 所以最右边的前驱是最优的，然后我们不考虑更新，而考虑转移点（即对应位置 $i$ 对应的一段由哪一个位置 $j$ `pre[i]` 转移过来）
> 
> 可以维护一个单调队列
> 
> - 队头维护转移点，如果满足约束则尽量向右移动
> 
> - 队尾维护最值 $s[j] * 2 - s[pre[j]]$ **(注意推导细节)**
> 
> 时间复杂度 $O(n)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 重要性质: 考虑最后的答案，最优解必然满足最后一段最小

using LL = long long;
const static int N = 4e7 + 10, M = 1e5 + 10, MOD = 1 << 30;

int n, type;
int a[N], b[N], p[M], l[M], r[M];

LL s[N], f[N];     // 以i结尾的最优方案
int q[N], pre[N];  // ATTENTION: pre同时记录以i结尾的最小段；与前述重要性质关联

void print(__int128 x) {
    if (x == 0)
        return;
    print(x / 10);
    cout << int(x % 10);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> type;
    if (type) {
        int x, y, z, m;
        cin >> x >> y >> z >> b[1] >> b[2] >> m;
        for (int i = 1; i <= m; ++i)
            cin >> p[i] >> l[i] >> r[i];
        for (int i = 3; i <= n; ++i)
            b[i] = ((LL)b[i - 1] * x + (LL)b[i - 2] * y + z) % MOD;
        for (int i = 1; i <= m; ++i)
            for (int j = p[i - 1] + 1; j <= p[i]; ++j) {
                a[j] = (b[j] % (r[i] - l[i] + 1)) + l[i];
                s[j] = s[j - 1] + a[j];
            }
    } else {
        for (int i = 1; i <= n; ++i) {
            cin >> a[i];
            s[i] = s[i - 1] + a[i];
        }
    }

    // ATTENTION 要非常注意此处我们仍然使用的是闭区间，但是预先加入 0 【一段可能的最左起始位置显然0】
    // 以及 在后续 while 循环中需要保证至少两个数，所以是 < 而不是 <=
    int hh = 0, tt = -1;
    q[++tt] = 0;
    for (int i = 1; i <= n; ++i) {
        // ATTENTION hh < tt instead of hh <= tt
        // 在满足区间和约束的情况下，如果可以往后走，就一直往后
        // 对应 get(j) = s*s[j] - pre[s[j]] <= s[i]
        while (hh < tt &&
               s[q[hh + 1]] - s[pre[q[hh + 1]]] <= s[i] - s[q[hh + 1]])
            hh++;
        pre[i] = q[hh];     // i 位置本段左侧的转移点为 q[hh]
        // 注意 维护的时候包含两部分值
        // s[j] - s[pre[j]] + s[j] >= s[i] - s[pre[i]] + s[i]
        // 为什么两边又加了一遍对应位置的 s 值？
        // ==> pre[i] 为 max_j 的 s[i] - s[j] >= s[j] - s[pre[j]]
        // ==> 即为 s[i] >= s[j] * 2 - s[pre[j]]
        //  故 维护 s[j] * 2 - s[pre[j]] 的最值即可
        // 对应 get(q[tt]) >= get(i)
        while (hh < tt && s[q[tt]] - s[pre[q[tt]]] + s[q[tt]] >= s[i] - s[pre[i]] + s[i])
            tt--;
        q[++tt] = i;
    }

    __int128 res = 0;
    for (int p = n; p; p = pre[p]) {
        __int128 t = s[p] - s[pre[p]];
        res += t * t;
    }
    print(res);

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


> [!NOTE] **[Luogu P3522 [POI2011]TEM-Temperature](https://www.luogu.com.cn/problem/P3522)** [TAG]
> 
> 题意: 
> 
> 对每一天的天气都给出上界和下界，求最长可能有多少个连续天的温度不下降

> [!TIP] **思路**
> 
> 重点在于 **连续天**，连续则可以进一步思考一段合法的连续段有何性质
> 
> luogu 的题解描述非常有歧义
> 
> 应当是：一段合法连续段必须满足【**本连续段内: 任意位置作为末尾，该位置的上界一定要大于等于前面的所有位置的下界**】
> 
> 则单调队列维护下界的最大值 (思考) ，并在维护过程中记录 res 即可
> 
> **重要: 问题转化为单调队列的思想**
> 
> **TODO: 感觉单调栈也可以做？**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const static int N = 1e6 + 10;

int n;
int l[N], r[N];

int q[N];  // 递减的序列

int main() {
    cin >> n;

    for (int i = 1; i <= n; ++i)
        cin >> l[i] >> r[i];

    int res = 0;
    int hh = 0, tt = -1;
    for (int i = 1; i <= n; ++i) {
        while (hh <= tt && l[q[hh]] > r[i])
            hh++;

        while (hh <= tt && l[q[tt]] <= l[i])
            tt--;
        q[++tt] = i;

        // cout << " hh = " << hh << " tt = " << tt << endl;
        // cout << " q[tt] = " << q[tt] << " q[hh-1] = " << q[hh - 1] << endl;

        // ATTENTION: 这里用 q[hh-1] 因为 q[hh-1] 是往左数第一个不符合条件的位置
        // 细节 思考
        res = max(res, q[tt] - q[hh - 1]);
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

> [!NOTE] **[Luogu P4544 [USACO10NOV]Buying Feed G](https://www.luogu.com.cn/problem/P4544)**
> 
> 题意: 
> 
> 见题目链接

> [!TIP] **思路**
> 
> 列公式 => 拆分 => 变量和常量 => 单调队列维护变量最值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 510, M = 10010;

struct Node {
    LL x, f, c;
    bool operator<(const Node& t) { return x < t.x; }
} xs[N];

int k, e, n;

LL f[N][M];  // 走过了前i个商店 当前拥有饲料j的最小花费
int q[M];    // 注意是 M 不是 N, 因为单调队列存的是第二维

LL get(int i, int k) {
    return f[i - 1][k] + (xs[i].x - xs[i - 1].x) * k * k - xs[i].c * k;
}

int main() {
    cin >> k >> e >> n;

    for (int i = 1; i <= n; ++i)
        cin >> xs[i].x >> xs[i].f >> xs[i].c;
    sort(xs + 1, xs + n + 1);

    // f[i][j] = min(f[i - 1][j],
    // 				 f[i - 1][k] + k * k * (x[i] - x[i - 1]) + (j - k) * c[i])
    // f[i][j] = min{
    //	f[i - 1][k] + k * k * (x[i] - x[i - 1]) - k * c[i] + j * c[i] }
    // 其中 k 显然有限制:  0<=j-k<=f[i]
    // 				=> k <= j && k >= j-f[i]
    // 重点在于在避免枚举的情况下求得对应的 k 和 f 值

    memset(f, 0x3f, sizeof f);
    f[0][0] = 0;

    for (int i = 1; i <= n; ++i) {
        int hh = 0, tt = -1;
        for (int j = 0; j <= k; ++j) {
            // 排除 k(q[hh]) 不合法的部分
            while (hh <= tt && q[hh] < j - xs[i].f)
                hh++;
            while (hh <= tt && get(i, q[tt]) >= get(i, j))
                tt--;
            q[++tt] = j;
            f[i][j] = get(i, q[hh]) + xs[i].c * j;
        }
    }
    cout << f[n][k] + ((LL)e - xs[n].x) * k * k << endl;

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

### 决策单调性优化

> TODO: https://www.luogu.com.cn/training/9352 Part 4.9 斜率优化动态规划

> [!NOTE] **[Luogu P3515 [POI2011]Lightning Conductor](https://www.luogu.com.cn/problem/P3515)** TODO
> 
> 题意: 
> 
> 给定一个长度为 $n$ 的序列 $\{a_n\}$，对于每个 $i\in [1,n]$ ，求出一个最小的非负整数 $p$ ，使得 $\forall j\in[1,n]$，都有 $a_j\le a_i+p-\sqrt{|i-j|}$
> 
> $1 \le n \le 5\times 10^{5}$，$0 \le a_i \le 10^{9}$

> [!TIP] **思路**
> 
> 意即任意两个元素之间都满足该不等式
> 
> 考虑顺序无关，直接找最小的和最大的
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

> [!NOTE] **[Luogu P4767 [IOI2000]邮局](https://www.luogu.com.cn/problem/P4767)**
> 
> 题意: TODO

> [!TIP] **思路**
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

> [!NOTE] **[Luogu P1973 [NOI2011] NOI 嘉年华](https://www.luogu.com.cn/problem/P1973)** [TAG]
> 
> 题意: 
> 
> 给出 $n$ 个区间, 让你把这些区间分成两份, 允许丢弃, 两份区间不能有交
> 
> Q1: 让两份中分到区间数最小的一份, 尽量得到更多的区间
> 
> Q2: 在第 $∀i$ 个区间必须不丢弃的情况下的上一问答案

> [!TIP] **思路**
> 
> 显然可以定义 `前 i 个时刻(已离散化)第一份选择了 j 个 此时第二份最多能选择多少个`
> 
> Q1 可以直接由上述状态维护计算得出
> 
> 考虑 Q2 需要想到**类似前后缀分解的思路**: 枚举中间一段必选(假定直接给第一份) 在此基础上再枚举前面和后面第一份各选了多少个
> 
> => 显然枚举复杂度 $O(n^4)$ 较高，**进一步考虑【左边选的越多，右边相应一定选的越少】利用单调性提前 break 降低真实复杂度**
> 
> 非常好的题 细节很多

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
using LL = long long;
const static int N = 410, INF = 0x3f3f3f3f;

int n;
int s[N], t[N];
vector<int> xs;
int get(int x) { return lower_bound(xs.begin(), xs.end(), x) - xs.begin() + 1; }

int c[N][N];  // 统计在某个时间段内的活动有多少个

// 前i个时刻(已离散化)第一份选择了j个 此时第二份最多能选择多少个
int f[N][N];
// 后i个时刻 第一份选择了j个 此时第二份最多能选择多少个
int g[N][N];
// 前面选到i 后面从j开始选 中间都给第一个时
// 【二者在一起最少的那个】最多能选多少个
int p[N][N];

int main() {
    cin >> n;

    xs.clear();
    for (int i = 0; i < n; ++i) {
        cin >> s[i] >> t[i];
        t[i] += s[i];
        xs.push_back(s[i]), xs.push_back(t[i]);
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());

    int m = xs.size();

    for (int i = 1; i <= m; ++i)
        for (int j = i; j <= m; ++j) {
            int l = xs[i - 1], r = xs[j - 1], cnt = 0;
            for (int k = 0; k < n; ++k)
                if (s[k] >= l && t[k] <= r)
                    cnt++;
            c[i][j] = cnt;
        }

    memset(f, 0xcf, sizeof f);  // -INF
    f[1][0] = 0;                // ATTENTION 注意转移
    for (int i = 1; i <= m; ++i)
        for (int j = 0; j <= n; ++j)
            // 枚举本段开始位置位置【最终形成的段不包含i处对应的时间点 细节】
            for (int k = 1; k <= i; ++k) {
                int x = c[k][i];
                // 最后一段放到第一份
                if (j >= x)
                    f[i][j] = max(f[i][j], f[k][j - x]);
                // 最后一段放到第二份
                f[i][j] = max(f[i][j], f[k][j] + x);  // 注意状态转移方程
            }

    {
        // Q1
        int res = 0;
        for (int i = 1; i <= n; ++i)
            res = max(res, min(f[m][i], i));
        cout << res << endl;
    }

    // Q2
    // 要求中间某个必选，则可以按照这个必选的区间作为中间部分，分别求左右侧选取的不同情况
    // 则 必然需要反向求后缀的 f 数组，假定命名其为 g

    // 求g
    memset(g, 0xcf, sizeof g);
    g[m][0] = 0;  // 边界
    for (int i = m; i >= 1; --i)
        for (int j = 0; j <= n; ++j)
            for (int k = i; k <= m; ++k) {
                int x = c[i][k];
                if (j >= x)
                    g[i][j] = max(g[i][j], g[k][j - x]);
                g[i][j] = max(g[i][j], g[k][j] + x);
            }

    // 求s
    for (int i = 1; i <= m; ++i)
        for (int j = i + 1; j <= m; ++j) {
            // ATTENTION 只能在这里初始化，对于其他部分都应该是0
            p[i][j] = -INF;
            // 左侧第一个选x个 右侧第一个选y个
            for (int x = 0; x <= n; ++x) {
                // ATTENTION 左边选的越多 右边能选的越少
                // 使用 break 排除一些方案【思考】
                if (f[i][x] < 0)
                    break;
                for (int y = 0; y + x <= n; ++y) {
                    if (g[j][y] < 0)
                        break;
                    // 第二个的 第一个的
                    int t = min(f[i][x] + g[j][y], x + y + c[i][j]);
                    // cout << " t = " << t << ' ';
                    p[i][j] = max(p[i][j], t);
                }
            }
        }

    for (int x = 1; x <= n; ++x) {
        int res = 0;
        for (int i = 1; i <= get(s[x - 1]); ++i)
            for (int j = get(t[x - 1]); j <= m; ++j) {
                res = max(res, p[i][j]);
            }

        cout << res << endl;
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

> [!NOTE] **[Luogu P3724 [AHOI2017/HNOI2017]大佬](https://www.luogu.com.cn/problem/P3724)**
> 
> 题意: TODO

> [!TIP] **思路**
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

> [!NOTE] **[Luogu P5574 [CmdOI2019]任务分配问题](https://www.luogu.com.cn/problem/P5574)**
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

### 单调栈 主要结合其他优化