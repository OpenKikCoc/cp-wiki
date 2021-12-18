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
```

<!-- tabs:end -->
</details>

<br>

* * *


## 习题

[「Luogu P1886」滑动窗口](https://loj.ac/problem/10175)

[「NOI2005」瑰丽华尔兹](https://www.luogu.com.cn/problem/P2254)

[「SCOI2010」股票交易](https://loj.ac/problem/10183)

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
<!-- tabs:start -->

$f[i]$ 表示前 $i$ 头牛 符合条件的最大

对于使用 $i$ 牛的情况：

$f[i] = max(f[i-j-1] + s[i] - s[i-j]);  1 <= j <= k$ 因为题目要求不超过k都可

对于不适用i牛的情况： $f[i] = f[i-1]$

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