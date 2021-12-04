前置知识: [单调队列](../../ds/monotonous-queue.md) 及 [单调栈](../../ds/monotonous-stack.md) 部分。

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

不了解背包 DP 的请先阅读 [背包 DP](../knapsack.md)。设 $f_{i,j}$ 表示前 $i$ 个物品装入承重为 $j$ 的背包的最大价值，朴素的转移方程为

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
