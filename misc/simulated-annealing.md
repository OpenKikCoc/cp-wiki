## 简介

模拟退火是一种随机化算法。当一个问题的方案数量极大（甚至是无穷的）而且不是一个单峰函数时，我们常使用模拟退火求解。

* * *

## 实现

根据 [爬山算法](./hill-climbing.md) 的过程，我们发现：对于一个当前最优解附近的非最优解，爬山算法直接舍去了这个解。而很多情况下，我们需要去接受这个非最优解从而跳出这个局部最优解，即为模拟退火算法。

> **什么是退火？**（选自百度百科）
>
> 退火是一种金属热处理工艺，指的是将金属缓慢加热到一定温度，保持足够时间，然后以适宜速度冷却。目的是降低硬度，改善切削加工性；消除残余应力，稳定尺寸，减少变形与裂纹倾向；细化晶粒，调整组织，消除组织缺陷。准确的说，退火是一种对材料的热处理工艺，包括金属材料、非金属材料。而且新材料的退火目的也与传统金属退火存在异同。

由于退火的规律引入了更多随机因素，那么我们得到最优解的概率会大大增加。于是我们可以去模拟这个过程，将目标函数作为能量函数。

### 模拟退火算法描述

先用一句话概括：如果新状态的解更优则修改答案，否则以一定概率接受新状态。

我们定义当前温度为 $T$，新状态与已知状态（由已知状态通过随机的方式得到）之间的能量（值）差为 $\Delta E$（$\Delta E\geqslant 0$），则发生状态转移（修改最优解）的概率为

$$
P(\Delta E)=
\begin{cases}
1&\text{新状态更优}\\
e^\frac{-\Delta E}{T}&\text{新状态更劣}
\end{cases}
$$

**注意**：我们有时为了使得到的解更有质量，会在模拟退火结束后，以当前温度在得到的解附近多次随机状态，尝试得到更优的解（其过程与模拟退火相似）。

### 如何退火（降温）？

模拟退火时我们有三个参数：初始温度 $T_0$，降温系数 $d$，终止温度 $T_k$。其中 $T_0$ 是一个比较大的数，$d$ 是一个非常接近 $1$ 但是小于 $1$ 的数，$T_k$ 是一个接近 $0$ 的正数。

首先让温度 $T=T_0$，然后按照上述步骤进行一次转移尝试，再让 $T=d\cdot T$。当 $T<T_k$ 时模拟退火过程结束，当前最优解即为最终的最优解。

注意为了使得解更为精确，我们通常不直接取当前解作为答案，而是在退火过程中维护遇到的所有解的最优值。

引用一张 [Wiki - Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) 的图片（随着温度的降低，跳跃越来越不随机，最优解也越来越稳定）。

![](./images/simulated-annealing.gif)

* * *

## 代码

此处代码以 [「BZOJ 3680」吊打 XXX](https://www.luogu.com.cn/problem/P1337)（求 $n$ 个点的带权类费马点）为例。


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

## 一些技巧

### 分块模拟退火

有时函数的峰很多，模拟退火难以跑出最优解。

此时可以把整个值域分成几段，每段跑一遍模拟退火，然后再取最优解。

### 卡时

有一个 `clock()` 函数，返回程序运行时间。

可以把主程序中的 `simulateAnneal();` 换成 `while ((double)clock()/CLOCKS_PER_SEC < MAX_TIME) simulateAnneal();`。这样子就会一直跑模拟退火，直到用时即将超过时间限制。

这里的 `MAX_TIME` 是一个自定义的略小于时限的数。

* * *

## 习题

- [「BZOJ 3680」吊打 XXX](https://www.luogu.com.cn/problem/P1337)
- [「JSOI 2016」炸弹攻击](https://loj.ac/problem/2076)
- [「HAOI 2006」均分数据](https://www.luogu.com.cn/problem/P2503)

> [!NOTE] **[LeetCode 1521. 找到最接近目标值的函数值](https://leetcode-cn.com/problems/find-a-value-of-a-mysterious-function-closest-to-target/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可以部分参考 [898. 子数组按位或操作](https://leetcode-cn.com/problems/bitwise-ors-of-subarrays/)
> 
> - 动态维护 TODO clear
> 
> - 模拟退火 TODO clear

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 动态维护**

```cpp
class Solution {
public:
    int closestToTarget(vector<int>& arr, int target) {
        int ans = abs(arr[0] - target);
        vector<int> valid = {arr[0]};
        for (int num : arr) {
            vector<int> validNew = {num};
            ans = min(ans, abs(num - target));
            for (int prev : valid) {
                validNew.push_back(prev & num);
                ans = min(ans, abs((prev & num) - target));
            }
            validNew.erase(unique(validNew.begin(), validNew.end()),
                           validNew.end());
            valid = validNew;
        }
        return ans;
    }
};
```

##### **C++ 模拟退火**

```cpp
class Solution {
public:
    //通过预处理，快速求解arr[L..R]的与值
    int pre[100001][20] = {0};

    int get(int L, int R, int target) {
        int val = 0;
        for (int i = 0, bit = 1; i < 20; i++, bit <<= 1)
            // 如果第 i 个bit 在 [L,R] 中全为 1，那么与值的该bit也必然为 1。
            if (pre[R][i] - pre[L - 1][i] == R - L + 1) { val |= bit; }
        return abs(val - target);
    }

    // 用模拟退火求解关于 L 的局部最优解
    int query(int L, int n, int target) {
        int dir[2] = {-1, 1};  // 两个方向
        int step = 1000;       // 初始步长
        int now = L;           // R 的起始位置
        int best = 100000000;  // 局部最优解

        while (step > 0) {
            int Lpos = now + step * dir[0];
            if (Lpos < L) Lpos = L;
            int Rpos = now + step * dir[1];
            if (Rpos > n) Rpos = n;
            // 向左右两个方向各走一步，求值
            int ldis = get(L, Lpos, target);
            int rdis = get(L, Rpos, target);
            int pbest = best;

            //更新位置及最优解
            if (ldis < best) {
                now = Lpos;
                best = ldis;
            }
            if (rdis < best) {
                now = Rpos;
                best = rdis;
            }

            //如果没有找到更优解，那就缩小步长
            if (pbest == best) { step /= 2; }
        }
        return best;
    }

    int closestToTarget(vector<int>& arr, int target) {
        int anw = 100000000;

        //统计前 i 个数字中，第 j 个bit 为 1 的数量。
        for (int i = 0; i < arr.size(); i++)
            for (int j = 0, bit = 1; j < 20; j++, bit <<= 1)
                pre[i + 1][j] = pre[i][j] + ((bit & arr[i]) ? 1 : 0);

        for (int i = 1; i <= arr.size(); i++)
            anw = min(anw, query(i, arr.size(), target));

        return anw;
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

> [!NOTE] **[LeetCode 1815. 得到新鲜甜甜圈的最多组数](https://leetcode-cn.com/problems/maximum-number-of-groups-getting-fresh-donuts/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 典型模拟退火 【数据范围不大 答案和顺序有关】
> 
> 状压也可 但需要对状态表示进行优化（编码）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 模拟退火**

```cpp
class Solution {
public:
    int n, m;
    vector<int> w;
    int res;
    
    int calc() {
        int ret = 1;
        for (int i = 0, s = 0; i < n; ++ i ) {
            s = (s + w[i]) % m;
            if (!s && i < n - 1)
                ret ++ ;
        }
        res = max(res, ret);
        return ret;
    }
    
    void simulate_anneal() {
        random_shuffle(w.begin(), w.end());
        for (double t = 1e6; t > 1e-5; t *= 0.97) {
            int a = rand() % n, b = rand() % n;
            int x = calc();
            swap(w[a], w[b]);
            int y = calc();
            int delta = x - y;
            if (!(exp(-delta / t) > (double)rand() / RAND_MAX))
                swap(w[a], w[b]);
        }
    }
    
    int maxHappyGroups(int batchSize, vector<int>& groups) {
        w = groups;
        n = w.size();
        m = batchSize;
        res = 0;
        for (int i = 0; i < 80; ++ i )
            simulate_anneal();
        return res;
    }
};
```

##### **C++ 状压**

```cpp
int c[10], d[10];
int pw[10];
int f[1000010];

class Solution {
public:
    int maxHappyGroups(int b, vector<int>& groups) {
        memset(c, 0, sizeof c);
        memset(f, 0, sizeof f);
        
        for (auto v : groups)
            c[v % b] ++ ;

        // 编码        
        int mx = 1;
        pw[0] = 1;
        for (int i = 0; i < b; ++ i ) {
            mx *= (c[i] + 1);
            pw[i + 1] = pw[i] * (c[i] + 1);
        }
        
        int res = 0;
        f[0] = 1;
        for (int i = 1; i < mx; ++ i ) {
            int x = i, s = 0;
            for (int j = 0; j < b; ++ j ) {
                // 映射
                d[j] = x % (c[j] + 1);
                x /= (c[j] + 1);

                if (d[j] > 0)
                    f[i] = max(f[i], f[i - pw[j]]);
                // d[j]频次 j值
                s = (s + d[j] * j) % b;
            }
            res = max(res, f[i]);
            if (s == 0)
                f[i] ++ ;
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