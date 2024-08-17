## 介绍

`std::bitset` 是标准库中的一个存储 `0/1` 的大小不可变容器。严格来讲，它并不属于 STL。

> [!TIP] **bitset 与 STL**
> 
> `bitset` 并不属于 STL，而是一种标准库中的 "Special Container"。事实上，它作为一种容器，也并不满足 STL 容器的要求。说它是适配器，它也并不依赖于其它 STL 容器作为底层实现。

`bitset` 就是通过固定的优化，使得一个字节的八个比特能分别储存 8 位的 `0/1`。

`vector` 的一个特化 `vector<bool>` 的储存方式同 `bitset` 一样，区别在于其支持动态开空间，`bitset` 则和我们一般的静态数组一样，是在编译时就开好了的。

然而，`bitset` 有一些好用的库函数，不仅方便，而且有时可以避免使用 for 循环而没有实质的速度优化。因此，一般不使用 `vector<bool>`。

## 使用（来自 oi-wiki 待学习）

### 头文件

```cpp
#include <bitset>
```

### 指定大小

```cpp
bitset<1000> bs;  // a bitset with 1000 bits
```

### 构造函数

- `bitset()`: 每一位都是 `false`。
- `bitset(unsigned long val)`: 设为 `val` 的二进制形式。
- `bitset(const string& str)`: 设为 $01$ 串 `str`。

### 运算符

- `operator []`: 访问其特定的一位。
- `operator ==/!=`: 比较两个 `bitset` 内容是否完全一样。
- `operator &/&=/|/| =/^/^=/~`: 进行按位与/或/异或/取反操作。**`bitset` 只能与 `bitset` 进行位运算**，若要和整型进行位运算，要先将整型转换为 `bitset`。
- `operator <</>>/<<=/>>=`: 进行二进制左移/右移。
- `operator <</>>`: 流运算符，这意味着你可以通过 `cin/cout` 进行输入输出。

### 成员函数

- `count()`: 返回 `true` 的数量。
- `size()`: 返回 `bitset` 的大小。
- `test(pos)`: 它和 `vector` 中的 `at()` 的作用是一样的，和 `[]` 运算符的区别就是越界检查。
- `any()`: 若存在某一位是 `true` 则返回 `true`，否则返回 `false`。
- `none()`: 若所有位都是 `false` 则返回 `true`，否则返回 `false`。
- `all()`:**C++11**，若所有位都是 `true` 则返回 `true`，否则返回 `false`。
-   1. `set()`: 将整个 `bitset` 设置成 `true`。
    2. `set(pos, val = true)`: 将某一位设置成 `true`/`false`。
-   1. `reset()`: 将整个 `bitset` 设置成 `false`。
    2. `reset(pos)`: 将某一位设置成 `false`。相当于 `set(pos, false)`。
-   1. `flip()`: 翻转每一位。（$0\leftrightarrow1$，相当于异或一个全是 $1$ 的 `bitset`）
    2. `flip(pos)`: 翻转某一位。
- `to_string()`: 返回转换成的字符串表达。
- `to_ulong()`: 返回转换成的 `unsigned long` 表达 (`long` 在 NT 及 32 位 POSIX 系统下与 `int` 一样，在 64 位 POSIX 下与 `long long` 一样）。
- `to_ullong()`:**C++11**，返回转换成的 `unsigned long long` 表达。

一些文档中没有的成员函数：

- `_Find_first()`: 返回 `bitset` 第一个 `true` 的下标，若没有 `true` 则返回 `bitset` 的大小。
- `_Find_next(pos)`: 返回 `pos` 后面（下标严格大于 `pos` 的位置）第一个 `true` 的下标，若 `pos` 后面没有 `true` 则返回 `bitset` 的大小。

## 应用

### [「LibreOJ β Round #2」贪心只能过样例](https://loj.ac/problem/515)

1. dp ：

> $f(i,j)$ 表示前 $i$ 个数的平方和能否为 $j$，那么 $f(i,j)=\bigvee\limits_{k=a}^bf(i-1,j-k^2)$（或起来）。

但如果直接做的话是 $O(n^5)$ 的，（看起来）过不了。

1. `bitset` 优化

   左移再或起来：[std::bitset](https://loj.ac/submission/395274)

3. [剪枝的暴力](https://loj.ac/submission/395673)

4. 手写 `bitset`

（只需要支持左移后或起来这一种操作）压 $64$ 位（`unsigned long long`）：[手写 bitset](https://loj.ac/submission/395619)

### [CF1097F Alex and a TV Show](https://codeforces.com/contest/1097/problem/F)

#### 题意

给你 $n$ 个可重集，四种操作：

1. 把某个可重集设为一个数。
2. 把某个可重集设为另外两个可重集加起来。
3. 把某个可重集设为从另外两个可重集中各选一个数的 $\gcd$。即：$A=\{\gcd(x,y)|x\in B,y\in C\}$。
4. 询问某个可重集中某个数的个数，**在模 2 意义下**。

可重集个数 $10^5$，操作个数 $10^6$，值域 $7000$。

#### 做法

看到“在模 $2$ 意义下”，可以想到用 `bitset` 维护每个可重集。

这样的话，操作 $1$ 直接设，操作 $2$ 就是异或（因为模 $2$），操作 $4$ 就是直接查，但 .. 操作 $3$ 怎么办？

我们可以尝试维护每个可重集的所有约数构成的可重集，这样的话，操作 $3$ 就是直接按位与。

我们可以把值域内每个数的约数构成的 `bitset` 预处理出来，这样操作 $1$ 就解决了。操作 $2$ 仍然是异或。

现在的问题是，如何通过一个可重集的约数构成的可重集得到该可重集中某个数的个数。

令原可重集为 $A$，其约数构成的可重集为 $A'$，我们要求 $A$ 中 $x$ 的个数，用 [莫比乌斯反演](math/number-theory/mobius.md) 推一推：

$$
\begin{aligned}&\sum\limits_{i\in A}[\frac i x=1]\\=&\sum\limits_{i\in A}\sum\limits_{d|\frac i x}\mu(d)\\=&\sum\limits_{d\in A',x|d}\mu(\frac d x)\end{aligned}
$$

由于是模 $2$ 意义下，$-1$ 和 $1$ 是一样的，只用看 $\frac d x$ 有没有平方因子即可。所以，可以对值域内每个数预处理出其倍数中除以它不含平方因子的位置构成的 `bitset`，求答案的时候先按位与再 `count()` 就好了。

这样的话，单次询问复杂度就是 $O(\frac v w)$（$v=7000,\,w=32$）。

至于预处理的部分，$O(v\sqrt v)$ 或者 $O(v^2)$ 预处理比较简单，$\log$ 预处理就如下面代码所示，复杂度为调和级数，所以是 $O(v\log v)$。


>
> ```cpp
> #include <bitset>
> #include <cctype>
> #include <cmath>
> #include <cstdio>
> #include <iostream>
> 
> using namespace std;
> 
> int read() {
>     int out = 0;
>     char c;
>     while (!isdigit(c = getchar()))
>         ;
>     for (; isdigit(c); c = getchar()) out = out * 10 + c - '0';
>     return out;
> }
> 
> const int N = 100005;
> const int M = 1000005;
> const int V = 7005;
> 
> bitset<V> pre[V], pre2[V], a[N], mu;
> int n, m, tot;
> char ans[M];
> 
> int main() {
>     int i, j, x, y, z;
> 
>     n = read();
>     m = read();
> 
>     mu.set();
>     for (i = 2; i * i < V; ++i) {
>         for (j = 1; i * i * j < V; ++j) { mu[i * i * j] = 0; }
>     }
>     for (i = 1; i < V; ++i) {
>         for (j = 1; i * j < V; ++j) {
>             pre[i * j][i] = 1;
>             pre2[i][i * j] = mu[j];
>         }
>     }
> 
>     while (m--) {
>         switch (read()) {
>             case 1:
>                 x = read();
>                 y = read();
>                 a[x] = pre[y];
>                 break;
>             case 2:
>                 x = read();
>                 y = read();
>                 z = read();
>                 a[x] = a[y] ^ a[z];
>                 break;
>             case 3:
>                 x = read();
>                 y = read();
>                 z = read();
>                 a[x] = a[y] & a[z];
>                 break;
>             case 4:
>                 x = read();
>                 y = read();
>                 ans[tot++] = ((a[x] & pre2[y]).count() & 1) + '0';
>                 break;
>         }
>     }
> 
>     printf("%s", ans);
> 
>     return 0;
> }
> ```


### 与埃氏筛结合

由于 `bitset` 快速的连续读写效率，使得它非常适合用于与埃氏筛结合打质数表。

使用的方式也很简单，只需要将埃氏筛中的布尔数组替换成 `bitset` 即可。

> [!TIP] **速度测试**

| 算法              | $5 \times 10^7$ | $10^8$ | $5 \times 10^8$ |
| ----------------- | --------------- | ------ | --------------- |
| 埃氏筛 + 布尔数组 | 386ms           | 773ms  | 4.41s           |
| 欧拉筛 + 布尔数组 | 257ms           | 521ms  | 2.70s           |
| 埃氏筛 +`bitset`  | 219ms           | 492ms  | 2.66s           |
| 欧拉筛 +`bitset`  | 332ms           | 661ms  | 3.21s           |

> 从测试结果中可知，时间复杂度 $O(n \log \log n)$ 的埃氏筛在使用 `bitset` 优化后速度甚至超过时间复杂度 $O(n)$ 的欧拉筛，而欧拉筛在使用 `bitset` 后会出现「负优化」的情况。




>
> ```cpp
> bitset<N> vis;
> void Prime(int n) {
>     vis.set();
>     vis[0] = vis[1] = 0;
>     for (int i = 2; i * i <= n; i++) {
>         if (vis[i]) {
>             for (int j = i << 1; j <= n; j += i) vis[j] = 0;
>         }
>     }
> }
> ```


### 与树分块结合

`bitset` 与树分块结合可以解决一类求树上多条路径信息并的问题，详见 [数据结构/树分块](ds/tree-decompose.md)。

### 与莫队结合

详见 [杂项/莫队配合 bitset](misc/mo-algo-with-bitset.md)。

### 计算高维偏序

详见 [FHR 课件](https://github.com/OI-wiki/libs/blob/master/lang/csl/FHR-分块bitset求高维偏序.pdf)。


## 习题

> [!NOTE] **[LeetCode 187. 重复的DNA序列](https://leetcode.cn/problems/repeated-dna-sequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> bitset 做法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<string> findRepeatedDnaSequences_2(string s) {
        vector<string> res;
        unordered_map<string, int> mp;
        for (int i = 0; i + 10 <= s.size(); ++ i ) {
            string str = s.substr(i, 10);
            if (mp[str] == 1)
                res.push_back(str);
            ++ mp[str] ;
        }
        return res;
    }
};
```

##### **C++ bitset**

```cpp
class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s) {
        unordered_map<string, int> cnt;
        for (int i = 0; i + 10 <= s.size(); i ++ )
            cnt[s.substr(i, 10)] ++ ;
        vector<string> res;
        for (auto [s, c]: cnt)
            if (c > 1)
                res.push_back(s);
        return res;
    }
};
```

##### **Python**

```python
# 用set去重(竟然时间和内存占用都要优些)
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        visited = set()
        res = set()
        for i in range(0, len(s) - 9):
            tmp = s[i:i+10]
            if tmp in visited:
                res.add(tmp)
            visited.add(tmp)
        return list(res)  
      
# 用hash + list
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:        
        my_hash = collections.defaultdict(int)
        for i in range(len(s) - 9):
            my_hash[s[i:i+10]] += 1
        res = []
        for key, val in my_hash.items():
            if val > 1:
                res.append(key)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> bitset优化 可拓展

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // bitset
    bool canPartition(vector<int>& nums) {
        bitset<10001> f;
        f[0] = 1;
        int sum = 0;
        for (auto x: nums) {
            f |= f << x;
            sum += x;
        }
        if (sum % 2) return false;
        return f[sum / 2];
    }
};

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n = nums.size(), m = 0;
        for (auto x: nums) m += x;
        if (m % 2) return false;
        m /= 2;
        vector<int> f(m + 1);
        f[0] = 1;
        for (auto x: nums)
            for (int j = m; j >= x; j -- )
                f[j] |= f[j - x];
        return f[m];
    }
};
```

##### **Python**

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # 两个子集的和要想等，说明nums之和应该为偶数，且每个子集的和应该是sum(nums)/2
        if sum(nums) % 2 == 1 or len(nums) == 1:
            return False
        # 这样就转化为了：从nums中任选数字，这些数字之和要等于sum(nums)/2，那么剩下的就是另一个子集
        # 这就是一个01背包问题，背包容积是sum(nums)/2,从N=len(nums)中选，每个物品的体积是nums[i]，问能否恰好装满这个包
        # 闫氏dp分析法
        # 状态表示：dp[i][j]表示遍历到第i位时是否可凑出target。
        # 状态计算：第i个物品装或不装：dp[i][j] = dp[i-1][j-nums[i]]装 or dp[i-1][j]不装,有一个为True就说明在前i个数中是存在符合要求的方案的
        # 注意：j = 0时初始化为True，从定义出发，j=0说明sum(nums)/2==0，说明sum(num)==0,那么相当于全0数组分割，当然是True
        # N, V = len(nums), int(sum(nums)/2)
        # 注意这里j也要开多一位，因为需要取到V
        # dp = [[False] * (V + 1) for _ in range(N+1)]
        # for i in range(N+1):
        #     dp[i][0] = True
        # nums = [0] + nums
        # for i in range(1, N+1):
        #     for j in range(V+1):
        #         if j < nums[i]:
        #             dp[i][j] = dp[i-1][j]
        #         else:
        #             dp[i][j] = dp[i-1][j-nums[i]] or dp[i-1][j]
        # return dp[N][V]
        # 状态压缩
        N, V = len(nums), int(sum(nums)/2)
        dp = [False] * (V + 1)
        dp[0] = True
        nums = [0] + nums
        for num in nums:
            for j in range(V, num-1, -1):
                dp[j] = dp[j-num] or dp[j]
        return dp[V]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1452. 收藏清单](https://leetcode.cn/problems/people-whose-list-of-favorite-companies-is-not-a-subset-of-another-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 多组字符串，返回不是其他组子集的组。
> 
> 排序后暴力即可
> 
> bitset可以优化 因为每一个字符串都不同 可以对字符串编码

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> peopleIndexes(vector<vector<string>>& a) {
        int n = a.size();
        unordered_map<string, int> vis;
        int vcnt = 0;  // 1~vcnt 闭区间
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < a[i].size(); ++j)
                if (!vis[a[i][j]]) vis[a[i][j]] = ++vcnt;
        bitset<50500> f[110];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < a[i].size(); ++j) f[i][vis[a[i][j]]] = 1;
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            bool flag = false;
            for (int j = 0; j < n; ++j)
                if (j != i) {
                    if ((f[i] & f[j]) == f[i]) {
                        flag = true;
                        break;
                    }
                }
            if (!flag) res.push_back(i);
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

> [!NOTE] **[LeetCode 3181. 执行操作可获得的最大总奖励 II](https://leetcode.cn/problems/maximum-total-reward-using-operations-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 根据题意及数据范围推导 值一定不超过 1e5 => 想到 bitset
> 
> 消除 bitset 部分 0 的姿势

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 根据题意理解及数据范围 总和一定不会超过 max(rewardValues)*2 => 思考
    // 可以用 bitset 快速维护
    const static int N = 5e4 + 10, M = 1e5 + 10;
    
    bitset<M> f;
    
    int maxTotalReward(vector<int>& rewardValues) {
        int n = rewardValues.size();
        sort(rewardValues.begin(), rewardValues.end());
        
        f[0] = 1;
        for (int i = 0; i < n; ++ i ) {
            int x = rewardValues[i];
            // ATTENTION 重点在于 约束只看较低的 x 位
            // for (int j = 0; j < x; ++ j )
            //     f[j + x] = f[j + x] || f[j];
            
            // ATTENTION: trick
            // 先左移 d 消除前面的 0, 再右移 d 恢复, 最后左移 x 实现状态转移
            int d = M - x;
            // f |= f << d >> d << x;
        }
        
        for (int j = M - 1; j >= 0; -- j )
            if (f[j])
                return j;
        return -1;
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