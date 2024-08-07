## 简介

RMQ 是英文 Range Maximum/Minimum Query 的缩写，表示区间最大（最小）值。

在笔者接下来的描述中，默认初始数组大小为 $n$。

在笔者接下来的描述中，默认时间复杂度标记方式为 $O($ 数据预处理 $) \sim O($ 单次询问 $)$。

## 单调栈

由于 **OI Wiki** 中已有此部分的描述，本文仅给出 [链接](ds/monotonous-stack.md)。这部分不再展开。

时间复杂度 $O(m\log m) \sim O(\log n)$

空间复杂度 $O(n)$

## ST 表

由于 **OI Wiki** 中已有此部分的描述，本文仅给出 [链接](ds/sparse-table.md)。这部分不再展开。

时间复杂度 $O(n\log n) \sim O(1)$

空间复杂度 $O(n\log n)$

## 线段树

由于 **OI Wiki** 中已有此部分的描述，本文仅给出 [链接](ds/seg.md)。这部分不再展开。

时间复杂度 $O(n) \sim O(\log n)$

空间复杂度 $O(n)$

## Four Russian

Four russian 是一个由四位俄罗斯籍的计算机科学家提出来的基于 ST 表的算法。

在 ST 表的基础上 Four russian 算法对其做出的改进是序列分块。

具体来说，我们将原数组——我们将其称之为数组 A——每 $S$ 个分成一块，总共 $n/S$ 块。

对于每一块我们预处理出来块内元素的最小值，建立一个长度为 $n/S$ 的数组 B，并对数组 B 采用 ST 表的方式预处理。

同时，我们对于数组 A 的每一个零散块也建立一个 ST 表。

询问的时候，我们可以将询问区间划分为不超过 1 个数组 B 上的连续块区间和不超过 2 个数组 A 上的整块内的连续区间。显然这些问题我们通过 ST 表上的区间查询解决。

在 $S=\log n$ 时候，预处理复杂度达到最优，为 $O((n / \log n)\log n+(n / \log n)\times\log n\times\log \log n)=O(n\log \log n)$。

时间复杂度 $O(n\log \log n) \sim O(1)$

空间复杂度 $O(n\log \log n)$

当然询问由于要跑三个 ST 表，该实现方法的常数较大。

> [!TIP] **一些小小的算法改进**
> 
> 我们发现，在询问的两个端点在数组 A 中属于不同的块的时候，数组 A 中块内的询问是关于每一块前缀或者后缀的询问。
> 
> 显然这些询问可以通过预处理答案在 $O(n)$ 的时间复杂度内被解决。
> 
> 这样子我们只需要在询问的时候进行至多一次 ST 表上的查询操作了。

> [!TIP] **一些玄学的算法改进**

由于 Four russian 算法以 ST 表为基础，而算法竞赛一般没有非常高的时间复杂度要求，所以 Four russian 算法一般都可以被 ST 表代替，在算法竞赛中并不实用。这里提供一种在算法竞赛中更加实用的 Four russian 改进算法。
    
我们将块大小设为 $\sqrt n$，然后预处理出每一块内前缀和后缀的 RMQ，再暴力预处理出任意连续的整块之间的 RMQ，时间复杂度为 $O(n)$。
    
查询时，对于左右端点不在同一块内的询问，我们可以直接 $O(1)$ 得到左端点所在块的后缀 RMQ，左端点和右端点之间的连续整块 RMQ，和右端点所在块的前缀 RMQ，答案即为三者之间的最值。
    
而对于左右端点在同一块内的询问，我们可以暴力求出两点之间的 RMQ，时间复杂度为 $O(\sqrt n)$，但是单个询问的左右端点在同一块内的期望为 $O(\frac{\sqrt n}{n})$，所以这种方法的时间复杂度为期望 $O(n)$。
    
而在算法竞赛中，我们并不用非常担心出题人卡掉这种算法，因为我们可以通过在 $\sqrt n$ 的基础上随机微调块大小，很大程度上避免算法在根据特定块大小构造的数据中出现最坏情况。并且如果出题人想要卡掉这种方法，则暴力有可能可以通过。
    
这是一种期望时间复杂度达到下界，并且代码实现难度和算法常数均较小的算法，因此在算法竞赛中比较实用。
    
以上做法参考了 [P3793 由乃救爷爷](https://www.luogu.com.cn/problem/P3793) 中的题解。

## 加减 1RMQ

若序列满足相邻两元素相差为 1，在这个序列上做 RMQ 可以成为加减 1RMQ，根究这个特性可以改进 Four Russian 算法，做到 $O(n) \sim O(1)$ 的时间复杂度，$O(n)$ 的空间复杂度。

由于 Four russian 算法的瓶颈在于块内 RMQ 问题，我们重点去讨论块内 RMQ 问题的优化。

由于相邻两个数字的差值为 $\pm 1$，所以在固定左端点数字时 长度不超过 $\log n$ 的右侧序列种类数为 $\sum_{i=1}^{i \leq \log n} 2^{i-1}$，而这个式子显然不超过 $n$。

这启示我们可以预处理所有不超过 $n$ 种情况的 最小值 - 第一个元素 的值。

在预处理的时候我们需要去预处理同一块内相邻两个数字之间的差，并且使用二进制将其表示出来。

在询问的时候我们找到询问区间对应的二进制表示，查表得出答案。

这样子 Four russian 预处理的时间复杂度就被优化到了 $O(n)$。

## 笛卡尔树在 RMQ 上的应用

不了解笛卡尔树的朋友请移步 [笛卡尔树](ds/cartesian-tree.md)。

不难发现，原序列上两个点之间的 min/max，等于笛卡尔树上两个点的 LCA 的权值。根据这一点就可以借助 $O(n) \sim O(1)$ 求解树上两个点之间的 LCA 进而求解 RMQ。$O(n) \sim O(1)$ 树上 LCA 在 [LCA - 标准 RMQ](graph/lca.md#rmq_1) 已经有描述，这里不再展开。

总结一下，笛卡尔树在 RMQ 上的应用，就是通过将普通 RMQ 问题转化为 LCA 问题，进而转化为加减 1 RMQ 问题进行求解，时间复杂度为 $O(n) \sim O(1)$。当然由于转化步数较多，$O(n) \sim O(1)$ RMQ 常数较大。

如果数据随机，还可以暴力在笛卡尔树上查找。此时的时间复杂度为期望 $O(n) \sim O(\log n)$，并且实际使用时这种算法的常数往往很小。

### 例题 [Luogu P3865【模板】ST 表](https://www.luogu.com.cn/problem/P3865)

如果数据随机，则我们还可以暴力在笛卡尔树上查找。此时的时间复杂度为期望 $O(n)-O(\log n)$，并且实际使用时这种算法的常数往往很小。

## 基于状压的线性 RMQ 算法

### 隐性要求

- 序列的长度 $n$ 满足 $\log_2{n} \leq 64$

### 前置知识

- [Sparse Table](ds/sparse-table.md)

- 基本位运算

- 前后缀极值

### 算法原理

将原序列 $A[1\cdots n]$ 分成每块长度为 $O(\log_2{n})$ 的 $O(\frac{n}{\log_2{n}})$ 块。

> 听说令块长为 $1.5\times \log_2{n}$ 时常数较小。

记录每块的最大值，并用 ST 表维护块间最大值，复杂度 $O(n)$。

记录块中每个位置的前、后缀最大值 $Pre[1\cdots n], Sub[1\cdots n]$（$Pre[i]$ 即 $A[i]$ 到其所在块的块首的最大值），复杂度 $O(n)$。

若查询的 $l,r$ 在两个不同块上，分别记为第 $bl,br$ 块，则最大值为 $[bl+1,br-1]$ 块间的最大值，以及 $Sub[l]$ 和 $Pre[r]$ 这三个数的较大值。

现在的问题在于若 $l,r$ 在同一块中怎么办。

将 $A[1\cdots r]$ 依次插入单调栈中，记录下标和值，满足值从栈底到栈顶递减，则 $A[l,r]$ 中的最大值为从栈底往上，单调栈中第一个满足其下标 $p \geq l$ 的值。

由于 $A[p]$ 是 $A[l,r]$ 中的最大值，因而在插入 $A[p]$ 时，$A[l\cdots p-1]$ 都被弹出，且在插入 $A[p+1\cdots r]$ 时不可能将 $A[p]$ 弹出。

而如果用 $0/1$ 表示每个数是否在栈中，就可以用整数状压，则 $p$ 为第 $l$ 位后的第一个 $1$ 的位置。

由于块大小为 $O(\log_2{n})$，因而最多不超过 $64$ 位，可以用一个整数存下（即隐性条件的原因）。


```cpp
#include <bits/stdc++.h>

const int MAXN = 1e5 + 5;
const int MAXM = 20;

struct RMQ {
    int N, A[MAXN];
    int blockSize;
    int S[MAXN][MAXM], Pow[MAXM], Log[MAXN];
    int Belong[MAXN], Pos[MAXN];
    int Pre[MAXN], Sub[MAXN];
    int F[MAXN];
    void buildST() {
        int cur = 0, id = 1;
        Pos[0] = -1;
        for (int i = 1; i <= N; ++i) {
            S[id][0] = std::max(S[id][0], A[i]);
            Belong[i] = id;
            if (Belong[i - 1] != Belong[i])
                Pos[i] = 0;
            else
                Pos[i] = Pos[i - 1] + 1;
            if (++cur == blockSize) {
                cur = 0;
                ++id;
            }
        }
        if (N % blockSize == 0) --id;
        Pow[0] = 1;
        for (int i = 1; i < MAXM; ++i) Pow[i] = Pow[i - 1] * 2;
        for (int i = 2; i <= id; ++i) Log[i] = Log[i / 2] + 1;
        for (int i = 1; i <= Log[id]; ++i) {
            for (int j = 1; j + Pow[i] - 1 <= id; ++j) {
                S[j][i] = std::max(S[j][i - 1], S[j + Pow[i - 1]][i - 1]);
            }
        }
    }
    void buildSubPre() {
        for (int i = 1; i <= N; ++i) {
            if (Belong[i] != Belong[i - 1])
                Pre[i] = A[i];
            else
                Pre[i] = std::max(Pre[i - 1], A[i]);
        }
        for (int i = N; i >= 1; --i) {
            if (Belong[i] != Belong[i + 1])
                Sub[i] = A[i];
            else
                Sub[i] = std::max(Sub[i + 1], A[i]);
        }
    }
    void buildBlock() {
        static int S[MAXN], top;
        for (int i = 1; i <= N; ++i) {
            if (Belong[i] != Belong[i - 1])
                top = 0;
            else
                F[i] = F[i - 1];
            while (top > 0 && A[S[top]] <= A[i]) F[i] &= ~(1 << Pos[S[top--]]);
            S[++top] = i;
            F[i] |= (1 << Pos[i]);
        }
    }
    void init() {
        for (int i = 1; i <= N; ++i) scanf("%d", &A[i]);
        blockSize = log2(N) * 1.5;
        buildST();
        buildSubPre();
        buildBlock();
    }
    int queryMax(int l, int r) {
        int bl = Belong[l], br = Belong[r];
        if (bl != br) {
            int ans1 = 0;
            if (br - bl > 1) {
                int p = Log[br - bl - 1];
                ans1 = std::max(S[bl + 1][p], S[br - Pow[p]][p]);
            }
            int ans2 = std::max(Sub[l], Pre[r]);
            return std::max(ans1, ans2);
        } else {
            return A[l + __builtin_ctz(F[r] >> Pos[l])];
        }
    }
} R;

int M;

int main() {
    scanf("%d%d", &R.N, &M);
    R.init();
    for (int i = 0, l, r; i < M; ++i) {
        scanf("%d%d", &l, &r);
        printf("%d\n", R.queryMax(l, r));
    }
    return 0;
}
```

### 习题

[\[BJOI 2020\]封印](https://loj.ac/problem/3298)：SAM+RMQ

> [!NOTE] **[AcWing 1273. 天才的记忆](https://www.acwing.com/problem/content/1275/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> RMQ 区间最大值问题 也叫跳表 ST表 本质是动态规划
> 
> 思想是倍增预处理+快速查询   【原数组不能变化】
> 
> $f[i, j]$ 从i开始长度是 $2^j$ 的区间当中 最大值是多少
> 
> init:
> $$
>   f[i, j] = max{f[i, j - 1], f[i + 2^(j-1), j - 1]}
> $$
> query:
> $$
>   f[l, k], f[r - 2^k + 1, k]; 【显然中间有部分交差】
> $$
> 
>  [易知 x + 2^k - 1 = r ==> x = r - 2^k + 1]

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 200010, M = 18;   // 2*10e5 取lg2 = 17

int n, m;
int w[N];
int f[N][M];

void init() {
    for (int j = 0; j < M; ++ j )
        for (int i = 1; i + (1 << j) - 1 <= n; ++ i)    // 当前区间不能超过终点n
            if (!j) f[i][j] = w[i];
            else f[i][j] = max(f[i][j - 1], f[i + (1 << j - 1)][j - 1]);
}

int query(int l, int r) {
    int len = r - l + 1;
    int k = log(len) / log(2);
    return max(f[l][k], f[r - (1 << k) + 1][k]);
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i ) cin >> w[i];
    
    init();
    
    cin >> m;
    while (m -- ) {
        int l, r;
        cin >> l >> r;
        cout << query(l, r) << endl;
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

> [!NOTE] **[LeetCode 654. 最大二叉树](https://leetcode.cn/problems/maximum-binary-tree/)**
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
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
// 设计到查询区间最值 显然可以RMQ
class Solution {
public:
    const static int N = 1010, M = 10;

    int f[N][M];
    void init() {
        memset(f, 0, sizeof f);
        for (int j = 0; j < M; ++ j )
            for (int i = 0; i + (1 << j) - 1 < n; ++ i )
                if (!j)
                    // f[i][j] = nums[i];
                    f[i][j] = i;
                else {
                    // f[i][j] = max(f[i][j - 1], f[i + (1 << j - 1)][j - 1]);
                    int a = f[i][j - 1], b = f[i + (1 << j - 1)][j - 1];
                    f[i][j] = nums[a] > nums[b] ? a : b;
                }

                    
    }
    int query(int l, int r) {
        int len = r - l + 1, k = log(len) / log(2);
        int a = f[l][k], b = f[r - (1 << k) + 1][k];
        return nums[a] > nums[b] ? a : b;
    }

    vector<int> nums;
    int n;

    TreeNode* dfs(int l, int r) {
        if (l > r)
            return nullptr;
        int k = query(l, r);
        auto root = new TreeNode(nums[k]);
        root->left = dfs(l, k - 1), root->right = dfs(k + 1, r);
        return root;
    }

    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        this->nums = nums, this->n = nums.size();
        init();
        return dfs(0, n - 1);
    }
};
```

##### **C++ 暴力**

```cpp
// 暴力做法就能过
class Solution {
public:
    TreeNode* dfs(vector<int>& nums, int l, int r) {
        if (l >= r) return nullptr;
        int p = l;
        for (int i = l; i < r; ++ i )
            if (nums[i] > nums[p])
                p = i;
        TreeNode* root = new TreeNode(nums[p]);
        root->left = dfs(nums, l, p);
        root->right = dfs(nums, p + 1, r);
        return root;
    }
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        int n = nums.size();
        return dfs(nums, 0, n);
    }
};
```

##### **Python**

```python
# 属于 重构二叉树的题型
# 采用递归算法，假设 build(l, r) 表示对数组中 [l, r] 闭区间的部分构造二叉树；
# 首先找出最大值及其所在位置 max_i，
# 然后构造一个新的结点 rt， 递归 build(l, max_i - 1) 和 build(max_i + 1, r) 分别作为 rt 的# 左右儿子，最后返回该结点 rt

class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if not nums:return 

        def build(l, r):
            if l > r:return 
            max_num, max_i = nums[l], l 
            for i in range(l + 1, r + 1):
                if max_num < nums[i]:
                    max_num = nums[i]
                    max_i = i  
            rt = TreeNode(max_num)
            rt.left = build(l, max_i - 1)
            rt.right = build(max_i + 1, r)
            return rt

        return build(0, len(nums) - 1)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2735. 收集巧克力](https://leetcode.cn/problems/collecting-chocolates/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然可以 rmq
> 
> 更优：根据性质 线性维护

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ RMQ**

```cpp
class Solution {
public:
    using LL = long long;
    // 考虑 每个位置的价格是固定的，则对于任意一个类型其在不同位置的价格已知
    
    // f[i]: 转 i 次的情况下，收集所有的最小成本
    //  则对于特定的 i，每个的物品都价格都是一定区域内的最小值
    const static int N = 1010, M = 11;
    
    int n;
    LL f[N][M];
    LL query(int l, int r) {
        int len = r - l + 1;
        int k = log(len) / log(2);
        return min(f[l][k], f[r - (1 << k) + 1][k]);
    }
    
    long long minCost(vector<int>& nums, int x) {
        int n = nums.size();
        memset(f, 0, sizeof f);
        for (int j = 0; j < M; ++ j )
            for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                if (!j)
                    f[i][j] = nums[i - 1];
                else
                    f[i][j] = min(f[i][j - 1], f[i + (1 << j - 1)][j - 1]);
        
        LL res = 1e16;
        for (int i = 0; i < n; ++ i ) {
            LL sum = LL(x) * i;
            // cout << " init: i = " << i << " sum = " << sum << endl;
            for (int j = 0; j < n; ++ j ) {
                int l = j - i, r = j;
                LL t = 0;
                if (l >= 0) {
                    t = query(l + 1, r + 1);
                } else {
                    t = min(query((l + n) % n + 1, n), query(1, r + 1));
                }
                // cout << " j = " << j << " t = " << t << endl;
                sum += t;
            }
            // cout << " final: i = " << i << " sum = " << sum << endl;
            res = min(res, sum);
        }
        return res;
    }
};
```

##### **C++ 最优**

```cpp
class Solution {
public:
    // 更优的办法: 考虑到伴随着操作次数的增加，每个类型所能触及的最小值【不断变小】
    //  => 维护该最小值即可 更进一步降低复杂度

    using LL = long long;

    long long minCost(vector<int>& nums, int x) {
        int n = nums.size();
        vector<int> f = nums;
        LL res = 1e16;
        for (int i = 0; i < n; ++ i ) {
            for (int j = 0; j < n; ++ j )
                f[j] = min(f[j], nums[(j + i) % n]);
            LL t = 0;
            for (auto v : f)
                t += v;
            res = min(res, 1ll * x * i + t);
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

> [!NOTE] **[LeetCode 3117. 划分数组得到最小的值之和](https://leetcode.cn/problems/minimum-sum-of-values-by-dividing-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数据结构优化大杂烩
> 
> - `[prefix_sum / RMQ] + [BIT / 单调队列]`
> 
> - `X + LogTrick`
> 
>   LogTrick: **给定一个数组，以某个右端点为结尾的所有子数组，其中不同的 或/与/lcm/gcd 值至多只有 logU 个**
> 
> => 本专题: RMQ 维护 `&` 性质

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ prefix_sum + BIT**

```cpp
class Solution {
public:
    // 搜索TLE (TODO: revisit 剪枝策略)
    //
    // 考虑 dp
    //  f[i][j] 表示考虑前i个元素切分成j段的最小值和
    //  => f[i][j] = min(f[x][j-1]) + nums[i]
    //     其中x需满足一定条件: and[x+1,j] = andVlues[j]
    //  由与操作性质容易推导 x必然锁定到一个连续区间 接下来就是查询这个区间的最小值
    // 
    // 拆解子问题:
    // 1. 如何计算得到x的区间 => 二分 => 需能快速获取特定区间的 and 值 => 前缀和/rmq
    // 2. 如何维护区间最小值 => BIT/单调队列(更优)
    //
    // 【数据结构优化DP】

    const static int N = 1e4 + 10, M = 11, K = 18, INF = 0x3f3f3f3f;

    // prefix sum
    int sum[N][K];
    int get_by_sum(int l, int r) {
        int ret = 0;
        for (int j = 0; j < K; ++ j )
            if (sum[r][j] - sum[l - 1][j] == r - l + 1)
                ret += 1 << j;
        return ret;
    }
    
    // BIT
    // - 区间最值
    // - 需要nxt数组记录对于第j段下次应当加入bit的位置
    int w[M][N], tr[M][N], nxt[M];
    int lowbit(int x) {
        return x & -x;
    }
    void bit_modify(int tr[], int w[], int x, int y) {
        w[x] = y;
        for (int i = x; i < N; i += lowbit(i))
            tr[i] = min(tr[i], y);  // min
    }
    int bit_query(int tr[], int w[], int l, int r) {
        int ret = INF;  // ATTENTION debug
        for (; l <= r; ) {
            ret = min(ret, w[r]);
            for ( -- r ; r >= l + lowbit(r); r -= lowbit(r))
                ret = min(ret, tr[r]);
        }
        return ret;
    }

    void init() {
        {
            memset(sum, 0, sizeof sum);
            for (int i = 1; i <= n; ++ i )
                for (int j = 0; j < K; ++ j )
                    sum[i][j] = sum[i - 1][j] + (nums[i - 1] >> j & 1);
        }
        {
            memset(w, 0x3f, sizeof w);
            memset(tr, 0x3f, sizeof tr);
            for (int i = 0; i < M; ++ i )
                nxt[i] = 1;
        }
    }

    int search(int L, int R, int x) {
        int l = L, r = R + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (get_by_sum(mid, R) >= x)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    int f[N][M];

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();

        init();

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                int target = andValues[j - 1];

                // ATTENTION 减少无效计算
                if (get_by_sum(1, i) > target)
                    continue;
                
                // ATTENTION: 思考
                // 找到 第一个 >= target 的位置  &  第一个 > target 的位置
                //  l,r 本质都是上一个轮次的下标
                int l = search(1, i, target), r = search(1, i, target + 1);
                if (l == i + 1 || get_by_sum(l, i) != target)
                    continue;

                r -- ;  // 实际上 r 的位置刚好比最右侧的分界点大 1
                
                auto & local_w = w[j - 1];      // 上一个轮次的
                auto & local_tr = tr[j - 1];
                for (; nxt[j - 1] < r; ++ nxt[j - 1]) {
                    int x = nxt[j - 1], y = f[x][j - 1];
                    bit_modify(local_tr, local_w, x, y);
                }

                // ATTENTION l-1: 思考 上一个区间的结束位置自l左侧一个位置开始
                int min_val = bit_query(local_tr, local_w, max(l - 1, 1), r);
                if (j == 1 && get_by_sum(1, i) == avs[j - 1])
                    min_val = 0;

                f[i][j] = min_val + nums[i - 1];
            }
        
        return f[n][m] < INF / 2 ? f[n][m] : -1;
    }
};
```

##### **C++ RMQ(学习记录&的方法) + BIT**

```cpp
class Solution {
public:
    // 搜索TLE (TODO: revisit 剪枝策略)
    //
    // 考虑 dp
    //  f[i][j] 表示考虑前i个元素切分成j段的最小值和
    //  => f[i][j] = min(f[x][j-1]) + nums[i]
    //     其中x需满足一定条件: and[x+1,j] = andVlues[j]
    //  由与操作性质容易推导 x必然锁定到一个连续区间 接下来就是查询这个区间的最小值
    // 
    // 拆解子问题:
    // 1. 如何计算得到x的区间 => 二分 => 需能快速获取特定区间的 and 值 => 前缀和/rmq
    // 2. 如何维护区间最小值 => BIT/单调队列(更优)
    //
    // 【数据结构优化DP】

    const static int N = 1e4 + 10, M = 11, K = 18, INF = 0x3f3f3f3f;

    // [no] prefix sum => RMQ
    int sum[N][K];
    int get_by_sum(int l, int r) {
        int len = r - l + 1;
        int k = log(len) / log(2);
        return sum[l][k] & sum[r - (1 << k) + 1][k];
    }
    
    // BIT
    // - 区间最值
    // - 需要nxt数组记录对于第j段下次应当加入bit的位置
    int w[M][N], tr[M][N], nxt[M];
    int lowbit(int x) {
        return x & -x;
    }
    void bit_modify(int tr[], int w[], int x, int y) {
        w[x] = y;
        for (int i = x; i < N; i += lowbit(i))
            tr[i] = min(tr[i], y);  // min
    }
    int bit_query(int tr[], int w[], int l, int r) {
        int ret = INF;  // ATTENTION debug
        for (; l <= r; ) {
            ret = min(ret, w[r]);
            for ( -- r ; r >= l + lowbit(r); r -= lowbit(r))
                ret = min(ret, tr[r]);
        }
        return ret;
    }

    void init() {
        {
            memset(sum, 0, sizeof sum);
            // 倍增RMQ
            for (int j = 0; j < K; ++ j )
                for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                    if (!j)
                        sum[i][j] = nums[i - 1];
                    else
                        // ATTENTION RMQ 维护性质变种
                        //  下标非常容易写疵...
                        sum[i][j] = sum[i][j - 1] & sum[i + (1 << j - 1)][j - 1];
        }
        {
            memset(w, 0x3f, sizeof w);
            memset(tr, 0x3f, sizeof tr);
            for (int i = 0; i < M; ++ i )
                nxt[i] = 1;
        }
    }

    int search(int L, int R, int x) {
        int l = L, r = R + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (get_by_sum(mid, R) >= x)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    int f[N][M];

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();

        init();

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                int target = andValues[j - 1];

                // ATTENTION 减少无效计算
                if (get_by_sum(1, i) > target)
                    continue;
                
                // ATTENTION: 思考
                // 找到 第一个 >= target 的位置  &  第一个 > target 的位置
                //  l,r 本质都是上一个轮次的下标
                int l = search(1, i, target), r = search(1, i, target + 1);
                if (l == i + 1 || get_by_sum(l, i) != target)
                    continue;

                r -- ;  // 实际上 r 的位置刚好比最右侧的分界点大 1
                
                auto & local_w = w[j - 1];      // 上一个轮次的
                auto & local_tr = tr[j - 1];
                for (; nxt[j - 1] < r; ++ nxt[j - 1]) {
                    int x = nxt[j - 1], y = f[x][j - 1];
                    bit_modify(local_tr, local_w, x, y);
                }

                // ATTENTION l-1: 思考 上一个区间的结束位置自l左侧一个位置开始
                int min_val = bit_query(local_tr, local_w, max(l - 1, 1), r);
                if (j == 1 && get_by_sum(1, i) == avs[j - 1])
                    min_val = 0;

                f[i][j] = min_val + nums[i - 1];
            }
        
        return f[n][m] < INF / 2 ? f[n][m] : -1;
    }
};
```

##### **C++ RMQ(前缀和写法略) + 单调队列**

```cpp
class Solution {
public:
    // 搜索TLE (TODO: revisit 剪枝策略)
    //
    // 考虑 dp
    //  f[i][j] 表示考虑前i个元素切分成j段的最小值和
    //  => f[i][j] = min(f[x][j-1]) + nums[i]
    //     其中x需满足一定条件: and[x+1,j] = andVlues[j]
    //  由与操作性质容易推导 x必然锁定到一个连续区间 接下来就是查询这个区间的最小值
    // 
    // 拆解子问题:
    // 1. 如何计算得到x的区间 => 二分 => 需能快速获取特定区间的 and 值 => 前缀和/rmq
    // 2. 如何维护区间最小值 => BIT/单调队列(更优)
    //
    // 【数据结构优化DP】

    const static int N = 1e4 + 10, M = 11, K = 18, INF = 0x3f3f3f3f;

    // [no] prefix sum => RMQ
    int sum[N][K];
    int get_by_sum(int l, int r) {
        int len = r - l + 1;
        int k = log(len) / log(2);
        return sum[l][k] & sum[r - (1 << k) + 1][k];
    }
    
    // [no] BIT => deque
    // - 区间最值
    // - 需要nxt数组记录对于第j段下次应当加入bit的位置
    deque<int> q[M];
    int nxt[M];

    void init() {
        {
            memset(sum, 0, sizeof sum);
            // 倍增RMQ
            for (int j = 0; j < K; ++ j )
                for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                    if (!j)
                        sum[i][j] = nums[i - 1];
                    else
                        // ATTENTION RMQ 维护性质变种
                        //  下标非常容易写疵...
                        sum[i][j] = sum[i][j - 1] & sum[i + (1 << j - 1)][j - 1];
        }
        {
            for (int i = 0; i < M; ++ i )
                nxt[i] = 1, q[i].clear();
        }
    }

    int search(int L, int R, int x) {
        int l = L, r = R + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (get_by_sum(mid, R) >= x)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    int f[N][M];

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();

        init();

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                int target = andValues[j - 1];

                // ATTENTION 减少无效计算
                if (get_by_sum(1, i) > target)
                    continue;
                
                // ATTENTION: 思考
                // 找到 第一个 >= target 的位置  &  第一个 > target 的位置
                //  l,r 本质都是上一个轮次的下标
                int l = search(1, i, target), r = search(1, i, target + 1);
                if (l == i + 1 || get_by_sum(l, i) != target)
                    continue;

                r -- ;  // 实际上 r 的位置刚好比最右侧的分界点大 1
                
                auto & local_q = q[j - 1];      // 上一个轮次的

                for (; nxt[j - 1] < r; ++ nxt[j - 1]) {
                    while (!local_q.empty() && f[local_q.back()][j -1] >= f[nxt[j - 1]][j - 1])
                        local_q.pop_back();
                    local_q.push_back(nxt[j - 1]);
                }
                // ATTENTION: 因为前面可能会加进去小于l-1的，所以pop_front要放在后面
                // ATTENTION l-1: 思考 上一个区间的结束位置自l左侧一个位置开始
                while (!local_q.empty() && local_q.front() < l - 1)
                    local_q.pop_front();

                int min_val = INF;
                if (!local_q.empty())
                    min_val = f[local_q.front()][j - 1];
                if (j == 1 && get_by_sum(1, i) == avs[j - 1])
                    min_val = 0;

                f[i][j] = min_val + nums[i - 1];
            }
        
        return f[n][m] < INF / 2 ? f[n][m] : -1;
    }
};
```

##### **C++ LogTrick (打掉一个log) + 单调队列(BIT略)**

```cpp
class Solution {
public:
    using PII = pair<int, int>;

    template <typename T, typename F = function<T(const T &, const T &)>>
    struct SparseTable {
        int n;
        vector<vector<T>> mat;
        F func;

        SparseTable() {}    // For vector
        SparseTable(const vector<T> & a, const F & f) : func(f) {
            n = a.size();
            if (n == 0)
                return;
            int maxLog = 32 - __builtin_clz(n);
            mat.resize(maxLog);
            mat[0] = a;
            mat[0].insert(mat[0].begin(), 0);
            for (int j = 1; j < maxLog; ++ j ) {
                mat[j].resize(n - (1 << j) + 1 + 1);
                // i = n - (1 << j) + 1, 故申请内存需要再+1
                for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                    mat[j][i] = func(mat[j - 1][i], mat[j - 1][i + (1 << j - 1)]);
            }
        }
        T query(int l, int r) const {
            assert(0 <= l && l <= r && r <= n - 1);
            int lg = 31 - __builtin_clz(r - l + 1);
            return func(mat[lg][l], mat[lg][r - (1 << lg) + 1]);
        }
    };

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();
        
        const int INF = 0x3f3f3f3f;
        vector<int> dp(n + 1, INF);
        dp[0] = 0;
        for (int parts = 1; parts <= m; ++ parts ) {
            SparseTable st(dp, [&](int i, int j) {return min(i, j);});
            vector<int> ndp(n + 1, INF);

            vector<PII> G;  // (and, left endpoint)
            for (int i = 1; i <= n; ++ i ) {
                int x = nums[i - 1];
                G.push_back({x, i});
                for (auto & g : G)
                    g.first &= x;
                G.erase(unique(G.begin(), G.end(), [](const PII & g1, const PII & g2) {
                    // ATTENTION trick: 对于相同的保留第一个
                    return g1.first == g2.first;
                }), G.end());

                // ATTENTION: G.size() <= log(bit_width)
                for (int j = 0; j < G.size(); ++ j ) {
                    int l = G[j].second, r = (j == G.size() - 1 ? i : G[j + 1].second - 1);
                    if (G[j].first == avs[parts - 1])
                        ndp[i] = min(ndp[i], x + st.query(l, r));
                }
            }

            swap(dp, ndp);
        }
        
        return dp[n] >= INF / 2 ? -1 : dp[n];
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
