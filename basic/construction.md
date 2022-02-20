从形式上来看，问题的答案往往具有某种规律性，使得在问题规模迅速增大的时候，仍然有机会比较容易地得到答案。

这要求解题时要思考问题规模增长对答案的影响，这种影响是否可以推广。例如，在设计动态规划方法的时候，要考虑从一个状态到后继状态的转移会造成什么影响。

## 例题

### 例题 1

> [!NOTE] **[Codeforces Round #384 (Div. 2) C.Vladik and fractions](http://codeforces.com/problemset/problem/743/C)**
> 
> 构造一组 $x,y,z$，使得对于给定的 $n$，满足 $\dfrac{1}{x}+\dfrac{1}{y}+\dfrac{1}{z}=\dfrac{2}{n}$

> [!TIP] **解题思路**
> 
> 从样例二可以看出本题的构造方法。
> 
> 显然 $n,n+1,n(n+1)$ 为一组合法解。特殊地，当 $n=1$ 时，无解，这是因为 $n+1$ 与 $n(n+1)$ 此时相等。

### 例题 2

> [!NOTE] **[Luogu P3599 Koishi Loves Construction](https://www.luogu.com.cn/problem/P3599)**
> 
> Task1：试判断能否构造并构造一个长度为 $n$ 的 $1\dots n$ 的排列，满足其 $n$ 个前缀和在模 $n$ 的意义下互不相同
> 
> Taks2：试判断能否构造并构造一个长度为 $n$ 的 $1\dots n$ 的排列，满足其 $n$ 个前缀积在模 $n$ 的意义下互不相同

> [!TIP] **解题思路**
> 
> 对于 task1：
> 
> 当 $n$ 为奇数时，无法构造出合法解；
> 
> 当 $n$ 为偶数时，可以构造一个形如 $n,1,n-2,3,\cdots$ 这样的数列。
> 
> 首先，我们可以发现 $n$ 必定出现在数列的第一位，否则 $n$ 出现前后的两个前缀和必然会陷入模意义下相等的尴尬境地；
> 
> 考虑通过构造前缀和序列的方式来获得原数列，可以发现前缀和序列两两之间的差在模意义下不能相等，因为前缀和序列的差分序列对应着原来的排列。
> 
> 因此我们尝试以前缀和数列在模意义下为
> $$
> 0,1,-1,2,-2,\cdots
> $$
> 
> 这样的形式来构造这个序列，不难发现它完美地满足所有限制条件。
> 
> 对于 task2：
> 
> 当 $n$ 为除 $4$ 以外的合数时，无法构造出合法解
> 
> 当 $n$ 为质数或 $4$ 时，可以构造一个形如 $1,\dfrac{2}{1},\dfrac{3}{2},\cdots,\dfrac{n-1}{n-2},n$ 这样的数列
> 
> 先考虑什么时候有解：
> 显然，当 $n$ 为合数时无解。因为对于一个合数来说，存在两个比它小的数 $p,q$ 使得 $p\times q \equiv 0 \pmod n$，如 $(3\times6)\%9=0$。那么，当 $p,q$ 均出现过后，数列的前缀积将一直为 $0$，故合数时无解。特殊地，我们可以发现 $4=2\times 2$，无满足条件的 $p,q$，因此存在合法解。
> 
> 我们考虑如何构造这个数列：
> 
> 和 task1 同样的思路，我们发现 $1$ 必定出现在数列的第一位，否则 $1$ 出现前后的两个前缀积必然相等；而 $n$ 必定出现在数列的最后一位，因为 $n$ 出现位置后的所有前缀积在模意义下都为 $0$。手玩几组样例以后发现，所有样例中均有一组合法解满足前缀积在模意义下为 $1,2,3,\cdots,n$，因此我们可以构造出上文所述的数列来满足这个条件。那么我们只需证明这 $n$ 个数互不相同即可。
> 
> 我们发现这些数均为 $1 \cdots n-2$ 的逆元 $+1$，因此各不相同，此题得解。

### 例题 3

> [!NOTE] **[AtCoder Grand Contest 032 B](https://atcoder.jp/contests/agc032/tasks/agc032_b)**
> 
> 给定一个整数 $N$，试构造一个节点数为 $N$ 无向图。令节点编号为 $1\ldots N$，要求其满足以下条件：
> 
> - 这是一个简单连通图。
> 
> - 存在一个整数 $S$ 使得对于任意节点，与其相邻节点的下标和为 $S$。
> 
> 保证输入数据有解。

> [!TIP] **解题思路**
> 
> 手玩一下 $n=3,4,5$ 的情况，我们可以找到一个构造思路。
> 
> 构造一个完全 $k$ 分图，保证这 $k$ 部分和相等。则每个点的 $S$ 均相等，为 $\dfrac{(k-1)\sum_{i=1}^{n}i}{k}$。
> 
> 如果 $n$ 为偶数，那么我们可以前后两两配对，即 $\{1,n\},\{2,n-1\}\cdots$
> 
> 如果 $n$ 为奇数，那么我们可以把 $n$ 单拿出来作为一组，剩余的 $n-1$ 个两两配对，即 $\{n\},\{1,n-1\},\{2,n-2\}\cdots$
> 
> 这样构造出的图在 $n\ge 3$ 时连通性易证，在此不加赘述。

### 例题 4

> [!NOTE] **BZOJ 4971「Lydsy1708 月赛」记忆中的背包**
> 
> 经过一天辛苦的工作，小 Q 进入了梦乡。他脑海中浮现出了刚进大学时学 01 背包的情景，那时还是大一萌新的小 Q 解决了一道简单的 01 背包问题。这个问题是这样的
> 
> 给定 $n$ 个物品，每个物品的体积分别为 $v_1,v_2,…,v_n$，请计算从中选择一些物品（也可以不选），使得总体积恰好为 $w$ 的方案数。因为答案可能非常大，你只需要输出答案对 $P$ 取模的结果。
> 
> 因为长期熬夜刷题，他只看到样例输入中的 $w$ 和 $P$，以及样例输出是 $k$，看不清到底有几个物品，也看不清每个物品的体积是多少。直到梦醒，小 Q 也没有看清 $n$ 和 $v$，请写一个程序，帮助小 Q 一起回忆曾经的样例输入。

> [!TIP] **解题思路**
> 
> 这道题是自由度最高的构造题之一了。这就导致了没有头绪，难以入手的情况。
> 
> 首先，不难发现模数是假的。由于我们自由构造数据，我们一定可以让方案数不超过模数。
> 
> 通过奇怪的方式，我们想到可以通过构造 $n$ 个 代价为 $1$ 的小物品和几个代价大于 $\dfrac{w}{2}$ 的大物品。
> 
> 由于大物品只能取一件，所以每个代价为 $x$ 的大物品对方案数的贡献为 $C_{n}^{w-x}$。
> 
> 令 $f_{i,j}$ 表示有 $i$ 个 $1$，方案数为 $j$ 的最小大物品数。
> 
> 用 dp 预处理出 $f$，通过计算可知只需预处理 $i\le 20$ 的所有值即可。

## 习题

> [!NOTE] **[LeetCode 667. 优美的排列 II](https://leetcode-cn.com/problems/beautiful-arrangement-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 构造题 找规律
> 
> 结论：【最大值最小值交替出现】
> 
> 此时不同 k 达到最大为 n-1

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> constructArray(int n, int k) {
        vector<int> res(n);
        for (int i = 0; i < n - k - 1; i ++ ) res[i] = i + 1;
        int u = n - k - 1;
        int i = n - k, j = n;
        while (u < n) {
            res[u ++ ] = i ++ ;
            if (u < n) res[u ++ ] = j -- ;
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

> [!NOTE] **[LeetCode 1253. 重构 2 行二进制矩阵](https://leetcode-cn.com/problems/reconstruct-a-2-row-binary-matrix/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心即可 需要注意优先处理列和为 2 的
> 
> 以及 处理 2 的时候就要判断 uv lv 是否合法 (WA 1)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> reconstructMatrix(int upper, int lower,
                                          vector<int>& colsum) {
        int n = colsum.size();
        int uv = upper, lv = lower;
        vector<vector<int>> g(2, vector<int>(n));

        int tot = upper + lower, che = 0;
        for (auto& v : colsum) che += v;
        if (che != tot) return vector<vector<int>>{};

        for (int i = 0; i < n; ++i)
            if (colsum[i] == 2) {
                g[0][i] = g[1][i] = 1;
                uv -= 1, lv -= 1;
                if (uv < 0 || lv < 0) return vector<vector<int>>{};
            }
        for (int i = 0; i < n; ++i) {
            if (colsum[i] == 1) {
                if (uv)
                    g[0][i] = 1, --uv;
                else if (lv)
                    g[1][i] = 1, --lv;
                else
                    return vector<vector<int>>{};
            }
        }

        return g;
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

> [!NOTE] **[LeetCode 1968. 构造元素不等于两相邻元素平均值的数组](https://leetcode-cn.com/problems/array-with-elements-not-equal-to-average-of-neighbors/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题
> 
> 显然排序再间隙排，构造摆动序列，此时必然满足题目要求

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> rearrangeArray(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<int> res(n);
        int p = 0;
        for (int i = 0; i < n; i += 2)
            res[i] = nums[p ++ ];
        for (int i = 1; i < n; i += 2)
            res[i] = nums[p ++ ];
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

> [!NOTE] **[AcWing 516. 神奇的幻方](https://www.acwing.com/problem/content/518/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 按题意模拟即可

const int N = 40;

int n;
int a[N][N];

int main() {
    cin >> n;
    int x = 1, y = n / 2 + 1;
    for (int i = 1; i <= n * n; ++ i ) {
        a[x][y] = i;
        if (x == 1 && y == n)
            x ++ ;
        else if (x == 1)
            x = n, y ++ ;
        else if (y == n)
            x -- , y = 1;
        else if (a[x - 1][y + 1])
            x ++ ;
        else
            x -- , y ++ ;
    }
    for (int i = 1; i <= n; ++ i ) {
        for (int j = 1; j <= n; ++ j )
            cout << a[i][j] << ' ';
        cout << endl;
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

> [!NOTE] **[AcWing 2268. 时态同步](https://www.acwing.com/problem/content/2270/)**
> 
> 题意: 如何使得叶子结点路径总和相同

> [!TIP] **思路**
> 
> 贪心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 贪心 很多时候可以作为转化的模版
// 如何使得叶子结点路径总和相同

using LL = long long;
const int N = 500010, M = N << 1;

int n, root;
int h[N], e[M], w[M], ne[M], idx;
LL d[N], res;

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u, int pa) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == pa)
            continue;
        dfs(j, u);
        d[u] = max(d[u], d[j] + w[i]);
    }
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == pa)
            continue;
        // 所有儿子到达当前节点的值必然需要相同
        res += d[u] - (d[j] + w[i]);
    }
}

int main() {
    cin >> n >> root;
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    
    dfs(root, -1);
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

> [!NOTE] **[LeetCode 2081. k 镜像数字的和](https://leetcode-cn.com/problems/sum-of-k-mirror-numbers/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典模型：**求k进制下的第n个回文数字**
> 
> - 折半搜索缩小至 `sqrt(n)` 规模
> - 将当前数翻转并追加到当前数后面
> - 显然形成 `奇/偶` 两种长度情况，根据有序性质推理先生成前者
> - 规定搜索的范围，`i` 需要在 `[10^k, 10^(k+1)]` 的范围内，使用 `[l, r]` 维护
> 
> 显然也可以打表... 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 基础知识: [求 k 进制下的第 n 个回文数]
    // 在此基础上加上 10 进制回文的判断，中间累加和即可
    // 
    // Knowledge: 1e9里的十进制回文数有109998个
    using LL = long long;
    
    bool check(LL x) {
        string s = to_string(x);
        for (int i = 0, j = s.size() - 1; i < j; ++ i , -- j )
            if (s[i] != s[j])
                return false;
        return true;
    }
    
    long long kMirror(int k, int n) {
        LL res = 0, l = 1;
        while (n) {
            LL r = l * k;
            for (int op = 0; op < 2; ++ op )
                for (int i = l; i < r && n; ++ i ) {
                    int x = (op ? i : i / k);   // 生成奇数还是偶数位 0代表奇数
                    LL conbined = i;
                    while (x) {
                        conbined = conbined * k + x % k;
                        x /= k;
                    }
                    // 本题要求 10 进制回文  特殊处理
                    if (check(conbined) && n) {
                        res += conbined;
                        n -- ;
                    }
                }
            l = r;
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