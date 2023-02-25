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

### 待细分

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

> [!NOTE] **[Codeforces A. Adding Digits](https://codeforces.com/problemset/problem/260/A)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> At first try to add to the right one digit from 0 to 9. If it is impossible write -1.
> 
> In other case, the remaining n–1 digits can be 0 because divisibility doesn’t change.

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Adding Digits
// Contest: Codeforces - Codeforces Round #158 (Div. 2)
// URL: https://codeforces.com/problemset/problem/260/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

int main() {
    int a, b, n;
    cin >> a >> b >> n;

    bool f = false;
    for (int i = 0; i < 10; ++i) {
        int t = a * 10 + i;
        if (t % b == 0) {
            f = true;
            a = t;
            break;
        }
    }
    if (f) {
        string res = to_string(a);
        for (int i = 0; i < n - 1; ++i)
            res.push_back('0');
        cout << res << endl;
    } else
        cout << -1 << endl;

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

> [!NOTE] **[Codeforces B. Flag Day](https://codeforces.com/problemset/problem/357/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 题意条件对解题简化很关键
> 
> 优雅实现的代码技巧

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Flag Day
// Contest: Codeforces - Codeforces Round #207 (Div. 2)
// URL: https://codeforces.com/problemset/problem/357/B
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 注意题意 最多有一个人出现在其他组合 且最多只能出现一次
// 这样就很好做了
const int N = 100010;

int n, m;
int r[N];

// https://codeforces.com/contest/357/submission/109691304
int mod_add(int x) { return x % 3; }

int main() {
    cin >> n >> m;
    while (m--) {
        int a, b, c;
        cin >> a >> b >> c;
        if (r[a]) {
            r[b] = mod_add(r[a]) + 1;
            r[c] = mod_add(r[b]) + 1;
        } else if (r[b]) {
            r[a] = mod_add(r[b]) + 1;
            r[c] = mod_add(r[a]) + 1;
        } else if (r[c]) {
            r[a] = mod_add(r[c]) + 1;
            r[b] = mod_add(r[a]) + 1;
        } else {
            r[a] = 1, r[b] = 2, r[c] = 3;
        }
    }
    for (int i = 1; i <= n; ++i)
        cout << r[i] << ' ';
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

> [!NOTE] **[Codeforces C. XOR and OR](https://codeforces.com/problemset/problem/282/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. XOR and OR
// Contest: Codeforces - Codeforces Round #173 (Div. 2)
// URL: https://codeforces.com/problemset/problem/282/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 思维题
// 暴力遍历转化的错误思路 WA：
// https://codeforces.com/contest/282/submission/109767746
// 本质上，串有 1 才可以转换 1数量可以变但无法被消除
// 故 都有1 或 都没有1

int main() {
    string s1, s2;
    cin >> s1 >> s2;

    bool f1 = false, f2 = false;
    for (auto c : s1)
        if (c == '1') {
            f1 = true;
            break;
        }
    for (auto c : s2)
        if (c == '1') {
            f2 = true;
            break;
        }
    cout << (s1.size() == s2.size() && f1 == f2 ? "YES" : "NO") << endl;

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

> [!NOTE] **[Codeforces C. Valera and Tubes](https://codeforces.com/problemset/problem/441/C)**
> 
> 题意: 
> 
> 我以为是放正方块 其实放一条可以弯折的管子
> 
> 前面每一个都只消耗俩方格 最后一个消耗剩余所有方格即可

> [!TIP] **思路**
> 
> 思维 构造方法很多
> 
> **有一个聚聚的超简洁代码**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Valera and Tubes
// Contest: Codeforces - Codeforces Round #252 (Div. 2)
// URL: https://codeforces.com/problemset/problem/441/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

#define x first
#define y second

using PII = pair<int, int>;

int main() {
    int n, m, k;
    cin >> n >> m >> k;

    vector<PII> ve;
    int u = 1, d = n, l = 1, r = m;
    for (;;) {
        for (int i = l; i <= r; ++i)
            ve.push_back({u, i});
        if (++u > d)
            break;

        for (int i = u; i <= d; ++i)
            ve.push_back({i, r});
        if (--r < l)
            break;

        for (int i = r; i >= l; --i)
            ve.push_back({d, i});
        if (--d < u)
            break;

        for (int i = d; i >= u; --i)
            ve.push_back({i, l});
        if (++l > r)
            break;
    }

    int t = n * m - (k - 1) * 2;
    cout << t;
    for (int i = 0; i < t; i++)
        cout << ' ' << ve[i].x << ' ' << ve[i].y;
    cout << endl;

    for (int i = t; i < n * m; i += 2)
        cout << 2 << ' ' << ve[i].x << ' ' << ve[i].y << ' ' << ve[i + 1].x
             << ' ' << ve[i + 1].y << endl;

    return 0;
}
```

##### **C++ 简洁代码**

```cpp
// Problem: C. Valera and Tubes
// Contest: Codeforces - Codeforces Round #252 (Div. 2)
// URL: https://codeforces.com/problemset/problem/441/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

void next(int n, int m, int &x, int &y) {
    if (x & 1)
        y++;
    else
        y--;
    if (y > m)
        x++, y--;
    if (y < 1)
        x++, y++;
}

void Print(int n, int m, int &x, int &y) {
    cout << x << " " << y << " ";
    next(n, m, x, y);
}

int main() {
    int n, m, k, x, y;
    cin >> n >> m >> k;
    x = y = 1;
    for (int i = 1; i <= k; i++) {
        if (i <= k - 1) {
            cout << 2 << " ";
            Print(n, m, x, y);
            Print(n, m, x, y);
            cout << endl;
        } else {
            cout << n * m - 2 * (k - 1) << " ";
            while (x <= n)
                Print(n, m, x, y);
            cout << endl;
        }
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces A. Escape from Stones](http://codeforces.com/problemset/problem/264/A)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 纯模拟会有精度损失
> 
> 思维题 根据题意来找规律

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Escape from Stones
// Contest: Codeforces - Codeforces Round #162 (Div. 1)
// URL: http://codeforces.com/problemset/problem/264/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// 模拟会因为double的精度损失造成错误答案
// http://codeforces.com/contest/264/submission/110602981
//
// 找规律: 向右躲时先输出 顺序
//        向左躲时后输出 倒序
//
// 快读也TLE
// http://codeforces.com/contest/264/submission/110603825

const int N = 1000010;

char s[N];

int main() {
    scanf("%s", s);

    int n = strlen(s);
    for (int i = 0; i < n; ++i)
        if (s[i] == 'r')
            printf("%d\n", i + 1);

    for (int i = n - 1; i >= 0; --i)
        if (s[i] == 'l')
            printf("%d\n", i + 1);

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

> [!NOTE] **[Codeforces B. Convex Shape](http://codeforces.com/problemset/problem/275/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 实现的代码技巧

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Convex Shape
// Contest: Codeforces - Codeforces Round #168 (Div. 2)
// URL: http://codeforces.com/problemset/problem/275/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

/*
5 5
WBBBW
WBBWW
WBBWW
BBBWW
BBWWW
*/

#include <bits/stdc++.h>
using namespace std;

const int N = 55;

int n, m, tot;
char g[N][N];

bool check() {
    for (int x1 = 0; x1 < n; ++x1)
        for (int y1 = 0; y1 < m; ++y1) {
            if (g[x1][y1] == 'W')
                continue;

            for (int x2 = 0; x2 < n; ++x2)
                for (int y2 = 0; y2 < m; ++y2) {
                    if (g[x2][y2] == 'W')
                        continue;

                    // 分别为两侧路径
                    bool f1 = true, f2 = true;

                    for (int i = min(x1, x2); i <= max(x1, x2); ++i) {
                        if (g[i][y1] == 'W')
                            f1 = false;
                        if (g[i][y2] == 'W')
                            f2 = false;
                    }
                    for (int i = min(y1, y2); i <= max(y1, y2); ++i) {
                        if (g[x1][i] == 'W')
                            f2 = false;
                        if (g[x2][i] == 'W')
                            f1 = false;
                    }
                    if (!f1 && !f2)
                        return false;
                }
        }
    return true;
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; ++i)
        cin >> g[i];

    cout << (check() ? "YES" : "NO") << endl;

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

> [!NOTE] **[Codeforces C. Anya and Ghosts](https://codeforces.com/problemset/problem/508/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 重复做
> 
> 因可以提前点蜡烛的实现技巧

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Anya and Ghosts
// Contest: Codeforces - Codeforces Round #288 (Div. 2)
// URL: https://codeforces.com/problemset/problem/508/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 贪心 思维 重复

const int N = 310;

int m, t, r;
int w[N];
unordered_map<int, bool> up;

int main() {
    cin >> m >> t >> r;
    for (int i = 1; i <= m; ++i)
        cin >> w[i];

    bool f = true;
    int res = 0;
    for (int i = 1; i <= m; ++i) {
        int cnt = 0;
        // 计算已点亮的 注意 点亮时间耗时1的细节
        // 以及因为可以在午夜前点亮 直接用map记录up
        // https://codeforces.com/contest/508/submission/110869906
        for (int j = w[i] - 1; j >= w[i] - t; --j)
            if (up[j])
                ++cnt;
        for (int j = w[i] - 1; j >= w[i] - t && cnt < r; --j)
            if (!up[j])
                up[j] = true, res++, cnt++;
        if (cnt < r) {
            f = false;
            break;
        }
    }
    cout << (f ? res : -1) << endl;

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

> [!NOTE] **[Codeforces C. Removing Columns](https://codeforces.com/problemset/problem/496/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 好想到做法 注意 corner case

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Removing Columns
// Contest: Codeforces - Codeforces Round #283 (Div. 2)
// URL: https://codeforces.com/problemset/problem/496/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 一开始想简单了 没有考虑复杂情况 显然WA
// https://codeforces.com/contest/496/submission/110871573
// 需要确定有序
//
// luogu也有操作结束之后删字符串的操作 达到类似 in_order 类似效果
// https://www.luogu.com.cn/problem/solution/CF496C
// http://hzwer.com/5685.html

const int N = 110;

int n, m;
char g[N][N];
bool st[N], in_order[N], t[N];

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; ++i)
        cin >> g[i];

    for (int j = 0; j < m; ++j) {
        bool f = false;
        memset(t, 0, sizeof t);

        for (int i = 1; i < n; ++i)
            if (!in_order[i] && g[i][j] < g[i - 1][j]) {
                f = true;
                break;
            } else if (g[i][j] > g[i - 1][j])
                t[i] = true;

        if (f)
            st[j] = true;
        else {
            int cnt = 0;
            for (int i = 1; i < n; ++i)
                if (t[i] || in_order[i])
                    in_order[i] = true, cnt++;
            // cout << "j = " << j << " cnt = " << cnt << endl;
            if (cnt == n - 1)
                break;
        }
    }

    int res = 0;
    for (int i = 0; i < m; ++i)
        if (st[i])
            ++res;
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

> [!NOTE] **[Codeforces C. Beautiful Sets of Points](https://codeforces.com/problemset/problem/268/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 构造

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Beautiful Sets of Points
// Contest: Codeforces - Codeforces Round #164 (Div. 2)
// URL: https://codeforces.com/problemset/problem/268/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 思维题
// 易知每行每列最多有一个点 此时显然最多有 min(n, m) + 1 个
// 因为 [0, 0] 不行 换斜对角线即可

int n, m;

int main() {
    cin >> n >> m;

    cout << min(n, m) + 1 << endl;
    for (int i = 0; i <= min(n, m); ++i)
        cout << i << ' ' << min(n, m) - i << endl;

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

> [!NOTE] **[Codeforces C. Devu and Partitioning of the Array](https://codeforces.com/problemset/problem/439/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题 细节case很多

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Devu and Partitioning of the Array
// Contest: Codeforces - Codeforces Round #251 (Div. 2)
// URL: https://codeforces.com/problemset/problem/439/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// cin 快读仍然TLE
// https://codeforces.com/contest/439/submission/111315959
// 尝试不用vector
//     RE
//     https://codeforces.com/contest/439/submission/111316060
//     目测应该是case考虑不完善
// 结论：输出NO的if条件需要加入 pe + (no - (k - p)) / 2 < p
// 仍然 RE
//     https://codeforces.com/contest/439/submission/111317575
//     问题在于输出时 pe 数量小于 p-1
// 结果WA 58
//     https://codeforces.com/contest/439/submission/111324015
//     考虑 p=0 输出单个奇数元素时同样不能超过k-1 否则最后行就出错了
//   随即AC
//     https://codeforces.com/contest/439/submission/111324667
// 简化代码如下

const int N = 100010;

int n, k, p;
int odd[N], even[N], po, pe;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> k >> p;
    for (int i = 0; i < n; ++i) {
        int x;
        cin >> x;
        if (x & 1)
            odd[po++] = x;
        else
            even[pe++] = x;
    }

    int no = po, ne = pe;
    int more = no - (k - p);

    if (k > n || no < k - p || more & 1 || pe + more / 2 < p)
        cout << "NO" << endl;
    else {
        cout << "YES" << endl;

        // 已有 po >= k - p
        // 输出奇数和部分 【不能超过k-1】
        int tot_odd = min(k - p, k - 1);
        for (int i = 0; i < tot_odd; ++i)
            cout << 1 << ' ' << odd[--po] << endl;

        // 输出偶数和部分 不能超过 k-1-tot_odd
        int tot_even = min(p, k - 1 - tot_odd);
        // 其中只用一个偶数的部分 可能为空
        int use_even = min(tot_even, pe);
        for (int i = 0; i < use_even; ++i)
            cout << 1 << ' ' << even[--pe] << endl;
        // 输出两个奇数的部分 可能为空
        for (int i = use_even; i < tot_even; ++i)
            cout << 2 << ' ' << odd[--po] << ' ' << odd[--po] << endl;

        // 输出最后一个部分
        int t = po + pe;
        cout << t;
        while (po)
            cout << ' ' << odd[--po];
        while (pe) {
            cout << ' ' << even[--pe];
        }
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

> [!NOTE] **[Codeforces C. Restore Graph](https://codeforces.com/problemset/problem/404/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常好的构造题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Restore Graph
// Contest: Codeforces - Codeforces Round #237 (Div. 2)
// URL: https://codeforces.com/problemset/problem/404/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 构造
// 边权为1 n个点 已知最短路距离 求图(边)
// 题目要求每个点的度数不超过k 其实就是bfs时层间每个点的连接限制
//
// WA 5
//    https://codeforces.com/contest/404/submission/111338086
// 考虑向下拓展时 显然已经有一条边从父连向本节点 此时最多可以再连k-1个而非k个
//    sz*nk 需要long long 否则 signed integer overflow
//    https://codeforces.com/contest/404/submission/111338653

using LL = long long;  // 避免bfs乘法溢出
using PII = pair<int, int>;
const int N = 100010;

LL n, k, idx;
PII d[N], e[N];
vector<int> deg[N];
int q[N];

bool bfs() {
    if (deg[0].size() > 1)
        return false;

    idx = 0;
    int hh = 0, tt = -1;
    for (auto& v : deg[0])
        q[++tt] = v;

    int depth = 0;
    while (hh <= tt) {
        int sz = tt - hh + 1;
        int tot = deg[++depth].size();
        int nk = (depth > 1 ? k - 1 : k);  // ATTENTION
        if (tot > (LL)sz * nk)
            return false;

        int has = 0;
        while (sz--) {
            int t = q[hh++];
            for (int i = 0; i < nk && has < tot; ++i, ++has) {
                int v = deg[depth][has];
                e[idx++] = {t, v};
                q[++tt] = v;
            }
        }
    }
    // 理想情况下 应该形成一颗树 如果中间有断层则失败
    if (idx != n - 1)
        return false;
    return true;
}

int main() {
    cin >> n >> k;
    for (int i = 1; i <= n; ++i) {
        int x;
        cin >> x;
        d[i] = {x, i};
    }

    for (int i = 1; i <= n; ++i) {
        auto& [dis, id] = d[i];
        deg[dis].push_back(id);
    }

    if (bfs()) {
        cout << idx << endl;
        for (int i = 0; i < idx; ++i)
            cout << e[i].first << ' ' << e[i].second << endl;
    } else
        cout << -1 << endl;

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

> [!NOTE] **[Codeforces Regular Bridge](http://codeforces.com/problemset/problem/550/D)** [TAG]
> 
> 题意: 
> 
> 给出一个 $k$ ，构造一个无向图，使得每个点的度数为 $k$ ，且存在一个桥
> 
> $k \leqslant 100$

> [!TIP] **思路**
> 
> 非常非常好的构造
> 
> 桥两侧对称，显然可以构造如下形式:
> 
> `[2, k-1, 1] --桥-- [1, k-1, 2]`
> 
> - 对于其中一侧（如左侧），易知 `k-1` 个点辅助割点保持桥性质，而 `2` 个点辅助 `k-1` 个点保持 `k度` 性质。同时 `2` 相互连边以保持辅助点的 `k度` 性质
> 
> - 现在 `k-1` 个点每个点已有 `3` 条边。还需要在除自己之外的 `k-2` 个点里连接 `k-3` 条边【**重点**】
> 
> - 原本想的每个点和它下一个点都不连 ==> 因为连续性，这样会造成中间的点与其 `左/右` 两个点都没有相连，最终边数不够
> 
> - **应当每隔一个点与其下一个点不相连（每隔一个点删除这个点与下一个点直接相连的边）** ==> **思考**
> 
> 核心在于：**对于剩下的 `k-1` 个点要连 `k-3` 条边的处理思路**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Regular Bridge
// Contest: Codeforces - Codeforces Round #306 (Div. 2)
// URL: https://codeforces.com/problemset/problem/550/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

int k;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> k;

    if (k & 1) {
        cout << "YES" << endl;
        int subver = k - 1;  // 和割点直接相连的必然有 k-1 个
        // 如果这样的点存在，额外需要两个来帮助这样的点保持k特性
        if (subver)
            subver += 2;
        int singleVer = subver + 1;
        int ver = singleVer * 2;
        int edges = ver * k / 2;
        cout << ver << ' ' << edges << endl;

        int cnt = 0;
        cnt++;
        cout << singleVer << ' ' << singleVer + 1 << endl;

        if (subver) {  // must > 2

            int maxv = ver + 1;
            cnt += 2;
            cout << 1 << ' ' << 2 << endl;
            cout << maxv - 1 << ' ' << maxv - 2 << endl;
            for (int i = 3; i <= subver; ++i) {
                cnt += 6;
                cout << singleVer << ' ' << i << endl;
                cout << 1 << ' ' << i << endl;
                cout << 2 << ' ' << i << endl;
                cout << singleVer + 1 << ' ' << maxv - i << endl;
                cout << maxv - 1 << ' ' << maxv - i << endl;
                cout << maxv - 2 << ' ' << maxv - i << endl;
            }
            // between subver
            // 这k-1个点分别已有3条边，只需要再加k-3条即可  (k为奇数)
            // 假定所有的k-1个点和下一个点都不连 ==>【ATTENTION】
            // 错就错在这里，并不能每个点都和下一个点不连【较显然】
            // 应当是每隔一个点，和下一个点不连

            // ATTENTION: 截止此时，k-1个点每个连k-3条边的思路都是正确的
            // 因为除去自己之外还有 k-2 个点，故必然可以
            // 接下来是删边方式，显然需要隔一个删一个，而非每个都删它与下一个紧邻的边
            for (int i = 3; i <= subver; ++i)
                for (int j = i + 1; j <= subver; ++j) {
                    if ((i & 1) && j == i + 1)
                        continue;
                    cnt += 2;
                    cout << i << ' ' << j << endl;
                    cout << maxv - i << ' ' << maxv - j << endl;
                }
            // cout << "DEBUG edges = " << edges << " cnt = " << cnt << endl;
        }
    } else {
        cout << "NO" << endl;
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

> [!NOTE] **[LeetCode 2573. 找出对应 LCP 矩阵的字符串](https://leetcode.cn/problems/find-the-string-with-lcp/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 构造
> 
> **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string findTheString(vector<vector<int>>& lcp) {
        int i = 0, n = lcp.size();
        string s(n, 0);
        for (char c = 'a'; c <= 'z'; ++ c ) {
            while (i < n && s[i])
                i ++ ;
            if (i == n) // 构造结束
                break;
            // 考虑填充第 i 个位置，以及相应的有 lcp 的所有位置（都与 i 相同）
            for (int j = i; j < n; ++ j )
                if (lcp[i][j])
                    s[j] = c;
        }
        for (int j = i; j < n; ++ j )
            if (s[j] == 0)  // 未填充完
                return "";
        
        for (int i = n - 1; i >= 0; -- i )
            for (int j = n - 1; j >= 0; -- j ) {
                int actual_lcp = 0;
                if (s[i] != s[j])
                    actual_lcp = 0;
                else if (i == n - 1 || j == n - 1)
                    actual_lcp = 1;
                else
                    actual_lcp = lcp[i + 1][j + 1] + 1;
                
                if (actual_lcp != lcp[i][j])
                    return "";
            }
        return s;
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

### 回文数构造

> [!NOTE] **[LeetCode 906. 超级回文数](https://leetcode.cn/problems/super-palindromes/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典题
> 
> 深入理解 `折半` 构造过程

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;

    bool check(LL x) {
        string s = to_string(x);
        for (int i = 0, j = s.size() - 1; i < j; ++ i , -- j )
            if (s[i] != s[j])
                return false;
        return true;
    }

    LL getFrontHalfPart(LL x) {
        string s = to_string(x);
        s = s.substr(0, s.size() / 2);
        return s.empty() ? 0 : stoll(s);
    }

    int superpalindromesInRange(string left, string right) {
        LL L = stoll(left), R = stoll(right);
        LL l = sqrt(L), r = sqrt(R), res = 0;
        for (int op = 0; op < 2; ++ op )
            // 必须 getFrontHalfPart(l) 而非直接使用 l
            // - 直接使用 l 显然是错误的，因为会遗漏部分数据
            // - 直接从 1 开始显然会 TLE
            for (LL i = getFrontHalfPart(l); i <= r; ++ i ) {
                LL x = (op ? i : i / 10);  // 先按奇数长度再按偶数长度, 这里进制 k=10
                LL conbined = i;
                while (x)
                    conbined = conbined * 10 + x % 10, x /= 10;
                if (conbined > r)  // 需要 quick fail
                    break;
                LL t = conbined * conbined;
                if (t >= L && t <= R && check(t))
                    res ++ ;
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

> [!NOTE] **[LeetCode 2217. 找到指定长度的回文数](https://leetcode-cn.com/problems/find-palindrome-with-fixed-length/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 观察 **折半** 构造

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 1e15 数据范围较大 观察知当前长度的第k大也即折半后的第k大 故直接折半构造
    using LL = long long;
    
    vector<long long> kthPalindrome(vector<int>& queries, int intLength) {
        int m = (intLength + 1) / 2;
        LL base = pow(10, m - 1), upper = base * 9;
        vector<LL> res;
        for (auto x : queries)
            // ATTENTION: base * 9, not base * 10
            // 因为这里 x 代表第几个，因为首位不可能为 0 ，个数显然无法超过 base * (10 - 1)
            if (x <= upper) {
                LL v = base + x - 1;
                string s = to_string(v);
                if (intLength & 1) {
                    int n = s.size();
                    for (int i = n - 2; i >= 0; -- i )
                        s.push_back(s[i]);
                    res.push_back(stoll(s));
                } else {
                    int n = s.size();
                    for (int i = n - 1; i >= 0; -- i )
                        s.push_back(s[i]);
                    res.push_back(stoll(s));
                }
            } else
                res.push_back(-1);
        return res;
    }
};
```

##### **C++ Other**

```cpp
class Solution {
public:
    typedef long long LL;
    vector<long long> kthPalindrome(vector<int>& q, int len) {
        vector<LL> res;
        int md = len + 1 >> 1;
        LL B = 1;
        for (int i = 1; i < md; ++i) B *= 10;
        for (int x : q) {
            LL t = B + x - 1;
            if (t >= 10 * B) {
                res.push_back(-1);
                continue;
            }
            LL y = t;
            if (len & 1) {
                y /= 10;
            }
            while (y) {
                t = t * 10 + y % 10;
                y /= 10;
            }
            res.push_back(t);
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

### 进制数 思想

> [!NOTE] **[Codeforces Pashmak and Buses](http://codeforces.com/problemset/problem/459/C)** [TAG]
> 
> 题意: 
> 
> 有 n 个学生用车，有 k 辆车（容量无限），总共 d 天，不希望有任意两个学生 d 天内都是一辆车，问能否合理安排。
> 
> $n,d≤1000$ , $k≤10^9$
> 
> 错误理解：最终没有两行完全一样
> 
> 正确理解：**构造 n 个不同的 d 位 k 进制数 ==> 每一列不完全一样**

> [!TIP] **思路**
> 
> 如果 $n>pow(k,d)$ 则无解
> 
> 否则依次构造进制数即可
> 
> 重在理解分析题意 **应用进制思想**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Pashmak and Buses
// Contest: Codeforces - Codeforces Round #261 (Div. 2)
// URL: https://codeforces.com/problemset/problem/459/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1010;

int n, k, d;

int g[N][N];

int main() {
    cin >> n >> k >> d;

    if (n > pow(k, d)) {
        cout << -1 << endl;
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        int t = i;
        for (int j = 0; j < d; ++j) {
            g[i][j] = t % k + 1;
            t /= k;
        }
    }

    for (int j = 0; j < d; ++j) {
        for (int i = 0; i < n; ++i)
            cout << g[i][j] << ' ';
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

> [!NOTE] **[Codeforces Fox and Minimal path](http://codeforces.com/problemset/problem/388/B)**
> 
> 题意: 
> 
> 要求构造一个含有 $N(1\le N\le 1000)$ 个节点的简单无向图，使得从 $1$ 号节点到 $2$ 号节点恰有 $K$ 条最短路径（$1\le K\le 10^9$）。
> 
> 输出你构造图的邻接矩阵表示。

> [!TIP] **思路**
> 
> 直观想法是分层，单个层内的节点数为对 k 因数分解的数值 ==> 据称因子较大会被卡
> 
> **转而对 k 二进制分解**
> 
> https://www.luogu.com.cn/blog/yjb/fox-and-minimal-path-ti-xie
> 
> **反复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Fox and Minimal path
// Contest: Codeforces - Codeforces Round #228 (Div. 1)
// URL: https://codeforces.com/problemset/problem/388/B
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const int N = 1e3 + 10;

int k;
int a[N], b[N], c[N];
int t = 2, h = 30;
bool g[N][N];

int main() {
    cin >> k;
    if (k == 1) {
        printf("2\nNY\nYN");
        return 0;
    }
    for (; !((1 << h) & k); h--)
        ;  //求图的层数
    for (int i = 1; i <= h; i++)
        a[i] = ++t;  //按优先从上到下，其次从右到左的顺序对每个点编号
    for (int i = 1; i <= h; i++)
        b[i] = ++t;
    for (int i = 1; i <= h; i++)
        c[i] = ++t;
    g[a[1]][1] = g[1][a[1]] = true;  //对1,2点连别
    g[b[1]][1] = g[1][b[1]] = true;
    if (k & 1)
        g[c[1]][1] = g[1][c[1]] = true;
    g[a[h]][2] = g[2][a[h]] = true;
    g[b[h]][2] = g[2][b[h]] = true;
    g[c[h]][2] = g[2][c[h]] = true;
    for (int i = 1; i < h; i++) {
        g[a[i]][a[i + 1]] = g[b[i]][b[i + 1]] = g[c[i]][c[i + 1]] =
            true;  //每一列从上到下连边
        g[a[i + 1]][a[i]] = g[b[i + 1]][b[i]] = g[c[i + 1]][c[i]] = true;
        g[a[i]][b[i + 1]] = g[b[i]][a[i + 1]] = true;  //右边两列交叉连别
        g[b[i + 1]][a[i]] = g[a[i + 1]][b[i]] = true;
        if (k & (1 << i))
            g[c[i + 1]][a[i]] = g[a[i]][c[i + 1]] = g[c[i + 1]][b[i]] =
                g[b[i]][c[i + 1]] = true;  //上文中的合并
    }
    cout << t << endl;
    for (int i = 1; i <= t; i++) {
        for (int j = 1; j <= t; j++)
            printf("%c", g[i][j] ? 'Y' : 'N');
        putchar('\n');
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

> [!NOTE] **[LeetCode 1040. 移动石子直到连续 II](https://leetcode.cn/problems/moving-stones-until-consecutive-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂分情况讨论 + 推理与构造

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> numMovesStonesII(vector<int>& stones) {
        sort(stones.begin(), stones.end());
        int n = stones.size();

        vector<int> res(2);
        // 先求最大值
        int m = stones.back() - stones[0] - (n - 1);
        // 分两种情况，取 min
        res[1] = m - min(stones[1] - stones[0] - 1, stones[n - 1] - stones[n - 2] - 1);

        // 再求最小值
        res[0] = n;
        for (int i = 0, j = 0; j < n; ++ j ) {
            while (stones[j] - stones[i] + 1 > n)
                i ++ ;
            m = j - i + 1;

            int r;
            // 中间这一段是紧邻的
            if (m == n - 1 && stones[j] - stones[i] == j - i)
                r = 2;
            // 其他情况
            else
                r = n - m;
            
            res[0] = min(res[0], r);
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

> [!NOTE] **[LeetCode 932. 漂亮数组](https://leetcode.cn/problems/beautiful-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推理 递归构造

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 考虑 i 都在左侧，j 都在右侧
    // 则左侧全放奇数，右侧全放偶数
    //
    // 但 k 位置未可知，所以必须递归执行本个构造流程

    vector<int> beautifulArray(int n) {
        if (n == 1)
            return {1};
        auto left = beautifulArray((n + 1) / 2);    // 如果 n 是奇数，左侧用的多一个
        auto right = beautifulArray(n / 2);

        vector<int> res;
        for (auto x : left)
            res.push_back(x * 2 - 1);
        for (auto x : right)
            res.push_back(x * 2);
        return res;
    }
};/
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2396. 严格回文的数字](https://leetcode.cn/problems/strictly-palindromic-number/)**
> 
> 题意: 
> 
> 问某个数 $n$ 将其分别转化为 $2 ~ n-2$ 进制，是否都能满足回文

> [!TIP] **思路**
> 
> 可以模拟，但思考推导知永远都是 $false$
> 
> 思维题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 模拟**

```cpp
class Solution {
public:
    string get(int x, int d) {
        if (!x)
            return "0";
        string ret;
        while (x)
            ret.push_back(x % d), x /= d;
        return ret;
    }
    bool isStrictlyPalindromic(int n) {
        for (int i = 2; i <= n - 2; ++ i ) {
            string s = get(n, i);
            int m = s.size();
            for (int j = 0, k = m - 1; j < k; ++ j , -- k )
                if (s[j] != s[k])
                    return false;
        }
        return true;
    }
};
```

##### **C++ 思维**

```cpp
class Solution {
public:
    bool isStrictlyPalindromic(int n) {
        return false;
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