区间类动态规划是线性动态规划的扩展，它在分阶段地划分问题时，与阶段中元素出现的顺序和由前一阶段的哪些元素合并而来有很大的关系。

令状态 $f(i,j)$ 表示将下标位置 $i$ 到 $j$ 的所有元素合并能获得的价值的最大值，那么 $f(i,j)=\max\{f(i,k)+f(k+1,j)+cost\}$，$cost$ 为将这两组元素合并起来的代价。

* * *

> [!NOTE] **区间 DP 的特点**
> 
> **合并**：即将两个或多个部分进行整合，当然也可以反过来；
> 
> **特征**：能将问题分解为能两两合并的形式；
> 
> **求解**：对整个问题设最优值，枚举合并点，将问题分解为左右两个部分，最后合并两个部分的最优值得到原问题的最优值。

* * *

## 习题



[NOIP 2007 矩阵取数游戏](https://vijos.org/p/1378)

[「IOI2000」邮局](https://www.luogu.com.cn/problem/P4767)


### 一维

> [!NOTE] **[AcWing 1068. 环形石子合并](https://www.acwing.com/problem/content/description/1070/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 区间 $dp$ 的常用技巧：一般涉及到环形区间（就是首尾可以合并），可以把整个环复制一遍，变成一条长链。
>
> 1. 状态表示：$f[i,j]$ 所有把从 $i$ 堆到 $j$ 堆合并成一堆的方案的最小/最大值
>
> 2. 状态转移：根据最后一次合并的不同：最后一次合并第 $k$ 堆：$f[l, r] = f[l,k] + f[k+1, r] + s[l,r]$
>
>    合并的过程看成：把两堆石子连一条边。
>
> 3. 区间 $dp$ 有两种实现方式：
>
>    (1) 迭代式（推荐）
>
>    - 最外层循环：循环长度
>    - 第二层循环：循环左端点
>    - 算一下右端点，再枚举分界点
>
>    (2) 记忆化搜索

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 410, INF = 0x3f3f3f3f;

int n;
int w[N], s[N];
int f[N][N], g[N][N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> w[i];
        w[i + n] = w[i];
    }

    for (int i = 1; i <= n * 2; i++) s[i] = s[i - 1] + w[i];

    memset(f, 0x3f, sizeof f);
    memset(g, -0x3f, sizeof g);

    for (int len = 1; len <= n; len++)
        for (int l = 1; l + len - 1 <= n * 2; l++) {
            int r = l + len - 1;
            if (l == r)
                f[l][r] = g[l][r] = 0;
            else {
                for (int k = l; k < r; k++) {
                    f[l][r] =
                        min(f[l][r], f[l][k] + f[k + 1][r] + s[r] - s[l - 1]);
                    g[l][r] =
                        max(g[l][r], g[l][k] + g[k + 1][r] + s[r] - s[l - 1]);
                }
            }
        }

    int minv = INF, maxv = -INF;
    for (int i = 1; i <= n; i++) {
        minv = min(minv, f[i][i + n - 1]);
        maxv = max(maxv, g[i][i + n - 1]);
    }

    cout << minv << endl << maxv << endl;

    return 0;
}
```

##### **Python**

```python
N = 410
f = [[float('inf')] * N for _ in range(N)]
g = [[float('-inf')] * N for _ in range(N)]
s = [0] * N
w = [0] * N

if __name__ == '__main__':
    n = int(input())
    nums = list(map(int, input().split()))
    w[1:] = nums * 2
    
    for i in range(1, 2 * n + 1): # 用prefix sum的思想快速求一段的数的和
        s[i] = s[i - 1] + w[i]
        
    for length in range(1, n + 1):
        l = 1
        while (l + length - 1) <= 2 * n:
            r = l + length - 1
            if length == 1:
                f[l][r] = g[l][r] = 0
            for k in range(l, r):
                f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r] + s[r] - s[l - 1])
                g[l][r] = max(g[l][r], g[l][k] + g[k + 1][r] + s[r] - s[l - 1])
            l += 1
                    
    minv, maxv = float('inf'), float('-inf')
    for i in range(1, n + 1):
        minv = min(minv, f[i][i + n - 1])
        maxv = max(maxv, g[i][i + n - 1])
        
    print(minv)
    print(maxv)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 320. 能量项链](https://www.acwing.com/problem/content/description/322/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 变种 记忆整理
>
> 1. 状态表示：$f[i,j]$ 表示将所有 $[L,R]$ 合并在一个矩阵（珠子）的方式
>
>    把2 3 5 10 （2 3） （3 5） （5 10） （10 2）===> 用 2 3 5 10 2表示 return f[1, 5]， 先考虑线性状态如何实现，再复制一遍就成环状了。 
>
> 2. 状态计算：$f[L,R] = max(f[L,K) + f[K, R] + W[L] * W[R] * W[K])$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 210, INF = 0x3f3f3f3f;

int n;
int w[N];
int f[N][N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> w[i];
        w[i + n] = w[i];
    }
    // for(int i = 1; i <= n; ++i) f[i][i] = c[i];   // 错误
    // 定义左闭右开区间 len=3本质还是len=2 [1,n+1) = [1, n]
    for (int len = 3; len <= n + 1; len++)
        for (int l = 1; l + len - 1 <= n * 2; l++) {
            int r = l + len - 1;
            for (int k = l + 1; k < r; k++)
                f[l][r] = max(f[l][r], f[l][k] + f[k][r] + w[l] * w[k] * w[r]);
        }

    int res = 0;
    for (int l = 1; l <= n; l++) res = max(res, f[l][l + n]);

    cout << res << endl;

    return 0;
}
```

##### **Python**

```python
N = 210 
w = [0] * N
f = [[0] * N for _ in range(N)]

if __name__=='__main__':
    n = int(input())
    nums = list(map(int, input().split()))
    w[1:] = 2 * nums
     
    for length in range(3, n + 2):  #踩坑1: 这道题长度可以遍历到 n+1
        for l in range(1, 2 * n - length + 2):
            r = l + length -1 
            for k in range(l + 1,r):  # 踩坑2: k 是从 l+1 开始遍历的
                f[l][r] = max(f[l][r], f[l][k] + f[k][r] + w[l] * w[k] * w[r])
    
    res = 0
    for l in range(1, n + 1):
        res = max(res, f[l][l + n])
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 479. 加分二叉树](https://www.acwing.com/problem/content/description/481/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 状态定义：$f[i, j]$ 表示中序遍历是 $w[i~j]$ 的所有二叉树的得分的最大值
>
> 2. 状态转移：以根节点的位置划分：$i <= k <= j$ 
>
>    当根节点在 k 时，最大的得分是： $f[i, k - 1] * f[k + 1, j] + w[k]$ 
>
>    $f[i, j]=max(f[i, k - 1] * f[k + 1, j] + w[k])$
>
> 3. 计算过程中，记录每个区间的最大值所对应的根节点编号，最后通过 $DFS$ 求出最大加分二叉树的前序遍历
>
> 4. 时间复杂度：状态数量是 $n^2$，每个状态计算量是 $n$, 总共是 $O(n^3)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 35;

int n;
int a[N];
int f[N][N], rt[N][N];

void dfs(int l, int r) {
    if (l > r) return;
    int v = rt[l][r];
    cout << v << ' ';
    dfs(l, v - 1);
    dfs(v + 1, r);
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    for (int i = 1; i <= n; ++ i )
        f[i][i] = a[i], rt[i][i] = i;
    
    for (int len = 2; len <= n; ++ len )
        for (int l = 1; l + len - 1 <= n; ++ l ) {
            int r = l + len - 1;
            for (int k = l; k <= r; ++ k ) {
                int left = k == l ? 1 : f[l][k - 1];
                int right = k == r ? 1 : f[k + 1][r];
                int score = left * right + a[k];
                if (f[l][r] < score) {
                    f[l][r] = score;
                    rt[l][r] = k;
                }
            }
        }
    
    cout << f[1][n] << endl;
    
    dfs(1, n);
    
    return 0;
}
```

##### **Python**

```python
N = 50
w = [0] * N 
f = [[0] * N for _ in range(N)]
g = [[0] * N for _ in range(N)]

def dfs(l, r):
    if l > r:
        return
    root = g[l][r]
    print(root, end = " ")
    dfs(l, root - 1)
    dfs(root + 1, r)

if __name__ == '__main__':
    n = int(input())
    w[1:] = list(map(int, input().split()))
    
    # 初始化（特例）
    for i in range(1, n + 1):
        f[i][i] = w[i]
        g[i][i] = i 
        
    # 长度就可以从2开始枚举
    for len in range(2, n + 1):
        for l in range(1, n - len + 2):
            r = l + len - 1 
            for k in range(l, r + 1):
                left = 1 if k == l else f[l][k - 1]
                right = 1 if k == r else f[k + 1][r]
                score = left * right + w[k]
                if f[l][r] < score:
                    f[l][r] = score
                    g[l][r] = k

    print(f[1][n])
    dfs(1, n)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1069. 凸多边形的划分](https://www.acwing.com/problem/content/description/1071/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1~n 就覆盖了所有情况 无需重复2n
>
> 高精度
>
> 1. 状态表示：f[l, r] 所有将(l,l+1),(l+1,l+2)...(r-1,r),(r,l) 这个多边形划分成若干个三角形的所有的方案；属性：min
>
> 2. 状态转移：划分的时候 可以根据其中任何一条边来进行划分。（每条边最后都会属于某个三角形）
>
>    f[l, r] = f[l,k] + f[k,r] + w[l] * w[r] * w[k]

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

const int N = 55, M = 35, INF = 1e9;

int n;
int w[N];
LL f[N][N][M];

void add(LL a[], LL b[]) {
    static LL c[M];
    memset(c, 0, sizeof c);
    for (int i = 0, t = 0; i < M; i++) {
        t += a[i] + b[i];
        c[i] = t % 10;
        t /= 10;
    }
    memcpy(a, c, sizeof c);
}

void mul(LL a[], LL b) {
    static LL c[M];
    memset(c, 0, sizeof c);
    LL t = 0;
    for (int i = 0; i < M; i++) {
        t += a[i] * b;
        c[i] = t % 10;
        t /= 10;
    }
    memcpy(a, c, sizeof c);
}

int cmp(LL a[], LL b[]) {
    for (int i = M - 1; i >= 0; i--)
        if (a[i] > b[i])
            return 1;
        else if (a[i] < b[i])
            return -1;
    return 0;
}

void print(LL a[]) {
    int k = M - 1;
    while (k && !a[k]) k--;
    while (k >= 0) cout << a[k--];
    cout << endl;
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> w[i];

    LL temp[M];
    for (int len = 3; len <= n; len++)
        for (int l = 1; l + len - 1 <= n; l++) {
            int r = l + len - 1;
            f[l][r][M - 1] = 1;
            for (int k = l + 1; k < r; k++) {
                memset(temp, 0, sizeof temp);
                temp[0] = w[l];
                mul(temp, w[k]);
                mul(temp, w[r]);
                add(temp, f[l][k]);
                add(temp, f[k][r]);
                if (cmp(f[l][r], temp) > 0) memcpy(f[l][r], temp, sizeof temp);
            }
        }

    print(f[1][n]);

    return 0;
}
```

##### **Python**

```python
N = 55 
M = 35 # 位数是30位，保险开到35
w = [0] * N
f = [[float('inf')] * N for _ in range(N)]

if __name__ == '__main__':
    n = int(input())
    w[1:] = list(map(int, input().split()))
    
    for len in range(1, n + 1): 
        for l in range(1, n - len +2):
            r = len + l - 1
            if len < 3:  # 从长度为 3 开始枚举，长度为 3 才能构成三角形，如果小于3，那f就是0 初始化为0即可（后续状态会由这里转移过来）
                f[l][r] = 0
            for k in range(l + 1, r):
                f[l][r] = min(f[l][r], f[l][k] + f[k][r] + w[l] * w[k] * w[r])
    print(f[1][n])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu [HNOI2010]合唱队](https://www.luogu.com.cn/problem/P3205)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 统计方案数 流程

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 显然区间dp 如下实现即可
//
// 对于子区间长度为1的情况写了好几个if判断 不够优雅
// 实际上只有一个人的方案只有一种 可以直接按照默认在左侧进来
// 即 初始化 f[i][i][0] = 1, f[i][i][1] = 0;即可 略

const int N = 1010, MOD = 19650827;

int n;
int a[N];
int f[N][N][2];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    for (int i = 1; i <= n; ++ i )
        f[i][i][0] = f[i][i][1] = 1;
    
    for (int len = 2; len <= n; ++ len )
        for (int l = 1; l + len - 1 <= n; ++ l ) {
            int r = l + len - 1;
            
            {
                int v = a[l];
                int tl = l + 1, tr = r, vl = 0, vr = 0;
                if (v < a[tl])
                    vl = f[tl][tr][0];
                if (v < a[tr])
                    vr = f[tl][tr][1];
                if (tl == tr)
                    f[l][r][0] = (f[l][r][0] + vl) % MOD;
                else
                    f[l][r][0] = (f[l][r][0] + (vl + vr) % MOD) % MOD;
            }
            {
                int v = a[r];
                int tl = l, tr = r - 1, vl = 0, vr = 0;
                if (v > a[tl])
                    vl = f[tl][tr][0];
                if (v > a[tr])
                    vr = f[tl][tr][1];
                if (tl == tr)
                    f[l][r][1] = (f[l][r][1] + vl) % MOD;
                else
                    f[l][r][1] = (f[l][r][1] + (vl + vr) % MOD) % MOD;
            }
        }
        
    cout << (f[1][n][0] + f[1][n][1]) % MOD << endl;
    
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

> [!NOTE] **[Luogu Zuma](https://www.luogu.com.cn/problem/CF607B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典
> 
> 题面回文但非回文处理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 510, INF = 0x3f3f3f3f;

int n, a[N], f[N][N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    memset(f, 0x3f, sizeof f);
    for (int i = 1; i <= n; ++ i )
        f[i][i] = 1;
    
    for (int len = 2; len <= n; ++ len )
        for (int l = 1; l + len - 1 <= n; ++ l ) {
            int r = l + len - 1;
            if (l + 1 == r) {
                // f[l][r - 1] = f[l + 1][r] = 1;
                f[l][r] = min(f[l][r - 1], f[l + 1][r]) + (a[l] != a[r]);
            } else {
                if (a[l] == a[r])
                    f[l][r] = f[l + 1][r - 1];
                for (int k = l; k < r; ++ k )
                    f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r]);
            }
        }
    cout << f[1][n] << endl;
    
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

> [!NOTE] **[LeetCode 87. 扰乱字符串](https://leetcode-cn.com/problems/scramble-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> **区间dp**
>
> 1. 状态表示：$f[i, j, k]$ 表示 $s_1[i ～ i+k-1]$ $（从 i 开始长度为 k 的区间）$与  $s_1[j ～ j+k-1]$ 所有匹配反感的集合；属性：集合是否为空。 ($k$ 表示字符串的长度)
>
> 2. 状态转移：将 $f[i, j, k]$ 表示的集合按照 $s_1$ 第一段的长度划分为 $k - 1$ 类，也就是第一段长度可以是：$[1, 2, ..., k - 1]$，令 $s_1$ 第一段的长度为 $u$, 由于可以扰乱位置，所以 $s_1[i ～ i+k-1]$ 和 $s_1[j ～ j+k-1]$ 有两种匹配反感：前前匹配&&后后匹配，或者前后匹配&&后前匹配，故有：
>
>    1）$f[i, j, u] and f[i + u, i + u, k - u]$
>
>    2）$f[i, j + k - u, u] and f[i + u, j, k - u]$
>
> 3. 时间复杂度：状态数：$O(n^3)$, 状态转移计算量是 $O(n)$, 总的：$O(n^4)$
>
> **暴力dfs**
>
> 递归判断两个字符串是否可以相互转化。
>
> 先去枚举s1 第一次分割的情况：左边：$i$ 个节点, 右边：$n - i$ 个节点
>
> 1. 如果 $s_1$ 的根节点不翻转
>
>    $(s1[:i], s2[:i]) and (s1[i:], s2[i:])$
>
> 2. 如果 $s_1$ 的根节点翻转
>
>    $(s1[:i], s2[-i:]) and (s1[i:], s2[:-i])$
>
> 如果s2可以有s1得到的话，意味着s2的右边i个字符 可以通过 s1的左边的i个字符干扰得到; s2的左边的(n - i)个字符 可以通过s1的右边(n - i)个字符干扰得到。



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 递归**

```cpp
class Solution {
public:
    bool isScramble(string s1, string s2) {
        if (s1 == s2) return true;
        string bs1 = s1, bs2 = s2;
        sort(bs1.begin(), bs1.end()), sort(bs2.begin(), bs2.end());
        if (bs1 != bs2) return false;

        int n = s1.size();
        for (int i = 1; i <= n - 1; i ++ ) {
            if (isScramble(s1.substr(0, i), s2.substr(0, i)) &&
                isScramble(s1.substr(i), s2.substr(i))) return true;
            if (isScramble(s1.substr(0, i), s2.substr(n - i)) &&
                isScramble(s1.substr(i), s2.substr(0, n - i))) return true;
        }

        return false;
    }
};
```

##### **C++ 1**

```cpp
class Solution {
public:
    bool isScramble(string s1, string s2) {
        int l1 = s1.size(), l2 = s2.size();
        if (l1 != l2) return false;
        if (!l1) return true;
        vector<vector<vector<bool>>> dp(l1 + 1, vector<vector<bool>>(l1, vector<bool>(l1, false)));
        for (int i = 0; i < l1; ++ i )
            for (int j = 0; j < l1; ++ j )
                dp[1][i][j] = s1[i] == s2[j];
              
        for (int len = 2; len <= l1; ++ len ) {
            for (int i = 0; i < l1 && i + len - 1 < l1; ++ i ) {
                for (int j = 0; j < l1 && j + len - 1 < l1; ++ j ) {
                    for (int k = 1; k < len; ++ k ) {
                        if(dp[k][i][j] && dp[len - k][i + k][j + k]) {
                            dp[len][i][j] = true;
                            break;
                        }
                        if(dp[k][i][j + len - k] && dp[len - k][i + k][j]) {
                            dp[len][i][j] = true;
                            break;
                        }
                    }
                }
            }
        }
        return dp[l1][0][0];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    bool isScramble(string s1, string s2) {
        int n = s1.size();
        vector<vector<vector<bool>>> f(n, vector<vector<bool>>(n, vector<bool>(n + 1)));
        for (int k = 1; k <= n; k ++ )
            for (int i = 0; i + k - 1 < n; i ++ )
                for (int j = 0; j + k - 1 < n; j ++ ) {
                    if (k == 1) {
                        if (s1[i] == s2[j]) f[i][j][k] = true;
                    } else {
                        for (int u = 1; u < k; u ++ ) {
                            if (f[i][j][u] && f[i + u][j + u][k - u] || f[i][j + k - u][u] && f[i + u][j][k - u]) {
                                f[i][j][k] = true;
                                break;
                            }
                        }
                    }
                }
        return f[0][0][n];
    }
};
```

##### **Python-区间dp**

```python
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        n, m = len(s1), len(s2)
        if n != m:return False
        if not n:return True
        f = [[[False] * (n + 1) for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if s1[i] == s2[j]:
                    f[i][j][1] = True

        for k in range(2, n + 1):
            for i in range(n + 1 - k):
                for j in range(n + 1 - k):
                    for u in range(1, k):
                        if (f[i][j][u] and f[i + u][j + u][k - u]) or (f[i][j + k -u][u] and f[i + u][j][k - u]):
                            f[i][j][k] = True
                            break
        return f[0][0][n]
```

##### **Python-dfs**

```python
import functools
class Solution:
    @functools.lru_cache(None)
    def isScramble(self, s1: str, s2: str) -> bool:
        if s1 == s2:return True
        if sorted(s1) != sorted(s2):
            return False
        for i in range(1, len(s1)):
            # 分割点：s1左==s2左 && s1右==s2右
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True
            # 分割点：（翻转）s1左==s2右 && s1右==s2左
            if self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:-i]):
                return True
        return False
```

##### 

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 暴搜方案数太多，考虑用dp
>
> 1. 状态表示: $f[i, j]$ 表示所有由 $s_1[1-i]$ 和 $s_2[1-j]$ 交错形成 $s_3[1-i+j]$ 的方案；属性：集合是否非空；$true/false$
>
> 2. 状态计算：
>
>    如果 $s_3[i+j]$ 匹配 $s_1[i]$ ，则问题就转化成了 $f[i−1, j]$；
>
>    如果 $s_3[i+j]$ 匹配 $s_2[j]$，则问题就转化成了 $f[i,j−1]$。两种情况只要有一种为真，则 $f[i, j]$ 就为真

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int l1 = s1.size(), l2 = s2.size(), l3 = s3.size();
        if (l1 + l2 != l3) return false;
        vector<vector<bool>> f(l1 + 1, vector<bool>(l2 + 1));
        f[0][0] = true;
        for (int i = 1; i <= l1; ++ i )
            if (f[i - 1][0] && s1[i - 1] == s3[i - 1])
                f[i][0] = true;
        for (int i = 1; i <= l2; ++ i )
            if (f[0][i - 1] && s2[i - 1] == s3[i - 1])
                f[0][i] = true;
        for (int i = 1; i <= l1; ++ i )
            for (int j = 1; j <= l2; ++ j )
                f[i][j] = (f[i - 1][j] && s1[i - 1] == s3[i + j - 1]) || (f[i][j - 1] && s2[j - 1] == s3[i + j - 1]);
        return f[l1][l2];
    }
};
```

##### **C++ 2**

```cpp
// yxc
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int n = s1.size(), m = s2.size();
        if (s3.size() != n + m) return false;

        vector<vector<bool>> f(n + 1, vector<bool>(m + 1));
        s1 = ' ' + s1, s2 = ' ' + s2, s3 = ' ' + s3;
        for (int i = 0; i <= n; i ++ )
            for (int j = 0; j <= m; j ++ )
                if (!i && !j) f[i][j] = true;
                else {
                    if (i && s1[i] == s3[i + j]) f[i][j] = f[i - 1][j];
                    if (j && s2[j] == s3[i + j]) f[i][j] = f[i][j] || f[i][j - 1];
                }

        return f[n][m];
    }
};
```

##### **Python**

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        n, m = len(s1), len(s2)
        if len(s3) != (n + m):return False
        f = [[False] * (m + 1) for _ in range(n + 1)]
        s1, s2, s3 = ' ' + s1, ' ' + s2, ' ' + s3

        # 初始化
        f[0][0] = True
        for i in range(1, n + 1):
            if f[i - 1][0] and s1[i] == s3[i]:
                f[i][0] = True
        for i in range(1, m + 1):
            if f[0][i - 1] and s2[i] ==s3[i]:
                f[0][i] = True
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                f[i][j] = (s1[i] == s3[i + j] and f[i - 1][j]) or (s2[j] == s3[i + j] and f[i][j - 1])
        return f[n][m]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 状态表示： $f[i, j]$ 表示戳爆从第 $i$ 到第 $j$ 个气球得到的最大金币数。
>
> 2. 状态转移： 假设在 $[i,j]$ 范围里最后戳破的一个气球是 $k$ 。注意此时 [i, j] 区间里只有 k 这个气球了，所以它在被扎破的瞬间的代价：$num[i−1] ∗ nums[k] ∗ nums[j+1] $
>
>    $f[i,j]=max(f[i, j], f[i, k−1] + num[i−1] ∗ nums[k] ∗ nums[j+1] + f[k+1, j])$
>
> 3. 时间复杂度：时间复杂度 $O(n^3)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        vector<int> a(n + 2, 1);
        for (int i = 1; i <= n; i ++ ) a[i] = nums[i - 1];
        vector<vector<int>> f(n + 2, vector<int>(n + 2));
        for (int len = 3; len <= n + 2; len ++ )
            for (int i = 0; i + len - 1 <= n + 1; i ++ ) {
                int j = i + len - 1;
                for (int k = i + 1; k < j; k ++ )
                    f[i][j] = max(f[i][j], f[i][k] + f[k][j] + a[i] * a[k] * a[j]);
            }

        return f[0][n + 1];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        vector<vector<int>> f(n + 2, vector<int>(n + 2));
        // for (int i = 1; i <= n; ++ i ) f[i][i] = nums[i]; // NOT
        for (int len = 1; len <= n; ++ len )
            for (int l = 1; l + len - 1 <= n; ++ l ) {
                int r = l + len - 1;
                for (int k = l; k <= r; ++ k )
                    f[l][r] = max(f[l][r], f[l][k - 1] + f[k + 1][r] + nums[l - 1] * nums[r + 1] * nums[k]);
            }
        return f[1][n];
    }
};
```


##### **Python**

```python
class Solution(object):
    def maxCoins(self, nums):
        n = len(nums)
        nums = [1] + nums + [1]
        f = [[0] * (n + 2) for i in range(n + 2)]

        for length in range(1, n + 1):
          	# l是从下标为1的气球开始的
            for l in range(1, n - length + 2):
                r = l + length - 1
                for k in range(l, r + 1):
                    f[l][r] = max(f[l][r], f[l][k - 1] + f[k + 1][r] + nums[l - 1] * nums[k] * nums[r + 1])
        return f[1][n]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 375. 猜数字大小 II](https://leetcode-cn.com/problems/guess-number-higher-or-lower-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 状态定义: $f[i, j]$ 表示在区间 $[i, j]$ 内所有target以及对应的所有选法。状态表示这些集合所用钱最多情况下的最小值。
> 2. 状态转移：假设选的数是 $k$ , 那么用钱最多的是：$max(f[i, k - 1], f[k + 1, j]) + k$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    const int inf = 1e9;
    int getMoneyAmount(int n) {
        int res = 0;
        vector<vector<int>> f(n + 1, vector<int>(n + 1, inf));
        for (int i = 0; i <= n; ++i) f[i][i] = 0;
        for (int len = 2; len <= n; ++ len ) {
            for (int l = 1; l + len - 1 <= n; ++ l ) {
                int r = l + len - 1;
                // 使用这种写法而非：
                // f[i][j] = INT_MAX;
                // for (int k = i; k <= j; k ++ )
                //     f[i][j] = min(f[i][j], max(f[i][k - 1], f[k + 1][j]) + k);
                // 的好处：不需要数组开到n+2 且 也还算清晰
                f[l][r] = min(l + f[l + 1][r], r + f[l][r - 1]);
                for (int k = l + 1; k < r; ++ k )
                    f[l][r] = min(f[l][r], k + max(f[l][k - 1], f[k + 1][r]));
            }
        }
        return f[1][n];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int getMoneyAmount(int n) {
        vector<vector<int>> f(n + 2, vector<int>(n + 2));
        for (int len = 2; len <= n; len ++ )
            for (int i = 1; i + len - 1 <= n; i ++ ) {
                int j = i + len - 1;
                f[i][j] = INT_MAX;
                for (int k = i; k <= j; k ++ )
                    f[i][j] = min(f[i][j], max(f[i][k - 1], f[k + 1][j]) + k);
            }
        return f[1][n];
    }
};
```

##### **Python**

```python
class Solution(object):
    def getMoneyAmount(self, n):
        f = [[0] * (n + 2) for _ in range(n + 2)]

        # 我们的最小子问题得是从2开始，因为从猜数范围从1开始的话，那就不用猜了
        for length in range(2, n + 1):
            for l in range(1, n + 2 - length):
                r = l + length - 1
                f[l][r] = float("inf")
                for k in range(l, r + 1):
                    f[l][r] = min(f[l][r], max(f[l][k - 1], f[k + 1][r]) + k)
        return f[1][n]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 486. 预测赢家](https://leetcode-cn.com/problems/predict-the-winner/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 经典
>
> **TODO 为何博弈不可的证明**
>
> 1. 状态定义: $f[i, j]$ 表示剩余数组为 $[i, j]$，并且你是先手，可以获得的分数减去对手分数的最大值。
>
> 2. 状态转移：
>
>    1）如果你先拿的 $a[i]$, 对手可以要么拿 $a[i +1]$ 要么拿 $a[j]$，他也会选择最有利自己的方式：$f[i +1, j]$, 这个表示对手减去你分数的最大值，但是求的是你的分数减去对手的分数，所以需要取反。
>
>    你的得分最大值是：$-f[i + 1, j] + a[i]$
>
>    2）如果你先拿的 $a[j]$, 同理：你的得分最大值：$-f[i, j - 1] + a[j]$
>
> 3. 区间dp可以用循环写，也可以用记忆化搜索。（循环写法简单些）

<details> 
<summary>详细代码</summary>
<!-- tabs:start -->


##### **C++**

```cpp
class Solution {
public:
    bool PredictTheWinner(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> f(n, vector<int>(n));
        for (int len = 1; len <= n; len ++ ) {
            for (int i = 0; i + len - 1 < n; i ++ ) {
                int j = i + len - 1;
                if (len == 1) f[i][j] = nums[i];
                else {
                    f[i][j] = max(nums[i] - f[i + 1][j], nums[j] - f[i][j - 1]);
                }
            }
        }
        return f[0][n - 1] >= 0;
    }
};
```

##### **Python**

```python
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        f = [[0] * n for _ in range(n)]
        
        for length in range(1, n + 1):
            for l in range(n + 1 - length):
                r = l + length - 1
                if length == 1:
                    f[l][r] = nums[l]
                else:
                    f[l][r] = max(nums[l] - f[l + 1][r], nums[r] - f[l][r - 1])

        return f[0][n - 1] >= 0
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 664. 奇怪的打印机](https://leetcode-cn.com/problems/strange-printer/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 经典区间dp
>
> 1. 状态表示：打印出 $[i, j]$ 的所有方案的集合；属性：$min$
> 2. 状态转移：根据长度，枚举所有可能的组合。枚举之前，做一个判断，如果 $s[i] == s[j]$，那就可以利用相同字符可以一起打印的特性进行转移：$f[i, j] = f[i, j - 1]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    const int INF = 0x3f3f3f3f;

    int strangePrinter(string s) {
        int n = s.size();
        vector<vector<int>> f(n + 1, vector<int>(n + 1, INF));

        for (int i = 1; i <= n; ++ i )
            f[i][i] = 1;
        
        for (int len = 2; len <= n; ++ len )
            for (int l = 1; l + len - 1 <= n; ++ l ) {
                int r = l + len - 1;
                if (s[l - 1] == s[r - 1])
                    f[l][r] = min(f[l][r], f[l][r - 1]);
                for (int k = l; k < r; ++ k )
                    f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r]);
            }
        return f[1][n];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    const int INF = 0x3f3f3f3f;
    int strangePrinter(string s) {
        int n = s.size();
        if (!n) return 0;
        vector<vector<int>> f(n + 1, vector<int>(n + 1));
        for (int i = 1; i <= n; ++ i )
            f[i][i] = 1;
        
        for (int len = 2; len <= n; ++ len )
            for (int l = 1; l + len - 1 <= n; ++ l ) {
                int r = l + len - 1;
                f[l][r] = f[l][r - 1] + 1;
                for (int i = l; i < r; ++ i )
                    if (s[i - 1] == s[r - 1])
                        f[l][r] = min(f[l][r], f[l][i - 1] + f[i + 1][r]);  // not r - 1
            }
        return f[1][n];
    }
};
```

##### **Python**

```python
class Solution:
    def strangePrinter(self, s: str) -> int:
        n = len(s)
        f = [[float('inf')] * (n + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            f[i][i] = 1
        
        for length in range(2, n + 1):
            for l in range(1, n + 2 - length):
                r = l + length - 1
                if s[l - 1] == s[r - 1]:
                    f[l][r] = f[l][r - 1]
                for k in range(l, r):
                    f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r])
        return f[1][n]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 730. 统计不同回文子序列](https://leetcode-cn.com/problems/count-different-palindromic-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **区间dp进阶 重复做**
> 
> **有一个更好的解法**
> 
> trick ==> **左右区间并非简单的 `i/j` 而是队列里的头和尾**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 更好**

```cpp
// yxc
class Solution {
public:
    int countPalindromicSubsequences(string s) {
        int n = s.size(), MOD = 1e9 + 7;
        vector<vector<int>> f(n + 2, vector<int>(n + 2, 1));
        for (int i = 1; i <= n; i ++ ) f[i][i] ++ ;
        for (int len = 2; len <= n; len ++ ) {
            deque<int> q[4];
            for (int i = 1; i <= n; i ++ ) {
                q[s[i - 1] - 'a'].push_back(i);
                int j = i - len + 1;
                if (j >= 1) {
                    for (int k = 0; k < 4; k ++ ) {
                        while (q[k].size() && q[k].front() < j) q[k].pop_front();
                        if (q[k].size()) {
                            f[j][i] ++ ;
                            int l = q[k].front(), r = q[k].back();
                            if (l < r)
                                f[j][i] = (f[j][i] + f[l + 1][r - 1]) % MOD;
                        }
                    }
                }
            }
        }
        return (f[1][n] + MOD - 1) % MOD;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    const int MOD = 1e9 + 7;
    int countPalindromicSubsequences(string S) {
        int n = S.size();
        vector<vector<int>> f(n + 1, vector<int>(n + 1));
        for (int i = 1; i <= n; ++ i )
            f[i][i] = 1;
        for (int len = 2; len <= n; ++ len )
            for (int l = 1; l + len - 1 <= n; ++ l ) {
                int r = l + len - 1;
                if (S[l - 1] == S[r - 1]) {
                    f[l][r] = 2 * f[l + 1][r - 1];
                    int L = l + 1, R = r - 1;
                    while (L <= R && S[L - 1] != S[l - 1])
                        ++ L ;
                    while (L <= R && S[R - 1] != S[r - 1])
                        -- R ;
                    if (L > R)
                        f[l][r] += 2;
                    else if (L == R)
                        f[l][r] += 1;
                    else
                        f[l][r] -= f[L + 1][R - 1];
                } else
                    f[l][r] = f[l + 1][r] + f[l][r - 1] - f[l + 1][r - 1];
                f[l][r] = (f[l][r] % MOD + MOD) % MOD;
            }
        return f[1][n];
    }
};
```

##### **Python-dp+单调队列**

```python
```



##### **Python**

```python
class Solution:
    def countPalindromicSubsequences(self, S: str) -> int:
        n = len(S)
        mod = int(1e9 +7)
        f = [[0] * (n + 1) for _ in range(n + 1)]
        
        # base case
        for i in range(1, n + 1):
            f[i][i] = 1
        for length in range(2, n + 1):
            for i in range(n - length + 1, 0, -1):  # 逆序遍历
                j = i + length - 1
                # 两端不能同时参与构成回文子序列
                if S[i - 1] != S[j - 1]:
                    f[i][j] = f[i + 1][j] + f[i][j - 1] - f[i + 1][j - 1]
                else:
                    f[i][j] = f[i + 1][j - 1] * 2
                    # check [i + 1, j - 1] 区间里是否存在和S[i - 1]相等的值
                    l, r = i + 1, j - 1
                    while l <= r and S[l - 1] != S[i - 1]: l += 1
                    while l <= r and S[r - 1] != S[i - 1]: r -= 1
                    # 不存在
                    if l > r: 
                        f[i][j] += 2
                    # 存在一个
                    elif l == r: 
                        f[i][j] += 1
                    # 存在 >= 2
                    else: 
                        f[i][j] -= f[l + 1][r - 1]
                    
                f[i][j] %= mod
        return f[1][n]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1246. 删除回文子数组](https://leetcode-cn.com/problems/palindrome-removal/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> Microsoft题库
>
> **Key**：$if$ $s[i] == s[j]$, 那么这两个字符可以在删除 $[i+1, j-1]$ 区间时顺带删掉: $f[i, j] = f[i+1, j-1]$
>
> 如果两个字符不相等，那就枚举中间变量 $k$, 以 $k$ 作为分割, $f[i, j] = f[i, k] + f[k + 1, j]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int minimumMoves(vector<int>& arr) {
        int n = arr.size();
        vector<vector<int>> f(n + 1, vector<int>(n + 1, inf));
        for (int i = 1; i <= n; ++i) f[i][i] = 1;
        for (int len = 2; len <= n; ++len)
            for (int l = 1; l + len - 1 <= n; ++l) {
                int r = l + len - 1;
                for (int k = l; k < r; ++k)
                    f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r]);
                if (arr[l - 1] == arr[r - 1])
                    if (len == 2)
                        f[l][r] = 1;
                    else
                        f[l][r] = min(f[l][r], f[l + 1][r - 1]);
            }
        return f[1][n];
    }
};
```

##### **Python**

```python
class Solution:
    def minimumMoves(self, arr: List[int]) -> int:
        n = len(arr)
        f = [[float('inf')] * (n + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            f[i][i] = 1
        
        for length in range(2, n + 1):
            for l in range(1, n - length + 2):
                r = l + length - 1
                if arr[l - 1] == arr[r - 1]:
                    if length == 2:
                        f[l][r] = 1
                    else:
                        f[l][r] = min(f[l][r], f[l + 1][r - 1])
                for k in range(l, r):
                    f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r])
        return f[1][n]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1531. 压缩字符串 II](https://leetcode-cn.com/problems/string-compression-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp 解释参考 [题解](https://leetcode-cn.com/problems/string-compression-ii/solution/zhuan-zai-dp-by-mike-meng/)
> 
> 状态定义 十分十分 trick 的状态转移与推理 ==> `前 i 个字符中最多选择 j 个字符进行删除`
> 
> 非常经典 细节居多 反复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
int dp[111][111];
class Solution {
public:
    int getLengthOfOptimalCompression(string s, int k) {
        int n = s.size();
        memset(dp, 0x3f, sizeof(dp));
        dp[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= k + 1; j++) {
                // 1. 删除字符 i
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]);
                // 2. 保留字符 i , 后续尽量选择保留与字符 i 相同的字符 ==> trick
                int cnt = 0, del = 0;
                for (int l = i; l <= n; l++) {
                    cnt += s[l - 1] == s[i - 1];
                    del += s[l - 1] != s[i - 1];
                    if (j + del - 1 <= k)
                        dp[l][j + del - 1] =
                            min(dp[l][j + del - 1], dp[i - 1][j - 1] + 1 +
                                                        (cnt >= 100  ? 3
                                                         : cnt >= 10 ? 2
                                                         : cnt >= 2  ? 1
                                                                     : 0));
                }
            }
        }
        return dp[n][k];
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

> [!NOTE] **[LeetCode 1770. 执行乘法运算的最大分数](https://leetcode-cn.com/problems/maximum-score-from-performing-multiplication-operations/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 状态定义：$f[l][r]$ 表示左侧使用 l 个右侧使用 r 个的分数 与传统区间 dp 定义略有不同
>
> 类似状压递推的**状态转移思想** 十分经典
>
> 2. 状态转移：除了$base case$外，$f[i, j]$ 由 $f[i - 1, j]$ 或 $f[i, j - 1]$ 转移得到，取两个中得分最大的一个。
> 3. 出口：满足 $i + j == m$ 的所有组合中的最大值。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    const static int N = 1010;
    int f[N][N];
    
    int maximumScore(vector<int>& nums, vector<int>& mp) {
        int n = nums.size(), m = mp.size();
        memset(f, 0xcf, sizeof f);
        
        f[0][0] = 0;
        for (int i = 1; i <= m; ++ i )
            f[0][i] = f[0][i - 1] + nums[n - i] * mp[i - 1];
        for (int i = 1; i <= m; ++ i )
            f[i][0] = f[i - 1][0] + nums[i - 1] * mp[i - 1];
        
        for (int len = 1; len <= m; ++ len )
            for (int l = 1; l < len; ++ l ) {
                int r = len - l;
                // ATTENTION
                f[l][r] = max(f[l - 1][r] + nums[l - 1] * mp[len - 1], f[l][r - 1] + nums[n - r] * mp[len - 1]);
            }
        int res = INT_MIN;
        for (int i = 0; i <= m; ++ i )
            res = max(res, f[i][m - i]);
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int maximumScore(vector<int>& nums, vector<int>& mp) {
        int n = nums.size(), m = mp.size();
        vector<vector<int>> f(m + 1, vector<int>(m + 1, -1e9));
        f[0][0] = 0;
        for (int i = 1; i <= m; ++ i )
            f[0][i] = f[0][i - 1] + nums[n - i] * mp[i - 1];
        for (int i = 1; i <= m; ++ i )
            f[i][0] = f[i - 1][0] + nums[i - 1] * mp[i - 1];

        for (int l = 1; l <= m; ++ l )
            for (int r = 1; l + r <= m; ++ r ) {
                f[l][r] = max(f[l - 1][r] + mp[l + r - 1] * nums[l - 1],
                    f[l][r - 1] + mp[l + r - 1] * nums[n - r]);
            }

        int res = -1e9;
        for (int i = 0; i <= m; ++i)
            res = max(res, f[i][m - i]);
        return res;
    }
};
```

##### **C++ 3**

```cpp
class Solution {
public:
    int maximumScore(vector<int>& nums, vector<int>& mp) {
        int n = nums.size(), m = mp.size();
        vector<vector<int>> f(m + 1, vector<int>(m + 1, -1e9));
        f[0][0] = 0;
        for (int l = 0; l < m; ++l)
            for (int r = 0; l + r < m; ++r) {
                // ATTENTION 取 max 必不可少
                f[l + 1][r] = max(f[l + 1][r], f[l][r] + mp[l + r] * nums[l]);
                f[l][r + 1] = max(f[l][r + 1], f[l][r] + mp[l + r] * nums[n - 1 - r]);
            }
        
        int res = -1e9;
        for (int i = 0; i <= m; ++i)
            res = max(res, f[i][m - i]);
        return res;
    }
};
```

##### **Python-区间dp**

```python
class Solution:
    def maximumScore(self, nums: List[int], multi: List[int]) -> int:
        n, m = len(nums), len(multi) 
        f = [[float('-inf')] * (m + 1) for _ in range(m + 1)]
        f[0][0] = 0
        for i in range(1, m + 1):
            f[i][0] = f[i - 1][0] + nums[i - 1] * multi[i - 1]
        for i in range(1, m + 1):
            f[0][i] = f[0][i - 1] + nums[n - i] * multi[i - 1]
        
        for i in range(1, m + 1):
            for j in range(1, m - i + 1):
                f[i][j] = max(f[i - 1][j] + nums[i - 1] * multi[i + j - 1], f[i][j - 1] + nums[n - j] * multi[i + j - 1])

        res = float('-inf')
        for i in range(m + 1):
            res = max(res, f[i][m - i])
        return res
```

##### **Python-记忆化dfs**

```python
# 记忆化递归(@cache)，一定要调用cache_clear()清除缓存，否则会超时
class Solution:
    def maximumScore(self, nums: List[int], multi: List[int]) -> int:
        n, m = len(nums), len(multi) 

        @cache
        def dfs(cur, l, r):
            if cur == m:
                return 0
            return max(multi[cur] * nums[l] + dfs(cur + 1, l + 1, r), multi[cur] * nums[r] + dfs(cur + 1, l, r - 1))

        res = dfs(0, 0, n - 1)
        dfs.cache_clear()
        return res
```



<!-- tabs:end -->
</details>

<br>

* * *

### 二维

> [!NOTE] **[AcWing 321. 棋盘分割](https://www.acwing.com/problem/content/description/323/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二维分割问题
> 
> 注意状态定义 记忆化搜索

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 15, M = 9;
const double INF = 1e9;

int n, m = 8;
int s[M][M];
double f[M][M][M][M][N];
double X;

int get_sum(int x1, int y1, int x2, int y2) {
    return s[x2][y2] - s[x2][y1 - 1] - s[x1 - 1][y2] + s[x1 - 1][y1 - 1];
}

double get(int x1, int y1, int x2, int y2) {
    double sum = get_sum(x1, y1, x2, y2) - X;
    return (double)sum * sum / n;
}

double dp(int x1, int y1, int x2, int y2, int k) {
    double &v = f[x1][y1][x2][y2][k];
    if (v >= 0) return v;
    if (k == 1) return v = get(x1, y1, x2, y2);

    v = INF;
    for (int i = x1; i < x2; i++) {
        v = min(v, get(x1, y1, i, y2) + dp(i + 1, y1, x2, y2, k - 1));
        v = min(v, get(i + 1, y1, x2, y2) + dp(x1, y1, i, y2, k - 1));
    }

    for (int i = y1; i < y2; i++) {
        v = min(v, get(x1, y1, x2, i) + dp(x1, i + 1, x2, y2, k - 1));
        v = min(v, get(x1, i + 1, x2, y2) + dp(x1, y1, x2, i, k - 1));
    }

    return v;
}

int main() {
    cin >> n;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= m; j++) {
            cin >> s[i][j];
            s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
        }

    X = (double)s[m][m] / n;
    memset(f, -1, sizeof f);
    printf("%.3lf\n", sqrt(dp(1, 1, 8, 8, n)));

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

> [!NOTE] **[LeetCode 1444. 切披萨的方案数](https://leetcode-cn.com/problems/number-of-ways-of-cutting-a-pizza/)**
> 
> 题意: 
> 
> rows * cols的矩阵 某些坐标有Apple 
> 
> 每次可以沿 【垂直/水平】 切一刀（【左/上】至少包含 1 个 apple）然后将【左/上】拿出去
> 
> 剩下的自问题 总共切 k-1 次分成 k 个都有苹果的块

> [!TIP] **思路**
> 
> 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int dp[55][55][15];  // 以 [i,j]为左上角 *再切* k刀的方案数
    int s[55][55];       // 以 [i,j]为左上角 包含苹果个数
    const int MOD = 1e9 + 7;
    int ways(vector<string>& pizza, int k) {
        int m = pizza.size(), n = pizza[0].size();
        for (int i = m; i > 0; --i)
            for (int j = n; j > 0; --j)
                s[i][j] = s[i + 1][j] + s[i][j + 1] - s[i + 1][j + 1] +
                          (pizza[i - 1][j - 1] == 'A');

        for (int i = m; i > 0; --i)
            for (int j = n; j > 0; --j) {
                if (s[i][j] == 0) continue;
                dp[i][j][0] = 1;
                for (int t = 1; t < k; ++t) {
                    for (int x = i + 1; x <= m; ++x)
                        if (s[i][j] - s[x][j])
                            dp[i][j][t] = (dp[i][j][t] + dp[x][j][t - 1]) % MOD;
                    for (int y = j + 1; y <= n; ++y)
                        if (s[i][j] - s[i][y])
                            dp[i][j][t] = (dp[i][j][t] + dp[i][y][t - 1]) % MOD;
                }
            }
        return dp[1][1][k - 1];
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

### 进阶

> [!NOTE] **[LeetCode 2019. 解出数学表达式的学生分数](https://leetcode-cn.com/problems/the-score-of-students-solving-math-expression/)**
> 
> [weekly-260](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-09-26_Weekly-260)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 区间DP进阶。中缀表达式计算 + 区间DP + 剪枝

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    stack<int> num;
    stack<char> op;
    void eval() {
        auto a = num.top(); num.pop();
        auto b = num.top(); num.pop();
        auto c = op.top(); op.pop();
        int r;
        if (c == '+')
            r = a + b;
        else
            r = a * b;
        num.push(r);
    }
    int calc(string s) {
        unordered_map<char, int> pr;
        pr['+'] = pr['-'] = 1, pr['*'] = pr['/'] = 2;
        for (int i = 0; i < s.size(); ++ i ) {
            char c = s[i];
            if (c == ' ')
                continue;
            if (isdigit(c)) {
                int x = 0, j = i;
                while (j < s.size() && isdigit(s[j]))
                    x = x * 10 + s[j] - '0', j ++ ;
                num.push(x);
                i = j - 1;
            } else {
                while (op.size() && pr[op.top()] >= pr[c])
                    eval();
                op.push(c);
            }
        }
        while (op.size())
            eval();
        return num.top();
    }

    int scoreOfStudents(string s, vector<int>& answers) {
        int tar = calc(s), n = s.size();

        unordered_set<int> f[32][32];
        for (int len = 1; len <= n; ++ len )
            for (int l = 0; l + len - 1 < n; l += 2 ) {
                int r = l + len - 1;
                if (l == r)
                    f[l][r].insert(s[l] - '0');
                else {
                    for (int k = l; k < r; k += 2)
                        for (auto v1 : f[l][k])
                            for (auto v2 : f[k + 2][r]) {
                                int t = 0;
                                if (s[k + 1] == '+')
                                    t = v1 + v2;
                                else
                                    t = v1 * v2;
                                if (t > 1000)
                                    continue;
                                f[l][r].insert(t);
                            }
                }
            }
        
        int res = 0;
        for (auto v : answers)
            if (v == tar)
                res += 5;
            else if (f[0][n - 1].count(v))
                res += 2;
        return res;
    }
};
```


##### **C++ 初版TLE**

```cpp
// TLE
class Solution {
public:
    using PON = pair<vector<char>, vector<int>>;
    using PONI = pair<PON, int>;
        
    int n, tar;
    unordered_set<int> S;
    set<PONI> hash;
    
    PON parse(string s) {
        vector<char> ops;
        vector<int> nums;
        int n = s.size();
        for (int i = 0; i < n; ++ i ) {
            int j = i, v = 0;
            while (j < n && isdigit(s[j]))
                v = v * 10 + s[j] - '0', j ++ ;
            nums.push_back(v);
            if (j < n)
                ops.push_back(s[j]);
            i = j;
        }
        return {ops, nums};
    }
    
    int op(char c, int a, int b) {
        if (c == '+')
            return a + b;
        return a * b;
    }
    
    void dfs(vector<char> ops, vector<int> nums, int cnt) {
        // MEM
        PONI t = {{ops, nums}, cnt};
        if (hash.count(t))
            return;
        hash.insert(t);
        
        if (ops.empty()) {
            // cout << "cnt = " << cnt << " nums[0] = " << nums[0] << endl;
            if (cnt == 0)
                this->tar = nums[0];
            else
                S.insert(nums[0]);
            return;
        }
        
        int n = ops.size(), p = 0;
        for (int i = 0; i < n; ++ i )
            if (ops[i] == '*') {
                p = i;
                break;
            }
        
        for (int i = 0; i < n; ++ i ) {
            int v = op(ops[i], nums[i], nums[i + 1]);
            // ATTENTION 增加一个剪枝
            if (v >= 1000)
                continue;
            
            vector<char> t_ops;
            vector<int> t_nums;
            for (int j = 0; j < i; ++ j )
                t_ops.push_back(ops[j]), t_nums.push_back(nums[j]);
            t_nums.push_back(v);
            for (int j = i + 1; j < n; ++ j )
                t_ops.push_back(ops[j]), t_nums.push_back(nums[j + 1]);
            dfs(t_ops, t_nums, cnt + (i != p));
        }
    }
    
    int scoreOfStudents(string s, vector<int>& answers) {
        this->n = s.size();
        auto [ops, nums] = parse(s);
        
        dfs(ops, nums, 0);
        
        // cout << "tar = " << tar << endl;
        
        int res = 0;
        for (auto v : answers)
            if (v == tar)
                res += 5;
            else if (S.count(v))
                res += 2;
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

> [!NOTE] **[LeetCode 1771. 由子序列构造的最长回文串的长度](https://leetcode-cn.com/problems/maximize-palindrome-length-from-subsequences/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 将两个字符串进行拼接 求拼接后字符串的最长回文子序列 但要保证答案对应原字符串中的子序列都非空
> 
> **注意添加条件的判断处理 ( l 和 r 的范围判断 )**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 2010;
    int f[N][N];
     
    int longestPalindrome(string word1, string word2) {
        string s = word1 + word2;
        int n1 = word1.size(), n2 = word2.size(), n = s.size();
        memset(f, 0, sizeof f);
        
        int res = 0;
        for (int i = 1; i <= n; ++ i ) f[i][i] = 1;
        for (int len = 2; len <= n; ++ len )
            for (int l = 1; l + len - 1 <= n; ++ l ) {
                int r = l + len - 1;
                if (s[l - 1] == s[r - 1]) {
                    f[l][r] = f[l + 1][r - 1] + 2;
                    // 只要加一个判断即可
                    if (l <= n1 && r > n1) res = max(res, f[l][r]);
                } else
                    f[l][r] = max(f[l + 1][r], f[l][r - 1]);
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

### 优化进阶

尤其是线性优化

> [!NOTE] **[LeetCode 1937. 扣分后的最大得分](https://leetcode-cn.com/problems/maximum-number-of-points-with-cost/)**
> 
> [weekly-250](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-07-18_Weekly-250)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本行状态依赖于上一行的状态，且本行内状态时前后有关联关系（区间修改/移动）
> 
> 尝试根据状态转移方程【拆掉绝对值表达式】
> 
> $O(nm^2)$ 优化为 $O(nm)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int INF = 0x3f3f3f3f;
    int n, m;
    vector<vector<int>> ps;
    vector<LL> f, g;
    
    long long maxPoints(vector<vector<int>>& points) {
        this->ps = points;
        this->n = ps.size(), this->m = ps[0].size();
        
        f = g = vector<LL>(m);
        
        for (int i = 0; i < m; ++ i )
            f[i] = ps[0][i];
        
        for (int i = 1; i < n; ++ i ) {
            g = f;
            {
                LL maxv = -INF;
                for (int j = 0; j < m; ++ j ) {
                    maxv = max(maxv, g[j] + j);
                    f[j] = max(f[j], ps[i][j] - j + maxv);
                }
            }
            {
                LL maxv = -INF;
                for (int j = m - 1; j >= 0; -- j ) {
                    maxv = max(maxv, g[j] - j);
                    f[j] = max(f[j], ps[i][j] + j + maxv);
                }
            }
        }
        return *max_element(f.begin(), f.end());
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

> [!NOTE] **[LeetCode 1977. 划分数字的方案数](https://leetcode-cn.com/problems/number-of-ways-to-separate-numbers/)**
> 
> [biweekly-59](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-08-21_Biweekly-59)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 当前 $f[i][j]$ 计算依赖于 $f[k][1 ~ j-1]$ 的求和 ，考虑前缀和优化
> 
> 另外需要字符串比较，使用 LCP 的思路【第一次见 LCP】



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    string s;
    vector<vector<int>> f, sum, lcp;
    
    // [r1结束长度为l的串] 是否 >= [r2结束长度为l的串]
    bool check(int r1, int r2, int l) {
        int l1 = r1 - l + 1, l2 = r2 - l + 1;
        if (l1 <= 0 || l2 <= 0)
            return false;
        int t = lcp[l1][l2];
        return t >= l || s[l1 + t - 1] > s[l2 + t - 1];
    }
    
    int numberOfCombinations(string num) {
        this->s = num;
        int n = s.size();
        f = sum = lcp = vector<vector<int>>(n + 1, vector<int>(n + 1));
        
        // lcp
        for (int i = n; i; -- i )
            for (int j = n; j; -- j )
                if (s[i - 1] != s[j - 1])
                    lcp[i][j] = 0;
                else {
                    lcp[i][j] = 1;
                    if (i < n && j < n)
                        lcp[i][j] += lcp[i + 1][j + 1];
                }
        
        // 初始化
        f[0][0] = 1;
        for (int i = 0; i <= n; ++ i )
            sum[0][i] = 1;  // sum[0][i] = sum[0][i - 1]
        
        // f[i][j] 前i个数 最后一个长度为j的方案数
        // sum[i][j] 以i结尾 长度不超过j的方案数总和
        for (int i = 1; i <= n; ++ i ) {
            for (int j = 1; j <= i; ++ j ) {
                int k = i - j;
                // 前缀和优化 将[枚举k结尾长度]的On降为O1
                if (s[k + 1 - 1] == '0')
                    f[i][j] = 0;    // 本段包含前缀0 非法
                else {
                    // case 1 长度小于j的都合法
                    f[i][j] = sum[k][j - 1];
                    // for (int t = 0; t < j; ++ t )
                    //     f[i][j] += f[k][t];
                    
                    // case 2 长度等于j的要比较大小
                    if (check(i, k, j))
                        f[i][j] = (f[i][j] + f[k][j]) % MOD;
                }
                // 更新
                sum[i][j] = (sum[i][j - 1] + f[i][j]) % MOD;
            }
            // 更新 根据定义，且j在内层循环所以必须这么写
            for (int j = i + 1; j <= n; ++ j )
                sum[i][j] = sum[i][j - 1];
        }
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            res = (res + f[n][i]) % MOD;    // add
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

> [!NOTE] **[LeetCode 546. 移除盒子](https://leetcode-cn.com/problems/remove-boxes/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> [acwing 题解](https://www.acwing.com/solution/content/6168/)
> 
> 区间DP进阶。基础版的问题为：最少需要多少次操作（操作的规则一样）能删除掉所有的盒子。
> 
> 状态定义
> 
> **使用 g 数组本质是区间 dp 的优化, 用 g 来辅助记录某个区间限制下的最大值**
> 
> **复杂区间 dp** 重复做
> 
> **经典优化 TODO 待集中整理**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准新理解**

```cpp
class Solution {
public:
    const static int N = 110;

    int f[N][N][N], g[N][N];

    int removeBoxes(vector<int>& boxes) {
        int n = boxes.size();
        memset(f, 0xcf, sizeof f);
        memset(g, 0xcf, sizeof g);

        for (int i = 0; i < n; ++ i )
            g[i][i] = f[i][i][1] = 1;

        for (int len = 2; len <= n; ++ len )
            for (int l = 0; l + len - 1 < n; ++ l ) {
                int r = l + len - 1;
                for (int k = 1; k <= len; ++ k ) {
                    if (k == 1)
                        f[l][r][k] = 1 + g[l + 1][r];
                    else
                        // u 可以取到最右侧时为 u + （k - 1) - 1 == r
                        // 也即 u == r - (k - 1) + 1
                        for (int u = l + 1; u <= r - (k - 1) + 1; ++ u ) {
                            if (boxes[u] != boxes[l])
                                continue;
                            int t = 0;
                            if (l + 1 <= u - 1)
                                t = g[l + 1][u - 1];
                            // ATTENTION: (k-1)^2 是相当于不删 u~r 的部分 而删 i 连上 u~r 的部分
                            f[l][r][k] = max(f[l][r][k], t + f[u][r][k - 1] - (k - 1) * (k - 1) + k * k);
                        }
                    g[l][r] = max(g[l][r], f[l][r][k]);
                }
            }
        return g[0][n - 1];
    }
};
```

##### **C++ 标准**

```cpp
class Solution {
public:
    const static int N = 110;

    // f[i][j][k] 所有将区间 [i, j] 清空，且最后删除i，且最后删除时的长度为k的最大取值
    // g[i][j] = max(f[i][j][0], f[i][j][1] ... f[i][j][k])
    int f[N][N][N], g[N][N];

    int removeBoxes(vector<int>& boxes) {
        int n = boxes.size();
        memset(f, 0xcf, sizeof f);
        memset(g, 0xcf, sizeof g);

        for (int len = 1; len <= n; ++ len )
            for (int l = 0; l + len - 1 < n; ++ l ) {
                int r = l + len - 1;
                for (int k = 1; k <= len; ++ k ) {
                    if (len == 1)
                        f[l][r][k] = 1;
                    else if (k == 1)
                        f[l][r][k] = 1 + g[l + 1][r];
                    else
                        // 枚举时因为 l 总是第一个被删除的数
                        // 因此不同点设置为【第二个会被删除的数】并以其作为 u 
                        // 此时 总取值为
                        //   `l`    +     `range_l`       +  `u as the beginning of range_r`
                        //   `l`    +  `g[l + 1][u - 1]`  +  `f[u][r][k - 1]`
                        // 又因为最终删除时 l 与 range_r 一体
                        //      其价值为 f[u][r][k-1] - (k-1)*(k-1) + k*k 【思考】
                        // 综上有以下代码实现
                        for (int u = l + 1; u <= r - k + 2; ++ u ) {
                            if (boxes[u] != boxes[l])
                                continue;
                            int t = 0;
                            if (l + 1 <= u - 1)
                                t = g[l + 1][u - 1];
                            f[l][r][k] = max(f[l][r][k], t + f[u][r][k - 1] - (k - 1) * (k - 1) + k * k);
                        }
                    g[l][r] = max(g[l][r], f[l][r][k]);
                }
            }
        return g[0][n - 1];
    }
};
```

##### **C++ 旧**

```cpp
class Solution {
public:
    // 与射气球类似 注意状态转移
/*
我们很容易陷入这样一个错误的思路：用 f(l, r) 来表示移除区间 [l, r] 内所有的盒子能得到的最大积分，
然后去探索某一种移除盒子的策略来进行状态转移。而实际上，我们并不能直接使用起始节点和结束节点决定最大分数，
因为这个分数并不只依赖于子序列，也依赖于之前的移动对当前数组的影响，这可能让最终的子序列不是一个连续的子串。
比如 {3,4,2,4,4}，如果先把 2 移除，3 个 4 会合并在一起，对答案的贡献是 3^2 = 9，
如果先移除左右两边的 4 再移除 2 这里 3 个 4 的贡献就是 1^2 + 2^2 = 5
最优的办法当然是先取走 2，但是这样剩下的 3 个 4 其实并不是原串的某个连续子串。
*/
    const int INF = 1e8;
    int removeBoxes(vector<int>& boxes) {
        int n = boxes.size();
        vector<vector<vector<int>>> f(n, vector<vector<int>>(n, vector<int>(n + 1, -INF)));
        vector<vector<int>> g(n, vector<int>(n, -INF));

        for (int len = 1; len <= n; ++ len )
            for (int l = 0; l + len - 1 < n; ++ l ) {
                int r = l + len - 1;
                for (int k = 1; k <= len; ++ k ) {
                    if (len == 1) f[l][r][k] = 1;
                    else if (k == 1) f[l][r][k] = 1 + g[l + 1][r];
                    else for (int u = l + 1; u <= r - k + 2; ++ u ) {
                        if (boxes[l] != boxes[u]) continue;
                        int t = 0;
                        if (l + 1 <= u - 1) t = g[l + 1][u - 1];
                        f[l][r][k] = max(f[l][r][k], t + f[u][r][k - 1] - (k - 1) * (k - 1) + k * k);
                    }
                    g[l][r] = max(g[l][r], f[l][r][k]);
                }
            }
        return g[0][n - 1];
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

> [!NOTE] **[LeetCode 1563. 石子游戏 V](https://leetcode-cn.com/problems/stone-game-v/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 题意理解有问题，其实这题是每次选中点切分成两组。
> 
> - 显然区间DP ==> TLE
> 
> - 区间DP基础上 + 记忆化 ==> AC
> 
> - 优化写法: [考虑不用记忆化搜索的优化](https://leetcode-cn.com/problems/stone-game-v/solution/on2dong-tai-gui-hua-jie-fa-by-huangyuyang/) ==> AC
> 
> **非常经典的优化（题目条件每个数都是正数 且本题特殊条件说明左右区间比大小 显然可以预处理）**
> 
> **TODO 集中整理**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ TLE**

```cpp
class Solution {
public:
    int stoneGameV(vector<int>& stoneValue) {
        int n = stoneValue.size();
        vector<int> sum(n + 1);
        for (int i = 1; i <= n; ++i) sum[i] = sum[i - 1] + stoneValue[i - 1];
        vector<vector<int>> f(n + 1, vector<int>(n + 1));
        for (int len = 2; len <= n; ++len)
            for (int l = 1; l + len - 1 <= n; ++l) {
                int r = l + len - 1;
                for (int k = l; k < r; k++) {
                    int lv = sum[k] - sum[l - 1];
                    int rv = sum[r] - sum[k];
                    if (lv > rv)
                        f[l][r] = max(f[l][r], rv + f[k + 1][r]);
                    else if (lv < rv)
                        f[l][r] = max(f[l][r], lv + f[l][k]);
                    else
                        f[l][r] = max(f[l][r], max(f[k + 1][r], f[l][k]) + lv);
                }
            }
        return f[1][n];
    }
};
```

##### **C++ AC**

```cpp
class Solution {
public:
    int vis[601][601], f[601][601];
    int idx = 101, a[10001], sum[10001];

    int dfs(int l, int r) {
        if (l == r) return 0;
        if (vis[l][r] == idx) return f[l][r];
        vis[l][r] = idx;
        int ans = 0;
        for (int i = l; i < r; i++) {
            int x = sum[i] - sum[l - 1], y = sum[r] - sum[i];
            int tmp = 0;
            if (x > y)
                tmp = dfs(i + 1, r) + y;
            else if (x < y)
                tmp = dfs(l, i) + x;
            else
                tmp = max(dfs(i + 1, r), dfs(l, i)) + x;
            ans = max(ans, tmp);
        }
        return f[l][r] = ans;
    }
    int stoneGameV(vector<int>& stoneValue) {
        int n = (int)stoneValue.size();
        ++idx;
        for (int i = 1; i <= n; i++)
            a[i] = stoneValue[i - 1], sum[i] = sum[i - 1] + a[i];
        return dfs(1, n);
    }
};
```

##### **C++ 优化 AC**

```cpp
const int N = 505;
int s[N][N], g[N][N], f[N][N], mxl[N][N], mxr[N][N];
class Solution {
public:
    int stoneGameV(vector<int>& a) {
        int n = a.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f[i][j] = g[i][j] = s[i][j] = 0;
                mxl[i][j] = mxr[i][j] = 0;
            }
        }
        for (int i = 0; i < n; i++) {
            s[i][i] = a[i];
            g[i][i] = i;
            for (int j = i + 1; j < n; j++) {
                s[i][j] = s[i][j - 1] + a[j];
                int now = g[i][j - 1];
                while (s[i][j] - s[i][now] > s[i][now]) {
                    now++;
                }
                g[i][j] = now;
            }
        }
        
        for (int len = 1; len <= n; len++) {
            for (int l = 0; l + len - 1 < n; l++) {
                int r = l + len - 1;
                int mid = g[l][r];
                int ls = s[l][mid];
                int rs = s[mid + 1][r];
                if (ls == rs) {
                    f[l][r] = max(f[l][r], mxl[l][mid]);
                    f[l][r] = max(f[l][r], mxr[mid + 1][r]);
                } else {
                    if (mid > l) {
                        int ls = s[l][mid - 1];
                        f[l][r] = max(f[l][r], mxl[l][mid - 1]);
                    }
                    if (mid < r) {
                        int rs = s[mid + 1][r];
                        f[l][r] = max(f[l][r], mxr[mid + 1][r]);
                    }
                }
                int v = f[l][r] + s[l][r];
                if (l == r) {
                    mxl[l][r] = mxr[l][r] = v;
                } else {
                    mxl[l][r] = max(v, mxl[l][r - 1]);
                    mxr[l][r] = max(v, mxr[l + 1][r]);
                }
            }
        }
        return f[0][n - 1];
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