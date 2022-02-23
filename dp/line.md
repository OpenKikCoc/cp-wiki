## 习题

### 一般线性（待细分）

> [!NOTE] **[Luogu 覆盖墙壁](https://www.luogu.com.cn/problem/P1990)**
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
#include <bits/stdc++.h>
using namespace std;

// https://www.acwing.com/solution/content/16126/

const int N = 1e6 + 10, MOD = 1e4;

int n;
int f[N][4];
// f[i][j] :
//      i-1 has been filled, the i-th state is j

int main() {
    cin >> n;
    
    f[0][3] = f[0][0] = 1;
    
    for (int i = 1; i <= n; ++ i ) {
        f[i][0] = f[i - 1][3];
        f[i][1] = (f[i - 1][0] + f[i - 1][2]) % MOD;
        f[i][2] = (f[i - 1][0] + f[i - 1][1]) % MOD;
        f[i][3] = (f[i - 1][3] + f[i - 1][0] + f[i - 1][1] + f[i - 1][2]) % MOD;
    }
    cout << f[n][0] << endl;
    
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

> [!NOTE] **[LeetCode 32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)**
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
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.size();
        vector<int> f(n + 1);
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            if (s[i - 1] == ')') {
                if (i - 1 >= 1 && s[i - 2] == '(')
                    f[i] = f[i - 2] + 2;
                else if (i - 2 - f[i - 1] >= 0 && s[i - 2 - f[i - 1]] == '(')
                    f[i] = f[i - 2 - f[i - 1]] + f[i - 1] + 2;
                res = max(res, f[i]);
            }
        return res;
    }
};
```

##### **Python**

```python
"""
1. 状态表示：f(i) 为以 i 为结尾的最长合法子串；初始时，f(0)=0
2. 状态转移时，我们仅考虑当前字符是 ) 的时候。如果上一个字符是 (，即 ...() 结尾的情况，则 f(i)=f(i−1)+2。
3. 如果上一个字符是 )，即 ...)) 的情况，则我们通过上一个字符的动规结果，判断是否能匹配末尾的 )。
		判断 s[i - f(i - 1) - 2] 是 (，即 ...((合法))，则可以转移 f(i)=f(i−1)+2+f(i−f(i−1)−2)。
最终答案为动规数组中的最大值。

"""

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        f = [0] * (n + 1)
        f[0] = 0
        res = 0
        for i in range(2, n + 1):
            if s[i-1] == ')':
                if s[i-2] == '(':
                    f[i] = f[i-2] + 2
                elif i - 2 - f[i-1] >= 0 and s[i-2-f[i-1]] == '(':
                    f[i] = f[i-1] + f[i-2-f[i-1]] + 2
            res = max(res, f[i])
        return res
      
      
"""
使用栈来模拟操作。栈顶保存当前扫描的时候，当前合法序列前的一个位置位置下标是多少。初始时栈中元素为-1。然后遍历整个字符串

如果s[i] =='('，那么把i进栈。
如果s[i] == ')',那么弹出栈顶元素 （代表栈顶的左括号匹配到了右括号）
如果此时栈为空，将i进栈。说明之前没有与之匹配的左括号，那么就将当前的位置入栈。
否则，i - st[-1]就是当前右括号对应的合法括号序列长度。
"""

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        res = 0
        stack = []
        stack.append(-1)
        for i in range(n):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)**
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
// case 61 / 63 会RE
// Line 17: Char 49: runtime error: signed integer overflow: 4472995859186094240 + 5516694892996182896 cannot be represented in type 'long' (solution.cpp)
// SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior prog_joined.cpp:26:49
// 即便是官方标答也是如此 略过

class Solution {
public:
    using LL = long long;
    int numDistinct(string s, string t) {
        int n = s.size(), m = t.size();
        if (n < m)
            return 0;
        vector<vector<LL>> f(n + 1, vector<LL>(m + 1));
        for (int i = 0; i <= n; ++ i )
            f[i][0] = 1;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j )
                if (s[i - 1] == t[j - 1])
                    // 可用可不用
                    f[i][j] = f[i - 1][j - 1] + f[i - 1][j];
                else
                    f[i][j] = f[i - 1][j];
        return f[n][m];
    }
};
```

##### **Python**

```python
# 两个字符串 + 一个序列的所有子序列是2**n（指数级别） ==> 考虑用dp
# 状态表示：s[1-i]的所有和t[1-j]相等的子序列；属性:count
# 状态转移：以s[i]是否包含在内作为划分：
# 1）不包含s[i]: s[i] != t[j]: f[i][j] = f[i-1][j] 
# 2）包含s[i]: f[i][j] = f[i-1][j-1]
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n+1):
            f[i][0] = 1  # 初始化很重要！！当t字符串为空时，是有意义的 为1
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                f[i][j] = f[i-1][j]
                if s[i-1] == t[j-1]:
                    f[i][j] += f[i-1][j-1]
        return f[n][m]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        int fmin = 1, fmax = 1, res = INT_MIN;
        for (int i = 1; i <= n; ++ i ) {
            int a = fmax, b = fmin;
            fmax = max(nums[i - 1], max(a * nums[i - 1], b * nums[i - 1]));
            fmin = min(nums[i - 1], min(a * nums[i - 1], b * nums[i - 1]));
            res = max(res, fmax);
        }
        return res;
    }
};

class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int res = nums[0];
        int f = nums[0], g = nums[0];
        for (int i = 1; i < nums.size(); i ++ ) {
            int a = nums[i], fa = f * a, ga = g * a;
            f = max(a, max(fa, ga));
            g = min(a, min(fa, ga));
            res = max(res, f);
        }
        return res;
    }
};
```

##### **Python**

```python
# 核心思想：记录当前的最大值 和 最小值
# 如果遇到负数，当前的最大值会变成最小值，最小值会变成最大值。
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        if not nums:return 0
        maxv, minv, res = nums[0], nums[0], nums[0]
        for i in range(1, n):
            mx, mn = maxv, minv
            maxv = max(nums[i], max(nums[i] * mx, nums[i] * mn))  # 如果mx和mn为0，那么此时最大值为nums[i] 
            minv = min(nums[i], min(nums[i] * mx, nums[i] * mn))
            res = max(maxv, res)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 413. 等差数列划分](https://leetcode-cn.com/problems/arithmetic-slices/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思路更清晰的解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 一般定义**

```cpp
class Solution {
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        int len = A.size(), res = 0;
        if (len < 3) return 0;
        int dp = 0;
        for (int i = 2; i < len; ++ i ) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                dp = 1 + dp;
                res += dp;
            } else {
                dp = 0;
            }
        }
        return res;
    }

    int numberOfArithmeticSlices(vector<int>& A) {
        int len = A.size();
        if (len < 3) return 0;
        int one = 0, two = 0, three = 0, res = 0;
        for (int i = 2; i < len; ++ i ) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                // 此处可以优化 因为只用到了two
                three = two >= 3 ? two + 1 : 3;
                res += three - 2;
                one = two, two = three;
            } else {
                one = two = three = 0;
            }
        }
        return res;
    }
};
```

##### **C++**

```cpp
// yxc 计算思路更清晰
class Solution {
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        for (int i = A.size() - 1; i > 0; i -- ) A[i] -= A[i - 1];
        int res = 0;
        for (int i = 1; i < A.size(); i ++ ) {
            int j = i;
            while (j < A.size() && A[j] == A[i]) j ++ ;
            int k = j - i;
            res += k * (k - 1) / 2;
            i = j - 1;
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

> [!NOTE] **[LeetCode 446. 等差数列划分 II - 子序列](https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 关于弱等差数列的转换，以及转换后求真等差数列。
> 
> 求真等差数列有两种方法：
> 
> 其一，我们可以对真弱等差数列的数量进行直接计数。真弱等差数列即为长度为 2 的弱等差数列，故其数量为 $(i, j)$ 对的格数，即为 $n * (n - 1) / 2$
> 
> 其二，对于 $f[i][A[i] - A[j]] += (f[j][A[i] - A[j]] + 1)，f[j][A[i] - A[j]]$ 是现有的弱等差数列个数，而 1 是根据 $A[i]$ 和 $A[j]$ 新建的子序列。根据性质二【若在弱等差数列后添加一个元素且保持等差，则一定得到一个等差数列】，新增加的序列必为等差数列。故 $f[j][A[i] - A[j]]$ 为新生成的等差数列的个数。
> 
> ==> 也因此 真正的等差数列可以由 $res += f[j][k]$ 统计而来
> 
> **经典 定义转化与计数方法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    typedef long long LL;
    int numberOfArithmeticSlices(vector<int>& A) {
        if (A.empty()) return 0;
        int n = A.size();
        int res = 0;
        vector<unordered_map<LL, int>> f(n + 1);
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < i; ++ j ) {
                LL k = (LL)A[i] - A[j];
                // 等同:
                // res += f[j][k];
                // f[i][k] = f[j][k] + 1
                int t = 0;
                if (f[j].count(k)) {
                    t = f[j][k];
                    res += t;
                }
                f[i][k] += t + 1;
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

> [!NOTE] **[LeetCode 639. 解码方法 2](https://leetcode-cn.com/problems/decode-ways-ii/)**
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
class Solution {
public:
    // 分情况讨论即可
    const int mod = 1e9 + 7;
    int numDecodings(string s) {
        int n = s.size();
        vector<int> f(n + 1);
        f[0] = 1;
        for (int i = 1; i <= n; ++ i )
            // i位 匹配j
            for (int j = 1; j <= 26; ++ j ) {
                char a = s[i - 1];
                if (j <= 9) {
                    if (a == '*' || a == j + '0') f[i] += f[i - 1];
                } else if (i >= 2) {
                    char b = s[i - 2];
                    int y = j / 10, x = j % 10;
                    // b 和 y 匹配     a 和 x 匹配
                    if ((b == y + '0' || b == '*' && y) && (a == x + '0' || a == '*' && x))
                        f[i] += f[i - 2];
                }
                f[i] %= mod;
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

> [!NOTE] **[LeetCode 688. “马”在棋盘上的概率](https://leetcode-cn.com/problems/knight-probability-in-chessboard/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 递推

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
double f[25][25][101];

class Solution {
public:
    double knightProbability(int n, int K, int r, int c) {
        memset(f, 0, sizeof f);
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                f[i][j][K] = 1;

        int dx[] = {-2, -1, 1, 2, 2, 1, -1, -2};
        int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};
        for (int k = K - 1; k >= 0; k -- )
            for (int i = 0; i < n; i ++ )
                for (int j = 0; j < n; j ++ )
                    for (int u = 0; u < 8; u ++ ) {
                        int x = i + dx[u], y = j + dy[u];
                        if (x >= 0 && x < n && y >= 0 && y < n)
                            f[i][j][k] += f[x][y][k + 1] / 8;
                    }
        return f[r][c][0];
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

> [!NOTE] **[LeetCode 790. 多米诺和托米诺平铺](https://leetcode-cn.com/problems/domino-and-tromino-tiling/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典线性 dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 递推
class Solution {
public:
    const int MOD = 1e9 + 7;

    int numTilings(int n) {
        vector<vector<int>> f(n, vector<int>(4, 0));
        f[0][0] = 1;
        f[0][3] = 1;

        for (int i = 1; i < n; ++ i ) {
            f[i][0] = f[i - 1][3];
            f[i][1] = (f[i - 1][0] + f[i - 1][2]) % MOD;
            f[i][2] = (f[i - 1][0] + f[i - 1][1]) % MOD;
            f[i][3] = ((( f[i - 1][0]
                        + f[i - 1][1]) % MOD
                        + f[i - 1][2]) % MOD
                        + f[i - 1][3]) % MOD;
        }
        return f[n - 1][3];
    }
};

// 快速幂
class Solution {
public:
    int numTilings(int n) {
        const int MOD = 1e9 + 7;
        int w[4][4] = {
            {1, 1, 1, 1},
            {0, 0, 1, 1},
            {0, 1, 0, 1},
            {1, 0, 0, 0}
        };
        vector<vector<int>> f(n + 1, vector<int>(4));
        f[0][0] = 1;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < 4; j ++ )
                for (int k = 0; k < 4; k ++ )
                    f[i + 1][k] = (f[i + 1][k] + f[i][j] * w[j][k]) % MOD;
        return f[n][0];
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

> [!NOTE] **[LeetCode 1621. 大小为 K 的不重叠线段的数目](https://leetcode-cn.com/problems/number-of-sets-of-k-non-overlapping-line-segments/)**
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
class Solution {
public:
    // f[i][k] 表示长度为i j段段方案数
    const int mod = 1e9 + 7;
    int numberOfSets(int n, int k) {
        vector<vector<pair<int, int>>> f(n + 1, vector<pair<int, int>>(k + 1));
        // first not     second has
        f[0][0].first = 1;
        for (int i = 1; i < n; ++i)
            for (int j = 0; j <= k; ++j) {
                f[i][j].first = (f[i - 1][j].first + f[i - 1][j].second) % mod;
                f[i][j].second = f[i - 1][j].second;
                if (j > 0) {
                    f[i][j].second =
                        (f[i][j].second + f[i - 1][j - 1].first) % mod;
                    f[i][j].second =
                        (f[i][j].second + f[i - 1][j - 1].second) % mod;
                }
            }
        return (f[n - 1][k].first + f[n - 1][k].second) % mod;
    }
};
```

##### **C++ 转化 组合数**

[题解](https://leetcode-cn.com/problems/number-of-sets-of-k-non-overlapping-line-segments/solution/da-xiao-wei-k-de-bu-zhong-die-xian-duan-de-shu-mu-/)

```cpp
class Solution {
public:
    const static int N = 2010, MOD = 1e9 + 7;

    int C[N][N];

    void init() {
        for (int i = 0; i < N; ++ i )
            for (int j = 0; j <= i; ++ j )
                if (!j)
                    C[i][j] = 1;
                else
                    C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) % MOD;
    }

    // C_{n+k-1}^{2*k}
    int numberOfSets(int n, int k) {
        init();
        return C[n + k - 1][2 * k];
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

### 复杂线性

> [!NOTE] **[LeetCode 689. 三个无重叠子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-of-3-non-overlapping-subarrays/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性 逆序dp找方案
> 
> 经典 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    vector<int> maxSumOfThreeSubarrays(vector<int> nums, int k) {
        int n = nums.size();
        vector<int> s(n + 1);
        for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + nums[i - 1];
        vector<vector<int>> f(n + 2, vector<int>(4));

        int x = n + 1, y = 3;
        for (int i = n - k + 1; i; i -- ) {
            for (int j = 1; j <= 3; j ++ )
                f[i][j] = max(f[i + 1][j], f[i + k][j - 1] + s[i + k - 1] - s[i - 1]);
            if (f[x][3] <= f[i][3]) x = i;
        }

        vector<int> res;
        while (y) {
            while (f[x][y] != f[x + k][y - 1] + s[x + k - 1] - s[x - 1]) x ++ ;
            res.push_back(x - 1);
            x += k, y -- ;
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> sum(n + 1, 0);

        reverse(nums.begin(), nums.end());

        for (int i = 1; i <= n; i++)
            sum[i] = sum[i - 1] + nums[i - 1];

        vector<vector<int>> f(n + 1, vector<int>(4, INT_MIN));

        for (int i = 0; i <= n; i++)
            f[i][0] = 0;

        for (int i = k; i <= n; i++)
            for (int j = 1; j <= 3; j++)
                f[i][j] = max(f[i - 1][j], f[i - k][j - 1] + sum[i] - sum[i - k]);


        int i = n, j = 3;
        vector<int> ans;
        while (j > 0) {
            if (f[i - 1][j] > f[i - k][j - 1] + sum[i] - sum[i - k]) {
                i--;
            } else {
                ans.push_back(n - i);
                i -= k;
                j--;
            }
        }

        return ans;
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

> [!NOTE] **[LeetCode 740. 删除与获得点数](https://leetcode-cn.com/problems/delete-and-earn/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始想的状态转移可以简化
> 
> **重点在于想到题意影响两侧本质就是上一个影响本个**
> 
> **同样与数据范围有关**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const int N = 10000;

    int deleteAndEarn(vector<int>& nums) {
        vector<int> cnt(N + 1);
        for (auto v : nums)
            cnt[v] ++ ;

        vector<vector<int>> f(N + 1, vector<int>(3));
        // 0 上一个被删 1 本身被删 2 不被删
        int res = 0;
        for (int i = 1, j = 0; i <= N; ++ i ) {
            auto x = i, y = cnt[x];
            f[i][0] = f[i - 1][1];
            f[i][1] = max(f[i - 1][0], f[i - 1][2]) + x * y;
            f[i][2] = max(f[i - 1][0], f[i - 1][2]);
            res = max(res, f[i][1]);
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
const int N = 10010;
int cnt[N], f[N][2];

class Solution {
public:
    int deleteAndEarn(vector<int>& nums) {
        memset(cnt, 0, sizeof cnt);
        memset(f, 0, sizeof f);
        for (auto x: nums) cnt[x] ++ ;
        int res = 0;
        for (int i = 1; i < N; i ++ ) {
            f[i][0] = max(f[i - 1][0], f[i - 1][1]);
            f[i][1] = f[i - 1][0] + i * cnt[i];
            res = max(res, max(f[i][0], f[i][1]));
        }
        return res;
    }
};
```

##### **Python**

```python
# 选了x 不能选择x + 1 和 x - 1；这和打家劫舍的题很像
# 状态机dp问题：有限制的选择问题（一般 有限制的选择问题 都可以用dp来做：背包问题也是有限制的选择问题）
# 用一个数组存储每个数字出现的次数；
# 考虑前i个数， 每个数都后面一个数都有影响；选择了i，就不能选i + 1；
# 对于每个数都有两种情况：选 :  f[i][1]: / 不选 : f[i][0]； 
# f[i][0] : 不选i，那i - 1  可选 可不选；1) 选i - 1 2）不选 i -1 ：f[i][0] = max(f[i-1][0], f[i-1][1])
# f[i][1]: 选择了i，那i-1一定不能选；f[i][1] = f[i-1][0] + i * cnt[i](i选一个 还是选两个 对其他数的影响都是一样的)
# 时间复杂度是O(N)， 一共有N个状态

class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        n = len(nums)
        cnt = [0] * 10010
        m = 0
        for x in nums:
            cnt[x] += 1 
            m = max(m, x)
        print(m)
        f = [[0] * 2 for i in range(m + 1)]
        for i in range(1, m + 1):
            f[i][0] = max(f[i - 1][0], f[i - 1][1])
            f[i][1] = f[i - 1][0] + cnt[i] * i 
        return max(f[m][0], f[m][1])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1187. 使数组严格递增](https://leetcode-cn.com/problems/make-array-strictly-increasing/)** [TAG]
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
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2) {
        // 数值离散化 并使用离散化结果更新原数组
        vector<int> p;
        for (auto x : arr1) p.push_back(x);
        for (auto x : arr2) p.push_back(x);
        sort(p.begin(), p.end());
        p.erase(unique(p.begin(), p.end()), p.end());
        for (auto & x : arr1) x = lower_bound(p.begin(), p.end(), x) - p.begin() + 1;
        for (auto & x : arr2) x = lower_bound(p.begin(), p.end(), x) - p.begin() + 1;
        
        int k = p.size();
        vector<int> u(k + 1);
        for (auto x : arr2) u[x] = 1;
        
        // f[i][j] 前i个数 末尾值最大为j 的替换次数
        int n = arr1.size();
        vector<vector<int>> f(n + 1, vector<int>(k + 1, inf));
        for (int j = 0; j <= k; ++ j ) f[0][j] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= k; ++ j ) {
                f[i][j] = f[i][j - 1];
                if (arr1[i - 1] == j)
                    f[i][j] = min(f[i][j], f[i - 1][j - 1]);
                if (u[j])
                    f[i][j] = min(f[i][j], f[i - 1][j - 1] + 1);
            }
        int ret = inf;
        for (int j = 1; j <= k; ++ j )
            ret = min(ret, f[n][j]);
        return ret == inf ? -1 : ret;
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

> [!NOTE] **[LeetCode 1223. 掷骰子模拟](https://leetcode-cn.com/problems/dice-roll-simulation/)**
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
const int mod = 1e9 + 7;
int dieSimulator(int n, vector<int>& rollMax) {
    // f[i][j][k] 到i位置 筛子j出现k次的方案数
    // for f[i][j][k] = f[i-1][j][k-1]
    vector<vector<vector<int>>> f(n + 1,
                                  vector<vector<int>>(6, vector<int>(16)));
    for (int i = 0; i < 6; ++i) f[1][i][1] = 1;

    for (int i = 2; i <= n; ++i)
        for (int j = 0; j < 6; ++j)
            for (int k = 0; k < 6; ++k) {  // 上个筛子为k
                if (j == k)
                    for (int t = 1; t < rollMax[j]; ++t)
                        f[i][j][t + 1] =
                            (f[i][j][t + 1] + f[i - 1][k][t]) % mod;
                else
                    for (int t = 1; t <= rollMax[k]; ++t)
                        f[i][j][1] = (f[i][j][1] + f[i - 1][k][t]) % mod;
            }
    int res = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= rollMax[i]; ++j) res = (res + f[n][i][j]) % mod;
    return res;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1388. 3n 块披萨](https://leetcode-cn.com/problems/pizza-with-3n-slices/)** [TAG]
> 
> 题意: 
> 
> 给一个长度为 3n 的环状序列，你可以在其中选择 n 个数，并且任意两个数不能相邻，求这 n 个数的最大值。

> [!TIP] **思路**
> 
> 环状序列相较于普通序列，相当于添加了一个限制：普通序列中的第一个和最后一个数不能同时选。
> 
> 这样一来，我们只需要对普通序列进行两遍动态即可得到答案，
> 
> - 第一遍动态规划中我们删去普通序列中的第一个数，表示我们不会选第一个数；
> - 第二遍动态规划中我们删去普通序列中的最后一个数，表示我们不会选最后一个数。
> 
> 将这两遍动态规划得到的结果去较大值，即为在环状序列上的答案。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int calculate(const vector<int>& slices) {
        int n = slices.size();
        int choose = (n + 1) / 3;
        vector<vector<int>> dp(n + 1, vector<int>(choose + 1));
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= choose; ++j)
                dp[i][j] = max(dp[i - 1][j], (i - 2 >= 0 ? dp[i - 2][j - 1] : 0) + slices[i - 1]);
        return dp[n][choose];
    }

    int maxSizeSlices(vector<int>& slices) {
        vector<int> v1(slices.begin() + 1, slices.end());
        vector<int> v2(slices.begin(), slices.end() - 1);
        int ans1 = calculate(v1);
        int ans2 = calculate(v2);
        return max(ans1, ans2);
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

> [!NOTE] **[LeetCode 1478. 安排邮筒](https://leetcode-cn.com/problems/allocate-mailboxes/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> [何逊的题解](https://leetcode-cn.com/problems/allocate-mailboxes/solution/dong-tai-gui-hua-shi-jian-fu-za-du-oknlognkong-jia/)
>
> `f[i][j]` 表示前 i 个建筑用 j 个邮箱的最短距离和 预处理 `dis[i][j]` 为从 i 到 j 使用一个邮箱时的消耗
>
> 则 `f[i][j] = min(f[d][j-1] + dis[d+1][i]) [0 < k < i-1] `
>
> 意即： 以 d 为最后一段的分界线 最后一段为 d+1~i 的最小消耗

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    long long f[105][105];
    long long dis[105][105];
    int minDistance(vector<int>& houses, int k) {
        int n = houses.size();
        sort(houses.begin(), houses.end());
        for (int i = 1; i <= n; ++i) {
            for (int j = i; j <= n; ++j) {
                int p = (j - i) >> 1;
                int mid = i + p;
                int pos = houses[mid - 1];
                long long res = 0;
                for (int t = i; t <= j; ++t) res += abs(houses[t - 1] - pos);
                dis[i][j] = res;
            }
        }
        memset(f, 0x3f, sizeof(f));
        f[0][0] = 0;
        for (int i = 1; i <= n; ++i)
            for (int t = 1; t <= i && t <= k; ++t)
                for (int j = i - 1; j >= 0; --j)
                    f[i][t] = min(f[i][t], f[j][t - 1] + dis[j + 1][i]);
        return f[n][k];
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

> [!NOTE] **[LeetCode 1575. 统计所有可行路径](https://leetcode-cn.com/problems/count-all-possible-routes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> $f[k][i-dis] += dp[j][i]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int mod = 1e9 + 7;
    void add(int& x, int y) {
        x += y;
        if (x >= mod) x -= mod;
    }
    int countRoutes(vector<int>& locations, int start, int finish, int fuel) {
        int n = locations.size();
        vector<vector<int>> f(n + 1, vector<int>(fuel + 1));
        f[start][fuel] = 1;
        for (int i = fuel; i >= 0; --i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) {
                    int dis = abs(locations[j] - locations[k]);
                    if (j != k && dis <= i) add(f[k][i - dis], f[j][i]);
                }
        int res = 0;
        for (int i = 0; i <= fuel; ++i) add(res, f[finish][i]);
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

> [!NOTE] **[LeetCode 1751. 最多可以参加的会议数目 II](https://leetcode-cn.com/problems/maximum-number-of-events-that-can-be-attended-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 如果权值1 显然可以堆
> 
> 线性dp 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    int maxValue(vector<vector<int>>& events, int k) {
        int n = events.size();
        sort(events.begin(), events.end(),
             [](const vector<int> & a, const vector<int> & b) {
                return a[1] < b[1];
        });
        events.insert(events.begin(), vector<int>{0, 0, 0});
        
        vector<vector<LL>> f(n + 1, vector<LL>(k + 1));
        f[0][0] = 0;
        
        for (int i = 1; i <= n; ++ i ) {
            int l = 0, r = i - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (events[mid + 1][1] >= events[i][0]) r = mid;
                else l = mid + 1;
            }
            
            f[i][0] = 0;
            for (int j = 1; j <= k; ++ j )
                f[i][j] = max(f[i - 1][j], f[l][j - 1] + events[i][2]);
        }
        return f[n][k];
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

> [!NOTE] **[LeetCode 1787. 使所有区间的异或结果为零](https://leetcode-cn.com/problems/make-the-xor-of-all-segments-equal-to-zero/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推理知: 以 k 为周期，后面每一个周期都与前面相同
> 
> 注意各类细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 直觉: 更改最靠右的元素 ==> WA
    // 考虑：分组
    const static int M = 1024;
    const int INF = 1e8;
    int s[M];
    int minChanges(vector<int>& nums, int k) {
        int n = nums.size(), m = (n + k - 1) / k;
        // 第 k 组为止异或状态值
        vector<vector<int>> f(k + 1, vector<int>(M, INF));
        f[0][0] = 0;

        // 所有列都先使用众数，随后修改众数个数最少的一列即可
        int sum = 0, minv = INF;
        for (int i = 1; i <= k; ++ i ) {
            memset(s, 0, sizeof s);
            int len = m;
            if (n % k && n % k < i) -- len; // 最后一行的某些列 不足m
            
            // 计数
            for (int j = 0; j < len; ++ j )
                s[nums[j * k + i - 1]] ++ ;
            // 众数 某一列用了一个全新的数
            int maxv = 0;
            for (int j = 0; j < M; ++ j )
                if (s[j])
                    maxv = max(maxv, s[j]);
            sum += len - maxv, minv = min(minv, maxv);
            
            // 某一列都用了原来的某个数
            for (int j = 0; j < M; ++ j )
                for (int u = 0; u < len; ++ u ) {
                    int x = nums[u * k + i - 1], cost = len - s[x];
                    f[i][j] = min(f[i][j], f[i - 1][j ^ x] + cost);
                }
        }
        return min(sum + minv, f[k][0]);
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

> [!NOTE] **[LeetCode 1959. K 次调整数组大小浪费的最小总空间](https://leetcode-cn.com/problems/minimum-total-space-wasted-with-k-resizing-operations/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然需要预处理【区间最值】和【区间和】
> 
> 随后 200 的数据范围写线性 dp 即可
> 
> 自己写了 RMQ 实际上可以直接区间 dp 来求区间最值
> 
> **唯一需要注意的点是初始化思路**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ RMQ**

```cpp
class Solution {
public:
    const static int N = 210, M = 9;
    
    vector<int> w;
    int n;
    int s[N];
    int f[N][M], g[N][N];
    
    void init() {
        for (int j = 0; j < M; ++ j )
            for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                if (!j)
                    f[i][j] = w[i - 1];
                else
                    f[i][j] = max(f[i][j - 1], f[i + (1 << j - 1)][j - 1]);
                
        memset(s, 0, sizeof s);
        for (int i = 1; i <= n; ++ i )
            s[i] = s[i - 1] + w[i - 1];
    }
    
    int query(int l, int r) {
        int len = r - l + 1;
        int k = log(len) / log(2);
        return max(f[l][k], f[r - (1 << k) + 1][k]);
    }
    
    int get(int l, int r) {
        int tot = s[r] - s[l - 1];
        int mxv = query(l, r);
        return mxv * (r - l + 1) - tot;
    }
    
    int minSpaceWastedKResizing(vector<int>& nums, int k) {
        this->w = nums;
        this->n = w.size();
        init();
        
        // 把整个数组分成 k + 1 个区间
        // 可以直接写到 k 是因为 g[i][0] 本已代表一个区间
        memset(g, 0x3f, sizeof g);
        for (int j = 0; j <= k; ++ j )
            g[0][j] = 0;
        for (int i = 1; i <= n; ++ i ) {
            g[i][0] = get(1, i);
            for (int j = 1; j <= i; ++ j )
                for (int t = 0; t < i; ++ t )
                    g[i][j] = min(g[i][j], g[t][j - 1] + get(t + 1, i));
        }
        return g[n][k];
    }
};
```

##### **C++ 无RMQ**

```cpp
// yxc
class Solution {
public:
    int minSpaceWastedKResizing(vector<int>& nums, int k) {
        k ++ ;
        int n = nums.size(), INF = 1e9;
        vector<vector<int>> f(n + 1, vector<int>(k + 1, INF));
        vector<int> s(n + 1);
        for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + nums[i - 1];
        f[0][0] = 0;
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= i && j <= k; j ++ )
                for (int u = i, h = 0; u; u -- ) {
                    h = max(h, nums[u - 1]);
                    f[i][j] = min(f[i][j], f[u - 1][j - 1] + h * (i - u + 1) - (s[i] - s[u - 1]));
                }
        return f[n][k];
    }
};
```

##### **C++ 官方 预处理O(n^2)**

```cpp
class Solution {
public:
    int minSpaceWastedKResizing(vector<int>& nums, int k) {
        int n = nums.size();

        // 预处理数组 g
        // 思路
        vector<vector<int>> g(n, vector<int>(n));
        for (int i = 0; i < n; ++i) {
            // 记录子数组的最大值
            int best = INT_MIN;
            // 记录子数组的和
            int total = 0;
            for (int j = i; j < n; ++j) {
                best = max(best, nums[j]);
                total += nums[j];
                g[i][j] = best * (j - i + 1) - total;
            }
        }
        
        vector<vector<int>> f(n, vector<int>(k + 2, INT_MAX / 2));
        for (int i = 0; i < n; ++i)
            for (int j = 1; j <= k + 1; ++j)
                for (int i0 = 0; i0 <= i; ++i0)
                    f[i][j] = min(f[i][j], (i0 == 0 ? 0 : f[i0 - 1][j - 1]) + g[i0][i]);

        return f[n - 1][k + 1];
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

> [!NOTE] **[LeetCode 1997. 访问完所有房间的第一天](https://leetcode-cn.com/problems/first-day-where-you-have-been-in-all-the-rooms/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分析题意尤为重要

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // ATTENTION:
    // 0 <= nextVisit[i] <= i
    // 综合题意条件
    // 说明第 i 个房间必定是两次访问第 i−1 个房间后到达的
    const static int MOD = 1e9 + 7;
    
    vector<int> f;
    
    int firstDayBeenInAllRooms(vector<int>& nextVisit) {
        int n = nextVisit.size();
        
        // 定义状态 f[i] 表示首次访问到房间 i 的日期 [房间编号0 - n-1]
        f = vector<int>(n);
        // 第一次到 [第0个房间] 需要0天
        f[0] = 0;
        for (int i = 1; i < n; ++ i ) {
            // 如果是第一次访问房间 i , 则i-1回访时回访到的地址 t 必然已经被经过了偶数次
            
            // i-1会回访房间t
            int t = nextVisit[i - 1];
            
            // 第一次到达第i房间 = 
            //    第一次到i-1 +      第二次到i-1       + 再到i
            f[i] = f[i - 1] + (f[i - 1] - f[t] + 1) + 1;
            f[i] = (f[i] % MOD + MOD) % MOD;
        }
        
        return f[n - 1];
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

> [!NOTE] **[LeetCode 2008. 出租车的最大盈利](https://leetcode-cn.com/problems/maximum-earnings-from-taxi/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 既可以以开始位置作为状态定义 也可以以结束为止
> 
> 结束位置定义可解释性更强
> 
> **容易得出定义 但要注意转移时直接【遍历位置而不是 rides 】 以及【最好预处理 rides 】**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;
    
    // 以 i 结尾
    LL f[N];
    
    long long maxTaxiEarnings(int n, vector<vector<int>>& rides) {
        memset(f, 0, sizeof f);
        
        sort(rides.begin(), rides.end(), [](const vector<int> & a, const vector<int> & b) {
            return a[1] < b[1];
        });
        
        // WRONG: cause this will not differ ride-and-ride at same end-time
        // for (auto & r : rides) ...
        
        // RIGHT:
        // i means end-time
        // ==================== case 1 ==================== PASS
        // for (int i = 1, j = 0; i <= n; ++ i ) {
        //      f[i] = f[i - 1];
        //    
        //      while (j < rides.size() && rides[j][1] == i) {
        //          int l = rides[j][0], r = rides[j][1], v = rides[j][2];
        //          f[i] = max(f[i], f[l] + r - l + v);
        //          j ++ ;
        //      }
        // }
        // ==================== case 2 ==================== PASS
        using PII = pair<int, int>;
        #define x first
        #define y second
        vector<vector<PII>> ve(n + 1);
        for (auto & r : rides)
            ve[r[1]].push_back({r[0], r[2]});
        for (int i = 1; i <= n; ++ i ) {
            f[i] = f[i - 1];
            for (auto & [s, v] : ve[i])
                f[i] = max(f[i], f[s] + i - s + v);
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

### 股票问题

> [!NOTE] **[LeetCode 121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)**
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
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int has = INT_MIN, no = 0;
        for (int i = 0; i < n; ++ i ) {
            int a = has, b = no;
            has = max(a, -prices[i]);
            no = max(b, a + prices[i]);
        }
        return no;
    }
};

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0;
        for (int i = 0, minp = INT_MAX; i < prices.size(); i ++ ) {
            res = max(res, prices[i] - minp);
            minp = min(minp, prices[i]);
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        min_v = float("+inf")
        res = 0  # 如果发现交易就会赔本，可以不卖卖，就是获利是0 （所以初始化为0）
        for i in range(n):
            res = max(res, prices[i] - min_v)
          	min_v = min(min_v, prices[i]) 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)**
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
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int has = INT_MIN, no = 0;
        for (auto & c : prices) {
            int nhas = max(has, no - c);
            int nno = max(has + c, no);
            has = nhas, no = nno;
        }
        return max(has, no);
    }
};

// trick
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0;
        for (int i = 0; i + 1 < prices.size(); i ++ )
            res += max(0, prices[i + 1] - prices[i]);
        return res;
    }
};
```

##### **Python**

```python
# 所有的交易 都没有交集。
# 遍历一次数组，低进高出，把正的价格差相加起来就是最终利润。
# 比如：[1,2,3]：在1买入，3卖出  等价于  每天都进行买入卖出。
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        res = 0  # 不做交易，不赔不赚。
        for i in range(1, n):
            res += max(0, prices[i] - prices[i-1])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)**
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
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int fstSell = 0, secSell = 0;
        int fstBuy = INT_MIN, secBuy = INT_MIN;
        for (int i = 1; i <= n; ++ i ) {
            fstSell = max(fstSell, fstBuy + prices[i - 1]);
            fstBuy = max(fstBuy, -prices[i-1]);
            secSell = max(secSell, secBuy + prices[i - 1]);
            secBuy = max(secBuy, fstSell - prices[i - 1]);
        }
        return secSell;
    }
};

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<int> f(n + 2);
        for (int i = 1, minp = INT_MAX; i <= n; i ++ ) {
            f[i] = max(f[i - 1], prices[i - 1] - minp);
            minp = min(minp, prices[i - 1]);
        }

        int res = 0;
        for (int i = n, maxp = 0; i; i -- ) {
            res = max(res, maxp - prices[i - 1] + f[i - 1]);
            maxp = max(maxp, prices[i - 1]);
        }

        return res;
    }
};
```

##### **Python**

```python
# 可以用dp来写；这里写另外一种思路，叫：！！！前后缀分解 （枚举两次交易的分界点）==> 涉及到分两次买入的情况都可以用这种思路 ： 可以枚举第二次交易的买入时间，比如是第i次交易买入的。（如何求这一类方案的最值呢？）
# f[i]: 在第1-i天 进行买卖一次的最大收益值（可以分为：1. 在第i天卖出；2.不在第i天卖出）
# 总收益就转换为：第一次交易是在前i-1天内完成的,可以表示成f[i-1]；第二次交易是在第i天买入，后面再卖出，最大收益是：第i天之后股票的最大值减去第i天的价格。
# 枚举完分界点后，就可以把一个问题分解成 两个独立的子问题；（搜Acwing--前后缀分解，会出来相关题型）

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        f = [0] * (n + 1)

        minv = float('inf')
        for i in range(1, n + 1):   #f从1开始，f[i]对应的是prices[i-1]
            f[i] = max(f[i-1], prices[i-1] - minv)
            minv = min(minv, prices[i-1])
        
        maxv = float('-inf')
        res = 0
        for i in range(n, 0, -1):
            res = max(res, maxv - prices[i-1] + f[i-1]) 
            maxv = max(maxv, prices[i-1])
        return res      
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)**
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
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        if (k > n / 2) {
            int res = 0;
            for (int i = 0; i < n - 1; ++ i )
                res += max(0, prices[i + 1] - prices[i]);
            return res;
        }
        vector<int> hs(k + 1, INT_MIN), no(k + 1);
        for (int i = 1; i <= n; ++ i )
            for (int c = 1; c <= k; ++ c ) {
                // hs[i][c] = max(hs[i-1][c], no[i-1][c-1] - prices[i-1]);
                // no[i][c] = max(no[i-1][c], hs[i-1][c] + prices[i-1]);
                no[c] = max(no[c], hs[c] + prices[i - 1]);
                hs[c] = max(hs[c], no[c - 1] - prices[i - 1]);
            }
        return no[k];
    }
};
```

##### **C++ yxc**

```cpp
// yxc
/*
LeetCode增强了本题的数据，非常卡常。

于是为了应对新数据，对视频中的代码做了如下优化：

1. 将vector换成了数组，大概会快50%。
2. 类似于背包问题优化空间，将原本的滚动二维数组，直接换成一维数组。
*/
int f[10001], g[10001];

class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int INF = 1e8;
        int n = prices.size();
        if (k > n / 2) {
            int res = 0;
            for (int i = 1; i < n; i ++ )
                if (prices[i] > prices[i - 1])
                    res += prices[i] - prices[i - 1];
            return res;
        }
        memset(f, -0x3f, sizeof f);
        memset(g, -0x3f, sizeof g);
        f[0] = 0;
        int res = 0;
        for (int i = 1; i <= n; i ++ )
            for (int j = k; j >= 0; j -- ) {
                g[j] = max(g[j], f[j] - prices[i - 1]);
                if (j) f[j] = max(f[j], g[j - 1] + prices[i - 1]);
            }
        for (int i = 1; i <= k; i ++ ) res = max(res, f[i]);
        return res;
    }
};
```

##### **Python**

```python
# 如何把交易状态描述清楚
# 第一个状态：手中有货； ==> 1）可以持有；2）卖出
# 第二个状态：手中没有货； ==> 1) 不买，就继续是没货；2）第二天买入，就是持有状态
# 状态转移的时候 是有权重的，+ w[i], - w[1]

# 状态表示：f[i, j, 0] : 前i天，已经做完j次交易，并且手中无货的购买方式的集合
#           f[i, j, 1] : 前i天，已经做完前j-1次交易，并且正在进行第j次交易，并且手中有货的购买方式的集合 ！ 注意：这里是正在进行第j次交易
# 状态机的状态表示，实质上是把i的状态进行了，方便后续状态计算； 属性：最大值
# 状态计算：就是状态机的转移

# 注意：
# 1. 初始化的问题：f[i,0,0]表示进行0次交易 手中无货的情况，那就是0，表示这个状态合法，可以从这个状态转移过来；状态不合法的时候，就要初始化无穷大
#    求最大，就初始化为负无穷；求最小，就初始化为最大，表示为：状态不合法，没办法从这个状态转移过来
# 2. 最后的结果输出问题：最后一定是进行了若干次完整的交易，手中无货才是完整交易（买了不卖，不是最优解，买要花钱）

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        # 假如一共有n天，那最多买卖n/2次（因为买卖不能在同一天），因此如果k>n/2的话，可以直接k=n/2
        # 那就是可以交易无限次; 特判。
        n = len(prices)
        ans = 0 
        if k >= n // 2:
            for i in range(n - 1):
                if prices[i+1] > prices[i]:
                    ans += prices[i+1] - prices[i]
            return ans

        n = len(prices)
        N = 1010
        f = [[float('-inf')] * 2 for _ in range(n + 5)]
        f[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                f[j][0] = max(f[j][0], f[j][1] + prices[i - 1])
                f[j][1] = max(f[j][1], f[j - 1][0] - prices[i - 1])
        res = 0
        for i in range(k + 1):
            res = max(res, f[i][0])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)**
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
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int hs = INT_MIN, no = 0, mm = 0;
        for (int i = 0; i < n; ++ i ) {
            // hs = max(hs, no[-2] - prices[i]);
            // no = max(no, hs[-1] + prices[i]);
            int t = no;
            no = max(no, hs + prices[i]);
            hs = max(hs, mm - prices[i]);
            mm = t;
        }
        return no;
    }
};
```

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) return 0;
        int n = prices.size(), INF = 1e8;
        vector<vector<int>> f(n, vector<int>(3, -INF));
        f[0][1] = -prices[0], f[0][0] = 0;
        for (int i = 1; i < n; i ++ ) {
            f[i][0] = max(f[i - 1][0], f[i - 1][2]);
            f[i][1] = max(f[i - 1][1], f[i - 1][0] - prices[i]);
            f[i][2] = f[i - 1][1] + prices[i];
        }
        return max(f[n - 1][0], max(f[n - 1][1], f[n - 1][2]));
    }
};
```

##### **Python**

```python
"""
以 线性 的方式 动态规划，考虑第 i 阶段/天 的状态，需要记录的参数有哪些：
第 i 天的 决策状态：
(j=0) 当前没有股票，且不处于冷冻期 （空仓）
(j=1) 当前有股票 （持仓）
(j=2) 当前没有股票，且处于冷冻期 （冷冻期）

状态机模型：
如果第 i 天是 空仓 (j=0) 状态，则 i-1 天可能是 空仓 (j=0) 或 冷冻期 (j=1) 的状态
如果第 i 天是 冷冻期 (j=2) 状态，则 i-1 天只可能是 持仓 (j=1) 状态 （卖出）
如果第 i 天是 持仓 (j=1) 状态，则 i-1 天可能是 持仓 (j=1) 状态 或 空仓 (j=0) 的状态 （买入）

状态表示f[i,j]—属性: 考虑前 i 天股市，当前第 i 天的状态是 j 的方案；方案的总利润 最大Max

入口：一开始是第0天，并且一定是处于可以买票的状态的，所以：f[0][2]=0；其他状态全部负无穷
出口： 最后一天票子留在手里肯定是不合算的：最后一天要么是我刚刚卖出去，要么是我处于冷冻期中（或出了冷冻期）
所以答案应该是在f[n][0]f[n][0]和f[n][2]f[n][2]中选，即 ans = max(f[n][0],f[n][2]);
"""

class Solution:
    def maxProfit(self, w: List[int]) -> int:
        n = len(w)
        f = [[float('-inf')] * 3 for _ in range(n + 1)]
        f[0][2] = 0  # 初始化，入口很重要

        for i in range(1, n + 1):
            f[i][0] = max(f[i-1][0], f[i-1][2] - w[i-1])
            f[i][1] = f[i-1][0] + w[i-1]
            f[i][2] = max(f[i-1][1], f[i-1][2]) 
        return max(f[n][1], f[n][2])
        # 2的状态可以由1转移过来，不会增加w值；但存在极端情况，如数列递减，这时不交易才是最大收益，就是f[n][2]，所以出口需要加上f[n][2]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)**
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
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        vector<int> fin(n + 1, INT_MIN / 2), fout(n + 1, 0);
        for (int i = 1; i <= n; ++ i ) {
            fin[i] = max(fin[i - 1], fout[i - 1] - prices[i - 1]);
            fout[i] = max(fout[i - 1], fin[i - 1] + prices[i - 1] - fee);
        }
        return fout[n];
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

### 打家劫舍

> [!NOTE] **[LeetCode 198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)**
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
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        int st = 0, no = 0;
        for (int i = 0; i < n; ++ i ) {
            int a = st, b = no;
            st = b + nums[i];
            no = max(a, b);
        }
        return max(st, no);
    }
};

// yxc
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        vector<int> f(n + 1), g(n + 1);
        for (int i = 1; i <= n; i ++ ) {
            f[i] = g[i - 1] + nums[i - 1];
            g[i] = max(f[i - 1], g[i - 1]);
        }
        return max(f[n], g[n]);
    }
};
```

##### **Python**

```python
# 正常dp思维：f[i]: 前i天 可以最多赚到多少钱；
# 1. 第i天不偷，那就是f[i-1] ；2. 第i天偷，那就只能i-2天偷！

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        f = [0] * (n + 1)
        for i in range(1, n + 1):
            f[i] = max(f[i-1], f[i-2] + nums[i-1])
        return f[n]
      
# 状态机dp解法
#  f(i) 表示考虑了前 ii 个房间，且盗窃了第 i 个房间所能得到的最大收益，g(i) 表示不盗窃第 i 个房间所能得到的最大收益
# f[i]：表示在第i家偷，那i-1家就不能偷，f[i] = g[i-1] + nums[i-1]
# g[i]: 表示不偷第i家，那i-1家也是可偷可不偷, g[i] = max(g[i-1], f[i-1])
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        f = [0] * (n + 1); g = [0] * (n + 1)
        # f[1] = nums[0]  
        for i in range(1, n + 1): # 由于没有直接初始化特判别f[1] = nums[0] 所以 这里需要从1开始计算
            f[i] = g[i-1] + nums[i-1]
            g[i] = max(f[i-1], g[i-1])
        return max(f[n], g[n])

      
# 状态机dp解法      
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        f = [0] * (n + 1); g =[0] * (n + 1)
        f[1] = nums[0]
        for i in range(2, n + 1): 
            f[i] = g[i - 1] + nums[i - 1]
            g[i] = max(f[i - 1], g[i -1])
        return max(f[n], g[n])

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)**
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
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return nums[0];
        int st = 0, no = 0;
        for (int i = 0; i < n - 1; ++ i ) {
            int a = st, b = no;
            st = b + nums[i];
            no = max(a, b);
        }
        int res1 = max(st, no);
        st = 0, no = 0;
        for (int i = 1; i < n; ++ i ) {
            int a = st, b = no;
            st = b + nums[i];
            no = max(a, b);
        }
        int res2 = max(st, no);
        return max(res1, res2);
    }
};
```

##### **Python**

```python
 # 相比上一道题，唯一的不同就是第一个 和 最后一个 不能同时选；我们可以将第一个房间单独分离进行讨论。分别是选择第一个房间和不选择第一个房间
# 1. 不选第一个：f[i]表示一定不选第1个 并且选第i个的金额；g[i]表示不选第1个，并且 不选第i个的金额。max(f[n], g[n]) (n可选 可不选，所以可以两个取max)
# 2. 选第一个：f'[i]选1，必选第i个；g'[i]选1，不选第i个。由于选了1了，所以最后一个点不能选，max = g'[n]
# 最后 取两个情况的最大值

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:return 0
        if n == 1:return nums[0]
        res = 0
        f = [0] * (n + 1); g = [0] * (n + 1)

        for i in range(2, n + 1):  # 不选1
            f[i] = g[i - 1] + nums[i - 1]
            g[i] = max(f[i - 1], g[i - 1])
        res = max(f[n], g[n]) # 不选1的话，那最大值是在第n个可选 可不选里取最大
        
        f[1] = nums[0]  # 选1 初始化
        g[1] = float('-inf') # 选1，g[1]就不合理，置为不可达数据 即可
        for i in range(2, n + 1):
            f[i] = g[i - 1] + nums[i - 1]
            g[i] = max(f[i - 1], g[i - 1])
        res = max(res, g[n]) # 由于选了第1个，所以这里只能用g[n]（不选n） 和之前的res作对比
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)**
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
class Solution {
public:
    int res = INT_MIN;
    pair<int, int> dfs(TreeNode* n) {
        if (!n) return {0, 0};
        auto l = dfs(n->left), r = dfs(n->right);
        int zero = max(l.first, l.second) + max(r.first, r.second);
        int one = max(l.first, 0) + max(0, r.first) + n->val;
        res = max(res, max(zero, one));
        return {zero, one};
    }
    int rob(TreeNode* root) {
        if (!root) return 0;
        dfs(root);
        return res;
    }
};


// yxc
class Solution {
public:
    unordered_map<TreeNode*, unordered_map<int, int>>f;

    int rob(TreeNode* root) {
        dfs(root);
        return max(f[root][0], f[root][1]);
    }

    void dfs(TreeNode *root) {
        if (!root) return;
        dfs(root->left);
        dfs(root->right);
        f[root][1] = root->val + f[root->left][0] + f[root->right][0];
        f[root][0] = max(f[root->left][0], f[root->left][1]) + max(f[root->right][0], f[root->right][1]);
    }
};
```

##### **Python**

```python
"""
(树形动规) O(n)O(n)
典型的树形DP问题。

状态表示：
f[i][0]表示已经偷完以 i 为根的子树，且不在 i 行窃的最大收益；
f[i][1]表示已经偷完以 i 为根的子树，且在 i 行窃的最大收益；

状态转移：
f[i][1]：因为在 i 行窃，所以在 i 的子节点不能行窃，只能从f[i->left][0]和f[i->right][0]转移；
f[i][0]：因为不在 i 行窃，所以对 i 的子节点没有限制，直接用左右子节点的最大收益转移即可；
时间复杂度分析：总共有 n 个状态，每个状态进行转移的计算量是 O(1)。所以总时间复杂度是 O(n)。

"""

class Solution:
    def rob(self, root: TreeNode) -> int:
        def dfs(u):
            if not u:return [0, 0]
            l = dfs(u.left)
            r = dfs(u.right)
            return [max(l[0],l[1]) + max(r[0], r[1]), l[0] + r[0] + u.val]

        f = dfs(root)
        return max(f[0], f[1])
```

<!-- tabs:end -->
</details>

<br>

* * *

### 一般递推

> [!NOTE] **[LeetCode 799. 香槟塔](https://leetcode-cn.com/problems/champagne-tower/)**
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
class Solution {
public:
    double champagneTower(int poured, int query_row, int query_glass) {
        vector<vector<double>> f(query_row + 1, vector<double>(query_row + 1));
        f[0][0] = poured;
        for (int i = 0; i < query_row; i ++ )
            for (int j = 0; j <= i; j ++ )
                if (f[i][j] > 1) {
                    double x = (f[i][j] - 1) / 2;
                    f[i + 1][j] += x, f[i + 1][j + 1] += x;
                }
        return min(1.0, f[query_row][query_glass]);
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

> [!NOTE] **[LeetCode 1182. 与目标颜色间的最短距离](https://leetcode-cn.com/problems/shortest-distance-to-target-color/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 自己做的时候想得是记忆化搜索
> 
> 显然有线性关系，可以从左 从右递推 ==> 线性dp 略
> 
> 考虑二分

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    vector<int> c[4];
    vector<int> shortestDistanceColor(vector<int>& colors, vector<vector<int>>& queries) {
        int n = colors.size();
        for (int i = 1; i <= 3; ++ i )
            c[i].push_back(-inf);
        for (int i = 0; i < n; ++ i )
            c[colors[i]].push_back(i);
        for (int i = 1; i <= 3; ++ i )
            c[i].push_back(inf);
        
        vector<int> res;
        for (auto & q : queries) {
            int id = q[0], cc = q[1];
            if (c[cc].size() == 2) res.push_back(-1);
            else if (colors[id] == cc) res.push_back(0);
            else {
                auto it1 = lower_bound(c[cc].begin(), c[cc].end(), id);
                auto it2 = it1 - 1;
                int v1 = *it1, v2 = *it2;
                res.push_back(min(id - v2, v1 - id));
            }
        }
        return res;
    }
};
```

##### **C++ 错误写法**

错误在于：从左侧计算值，某个位置起只能向右走时，右侧可能不可达，造成右侧记忆值为 inf 从而不为 -1 最终造成结果有误

```cpp
// 错误代码
const int N = 50010;
const int inf = 0x3f3f3f3f;

int f[N][3];

class Solution {
public:
    vector<int> cs;
    int n;
    
    // dir = 3 同时向左向右 1向左 2向右
    int dfs(int x, int c, int dir) {
        string d;
        if (f[x][c] != -1) return f[x][c];
        if (cs[x] - 1 == c) return f[x][c] = 0;
        int ret = inf;
        if ((dir >> 1 & 1) && x + 1 < n) ret = min(ret, dfs(x + 1, c, 2) + 1);
        if ((dir & 1) && x - 1 >= 0) ret = min(ret, dfs(x - 1, c, 1) + 1);
        return f[x][c] = ret;
    }
    
    void init() {
        memset(f, -1, sizeof f);
        for (int i = 0; i < N; ++ i )
            for (int j = 0; j < 3; ++ j ) f[i][j] = -1;
        for (int i = 0; i < n; ++ i ) {
            dfs(i, 0, 3, i);
            dfs(i, 1, 3, i);
            dfs(i, 2, 3, i);
        }
    }
    
    vector<int> shortestDistanceColor(vector<int>& colors, vector<vector<int>>& queries) {
        this->cs = colors;
        n = cs.size();
        init();
        
        vector<int> res;
        for (auto & q : queries) {
            int v = f[q[0]][q[1] - 1];
            if (v > inf / 2) res.push_back(-1);
            else res.push_back(v);
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

> [!NOTE] **[LeetCode 1220. 统计元音字母序列的数目](https://leetcode-cn.com/problems/count-vowels-permutation/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性dp 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int mod = 1e9 + 7;
    int countVowelPermutation(int n) {
        vector<vector<long long>> f(n + 1, vector<long long>(5));
        f[1][0] = f[1][1] = f[1][2] = f[1][3] = f[1][4] = 1;
        for (int i = 2; i <= n; ++ i ) {
            f[i][0] = (f[i - 1][1] + f[i - 1][2] + f[i - 1][4]) % mod;
            f[i][1] = (f[i - 1][0] + f[i - 1][2]) % mod;
            f[i][2] = (f[i - 1][1] + f[i - 1][3]) % mod;
            f[i][3] = (f[i - 1][2]) % mod;
            f[i][4] = (f[i - 1][2] + f[i - 1][3]) % mod;
        }
        long long res = 0;
        for (int i = 0; i < 5; ++ i ) res = (res + f[n][i]) % mod;
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

> [!NOTE] **[LeetCode 1262. 可被三整除的最大和](https://leetcode-cn.com/problems/greatest-sum-divisible-by-three/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 自己写了不少垃圾代码 只留一个标准代码

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int maxSumDivThree(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> f(n + 1, vector<int>(3, -inf));
        f[0][0] = 0;
        for (int i = 1; i <= n; ++i)
            for (int j = 0; j < 3; ++j)
                f[i][(j + nums[i - 1]) % 3] = max(
                    f[i - 1][(j + nums[i - 1]) % 3], f[i - 1][j] + nums[i - 1]);
        return f[n][0];
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

> [!NOTE] **[LeetCode 1335. 工作计划的最低难度](https://leetcode-cn.com/problems/minimum-difficulty-of-a-job-schedule/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 把 n 个 job 分成连续的 d 段，每一段的代价是本段最大数值的值
> 
> 线性dp即可
> 
> 这里每次都计算了 k+1 ~ i 的最大值 也可以预处理减少一些复杂度 这里的数据范围 预处理不太重要

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 线性dp 第i个工作作为第j天结束 返回 dp[n][d];
    // dp[i][j] = min(dp[i][j], dp[k][j-1] + max(k+1...i));   // j-1 < k < i
    int minDifficulty(vector<int>& jobDifficulty, int d) {
        int n = jobDifficulty.size();
        if (n < d) return -1;
        vector<vector<int>> dp(n + 1, vector<int>(d + 1, INT_MAX));
        dp[0][0] = 0;

        for (int i = 1; i <= n; ++i) {  // 第i天
            // 作为第j天的结束 1 <= j <= min(i,d);
            for (int j = 1; j <= i && j <= d; ++j) {
                for (int k = j - 1; k < i;
                     ++k) {  // 选择j-1天结束的位置k  j-1 <= k < i
                    if (dp[k][j - 1] == INT_MAX)
                        continue;  // 不可能直接跳过 否则会有 INT_MAX+v
                    int maxv = INT_MIN;
                    // 从 k+1 ~ i 选择最大的作为本天的消耗
                    for (int p = k + 1; p <= i; ++p)
                        maxv = max(maxv, jobDifficulty[p - 1]);
                    dp[i][j] = min(dp[i][j], dp[k][j - 1] + maxv);
                }
            }
        }
        return dp[n][d];
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

> [!NOTE] **[LeetCode 1639. 通过给定词典构造目标字符串的方案数](https://leetcode-cn.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性 dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
    const int mod = 1e9 + 7;
    int numWays(vector<string>& words, string target) {
        int n = words[0].size(), m = target.size();
        vector<vector<int>> cnt(n + 1, vector<int>(26));
        vector<vector<long long>> f(n + 1, vector<long long>(m + 1));
        for (auto& w : words)
            for (int i = 0; i < n; ++i) ++cnt[i + 1][w[i] - 'a'];
        for (int i = 0; i <= n; ++i) f[i][0] = 1;
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= m; ++j)
                f[i][j] = (f[i - 1][j] +
                           f[i - 1][j - 1] * cnt[i][target[j - 1] - 'a']) %
                          mod;
        return f[n][m];
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

> [!NOTE] **[LeetCode 1680. 连接连续二进制数字](https://leetcode-cn.com/problems/concatenation-of-consecutive-binary-numbers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性递推即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    // 连接起来的数字本质是
    typedef long long LL;
    const LL mod = 1e9 + 7;
    LL getw(LL x) {
        LL res = 0;
        while (x) {
            x >>= 1;
            ++ res;
        }
        return res;
    }
    int concatenatedBinary(int n) {
        vector<LL> f(n + 1);
        f[1] = 1;
        for (int i = 2; i <= n; ++ i ) {
            LL t = (LL)pow(2, getw(i)) % mod;
            f[i] = f[i - 1] * t % mod + i;
            f[i] %= mod;
        }
        return f[n];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int concatenatedBinary(int n) {
        long long res = 0, mod = 1e9 + 7;
        vector<int> g(n + 1);
        for (int i = 1; i <= n; i ++ ) {
            g[i] = g[i / 2] + 1;
            (res = (res << g[i]) + i) %= mod;
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

> [!NOTE] **[LeetCode 1955. 统计特殊子序列的数目](https://leetcode-cn.com/problems/count-number-of-special-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 核心在于 **状态定义** 和 **转移**
> 
> 前 i 个位置分别构成 0 / 01 / 012 形式序列的方案数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    int countSpecialSubsequences(vector<int>& nums) {
        LL a = 0, b = 0, c = 0;
        for (auto x : nums) {
            if (x == 0)
                // 不选本个 a
                // 选本个 则可以与前面连也可以不连 共a+1
                // 合计 a*2+1
                a = (a * 2 + 1) % MOD;
            if (x == 1)
                b = (b * 2 + a) % MOD;
            if (x == 2)
                c = (c * 2 + b) % MOD;
        }
        return c;
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

### 复杂递推

#### 数学递推 dp

> [!NOTE] **[LeetCode 1259. 不相交的握手](https://leetcode-cn.com/problems/handshakes-that-dont-cross/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 划分
> 
> $$
> dp[n]= \sum_{j=1}^{n/2} dp[2*j−2]*dp[n−2*j]
> $$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int mod = 1e9 + 7;
    int numberOfWays(int num_people) {
        int n = num_people;
        vector<long long> f(n + 1);
        f[0] = 1;
        // i 个人的情况
        // 第n个人与第j个相连
        for (int i = 2; i <= n; i += 2)
            // f[i] = 0;
            for (int j = 1; j < i; j += 2)
                f[i] += f[j - 1] * f[i - j - 1] % mod, f[i] %= mod;
        return f[n];
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

> [!NOTE] **[LeetCode 1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 组合数 其中左右边界限制约束组合数计算
> 
> 直接dp
> 
> 压掉一维：
> 
> > 当 `f[i][j] = f[i - 1][j] + f[i - ][j] + f[i - 1][j + 1]` 的形式时 
> > 
> > 【形式可能略微有所变动，但基本是本维依赖上一维度，且依赖上一维度某个循环更新顺序】
> > 
> > ```cpp
> > for (int j = 0; j <= longest; ++ j ) {
> >     f[i][j] = f[i - 1][j - 1] + f[i - 1][j] + f[i - 1][j + 1];
> >
> >     // === >
> >     //     已被本维度覆盖    未被覆盖的部分
> >     f[j] = f[j - 1]  +   f[j] + f[j + 1];
> >     // 显然需要临时变量每次记录当前 j [左侧且上一维度] 的值
> > }
> >     // === > 
> > int t = 0;  // memo the f[i - 1][j - 1]
> > for (int j = 0; j < longest; ++ j ) {
> >     int t_next = f[j];
> >     f[j] = t + f[j] + f[j + 1];
> >     t = t_next;
> > }
> > ```
> > 
> > 显然，`int t_next = f[j]; ... t = t_next;` 这样的操作可以直接用一个 swap 实现
> > ```cpp
> > int t = 0;
> > for (int j = 0; j <= longest; ++ j ) {
> >     swap(t, f[j]);
> >     f[j] = (f[j] + t + f[j + 1]) % MOD;
> > }
> > ```
> 
> **经典优化 待归类整理**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int MOD = 1e9 + 7;

    int numWays(int steps, int arrLen) {
        int longest = min(steps, arrLen - 1);
        vector<vector<int>> f(steps + 1, vector<int>(longest + 1));
        f[0][0] = 1;
        for (int i = 1; i <= steps; ++ i )
            for (int j = 0; j <= longest; ++ j ) {
                f[i][j] = f[i - 1][j];
                if (j > 0)
                    (f[i][j] += f[i - 1][j - 1]) %= MOD;
                if (j < longest)
                    (f[i][j] += f[i - 1][j + 1]) %= MOD;
            }
        return f[steps][0];
    }
};
```

##### **C++ 优化**

```cpp
class Solution {
public:
    using LL = long long;
    const static int MOD = 1e9 + 7;

    int numWays(int steps, int arrLen) {
        int longest = min(steps, arrLen - 1);
        vector<LL> f(longest + 1 + 1);
        f[0] = 1;
        for (int i = 1; i <= steps; ++ i ) {
            LL t = 0;   // avoid overflow
            for (int j = 0; j <= longest; ++ j ) {
                LL tt = f[j];
                f[j] = (f[j] + t + f[j + 1]) % MOD;
                t = tt;
            }
        }
        return f[0];
    }
};
```

##### **C++ 优化 swap**

```cpp
class Solution {
public:
    using LL = long long;
    const static int MOD = 1e9 + 7;

    int numWays(int steps, int arrLen) {
        int longest = min(steps, arrLen - 1);
        vector<LL> f(longest + 1);
        f[0] = 1;
        for (int i = 1; i <= steps; ++ i ) {
            LL t = 0;   // avoid overflow
            for (int j = 0; j <= longest; ++ j ) {
                swap(t, f[j]);
                // 可以直接给数组长度再加一 避免下面的 j != longest 的判断
                f[j] = (f[j] + t + (j != longest ? f[j + 1] : 0)) % MOD;
            }
        }
        return f[0];
    }
};
```

##### **C++ 另一 经典优化**

```cpp
class Solution {
    const int mod = 1e9 + 7;
    int numWays(int steps, int arrLen) {
        vector<long long> f(steps + 1);
        f[0] = 1;
        for (int s = 1; s <= steps; ++s) {
            long long tmp = 0;
            for (int i = 0; i < min(min(arrLen, s + 1), steps - s + 1); ++i) {
                swap(f[i], tmp);
                f[i] = (tmp + f[i] + f[i + 1]) % mod;
            }
        }
        return f[0];
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

> [!NOTE] **[LeetCode 1411. 给 N x 3 网格图涂色的方案数](https://leetcode-cn.com/problems/number-of-ways-to-paint-n-x-3-grid/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数学找规律

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    long long mod = 1e9 + 7;
    int numOfWays(int n) {
        long long last3 = 6, last2 = 6;
        for (int i = 2; i <= n; ++i) {
            long long nlast3 = last3 * 2 % mod + last2 * 2 % mod;  // pass
            long long nlast2 = last3 * 2 % mod + last2 * 3 % mod;
            last3 = nlast3 % mod;
            last2 = nlast2 % mod;
        }
        return (last3 + last2) % mod;
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

> [!NOTE] **[LeetCode 1420. 生成数组](https://leetcode-cn.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂线性递推 + 前缀和优化
> 
> TODO: 重复做 更优雅的实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 55, M = 110, MOD = 1e9 + 7;

    // 有 i 个数，搜索代价为 j ，最大值为 k 的所有方案
    int f[N][N][M];

    int numOfArrays(int n, int m, int _k) {
        if (!_k)
            return 0;
        // 只有一个数，搜索代价为 1 的方案
        memset(f, 0, sizeof f);
        for (int i = 1; i <= m; ++ i )
            f[1][1][i] = 1;
        
        // i: 数的个数
        for (int i = 2; i <= n; ++ i )
            // j: 搜索代价
            for (int j = 1; j <= _k && j <= i; ++ j ) {
                // 优化代码:
                int sum = 0;

                // k: 最大值
                for (int k = 1; k <= m; ++ k ) {
                    // 1. 最大值出现在前 i - 1 个元素中，则数组末尾的元素可以从 1 到 k 中随便取
                    f[i][j][k] = (LL)f[i - 1][j][k] * k % MOD;
                    // 2. 最大值出现在数组末尾，则此前搜索代价为 j - 1
                    // for (int x = 0; x < k; ++ x )
                    //     f[i][j][k] = ((LL)f[i][j][k] + f[i - 1][j - 1][x]) % MOD;
                    // 优化代码:
                    f[i][j][k] = ((LL)f[i][j][k] + sum) % MOD;
                    sum = (sum + f[i - 1][j - 1][k]) % MOD;
                }
            }
        
        int res = 0;
        for (int i = 1; i <= m; ++ i )
            res = (res + f[n][_k][i]) % MOD;
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

> [!NOTE] **[LeetCode 1467. 两个盒子中球的颜色数相同的概率](https://leetcode-cn.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp + 组合数
> 
> [题解](https://leetcode-cn.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/solution/cdong-tai-gui-hua-bi-sai-de-shi-hou-bei-fan-yi-ken/)
> 
> 重复做 todo

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    double getProbability(vector<int>& balls) {
        // 颜色数和球的数量
        const int k = balls.size();
        const int n = accumulate(balls.begin(), balls.end(), 0) / 2;
        // 预处理阶乘
        vector<double> fact;
        fact.push_back(1.0);
        for (int i = 1; i <= 2 * n; ++i) { fact.push_back(fact[i - 1] * i); }
        // 总的排列方法数
        double total = fact[2 * n];
        for (auto ball : balls) { total /= fact[ball]; }
        // 动态规划
        vector<vector<double>> dp(2 * n + 1, vector<double>(2 * k + 1, 0.0));
        dp[0][k] = 1.0;
        int num = 0;
        for (int i = 0; i < k; ++i) {
            vector<vector<double>> next(2 * n + 1,
                                        vector<double>(2 * k + 1, 0.0));
            for (int j = 0; j <= balls[i]; ++j) {
                int trans = 0;
                trans = j == 0 ? -1 : trans;
                trans = j == balls[i] ? 1 : trans;
                for (int front = 0; front <= 2 * n; ++front)
                    for (int color = 0; color <= 2 * k; ++color) {
                        if (dp[front][color] == 0) continue;
                        double ways = dp[front][color];
                        ways *= fact[front + j] / (fact[front] * fact[j]);
                        ways *= fact[num - front + balls[i] - j] /
                                (fact[num - front] * fact[balls[i] - j]);
                        next[front + j][color + trans] += ways;
                    }
            }
            swap(dp, next);
            num += balls[i];
        }
        return dp[n][k] / total;
    }
};

// mskadr
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1473. 粉刷房子 III](https://leetcode-cn.com/problems/paint-house-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp 状态：$dp[i][j][k]$ 第 i 个房子 用第 j 种颜色此时形成了 k 个街区

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int dp[105][22][105];
    int minCost(vector<int>& houses, vector<vector<int>>& cost, int m, int n,
                int target) {
        for (int i = 0; i <= m; ++i)
            for (int j = 0; j <= n; ++j)
                for (int k = 0; k <= target; ++k) dp[i][j][k] = 1e9;
        for (int j = 1; j <= n; ++j) {
            if (houses[0] && houses[0] != j) continue;
            int c = houses[0] ? 0 : cost[0][j - 1];
            dp[1][j][1] = c;
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (houses[i - 1] && houses[i - 1] != j) continue;
                int c = houses[i - 1] ? 0 : cost[i - 1][j - 1];
                for (int k = 1; k <= target; ++k) {
                    for (int l = 1; l <= n; ++l) {
                        if (l == j)
                            dp[i][j][k] = min(dp[i][j][k], dp[i - 1][l][k] + c);
                        else
                            dp[i][j][k] =
                                min(dp[i][j][k], dp[i - 1][l][k - 1] + c);
                    }
                }
            }
        }
        int res = 1e9;
        for (int j = 1; j <= n; ++j) res = min(res, dp[m][j][target]);
        return res == 1e9 ? -1 : res;
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

> [!NOTE] **[LeetCode 1739. 放置盒子](https://leetcode-cn.com/problems/building-boxes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 如果摆 k 层，共可以摆
> 
> $$
> (1 * 2 + 2 * 3 + ... + k * (k + 1)) / 2 = k * (k + 1) * (k + 2) / 6
> $$
> 
> 个方块
> 
> 考虑满 k 个之后在底面再铺，已有方块为 k * (k - 1) / 2
> 
> **随后令 k = 1 则第 k 次增加接触地面面积可以增加 k 个放置总量**
> 
> 自己做法一开始超时 优化即AC

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    int minimumBoxes(int n) {
        int sum = 0, k = 1;
        while (sum + k * (k + 1) / 2 <= n) {
            sum += k * (k + 1) / 2;
            k ++ ;
        }
        
        int res = k * (k - 1) / 2;
        k = 1;
        while (sum < n) {
            sum += k;
            k ++ ;
            res ++ ;
        }
        return res;
    }
};
```

##### **C++ TLE**

本质先求出必定大于等于 n 的方块塔，再挨个往下减

二分的部分计算较多，超时。。。距离 AC 一步之遥

```cpp
class Solution {
public:
    using LL = long long;
    int calc(int x) {
        return (LL)x * (x + 1) / 2;
    }
    LL get(int x) {
        LL res = 0;
        while (x) {
            res += calc(x);
            -- x ;
        }
        return res;
    }
    int minimumBoxes(int n) {
        int l = 0, r = n;
        while (l < r) {
            int m = l + r >> 1;
            if (get(m) < n) l = m + 1;
            else r = m;
        }
        // cout << "l = " << l << " get = " << get(l) << endl;
        int res = calc(l), tot = get(l), x = l;
        while (tot - x >= n) {
            tot -= x;
            res -= 1;
            x -- ;
        }
        return res;
    }
};
```

##### **C++ 优化后AC**

```cpp
class Solution {
public:
    int minimumBoxes(int n) {
        int sum = 0, k = 1;
        for (;;) {
            sum += k * (k + 1) / 2;
            if (sum >= n) break;
            k ++ ;
        }
        
        int res = k * (k + 1) / 2;
        while (sum - k >= n) {
            sum -= k;
            k -- ;
            res -- ;
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

> [!NOTE] **[LeetCode 1866. 恰有 K 根木棍可以看到的排列数目](https://leetcode-cn.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典题目 dp时考虑当前枚举的是所有当中最小的即可
> 
> 和另一题假定枚举的是当前 `最大/最高` 的类似
> 
> > 另一种思路是 园排列
> > 
> > 本质是第一类斯特林数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    const static int N = 1010;
    LL f[N][N];    // 用了高度1-i 左侧可以看到j个 最终 f[n][k]
    
    int rearrangeSticks(int n, int k) {
        f[1][1] = 1;
        for (int i = 2; i <= n; ++ i )
            for (int j = 1; j <= i; ++ j )
                f[i][j] = (f[i - 1][j - 1] + f[i - 1][j] * (i - 1) % MOD) % MOD;
        
        return f[n][k];
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

> [!NOTE] **[LeetCode 1883. 准时抵达会议现场的最小跳过休息次数](https://leetcode-cn.com/problems/minimum-skips-to-arrive-at-meeting-on-time/)** [TAG]
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
class Solution {
public:
    const double eps = 1e-8, INF = 1e9;
    const static int N = 1010;
    double f[N][N];

    int minSkips(vector<int>& dist, int speed, int hoursBefore) {
        int n = dist.size();
        for (int i = 1; i <= n; ++ i ) {
            double t = (double)dist[i - 1] / speed;
            for (int j = 0; j <= i; ++ j ) {
                f[i][j] = INF;
                if (j <= i - 1)
                    f[i][j] = ceil(f[i - 1][j] + t - eps);
                if (j)
                    f[i][j] = min(f[i][j], f[i - 1][j - 1] + t);
            }
        }
        for (int i = 0; i <= n; ++ i )
            if (f[n][i] <= hoursBefore)
                return i;
        return -1;
    }
};
```

##### **C++ 习惯写法**

```cpp
class Solution {
public:
    const double eps = 1e-8, INF = 2e9;
    const static int N = 1010;
    double f[N][N];
    
    int minSkips(vector<int>& dist, int speed, int hoursBefore) {
        int n = dist.size();
        for (int i = 0; i <= n; ++ i )
            for (int j = 0; j <= n; ++ j )
                f[i][j] = INF;
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            f[i][0] = ceil(f[i - 1][0] - eps) + (double)dist[i - 1] / speed;
        
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= i; ++ j ) {
                double t = (double)dist[i - 1] / speed;
                f[i][j] = min(ceil(f[i - 1][j] - eps), f[i - 1][j - 1]) + t;
            }
        for (int i = 0; i <= n; ++ i )
            if (f[n][i] <= hoursBefore)
                return i;
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

> [!NOTE] **[LeetCode 1987. 不同的好子序列数目](https://leetcode-cn.com/problems/number-of-unique-good-subsequences/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> > https://www.acwing.com/solution/content/65104/
> 
> 重复做 TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int MOD = 1e9 + 7;
    
    int numberOfUniqueGoodSubsequences(string binary) {
        int n = binary.size();
        
        // f[i]表示从第i个字符往后，构造出1开头的不同子序列的个数
        // g[i]表示从第i个字符往后，构造出0开头的不同子序列的个数
        // Case 1: 第i个字符为1，此时显然有 g[i] = g[i + 1]，考虑 f[i]
        //           第一类:使用第i个字符       第二类:不使用第i个字符
        // f[i] = (f[i + 1] + g[i + 1] + 1)  +  (f[i + 1])
        // 其中，两类内部无重复，但两类之间有重复。
        // 重复即第二类一定完全是第一类的子集，去除第二类的计数即可。
        // Case 2: 第i个字符为0，此时显然有 f[i] = f[i + 1]，类似考虑 g[i] 即可
        // ...
        //   初始化: f[n] = g[n] = 0;
        int f = 0, g = 0;
        bool has_zero = false;
        
        for (int i = n - 1; i >= 0; -- i )
            if (binary[i] == '0') {
                g = (f + g + 1) % MOD;
                has_zero = true;
            } else
                f = (f + g + 1) % MOD;
        
        return f + has_zero;
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

#### 栅栏涂色

> [!NOTE] **[LeetCode 276. 栅栏涂色](https://leetcode-cn.com/problems/paint-fence/)**
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
// 对于本题条件较简单，可以优化滚动数组至常量 略
class Solution {
public:
    const static int N = 55, M = 1e5 + 10;

    int f[N][M][2];

    int numWays(int n, int k) {
        memset(f, 0, sizeof f);
        
        f[0][0][1] = 1;
        int s = 1;        
        for (int i = 1; i <= n; ++ i ) {
            int t = 0;
            for (int j = 1; j <= k; ++ j ) {
                // 颜色不同的所有
                f[i][j][0] = s - f[i - 1][j][0] - f[i - 1][j][1];
                // 上一个和本个颜色相同
                f[i][j][1] = f[i - 1][j][0];

                t += f[i][j][0] + f[i][j][1];
            }
            s = t;
        }
        return s;
    }
};
```

##### **C++ 优化1**

无需讨论某个具体的颜色 直接标识与上一个是否相同

```cpp
// 仍然可优化至常量
class Solution {
public:
    const static int N = 55, M = 1e5 + 10;

    // f[i][0] 和前一个不一样 f[i][1] 和前一个一样
    int f[N][2];

    int numWays(int n, int k) {
        memset(f, 0, sizeof f);

        f[1][0] = k, f[1][1] = 0;
        for (int i = 2; i <= n; ++ i ) {
            f[i][0] = (f[i - 1][0] + f[i - 1][1]) * (k - 1);
            f[i][1] = f[i - 1][0];
        }
        
        return f[n][0] + f[n][1];
    }
};
```

##### **C++ 优化2 更进一步**

- 第 n 个栅栏如果和上一个不同颜色，则有 $f[i-1] * (k-1)$ 个方案数
- 第 n 个栅栏如果和上一个同颜色，那么上一个和前一个就不能同颜色，则有 $f[i-2] * (k-1)$
- 第 n 个栅栏上色方案数合计：$f[i-1] * (k-1) + f[i-2] * (k-1)$

```cpp
class Solution {
public:
    const static int N = 55, M = 1e5 + 10;

    int f[N];

    int numWays(int n, int k) {
        memset(f, 0, sizeof f);

        f[1] = k, f[2] = k * k;
        for (int i = 3; i <= n; ++ i )
            //          颜色不同       +       颜色相同
            f[i] = f[i - 1] * (k - 1) + f[i - 2] * (k - 1);
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

#### 拓展 环形涂色问题

![cicle](https://camo.githubusercontent.com/5c6e5705b55b7bf9faf0feba908d6faee0354ab7bc454de9bd0c9f464e5624e5/68747470733a2f2f706963342e7a68696d672e636f6d2f38302f76322d35393636613564336161626562326331653765396261326333613832646534305f31343430772e6a7067)


### 分层最短路

> [!NOTE] **[LeetCode 1824. 最少侧跳次数](https://leetcode-cn.com/problems/minimum-sideway-jumps/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然分层最短路 考虑递推dp
>  
> > 分层最短路的特点 只能从上一层到达下一层 拓扑无环
> > 
> > 考虑当前层只由上一层的点来更新 不考虑本层内
> > 
> > if (b[i] != k + 1) 本质是上一层直接向右走时无障碍
> 
> > **本质上是转移思路的转化**
> 
> 双端队列也能过 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 500010, INF = 1e8;

int f[N][3];

class Solution {
public:
    // 注意题意 可以跳的时候不借助障碍
    
    int minSideJumps(vector<int>& b) {
        int n = b.size() - 1;
        f[0][0] = f[0][2] = 1, f[0][1] = 0;
        
        for (int i = 1; i <= n; ++ i )
            for (int j = 0; j < 3; ++ j )
                if (b[i] == j + 1)
                    f[i][j] = INF;
                else {
                    f[i][j] = INF;
                    for (int k = 0; k < 3; ++ k )
                        if (b[i] != k + 1)  // ATTENTION
                            f[i][j] = min(f[i][j], f[i - 1][k] + (k != j));
                }
        return min(f[n][0], min(f[n][1], f[n][2]));
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