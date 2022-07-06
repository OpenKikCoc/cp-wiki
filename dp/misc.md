## 习题

> [!NOTE] **[LeetCode 1889. 装包裹的最小浪费空间](https://leetcode-cn.com/problems/minimum-space-wasted-from-packaging/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 考虑 每个箱子只会装一个包裹 则每个包裹都只需要大于等于其尺寸的最小的箱子即可
> 
> 显然需要遍历供应商
> 
> 如果暴力模拟：
> 
> > 每个包裹二分找对应的箱子 复杂度可能高达 1e5 * 1e5 * log(1e5)
> > 
> > 考虑逆向 对供应商的箱子二分找对应的包裹 则可以划分数个区间 并利用包裹前缀和快速计算浪费的空间 时间复杂度 1e5 * log(1e5)
> 
> 重复做 TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const LL MOD = 1e9 + 7, INF = 1e18;
    
    int minWastedSpace(vector<int>& packages, vector<vector<int>>& boxes) {
        int n = packages.size();
        sort(packages.begin(), packages.end());
        LL sum = accumulate(packages.begin(), packages.end(), 0ll);
        
        LL res = INF;
        for (auto & b : boxes) {
            sort(b.begin(), b.end());
            
            // 供应商不满足要求
            if (b.back() < packages.back())
                continue;
            LL t = -sum, last = 0;
            for (auto x : b) {
                int l = 1, r = n + 1;
                // 找到大于当前的第一个 往前一个就是小于等于的最后一个
                while (l < r) {
                    int m = l + r >> 1;
                    if (packages[m - 1] <= x)
                        l = m + 1;
                    else
                        r = m;
                }
                int next = l - 1;
                if (next <= last)
                    continue;
                t += (next - last) * x;
                last = next;
            }
            res = min(res, t);
        }
        
        if (res == INF)
            return -1;
        return res % MOD;
    }
};
```

##### **C++ 重复利用last**

```cpp
class Solution {
public:
    using LL = long long;
    const LL MOD = 1e9 + 7, INF = 1e18;
    
    int minWastedSpace(vector<int>& packages, vector<vector<int>>& boxes) {
        int n = packages.size();
        sort(packages.begin(), packages.end());
        LL sum = accumulate(packages.begin(), packages.end(), 0ll);
        
        LL res = INF;
        for (auto & b : boxes) {
            sort(b.begin(), b.end());
            
            // 供应商不满足要求
            if (b.back() < packages.back())
                continue;
            LL t = -sum, last = 0;
            for (auto x : b) {
                int l = last + 1, r = n + 1;
                // 找到大于当前的第一个 往前一个就是小于等于的最后一个
                while (l < r) {
                    int m = l + r >> 1;
                    if (packages[m - 1] <= x)
                        l = m + 1;
                    else
                        r = m;
                }
                int next = l - 1;
                if (next <= last)
                    continue;
                t += (next - last) * x;
                last = next;
            }
            res = min(res, t);
        }
        
        if (res == INF)
            return -1;
        return res % MOD;
    }
};


class Solution {
public:
    using LL = long long;
    const LL MOD = 1e9 + 7, INF = 1e18;
    
    int minWastedSpace(vector<int>& packages, vector<vector<int>>& boxes) {
        int n = packages.size();
        sort(packages.begin(), packages.end());
        LL sum = accumulate(packages.begin(), packages.end(), 0ll);
        
        LL res = INF;
        for (auto & b : boxes) {
            sort(b.begin(), b.end());
            
            // 供应商不满足要求
            if (b.back() < packages.back())
                continue;
            LL t = -sum, last = -1;
            for (auto x : b) {
                int l = last + 1, r = n;
                // 找到大于当前的第一个 往前一个就是小于等于的最后一个
                while (l < r) {
                    int m = l + r >> 1;
                    if (packages[m] <= x)
                        l = m + 1;
                    else
                        r = m;
                }
                int next = l - 1;
                if (next <= last)
                    continue;
                t += (next - last) * x;
                last = next;
            }
            res = min(res, t);
        }
        
        if (res == INF)
            return -1;
        return res % MOD;
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

> [!NOTE] **[LeetCode 1937. 扣分后的最大得分](https://leetcode-cn.com/problems/maximum-number-of-points-with-cost/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 尝试根据状态转移方程【拆掉绝对值表达式】
> 
> **经典DP优化**

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

> [!NOTE] **[LeetCode 1977. 划分数字的方案数](https://leetcode-cn.com/problems/number-of-ways-to-separate-numbers/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 dp 优化
> 
> 较显然的，可以得知状态定义及转移方程
> 
> 分析知当前取值需对上一个结尾的数的所有情况求和，故【维护一个前缀和数组】
> 
> 另外当长度相等时需考虑字符串比较，此时需【预处理 LCP 】

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

> [!NOTE] **[Codeforces Good Sequences](http://codeforces.com/problemset/problem/264/B)**
> 
> 题意: 
> 
> 如果一个序列 $x1,x2,...,xk$ 能够满足一下三个条件，那就是一个“好序列”：
> 
> 1. 该序列是严格上升的，即 $x[i] < x[i+1]$（1<=i<=k-1)
> 2. 任意两个相邻的元素是非互质的，即 $gcd(x[i],x[i+1]) > 1$ (1<=i<=k-1)
> 3. 所有的数字都是“好整数”
> 
> 求长度最长的“好序列”

> [!TIP] **思路**
> 
> 暴力显然有 $O(n^2*logn)$ (其中 log 为 gcd 复杂度) 的解决方案
> 
> 考虑数据范围较大，**无需枚举之前选哪个数，只需要关心之前的包含哪个因子**
> 
> ATTENTION: **需要先找到最值，再用最值更新所有包含的因子的 $f$ 值**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Good Sequences
// Contest: Codeforces - Codeforces Round #162 (Div. 1)
// URL: https://codeforces.com/problemset/problem/264/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10;

int primes[N], cnt;
bool st[N];
void init() {
    memset(st, 0, sizeof st);
    cnt = 0;
    for (int i = 2; i < N; ++i) {
        if (!st[i])
            primes[cnt++] = i;
        for (int j = 0; primes[j] <= (N - 1) / i; ++j) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0)
                break;
        }
    }
}

int n;
// 状态定义: 以 primes[x]=i 结尾的最大dp值
// ATTENTION: 先找最大 再每次更新
int f[N];

int main() {
    // case 21 6w+数据 TLE
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();
    memset(f, 0, sizeof f);

    cin >> n;

    for (int i = 0, x; i < n; ++i) {
        cin >> x;

        int maxv = 0;
        for (int j = 0, y = x; j < cnt && y >= primes[j]; ++j)
            if (y % primes[j] == 0) {
                // 找到可以接的最大的
                maxv = max(maxv, f[primes[j]]);
                // 也要除 避免超时
                while (y % primes[j] == 0)
                    y /= primes[j];
            }

        for (int j = 0; j < cnt && x >= primes[j]; ++j)
            if (x % primes[j] == 0) {
                f[primes[j]] = maxv + 1;
                while (x % primes[j] == 0)
                    x /= primes[j];
            }
        if (x > 1)
            f[x] = maxv + 1;
    }
    int res = 1;  // 输入 1,1 的 case , 显然最少长度为 1
    for (int i = 0; i < cnt; ++i)
        res = max(res, f[primes[i]]);
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

### 转化模型

> [!NOTE] **[LeetCode 2321. 拼接数组的最大分数](https://leetcode.cn/problems/maximum-score-of-spliced-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思考必然是其中一个数组的一个连续子段发生变化
> 
> 两数组做差来获取变化的量，随后求 `最大子序和` 即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int get(vector<int> & n1, vector<int> & n2) {
        int n = n1.size();
        vector<int> t(n);
        int sum = 0, diff = 0, pre = 0;
        for (int i = 0; i < n; ++ i ) {
            t[i] = n2[i] - n1[i];
            sum += n1[i];
            pre = max(0, pre) + t[i];
            diff = max(diff, pre);
        }
        return sum + diff;
    }
    
    int maximumsSplicedArray(vector<int>& nums1, vector<int>& nums2) {
        return max(get(nums1, nums2), get(nums2, nums1));
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

> [!NOTE] **[Codeforces A. Flipping Game](https://codeforces.com/problemset/problem/327/A)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 转化为 **最大子序和** 模型以精妙的以线性复杂度处理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 最大子序和**

找到一个区间 区间内 [0的数量 - 1的数量] 差值最大

==>

**计数 最大子序和模型**

把 0 翻转我们就加 1

将 1 翻转我们就加 -1

那么我们只需要计算子序列和最大就可以了

再加上原先的 1 的和 就是最大的 1 的数量

```cpp
// Problem: A. Flipping Game
// Contest: Codeforces - Codeforces Round #191 (Div. 2)
// URL: https://codeforces.com/problemset/problem/327/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const int N = 110;

int f[N];

int main() {
    int n;
    cin >> n;

    int tot = 0;
    for (int i = 0; i < n; ++i) {
        int x;
        cin >> x;
        if (x) {
            ++tot;
            f[i + 1] = max(f[i] - 1, -1);
        } else
            f[i + 1] = max(f[i] + 1, 1);
    }

    int pre = -1e9;
    for (int i = 1; i <= n; ++i)
        pre = max(pre, f[i]);
    cout << pre + tot << endl;

    return 0;
}
```

##### **C++ 前缀和暴力**

```cpp
// Problem: A. Flipping Game
// Contest: Codeforces - Codeforces Round #191 (Div. 2)
// URL: https://codeforces.com/problemset/problem/327/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 找到一个区间 区间内 [0的数量 - 1的数量] 差值最大
// 数据范围显然可以暴力

const int N = 110;

int n;
int s0[N], s1[N];

int main() {
    cin >> n;
    for (int i = 0; i < n; ++i) {
        int x;
        cin >> x;
        if (x) {
            s0[i + 1] = s0[i];
            s1[i + 1] = s1[i] + 1;
        } else {
            s0[i + 1] = s0[i] + 1;
            s1[i + 1] = s1[i];
        }
    }

    // -1e9 cause it needs EXECTLY one operation
    int res = -1e9;
    for (int l = 1; l <= n; ++l)
        for (int r = l; r <= n; ++r)
            res = max(res, s0[r] - s0[l - 1] - s1[r] + s1[l - 1]);
    cout << res + s1[n] << endl;

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

> [!NOTE] **[LeetCode 2272. 最大波动的子字符串](https://leetcode.cn/problems/substring-with-largest-variance/)** [TAG]
> 
> 题意: 
> 
> 求其所有子串中：出现 `次数最多` 与 `次数最少` 的 `两种字符` 的最大 `次数差值`。

> [!TIP] **思路**
> 
> 数据范围显然不能枚举区间，考虑枚举两类 `最多/最少` 的字符
> 
> 则在遍历整个串的过程中，遇到字符分别 `+1/-1` ，所求即转化为 `最大子序和` 。
> 
> **问题在于如何维护 `必然包含两个字符` 的最大子序和 (diff_with_b)**
> 
> **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 显然无法枚举区间，考虑枚举两个元素
    int largestVariance(string s) {
        int res = 0;
        for (char c1 = 'a'; c1 <= 'z'; ++ c1 )      // 较大的
            for (char c2 = 'a'; c2 <= 'z'; ++ c2 )  // 较小的
                if (c1 != c2) {
                    // diff 表示 c1-c2 差值; diff_with_b 表示包含了b的次数差值，初始化-inf
                    int diff = 0, diff_with_b = -1e8, t = 0;
                    for (auto c : s) {
                        if (c == c1)
                            diff ++ , diff_with_b ++ ;
                        else if (c == c2)
                            // ATTENTION: 若 diff < 0 则重置为 0
                            diff_with_b = -- diff , diff = max(diff, 0);
                        t = max(t, diff_with_b);
                    }
                    res = max(res, t);
                }
        return res;
    }
};
```

##### **C++ 另一**

```cpp
class Solution {
public:
    // 显然无法枚举区间，考虑枚举两个元素
    int largestVariance(string s) {
        int res = 0;
        for (char c1 = 'a'; c1 <= 'z'; ++ c1 )      // 较大的
            for (char c2 = 'a'; c2 <= 'z'; ++ c2 )  // 较小的
                if (c1 != c2) {
                    // diff 表示 c1-c2 差值; diff_with_b 表示包含了b的次数差值，初始化-inf
                    int diff = 0, diff_with_b = -1e8, t = 0;
                    for (auto c : s) {
                        if (c == c1)
                            // 思考：diff_with_b 如何保证必然包含 b [更新逻辑 不能取max(0, xxx)]
                            diff = max(0, diff) + 1, diff_with_b = diff_with_b + 1;
                        else if (c == c2)
                            // ATTENTION: 转移; diff 重置
                            diff = diff - 1, diff_with_b = diff, diff = max(diff, 0);
                        t = max(t, diff_with_b);
                    }
                    res = max(res, t);
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

> [!NOTE] **[Codeforces C. George and Job](https://codeforces.com/problemset/problem/467/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 状态定义是核心 多增加感觉
> 
> 一开始想的还是以 i 为结束分为 k 段
> 
> 实际上可以是 在 i 及之前就分为 k 段

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. George and Job
// Contest: Codeforces - Codeforces Round #267 (Div. 2)
// URL: https://codeforces.com/problemset/problem/467/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// note 增加经验
// TLE https://codeforces.com/contest/467/submission/109681416
// WA  https://codeforces.com/contest/467/submission/109682744
// 本题第三重循环要求前面的最值 显然可以直接用前面某个位置的值【需转换状态定义】
// 一开始想成三重循环
using LL = long long;
const int N = 5010;

int n, m, k;
LL s[N], f[N][N];

int main() {
    cin >> n >> m >> k;
    for (int i = 1; i <= n; ++i)
        cin >> s[i], s[i] += s[i - 1];

    for (int i = 1; i <= k; ++i)
        for (int j = max(i, m); j <= n; ++j)
            f[i][j] = max(f[i][j - 1], f[i - 1][j - m] + s[j] - s[j - m]);

    LL res = 0;
    for (int i = 1; i <= n; ++i)
        res = max(res, f[k][i]);
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


> [!NOTE] **[Codeforces C. Tourist Problem](https://codeforces.com/problemset/problem/340/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数论题 规律 推导 优化
> 
> 非常好的题 反复做
> 
> 其中**将计算两重循环绝对值差转化为前缀和的思路**非常精妙 有可拓展性

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Tourist Problem
// Contest: Codeforces - Codeforces Round #198 (Div. 2)
// URL: https://codeforces.com/problemset/problem/340/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 思维 数学
// 根据题意推导：
//    所有 |s[i] - s[j]| 都出现 (n - 1)! 次
//    从 0 开始的 |s[i] - 0| 同样出现 (n - 1)! 次
// 总情况 n! 种
// 答案：
//    for (int i = 1; i <= n; ++ i )
//        for (int j = 0; j <= n; ++ j )
//            t += abs(a[i] - a[j])
//    t * (n - 1)! / n!
//    也即 t / n
// 直接枚举 abs(a[i] - a[j]) 显然 n^2 超时
// 考虑排序维护前缀和 【此时 abs符号可以去掉】
// 两层循环中有一部分可以反过来 值相同 所以可以直接计算一半的部分
// 则计算绝对值差变为 2 * (s[i] * (i - 1) - s[i - 1])

using LL = long long;
const int N = 100010;

LL n;
LL a[N], s[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];
    sort(a + 1, a + n + 1);
    for (int i = 1; i <= n; ++i)
        s[i] = a[i] + s[i - 1];

    LL t = s[n];  // a[i] - 0
    for (int i = 1; i <= n; ++i)
        t += 2 * (a[i] * (i - 1) - s[i - 1]);
    LL g = __gcd(t, n);

    cout << t / g << ' ' << n / g << endl;

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