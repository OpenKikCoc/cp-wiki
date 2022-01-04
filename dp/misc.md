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