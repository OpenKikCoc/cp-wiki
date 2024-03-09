## 习题

> [!NOTE] **[LeetCode 1156. 单字符重复子串的最大长度](https://leetcode-cn.com/problems/swap-for-longest-repeated-character-substring/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 记录连续值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
ass Solution {
public:
    int maxRepOpt1(string text) {
        int n = text.size();
        
        vector<char> s(n + 2);
        vector<int> L(n + 2), R(n + 2);
        vector<int> cnt(27);
        
        for (int i = 1; i <= n; ++ i ) s[i] = text[i - 1];
        s[0] = s[n + 1] = 0;
        
        int res = 0;
        for (int i = 1; i <= n; ++ i ) {
            L[i] = s[i] == s[i - 1] ? L[i - 1] + 1 : 1;
            res = max(res, L[i]);
        }
        for (int i = n; i >= 1; -- i ) {
            R[i] = s[i] == s[i + 1] ? R[i + 1] + 1 : 1;
            res = max(res, R[i]);   // 和上一个for循环计算的是同一个内容 其实可以省略
        }
        for (int i = 1; i <= n; ++ i )
            cnt[s[i] - 'a'] ++ ;
        
        for (int i = 1; i <= n; ++ i )
            if (s[i - 1] == s[i + 1]) {
                if (s[i] != s[i - 1]) {
                    int t = L[i - 1] + R[i + 1];
                    // != cnt 则还有一个不在这两段的字母可以交换过来
                    if (L[i - 1] + R[i + 1] != cnt[s[i - 1] - 'a']) ++ t ;
                    res = max(res, t);
                }
                // 若 s[i - 1] == s[i + 1] == s[i] 当无事发生过 ovo
            } else {
                if (i > 1 && s[i - 1] != s[i] && L[i - 1] != cnt[s[i - 1] - 'a'])
                    res = max(res, L[i - 1] + 1);
                if (i < n && s[i + 1] != s[i] && R[i + 1] != cnt[s[i + 1] - 'a'])
                    res = max(res, R[i + 1] + 1);
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

> [!NOTE] **[LeetCode 1186. 删除一次得到子数组最大和](https://leetcode-cn.com/problems/maximum-subarray-sum-with-one-deletion/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 枚举每个位置
> 
> 1. 单独使用 不删除
> 2. 和左右合并 删 or 不删
> 3. 和左 or 右合并 删 or 不删
> 
> 题解区有dp解法 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int maximumSum(vector<int>& arr) {
        int n = arr.size();
        vector<int> l(n + 2), r(n + 2);
        for (int i = 1; i <= n; ++ i )
            l[i] = max(l[i - 1], 0) + arr[i - 1];
        for (int i = n; i >= 1; -- i )
            r[i] = max(r[i + 1], 0) + arr[i - 1];
        int res = -inf;
        for (int i = 1; i <= n; ++ i ) {
            res = max(res, arr[i - 1]);
            int oth = l[i - 1] + r[i + 1];
            if(i > 1 && i < n) res = max(res, max(oth, oth + arr[i - 1]));
            if (i > 1) res = max(res, max(l[i - 1], l[i - 1] + arr[i - 1]));
            if (i < n) res = max(res, max(r[i + 1], r[i + 1] + arr[i - 1]));
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

> [!NOTE] **[LeetCode 1856. 子数组最小乘积的最大值](https://leetcode-cn.com/problems/maximum-subarray-min-product/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 单调栈 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    int maxSumMinProduct(vector<int>& nums) {
        int n = nums.size();
        vector<LL> s(n + 1);
        for (int i = 1; i <= n; ++ i )
            s[i] = s[i - 1] + nums[i - 1];
        
        // 两侧第一个小于其的位置
        vector<int> l(n + 1, 0), r(n + 1, n + 1);
        stack<int> st;
        for (int i = 1; i <= n; ++ i ) {
            while (st.size() && nums[st.top() - 1] > nums[i - 1]) {
                r[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }
        while (st.size())
            st.pop();
        for (int i = n; i >= 1; -- i ) {
            while (st.size() && nums[st.top() - 1] > nums[i - 1]) {
                l[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }
        
        LL res = 0;
        for (int i = 1; i <= n; ++ i ) {
            int v = nums[i - 1], lid = l[i], rid = r[i];
            LL t = (LL)v * (s[rid - 1] - s[lid]);
            res = max(res, t);
        }
        
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

> [!NOTE] **[LeetCode 2281. 巫师的总力量和](https://leetcode.cn/problems/sum-of-total-strength-of-wizards/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 考虑当前值作为最小值，有哪些区间受影响
> 
> 显然有单调栈求左右边界，**注意本题数值可能重复，则需要去重（一侧严格小于，另一侧小于等于）**
> 
> 随后对区间内的所有数组求和即可
> 
> **问题在于时间复杂度，显然可以【公式转化，使用前缀和的前缀和来 $O(1)$ 查询】**
> 
> **深刻理解 重复做**
> 
> TODO: 整理公式转化过程

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // https://leetcode.cn/problems/sum-of-total-strength-of-wizards/solution/ji-suan-mei-ge-shu-zi-zuo-wei-zui-xiao-z-3jvr/
    using LL = long long;
    const static int N = 1e5 + 10, MOD = 1e9 + 7;
    
    int n;
    int stk[N], top;
    int l[N], r[N];
    LL s[N], ss[N]; // 原数组前缀和，以及该前缀和的前缀和
    
    int totalStrength(vector<int>& a) {
        n = a.size();
        
        // 求右侧【严格小于】当前值的位置
        {
            top = 0;
            for (int i = 1; i <= n; ++ i ) {
                while (top && a[stk[top - 1] - 1] > a[i - 1])
                    r[stk[top - 1]] = i, top -- ;
                stk[top ++ ] = i;
            }
            while (top)
                r[stk[top - 1]] = n + 1, top -- ;
        }
        // 求左侧【小于等于】当前值的位置
        {
            top = 0;
            for (int i = n; i >= 1; -- i ) {
                while (top && a[stk[top - 1] - 1] >= a[i - 1])  // ATTENTION >= 去重 其实改任意一侧都可以
                    l[stk[top - 1]] = i, top -- ;
                stk[top ++ ] = i;
            }
            while (top)
                l[stk[top - 1]] = 0, top -- ;
        }
        
        memset(s, 0, sizeof s), memset(ss, 0, sizeof ss);
        for (int i = 1; i <= n; ++ i )
            s[i] = (s[i - 1] + a[i - 1]) % MOD;
        for (int i = 1; i <= n; ++ i )
            ss[i] = (ss[i - 1] + s[i]) % MOD;
        
        LL res = 0;
        for (int i = 1; i <= n; ++ i ) {
            int lv = l[i], rv = r[i];
            LL t = a[i - 1];
            
            // cout << " i = " << i << " lv = " << lv << " rv = " << rv << endl;
            // [lv+1,i], [i,rv-1]
            // 以i为右边界起始点，则：
            // - 每个右边界都被使用 i-lv 次，共计 ss[rv-1]-ss[i-1],
            // - 每个左边界都被使用 rv-i 次，共计 ss[i-1]-ss[lv-1]               // ATTENTION ss[lv-1]
            LL tot = (LL)(i - lv) * (ss[rv - 1] - ss[i - 1]) % MOD - (rv - i) * (ss[i - 1] - (lv ? ss[lv - 1] : 0)) % MOD;
            // cout << " i = " << i << " tot = " << tot << endl;
            // cout << " ... " << (i - lv) << " " << ss[rv-1]-ss[i-1] << " " << rv-i << " " << ss[i-1]-ss[lv] << endl;
            
            res = (res + (tot + MOD) % MOD * t % MOD) % MOD;
            
            /*
            for (int j = lv + 1; j <= i; ++ j )
                for (int k = i; k <= rv - 1; ++ k )
                    res = (res + (s[k] - s[j - 1] + MOD) % MOD * t % MOD) % MOD;
            */
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

> [!NOTE] **[LeetCode 1960. 两个回文子字符串长度的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常好的 manacher 应用题 重复做
> 
> 以及【前缀和后缀分解】的经典思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;
    
    char b[N];
    int p[N];
    
    void manacher(int n) {
        p[0] = 1;
        int id = 0, mx = 0;
        for (int i = 1; i < n; ++ i ) {
            p[i] = mx > i ? min(p[2 * id - i], mx - i) : 1;
            while (i >= p[i] && b[i - p[i]] == b[i + p[i]])
                p[i] ++ ;
            if (i + p[i] > mx)
                id = i, mx = i + p[i];
        }
    }
    
    long long maxProduct(string s) {
        int n = s.size();
        for (int i = 0; i < n; ++ i )
            b[i] = s[i];
        b[n] = 0;
        
        manacher(n);
        
        vector<int> f(n), g(n);
        // i 指前缀下标
        // j 指当前扫到的中心
        // 非常巧妙的双指针优化
        for (int i = 0, j = 0, mx = 0; i < n; ++ i ) {
            while (j + p[j] - 1 < i) {
                mx = max(mx, p[j]);
                j ++ ;
            }
            mx = max(mx, i - j + 1);
            f[i] = mx;
        }
        for (int i = n - 1, j = n - 1, mx = 0; i >= 0; -- i ) {
            while (j - p[j] + 1 > i) {
                mx = max(mx, p[j]);
                j -- ;
            }
            mx = max(mx, j - i + 1);
            g[i] = mx;
        }
        
        LL res = 0;
        for (int i = 0; i < n - 1; ++ i )
            res = max(res, (LL)(f[i] * 2 - 1) * (g[i + 1] * 2 - 1));
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

> [!NOTE] **[LeetCode 2054. 两个最好的不重叠活动](https://leetcode-cn.com/problems/two-best-non-overlapping-events/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 普适思路：离散化 + 分别统计左右侧
> 
> 也可 二分

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 前后缀分解**

```cpp
class Solution {
public:
    vector<int> xs;
    
    int find(int x) {
        return lower_bound(xs.begin(), xs.end(), x) - xs.begin();
    }
    
    int maxTwoEvents(vector<vector<int>>& events) {
        for (auto & e : events)
            xs.push_back(e[0]), xs.push_back(e[1]);
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());
        
        int n = xs.size(), m = events.size();
        
        vector<int> l(n), r(n);
        {
            sort(events.begin(), events.end(), [](const vector<int> & a, const vector<int> & b) {
                return a[1] < b[1];
            });
            for (int i = 0, p = 0; i < n; ++ i ) {
                int ed = xs[i];
                if (i)
                    l[i] = l[i - 1];    // 继承左侧
                while (p < m && events[p][1] <= ed)
                    l[i] = max(l[i], events[p][2]), p ++ ;
            }
        }
        {
            sort(events.begin(), events.end(), [](const vector<int> & a, const vector<int> & b) {
                return a[0] > b[0];
            });
            for (int i = n - 1, p = 0; i >= 0; -- i ) {
                int st = xs[i];
                if (i < n - 1)
                    r[i] = r[i + 1];    // 继承右侧
                while (p < m && events[p][0] >= st)
                    r[i] = max(r[i], events[p][2]), p ++ ;
            }
        }
        
        int res = 0;
        for (int i = 0; i < n - 1; ++ i )
            res = max(res, l[i] + r[i + 1]);
        res = max(res, l[n - 1]);   // 1个
        return res;
    }
};
```

##### **C++ 二分**

```cpp
class Solution {
public:
    int maxTwoEvents(vector<vector<int>>& q) {
        int n = q.size();
        vector<int> p(n);
        for (int i = 0; i < n; i ++ ) p[i] = i;
        sort(p.begin(), p.end(), [&](int a, int b) {
            return q[a][1] < q[b][1];
        });
        vector<int> f(n);
        f[0] = q[p[0]][2];
        for (int i = 1; i < n; i ++ )
            f[i] = max(f[i - 1], q[p[i]][2]);
        
        int res = 0;
        for (int i = 0; i < n; i ++ ) {
            int l = 0, r = n - 1;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (q[p[mid]][1] < q[i][0]) l = mid;
                else r = mid - 1;
            }
            int s = q[i][2];
            if (q[p[r]][1] < q[i][0]) s += f[r];
            res = max(res, s);
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

> [!NOTE] **[LeetCode 2163. 删除元素后和的最小差值](https://leetcode-cn.com/problems/minimum-difference-in-sums-after-removal-of-elements/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一直在想排序贪心，实际上可以使用前后缀分解的思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    
    long long minimumDifference(vector<int>& nums) {
        int m = nums.size(), n = m / 3;
        vector<LL> l(m + 2, INT_MAX), r(m + 2, INT_MAX);
        
        {
            priority_queue<int> heap;
            LL s = 0;
            for (int i = 1; i <= n; ++ i )
                heap.push(nums[i - 1]), s += nums[i - 1];
            l[n] = s;
            for (int i = n + 1; i <= m; ++ i ) {
                heap.push(nums[i - 1]);
                int t = heap.top(); heap.pop();
                s = s - t + nums[i - 1];
                l[i] = s;
            }
        }
        {
            priority_queue<int, vector<int>, greater<int>> heap;
            LL s = 0;
            for (int i = m; i >= n * 2 + 1; -- i )
                heap.push(nums[i - 1]), s += nums[i - 1];
            r[2 * n + 1] = s;   // ATTENTION
            for (int i = 2 * n; i >= 1; -- i ) {
                heap.push(nums[i - 1]);
                int t = heap.top(); heap.pop();
                s = s - t + nums[i - 1];
                r[i] = s;
            }
        }
        
        LL res = 1e18;
        for (int i = n; i <= 2 * n; ++ i )
            res = min(res, l[i] - r[i + 1]);
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

> [!NOTE] **[LeetCode 2167. 移除所有载有违禁货物车厢所需的最少时间](https://leetcode-cn.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **核心在于推理得到三段不交叉的结论**
> 
> 重点在于理清楚【对于任意一个 1 只会作为前缀被消除或作为后缀或在中间被消除，且**三段不交叉**】
> 
> 明确以上结论 剩下的就很清晰 直接前后缀分解即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minimumTime(string s) {
        int n = s.size();
        vector<int> l(n + 2), r(n + 2);
        
        l[0] = 0;
        for (int i = 1; i <= n; ++ i )
            if (s[i - 1] == '0')
                l[i] = l[i - 1];
            else
                l[i] = min(l[i - 1] + 2, i);
        
        r[n + 1] = 0;
        for (int i = n; i >= 1; -- i )
            if (s[i - 1] == '0')
                r[i] = r[i + 1];
            else
                r[i] = min(r[i + 1] + 2, n - i + 1);
        
        int res = 1e9;
        for (int i = 1; i <= n; ++ i )
            res = min(res, l[i - 1] + r[i]);
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

> [!NOTE] **[Codeforces C. Ladder](http://codeforces.com/problemset/problem/279/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 题意转换 线性扫描即可
> 
> $r[a] >= l[b]$ 的小细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Ladder
// Contest: Codeforces - Codeforces Round #171 (Div. 2)
// URL: http://codeforces.com/problemset/problem/279/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int n, m;
int a[N], l[N], r[N];

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    for (int i = 1; i <= n; ++i) {
        int j = i + 1;
        while (j <= n && a[j] >= a[j - 1])
            ++j;
        for (int k = i; k < j; ++k)
            r[k] = j - 1;
        i = j - 1;
    }

    for (int i = n; i >= 1; --i) {
        int j = i - 1;
        while (j >= 1 && a[j] >= a[j + 1])
            --j;
        for (int k = i; k > j; --k)
            l[k] = j + 1;
        i = j + 1;
    }

    while (m--) {
        int a, b;
        cin >> a >> b;

        // if (r[a] == l[b] || r[a] == r[b] || l[a] == l[b])
        // WA
        // http://codeforces.com/contest/279/submission/110815163
        //
        // 应该是有一个 corner case :
        // 分析应该是 544[3455433]445 这样的 r[a] >= l[b] 的情况
        // 而该情况也可以写为 r[a] - l[b] >= 0
        //
        // 1. AC
        // http://codeforces.com/contest/279/submission/110819576
        // if (r[a] >= l[b] || r[a] == r[b] || l[a] == l[b])
        //
        // 2. AC
        // http://codeforces.com/contest/279/submission/110816646
        if (r[a] - l[b] >= 0)
            cout << "Yes" << endl;
        else
            cout << "No" << endl;
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

> [!NOTE] **[Codeforces DZY Loves Sequences](https://codeforces.com/problemset/problem/446/A)**
> 
> 题意: 
> 
> 最多修改一个数 求最大的连续上升子段长度

> [!TIP] **思路**
> 
> 注意题目要求的是【连续子段】而不是【子序列】，会好做很多
> 
> TODO: 思考【子序列】的进阶解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. DZY Loves Sequences
// Contest: Codeforces - Codeforces Round #FF (Div. 1)
// URL: https://codeforces.com/problemset/problem/446/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10, INF = 0x3f3f3f3f;

int n, a[N];
int l[N], r[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    for (int i = 1; i <= n; ++i)
        if (a[i] > a[i - 1])
            l[i] = l[i - 1] + 1;
        else
            l[i] = 1;
    for (int i = n; i >= 1; --i)
        if (a[i] < a[i + 1])
            r[i] = r[i + 1] + 1;
        else
            r[i] = 1;

    int res = 0;
    a[0] = INF, a[n + 1] = -INF;
    for (int i = 1; i <= n; ++i) {
        int t = 0;
        if (a[i + 1] - a[i - 1] > 1)
            t = l[i - 1] + r[i + 1];
        else
            t = max(l[i - 1], r[i + 1]);
        res = max(res, t + 1);
    }
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

> [!NOTE] **[Codeforces Pair of Numbers](http://codeforces.com/problemset/problem/359/D)** [TAG] 经典进阶
> 
> 题意: 
> 
> 有一个长度为 N 的正整数数列 $a_1 , a_2 , \cdots , a_n$ 
> 
> 现在他想找到这个数列中最长的一个区间，满足区间中有一个数 $x$ 可以整除区间中任意数。

> [!TIP] **思路**
> 
> 简单推理，这个数必然是最小数且是连续数字的 $gcd$
> 
> 有一个较为显然的思路是 $rmq$ 维护区间 $min / gcd$
> 
> 实际上，**可以直接前后缀分解，维护当前数字作为该数的左右延伸距离**
> 
> 实现时，**直接跳到 `l[l[i]-1]` 和 `r[r[i]+1]` 的 trick 技巧**
> 
> 另外，**注意 `l r` 保存下标而非长度**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Pair of Numbers
// Contest: Codeforces - Codeforces Round #209 (Div. 2)
// URL: https://codeforces.com/problemset/problem/359/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 3e5 + 10;

int n, a[N];
int l[N], r[N];  // 注意 l r 保存下标而非长度

int gcd(int a, int b) {
    if (!b)
        return a;
    return gcd(b, a % b);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    for (int i = 1; i <= n; ++i) {
        l[i] = i;  // init
        // ATTENTION
        while (a[l[i] - 1] % a[i] == 0 && l[i] > 1)
            l[i] = l[l[i] - 1];
    }
    for (int i = n; i >= 1; --i) {
        r[i] = i;
        while (a[r[i] + 1] % a[i] == 0 && r[i] < n)
            r[i] = r[r[i] + 1];
    }

    int len = -1;
    vector<int> xs;
    for (int i = 1; i <= n; ++i)
        if (r[i] - l[i] > len) {
            len = r[i] - l[i];
            xs = {l[i]};
        } else if (r[i] - l[i] == len && xs.back() != l[i])
            // ATTENTION 这里需要特判 xs.back() != l[i]
            // (一段相同的数, 其 li ri 都一样)
            xs.push_back(l[i]);
    cout << xs.size() << ' ' << len << endl;
    for (auto x : xs)
        cout << x << ' ';
    cout << endl;

    return 0;
}
```

##### **C++ RMQ**

```cpp
// Problem: D. Pair of Numbers
// Contest: Codeforces - Codeforces Round #209 (Div. 2)
// URL: https://codeforces.com/problemset/problem/359/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const static int N = 3e5 + 10, M = 20;

int n, a[N];
int f1[N][M], f2[N][M];

int gcd(int a, int b) {
    if (!b)
        return a;
    return gcd(b, a % b);
}

void init() {
    memset(f1, 0, sizeof f1);
    memset(f2, 0x3f, sizeof f2);
    for (int j = 0; j < M; ++j)
        for (int i = 1; i + (1 << j) - 1 <= n; ++i)
            if (!j)
                f1[i][j] = f2[i][j] = a[i];
            else {
                f1[i][j] = min(f1[i][j - 1], f1[i + (1 << j - 1)][j - 1]);
                f2[i][j] = gcd(f2[i][j - 1], f2[i + (1 << j - 1)][j - 1]);
            }
}

PII query(int l, int r) {
    int len = r - l + 1;
    int k = log(len) / log(2);
    return {min(f1[l][k], f1[r - (1 << k) + 1][k]),
            gcd(f2[l][k], f2[r - (1 << k) + 1][k])};
}

vector<int> check(int m) {
    vector<int> ret;
    for (int i = m; i <= n; ++i) {
        auto [k, v] = query(i - m + 1, i);
        if (k == v)
            ret.push_back(i - m + 1);
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    init();

    int l = 1, r = n + 1;
    while (l < r) {
        int m = l + r >> 1;
        auto xs = check(m);
        if (xs.size() > 0)
            l = m + 1;
        else
            r = m;
    }
    auto xs = check(l - 1);
    cout << xs.size() << ' ' << l - 2 << endl;
    for (auto x : xs)
        cout << x << ' ';
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

> [!NOTE] **[Codeforces Pashmak and Parmida's problem](http://codeforces.com/problemset/problem/459/D)**
> 
> 题意: 
> 
> 给定公式 求满足左右侧指定关系的总数

> [!TIP] **思路**
> 
> 分析 简单前后缀分解 + BIT

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Pashmak and Parmida's problem
// Contest: Codeforces - Codeforces Round #261 (Div. 2)
// URL: https://codeforces.com/problemset/problem/459/D
// Memory Limit: 256 MB
// Time Limit: 3000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e6 + 10;

int n, a[N];
LL l[N], r[N];

LL tr[N];
void init() { memset(tr, 0, sizeof tr); }
int lowbit(int x) { return x & -x; }
void add(int x, LL c) {
    for (int i = x; i < N; i += lowbit(i))
        tr[i] += c;
}
LL sum(int x) {
    LL ret = 0;
    for (int i = x; i; i -= lowbit(i))
        ret += tr[i];
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    {
        unordered_map<int, LL> hash;
        for (int i = 1; i <= n; ++i) {
            hash[a[i]]++;
            l[i] = hash[a[i]];
        }
    }
    {
        unordered_map<int, int> hash;
        for (int i = n; i >= 1; --i) {
            hash[a[i]]++;
            r[i] = hash[a[i]];
        }
    }

    init();
    // 求 l[i] > r[j] 的个数和
    LL res = 0;
    for (int j = 1; j <= n; ++j) {
        res += sum(N - 1) - sum(r[j]);
        add(l[j], 1);
    }
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

> [!NOTE] **[Codeforces Mike and Feet](http://codeforces.com/problemset/problem/547/B)**
> 
> 题意: 
> 
> $n$ 个值代表 $n$ 个熊的高度
> 
> 对于 $size$ 为 $x$ 的 $group$，$strength$ 值为这个 $group$ 中熊的最小的高度值

> [!TIP] **思路**
> 
> 经典前后缀分解
> 
> - $f$ 标记长度小于等于某个值的其最大 $strength$
> 
> - 最后逆序更新一遍即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Mike and Feet
// Contest: Codeforces - Codeforces Round #305 (Div. 1)
// URL: https://codeforces.com/problemset/problem/547/B
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 2e5 + 10;

int n;
int a[N];
int l[N], r[N], f[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    stack<int> st;
    {
        for (int i = n; i >= 1; --i) {
            while (st.size() && a[i] < a[st.top()]) {
                l[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }
        while (st.size()) {
            l[st.top()] = 0;
            st.pop();
        }
    }
    {
        for (int i = 1; i <= n; ++i) {
            while (st.size() && a[i] < a[st.top()]) {
                r[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }
        while (st.size()) {
            r[st.top()] = n + 1;
            st.pop();
        }
    }

    memset(f, 0, sizeof f);
    for (int i = 1; i <= n; ++i) {
        int d = r[i] - l[i] - 1;
        f[d] = max(f[d], a[i]);
    }
    for (int i = n; i >= 1; --i)
        f[i] = max(f[i], f[i + 1]);

    for (int i = 1; i <= n; ++i)
        cout << f[i] << ' ';
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

> [!NOTE] **[Codeforces Vasya and Robot](http://codeforces.com/problemset/problem/354/A)**
> 
> 题意: 
> 
> 一个序列 $a$ ，每次可以从左边或右边取走一个，从左边取消耗 $l \times a_i$ ，从右边取消耗 $r \times a_i$ 。
> 
> 连续取走左边的额外消耗 $ql$ ，连续取走右边的额外消耗 $qr$ 能量。最小化取走所有物品的价值。

> [!TIP] **思路**
> 
> 最终必然某个位置左侧全是左手取的，右侧全是右手取的。
> 
> 枚举中间断点 $x$ ，使用前缀后缀和维护代价。
> 
> 经典前后缀分解应用 要想的到

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Vasya and Robot
// Contest: Codeforces - Codeforces Round #206 (Div. 1)
// URL: https://codeforces.com/problemset/problem/354/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10;

int n, w[N];
int l, r, ql, qr;

int ls[N], rs[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> l >> r >> ql >> qr;
    for (int i = 1; i <= n; ++i)
        cin >> w[i];

    ls[0] = 0, rs[n + 1] = 0;
    for (int i = 1; i <= n; ++i)
        ls[i] = ls[i - 1] + w[i];
    for (int i = n; i >= 1; --i)
        rs[i] = rs[i + 1] + w[i];

    int res = 1e9;
    // 枚举结束时左手取截止到的位置
    for (int i = 0; i <= n; ++i) {
        int left = i, right = n - i;
        int t = ls[i] * l + rs[i + 1] * r;
        // 取最优时必然是左右间隔拿，如果不能间隔拿则需要连续取的代价
        if (right >= left + 2 || left >= right + 2)
            t += (left > right ? ql : qr) * (abs(right - left) - 1);
        res = min(res, t);
    }
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

> [!NOTE] **[LeetCode 1930. 长度为 3 的不同回文子序列](https://leetcode-cn.com/problems/unique-length-3-palindromic-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 注意 l & r 避免数值溢出
> 
> 另有 trick 的思想和解法: set 可以用数组标记替代

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        int n = s.size();
        vector<vector<int>> sum(n + 1, vector<int>(26));
        
        for (int i = 1; i <= n; ++ i ) {
            int v = s[i - 1] - 'a';
            for (int j = 0; j < 26; ++ j )
                if (j == v)
                    sum[i][j] = sum[i - 1][j] + 1;
                else
                    sum[i][j] = sum[i - 1][j];
        }
        
        unordered_set<string> S;
        for (int i = 2; i < n; ++ i ) {
            int v = s[i - 1] - 'a';
            for (int j = 0; j < 26; ++ j ) {
                int l = sum[i - 1][j], r = sum[n][j] - sum[i][j];
                if (l && r) {   // ATTENTION && 而不是 *
                    string t;
                    t.push_back('a' + j);
                    t.push_back('a' + v);
                    t.push_back('a' + j);
                    S.insert(t);
                }
            }
        }
        return S.size();
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        int n = s.size();
        vector<int> u(26), v(26);
        for (int i = 0; i < n; ++i)
            u[s[i]-'a'] ++;
        vector<vector<int>> f(26, vector<int>(26));
        for (int i = 0; i < n; ++i) {
            u[s[i]-'a'] --;
            for (int c = 0; c < 26; ++c)
                if (u[c] && v[c]) f[s[i]-'a'][c] = 1;
            v[s[i]-'a'] ++;
        }
        int res = 0;
        for (int a = 0; a < 26; ++a)
            for (int b = 0; b < 26; ++b)
                res += f[a][b];
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

> [!NOTE] **[LeetCode 2484. 统计回文子序列数目](https://leetcode.cn/problems/count-palindromic-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> `长度为 3 的不同回文子序列` 标准前后缀分解
> 
> 枚举中间位置，并遍历左右侧的可能值（100 个）即可
>
> 注意直接求解时求的是当前位置结尾的数量，要加上前面的以修订为前缀和
> 
> **加快速度**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 长度为 5 则可能性有 1000 种
    const static int N = 1e4 + 10, M = 100, MOD = 1e9 + 7;
    using LL = long long;
    
    int c[10];
    LL l[N][M], r[N][M];
    
    int countPalindromes(string s) {
        {
            memset(l, 0, sizeof l);
            memset(r, 0, sizeof r);
        }
        
        int n = s.size();
        
        // l
        {
            memset(c, 0, sizeof c);
            for (int i = 0; i < n; ++ i ) {
                int t = s[i] - '0';
                for (int j = 0; j < 10; ++ j )
                    l[i][j * 10 + t] = (l[i][j * 10 + t] + c[j]) % MOD;
                c[t] ++ ;
                
                if (i) {
                    for (int j = 0; j < M; ++ j )
                        l[i][j] = (l[i][j] + l[i - 1][j]) % MOD;
                }
            }
        }
        // r
        {
            memset(c, 0, sizeof c);
            for (int i = n - 1; i >= 0; -- i ) {
                int t = s[i] - '0';
                for (int j = 0; j < 10; ++ j )
                    r[i][j * 10 + t] = (r[i][j * 10 + t] + c[j]) % MOD;
                c[t] ++ ;
                
                if (i < n - 1) {
                    for (int j = 0; j < M; ++ j )
                        r[i][j] = (r[i][j] + r[i + 1][j]) % MOD;
                }
            }
        }
        
        LL res = 0;
        for (int i = 2; i < n - 2; ++ i ) {
            int t = s[i] - '0';
            int last = res;
            for (int j = 0; j < M; ++ j )
                res = (res + l[i - 1][j] * r[i + 1][j] % MOD) % MOD;
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

> [!NOTE] **[LeetCode 2818. 操作使得分最大](https://leetcode.cn/problems/apply-operations-to-maximize-score/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典的前后缀分解 结合素数筛和快速幂

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using LL = long long;
const static int N = 1e5 + 10, MOD = 1e9 + 7;
static bool f = false;

int primes[N], cnt;
bool st[N];

void init() {
    if (f)
        return;
    f = true;
    
    memset(st, 0, sizeof st);
    cnt = 0;
    for (int i = 2; i < N; ++ i ) {
        if (!st[i])
            primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= (N - 1) / i; ++ j ) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0)
                break;
        }
    }
}

class Solution {
public:
    // 某个位置可以被多次选 只要其被包含在不同的子数组中 且是最靠左的位置（单调栈）
    //  则 考虑每一个位置的元素可以作为发挥作用的元素 其左右端点可以延伸到多远 => 对应有多少个区间可用
    //  最后排序 依次取用区间即可
    
    using PIL = pair<int, LL>;
    
    int ps[N];
    int n;
    
    int l[N], r[N];
    
    LL qmi(LL a, LL b) {
        LL ret = 1;
        while (b) {
            if (b & 1)
                ret = ret * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return ret;
    }
    
    int maximumScore(vector<int>& nums, int k) {
        init();
        this->n = nums.size();
        
        for (int i = 0; i < n; ++ i ) {
            int t = 0, x = nums[i];
            for (int j = 0; primes[j] <= x; ++ j )
                if (x % primes[j] == 0) {
                    t ++ ;
                    while (x % primes[j] == 0)
                        x /= primes[j];
                }
                    
            ps[i] = t;
        }
        
        {
            // 向左最多延伸到哪里 默认显然是到 -1
            for (int i = 0; i < n; ++ i )
                l[i] = -1;
            
            stack<int> stk;
            for (int i = n - 1; i >= 0; -- i ) {
                int x = ps[i];
                while (stk.size() && ps[stk.top()] <= x) {
                    l[stk.top()] = i;
                    stk.pop();
                }
                stk.push(i);
            }
        }
        {
            // 向右
            for (int i = 0; i < n; ++ i )
                r[i] = n;
            
            stack<int> stk;
            for (int i = 0; i < n; ++ i ) {
                int x = ps[i];
                while (stk.size() && ps[stk.top()] < x) {   // 必须要大于 才能发挥作用
                    r[stk.top()] = i;
                    stk.pop();
                }
                stk.push(i);
            }
        }
        
        priority_queue<PIL> q;
        for (int i = 0; i < n; ++ i ) {
            LL x = i - l[i], y = r[i] - i;
            q.push({nums[i], x * y});
        }
        
        LL res = 1, tot = k;
        for (; tot && q.size();) {
            auto [k, v] = q.top(); q.pop();
            LL cost = min(v, tot);
            tot -= cost;
            
            res = (res * qmi(k, cost)) % MOD;
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

> [!NOTE] **[LeetCode 2866. 美丽塔 II](https://leetcode.cn/problems/beautiful-towers-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典前后缀分解 推导思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 考虑枚举中心位置
    // 则 对于一个具体的中心位置 i 其高度必然为 maxHeight[i]
    //   左右延展时必须小于等于当前值 且小于等于历史最小值 => 显然不能暴力计算
    // 考虑前后缀分解 => 这样答案很好求
    // 问题在于前后缀求解什么
    //      => l: 类似 LIS 的总和, 当高度增加时很好处理 高度减少时需要回退 => 单调栈
    
    using LL = long long;
    using PII = pair<int, int>;
    const static int N = 1e5 + 10;
    
    LL l[N], r[N];
    
    long long maximumSumOfHeights(vector<int>& maxHeights) {
        int n = maxHeights.size();
        memset(l, 0, sizeof l), memset(r, 0, sizeof r);
        {
            stack<PII> st;  // [height, cnt]
            LL sum = 0;
            for (int i = 1; i <= n; ++ i ) {
                int x = maxHeights[i - 1];
                sum += x;
                
                PII t = {x, 1};
                while (st.size()) {
                    auto [height, cnt] = st.top();
                    if (height < t.first)
                        break;
                    st.pop();
                    
                    int diff = height - x;
                    sum -= (LL)diff * cnt;
                    t.second += cnt;
                }
                st.push(t);
                l[i] = sum;
            }
        }
        {
            stack<PII> st;
            LL sum = 0;
            for (int i = n; i >= 1; -- i ) {
                int x = maxHeights[i - 1];
                sum += x;
                
                PII t = {x, 1};
                while (st.size()) {
                    auto [height, cnt] = st.top();
                    if (height < t.first)
                        break;
                    st.pop();
                    
                    int diff = height - x;
                    sum -= (LL)diff * cnt;
                    t.second += cnt;
                }
                st.push(t);
                r[i] = sum;
            }
        }
        LL res = 0;
        for (int i = 1; i <= n; ++ i )
            res = max(res, l[i] + r[i + 1]);
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

> [!NOTE] **[LeetCode 2906. 构造乘积矩阵](https://leetcode.cn/problems/construct-product-matrix/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 前后缀分解 重点在想到

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 初看认为是求逆元
    // 实际上 只需要前后缀分解即可...
    using LL = long long;
    const static int N = 1e5 + 10, MOD = 12345;

    LL l[N], r[N];
    
    vector<vector<int>> constructProductMatrix(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        
        memset(l, 0, sizeof l), memset(r, 0, sizeof r);
        
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j ) {
                int x = grid[i][j];
                int idx = i * m + j;
                l[idx] = (idx ? l[idx - 1] : 1) * x % MOD;
            }
        for (int i = n - 1; i >= 0; -- i )
            for (int j = m - 1; j >= 0; -- j ) {
                int x = grid[i][j];
                int idx = i * m + j;
                r[idx] = (idx < n * m - 1 ? r[idx + 1] : 1) * x % MOD;
            }
        
        vector<vector<int>> res(n, vector<int>(m));
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j ) {
                LL idx = i * m + j;
                LL t = (idx ? l[idx - 1] : 1) * (idx < n * m - 1 ? r[idx + 1] : 1) % MOD;
                res[i][j] = t;
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

> [!NOTE] **[LeetCode 3003. 执行操作后的最大分割数量](https://leetcode.cn/problems/maximize-the-number-of-partitions-after-operations/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推导 转换思路 结合前后缀分解降低复杂度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 首先 梳理明确:
    //      假设不产生修改 则 [无论从左/从右侧开始贪心分段 段的总数都是相同的] 不同的是区间具体的左右边界
    //
    // 考虑前后缀分解 枚举在某个位置修改 以及改成不同字符对整体带来的影响

    using PII = pair<int, int>; // seg-index, mask
    const static int N = 1e4 + 10;

    int K;
    PII l[N], r[N];

    void update(int bit, int & idx, int & size, int & mask) {
        if (mask & bit)
            return;
        
        if (size + 1 > K) {
            // 需要从当前位置作为起始 新增一段
            idx ++ ;
            mask = bit;
            size = 1;
        } else {
            mask |= bit;
            size ++ ;
        }
    }

    int maxPartitionsAfterOperations(string s, int k) {
        // ATTENTION 必须特判
        if (k == 26)
            return 1;

        this->K = k;
        int n = s.size();
        for (int i = 1, idx = 1, size = 0, mask = 0; i <= n; ++ i ) {
            int bit = 1 << (s[i - 1] - 'a');
            update(bit, idx, size, mask);
            l[i] = {idx, mask};
        }
        for (int i = n, idx = 1, size = 0, mask = 0; i >= 1; -- i ) {
            int bit = 1 << (s[i - 1] - 'a');
            update(bit, idx, size, mask);
            r[i] = {idx, mask};
        }

        int res = l[n].first;   // 默认不修改的情况下

        for (int i = 1; i <= n; ++ i ) {
            auto [l_idx, l_mask] = l[i - 1];
            auto [r_idx, r_mask] = r[i + 1];
            int union_size = __builtin_popcount(l_mask | r_mask);
            int tot = l_idx + r_idx;    // 假设从当前位置割裂 总共有多少段
            if (union_size < k) {
                // 无论 i 位置上取什么值 左右都会合并
                tot -- ;
            } else if (union_size < 26) {
                // 看当前位置是否 “能够独立”
                bool f1 = (l_idx == 0 || __builtin_popcount(l_mask) == k);
                bool f2 = (r_idx == 0 || __builtin_popcount(r_mask) == k);
                if (f1 && f2)
                    tot ++ ;
            }
            // 否则 i 位置上只能归并到 左/右 中的一个 => tot 不变

            res = max(res, tot);
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