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

> [!NOTE] **[LeetCode [2163. 删除元素后和的最小差值](https://leetcode-cn.com/problems/minimum-difference-in-sums-after-removal-of-elements/)** [TAG]
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
