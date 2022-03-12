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

> [!NOTE] **[LeetCode [2167. 移除所有载有违禁货物车厢所需的最少时间](https://leetcode-cn.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/)** [TAG]
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