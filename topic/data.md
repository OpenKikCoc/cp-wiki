## 习题

> [!NOTE] **[LeetCode 1198. 找出所有行中最小公共元素](https://leetcode-cn.com/problems/find-smallest-common-element-in-all-rows/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 并非使用堆。
> 
> 正解：统计每行出现的元素，若某元素数量达 n 则为答案。
> 
> 【关键在于题中所给的数据范围: 所有数值小于 10000 】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int smallestCommonElement(vector<vector<int>>& mat) {
        int n = mat.size(), m = mat[0].size();
        vector<int> cnt(10010);
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (j == 0 || mat[i][j] != mat[i][j - 1])
                    ++ cnt[mat[i][j]];
        for (int i = 1; i <= 10000; ++ i)
            if (cnt[i] == n)
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

> [!NOTE] **[LeetCode 1326. 灌溉花园的最少水龙头数目](https://leetcode-cn.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本题限制 `ranges[i] <= 100`
> 
> -   直观做法
>
>     超时 34/35个测试用例 `[输入10000]`
>
>     这种显然在每个区间都很短的情况下很慢 比如超时的用例 [0,1,....1,1,1,0,...]
>
> -   trick 做法
>
> -   dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ TLE**

```cpp
class Solution {
public:
    // 贪心 优先右边界大 右边界相同左边界越小越好
    // （这个位置本身范围为0的水龙头不可用
    int minTaps(int n, vector<int>& ranges) {
        vector<pair<int, int>> v;  // 需要排序 直接用pair<r, 0-l>
        for (int i = 0; i <= n; ++i) {
            if (!ranges[i]) continue;  // 跳过0
            v.push_back({min(ranges[i] + i, n), ranges[i] - i});
        }
        sort(v.begin(), v.end());
        // 随后从后向前
        int p = v.size() - 1, mx = n, res = 0;
        while (p >= 0) {
            ++res;
            int nmx = mx;
            // 从 r >= mx 的选出最靠左的一个来用
            while (p >= 0 && v[p].first >= mx) {
                nmx = min(nmx, -v[p].second);
                --p;
            }
            if (p < 0) {
                if (nmx <= 0)
                    return res;
                else
                    return -1;
            }
            mx = nmx;
            if (mx <= 0) return res;
        }
        return -1;
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    // 考虑：每一个range都很大的情况【当然这道题限定了ranges<=100】
    // 复杂度不会高于 n*100 = 1e6
    int minTaps(int n, vector<int>& ranges) {
        vector<int> v(n + 1);
        for (int i = 0; i <= n; ++i) {
            if (!ranges[i]) continue;  // 跳过0
            int l = max(0, i - ranges[i]), r = min(n, i + ranges[i]);
            while (l < r && v[l] < r) v[l++] = r;
        }
        int res = 0;
        int cur = 0;
        while (cur < n) {
            if (!v[cur]) return -1;
            cur = v[cur];
            ++res;
        }
        return res;
    }
};
```

##### **C++ dp**

```cpp
class Solution {
public:
    int minTaps(int n, vector<int>& ranges) {
        ++n;
        const int INF = 1e9;
        vector<int> f(n + 1, INF);
        f[0] = 0;
        vector<vector<int>> v(n);
        for (int i = 0; i < n; ++i) {
            if (!ranges[i]) continue;
            int l = max(0, i - ranges[i]), r = min(n - 1, i + ranges[i]);
            v[r - 1].push_back(l);
        }
        for (int i = 0; i < n - 1; ++i)
            for (auto j : v[i])
                for (int k = j; k <= i; ++k) f[i + 1] = min(f[i + 1], f[k] + 1);
        return f[n - 1] >= INF ? -1 : f[n - 1];
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

> [!NOTE] **[LeetCode 1766. 互质树](https://leetcode-cn.com/problems/tree-of-coprimes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> > 如果用蛮力检查一个节点的所有的祖先节点 那么一个节点的祖先节点最多能有 n−1 个 显然会超时
> 
> 原先首先 dfs 找序并反向建图 随后依据反向图向上搜索
> 
> 这样显然会在祖先极多的时候爆掉
> 
> 需要充分利用值域 ( 50 ) 的条件
> 
> 以及注意 **双向边建图源点遍历时加 fa 即可 无需反向建图**
> 
> TODO 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> g, st;
    vector<int> nums, res, depth;
    
    int gcd(int a, int b) {
        return b ? gcd(b, a % b) : a;
    }
    
    void dfs(int u, int fa) {
        int v = nums[u];
        for (int i = 1; i <= 50; ++ i )
            if (st[i].size() && gcd(v, i) == 1)
                if (res[u] == -1 || depth[res[u]] < depth[st[i].back()])
                    res[u] = st[i].back();
        
        st[v].push_back(u);
        for (auto v : g[u]) {
            if (v == fa) continue;
            depth[v] = depth[u] + 1;
            dfs(v, u);
        }
            
        st[v].pop_back();
    }
    
    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        this->nums = nums;
        int n = nums.size();
        g.resize(n), st.resize(55), res.resize(n, -1), depth.resize(n, 0);
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            g[a].push_back(b), g[b].push_back(a);
        }
        
        dfs(0, -1);
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

> [!NOTE] **[LeetCode 1906. 查询差绝对值的最小值](https://leetcode-cn.com/problems/minimum-absolute-difference-queries/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 根据数据集范围推测直接使用前缀和统计查询区间中出现的所有数字，去重遍历即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // ATTENTION num[i] <= 100
    const static int N = 1e5 + 10, M = 110;
    int cnt[N][M], t[M], n;
    
    vector<int> minDifference(vector<int>& nums, vector<vector<int>>& queries) {
        n = nums.size();
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= 100; ++ j ) {
                cnt[i][j] = cnt[i - 1][j];
                if (nums[i - 1] == j)
                    cnt[i][j] ++ ;
            }
        
        vector<int> res;
        for (auto & q : queries) {
            int l = q[0], r = q[1];
            vector<int> t;
            for (int i = 1; i <= 100; ++ i )
                if (cnt[r + 1][i] > cnt[l][i])
                    t.push_back(i);
            sort(t.begin(), t.end());
            int sz = t.size(), ret = 2e9;
            for (int i = 1; i < sz; ++ i )
                ret = min(ret, t[i] - t[i - 1]);
            if (ret == 2e9)
                ret = -1;
            res.push_back(ret);
        }
        return res;
    }
};
```

##### **Python**

```python
# python
# 前缀和
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1981. 最小化目标值与所选元素的差](https://leetcode-cn.com/problems/minimize-the-difference-between-target-and-chosen-elements/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 较显然的是一个分组背包问题
> 
> 直接分组写显然 MLE ，可以滚动数组来做
> 
> 鉴于题目所述数据范围，直接标记某值是否可达即可（背包特性较弱）
> 
> 可 bitset 优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 5000; // 70*70
    
    int n, m;
    vector<vector<int>> g;
    vector<bool> st;
    
    int minimizeTheDifference(vector<vector<int>>& mat, int target) {
        this->g = mat;
        this->n = g.size(), this->m = g[0].size();
        this->st = vector<bool>(N);
        
        st[0] = true;
        for (int i = 0; i < n; ++ i ) {
            auto t = vector<bool>(N);
            for (int j = 0; j < m; ++ j ) {
                int v = g[i][j];
                for (int k = 0; k < N; ++ k )
                    if (st[k] && k + v < N)
                        t[k + v] = true;
            }
            st = t;
        }
        
        // 这一坨显然可以直接一个for-loop... 略
        int res = INT_MAX;
        {
            int p = target;
            while (p && !st[p])
                p -- ;
            if (p)
                res = min(res, target - p);
        }
        {
            int p = target;
            while (p < N && !st[p])
                p ++ ;
            if (p < N)
                res = min(res, p - target);
        }
        return res;
    }
};
```

##### **C++ bitset**

```cpp
// bitset优化
class Solution {
public:
    const static int M = 5000;

    int minimizeTheDifference(vector<vector<int>>& mat, int target) {
        int n = mat.size(), m = mat[0].size();
        bitset<M> b;
        b.set(0);
        for (int i = 0; i < n; ++ i ) {
            bitset<M> t;
            for (auto x : mat[i])
                t |= b << x;
            b = t;
        }
        int res = M;
        for (int i = 0; i < M; ++ i )
            if (b[i])
                res = min(res, abs(i - target));
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