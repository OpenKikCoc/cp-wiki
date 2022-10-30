# 离线大类




### 非典型数据结构的简单离线

> [!NOTE] **[LeetCode 2343. 裁剪数字后查询第 K 小的数字](https://leetcode.cn/problems/query-kth-smallest-trimmed-number/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单离线操作即可
> 
> 另有基数排序做法 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PSI = pair<string, int>;      // string, i
    using TIII = tuple<int, int, int>;  // trim, k, i
    
    vector<int> smallestTrimmedNumbers(vector<string>& nums, vector<vector<int>>& queries) {
        vector<TIII> qs;
        for (int i = 0; i < queries.size(); ++ i )
            qs.push_back({queries[i][1], queries[i][0], i});
        sort(qs.begin(), qs.end());
        reverse(qs.begin(), qs.end());
        
        int n = nums.size(), m = nums[0].size();
        vector<PSI> ns;
        for (int i = 0; i < n; ++ i )
            ns.push_back({nums[i], i});
        sort(ns.begin(), ns.end());
        
        vector<int> res(queries.size());
        for (auto [trim, k, i] : qs) {
            if (ns[0].first.size() > trim) {
                int idx = ns[0].first.size() - trim;
                for (int j = 0; j < n; ++ j ) {
                    string s = ns[j].first;
                    ns[j].first = s.substr(idx);
                }
                sort(ns.begin(), ns.end());
            }
            res[i] = ns[k - 1].second;
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

> [!NOTE] **[LeetCode 6227. 下一个更大元素 IV](https://leetcode.cn/problems/next-greater-element-iv/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 显然离线思想 => 按大小降序和坐标顺序排序 依次加入并查询
> 
> 可以不用 BIT 而是直接使用 set 进一步优化时间复杂度 => **相同放一起 经典离线处理思路**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ BIT**

```cpp
class Solution {
public:
    // 从大到小 从前往后 查询并加入
    using PII = pair<int, int>;
    const static int N = 1e5 + 10;
    
    int tr[N];
    int lowbit(int x) {
        return x & -x;
    }
    void add(int x, int y) {
        for (int i = x; i < N; i += lowbit(i))
            tr[i] += y;
    }
    int query(int x) {
        int ret = 0;
        for (int i = x; i; i -= lowbit(i))
            ret += tr[i];
        return ret;
    }
    
    bool check(int m, int p) {
        return query(m) - query(p) < 2;
    }
    
    vector<int> secondGreaterElement(vector<int>& nums) {
        int n = nums.size();
        
        vector<PII> xs;
        for (int i = 1; i <= n; ++ i )
            xs.push_back({nums[i - 1], -i});
        sort(xs.begin(), xs.end());
        reverse(xs.begin(), xs.end());
        
        memset(tr, 0, sizeof tr);
        vector<int> res(n, -1);
        for (int i = 1; i <= n; ++ i ) {
            auto [x, p] = xs[i - 1];
            p = -p;
            
            int l = p, r = n + 1;
            while (l < r) {
                int m = l + r >> 1;
                if (check(m, p))
                    l = m + 1;
                else
                    r = m;
            }
            if (l != n + 1)
                res[p - 1] = nums[l - 1];	// ATTENTION p-1 instead of i-1
            
            if (query(p) - query(p - 1) == 0)
                add(p, 1);
        }
        
        return res;
    }
};
```

##### **C++ set**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    
    vector<int> secondGreaterElement(vector<int>& nums) {
        int n = nums.size();
        vector<PII> xs;
        for (int i = 0; i < n; ++ i )
            xs.push_back({nums[i], i});
        sort(xs.begin(), xs.end()); // 不关心第二维 因为后面会 while 统一处理
        reverse(xs.begin(), xs.end());
        
        vector<int> res(n, -1);
        
        set<int> S;
        S.insert(n + 10), S.insert(n + 11); // 哨兵
        
        for (int i = 0; i < n; ++ i ) {
            auto [x, p] = xs[i];
            int j = i + 1;
            while (j < n && xs[j].first == x)
                j ++ ;
            
            // 这一堆相同的数 放在一起统一处理
            for (int k = i; k < j; ++ k ) {
                auto it = S.lower_bound(xs[k].second);  // ATTENTION 加入的是坐标
                it ++ ;
                if (*it < n)
                    res[xs[k].second] = nums[*it];
            }
            for (int k = i; k < j; ++ k )
                S.insert(xs[k].second);                 // ATTENTION 加入的是坐标
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