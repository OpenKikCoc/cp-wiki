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
