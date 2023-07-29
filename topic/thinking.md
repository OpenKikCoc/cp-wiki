# 习题

> [!NOTE] **[LeetCode 2790. 长度递增组的最大数目](https://leetcode.cn/problems/maximum-number-of-groups-with-increasing-length/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常经典的模型
> 
> 贪心排序 构造思维

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 考虑 按照数量降序
    // 第i次需要把前i大的各自-1 减完之后有可能发生顺序变化
    // 显然不能用堆枚举 复杂度无法接受
    using LL = long long;

    vector<int> uls;

    int maxIncreasingGroups(vector<int>& usageLimits) {
        this->uls = usageLimits;
        sort(uls.begin(), uls.end());
        
        LL rest = 0, res = 0;
        for (auto x : uls) {
            // 直接加  因为不管是“构造不了”还是“构造完有余”都可以给后面用
            // ATTENTION 思维与证明 为什么这样一定可以构造
            // 以柱状图的形式思考
            // 每次新加一个数构造相当于基于原本的再追加一列【一定能够通过置换使得其可用】
            rest += x;
            if (rest >= res + 1) {
                rest -= res + 1;
                res ++ ;
            }
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