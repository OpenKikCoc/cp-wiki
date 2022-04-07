## 拓展摩尔投票

> [!NOTE] **ATTENTION**
> 
> 如果要选出 $N$ 个候选人，并且要求每个人的得票都超过总票数的 $\frac{1}{N+1}$ 。
> 
> 可以把票分为 2 个部分：
> 
> - 投给了最多 $N$ 个候选人的一部分
> 
> - 被抵消的一部分
> 
> 后者可以划分为若干个 $N+1$ 元组，每个元组内的票都来自不同的候选人。
> 
> 因此，只有那些属于第一部分的人的票数可能超过总数的 $\frac{1}{N+1}$ 。

## 区间摩尔众数 (线段树上摩尔投票)

> [!NOTE] **ATTENTION**
> 
> 摩尔投票不符合结合律，但只关心 `是谁` 而不关心 `具体多少`
> 
> 故 **当绝对众数存在时，可以任意交换摩尔投票的计票顺序，而不改变选出的候选人**
> 
> 在合并左右两个区间时，相当于是先对左右两边分别进行摩尔投票，再对两边抵消后的结果进行摩尔投票。

## 习题

> [!NOTE] **[LeetCode 169. 多数元素](https://leetcode-cn.com/problems/majority-element/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int r, c = 0;
        for (auto x: nums)
            if (!c) r = x, c = 1;
            else if (r == x) c ++ ;
            else c -- ;
        return r;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int n = nums.size();
        int res = nums[0], vote = 1;
        for (int i = 1; i < n; ++ i ) {
            if (nums[i] != res) {
                -- vote ;
                if (!vote) {
                    vote = 1;
                    res = nums[i];
                }
            } else
                ++ vote ;
        }
        return res;
    }
};
```

##### **Python**

```python
# 投票计数法
class Solution:
    def twoSum(self, arr: List[int], target: int) -> List[int]:
        n = len(arr)
        sumn = 0
        l, r = 0, n - 1
        while l < r:
            sumn = arr[l] + arr[r]
            if sumn > target:
                r -= 1
            elif sumn < target:
                l += 1
            else:return [l + 1, r + 1]
        return [-1, -1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 229. 求众数 II](https://leetcode-cn.com/problems/majority-element-ii/)**
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
    vector<int> majorityElement(vector<int>& nums) {
        int r1, r2, c1 = 0, c2 = 0;
        for (auto x: nums)
            if (c1 && x == r1) c1 ++ ;
            else if (c2 && x == r2) c2 ++ ;
            else if (!c1) r1 = x, c1 ++ ;
            else if (!c2) r2 = x, c2 ++ ;
            else c1 --, c2 -- ;
        c1 = 0, c2 = 0;
        for (auto x: nums)
            if (x == r1) c1 ++ ;
            else if (x == r2) c2 ++ ;

        vector<int> res;
        int n = nums.size();
        if (c1 > n / 3) res.push_back(r1);
        if (c2 > n / 3) res.push_back(r2);
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

> [!NOTE] **[LeetCode 1287. 有序数组中出现次数超过25%的元素](https://leetcode-cn.com/problems/element-appearing-more-than-25-in-sorted-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 超过25%的数唯一 说明这个数就是出现次数最多的数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findSpecialInteger(vector<int>& arr) {
        int n = arr.size();
        int res = arr[0], cnt = 1;
        vector<pair<int, int>> ve;
        for (int i = 1; i < n; ++i) {
            if (arr[i] == res)
                ++cnt;
            else {
                ve.push_back({cnt, res});
                res = arr[i];
                cnt = 1;
            }
        }
        ve.push_back({cnt, res});
        sort(ve.begin(), ve.end());
        return ve.back().second;
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