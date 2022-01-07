## 习题

> [!NOTE] **[LeetCode 41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思路：一个萝卜一个坑。
> 
> 把每个数放到该在的位置，注意：这里是从1开始的，所以还有一个下标映射的关系。
> 
> 外层循环遍历整个数组，内层用while判断当前数字是否在该在的位置，如果不在的话，就一直交换。跳出while循环时，如果当前数字不等于下标，只能说明它重复了。
> 
> 最后再从头到位遍历一遍数组，第一个与下标不对应的数字 就是 缺失的第一个正数。
> 
> O(N) + S(1)


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; ++ i )
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i])
                swap(nums[nums[i] - 1], nums[i]);
        for (int i = 0; i < n; ++ i )
            if(nums[i] != i + 1) return i + 1;
        return n + 1;
    }
};
```

##### **Python**

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 0 <= nums[i]-1 < n and nums[nums[i]-1] != nums[i]:
                # tmp = nums[i] - 1
                # nums[i], nums[tmp] = nums[tmp], nums[i]
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
            # 不能在跳出while循环就直接判断，这个和“寻找重复数字”不一样的点在于，当前值不在该在的位置的时候，确实可以判断这就是个重复数字。但是不能判断它为 缺失的第一个正数，因为可能有比当前数更小的数还在后面没有被处理。
            # if nums[i] != i+1:
            #     return i + 1
        for i in range(n):
            if nums[i] != i+1:
                return i + 1
        return n + 1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 442. 数组中重复的数据](https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/)**
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
// 标准写法
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; ++ i )
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i])
                swap(nums[nums[i] - 1], nums[i]);
        vector<int> res;
        for (int i = 0; i < n; ++ i )
            if (nums[i] != i + 1)
                res.push_back(nums[i]);
        return res;
    }
};
```

##### **C++ 不带修改**

```cpp
class Solution {
public:
    int duplicateInArray(vector<int>& nums) {
        int l = 1, r = nums.size() - 1;
        while (l < r) {
            int m = l + r >> 1;
            
            int s = 0;
            for (auto v : nums)
                if (v >= l && v <= m)
                    ++ s ;
            
            if (s > m - l + 1)
                r = m;
            else
                l = m + 1;
        }
        return l;
    }
};
```

##### **C++ yxc trick**

```cpp
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> res;
        for (auto x: nums) {
            int p = abs(x) - 1;
            nums[p] *= -1;
            if (nums[p] > 0) res.push_back(abs(x));
        }
        return res;
    }
};
```

##### **Python**

```python
# 一个萝卜一个坑。
# 最后遍历整理完的数组的时候，当前数 不等于 下标的，那就是重复的数字
# O(N) + S(1)

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = []
        for i in range(n):
            while nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        for i in range(n):
            if nums[i] != i+1:
                res.append(nums[i])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)**
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
// 标准写法
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; ++ i )
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i])
                swap(nums[nums[i] - 1], nums[i]);
        vector<int> res;
        for (int i = 0; i < n; ++ i )
            if (nums[i] != i + 1)
                res.push_back(i + 1);
        return res;
    }
};
```

##### **C++ yxc trick**

```cpp
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        vector<int> res;
        for (auto x : nums) {
            x = abs(x);
            if (nums[x - 1] > 0) nums[x - 1] *= -1;
        }
        for (int i = 0; i < n; ++ i )
            if (nums[i] > 0) res.push_back(i + 1);
        return res;
    }

    vector<int> findDisappearedNumbers_2(vector<int>& nums) {
        int n = nums.size();
        vector<int> res;
        for (int i = 0; i < n; ++ i )
            while (nums[nums[i] - 1] != nums[i]) swap(nums[nums[i] - 1], nums[i]); 
        for (int i = 0; i < n; ++ i ) if (nums[i] != i + 1) res.push_back(i + 1);
        return res;
    }
};
```

##### **Python**

```python
# 一个萝卜一个坑
# 消失的数字是 整理完数组后，不相等的数字的【下标+1】

class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = []
        for i in range(n):
            while nums[nums[i]-1] != nums[i]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        for i in range(n):
            if nums[i] != i+1:
                res.append(i+1)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 快慢指针也可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        for (int i = 0; i < nums.size(); ++ i ) {
            while (nums[i] != i + 1) {
                if (nums[nums[i] - 1] == nums[i])
                    return nums[i];
                swap(nums[i], nums[nums[i] - 1]);
            }
        }

        return -1;
    }

    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        int p = 0;
        while (p < n) {
            while (nums[p] != p && nums[nums[p]] != nums[p])
                swap(nums[p], nums[nums[p]]);
            if (nums[p] == nums[nums[p]])
                return nums[p];
            ++p;
        }
        return 0;
    }
};
```

##### **C++ 快慢指针**

要求不修改原数组（快慢指针）：

```cpp
// yxc
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int a = 0, b = 0;
        while (true) {
            a = nums[a];
            b = nums[nums[b]];
            if (a == b) {
                a = 0;
                while (a != b) {
                    a = nums[a];
                    b = nums[b];
                }
                return a;
            }
        }

        return -1;
    }
};
```


##### **Python**

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        l,r = 1, len(nums)-1
        while l <r :
            m = l + (r - l) // 2
            cnt = 0
            for num in nums:
                if num <= m:
                    cnt += 1
            if cnt <= m:
                l = m + 1
            else:r = m
        return l
```

<!-- tabs:end -->
</details>

<br>

* * *