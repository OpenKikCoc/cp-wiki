二分查找（英语：binary search），也称折半搜索（英语：half-interval search）、对数搜索（英语：logarithmic search），是用来在一个有序数组中查找某一元素的算法。

## 关于二分 STL

C++ 标准库中实现了查找首个不小于给定值的元素的函数 [`std::lower_bound`](https://zh.cppreference.com/w/cpp/algorithm/lower_bound) 和查找首个大于给定值的元素的函数 [`std::upper_bound`](https://zh.cppreference.com/w/cpp/algorithm/upper_bound)，二者均定义于头文件 `<algorithm>` 中。

二者均采用二分实现，所以调用前必须保证元素有序。

> [!WARNING] 再次提醒: **`lower_bound` 和 `upper_bound` 的时间复杂度**
> 
> 在一般的数组里，这两个函数的时间复杂度均为 $O(\log n)$，但在 `set` 等关联式容器中，直接调用 `lower_bound(s.begin(),s.end(),val)` 的时间复杂度是 $O(n)$ 的。
> 
> `set` 等关联式容器中已经封装了 `lower_bound` 等函数（像 `s.lower_bound(val)` 这样），这样调用的时间复杂度是 $O(\log n)$ 的。

## 二分应用

### 一般二分

<details>
<summary>详细代码</summary>
<!-- tabs:start -->
###### **C++**

```cpp
int lower_bound(vector<int>& nums, int l, int r, int target) {
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] < target)
            l = mid + 1;
        else
            r = mid;
    }
    return l;
}
int uper_bound(vector<int>& nums, int l, int r, int target) {
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] <= target)
            l = mid + 1;
        else
            r = mid;
    }
    return l - 1;
}

// 如果需在查找不到的时候返回 -1 则需要加两行check
int lower_bound() {
    // ...
    // 最后要检查 l 越界的情况
    if (l >= nums.size() || nums[l] != target) return -1;
    return l;
}

int uper_bound() {
    // ...
    // 最后要检查 r 越界的情况 这里 l == r
    if (l <= 0 || nums[l - 1] != target) return -1;  // if(l == 0 ... )
    return l - 1;
}
```

###### **Python**

```python
def lower_bound(array, first, last, value): # 返回[first, last)内第一个不小于value的值的位置
    while first < last: # 搜索区间[first, last)不为空
        mid = first + (last - first) // 2  # 防溢出
        if array[mid] < value: first = mid + 1
        else: last = mid
    return first  # last也行，因为[first, last)为空的时候它们重合
```
<!-- tabs:end -->
</details>

<br>


> [!NOTE]
> 
> 对于 $n$ 是有符号数且 $n\ge 0$ 时，`n >> 1` 比 `n / 2` 指令数更少。

### 最大值最小化

要求满足某种条件的最大值的最小可能情况（最大值最小化），首先的想法是从小到大枚举这个作为答案的「最大值」，然后去判断是否合法。

若答案单调，就可以使用二分搜索法来更快地找到答案。因此，要想使用二分搜索法来解这种「最大值最小化」的题目，需要满足以下三个条件：

1. 答案在一个固定区间内；
2. 可能查找一个符合条件的值不是很容易，但是要求能比较容易地判断某个值是否是符合条件的；
3. 可行解对于区间满足一定的单调性。换言之，如果 $x$ 是符合条件的，那么有 $x + 1$ 或者 $x - 1$ 也符合条件。（这样下来就满足了上面提到的单调性）

最小值最大化同理。

### 二分答案

解题的时候往往会考虑枚举答案然后检验枚举的值是否正确。若满足单调性，则满足使用二分法的条件。把这里的枚举换成二分，就变成了“二分答案”。

> [!NOTE] **[Luogu P1873 砍树](https://www.luogu.com.cn/problem/P1873)**
> 
> 伐木工人米尔科需要砍倒 $M$ 米长的木材。这是一个对米尔科来说很容易的工作，因为他有一个漂亮的新伐木机，可以像野火一样砍倒森林。不过，米尔科只被允许砍倒单行树木。
> 
> 米尔科的伐木机工作过程如下：米尔科设置一个高度参数 $H$（米），伐木机升起一个巨大的锯片到高度 $H$，并锯掉所有的树比 $H$ 高的部分（当然，树木不高于 $H$ 米的部分保持不变）。米尔科就行到树木被锯下的部分。
> 
> 例如，如果一行树的高度分别为 $20,~15,~10,~17$，米尔科把锯片升到 $15$ 米的高度，切割后树木剩下的高度将是 $15,~15,~10,~15$，而米尔科将从第 $1$ 棵树得到 $5$ 米木材，从第 $4$ 棵树得到 $2$ 米木材，共 $7$ 米木材。
> 
> 米尔科非常关注生态保护，所以他不会砍掉过多的木材。这正是他尽可能高地设定伐木机锯片的原因。你的任务是帮助米尔科找到伐木机锯片的最大的整数高度 $H$，使得他能得到木材至少为 $M$ 米。即，如果再升高 $1$ 米锯片，则他将得不到 $M$ 米木材。

> [!TIP] **解题思路**
> 我们可以在 $1$ 到 $10^9$ 中枚举答案，但是这种朴素写法肯定拿不到满分，因为从 $1$ 枚举到 $10^9$ 太耗时间。我们可以在 $[1,~10^9]$ 的区间上进行二分作为答案，然后检查各个答案的可行性（一般使用贪心法）。**这就是二分答案。**

## 三分法

三分法可以用来查找凸函数的最大（小）值。

- 如果 `lmid` 和 `rmid` 在最大（小）值的同一侧：由于单调性，一定是二者中较大（小）的那个离最值近一些，较远的那个点对应的区间不可能包含最值，所以可以舍弃。
- 如果在两侧：由于最值在二者中间，我们舍弃两侧的一个区间后，也不会影响最值，所以可以舍弃。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
lmid = left + (right - left >> 1);
rmid = lmid + (right - lmid >> 1);  // 对右侧区间取半
if (cal(lmid) > cal(rmid))
    right = rmid;
else
    left = lmid;
```
##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>



## 分数规划

参见：[分数规划](misc/frac-programming.md)

分数规划通常描述为下列问题：每个物品有两个属性 $c_i$，$d_i$，要求通过某种方式选出若干个，使得 $\frac{\sum{c_i}}{\sum{d_i}}$ 最大或最小。

经典的例子有最优比率环、最优比率生成树等等。

分数规划可以用二分法来解决。

## 习题

### 一般二分应用

> [!NOTE] **[AcWing 102. 最佳牛围栏](https://www.acwing.com/problem/content/104/)**
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
#include<bits/stdc++.h>
using namespace std;

const int N = 100010;

int n, f;
double a[N], s[N];

bool check(double avg) {
    // - avg 变为：求是否存在长度大于等于f的一段和大于0
    for (int i = 1; i <= n; ++ i ) s[i] = s[i - 1] + a[i] - avg;
    
    double mins = 0;
    for (int k = f; k <= n; ++ k ) {
        mins = min(mins, s[k - f]);
        if (s[k] >= mins) return true;
    }
    return false;
}

int main() {
    cin >> n >> f;
    double l = 0, r = 0;
    for (int i = 1; i <= n; ++ i ) {
        cin >> a[i];
        r = max(r, a[i]);
    }
    while (r - l > 1e-5) {
        double mid = (l + r) / 2.0;
        if (check(mid)) l = mid;
        else r = mid;
    }
    // 因为精度和二分边界问题(边界其实不明显),所以我们需要是右边界.
    // 精度往往会使得l和r有极为细微的差异(如果你写l*1000你会发现样例答案是6499),
    // 而且我们这道题目r是最高边界.
    cout << (int)(r * 1000) << endl;
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

> [!NOTE] **[Luogu [NOIP2015 提高组] 跳石头](https://www.luogu.com.cn/problem/P2678)**
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
#include <bits/stdc++.h>
using namespace std;

const int N = 5e4 + 10;

int s, n, m;
int d[N];

// 如果不合法 ----> [j == i + 1]  or  [c < n + 1 - m]
bool check(int mid) {
    int c = 0;
    for (int i = 0; i <= n; ++ i ) {
        int j = i + 1;
        while (j <= n + 1 && d[j] - d[i] < mid)
            j ++ ;
        // f[j] - d[i] >= mid
        c ++ ;
        i = j - 1;
    }
    return c >= n + 1 - m;
}

int main() {
    cin >> s >> n >> m;
    
    d[0] = 0, d[n + 1] = s;
    for (int i = 1; i <= n; ++ i )
        cin >> d[i];
    
    int l = 1, r = s + 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (check(mid))
            l = mid + 1;
        else
            r = mid;
    }
    cout << l - 1 << endl;
    
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

> [!NOTE] **[Luogu 寻找段落](https://www.luogu.com.cn/problem/P1419)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 维护 idx 之前 [s, t] 长度的区间窗口

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;
const double eps = 1e-5;

int n, s, t;
int a[N], q[N];
double b[N], c[N];

bool check(double m) {
    for (int i = 1; i <= n; ++ i )
        b[i] = (double)a[i] - m;
    for (int i = 1; i <= n; ++ i )
        c[i] = c[i - 1] + b[i];
    
    // 经典 维护 idx 之前 [s, t] 长度的区间窗口
    int hh = 0, tt = -1;
    for (int i = s; i <= n; ++ i ) {
        while (hh <= tt && q[hh] < i - t)
            hh ++ ;

        // [弹尾部可以放前面] 以本段新加入的i-s作为起点 长度恰好为s
        while (hh <= tt && c[q[tt]] >= c[i - s])
            tt -- ;
        q[ ++ tt] = i - s;

        if (c[i] - c[q[hh]] >= 0)
            return true;
    }
    return false;
}

int main() {
    cin >> n >> s >> t;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    double l = -1e4, r = 1e4;
    while (r - l > eps) {
        double m = (l + r) / 2;
        if (check(m))
            l = m;
        else
            r = m;
    }
    printf("%.3lf\n", l);
    
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

> [!NOTE] **[AcWing 113. 特殊排序](https://www.acwing.com/problem/content/115/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> note

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Forward declaration of compare API.
// bool compare(int a, int b);
// return bool means whether a is less than b.

class Solution {
public:
    // 基于插入排序
    vector<int> specialSort(int N) {
        vector<int> res(1, 1);
        for (int i = 2; i <= N; ++ i ) {
            int l = 0, r = res.size() - 1;
            // 找到一个大于等于i的位置 插入
            while (l < r) {
                int mid = l + r >> 1;
                if (compare(res[mid], i)) l = mid + 1;
                else r = mid;
            }
            
            res.push_back(i);
            for (int j = res.size() - 2; j > r; -- j ) swap(res[j], res[j + 1]);
            if (compare(i, res[r])) swap(res[r], res[r + 1]);
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

> [!NOTE] **[LeetCode 4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)**
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
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n1 = nums1.size(), n2 = nums2.size();
        if (n1 > n2)
            return findMedianSortedArrays(nums2, nums1);
        
        // 找到一个左半边的结束位置, 使得两边有严格大小关系
        int l = 0, r = n1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            int l1 = mid;
            int l2 = (n1 + n2) / 2 - l1;
            if (nums1[l1] < nums2[l2 - 1])  // ATTENTION 个数 -> 下标  偏移
                l = mid + 1;
            else
                r = mid;
        }

        int l1 = l, l2 = (n1 + n2) / 2 - l1;
        int lv1 = l1 ? nums1[l1 - 1] : INT_MIN;
        int rv1 = l1 < n1 ? nums1[l1] : INT_MAX;
        int lv2 = l2 ? nums2[l2 - 1] : INT_MIN;
        int rv2 = l2 < n2 ? nums2[l2] : INT_MAX;
        if ((n1 + n2) & 1)
            return min(rv1, rv2);
        return (max(lv1, lv2) + min(rv1, rv2)) / 2.0;
    }
};
```

##### **Python**

```python
"""
1. 假设 len(nums1) < len(nums2), 如果不是的话，那就反过来用一遍算法即可【保证 j 不越界】；
2. 用二分来枚举求解 中位数 在 nums1数列中的位置
3. i 为 nums1递增数列中，【中位数】划分的右边界：也就是包括 nums1[i]在内，nums1 中后面的数值都是在【中位数】的右边
   i 的取值范围： [空, n1-1]， 因为可能 nums1 中所有的数都是在 中位数 的右边，也就是 i 左边的值 都是 <= 中位数
4. 根据 i 的位置, 可以得到 在nums2 中有多少个数被划分到 中位数 的 左侧 j = (n1+n2) // 2 - i, j 表示第几个数，对应的下标是 j-1
5. 比较 中位数 的右边 的第一个数 nums1[i] 和 中位数左边的第一数的大小 nums2[j-1] ，如果nums1[i]小一些，说明 i 当前还需要再往 r的方向走， 则 l = i + 1
6. 中位数 一定是在右半部分（向下取整的原因导致的）
"""

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1, n2 = len(nums1), len(nums2)
        if n1 > n2:return self.findMedianSortedArrays(nums2, nums1)

        l, r = 0, n1
        while l < r:
            i = l + (r - l) // 2
            j = (n1 + n2) // 2 - i 
            if nums1[i] < nums2[j-1]:
                l = i + 1
            else:
                r = i 
        
        """
        i + j 总长度是向下取整，中位数左边的总长度 是 <= 2 // (n1+n2)
        这里是做一个，边界值的判断
        """
        i = l
        j = (n1 + n2) // 2 - i 
        
        lv1 = nums1[i-1] if i > 0 else float('-inf')
        rv1 = nums1[i] if i < n1 else float('inf')
        lv2 = nums2[j-1] if j > 0 else float('-inf')
        rv2 = nums2[j] if j < n2 else float('inf')
        if (n1 + n2) & 1:
            return min(rv1, rv2)
        return (max(lv1, lv2) + min(rv1, rv2)) / 2
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)**
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
    vector<int> searchRange(vector<int>& nums, int target) {
        int n = nums.size();
        vector<int> res(2);
        res[0] = res[1] = -1;
        int l = 0, r = n;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < target) l = mid + 1;
            else r = mid;
        }
        res[0] = l < n && nums[l] == target ? l : -1;
        l = 0, r = n;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] <= target) l = mid + 1;
            else r = mid;
        }
        res[1] = l > 0 && nums[l - 1] == target ? l - 1 : -1;
        return res; 
    }
};
```

##### **Python**

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        l, r = 0, n
        p1 = 0
        while l < r:  #找到等于target的目标值的位置;
            m = l + (r - l) // 2
            if nums[m] < target:
                l = m + 1
            else:r = m
        if 0 <= l < n and nums[l] == target:
            p1 = l 
        else:return [-1, -1]

        l, r = 0, n 
        while l < r:   #找到第一个比target大的整数
            m = l + (r - l) // 2
            if nums[m] <= target:
                l = m + 1
            else:r = m
        p2 = l - 1  #不用再做判断了，因为如果不存在target，第一步已经返回了[-1,-1]
        return [p1, p2]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 重点在于 **正确性证明**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int n = nums.size();
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < nums[mid + 1])
                l = mid + 1;
            else
                r = mid;
        }
        return l;
    }
};
```

##### **Python**

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n
        while l < r:
            m = l + (r - l) // 2 
            if m + 1 < len(nums) and nums[m] < nums[m+1]:
                l = m + 1 
            else:
                r = m 
        return l 
      
      
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n - 1 
        while l < r:
            m = l + (r - l) // 2 
            if nums[m] < nums[m+1]:
                l = m + 1 
            else:
                r = m 
        return l       
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1901. 寻找峰值 II](https://leetcode.cn/problems/find-a-peak-element-ii)**
> 
> 题意: 
> 
> 保证相邻元素都不相等的矩阵，求任意一个峰值（大于所有相邻的元素）

> [!TIP] **思路**
> 
> 重点仍然在于 **正确性推导**
> 
> 对每一行考虑其最大值所在的列（实际上通过二分），则需要关注该列相邻两行的值的关系
> 
> - 如果相邻行更大，则峰值一定出现在相邻行所在的区间
> 
> - 如果相邻行更小，则峰值一定出现在当前行所在区间

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 510;

    int p[N];   // 记录每一行最大值的下标 [如果有多个最大值呢]

    vector<int> findPeakGrid(vector<vector<int>>& mat) {
        int n = mat.size(), m = mat[0].size();
        for (int i = 0; i < n; ++ i ) {
            int t = 0;
            for (int j = 0; j < m; ++ j )
                if (mat[i][j] > mat[i][t])
                    t = j;
            p[i] = t;
        }

        int l = 0, r = n - 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            int t = p[m];
            if (mat[m][t] < mat[m + 1][t])  // ATTENTION 找相同列比较
                l = m + 1;
            else
                r = m;
        }
        return {l, p[l]};
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

> [!NOTE] **[LeetCode 274. H 指数](https://leetcode.cn/problems/h-index/)**
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
    int hIndex(vector<int>& citations) {
        sort(citations.begin(), citations.end(), greater<int>());
        for (int h = citations.size(); h; -- h )
            if (citations[h - 1] >= h) return h;
        return 0;
    }

    int hIndex_2(vector<int>& citations) {
        int l = 0, r = citations.size() + 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            
            int c = 0;
            for (auto & v : citations)
                if (v >= m)
                    c ++ ;

            if (c >= m)
                l = m + 1;
            else
                r = m;
        }
        return l - 1;
    }
};
```

##### **Python**

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        l = 0
        r = n = len(citations)
        while l<r:
            mid = l + (r-l)//2
            if citations[mid] < n - mid:
                l = mid+1
            else:
                r = mid
        return n-l
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 275. H 指数 II](https://leetcode.cn/problems/h-index-ii/)**
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
    int hIndex(vector<int>& citations) {
        // 对于某个下标 i , 其位置及其右侧的值均大于等于 c[i] 也即个数为 n - i
        // 故找到最左侧的 i
        int n = citations.size(), l = 0, r = n;
        // if (!n) return 0;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (citations[m] < n - m)
                l = m + 1;
            else
                r = m;
        }
        return n - l;
    }
};
```

##### **Python**

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        l = 0
        r = n = len(citations)
        while l<r:
            mid = l + (r-l)//2
            if citations[mid] < n - mid:
                l = mid+1
            else:
                r = mid
        return n-l
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 540. 有序数组中的单一元素](https://leetcode.cn/problems/single-element-in-a-sorted-array/)**
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
    int singleNonDuplicate(vector<int> & nums) {
        nums.push_back(nums.back() + 1);
        int l = 0, r = nums.size() / 2 - 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (nums[m * 2] != nums[m * 2 + 1]) r = m;
            else l = m + 1;
        }
        return nums[l * 2];
    }

    int singleNonDuplicate_2(vector<int>& nums) {
        int res = 0;
        for (auto v : nums) res ^= v;
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

> [!NOTE] **[LeetCode 1818 绝对差值和](https://leetcode.cn/problems/minimum-absolute-sum-difference/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然二分
> 
> 注意再开个数组排序 如果直接用 `set<int>` 超时
> 
> 注意取左右两侧的写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        LL s = 0;
        vector<int> ve;
        for (int i = 0; i < n; ++ i ) {
            s += abs(nums1[i] - nums2[i]);
            ve.push_back(nums1[i]);
        }
        sort(ve.begin(), ve.end());
        
        LL res = s;
        for (int i = 0; i < n; ++ i ) {
            int t = lower_bound(ve.begin(), ve.end(), nums2[i]) - ve.begin();
            if (t < n)
                res = min(res, s - abs(nums1[i] - nums2[i]) + abs(ve[t] - nums2[i]));
            if (t > 0)
                res = min(res, s - abs(nums1[i] - nums2[i]) + abs(ve[t - 1] - nums2[i]));
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

> [!NOTE] **[LeetCode 2080. 区间内查询数字的频率](https://leetcode.cn/problems/range-frequency-queries/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典二分应用，【二分下标】
> 
> 注意加引用

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class RangeFreqQuery {
public:
    const static int N = 10010;
    
    vector<int> vs[N];
    
    RangeFreqQuery(vector<int>& arr) {
        for (int i = 0; i < N; ++ i )
            vs[i].clear();
        
        int n = arr.size();
        for (int i = 0; i < n; ++ i )
            vs[arr[i]].push_back(i);
    }
    
    int query(int left, int right, int value) {
        auto & ve = vs[value];
        auto l = lower_bound(ve.begin(), ve.end(), left);
        auto r = lower_bound(ve.begin(), ve.end(), right + 1);
        return r - l;
    }
};

/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery* obj = new RangeFreqQuery(arr);
 * int param_1 = obj->query(left,right,value);
 */
```

##### **C++ 并不关键的简单优化**

```cpp
class RangeFreqQuery {
public:
    const static int N = 10010;
    
    vector<int> vs[N];
    
    RangeFreqQuery(vector<int>& arr) {
        for (int i = 0; i < N; ++ i )
            vs[i].clear();
        
        int n = arr.size();
        for (int i = 0; i < n; ++ i )
            vs[arr[i]].push_back(i);
    }
    
    int query(int left, int right, int value) {
        auto & ve = vs[value];
        if (ve.empty())
            return 0;
        vector<int>::iterator l, r;
        if (left <= ve.front())
            l = ve.begin();
        else
            l = lower_bound(ve.begin(), ve.end(), left);
        if (right >= ve.back())
            r = ve.end();
        else
            r = lower_bound(ve.begin(), ve.end(), right + 1);
        return r - l;
    }
};

/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery* obj = new RangeFreqQuery(arr);
 * int param_1 = obj->query(left,right,value);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 二分答案

> [!NOTE] **[LeetCode 410. 分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分 或 巧妙状态定义

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 状态定义**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int splitArray(vector<int>& nums, int m) {
        int n = nums.size();
        vector<int> psum(n + 1);
        for (int i = 1; i <= n; ++ i ) psum[i] = psum[i - 1] + nums[i - 1];
        vector<vector<int>> f(n + 1, vector<int>(m + 1, inf));
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i ) {
            for (int j = 1; j <= i && j <= m; ++ j )
                for (int k = 0; k < i; ++ k )
                    f[i][j] = min(f[i][j], max(f[k][j - 1], psum[i] - psum[k]));
                
        }
        return f[n][m];
    }
};
```

##### **C++ 二分1**

```cpp
// 二分
class Solution {
public:
    vector<int> nums;
    int n, m;
    
    bool check(int mid) {
        int c = 0;
        for (int i = 0; i < n; ++ i ) {
            int j = i, s = 0;
            while (j < n && s + nums[j] <= mid)
                s += nums[j], j ++ ;
            if (i == j)
                return true;
            c ++ ;
            i = j - 1;
        }
        return c > m;
    }

    int splitArray(vector<int>& nums, int m) {
        this->nums = nums;
        this->n = nums.size(), this->m = m;
        int l = 0, r = INT_MAX;
        while (l < r) {
            int mid = l + (r - l) / 2;
            // 不合法 要增加
            if (check(mid))
                l = mid + 1;
            else
                r = mid;
        }
        return l;
    }
};

```

##### **C++ 二分2**

```cpp
// yxc 二分
class Solution {
public:
    bool check(vector<int>& nums, int m, int mid) {
        int sum = 0, cnt = 0;
        for (auto x: nums) {
            if (x > mid) return false;
            if (sum + x > mid) {
                cnt ++ ;
                sum = x;
            } else {
                sum += x;
            }
        }
        if (sum) cnt ++ ;
        return cnt <= m;
    }

    int splitArray(vector<int>& nums, int m) {
        int l = 0, r = INT_MAX;
        while (l < r) {
            int mid = (long long)l + r >> 1;
            if (check(nums, m, mid)) r = mid;
            else l = mid + 1;
        }
        return r;
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

> [!NOTE] **[LeetCode 475. 供暖器](https://leetcode.cn/problems/heaters/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然二分 注意check函数写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    vector<int> cs, ws;
    bool check(int m) {
        // not empty
        int p1 = 0, p2 = 0;
        while (p1 < cs.size() && p2 < ws.size()) {
            int x1 = cs[p1], x2 = ws[p2];
            if (abs(x2 - x1) <= m) ++ p1;
            else ++ p2;
        }
        return p1 == cs.size();
    }
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        cs = houses, ws = heaters;
        sort(cs.begin(), cs.end());
        sort(ws.begin(), ws.end());
        int l = 0, r = inf;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (check(m)) r = m;
            else l = m + 1;
        }
        return l;
    }
};

// ----------------------------------------
// yxc check函数写法
    bool check(int mid) {
        for (int i = 0, j = 0; i < houses.size(); i ++ ) {
            while (j < heaters.size() && abs(heaters[j] - houses[i]) > mid)
                j ++ ;
            if (j >= heaters.size()) return false;
        }
        return true;
    }
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 778. 水位上升的泳池中游泳](https://leetcode.cn/problems/swim-in-rising-water/)**
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
    int n;
    vector<vector<int>> g;
    vector<vector<bool>> st;
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

    bool dfs(int x, int y, int mid) {
        if (x == n - 1 && y == n - 1) return true;
        st[x][y] = true;
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < n && g[a][b] <= mid && !st[a][b])
                if (dfs(a, b, mid)) return true;
        }
        return false;
    }

    bool check(int mid) {
        if (g[0][0] > mid) return false;
        st = vector<vector<bool>>(n, vector<bool>(n));
        return dfs(0, 0, mid);
    }

    int swimInWater(vector<vector<int>>& grid) {
        g = grid;
        n = g.size();
        int l = 0, r = n * n - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (check(mid)) r = mid;
            else l = mid + 1;
        }
        return r;
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

> [!NOTE] **[LeetCode 793. 阶乘函数后 K 个零](https://leetcode.cn/problems/preimage-size-of-factorial-zeroes-function/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分性质 + 数学

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;

    LL f(LL x) {
        LL tot = 0;
        while (x)
            tot += x / 5, x /= 5;
        return tot;
    }

    int calc(int k) {
        if (k < 0)
            return 0;
        LL l = 0, r = 1e18;
        while (l < r) {
            LL m = (l + r) >> 1;
            // 求第一个大于k的
            if (f(m) <= k)
                l = m + 1;
            else
                r = m;
        }
        return (int)l;
    }

    int preimageSizeFZF(int k) {
        return calc(k) - calc(k - 1);
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

> [!NOTE] **[LeetCode 1292. 元素和小于等于阈值的正方形的最大边长](https://leetcode.cn/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单二分check + 前缀和

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int m, n;
    bool check(vector<vector<int>>& sum, int l, int tar) {
        // 遍历右下角坐标
        for (int i = l; i <= m; ++i)
            for (int j = l; j <= n; ++j) {
                int v = sum[i][j] - sum[i - l][j] - sum[i][j - l] +
                        sum[i - l][j - l];
                if (v <= tar) return true;
            }
        return false;
    }
    int maxSideLength(vector<vector<int>>& mat, int threshold) {
        m = mat.size(), n = mat[0].size();
        vector<vector<int>> sum(m + 1, vector<int>(n + 1));
        for (int i = 1; i <= m; ++i)
            for (int j = 1; j <= n; ++j)
                sum[i][j] = sum[i - 1][j] - sum[i - 1][j - 1] + sum[i][j - 1] +
                            mat[i - 1][j - 1];
        int l = 0, r = min(m, n) + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(sum, mid, threshold))
                l = mid + 1;
            else
                r = mid;
        }
        return l ? l - 1 : 0;
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

> [!NOTE] **[LeetCode 1300. 转变数组后最接近目标值的数组和](https://leetcode.cn/problems/sum-of-mutated-array-closest-to-target/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 显然二分check
>
> 需要注意：
>
> **1) 题目要求返回的方案值最小，因而右边界 r 需要设置为数组最大值而非任意大。**
>
> **2) 题目要求比较的是绝对值，因而使用 `<` 以及 `l = m + 1` ，求出比 target 大的方案**
>
> **3) 比较比 target 小的方案和 2) 中方案何种最有并返回**
> 
> 题解区在计算 check 的时候有 排序 / 前缀和 的小优化，可以参考，但不关键。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int check(vector<int>& arr, int m) {
        int sum = 0;
        for (auto v : arr) {
            if (v >= m)
                sum += m;
            else
                sum += v;
        }
        return sum;
    }
    int findBestValue(vector<int>& arr, int target) {
        int n = arr.size(), maxv = 0;
        for (auto v : arr) maxv = max(maxv, v);
        int l = 0, r = maxv;
        while (l < r) {
            int m = l + (r - l) / 2;
            // sum < target 说明m需要变大
            if (check(arr, m) < target)
                l = m + 1;
            else
                r = m;
        }
        if (check(arr, l) - target < target - check(arr, l - 1)) return l;
        return l - 1;
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

> [!NOTE] **[LeetCode 1552. 两球之间的磁力](https://leetcode.cn/problems/magnetic-force-between-two-balls/)**
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
    int check(vector<int>& pos, int dis) {
        int cnt = 0, l = INT_MIN / 2;
        for (auto v : pos) {
            if (v - dis >= l) {
                ++cnt;
                l = v;
            }
        }
        return cnt;
    }
    int maxDistance(vector<int>& position, int m) {
        sort(position.begin(), position.end());
        int n = position.size();
        int minl = INT_MAX, maxl = INT_MIN;
        for (int i = 1; i < n; ++i) {
            minl = min(minl, position[i] - position[i - 1]);
            maxl = max(maxl, position[i] - position[0]);  // not pos[i] - pos[i-1]
        }
        int l = minl, r = maxl + 1, checkv;
        while (l < r) {
            int mid = l + (r - l) / 2;
            int checkv = check(position, mid);
            if (checkv >= m)
                l = mid + 1;
            else
                r = mid;
        }
        return l - 1;
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

> [!NOTE] **[LeetCode 1648. 销售价值减少的颜色球](https://leetcode.cn/problems/sell-diminishing-valued-colored-balls/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 使用 `堆` 找到可选最大值，复杂度 $O(orders * log n)$ ，数据较大会超
> 
> 考虑：必然存在一个阈值 T 使得结束后所有颜色球个数 x <= T。
> 
> 故二分先查找 T，随后统计即可
> 
> **贪心做法 细节很多**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    typedef long long LL;
    const int mod = 1e9 + 7;
    int maxProfit(vector<int>& inventory, int orders) {
        int n = inventory.size();
        int l = 0, r = 1e9 + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            // 大于等于 mid 的总个数 二分结束时 s<=orders 也即 l=r=T
            long long s = 0;
            for (int i = 0; i < n; ++i)
                if (inventory[i] >= mid) s += (inventory[i] - mid + 1);
            if (s > orders)
                l = mid + 1;
            else
                r = mid;
        }
        LL res = 0;
        for (auto& rv : inventory)
            if (rv >= l) {
                LL lv = l, cnt = rv - lv + 1;
                (res += ((lv + rv) * cnt / 2) % mod) %= mod;
                orders -= cnt;
            }
        (res += LL(orders) * (l - 1) % mod) %= mod;
        return res;
    }
};
```

##### **C++ 贪心 TODO**

```cpp
class Solution {
public:
    using LL = long long;
    const static int MOD = 1e9 + 7;

    LL qpow(LL a, LL b) {
        LL ret = 1;
        while (b) {
            if (b & 1)
                ret = ret * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return ret;
    }

    int maxProfit(vector<int>& inventory, int orders) {
        int n = inventory.size();

        LL res = 0, sum = 0, inver = qpow(2, MOD - 2);
        for (auto x : inventory)
            res = (res + (LL)x * (x + 1) % MOD * inver % MOD) % MOD, sum += x;
        
        if (sum <= orders)
            return res;

        inventory.push_back(0); // 哨兵节点
        sort(inventory.begin(), inventory.end(), greater<int>());

        LL tot = res, k = orders;
        for (int i = 0; ; ++ i ) {
            LL x = inventory[i], cost = (LL)(inventory[i] - inventory[i + 1]) * (i + 1);
            tot = tot - x * (x + 1) % MOD * inver % MOD;
            if (cost <= k) {
                k -= cost;
                continue;
            }
            x -= k / (i + 1);
            LL a = k % (i + 1), b = i + 1 - a;
            tot = (tot + a * (x - 1 + 1) % MOD * (x - 1) % MOD * inver % MOD + b * (x + 1) % MOD * x % MOD * inver % MOD) % MOD;
            break;
        }
        return (res - tot + MOD) % MOD;
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

> [!NOTE] **[LeetCode 2333. 最小差值平方和](https://leetcode.cn/problems/minimum-sum-of-squared-difference/) [TAG]**
> 
> 题意: 
> 
> 同 [LeetCode 1648. 销售价值减少的颜色球](https://leetcode.cn/problems/sell-diminishing-valued-colored-balls/) 稍作修改

> [!TIP] **思路**
> 
> - **二分**
> 
> - **贪心维护**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 二分**

```cpp
class Solution {
public:
    using LL = long long;
    
    long long minSumSquareDiff(vector<int>& nums1, vector<int>& nums2, int k1, int k2) {
        int n = nums1.size(), k = k1 + k2;
        vector<LL> d(n);
        for (int i = 0; i < n; ++ i )
            d[i] = abs(nums1[i] - nums2[i]);
        
        int l = 0, r = 1e9; // ATTENTION r
        while (l < r) {
            int mid = l + (r - l) / 2;
            LL s = 0;
            for (auto x : d)
                if (x >= mid)
                    s += x - mid;
            if (s > k)
                l = mid + 1;
            else
                r = mid;
        }
        
        // 当前的 l 能够实现不超过 k 的消耗
        if (!l)
            return 0;   // needed
        
        LL res = 0, tot = k;
        for (auto x : d)
            if (x >= l)
                tot -= x - l;
        // tot 为剩余
        
        for (auto x : d)
            if (x >= l) {
                // 较大的部分看能减多少
                LL t = l - min(1ll, max(0ll, tot -- )); // ATTENTION: trick 写法
                res += t * t; 
            } else
                // 较小的部分直接加
                res += x * x;
        
        return res;
    }
};
```

##### **C++ 贪心**

```cpp
class Solution {
public:
    using LL = long long;
    
    long long minSumSquareDiff(vector<int>& nums1, vector<int>& nums2, int k1, int k2) {
        int n = nums1.size(), k = k1 + k2;
        vector<LL> d(n);
        for (int i = 0; i < n; ++ i )
            d[i] = abs(nums1[i] - nums2[i]);
        
        LL res = 0, sum = 0;
        for (int i = 0; i < n; ++ i )
            sum += d[i], res += d[i] * d[i];
        
        d.push_back(0); // 哨兵
        sort(d.begin(), d.end(), greater<int>());   // 从大到小排序
        
        if (sum <= k)
            return 0;   // needed 否则后面 for-loop 会越界
        
        // 从前往后削峰，看能削多少
        for (int i = 0; ; ++ i ) {
            LL x = d[i], cost = (d[i] - d[i + 1]) * (i + 1);
            res -= x * x;
            if (cost <= k) {
                k -= cost;
                continue;
            }
            x -= k / (i + 1); // 大一的数值
            // ATTENTION: 细节计算
            LL a = k % (i + 1), b = i + 1 - a;
            res += a * (x - 1) * (x - 1) + b * x * x;
            break;
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

> [!NOTE] **[LeetCode 1870. 准时到达的列车最小时速](https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> d;
    int n;
    double h;
    
    bool check(int m) {
        double ret = 0;
        for (int i = 0; i < n - 1; ++ i )
            ret += (int)((d[i] + m - 1) / m);
        return ret + double(d[n - 1]) / m <= h;
    }
    
    int minSpeedOnTime(vector<int>& dist, double hour) {
        this->d = dist;
        this->n = d.size();
        this->h = hour;
        
        int l = 1, r = 1e7;
        while (l < r) {
            int m = l + r >> 1;
            if (check(m))
                r = m;
            else
                l = m + 1;
        }
        return check(l) ? l : -1;
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

> [!NOTE] **[Codeforces C. Present](https://codeforces.com/problemset/problem/460/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心+二分+差分 应用 边界case

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Present
// Contest: Codeforces - Codeforces Round #262 (Div. 2)
// URL: https://codeforces.com/problemset/problem/460/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 极大化最小值 显然二分
// 但这里涉及到区间修改 不难想到从左向右扫描 只需关注当前和右侧
// 故二分内部实现线性扫描累加即可 注意会修改差分数组所以新建个临时数组

const int N = 100010;

int n, m, w;
int a[N], d[N], t[N];

bool check(int x) {
    int s = 0, c = 0;
    for (int i = 1; i <= n; ++i) {
        if (s + t[i] < x) {
            int cost = x - t[i] - s;
            t[i] += cost;
            if (i + w <= n)
                t[i + w] -= cost;
            c += cost;
            if (c > m)
                return false;
        }
        s += t[i];
    }

    return true;
}

int main() {
    cin >> n >> m >> w;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];
    for (int i = 1; i <= n; ++i)
        d[i] = a[i] - a[i - 1];

    // r = 1e9 + 2e5 cause `a[i] + m` can be 1e9 + 1e5
    // https://codeforces.com/contest/460/submission/111331064
    int l = 0, r = 1e9 + 2e5;
    while (l < r) {
        memcpy(t, d, sizeof d);
        int mid = l + r >> 1;
        if (check(mid))
            l = mid + 1;
        else
            r = mid;
    }

    cout << l - 1 << endl;

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

> [!NOTE] **[Codeforces Renting Bikes](http://codeforces.com/problemset/problem/363/D)**
> 
> 题意: 
> 
> 每个人有一定钱 $b_i$ ，自行车有一定价格 $p_i$ ，另有公共资金 a
>  
> 求最多可以多少人买车，以及个人出钱最少为多少

> [!TIP] **思路**
> 
> 较显然的有二分性质，故先二分可以买车的个数
> 
> 个人出钱最少时，显然优先个人出钱不足公共补，再看公共的留下多少抵扣个人出的部分
> 
> 注意二分时如果 $mid > m$ 直接 `return INF`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Renting Bikes
// Contest: Codeforces - Codeforces Round #211 (Div. 2)
// URL: https://codeforces.com/problemset/problem/363/D
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10;

int n, m, a;
LL b[N], p[N], t[N];

LL check(int mid) {
    if (mid > m)
        return 2e9;
    for (int i = 0; i < mid; ++i)
        t[i] = b[n - 1 - i];
    reverse(t, t + mid);
    LL x = 0;
    for (int i = 0; i < mid; ++i)
        if (t[i] < p[i])
            x += p[i] - t[i];
    return x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m >> a;
    for (int i = 0; i < n; ++i)
        cin >> b[i];
    sort(b, b + n);
    for (int i = 0; i < m; ++i)
        cin >> p[i];
    sort(p, p + m);

    int l = 0, r = n + 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (check(mid) <= a)
            l = mid + 1;
        else
            r = mid;
    }
    LL x = check(l - 1);
    LL left = a - x, res = 0;
    for (int i = 0; i < l - 1; ++i)
        res += min(t[i], p[i]); // 个人出的所有累计
    // cout << " res = " << res << " left = " << left << endl;

    cout << l - 1 << ' ' << max(0ll, res - left) << endl;

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

> [!NOTE] **[Codeforces D. Characteristics of Rectangles](https://codeforces.com/problemset/problem/333/D)**
> 
> 题意: 
> 
> 找到四个位置使得其作为矩形的四个角，且四个位置的最小值最大。求该最值。

> [!TIP] **思路**
> 
> 显然有极大化最小值，可以二分来做
> 
> **重难点是二分之后的检验思维 标记是否有两列满足条件**
> 
> 优化思维 反复思考总结

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Characteristics of Rectangles
// Contest: Codeforces - Codeforces Round #194 (Div. 1)
// URL: https://codeforces.com/problemset/problem/333/D
// Memory Limit: 256 MB
// Time Limit: 3000 ms

#include <bits/stdc++.h>
using namespace std;

// 最小值最大，考虑二分
// 重点在怎么二分：
using LL = long long;
const static int N = 1010;

int n, m;
LL g[N][N];
bool st[N][N];

bool check(int mid) {
    memset(st, 0, sizeof st);
    for (int i = 1; i <= n; ++i) {
        // q 记录当前行可用的点 (纵坐标)
        static int q[N * N];
        int cnt = 0;
        for (int j = 1; j <= m; ++j)
            if (g[i][j] >= mid)
                q[++cnt] = j;

        // ATTENTION
        // st 对前后两个纵坐标打标记，说明存在这两列属于同一行
        for (int j = 1; j <= cnt; ++j)
            for (int k = 1; k < j; ++k)
                if (st[q[k]][q[j]])
                    return true;
                else
                    st[q[k]][q[j]] = true;
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m;

    LL l = 0, r = 1e9;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            cin >> g[i][j], r = max(r, g[i][j]);

    r++;
    while (l < r) {
        LL m = l + r >> 1;
        if (check(m))
            l = m + 1;
        else
            r = m;
    }
    cout << l - 1 << endl;

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

> [!NOTE] **[Codeforces Read Time](http://codeforces.com/problemset/problem/343/C)**
> 
> 题意: 
> 
> 给你 $n$ 个探头，$m$ 个要读的轨道
> 
> $n$ 个探头的初始位置是 $h1...hn$（从小到大），$m$ 个轨道的位置为 $p1...pn$ (也是从小到大)
> 
> 探头可以左移或右移，这些探头可以一起动，每移动一格的时间为 1，探头读轨道不计时间，如果要读的轨道上就有探头那么就不需要时间，找最小的时间来读完这些轨道。

> [!TIP] **思路**
> 
> 较显然的，总耗时有二分性质，可以对总时间二分
> 
> **问题在于 check 的计算逻辑**
> 
> 每个探头可以左右移动，那么它的移动轨迹如何讨论起来比较复杂
> 
> - 推理 知应当由最左侧的探头覆盖尽量靠左侧的轨道
> 
> - 枚举探头 并维护可以覆盖的最右侧轨道 双指针推进

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Read Time
// Contest: Codeforces - Codeforces Round #200 (Div. 1)
// URL: https://codeforces.com/problemset/problem/343/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10;

int n, m;
LL h[N], p[N];

// ATTENTION 假定覆盖 p[l], p[r], 则最小消耗为该表达式
inline LL get(int i, int l, int r) {
    return abs(p[r] - p[l]) + min(abs(p[l] - h[i]), abs(p[r] - h[i]));
}

bool check(LL mid) {
    // ATTENTION 显然需要从最左侧的来从左往右去覆盖
    for (int i = 1, l = 1, r = 1; i <= n; ++i) {
        LL t = get(i, l, r);
        while (r <= m && t <= mid)  // 推进可以移动到的目标位置
            r++, t = get(i, l, r);
        l = r;  // ATTENTION
        if (r > m)
            return true;  // 已经可以覆盖所有轨道
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m;
    for (int i = 1; i <= n; ++i)
        cin >> h[i];
    for (int i = 1; i <= m; ++i)
        cin >> p[i];

    LL l = 0, r = 1e11;
    while (l < r) {
        LL mid = l + r >> 1;
        if (check(mid))
            r = mid;
        else
            l = mid + 1;
    }
    cout << l << endl;

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

> [!NOTE] **[LeetCode 2258. 逃离火灾](https://leetcode.cn/problems/escape-the-spreading-fire/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 1A
> 
> 较显然的：二分答案
> 
> 注意预处理距离，然后再判断。而非在判断内执行多源BFS

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    using LL = long long;
    const static int N = 2e4 + 10, INF = 1e9;
    
    int n, m;
    vector<vector<int>> g;
    vector<vector<LL>> d, t;
    
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    PII q[N];
    
    LL mid;
    vector<vector<bool>> st;
    
    bool dfs(int x, int y) {
        st[x][y] = true;
        if (x == n - 1 && y == m - 1)
            return t[x][y] + mid <= d[x][y];
        if (t[x][y] + mid >= d[x][y])
            return false;
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                continue;
            if (g[nx][ny] != 0 || st[nx][ny])
                continue;
            if (dfs(nx, ny))
                return true;
        }
        return false;
    }
    
    // 能否可以到达目标点
    // ==> 有一条 t + mid <= d 的路径
    bool check() {
        st = vector<vector<bool>>(n, vector<bool>(m));
        return dfs(0, 0);
        // return true;
    }
    
    int maximumMinutes(vector<vector<int>>& grid) {
        this->g = grid, this->n = g.size(), this->m = g[0].size();
        
        {
            this->d = vector<vector<LL>>(n, vector<LL>(m, (LL)2e9));
            int hh = 0, tt = -1;
            for (int i = 0; i < n; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (grid[i][j] == 1)
                        q[ ++ tt] = {i, j}, d[i][j] = 0;
            
            while (hh <= tt) {
                auto [x, y] = q[hh ++ ];
                for (int i = 0; i < 4; ++ i ) {
                    int nx = x + dx[i], ny = y + dy[i];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                        continue;
                    if (g[nx][ny] != 0)
                        continue;
                    if (d[nx][ny] > d[x][y] + 1) {
                        d[nx][ny] = d[x][y] + 1;
                        q[ ++ tt] = {nx, ny};
                    }
                }
            }
        }
        {
            this->t = vector<vector<LL>>(n, vector<LL>(m, (LL)2e9));
            int hh = 0, tt = -1;
            q[ ++ tt] = {0, 0}, t[0][0] = 0;
            while (hh <= tt) {
                auto [x, y] = q[hh ++ ];
                for (int i = 0; i < 4; ++ i ) {
                    int nx = x + dx[i], ny = y + dy[i];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                        continue;
                    if (g[nx][ny] != 0)
                        continue;
                    if (t[nx][ny] > t[x][y] + 1) {
                        t[nx][ny] = t[x][y] + 1;
                        q[ ++ tt] = {nx, ny};
                    }
                }
            }
        }
        
        
        LL l = 0, r = INF + 1;
        while (l < r) {
            mid = l + (r - l) / 2;
            if (check())
                l = mid + 1;
            else
                r = mid;
        }
        return l - 1;
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

> [!NOTE] **[LeetCode 774. 最小化去加油站的最大距离](https://leetcode.cn/problems/minimize-max-distance-to-gas-station/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 浮点数二分
> 
> 细节：注意 `floor` 取值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, k;
    vector<int> sts;

    bool check(double m) {
        int c = 0;
        for (int i = 1; i < n; ++ i ) {
            int d = sts[i] - sts[i - 1];
            int t = floor(double(d) / m - 1e-8);
            c += t;
            if (c > k)
                return false;
        }
        return true;
    }

    double minmaxGasDist(vector<int>& stations, int k) {
        this->sts = stations, this->k = k, this->n = sts.size();
        double l = 0, r = 0;
        for (int i = 1; i < n; ++ i )
            r = max(r, (double)(sts[i] - sts[i - 1]));
        
        while (fabs(r - l) > 1e-8) {
            double m = (l + r) / 2.0;
            if (check(m))
                r = m;
            else
                l = m;
        }
        return l;
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

> [!NOTE] **[LeetCode 1231. 分享巧克力](https://leetcode.cn/problems/divide-chocolate)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分答案
> 
> 注意 `for-loop` 校验过程中如满足条件需要提前 break (直接 `return false` 的话仍然需要在 return 之前判断 c 的值，略有点麻烦)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, k;
    vector<int> ss;

    bool check(int m) {
        int c = 0;
        for (int i = 0; i < n; ++ i ) {
            int j = i + 1, t = ss[i];
            while (j < n && t < m)
                t += ss[j ++ ];
            if (t < m)
                break;
            c ++ ;
            i = j - 1;
        }
        return c >= k + 1;
    }

    int maximizeSweetness(vector<int>& sweetness, int k) {
        this->ss = sweetness;
        this->n = ss.size(), this->k = k;
        int l = 0, r = 1e9 + 10;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (check(m))
                l = m + 1;
            else
                r = m;
        }
        return l - 1;
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

> [!NOTE] **[LeetCode 2513. 最小化两个数组中的最大值](https://leetcode.cn/problems/minimize-the-maximum-of-two-arrays/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分即可
> 
> **重点是理清楚二分校验逻辑**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 本质是容斥
    // 假定答案为 n 则 n 以内不能给 n1 的有 x1 个，不能给 n2 的有 x2 个，两个都不能的有 x3 个
    // 则可以用的数量为  个
    
    using LL = long long;
    
    LL d1, d2, lcm, u1, u2;
    
    bool check(LL m) {
        LL x1 = m / d1, x2 = m / d2, x3 = m / lcm;
        x1 -= x3, x2 -= x3;
        m -= x1 + x2 + x3;
        return m >= max(0ll, u2 - x1) + max(0ll, u1 - x2);
    }
    
    int minimizeSet(int divisor1, int divisor2, int uniqueCnt1, int uniqueCnt2) {
        this->d1 = divisor1, this->d2 = divisor2, this->u1 = uniqueCnt1, this->u2 = uniqueCnt2;
        this->lcm = d1 / __gcd(d1, d2) * d2;
        
        LL l = 0ll, r = (LL)1e12;
        while (l < r) {
            LL m = l + (r - l) / 2;
            if (check(m))
                r = m;
            else
                l = m + 1;
        }
        return l;
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

> [!NOTE] **[LeetCode 3007. 价值和小于等于 K 的最大数字](https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准二分 内部计算细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 10^15 = 45bit
    using LL = long long;
    
    bool check(LL m, LL k, int x) {
        LL res = 0;
        // bit index, start from 1
        for (int i = x; i < 50; i += x ) {
            int t = i - 1;  // actual index
            LL v = 1ll << t;
            if (v > m)
                break;
            
            bool right_exist = x > 1;
            LL right = m % v;
            LL left = m >> (t + 1);
            
            // ATTENTION 计数规则: 左侧可选范围，右侧随便选
            LL add = max(left, 0ll) * v;
            
            if (m >> t & 1) // 如果上限可以取到
                add += (right + 1);
            
            res += add;
        }
        return res <= k;
    }
    
    long long findMaximumNumber(long long k, int x) {
        // 找到第一个不符合条件的
        LL l = 1, r = 1e15;
        while (l < r) {
            LL m = l + (r - l) / 2;
            if (check(m, k, x))
                l = m + 1;
            else
                r = m;
        }
        return l - 1;
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

> [!NOTE] **[LeetCode 3048. 标记所有下标的最早秒数 I](https://leetcode.cn/problems/earliest-second-to-mark-indices-i/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然有二分性质 核心在于二分校验逻辑
> 
> 标记 x 时必然发生在 changeIndices[i] == x 时，有较严格条件限制，则可假定标记行为一定发生在末尾，并从前往后贪心填充即可【从前往后比从后往前实现更简单】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    using PII = pair<int, int>;
    const static int N = 5e3 + 10;
    
    int n;
    int ns[N];
    vector<int> cs[N];
    
    bool check(int m) {
        if (m < n)
            return false;

        vector<PII> xs;     // ATTENTION 如果是逆序往前推导 实现会非常困难 考虑从前往后尽可能早的填充靠前的
        for (int i = 1; i <= n; ++ i ) {
            auto it = lower_bound(cs[i].begin(), cs[i].end(), m + 1);
            if (it == cs[i].begin())
                return false;
            it -- ;
            xs.push_back({*it, i});
        }
        
        sort(xs.begin(), xs.end());
        int tot = 0, last = 0;
        
        for (auto [idx, v] : xs) {
            tot += idx - last;  // 新增的空位置
            if (tot < ns[v] + 1)
                return false;   // 无法填充当前数字
            tot -= ns[v] + 1;
            last = idx;
        }
        return true;
    }
    
    int earliestSecondToMarkIndices(vector<int>& nums, vector<int>& changeIndices) {
        this->n = nums.size();
        for (int i = 1; i <= n; ++ i )
            ns[i] = nums[i - 1];
        for (int i = 1; i <= changeIndices.size(); ++ i )
            cs[changeIndices[i - 1]].push_back(i);
        
        for (int i = 1; i <= n; ++ i )
            if (cs[i].empty())
                return -1;
        
        int l = 0, r = 2e9;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (check(m))
                r = m;
            else
                l = m + 1;
        }
        return check(l) ? l : -1;
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

> [!NOTE] **[LeetCode 3049. 标记所有下标的最早秒数 II](https://leetcode.cn/problems/earliest-second-to-mark-indices-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 需要非常注意的是...题意和前面不一样...
> 
> 本题不对标记时机做要求，则问题得以简化
> 
> 校验函数需结合贪心思想 使用反悔堆维护发生 "快速消除" 的下标及相应代价即可
> 
> 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    using PII = pair<int, int>;
    const static int N = 5e3 + 10;
    
    int n;
    int ns[N];
    // vector<int> cs[N];
    int ci[N], first[N];
    LL sum = 0;
    
    bool check(int m) {
        if (m < n)
            return false;
        
        // 显然如果能一次撤销(直接打到0) 都应当一次撤销
        // 如果对于某个位置来说无法撤销 则应当尝试回滚之前的撤销取消 取消的时候应当取消回滚cost最小的
        //  ATTENTION 由于swap的情况存在 必须逆序遍历[思考]
        priority_queue<int, vector<int>, greater<int>> heap;     // cost
        
        // ATTENTION 因为需要关注最左侧的位置 因而不能像T3那样跳跃访问
        //  需要使用原始 changeIndices
        LL cnt = 0, save = 0;
        for (int i = m; i >= 1; -- i ) {
            int idx = ci[i];
            if (ns[idx] <= 1 || first[idx] < i) {   // 当前位置可以作为前面某个位置的标记操作占位
                cnt ++ ;
                continue;
            }

            if (cnt) {
                heap.push(ns[idx]);
                save += ns[idx];
                cnt -- ;
            } else {
                if (!heap.empty() && heap.top() <= ns[idx]) {
                    // swap
                    int t = heap.top();
                    save += ns[idx] - t;
                    heap.pop(), heap.push(ns[idx]);
                }
                cnt ++ ;    // ATTENTION 当前位置本身没有新增操作 可以保留这个时间【思考】
            }
        }
        // sum + n: 理论上总的操作次数
        return cnt >= sum + n - save - heap.size();
    }
    
    int earliestSecondToMarkIndices(vector<int>& nums, vector<int>& changeIndices) {
        this->n = nums.size();
        for (int i = 1; i <= n; ++ i )
            ns[i] = nums[i - 1];
        
        memset(first, 0, sizeof first);
        for (int i = changeIndices.size(); i >= 1; -- i )      // reverse for `first`
            ci[i] = changeIndices[i - 1], first[changeIndices[i - 1]] = i;

        this->sum = 0;
        for (auto x : nums)
            sum += x;
        
        int l = 0, r = changeIndices.size() + 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (check(m))
                r = m;
            else
                l = m + 1;
        }
        return (l <= changeIndices.size()/*ATTENTION*/ && check(l)) ? l : -1;
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

> [!NOTE] **[LeetCode 3116. 单面值组合的第 K 小金额](https://leetcode.cn/problems/kth-smallest-amount-with-single-denomination-combination/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 属于 [878. 第 N 个神奇数字](https://leetcode.cn/problems/nth-magical-number/) [1201. 丑数 III](https://leetcode.cn/problems/ugly-number-iii/) 的扩展
> 
> 数据范围下模拟显然不现实
> 
> 考虑枚举具体的值 => 引入问题: 需要去重 => 结合 **容斥原理** 去重 => 结合数据范围**二进制枚举**所有集合 (子集枚举)
> 
> 【容斥原理一般化】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // k 太大了  显然没办法模拟枚举
    //
    // 考虑数据范围：
    // - 币种类不超过15 且两两不同
    // - 每个coin的币值不超过25
    //
    // => 二分答案 校验函数里子集枚举去重
    
    using LL = long long;
    const static int N = (1 << 15) + 10;
    
    vector<int> coins;
    int n;
    
    LL v[N];
    
    LL check(LL m) {
        LL ret = 0;
        for (int i = 1; i < 1 << n; ++ i ) {
            int c = __builtin_popcount(i);
            if (c & 1)
                ret += m / v[i];
            else
                ret -= m / v[i];
        }
        return ret;
    }
    
    long long findKthSmallest(vector<int>& coins, int k) {
        this->coins = coins;
        this->n = coins.size();
        
        {
            v[0] = 0;
            for (int i = 1; i < 1 << n; ++ i ) {
                LL t = 1;
                vector<int> xs;
                for (int j = 0; j < n; ++ j )
                    if (i >> j & 1) {
                        t = t / __gcd(t, (LL)coins[j]) * coins[j];
                        xs.push_back(coins[j]);
                    }
                v[i] = t;
            }
        }
        
        LL l = 0, r = 1e15;
        while (l < r) {
            LL m = l + (r - l) / 2;
            if (check(m) < k)
                l = m + 1;
            else
                r = m;
        }
        return l;
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

### 二分套二分

> [!NOTE] **[LeetCode 2040. 两个有序数组的第 K 小乘积](https://leetcode.cn/problems/kth-smallest-product-of-two-sorted-arrays/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 比较显然的 二分套二分即可
> 
> 需要注意的点是数据有负数
> 
> **细节综合题**
> 
> - **在选取左右边界时需要事先考虑计算最小最大值（计算方式）**
> 
> - **check 中 v 的值不可知 不好二分 根据正负划分情况即可解决**
> 
> 【**非常好的二分综合应用题**】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    
    vector<int> ns1, ns2;
    int n1, n2;
    LL k;   // ATTENTION
    
    bool check(LL mid) {
        LL c = 0;
        for (auto v : ns1) {
            int l = 0, r = n2;
            // 无法直接二分 故根据v的正负性质来二分
            if (v > 0) {
                // 找到 > mid 的第一个数
                // 左侧的所有即为所求（不包括当前）
                while (l < r) {
                    int m = l + (r - l) / 2;
                    if ((LL)ns2[m] * v <= mid)
                        l = m + 1;
                    else
                        r = m;
                }
                c += l;
            } else if (v < 0) {
                // 找到 <= mid 的第一个数
                // 右侧的所有即为所求（包括当前）
                while (l < r) {
                    int m = l + (r - l) / 2;
                    if ((LL)ns2[m] * v > mid)
                        l = m + 1;
                    else
                        r = m;
                }
                c += (n2 - l);
            } else {
                // v == 0
                if (mid >= 0)
                    c += n2;
            }
        }
        return c >= k;
    }
    
    // 经典求两数组乘积最值
    pair<LL, LL> getMinMax() {
        LL lv1 = ns1[0], rv1 = ns1[n1 - 1];
        LL lv2 = ns2[0], rv2 = ns2[n2 - 1];
        vector<LL> ve = {lv1 * lv2, lv1 * rv2, lv2 * rv1, rv1 * rv2};
        LL minv = LONG_MAX, maxv = LONG_MIN;
        for (auto v : ve)
            minv = min(minv, v), maxv = max(maxv, v);
        return {minv, maxv};
    }
    
    long long kthSmallestProduct(vector<int>& nums1, vector<int>& nums2, long long k) {
        this->ns1 = nums1, this->ns2 = nums2;
        this->n1 = ns1.size(), this->n2 = ns2.size(), this->k = k;
        auto [l, r] = getMinMax();
        while (l < r) {
            LL m = l + (r - l) / 2;
            if (check(m))
                r = m;
            else
                l = m + 1;
        }
        return l;
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

### 三分 [同样可以三分答案]

> [!NOTE] **[AcWing 1420. 通电围栏](https://www.acwing.com/problem/content/1422/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 三分 要求：必须是凸/凹函数
> 
> 显然有三分性质 三分套三分

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

#define x first
#define y second

using PDD = pair<double, double>;
const int N = 160;
const double eps = 1e-6;

int n;
struct Segment {
    PDD a, b;
} seg[N];

double get_dist(PDD a, PDD b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double f(double x, double y) {
    double s = 0;
    for (int i = 0; i < n; ++ i ) {
        auto a = seg[i].a, b = seg[i].b;
        double da = get_dist({x, y}, a);
        double db = get_dist({x, y}, b);
        // 计算距离
        // Case1 到端点距离
        // Case2 x / y 坐标差 两种情况
        double d = min(da, db);
        if (x >= a.x && x <= b.x) d = fabs(y - a.y);
        else if (y >= a.y && y <= b.y) d = fabs(x - a.x);
        s += d;
    }
    return s;
}

// 三分纵坐标
double g(double x, double & y) {
    double l = 0, r = 100;
    while (r - l > eps) {
        double m1 = l + (r - l) / 3, m2 = l + (r - l) / 3 * 2;
        if (f(x, m1) >= f(x, m2)) l = m1;
        else r = m2;
    }
    y = r;
    return f(x, y);
}

int main() {
    cin >> n;
    for (int i = 0; i < n; ++ i ) {
        double x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        if (x1 > x2) swap(x1, x2);
        if (y1 > y2) swap(y1, y2);
        seg[i] = {{x1, y1}, {x2, y2}};
    }
    
    // 三分横坐标
    double l = 0, r = 100, y;
    while (r - l > eps) {
        double m1 = l + (r - l) / 3, m2 = l + (r - l) / 3 * 2;
        if (g(m1, y) >= g(m2, y)) l = m1;
        else r = m2;
    }
    double d = f(r, y);
    printf("%.1lf %.1lf %.1lf\n", r, y, d);
    
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

> [!NOTE] **[LeetCode 1515. 服务中心的最佳位置](https://leetcode.cn/problems/best-position-for-a-service-centre/)** [TAG]
> 
> 题意: POJ2420 原题

> [!TIP] **思路**
> 
> 求凸函数极值，有多种算法
> 
> - 三分
> - 模拟退火
> - 梯度下降
> - 还有大佬直接调scipy的优化库

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 模拟退火**

```cpp
class Solution {
public:
    using ld = double;
    const ld pi = 3.1415926535897932384626;
    double getMinDistSum(vector<vector<int>>& p) {
        ld x = 0, y = 0;
        int n = p.size();
        for (int i = 0; i < n; ++i) {
            x += p[i][0];
            y += p[i][1];
        }
        x /= n;
        y /= n;

        auto go = [&](ld x, ld y) {
            ld ret = 0;
            for (int i = 0; i < n; ++i)
                ret += sqrt((x - p[i][0]) * (x - p[i][0]) +
                            (y - p[i][1]) * (y - p[i][1]));
            return ret;
        };

        ld T = 100;
        const ld eps = 1e-8;
        while (T > eps) {
            T *= 0.99;
            ld rd = (rand() % 10000 + 1) / 10000.0;
            ld a = 2 * pi * rd;
            ld tx = x + T * cos(a), ty = y + T * sin(a);
            auto d = go(tx, ty) - go(x, y);
            if (d < 0) { x = tx, y = ty; }
        }

        return go(x, y);
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

### 旋转排序数组

> [!WARNING]  右端点有意义
> 
> `[l, r]` 而非 `[l, r)` 并且注意和右侧比较（such as `nums[mid] < nums[r]`）

> [!NOTE] **[LeetCode 153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)**
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
    int findMin(vector<int>& nums) {
        int n = nums.size();
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] > nums[r])
                l = mid + 1;
            else if (nums[mid] < nums[r])
                r = mid;
            // else 
        }
        return nums[l];
    }
};
```

##### **Python**

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n - 1
        while l < r:
            m = l + (r - l) // 2 
            if nums[m] > nums[r]:
                l = m + 1 
            elif nums[m] < nums[r]:
                r = m 
            else:
                r -= 1  # 当存在重复元素时
        return nums[l]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 154. 寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/)**
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
    int findMin(vector<int>& nums) {
        int n = nums.size();
        int l = 0, r = n - 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (nums[m] > nums[r])
                l = m + 1;
            else if (nums[m] < nums[r])
                r = m;
            else -- r ;
        }
        return nums[l];
    }
};
```

##### **Python**

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n - 1
        while l < r:
            m = l + (r - l) // 2 
            if nums[m] > nums[r]:
                l = m + 1 
            elif nums[m] < nums[r]:
                r = m 
            else:
                r -= 1  # 当存在重复元素时
        return nums[l]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)**
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
    int search(vector<int>& nums, int target) {
        int n = nums.size();
        int l = 0, r = n - 1;
        while (l < r) {
            int m = l + r >> 1;
            if (nums[m] < nums[r]) {
                if (target > nums[m] && target <= nums[r])
                    l = m + 1;
                else
                    r = m;
            } else if (nums[m] > nums[r]) {
                if (target > nums[m] || target <= nums[r])
                    l = m + 1;
                else
                    r = m;
            }
            // else
        }
        return target == nums[l] ? l : -1;
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

> [!NOTE] **[LeetCode 81. 搜索旋转排序数组 II](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)**
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
    bool search(vector<int>& nums, int target) {
        int n = nums.size();
        // 旋转排序树组 右端点有意义
        int l = 0, r = n - 1;
        while (l < r) {
            int m = l + r >> 1;
            if (nums[m] < nums[r]) {
                if (target > nums[m] && target <= nums[r])
                    l = m + 1;
                else
                    r = m;
            } else if (nums[m] > nums[r]) {
                if (target > nums[m] || target <= nums[r])
                    l = m + 1;
                else
                    r = m;
            } else
                r -- ;
        }
        return nums[l] == target;
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

### 抽象二分思想

> [!NOTE] **[LeetCode 74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)**
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
    // 根据题目 【每行的第一个整数大于前一行的最后一个整数】可以直接一维二分
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        int n = matrix.size(), m = matrix[0].size();

        int l = 0, r = n * m - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (matrix[mid / m][mid % m] >= target) r = mid;
            else l = mid + 1;
        }

        return matrix[r / m][r % m] == target;
    }
};

    bool searchMatrix2(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        if (!m) return false;
        int n = matrix[0].size();
        int u = 0, r = n - 1;
        while (u < m && r >= 0) {
            if (matrix[u][r] == target) return true;
            else if (matrix[u][r] < target) ++ u ;
            else -- r ;
        }
        return false;
    }
};
```

##### **Python**

```python
class Solution:
    def searchMatrix(self, arr: List[List[int]], target: int) -> bool:
        if not arr:return False
        n, m = len(arr), len(arr[0])
        i, j = 0, m - 1 
        while i < n and j >= 0:
            if arr[i][j] == target:
                return True 
            elif arr[i][j] > target:
                j -= 1
            else:i += 1 
        return False
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)**
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
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        int n = matrix.size(), m = matrix[0].size();
        int i = 0, j = m - 1;
        while (i < n && j >= 0) {
            int t = matrix[i][j];
            if (t == target) return true;
            else if (t > target) j -- ;
            else i ++ ;
        }
        return false;
    }
};
```

##### **Python**

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        if not matrix:return False
        i, j = 0, len(matrix[0]) - 1
        while i < len(matrix) and j >= 0:
            if matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
            else:return True 
        return False
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 378. 有序矩阵中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int l = INT_MIN, r = INT_MAX;
        while (l < r) {
            int mid = (long long)l + r >> 1;
            int i = matrix[0].size() - 1, cnt = 0;
            for (int j = 0; j < matrix.size(); j ++ ) {
                while (i >= 0 && matrix[j][i] > mid) i -- ;
                cnt += i + 1;
            }
            if (cnt >= k) r = mid;
            else l = mid + 1;
        }
        return r;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    bool check(vector<vector<int>>& mat, int m, int k, int n) {
        int i = n-1, j = 0;
        int cnt = 0;
        while (i >= 0 && j < n) {
            if (mat[i][j] <= m)
                cnt += i + 1, ++ j ;
            else
                -- i ;
        }
        return cnt >= k;
    }
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = matrix.size();
        int l = matrix[0][0], r = matrix[n - 1][n - 1];
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(matrix, mid, k, n)) {
                // 个数大于等于k
                r = mid;
            } else
                l = mid + 1;   // 个数小于k
        }
        return l;
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