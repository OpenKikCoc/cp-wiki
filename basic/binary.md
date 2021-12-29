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

> [!NOTE] **[LeetCode 4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)**
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
        if (n1 > n2) return findMedianSortedArrays(nums2, nums1);
        // l 是 nums1 中划分点的右边界 取值范围 [0, n1]
        // 对应：l 左侧的值为 [空, n1-1]
        // 也即：l 左侧的值 <= 中点值
        int l = 0, r = n1;
        // 找到第一个大于等于 lv2 的位置 l 
        while (l < r) {
            // 左侧有 i 个  总共应有(n1+n2)/2个
            int i = l + (r - l) / 2;
            // nums2 右边界 因为n1和n2大小关系 j永远不会取0(除非n1=n2)
            int j = (n1 + n2) / 2 - i;
            if (nums1[i] < nums2[j-1]) l = i + 1;  // rv1 和 lv2 对比   nums1 选的太少
            else r = i;
        }
        int i = l, j = (n1 + n2) / 2 - i;
        int lv1 = i ? nums1[i - 1] : INT_MIN;
        int rv1 = i < n1 ? nums1[i] : INT_MAX;
        int lv2 = j ? nums2[j - 1] : INT_MIN;
        int rv2 = j < n2 ? nums2[j] : INT_MAX;
        if((n1 + n2) & 1) return min(rv1, rv2);
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

> [!NOTE] **[LeetCode 34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)**
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

> [!NOTE] **[LeetCode 162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)**
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

> [!NOTE] **[LeetCode 274. H 指数](https://leetcode-cn.com/problems/h-index/)**
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

> [!NOTE] **[LeetCode 275. H 指数 II](https://leetcode-cn.com/problems/h-index-ii/)**
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

> [!NOTE] **[LeetCode 540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)**
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

### 二分答案

> [!NOTE] **[LeetCode 410. 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/)**
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

> [!NOTE] **[LeetCode 475. 供暖器](https://leetcode-cn.com/problems/heaters/)**
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

> [!NOTE] **[LeetCode 778. 水位上升的泳池中游泳](https://leetcode-cn.com/problems/swim-in-rising-water/)**
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

> [!NOTE] **[LeetCode 793. 阶乘函数后 K 个零](https://leetcode-cn.com/problems/preimage-size-of-factorial-zeroes-function/)**
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

### 三分

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


### 旋转排序数组

> [!WARNING]  右端点有意义
> 
> `[l, r]` 而非 `[l, r)` 并且注意和右侧比较（such as `nums[mid] < nums[r]`）

> [!NOTE] **[LeetCode 153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)**
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

> [!NOTE] **[LeetCode 154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)**
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

> [!NOTE] **[LeetCode 33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)**
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

> [!NOTE] **[LeetCode 81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)**
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

> [!NOTE] **[LeetCode 74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)**
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

> [!NOTE] **[LeetCode 240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)**
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

> [!NOTE] **[LeetCode 378. 有序矩阵中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/)**
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