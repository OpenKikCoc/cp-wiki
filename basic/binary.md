二分查找（英语：binary search），也称折半搜索（英语：half-interval search）、对数搜索（英语：logarithmic search），是用来在一个有序数组中查找某一元素的算法。

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

### STL 的二分查找

C++ 标准库中实现了查找首个不小于给定值的元素的函数 [`std::lower_bound`](https://zh.cppreference.com/w/cpp/algorithm/lower_bound) 和查找首个大于给定值的元素的函数 [`std::upper_bound`](https://zh.cppreference.com/w/cpp/algorithm/upper_bound)，二者均定义于头文件 `<algorithm>` 中。

二者均采用二分实现，所以调用前必须保证元素有序。

> [!WARNING] 再次提醒: **`lower_bound` 和 `upper_bound` 的时间复杂度**
> 
> 在一般的数组里，这两个函数的时间复杂度均为 $O(\log n)$，但在 `set` 等关联式容器中，直接调用 `lower_bound(s.begin(),s.end(),val)` 的时间复杂度是 $O(n)$ 的。
> 
> `set` 等关联式容器中已经封装了 `lower_bound` 等函数（像 `s.lower_bound(val)` 这样），这样调用的时间复杂度是 $O(\log n)$ 的。

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


### 旋转排序数组 (右端点有意义)

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