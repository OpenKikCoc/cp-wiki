
## 快速排序

快速排序（英语：Quicksort），又称分区交换排序（英语：partition-exchange sort），简称快排，是一种被广泛运用的排序算法。

### 原理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// C++ Version
void qsort(int l, int r) {
    if (l >= r) return;
    int i = l - 1, j = r + 1, x = a[l + r >> 1];
    while (i < j) {
        do i ++ ; while (a[i] < x);
        do j -- ; while (a[j] > x);
        if (i < j) swap(a[i], a[j]);
    }
    // ATTENTION 只能用 j 
    // Luogu https://www.luogu.com.cn/problem/P1177
    qsort(l, j);
    qsort(j + 1, r);
}

int main() {
    cin >> n;
    for (int i = 0; i < n; ++ i ) cin >> a[i];
    
    qsort(0, n - 1);
    
    for (int i = 0; i < n; ++ i ) cout << a[i] << ' ';
    cout << endl;
    return 0;
}
```

##### **Python**

```python
# Python Version
# 荷兰国旗的写法 不会超时
def partition(l, r):
	import random
	x = random.randint(l, r)
	arr[x], arr[r] = arr[r], arr[x]
	less, more = l - 1, r
	
	while l < more:
		if arr[l] < arr[r]:
			less += 1
			arr[l], arr[less] = arr[less], arr[l]
			l += 1
		elif arr[l] > arr[r]:
			more -= 1
			arr[l], arr[more] = arr[more], arr[l]
		else:l += 1
	arr[more], arr[r] = arr[r], arr[more]
	return [less + 1, more - 1]
	
 
def quick_sort(l, r):
	if l < r:
		p = partition(l, r)
		quick_sort(l, p[0] - 1)
		quick_sort(p[1] + 1, r)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    quick_sort(0, n - 1)
    for i in arr:
        print(i, end=' ')
```

<!-- tabs:end -->
</details>

<br>

### 优化思想 pivot

较为常见的优化思路有以下三种[^ref3]。

- 通过 **三数取中（即选取第一个、最后一个以及中间的元素中的中位数）** 的方法来选择两个子序列的分界元素（即比较基准）。这样可以避免极端数据（如升序序列或降序序列）带来的退化；
- 当序列较短时，使用 **插入排序** 的效率更高；
- 每趟排序后，**将与分界元素相等的元素聚集在分界元素周围**，这样可以避免极端数据（如序列中大部分元素都相等）带来的退化。

下面列举了几种较为成熟的快速排序优化方式。

#### 三路快速排序

三路快速排序（英语：3-way Radix Quicksort）是快速排序和 `基数排序` 的混合。它的算法思想基于 [荷兰国旗问题](https://en.wikipedia.org/wiki/Dutch_national_flag_problem) 的解法。

与原始的快速排序不同，三路快速排序在随机选取分界点 $m$ 后，将待排数列划分为三个部分：小于 $m$、等于 $m$ 以及大于 $m$。这样做即实现了将与分界元素相等的元素聚集在分界元素周围这一效果。

三路快速排序在处理含有多个重复值的数组时，效率远高于原始快速排序。其最佳时间复杂度为 $O(n)$。

#### 内省排序

内省排序（英语：Introsort 或 Introspective sort）是快速排序和 堆排序 的结合，由 David Musser 于 1997 年发明。内省排序其实是对快速排序的一种优化，保证了最差时间复杂度为 $O(n\log n)$。

内省排序将快速排序的最大递归深度限制为 $\lfloor \log_2n \rfloor$，超过限制时就转换为堆排序。这样既保留了快速排序内存访问的局部性，又可以防止快速排序在某些情况下性能退化为 $O(n^2)$。

从 2000 年 6 月起，SGI C++ STL 的 `stl_algo.h` 中 `sort()` 函数的实现采用了内省排序算法。

## 线性找第 k 大的数

可以证明，在期望意义下，程序的时间复杂度为 $O(n)$。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
int quick_sort(int q[], int l, int r, int k) {
    if (l >= r) return q[l];

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j) {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }

    if (j - l + 1 >= k) return quick_sort(q, l, j, k);
    else return quick_sort(q, j + 1, r, k - (j - l + 1));
}

int main() {
    int n, k;
    scanf("%d%d", &n, &k);

    for (int i = 0; i < n; i ++ ) scanf("%d", &q[i]);

    cout << quick_sort(q, 0, n - 1, k) << endl;

    return 0;
}
```

##### **Python**

```python
#本题对python不太友好，很容易超时
def partition(l, r):
    import random
    x = random.randint(l, r)
    nums[x], nums[l] = nums[l], nums[x]
    pivot = nums[l]
    while l < r:
        while l < r and nums[r] > pivot:r -= 1
        nums[l] = nums[r]
        while l < r and nums[l] < pivot:l += 1
        nums[r] = nums[l]
    nums[l] = pivot  # 踩坑！
    return l

def quick_sort(l, r):
    if l < r:
        idx = partition(l, r)
        if idx != k - 1:
            if idx > k - 1:
                partition(l, idx - 1)
            elif idx < k - 1:
                partition(idx + 1, r)
    return nums[k - 1]
    

if __name__ == '__main__':
    n, k = map(int, input().split())
    nums = list(map(int, input().split()))
    print(quick_sort(0, n - 1))
```

<!-- tabs:end -->
</details>

<br>

归并排序（英语：merge sort）是一种采用了 [分治](basic/divide-and-conquer.md) 思想的排序算法。

## 归并排序

归并排序是一种稳定的排序算法。

归并排序的最优时间复杂度、平均时间复杂度和最坏时间复杂度均为 $O(n\log n)$。

归并排序的空间复杂度为 $O(n)$。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// C++ Version
void merge_sort(int q[], int l, int r) {
    if (l >= r) return;

    int mid = l + r >> 1;

    merge_sort(q, l, mid), merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];
    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}
```

##### **Python**

```python
# Python Version
def mergeSort(arr, l, r):
    if l < r:
        m = l + (r - l) // 2
        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        p1, p2, c = l, m + 1, []   # 踩坑： p1 = l
        while p1 <= m and p2 <= r:
            if arr[p1] <= arr[p2]:
                c.append(arr[p1])
                p1 += 1
            else:
                c.append(arr[p2])
                p2 += 1
        while p1 <= m:
            c.append(arr[p1])
            p1 += 1
        while p2 <= r:
            c.append(arr[p2])
            p2 += 1
        for i in range(len(c)):
            arr[l + i] = c[i]

if __name__=="__main__":
    n = int(input())
    nums = list(map(int, input().split()))
    mergeSort(nums, 0, n - 1)
    for i in nums:
        print(i, end=' ')
```

<!-- tabs:end -->
</details>

<br>

## 逆序对


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
LL merge_sort(int q[], int l, int r) {
    if (l >= r) return 0;

    int mid = l + r >> 1;

    LL res = merge_sort(q, l, mid) + merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else {
            res += mid - i + 1;
            tmp[k ++ ] = q[j ++ ];
        }
    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];

    return res;
}
```

### **Python**

```python
# Python Version
def merge_sort(l, r):
    global cnt 
    if l >= r:return
    #注意要点是全局变量在函数内部的声明, 如果直接在函数用局部变量cnt,递归的时候是不会传递引用的。
    m = l + (r - l) // 2
    merge_sort(l, m)
    merge_sort(m + 1, r)
    
    p1, p2, tmp = l, m + 1 , []
    while p1 <= m and p2 <= r:
        if arr[p1] <= arr[p2]:
            tmp.append(arr[p1])
            p1 += 1
        else:
            tmp.append(arr[p2])
            p2 += 1
            cnt += m - p1 + 1
    while p1 <= m:
        tmp.append(arr[p1])
        p1 += 1
    while p2 <= r:
        tmp.append(arr[p2])
        p2 += 1
    for i in range(len(tmp)):
        arr[l + i] = tmp[i]
```

<!-- tabs:end -->
</details>

<br>

另外，逆序对也可以用 [树状数组](ds/fenwick.md)、[线段树](ds/seg.md) 等数据结构求解。这三种方法的时间复杂度都是 $O(n \log n)$。


## 堆排序

> [!NOTE] **[AcWing 838. 堆排序](https://www.acwing.com/problem/content/840/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int n, m;
int h[N], cnt;

void down(int u) {
    int t = u;
    if (u * 2 <= cnt && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= cnt && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t) {
        swap(h[u], h[t]);
        down(t);
    }
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) cin >> h[i];
    
    cnt = n;
    
    for (int i = n / 2; i; -- i ) down(i);
    
    while (m -- ) {
        cout << h[1] << ' ';
        h[1] = h[cnt -- ];
        down(1);
    }
    cout << endl;
    return 0;
}
```

##### **Python**

```python
def down(i):
    min_idx = i
    if 2 * i < len(nums) and nums[2 * i] < nums[i]:
        min_idx = 2 * i
    if 2 * i + 1 < len(nums) and nums[2 * i + 1] < nums[min_idx]:
        min_idx = 2 * i + 1
    if min_idx != i:
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
        down(min_idx)


def pop():
    x = nums[1]
    nums[1] = nums[-1]
    nums.pop()
    down(i)
    return x


if __name__ == '__main__':
    n, m = map(int, input().split())
    nums = [0] + list(map(int, input().split()))

    for i in range(len(nums) // 2, 0, -1):
        down(i)
    res = []
    for j in range(m):
        res.append(pop())
    print(' '.join(map(str, res)))

```

<!-- tabs:end -->
</details>

<br>

* * *

## 基数排序 (部分高级数据结构的基础)

```cpp
inline void radix_sort(unsigned A[], int len)
{
    unsigned *B = new unsigned[len];
    int r = 65535, L[r + 1] = {0}, H[r + 1] = {0};
    for (int i = 0; i < len; ++i)
        L[A[i] & r]++, H[(A[i] >> 16) & r]++;   // 计数
    for (int i = 1; i <= r; ++i)
        L[i] += L[i - 1], H[i] += H[i - 1];     // 求前缀和
    for (int i = len - 1; i >= 0; --i)
        B[--L[A[i] & r]] = A[i];                // 对低位进行计数排序
    for (int i = len - 1; i >= 0; --i)
        A[--H[(B[i] >> 16) & r]] = B[i];        // 对高位进行计数排序
    delete[] B;
}
```

## 习题

### 快排 partitiong

> [!NOTE] **[LeetCode 215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> BFPRT

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    // k 保持不变 持续缩小范围
    int quick_sort(vector<int> & nums, int l, int r, int k) {
        if (l >= r) // ATTENTION
            return nums[k];
        int i = l - 1, j = r + 1, x = nums[l + r >> 1];
        while (i < j) {
            do i ++ ; while (nums[i] > x);  // ATTENTION >
            do j -- ; while (nums[j] < x);
            if (i < j)
                swap(nums[i], nums[j]);
        }
        if (k <= j)
            return quick_sort(nums, l, j, k);
        else
            return quick_sort(nums, j + 1, r, k);
    }

    int findKthLargest(vector<int>& nums, int k) {
        return quick_sort(nums, 0, nums.size() - 1, k - 1);
    }
};

class Solution {
public:
    int quick_sort(vector<int> & nums, int l, int r, int k) {
        if (l >= r)
            return nums[l];
        int i = l - 1, j = r + 1, x = nums[l + r >> 1];
        while (i < j) {
            do i ++ ; while (nums[i] > x);
            do j -- ; while (nums[j] < x);
            if (i < j)
                swap(nums[i], nums[j]);
        }
        if (j - l + 1 >= k)
            return quick_sort(nums, l, j, k);
        else
            return quick_sort(nums, j + 1, r, k - (j - l + 1));
    }

    int findKthLargest(vector<int>& nums, int k) {
        return quick_sort(nums, 0, nums.size() - 1, k);
    }
};
```

##### **C++ 废弃**

```cpp
class Solution {
public:
    int partition(vector<int>& nums, int l, int r) {
        int pivot = nums[l];
        while (l < r) {
            while (r > l && nums[r] >= pivot) -- r ;
            nums[l] = nums[r];
            while (r > l && nums[l] <= pivot) ++ l ;
            nums[r] = nums[l];
        }
        nums[l] = pivot;
        return l;
    }
    int findKthLargest(vector<int>& nums, int k) {
        int l = 0, r = nums.size() - 1;
        int tar = r - k + 1;
        while (l < r) {
            int idx = partition(nums, l, r);
            if (idx < tar) l = idx + 1;
            else r = idx;
        }
        return l < nums.size() ? nums[l] : -1;
    }
};
```

##### **Python**

```python
# 用堆: 时间复杂度O(n*logn)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        maxHeap = []
        for x in nums:
            heapq.heappush(maxHeap, -x)
        for _ in range(k - 1):
            heapq.heappop(maxHeap)
        return -maxHeap[0]



# 用快排 O(n)
class Solution:
    def findKthLargest(self, arr: List[int], k: int) -> int:
        def partition(l, r):
            x = random.randint(l, r)
            arr[x], arr[r] = arr[r], arr[x]
            less, more = l - 1, r 
            while l < more:
                if arr[l] < arr[r]:
                    less += 1
                    arr[l], arr[less] = arr[less], arr[l]
                    l += 1
                elif arr[l] > arr[r]:
                    more -= 1
                    arr[l], arr[more] = arr[more], arr[l]
                else:l += 1
            arr[more], arr[r] = arr[r], arr[more]
            return less+1 # 返回的这个位置，是这个数的位置一定是确定好的，比这个数小的都在左边，大于或者等于的都在右边

        n = len(arr)
        l, r = 0, n - 1
        while l < r:
            q = partition(l, r)
            if q < n - k:
                l = q + 1
            else:r = q 
        return arr[l]  # 跳出循环 推出的时候 一定是 l == r == n -k
```

<!-- tabs:end -->
</details>

<br>

* * *

### 三路排序

> [!NOTE] **[SwordOffer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode.cn/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)**
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
    void exchange(vector<int> &array) {
        int l = 0, r = array.size() - 1;
        while (l < r) {
            while (l < r && array[l] & 1)
                ++ l ;
            while (l < r && !(array[r] & 1))
                -- r ;
            if (l < r)
                swap(array[l], array[r]);
        }
    }
};
```

##### **Python**

```python
# python3
# 双指针法: l 指向开头， r 指向尾部。开始循环处理。
# 当左边的数为奇数时， l+=1；直到遇到一个偶数
# 当右边的数位偶数时，r-=1；直到遇到一个奇数
# 把这两个数交换，然后继续下一个循环。

class Solution(object):
    def reOrderArray(self, arr):
        l, r = 0, len(arr) -1
        while l < r :
            while l < r and arr[l] % 2 == 1:
                l += 1
            while l < r and arr[r] % 2 == 0:
                r -= 1
            arr[l], arr[r] = arr[r], arr[l]
        return arr
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 75. 颜色分类](https://leetcode.cn/problems/sort-colors/)**
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
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        // p <= c2 , cause [2,0,1]
        for (int c0 = 0, p = 0, c2 = n - 1; p <= c2; ) {
            if (nums[p] == 0)
                swap(nums[p ++ ], nums[c0 ++ ]);
            else if (nums[p] == 2)
                swap(nums[p], nums[c2 -- ]);
            else
                p ++ ;
        }
    }
};
```

##### **Python**

```python
# 荷兰国旗问题；快排partition部分的思想

# 左神的partition部分思想
class Solution:
    def sortColors(self, arr: List[int]) -> None:
        l, r = 0, len(arr) - 1
        less, more = l - 1 , r + 1
        while l < more:
            if arr[l] < 1:
                less += 1
                arr[less], arr[l] = arr[l], arr[less]
                l += 1
            elif arr[l] > 1:
                more -= 1
                arr[more], arr[l] = arr[l], arr[more]
            else:
                l += 1 



# 3个指针：保证[0, j-1]都是0； [j, i-1]都是1；[k+1,n-1]都是2；然后i 和 k两个指针逼近
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        i, j, k = 0, 0, len(nums) - 1
        while i <= k:
            if nums[i] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[k] = nums[k], nums[i]
                k -= 1
            else:i += 1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 324. 摆动排序 II](https://leetcode.cn/problems/wiggle-sort-ii/)**
> 
> 题意: 非常 trick

> [!TIP] **思路**
> 
> 我们先用快速选择算法求出中位数mid，C++中可以调用 nth_element()函数。
> 
> 将所有数分成三种：小于mid的数、等于mid的数和大于mid的数。 然后对数组排序，使得大于mid的数在最前面，等于mid的数在中间，小于mid的数在最后面。 这一步可以直接利用三数排序，三数排序算法可以参考LeetCode 75. Sort Colors。
> 
> 然后我们将排好序的数组重排，将前半段依次放到奇数位置上，将后半段依次放到偶数位置上。此时就会有： nums[0] < nums[1] > nums[2] < nums[3] ...
> 
> 这一步重排我们可以在三数排序时做，只需在排序时做一个数组下标映射即可： `i => (1 + 2 * i) % (n | 1)` 该映射可以将数组前一半映射到奇数位置上，数组后一半映射到偶数位置上。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    void wiggleSort(vector<int>& nums) {
        int n = nums.size();
        auto midptr = nums.begin() + n / 2;
        nth_element(nums.begin(), midptr, nums.end());
        int mid = *midptr;

        #define A(i) nums[(1 + 2 * i) % (n | 1)]

        int i = 0, j = 0, k = n - 1;
        while (j <= k)
            if (A(j) > mid) swap(A(i ++ ), A(j ++ ));
            else if (A(j) < mid) swap(A(j), A(k -- ));
            else j ++ ;
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

### 逆序对

> [!NOTE] **[AcWing 107. 超快速排序](https://www.acwing.com/problem/content/109/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标记本题：求冒泡排序的交换次数 也即【逆序对的数量】
> 
> 每次交换 如果交换相邻两个数 则恰好个数1

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 500010;

int n;
LL q[N], w[N];

LL merge_sort(int l, int r) {
    if (l == r) return 0;
    int mid = l + r >> 1;
    LL res = merge_sort(l, mid) + merge_sort(mid + 1, r);
    int i = l, j = mid + 1, k = 0;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) w[k ++ ] = q[i ++ ];
        else {
            res += mid + 1 - i;
            w[k ++ ] = q[j ++ ];
        }
    while (i <= mid) w[k ++ ] = q[i ++ ];
    while (j <= r) w[k ++ ] = q[j ++ ];
    for (int i = l, j = 0; i <= r; ++ i, ++ j ) q[i] = w[j];
    return res;
}

int main() {
    while (cin >> n, n) {
        for (int i = 0; i < n; ++ i ) cin >> q[i];
        cout << merge_sort(0, n - 1) << endl;
    }
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

> [!NOTE] **[LeetCode 2938. 区分黑球与白球](https://leetcode.cn/problems/separate-black-and-white-balls/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 冒泡排序计数 转化为 逆序对

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    
    string s, t;
    int n;
    
    LL merge_sort(int l, int r) {
        if (l == r)
            return 0;
        int mid = l + r >> 1;
        LL res = merge_sort(l, mid) + merge_sort(mid + 1, r);
        int i = l, j = mid + 1, k = 0;
        while (i <= mid && j <= r)
            if (s[i] <= s[j])           // 白色0在左侧
                t[k ++ ] = s[i ++ ];
            else {
                res += mid + 1 - i;
                t[k ++ ] = s[j ++ ];
            }
        while (i <= mid)
            t[k ++ ] = s[i ++ ];
        while (j <= r)
            t[k ++ ] = s[j ++ ];
        for (int i = l, j = 0; i <= r; ++ i , ++ j )
            s[i] = t[j];
        return res;
    }
    
    long long minimumSteps(string s) {
        this->s = s, this->n = s.size();
        this->t = s;
        return merge_sort(0, n - 1);
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

> [!NOTE] **[Luogu [NOIP2013 提高组] 火柴排队](https://www.luogu.com.cn/problem/P1966)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **经典** 推理转化为求逆序对

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10, MOD = 1e8 - 3;

int n;
int x[N], t[N];
struct Node {
    int x, id;
} a[N], b[N];

int merge(int l, int r) {
    if (l >= r)
        return 0;
    
    int m = l + r >> 1;
    int ret = (merge(l, m) + merge(m + 1, r)) % MOD;
    int i = l, j = m + 1, k = 0;
    while (i <= m && j <= r)
        if (x[i] <= x[j])
            t[k ++ ] = x[i ++ ];
        else
            ret = (ret + m + 1 - i) % MOD, t[k ++ ]= x[j ++ ];
    
    while (i <= m)
        t[k ++ ] = x[i ++ ];
    while (j <= r)
        t[k ++ ] = x[j ++ ];

    for (int i = l, j = 0; i <= r; ++ i , ++ j )
        x[i] = t[j];
    return ret;
}

int main() {
    cin >> n;
    for (int i = 0; i < n; ++ i )
        cin >> a[i].x, a[i].id = i;
    for (int i = 0; i < n; ++ i )
        cin >> b[i].x, b[i].id = i;
    
    auto cmp = [](const Node & a, const Node & b) -> bool {
        return a.x < b.x;
    };
    sort(a, a + n, cmp);
    sort(b, b + n, cmp);
    
    for (int i = 0; i < n; ++ i )
        x[a[i].id] = b[i].id;
    
    cout << merge(0, n - 1) << endl;
    
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

> [!NOTE] **[LeetCode 493. 翻转对](https://leetcode.cn/problems/reverse-pairs/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 逆序对变种 单独统计再归并

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> w;
    int merge_sort(vector<int> & nums, int l, int r) {
        if (l >= r) return 0;
        int m = l + (r - l) / 2;
        int ret = merge_sort(nums, l, m) + merge_sort(nums, m + 1, r);
        // 统计
        for (int i = l, j = m + 1; i <= m; ++ i ) {
            while (j <= r && nums[j] * 2ll < nums[i]) ++ j ;
            ret += j - (m + 1);
        }

        w.clear();
        int i = l, j = m + 1;
        while (i <= m && j <= r) {
            if (nums[i] <= nums[j]) w.push_back(nums[i ++ ]);
            else w.push_back(nums[j ++ ]);
        }
        while (i <= m) w.push_back(nums[i ++ ]);
        while (j <= r) w.push_back(nums[j ++ ]);
        for (int i = l, j = 0; j < w.size(); ++ i , ++ j ) nums[i] = w[j];
        return ret;
    }
    int reversePairs(vector<int>& nums) {
        return merge_sort(nums, 0, nums.size() - 1);
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

> [!NOTE] **[LeetCode 1850. 邻位交换的最小次数](https://leetcode.cn/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 转化为归并排序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int getMinSwaps(string num, int k) {
        string ori = num;
        while (k -- )
            next_permutation(num.begin(), num.end());
        
        int n = ori.size();
        vector<int> c(n);
        int cnt[10] = {0};
        for (int i = 0; i < n; ++ i ) {
            int x = ori[i] - '0';
            cnt[x] ++ ;
            // 找对应的该数的位置
            for (int j = 0, k = cnt[x]; j < n; j ++ )
                if (num[j] - '0' == x && -- k == 0) {
                    c[i] = j;
                    break;
                }
        }
        int res = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = i + 1; j < n; ++ j )
                if (c[i] > c[j])
                    res ++ ;
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

> [!NOTE] **[LeetCode 2193. 得到回文串的最少操作次数](https://leetcode.cn/problems/minimum-number-of-moves-to-make-palindrome/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂贪心 => 将原题转化为逆序对问题
> 
> 数据范围更大的同样题目: https://www.luogu.com.cn/problem/P5041
> 
> 关于回文串 & 逆序对的一些重要特性：ref **[Codeforces C. Palindrome Transformation](https://codeforces.com/problemset/problem/486/C)**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ TODO**

```cpp
class Solution {
public:
    int minMovesToMakePalindrome(string s) {
        int n = s.size();
        // 记录下标
        vector<int> p[26];
        for (int i = 0; i < n; ++ i )
            p[s[i] - 'a'].push_back(i);
        
        // 生成反串以及对应的在原串中的字符下标 ==> TODO
        // 原理 ref: 
        int cnt[26] = {};
        auto t = s; reverse(t.begin(), t.end());
        vector<int> ve(n);
        for (int i = 0; i < n; ++ i )
            ve[i] = p[t[i] - 'a'][cnt[t[i] - 'a'] ++ ];
        
        int res = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < i; ++ j )
                if (ve[j] > ve[i])
                    res ++ ;
        return res / 2; // ==> TODO
    }
};
```

##### **C++ 标准贪心**

```cpp
class Solution {
public:
    // 贪心的思想：
    // 每一轮对于最左侧的字符 α，找到最大的下标 j，满足 sj=α，将 sj 移动到最右侧，然后同时去掉最左侧和最右侧的字符。
    // 如果找不到 jj，由于题意保证可以得到回文串，则说明 alpha 在当前字符串中仅出现了一次，需要放到最中间的位置上。
    // TODO: 可信证明
    int minMovesToMakePalindrome(string s) {
        int n = s.size(), res = 0;
        for (int i = 0; i < n; ++ i ) {
            int p = -1;
            for (int j = n - 1; j > i; -- j )
                if (s[j] == s[i]) {
                    p = j;
                    break;
                }
            
            if (p == -1) {
                // 奇数个 移动到中间
                res += s.size() / 2 - i;
                // 右边界不变 直接continue
                continue;
            }
            
            for (int j = p; j < n - 1; ++ j )
                s[j] = s[j + 1], res ++ ;
            n -- ;
        }
        return res;
    }
};
```

##### **C++ 贪心 + bit优化**

```cpp
class Solution {
public:
    // 贪心的思想：
    // 每一轮对于最左侧的字符 α，找到最大的下标 j，满足 sj=α，将 sj 移动到最右侧，然后同时去掉最左侧和最右侧的字符。
    // 如果找不到 jj，由于题意保证可以得到回文串，则说明 alpha 在当前字符串中仅出现了一次，需要放到最中间的位置上。
    // TODO: 可信证明
    const static int N = 2e3 + 10;
    
    int tr[N];
    void init() {
        memset(tr, 0, sizeof tr);
    }
    int lowbit(int x) {
        return x & -x;
    }
    void add(int x, int c) {
        for (int i = x; i < N; i += lowbit(i))
            tr[i] += c;
    }
    int query(int x) {
        int ret = 0;
        for (int i = x; i; i -= lowbit(i))
            ret += tr[i];
        return ret;
    }
    
    int minMovesToMakePalindrome(string s) {
        int n = s.size(), res = 0;
        
        // 预处理每个字符的位置
        vector<deque<int>> p(26);
        for (int i = 0; i < n; ++ i ) {
            p[s[i] - 'a'].push_back(i);
            add(i + 1, 1);  // 方便统计某个区间有多少个数
        }
        
        // t 已删除的个数
        int t = 0, odd = -1;
        for (int i = 0; i < n; ++ i ) {
            int c = s[i] - 'a';
            if (p[c].empty())
                continue;
            
            if (p[c].size() == 1) {
                // 奇数个
                odd = i;
                p[c].pop_back();
                continue;
            }
            
            // 总数 - 已删除的数 - 减去前面挖空的位置 ==> 后面数字的个数
            res += n - t - query(p[c].back() + 1);
            
            add(p[c].back() + 1, -1);   // 挖空当前
            p[c].pop_back(); p[c].pop_front();
            t ++ ;
        }
        if (odd != -1)
            res += n / 2 - query(odd);
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

### 桶排序

> [!NOTE] **[LeetCode 164. 最大间距](https://leetcode.cn/problems/maximum-gap/)**
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
// yxc
class Solution {
public:
    struct Range {
        int min, max;
        bool used;
        Range() : min(INT_MAX), max(INT_MIN), used(false){}
    };

    int maximumGap(vector<int>& nums) {
        int n = nums.size();
        int Min = INT_MAX, Max = INT_MIN;
        for (auto x: nums) {
            Min = min(Min, x);
            Max = max(Max, x);
        }
        if (n < 2 || Min == Max) return 0;
        vector<Range> r(n - 1);
        int len = (Max - Min + n - 2) / (n - 1);
        for (auto x: nums) {
            if (x == Min) continue;
            int k = (x - Min - 1) / len;
            r[k].used = true;
            r[k].min = min(r[k].min, x);
            r[k].max = max(r[k].max, x);
        }
        int res = 0;
        for (int i = 0, last = Min; i < n - 1; i ++ )
            if (r[i].used) {
                res = max(res, r[i].min - last);
                last = r[i].max;
            }
        return res;
    }
};
```

##### **C++ 2**

```cpp

class Solution {
public:
    class Bucket {
    public:
        bool used = false;
        int minval = numeric_limits<int>::max();        // same as INT_MAX
        int maxval = numeric_limits<int>::min();        // same as INT_MIN
    };

    int maximumGap(vector<int>& nums) {
        if (nums.empty() || nums.size() < 2)
            return 0;

        int mini = *min_element(nums.begin(), nums.end()),
            maxi = *max_element(nums.begin(), nums.end());

        int bucketSize = max(1, (maxi - mini) / ((int)nums.size() - 1));        // bucket size or capacity
        int bucketNum = (maxi - mini) / bucketSize + 1;                         // number of buckets
        vector<Bucket> buckets(bucketNum);

        for (auto&& num : nums) {
            int bucketIdx = (num - mini) / bucketSize;                          // locating correct bucket
            buckets[bucketIdx].used = true;
            buckets[bucketIdx].minval = min(num, buckets[bucketIdx].minval);
            buckets[bucketIdx].maxval = max(num, buckets[bucketIdx].maxval);
        }

        int prevBucketMax = mini, maxGap = 0;
        for (auto&& bucket : buckets) {
            if (!bucket.used)
                continue;

            maxGap = max(maxGap, bucket.minval - prevBucketMax);
            prevBucketMax = bucket.maxval;
        }

        return maxGap;
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