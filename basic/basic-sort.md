
## 快速排序

快速排序（英语：Quicksort），又称分区交换排序（英语：partition-exchange sort），简称快排，是一种被广泛运用的排序算法。

### 原理

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

#### 内省排序[^ref4]

内省排序（英语：Introsort 或 Introspective sort）是快速排序和 [堆排序](./heap-sort.md) 的结合，由 David Musser 于 1997 年发明。内省排序其实是对快速排序的一种优化，保证了最差时间复杂度为 $O(n\log n)$。

内省排序将快速排序的最大递归深度限制为 $\lfloor \log_2n \rfloor$，超过限制时就转换为堆排序。这样既保留了快速排序内存访问的局部性，又可以防止快速排序在某些情况下性能退化为 $O(n^2)$。

从 2000 年 6 月起，SGI C++ STL 的 `stl_algo.h` 中 `sort()` 函数的实现采用了内省排序算法。

## 线性找第 k 大的数

可以证明，在期望意义下，程序的时间复杂度为 $O(n)$。

<!-- tabs:start -->

##### **cpp**

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

##### **python**

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

归并排序（英语：merge sort）是一种采用了 [分治](./divide-and-conquer.md) 思想的排序算法。

## 归并排序

归并排序是一种稳定的排序算法。

归并排序的最优时间复杂度、平均时间复杂度和最坏时间复杂度均为 $O(n\log n)$。

归并排序的空间复杂度为 $O(n)$。

### 代码实现

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

## 逆序对


<!-- tabs:start -->

##### **C++**

```c++
LL merge_sort(int q[], int l, int r)
{
    if (l >= r) return 0;

    int mid = l + r >> 1;

    LL res = merge_sort(q, l, mid) + merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else
        {
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

另外，逆序对也可以用 [树状数组](ds/fenwick.md)、[线段树](ds/seg.md) 等数据结构求解。这三种方法的时间复杂度都是 $O(n \log n)$。
