**排序算法**（英语：Sorting algorithm）是一种将一组特定的数据按某种顺序进行排列的算法。排序算法多种多样，性质也大多不同。

> [!TIP] **稳定性**
> 
> 基数排序、计数排序、插入排序、冒泡排序、归并排序是稳定排序。
> 
> 选择排序、堆排序、快速排序不是稳定排序。


> [!TIP] **时间复杂度**
> 
> 基于比较的排序算法的时间复杂度下限是 $O(n\log n)$ 的。
> 
> 当然也有不是 $O(n\log n)$ 的。例如，`计数排序` 的时间复杂度是 $O(n+w)$，其中 $w$ 代表输入数据的值域大小。
> 
> 以下是几种排序算法的比较。

![几种排序算法的比较](./images/sort-intro-1.apng)

> [!NOTE] **[LeetCode 912. 排序数组](https://leetcode.cn/problems/sort-an-array)**
> 
> 题意: 输入整数数组，实现升序排序

> [!TIP] **思路**
> 
> 标准实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 快排**

```cpp
class Solution {
public:
    vector<int> ns;

    void quick_sort(int l, int r) {
        if (l >= r)
            return;
        int i = l - 1, j = r + 1, x = ns[l + r >> 1];
        while (i < j) {
            do i ++ ; while (ns[i] < x);
            do j -- ; while (ns[j] > x);
            if (i < j)
                swap(ns[i], ns[j]);
        }
        quick_sort(l, j);
        quick_sort(j + 1, r);
    }

    vector<int> sortArray(vector<int>& nums) {
        this->ns = nums;
        quick_sort(0, ns.size() - 1);
        return ns;
    }
};
```

##### **C++ 归并**

```cpp
class Solution {
public:
    const static int N = 5e4 + 10;

    vector<int> ns;
    int tmp[N];

    void merge_sort(int l, int r) {
        if (l >= r)
            return;
        int mid = l + r >> 1;
        merge_sort(l, mid), merge_sort(mid + 1, r);

        int k = 0, i = l, j = mid + 1;
        while (i <= mid && j <= r) {
            if (ns[i] <= ns[j])
                tmp[k ++ ] = ns[i ++ ];
            else
                tmp[k ++ ] = ns[j ++ ];
        }

        while (i <= mid)
            tmp[k ++ ] = ns[i ++ ];
        while (j <= r)
            tmp[k ++ ] = ns[j ++ ];
        
        for (int i = l, j = 0; i <= r; ++ i, ++ j )
            ns[i] = tmp[j];
    }

    vector<int> sortArray(vector<int>& nums) {
        this->ns = nums;
        merge_sort(0, ns.size() - 1);
        return ns;
    }
};
```

##### **C++ 堆排**

```cpp
class Solution {
public:
    vector<int> ns;
    int cnt;

    void heapify(int u) {
        int t = u;
        // 1. 下标需要恢复 -1
        // 2. 不需要从小到大输出 反而需要从大到小整理(大的放末尾并收缩) 故使用大顶堆
        /*
        if (u * 2 <= cnt && ns[u * 2 - 1] > ns[t - 1])
            t = u * 2;
        if (u * 2 + 1 <= cnt && ns[u * 2 + 1 - 1] > ns[t - 1])
            t = u * 2 + 1;
        */
        int lc = u << 1, rc = u << 1 | 1;
        if (lc <= cnt && ns[lc - 1] > ns[t - 1])
            t = lc;
        if (rc <= cnt && ns[rc - 1] > ns[t - 1])
            t = rc;
        if (u != t) {
            swap(ns[u - 1], ns[t - 1]);
            heapify(t);
        }
    }

    void heap_sort(int n) {
        this->cnt = n;

        // 1. 建堆
        // 数组下标从0开始 故使用 (n-2)/2 与 i>=0
        // for (int i = (n - 1) / 2; i >= 0; -- i )
        //
        // 实际上，下标从 1 起始会更好写
        for (int i = n / 2; i; -- i )
            heapify(i);
        
        // 2. 排序
        // 当前堆顶已经是最大值 考虑挨个swap
        for (int i = n; i >= 2; -- i ) {
            swap(ns[i - 1], ns[1 - 1]);
            cnt -- ;
            heapify(1);
        }
    }

    vector<int> sortArray(vector<int>& nums) {
        this->ns = nums;
        heap_sort(ns.size());
        return ns;
    }
};
```

##### **C++ 冒泡 + 选择 + 插入**

```cpp
class Solution {
public:
    void bubbleSort(vector<int> & nums) {
        // 1. p 定位待放置的元素
        // 2. i 从头到位扫描并迁移元素
        for (int p = nums.size() - 1; p; -- p )
            for (int i = 1; i <= p; ++ i )
                if (nums[i - 1] > nums[i])
                    swap(nums[i - 1], nums[i]);
    }

    vector<int> sortArray(vector<int>& nums) {
        bubbleSort(nums);
        return nums;
    }
};
```

##### **C++ 希尔**

```cpp
// 希尔排序可以看作是一个冒泡排序或者插入排序的变形
//
// 在每次的排序的时候都把数组拆分成若干个序列，一个序列的相邻的元素索引相隔的固定的距离gap，每一轮对这些序列进行冒泡或者插入排序，然后再缩小gap得到新的序列一一排序，直到gap为1
```

##### **C++ 计数**

```cpp
class Solution {
public:
    void countSort(vector<int> & nums) {
        int minv = INT_MAX, maxv = INT_MIN;
        for (auto x : nums)
            minv = min(minv, x), maxv = max(maxv, x);
        
        vector<int> count(maxv - minv + 1);
        for (auto x : nums)
            count[x - minv] ++ ;
        
        for (int i = 0, j = 0; i < count.size(); ++ i )
            for (int z = 0; z < count[i]; ++ z )
                nums[j ++ ] = i + minv;
    }

    vector<int> sortArray(vector<int>& nums) {
        countSort(nums);
        return nums;
    }
};
```

##### **C++ 桶排序**

```cpp
// 先将所有元素分桶，在单个桶内执行其他排序算法
```

##### **C++ 基数**

```cpp
// 类似桶排，只是分类规则是按照数位从低到高，对于每一个具体数位执行一次操作后在当前数位即有序
// 迭代其他数位，同时保持桶中已有的顺序即可
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *