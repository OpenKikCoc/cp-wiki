

堆是一棵树，其每个节点都有一个键值，且每个节点的键值都大于等于/小于等于其父亲的键值。

每个节点的键值都大于等于其父亲键值的堆叫做小根堆，否则叫做大根堆。[STL 中的 `priority_queue`](/lang/csl/container-adapter/#_13) 其实就是一个大根堆。

（小根）堆主要支持的操作有：插入一个数、查询最小值、删除最小值、合并两个堆、减小一个元素的值。

一些功能强大的堆（可并堆）还能（高效地）支持 merge 等操作。

一些功能更强大的堆还支持可持久化，也就是对任意历史版本进行查询或者操作，产生新的版本。

## 堆的分类

|         操作\\数据结构        |                                  配对堆                                  |      二叉堆     |      左偏树     |      二项堆     |    斐波那契堆    |
| :---------------------: | :-------------------------------------------------------------------: | :----------: | :----------: | :----------: | :---------: |
|        插入（insert）       |                                 $O(1)$                                |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$    |    $O(1)$   |
|     查询最小值（find-min）     |                                 $O(1)$                                |    $O(1)$    |    $O(1)$    |  $O(\log n)$ |    $O(1)$   |
|    删除最小值（delete-min）    |                              $O(\log n)$                              |  $O(\log n)$ |  $O(\log n)$ |  $O(\log n)$ | $O(\log n)$ |
|        合并 (merge)       |                                 $O(1)$                                |    $O(n)$    |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$   |
| 减小一个元素的值 (decrease-key) | $o(\log n)$（下界 $\Omega(\log \log n)$，上界 $O(2^{2\sqrt{\log \log n}})$) |  $O(\log n)$ |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$   |
|         是否支持可持久化        |                                $\times$                               | $\checkmark$ | $\checkmark$ | $\checkmark$ |   $\times$  |

习惯上，不加限定提到“堆”时往往都指二叉堆。

> [!NOTE] **[AcWing 839. 模拟堆](https://www.acwing.com/problem/content/841/)**
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

int h[N], ph[N], hp[N], cnt;

void heap_swap(int a, int b) {
    swap(ph[hp[a]], ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void down(int u) {
    int t = u;
    if (u * 2 <= cnt && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= cnt && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t) {
        heap_swap(u, t);
        down(t);
    }
}

void up(int u) {
    while (u / 2 && h[u] < h[u / 2]) {
        heap_swap(u, u / 2);
        u >>= 1;
    }
}

int main() {
    int n, m = 0;
    cin >> n;
    while (n -- ) {
        char op[5];
        int k, x;
        cin >> op;
        if (op[0] == 'I') {
            cin >> x;
            cnt ++ ;
            m ++ ;
            ph[m] = cnt, hp[cnt] = m;
            h[cnt] = x;
            up(cnt);
        } else if (op[0] == 'P' && op[1] == 'M') cout << h[1] << endl;
        else if (op[0] == 'D' && op[1] == 'M') {
            heap_swap(1, cnt);
            cnt -- ;
            down(1);
        } else if (op[0] == 'D') {
            cin >> k;
            k = ph[k];
            heap_swap(k, cnt);
            cnt -- ;
            up(k);
            down(k);
        } else {
            cin >> k >> x;
            k = ph[k];
            h[k] = x;
            up(k);
            down(k);
        }
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

## 习题

### heap

> [!NOTE] **[LeetCode 373. 查找和最小的K对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)**
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
    typedef tuple<int, int, int> tpi3;
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        if (nums1.empty() || nums2.empty()) return {};
        int n = nums1.size(), m = nums2.size();
        priority_queue<tpi3, vector<tpi3>, greater<tpi3>> heap;
        for (int i = 0; i < m; ++ i )
            heap.push({nums2[i] + nums1[0], 0, i});
        vector<vector<int>> res;
        while (k -- && heap.size()) {
            auto [v, p1, p2] = heap.top(); heap.pop();
            res.push_back({nums1[p1], nums2[p2]});
            if (p1 + 1 < n)
                heap.push({nums1[p1 + 1] + nums2[p2], p1 + 1, p2});
        }
        return res;
    }
};

// tuple 也可以直接用 vector
// yxc
class Solution {
public:
    typedef vector<int> VI;
    vector<vector<int>> kSmallestPairs(vector<int>& a, vector<int>& b, int k) {
        if (a.empty() || b.empty()) return {};
        int n = a.size(), m = b.size();
        priority_queue<VI, vector<VI>, greater<VI>> heap;
        for (int i = 0; i < m; i ++ ) heap.push({b[i] + a[0], 0, i});
        vector<VI> res;
        while (k -- && heap.size()) {
            auto t = heap.top();
            heap.pop();
            res.push_back({a[t[1]], b[t[2]]});
            if (t[1] + 1 < n)
                heap.push({a[t[1] + 1] + b[t[2]], t[1] + 1, t[2]});
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

> [!NOTE] **[LeetCode 658. 找到 K 个最接近的元素](https://leetcode-cn.com/problems/find-k-closest-elements/)**
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
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        priority_queue<pair<int, int>> heap;
        for (auto v: arr) {
            heap.push({abs(x - v), v});
            if (heap.size() > k) heap.pop();
        }
        vector<int> res;
        while (heap.size()) res.push_back(heap.top().second), heap.pop();
        sort(res.begin(), res.end());
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

> [!NOTE] **[LeetCode 692. 前K个高频单词](https://leetcode-cn.com/problems/top-k-frequent-words/)**
> 
> 题意: 尝试以 $O(n log k)$ 时间复杂度和 $O(n)$ 空间复杂度解决。

> [!TIP] **思路**
> 
> 原地建堆 复杂度要求

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PIS = pair<int, string>;
    
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> cnt;
        for (auto& w: words) cnt[w] ++ ;

        vector<PIS> ws;
        for (auto [k, v]: cnt) ws.push_back({v, k});
        auto cmp = [](PIS a, PIS b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second > b.second;
        };
        make_heap(ws.begin(), ws.end(), cmp);
        
        vector<string> res;
        while (k -- ) {
            res.push_back(ws[0].second);
            pop_heap(ws.begin(), ws.end(), cmp);
            ws.pop_back();
        }

        return res;
    }
};
```

##### **Python**

```python
# 1. 用哈希表统计每个单词出现的次数； 2. 原地建堆，取前k个；
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # my_cnt = collections.Counter(words) 也可以直接用计数器 进行计数。 
        my_dict = collections.defaultdict(int)
        for c in words:
            my_dict[c] += 1 
        
        q = []
        for key, val in my_dict.items():
            heapq.heappush(q, (-val, key))  #  堆的元素可以是元组/列表 类型; 小根堆 所以 取负数
        res = []
        for _ in range(k):
            res.append(heapq.heappop(q)[1])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

### multiset

> [!NOTE] **[LeetCode 218. 天际线问题](https://leetcode-cn.com/problems/the-skyline-problem/)**
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
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
        vector<pair<int, int>> h;
        multiset<int> m;
        vector<vector<int>> res;
        for (auto &b : buildings) {
            h.push_back({b[0], -b[2]});
            h.push_back({b[1], b[2]});
        }
        sort(h.begin(), h.end());
        m.insert(0);
        int prev = 0;
        for (auto i : h) {
            if (i.second < 0)
                m.insert(-i.second);
            else
                m.erase(m.find(i.second));
            int cur = *m.rbegin();
            if (cur != prev) {
                res.push_back({i.first, cur});
                prev = cur;
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

> [!NOTE] **[LeetCode 480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)**
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
    int k;
    // r.size() >= l.size()
    multiset<int> left, right;

    double get_medium() {
        if (k % 2) return *right.begin();
        // 这里写double 处理整数溢出
        return ((double)*left.rbegin() + *right.begin()) / 2;
    }
    vector<double> medianSlidingWindow(vector<int>& nums, int _k) {
        k = _k;
        for (int i = 0; i < k; ++ i ) right.insert(nums[i]);
        // 较小的一半压入left
        for (int i = 0; i < k / 2; ++ i ) {
            left.insert(*right.begin());
            right.erase(right.begin());
        }
        vector<double> res;
        res.push_back(get_medium());

        for (int i = k; i < nums.size(); ++ i ) {
            int x = nums[i], y = nums[i - k];

            if (x >= *right.begin()) right.insert(x);
            else left.insert(x);
            if (y >= *right.begin()) right.erase(right.find(y));
            else left.erase(left.find(y));

            while (left.size() > right.size()) {
                right.insert(*left.rbegin());
                // rbegin 必须写find
                left.erase(left.find(*left.rbegin()));
            }
            while (left.size() + 1 < right.size()) {
                left.insert(*right.begin());
                right.erase(right.begin());
            }
            res.push_back(get_medium());
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