
在学习单调队列前，让我们先来看一道例题。

## 例题

[Sliding Window](http://poj.org/problem?id=2823)

本题大意是给出一个长度为 $n$ 的数组，编程输出每 $k$ 个连续的数中的最大值和最小值。

最暴力的想法很简单，对于每一段 $i \sim i+k-1$ 的序列，逐个比较来找出最大值（和最小值），时间复杂度约为 $O(n \times k)$。

很显然，这其中进行了大量重复工作，除了开头 $k-1$ 个和结尾 $k-1$ 个数之外，每个数都进行了 $k$ 次比较，而题中 $100\%$ 的数据为 $n \le 1000000$，当 $k$ 稍大的情况下，显然会 TLE。

这时所用到的就是单调队列了。

## 概念

顾名思义，单调队列的重点分为 **单调" 和 **队列**

"单调" 指的是元素的的 **规律"——递增（或递减）

"队列" 指的是元素只能从队头和队尾进行操作

Ps. 单调队列中的 **队列" 与正常的队列有一定的区别，稍后会提到

## 例题分析

有了上面 **单调队列" 的概念，很容易想到用单调队列进行优化。

要求的是每连续的 $k$ 个数中的最大（最小）值，很明显，当一个数进入所要 **寻找" 最大值的范围中时，若这个数比其前面（先进队）的数要大，显然，前面的数会比这个数先出队且不再可能是最大值。

也就是说——当满足以上条件时，可将前面的数 **弹出"，再将该数真正 push 进队尾。

这就相当于维护了一个递减的队列，符合单调队列的定义，减少了重复的比较次数，不仅如此，由于维护出的队伍是查询范围内的且是递减的，队头必定是该查询区域内的最大值，因此输出时只需输出队头即可。

显而易见的是，在这样的算法中，每个数只要进队与出队各一次，因此时间复杂度被降到了 $O(N)$。

而由于查询区间长度是固定的，超出查询空间的值再大也不能输出，因此还需要 site 数组记录第 $i$ 个队中的数在原数组中的位置，以弹出越界的队头。

例如我们构造一个单调递增的队列会如下：

原序列为：

```text
1 3 -1 -3 5 3 6 7
```

因为我们始终要维护队列保证其 **递增** 的特点，所以会有如下的事情发生：

| 操作                              | 队列状态      |
| ------------------------------- | --------- |
| 1 入队                            | `{1}`     |
| 3 比 1 大，3 入队                    | `{1 3}`   |
| -1 比队列中所有元素小，所以清空队列 -1 入队       | `{-1}`    |
| -3 比队列中所有元素小，所以清空队列 -3 入队       | `{-3}`    |
| 5 比 -3 大，直接入队                   | `{-3 5}`  |
| 3 比 5 小，5 出队，3 入队               | `{-3 3}`  |
| -3 已经在窗体外，所以 -3 出队；6 比 3 大，6 入队 | `{3 6}`   |
| 7 比 6 大，7 入队                    | `{3 6 7}` |

> [!NOTE] **例题参考代码**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

Ps. 此处的 **队列" 跟普通队列的一大不同就在于可以从队尾进行操作，STL 中有类似的数据结构 deque。

> [!NOTE] **例题 2 [Luogu P2698 Flowerpot S ](https://www.luogu.com.cn/problem/P2698)**
> 
> 给出 $N$ 滴水的坐标，$y$ 表示水滴的高度，$x$ 表示它下落到 $x$ 轴的位置。每滴水以每秒 1 个单位长度的速度下落。你需要把花盆放在 $x$ 轴上的某个位置，使得从被花盆接着的第 1 滴水开始，到被花盆接着的最后 1 滴水结束，之间的时间差至少为 $D$。
> 
> 我们认为，只要水滴落到 $x$ 轴上，与花盆的边沿对齐，就认为被接住。给出 $N$ 滴水的坐标和 $D$ 的大小，请算出最小的花盆的宽度 $W$。$1\leq N \leq 100000 , 1 \leq D \leq 1000000, 0 \leq x,y\leq 10^6$

将所有水滴按照 $x$ 坐标排序之后，题意可以转化为求一个 $x$ 坐标差最小的区间使得这个区间内 $y$ 坐标的最大值和最小值之差至少为 $D$。我们发现这道题和上一道例题有相似之处，就是都与一个区间内的最大值最小值有关，但是这道题区间的大小不确定，而且区间大小本身还是我们要求的答案。

我们依然可以使用一个递增，一个递减两个单调队列在 $R$ 不断后移时维护 $[L,R]$ 内的最大值和最小值，不过此时我们发现，如果 $L$ 固定，那么 $[L,R]$ 内的最大值只会越来越大，最小值只会越来越小，所以设 $f(R) = \max[L,R]-\min[L,R]$，则 $f(R)$ 是个关于 $R$ 的递增函数，故 $f(R)\geq D  \Rightarrow f(r)\geq D,R\lt r \leq N$。这说明对于每个固定的 $L$，向右第一个满足条件的 $R$ 就是最优答案。
所以我们整体求解的过程就是，先固定 $L$，从前往后移动 $R$，使用两个单调队列维护 $[L,R]$ 的最值。当找到了第一个满足条件的 $R$，就更新答案并将 $L$ 也向后移动。随着 $L$ 向后移动，两个单调队列都需及时弹出队头。这样，直到 $R$ 移到最后，每个元素依然是各进出队列一次，保证了 $O(n)$ 的时间复杂度。



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>


## 习题

### 一般单调队列

> [!NOTE] **[AcWing 154. 滑动窗口](https://www.acwing.com/problem/content/156/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1000010;

int a[N];
int q1[N], q2[N];
int f1[N], f2[N];

int main() {
    int n, k;
    cin >> n >> k;
    for (int i = 0; i < n; ++ i ) cin >> a[i];
    
    
    int hh1 = 0, tt1 = -1, hh2 = 0, tt2 = -1;
    for (int i = 0; i < n; ++ i ) {
        if (hh1 <= tt1 && q1[hh1] <= i - k) ++ hh1 ;
        if (hh2 <= tt2 && q2[hh2] <= i - k) ++ hh2 ;
        
        while (hh1 <= tt1 && a[q1[tt1]] > a[i]) -- tt1 ;
        while (hh2 <= tt2 && a[q2[tt2]] < a[i]) -- tt2 ;
        
        q1[ ++ tt1] = i;
        q2[ ++ tt2] = i;
        
        f1[i] = a[q1[hh1]];
        f2[i] = a[q2[hh2]];
    }
    for (int i = k - 1; i < n; ++ i ) cout << f1[i] << ' ';
    cout << endl;
    for (int i = k - 1; i < n; ++ i ) cout << f2[i] << ' ';
    cout << endl;
    return 0;
}
```

##### **Python**

```python
"""
1. 维护一个最大/最小队列
2. 一般先用暴力做，有了思路后，再用单调队列进行优化
3. 一般需要求区间最值问题，就可以用单调队列进行优化（单调最小/大队列）

注意：
和单调栈不同，单调栈主要用于记录左边/右边第一个比它大/小的数值。

"""

#暴力做法
# n = len(nums)
# res = []
# for r in range(n):
#     max_v = float('-inf')
#     start = max(0, r - k + 1)
#     for l in range(start, r + 1):
#         max_v = max(nums[l],max_v)
#     if r >= k-1:
#         res.append(max_v)


import collections
if __name__ == '__main__':
    n, k = map(int, input().split())
    nums = list(map(int, input().split()))
    
    q1, res1 = collections.deque(), []  #队列里存的是下标，方便定位找到元素
    for r in range(n):
        if q1 and q1[0] <= r - k:
            q1.popleft()
        while q1 and nums[q1[-1]] > nums[r]:
            q1.pop()
        q1.append(r)
        
        if r >= k - 1:
            res1.append(nums[q1[0]])
        
    for i in range(len(res1)):
        print(res1[i], end = ' ')
    print()
    
    q2, res2 = collections.deque(), [] 
    for r in range(n):
        if q2 and q2[0] <= r - k:
            q2.popleft()
        while q2 and nums[q2[-1]] < nums[r]:
            q2.pop()
        q2.append(r)
        
        if r >= k - 1:
            res2.append(nums[q2[0]])
            
    for i in range(len(res2)):
        print(res2[i], end = ' ')
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1499. 满足不等式的最大值](https://leetcode-cn.com/problems/max-value-of-equation/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 动态维护单调队列

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findMaxValueOfEquation(vector<vector<int>>& points, int k) {
        // 动态维护队列的情况下 (对于i<j)所求yi + yj + |xi - xj| 即 yi+yj+xj-xi
        // 所以在满足 xj-xi <= k的情况下找最大的yi-xi pair->first yi-xi
        // pair->second xi
        priority_queue<pair<int, int>> pq;
        int res = -1e9;
        for (auto v : points) {
            int x = v[0], y = v[1];
            while (!pq.empty() && pq.top().second < x - k) pq.pop();
            if (!pq.empty()) res = max(res, pq.top().first + y + x);
            pq.push({y - x, x});
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

### 结合前后缀分解

> [!NOTE] **[LeetCode 1888. 使二进制字符串字符交替的最少反转次数](https://leetcode-cn.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/)**
> 
> [weekly-244](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-06-06_Weekly-244) 详细看左侧链接
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可以滑动窗口也可以前后缀分解 更优的做法是滑动窗口

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int minFlips(string s) {
        int n = s.size(), cnt = 0;
        // 将字符串变成 01 串需要反转的次数
        string tar = "01";
        for (int i = 0; i < n; ++ i )
            cnt += (s[i] != tar[i % 2]);
        
        int res = min(cnt, n - cnt);
        s += s;
        for (int i = 0; i < n; ++ i ) {
            cnt -= (s[i] != tar[i % 2]);
            cnt += (s[i + n] != tar[(i + n) % 2]);
            res = min(res, min(cnt, n - cnt));
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        s = s + s;
        
        string a, b;    // a = "0101...", b = "1010..."
        for (int i = 0; i < 2 * n; ++ i )
            a.push_back('0' + i % 2), b.push_back('0' + (i + 1) % 2);
        
        int res = n, da = 0, db = 0;
        for (int i = 0; i < 2 * n; ++ i ) {
            if (s[i] != a[i])
                da ++ ;
            if (s[i] != b[i])
                db ++ ;
            
            // 维护窗口的实现
            if (i >= n) {
                if (s[i - n] != a[i - n])
                    da -- ;
                if (s[i - n] != b[i - n])
                    db -- ;
            }
            if (i >= n - 1)
                res = min(res, min(da, db));
        }
        return res;
    }
};
```

##### **C++ 3 常见形式**

```cpp
// 转化为常见方式
class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        s = s + s;
        
        string a, b;    // a = "0101...", b = "1010..."
        for (int i = 0; i < 2 * n; ++ i )
            a.push_back('0' + i % 2), b.push_back('0' + (i + 1) % 2);
        
        int res = n, da = 0, db = 0;
        for (int i = 0; i < n; ++ i ) {
            if (s[i] != a[i])
                da ++ ;
            if (s[i] != b[i])
                db ++ ;
        }
        
        res = min(res, min(da, db));
        
        for (int i = n; i < 2 * n; ++ i ) {
            if (s[i] != a[i])
                da ++ ;
            if (s[i] != b[i])
                db ++ ;
            
            {
                if (s[i - n] != a[i - n])
                    da -- ;
                if (s[i - n] != b[i - n])
                    db -- ;
            }
            
            res = min(res, min(da, db));
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