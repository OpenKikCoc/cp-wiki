## 何为单调栈

顾名思义，单调栈即满足单调性的栈结构。与单调队列相比，其只在一端进行进出。

为了描述方便，以下举例及伪代码以维护一个整数的单调递增栈为例。

## 如何使用单调栈

### 插入

将一个元素插入单调栈时，为了维护栈的单调性，需要在保证将该元素插入到栈顶后整个栈满足单调性的前提下弹出最少的元素。

例如，栈中自顶向下的元素为 $\{0,11,45,81\}$。

![](images/monotonous-stack-before.svg)

插入元素 $14$ 时为了保证单调性需要依次弹出元素 $0,11$，操作后栈变为 $\{14,45,81\}$。

![](images/monotonous-stack-after.svg)

用伪代码描述如下：

```text
insert x
while !sta.empty() && sta.top()<x
    sta.pop()
sta.push(x)
```

### 使用

自然就是从栈顶读出来一个元素，该元素满足单调性的某一端。

例如举例中取出的即栈中的最小值。

## 应用

> [!NOTE] **[POJ3250 Bad Hair Day](http://poj.org/problem?id=3250)**
> 
> 有 $N$ 头牛从左到右排成一排，每头牛有一个高度 $h_i$，设左数第 $i$ 头牛与「它右边第一头高度 $≥h_i$」的牛之间有 $c_i$ 头牛，试求 $\sum_{i=1}^{N} c_i$。

比较基础的应用有这一题，就是个单调栈的简单应用，记录每头牛被弹出的位置，如果没有被弹出过则为最远端，稍微处理一下即可计算出题目所需结果。

另外，单调栈也可以用于离线解决 RMQ 问题。

我们可以把所有询问按右端点排序，然后每次在序列上从左往右扫描到当前询问的右端点处，并把扫描到的元素插入到单调栈中。这样，每次回答询问时，单调栈中存储的值都是位置 $\le r$ 的、可能成为答案的决策点，并且这些元素满足单调性质。此时，单调栈上第一个位置 $\ge l$ 的元素就是当前询问的答案，这个过程可以用二分查找实现。使用单调栈解决 RMQ 问题的时间复杂度为 $O(q\log q + q\log n)$，空间复杂度为 $O(n)$。


## 习题

### 一般单调栈

> [!NOTE] **[AcWing 830. 单调栈](https://www.acwing.com/problem/content/832/)**
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

int a[N];
int st[N], top = 0;
int res[N];

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; ++ i ) cin >> a[i];
    
    memset(res, -1, sizeof res);
    for (int i = n - 1; i >= 0; -- i ) {
        while (top && a[st[top - 1]] > a[i]) {
            res[st[top - 1]] = a[i];
            top -- ;
        }
        st[top ++ ] = i;
    }
    for (int i = 0; i < n; ++ i ) cout << res[i] << ' ';
    cout << endl;
    return 0;
}
```

##### **Python**

```python
"""
逆序遍历，从 左 到 右 维护一个【单调递增栈】
当前数 比 栈顶元素小时，那么当前数 就是栈顶元素就是 当前数的左边第一个比它小的数
"""
if __name__ == '__main__':
    n = int(input())
    res = [-1] * n
    nums = list(map(int, input().split()))
    stack = []

    for i in range(n - 1, -1, -1):
        while stack and nums[stack[-1]] > nums[i]:
            res[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)
    for i in range(len(res)):
        print(res[i], end = ' ')

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1413. 矩形牛棚](https://www.acwing.com/problem/content/1415/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 单调栈
> 
> 最大矩形

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 3010;

int n, m;
int g[N][N], h[N][N];
// int l[N], r[N], stk[N];
int stk[N];

int work(int h[]) {
    // top = 0 ----> idx-0 DO NOT store value
    int top = 0, res = 0;
    for (int i = 1; i <= m + 1; ++ i ) {
        while (top && h[stk[top]] >= h[i]) {
            int l = stk[top -- ];
            res = max(res, h[l] * (top == 0 ? i - 1 : i - stk[top] - 1));
        }
        stk[ ++ top] = i;
    }
    // cout << "res = " << res << endl;
    return res;
}

int main() {
    int p;
    cin >> n >> m >> p;
    while (p -- ) {
        int x, y;
        cin >> x >> y;
        g[x][y] = 1;
    }
    // 计算本行本列向上最多有多少可用位置
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            if (!g[i][j])
                h[i][j] = h[i - 1][j] + 1;
    
    // for (int i = 1; i <= n; ++ i )
    //     for (int j = 1; j <= m; ++ j )
    //         cout << h[i][j] << " \n"[j == m];
    
    int res = 0;
    for (int i = 1; i <= n; ++ i )
        res = max(res, work(h[i]));
    cout << res << endl;
    
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

> [!NOTE] **[LeetCode 1944. 队列中可以看到的人数](https://leetcode-cn.com/problems/number-of-visible-people-in-a-queue/)**
> 
> [biweekly-57](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-07-24_Biweekly-57/)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思考 抽象出模型 进而使用单调栈优化操作
> 
> 此类题目显然先想考虑 按高度加入 or 按方向加入
> 
> 本题按方向先加入右侧的点 需注意找到的是当前点右侧第一个大于等于当前点的数作为右边界
> 
> > 部分题解使用了大于等于当前点的最后一个数，是错误的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10;
    
    int stk[N], top;
    
    vector<int> canSeePersonsCount(vector<int>& heights) {
        int n = heights.size();
        this->top = 0;
        
        vector<int> res(n);
        for (int i = n - 1; i >= 0; -- i ) {
            int cnt = 0;
            while (top && heights[stk[top - 1]] < heights[i])
                top -- , cnt ++ ;
            if (top)
                cnt ++ ;
            res[i] = cnt;
            // 等于 后面的看不到
            while (top && heights[stk[top - 1]] == heights[i])
                top -- ;
            stk[top ++ ] = i;
        }
        return res;
    }
};
```

##### **C++ 数组简化栈操作**

```cpp
class Solution {
public:
    vector<int> canSeePersonsCount(vector<int>& heights) {
        int n = heights.size();
        vector<int> ans(n), v;
        for (int i = n - 1; i >= 0; i -= 1) {
            int j = lower_bound(v.begin(), v.end(), heights[i], greater<int>()) - v.begin();
            ans[i] = v.size() - j + (j != 0);
            while (not v.empty() and v.back() <= heights[i])
                v.pop_back();
            v.push_back(heights[i]);
        }
        return ans;
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

> [!NOTE] **[LeetCode 496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/)**
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
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        int n = nums2.size();
        unordered_map<int, int> hash;
        stack<int> st;
        for (int i = 0; i < n; ++ i ) {
            while (!st.empty() && st.top() < nums2[i]) {
                hash[st.top()] = nums2[i];
                st.pop();
            }
            st.push(nums2[i]);
        }
        // while (!st.empty()) hash[st.top()] = -1, st.pop();
        vector<int> res;
        for (auto v : nums1)
            if (hash.count(v)) res.push_back(hash[v]);
            else res.push_back(-1);
        return res;
    }
};
```

##### **Python**

```python
"""
# 模板
# 当前数向右找第一个比自己大的位置：从左向右维护一个单调递减栈
def nextBiggerElement(nums: list):
    n=len(nums)
    #先初始化所有res，只用改变存在的；
    #小技巧：stack里保存的是数组下标
    res,stack=[-1]*n,[]
    for i in range(n):
        while stack and nums[stack[-1]]<nums[i]:
            #当前数nums[i]大于栈顶元素时，就把栈顶元素对应的res的值更新为当前元素
            #然后把栈顶元素pop出去，继续比较
            res[stack[-1]=nums[i]
            stack.pop()
        #当前数小于或者等于栈顶元素时，直接把当前数的下标append到栈中        
        stack.append(i)
    return res
                   
                
# 当前数向左找第一个比自己的大的位置：从左边向右边维护一个单调递减栈         
def nextBiggerElement(nums: list):
  	n=len(nums)
    res,stack=[-1]*(n),[]
    for i in range(n-1，-1，-1):
      	while stack and nums[stack[-1]]<nums[i]:
            res[stack[-1]]=nums[i]
            stack.pop()
        stack.append(i)
    return res   
"""

#写法1:从前向后遍历
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        my_dic={}
        n=len(nums2)
        stack=[]
        for i in range(n):
            while stack and nums2[stack[-1]]<nums2[i]:
                my_dic[nums2[stack[-1]]]=nums2[i]
                stack.pop()
            stack.append(i)
        return [my_dic.get(x,-1)for x in nums1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)**
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
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; ++ i ) nums.push_back(nums[i]);
        vector<int> res(n, -1);
        stack<int> st;  // 记录下标
        for (int i = 0; i < 2 * n; ++ i ) {
            while (!st.empty() && nums[st.top()] < nums[i]) {
                if (i - st.top() < n) res[st.top() % n] = nums[i];
                st.pop();
            }
            st.push(i);
        }
        return res;
    }
};


// yxc
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        nums.insert(nums.end(), nums.begin(), nums.end());
        stack<int> stk;
        vector<int> res(n);
        for (int i = n * 2 - 1; i >= 0; i -- ) {
            int x = nums[i];
            while (stk.size() && x >= stk.top()) stk.pop();
            if (i < n) {
                if (stk.empty()) res[i] = -1;
                else res[i] = stk.top();
            }
            stk.push(x);
        }
        return res;
    }
};
```

##### **Python**

```python
"""
# 模板
# 当前数向右找第一个比自己大的位置：从左向右维护一个单调递减栈
def nextBiggerElement(nums: list):
    n=len(nums)
    #先初始化所有res，只用改变存在的；
    #小技巧：stack里保存的是数组下标
    res,stack=[-1]*n,[]
    for i in range(n):
        while stack and nums[stack[-1]]<nums[i]:
            #当前数nums[i]大于栈顶元素时，就把栈顶元素对应的res的值更新为当前元素
            #然后把栈顶元素pop出去，继续比较
            res[stack[-1]=nums[i]
            stack.pop()
        #当前数小于或者等于栈顶元素时，直接把当前数的下标append到栈中        
        stack.append(i)
    return res
                   
                
# 当前数向左找第一个比自己的大的位置：从左边向右边维护一个单调递减栈         
def nextBiggerElement(nums: list):
  	n=len(nums)
    res,stack=[-1]*(n),[]
    for i in range(n-1，-1，-1):
      	while stack and nums[stack[-1]]<nums[i]:
            res[stack[-1]]=nums[i]
            stack.pop()
        stack.append(i)
    return res   
"""

class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n=len(nums)
        for i in range(n):
            nums.append(nums[i])
        stack=[]
        res = [-1] * n
        for i in range (2*n):
            while stack and nums[stack[-1]] < nums[i]:
                #保证是在有效区间内，一定要有这个判断 不然可能会报错
                if stack[-1] > i - n:
                    res[stack[-1] % n] = nums[i]
                		stack.pop()
            stack.append(i)
        return res
   
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 556. 下一个更大元素 III](https://leetcode-cn.com/problems/next-greater-element-iii/)**
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
    int nextGreaterElement(int n) {
        string s = to_string(n);
        int k = s.size() - 1;
        while (k && s[k - 1] >= s[k]) -- k;
        if (!k) return -1;
        int t = k;
        while (t + 1 < s.size() && s[t + 1] > s[k - 1]) ++ t;
        swap(s[k - 1], s[t]);
        reverse(s.begin() + k, s.end());
        long long r = stoll(s);
        if (r > INT_MAX) return -1;
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

> [!NOTE] **[LeetCode 735. 行星碰撞](https://leetcode-cn.com/problems/asteroid-collision/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然单调栈 有更简单写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 单调栈**

```cpp
class Solution {
public:
    vector<int> asteroidCollision(vector<int>& a) {
        int n = a.size();
        vector<int> res;
        stack<int> st;
        for (int i = 0; i < n; ++ i )
            if (a[i] < 0) {
                int t = a[i];
                while (st.size() && st.top() < -t)
                    st.pop();
                if (st.size()) {
                    if (st.top() == -t)
                        st.pop();
                } else
                    res.push_back(t);
            } else
                st.push(a[i]);

        vector<int> t;
        while (st.size())
            t.push_back(st.top()), st.pop();
        reverse(t.begin(), t.end());
        for (auto v : t)
            res.push_back(v);
        return res;
    }
};
```

##### **C++ 简单trick写法**

```cpp
class Solution {
public:
    vector<int> asteroidCollision(vector<int>& asteroids) {
        vector<int> res;
        for (auto x : asteroids)
            if (x > 0) res.push_back(x);
            else {
                while (res.size() && res.back() > 0 && res.back() < -x)
                    res.pop_back();
                if (res.size() && res.back() == -x)
                    res.pop_back();
                else if (res.empty() || res.back() < 0)
                    res.push_back(x);
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

> [!NOTE] **[LeetCode 2334. 元素值大于变化阈值的子数组](https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 较显然的有单调栈做法
> 
> - **另有并查集思录**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 单调栈**

```cpp
class Solution {
public:
    int validSubarraySize(vector<int>& nums, int threshold) {
        int n = nums.size();
        vector<int> l(n, -1), r(n, n);
        stack<int> st;
        // 考虑某一个区间，受该区间内最小的值影响
        // 那么去找某个位置作为最小的值可以延伸的左右边界
        {
            for (int i = n - 1; i >= 0; -- i ) {
                while (st.size() && nums[st.top()] > nums[i]) {
                    l[st.top()] = i;
                    st.pop();
                }
                st.push(i);
            }
            while (st.size())
                st.pop();
        }
        {
            for (int i = 0; i < n; ++ i ) {
                while (st.size() && nums[st.top()] > nums[i]) {
                    r[st.top()] = i;
                    st.pop();
                }
                st.push(i);
            }
            while (st.size())
                st.pop();
        }
        
        for (int i = 0; i < n; ++ i ) {
            int lx = l[i], rx = r[i];
            int k = rx - lx - 1;
            if (nums[i] > threshold / k)
                return k;
        }
        return -1;
    }
};
```

##### **C++ 并查集 思维**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 1e5 + 10;
    
    int fa[N], sz[N];
    void init() {
        for (int i = 0; i < N; ++ i )
            fa[i] = i, sz[i] = 1;
    }
    int find(int x) {
        if (fa[x] != x)
            fa[x] = find(fa[x]);
        return fa[x];
    }
    
    int validSubarraySize(vector<int>& nums, int threshold) {
        init();
        
        vector<PII> xs;
        int n = nums.size();
        for (int i = 0; i < n; ++ i )
            xs.push_back({nums[i], i});
        
        sort(xs.begin(), xs.end());
        // ATTENTION: 思维 从大到小去逐个合并
        for (int i = n - 1; i >= 0; -- i ) {
            int pi = xs[i].second, pj = find(pi + 1);   // ATTENTION 细节
            fa[pi] = pj;
            sz[pj] += sz[pi];
            if (xs[i].first > threshold / (sz[pj] - 1)) // -1 cause we link n-th when i=n-1
                return sz[pj] - 1;
        }
        return -1;
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

### 进阶（推导）

> [!NOTE] **[LeetCode 42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 和 0084.柱状图中最大的矩形同思路 同思路
> 
> **以及纵向计算的双指针优化思路**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        stack<int> st;
        int res = 0;
        for (int i = 0; i < n; ++ i ) {
            while (st.size() && height[st.top()] <= height[i]) {
                int t = st.top(); st.pop();
                if (st.size()) {
                    int l = st.top();
                    res += (min(height[i], height[l]) - height[t]) * (i - l - 1);
                }
            }
            st.push(i);
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
// 考虑某个位置作为最低点可以接的雨水，受到左右两侧第一个比它高的高度的限制
// 维护单调递减栈，每次考虑顶部元素
// 则当前【h[i]与新的顶部】即为左右两侧第一个比它高的高度，直接计数即可
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size(), res = 0;
        stack<int> st;
        for (int i = 0; i < n; ++ i ) {
            while (st.size() && height[st.top()] < height[i]) {
                int t = st.top(); st.pop();
                if (st.size()) {
                    res += (min(height[i], height[st.top()]) - height[t]) * (i - st.top() - 1);
                }
            }
            st.push(i);
        }
        return res;
    }
};
```

##### **Python**

```python
# 维护一个【单调递减栈】
#处理当前数时，需要把栈里小于或者等于它的数值都弹出去
#高度：栈顶元素和上一个元素的高度差；宽度：是1
#宽度：当前柱子的左边界到下一个柱子的右边界


class Solution:
    def trap(self, h: List[int]) -> int:
        stack = []  # 栈里存储的是下标
        res = 0; n = len(h)
        for i in range(n):
            while stack and h[stack[-1]] < h[i]:
                t = stack.pop()
                if stack:
                    res += (min(h[i], h[stack[-1]]) - h[t]) * (i - stack[-1] - 1)
            stack.append(i)
        return res
```

<!-- tabs:end -->
</details>

<br>

> [!TIP] **更进一步的思路**
> 
> 前面做法本质是求每行（横向）累积的雨水。
> 
> 实际上可以通过记录某位置两侧分别最大的高度，来直接累积每列（纵向）累积的雨水。
> 
> 别记录的过程可以双指针优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int l = 0, r = height.size() - 1;
        int lmax = 0, rmax = 0, res = 0;
        while (l < r) {
            if (height[l] < height[r]) {
                if (height[l] > lmax)
                    lmax = height[l ++ ];
                else
                    res += lmax - height[l ++ ];
            } else {
                if (height[r] > rmax)
                    rmax = height[r -- ];
                else
                    res += rmax - height[r -- ];
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

> [!NOTE] **[LeetCode 84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 和 0042 接雨水 同思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        int n = heights.size(), res = 0;
        stack<int> st;
        for (int i = 0; i < n; ++ i ) {
            while (st.size() && heights[st.top()] > heights[i]) {
                int t = st.top(); st.pop();
                res = max(res, heights[t] * (i - st.top() - 1));
            }
            st.push(i);
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
// 考虑某个位置作为最高点的矩形面积，受到左右两侧第一个比它低的高度的限制
// 维护单调递增栈，每次考虑顶部元素
// 则当前【h[i]与新的顶部】即为左右两侧第一个比它高的高度，直接计数即可
// TRICK 在末尾加入0方便处理case
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        heights.push_back(0);
        int n = heights.size(), res = 0;
        stack<int> st;
        for (int i = 0; i < n; ++ i ) {
            while (st.size() && heights[st.top()] > heights[i]) {
                int t = st.top(); st.pop();
                if (st.empty())
                    res = max(res, heights[t] * (i - (-1) - 1));
                else
                    res = max(res, heights[t] * (i - st.top() - 1));
            }
            st.push(i);
        }
        return res;
    }
};

// TRICK 在起始也加入0简化判断逻辑
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        int n = heights.size(), res = 0;
        stack<int> st;
        for (int i = 0; i < n; ++ i ) {
            while (st.size() && heights[st.top()] > heights[i]) {
                int t = st.top(); st.pop();
                res = max(res, heights[t] * (i - st.top() - 1));
            }
            st.push(i);
        }
        return res;
    }
};
```

##### **Python**

```python
#方法：枚举所有柱形的上边界，作为整个矩形的上边界。
#然后找出左右边界：1. 找出左边离它最近的并且比它小的柱形；2.找出右边离它最近并且比它小的柱形。
#这就是要找“在一个数组中，每个数的左边第一个比它小/大的数”，于是可以想到用单调栈来解决这类问题。

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n=len(heights)
        left,right=[0]*n,[0]*n
        #栈里保存的是元素下标
        stack=[]
        res=0
        for i in range(n):
            while stack and heights[stack[-1]]>=heights[i]:
                stack.pop()
            if not stack:
                left[i]=-1
            else:
                left[i]=stack[-1]
            stack.append(i)
        while len(stack) > 0: stack.pop() # 先清空单调栈
        #stack.clear()
        for i in range(n-1,-1,-1):
            while stack and heights[stack[-1]]>=heights[i]:
                stack.pop()
            if not stack:
                #右边界的下一个位置    
                right[i]=n
            else:
                right[i]=stack[-1]
            stack.append(i)
        #更新答案：
        for i in range(n):
            res=max(res,heights[i]*(right[i]-left[i]-1))
        return res
      
"""
和 接雨水 类似；维护 单调递增栈，找到左右两边 第一个 比当前数 小

"""
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 尾部追加一个0 保证最终可以将栈排空（不会有遗漏某些情况的可能）
        heights.append(0)
        n = len(heights)
        res = 0
        stack = []  # 维护一个单调递增栈
        for i in range(n):
            while stack and heights[stack[-1]] > heights[i]:
                t = stack[-1]
                stack.pop()
                if not stack:
                    res = max(res, heights[t] * (i - (-1) - 1))
                else:
                    res = max(res, heights[t] * (i - stack[-1] - 1))
            stack.append(i)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)**
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
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (!m) return 0;
        int n = matrix[0].size();
        vector<int> h(n + 1); // 默认h[n] = 0;
        int res = 0;
        for (int i = 0; i < m; ++ i ) {
            for (int j = 0; j < n; ++ j )
                h[j] = matrix[i][j] == '1' ? h[j] + 1 : 0;
            res = max(res, maxArea(h, n));
        }
        return res;
    }
    int maxArea(vector<int>& h, int n) {
        int res = 0;
        stack<int> s;
        for (int i = 0; i <= n; ++ i ) {
            while (!s.empty() && h[s.top()] >= h[i]) {
                int l = s.top(); s.pop();
                res = max(res, h[l] * (s.empty() ? i : i - s.top() - 1));
            }
            s.push(i);
        }
        return res;
    }
};
```

##### **Python**

```python
"""
1. 将 Largest Rectangle in Histogram 问题扩展到二维。
2. 一行一行考虑，类比 Largest Rectangle in Histogram，一行内所有柱形条的高度 heights 就是当前 (i, j) 位置能往上延伸的最大高度。
3. 直接套用 Largest Rectangle in Histogram 的单调栈算法即可。

枚举每一行的时间复杂度是 O(n)，行内单调栈的时间复杂度是 O(m)，故总时间复杂度为 $O(nm)
"""
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # the same as 84.
        if not matrix or not matrix[0]: return 0
        res, n = 0, len(matrix[0])
        heights = [0] * (n + 1)
        for row in matrix:
            for i in range(n):
                if row[i] == "1":
                    heights[i] += 1
                else:
                    heights[i] = 0
            stack = [-1]
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    res = max(res, heights[stack.pop()] * (i - stack[-1] - 1)) # height x width
                stack.append(i)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces Maximum Submatrix 2](http://codeforces.com/problemset/problem/375/B)**
> 
> 题意: 
> 
> 最大矩形 + 行可排序

> [!TIP] **思路**
> 
> 最大矩形变体，非常有意思
> 
> 按列考虑，对行进行排序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Maximum Submatrix 2
// Contest: Codeforces - Codeforces Round #221 (Div. 1)
// URL: https://codeforces.com/problemset/problem/375/B
// Memory Limit: 512 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// 本题题意重点在于【可以重新排列行】

const static int N = 5e3 + 10;

int n, m;
char g[N][N];
int suf[N][N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m;

    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            cin >> g[i][j];

    for (int i = 1; i <= n; ++i)
        for (int j = m; j >= 1; --j)
            if (g[i][j] == '1')
                suf[i][j] = suf[i][j + 1] + 1;
            else
                suf[i][j] = 0;

    int res = 0;
    // 按列考虑，对行进行排序
    for (int j = 1; j <= m; ++j) {
        static int t[N];
        for (int i = 1; i <= n; ++i)
            t[i] = suf[i][j];
        sort(t + 1, t + n + 1);
        for (int i = 1; i <= n; ++i)
            res = max(res, t[i] * (n - i + 1));
    }
    cout << res << endl;

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


> [!NOTE] **[LeetCode 1944. 队列中可以看到的人数](https://leetcode-cn.com/problems/number-of-visible-people-in-a-queue/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 此类题目显然先想考虑 `按高度加入` or `按方向加入`
> 
> 本题按方向先加入右侧的点 需注意找到的是当前点右侧第一个大于等于当前点的数作为右边界
> 
> > 部分题解使用了大于等于当前点的最后一个数，是错误的
> 
> **数组可以简化栈操作**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10;
    
    int stk[N], top;
    
    vector<int> canSeePersonsCount(vector<int>& heights) {
        int n = heights.size();
        this->top = 0;
        
        vector<int> res(n);
        for (int i = n - 1; i >= 0; -- i ) {
            int cnt = 0;
            while (top && heights[stk[top - 1]] < heights[i])
                top -- , cnt ++ ;
            if (top)
                cnt ++ ;
            res[i] = cnt;
            // 等于 后面的看不到
            while (top && heights[stk[top - 1]] == heights[i])
                top -- ;
            stk[top ++ ] = i;
        }
        return res;
    }
};
```

##### **C++ 数组简化栈操作**

```cpp
class Solution {
public:
    vector<int> canSeePersonsCount(vector<int>& heights) {
        int n = heights.size();
        vector<int> ans(n), v;
        for (int i = n - 1; i >= 0; i -= 1) {
            int j = lower_bound(v.begin(), v.end(), heights[i], greater<int>()) - v.begin();
            ans[i] = v.size() - j + (j != 0);
            while (not v.empty() and v.back() <= heights[i])
                v.pop_back();
            v.push_back(heights[i]);
        }
        return ans;
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

> [!NOTE] **[Codeforces Psychos in a Line](http://codeforces.com/problemset/problem/319/B)** [TAG]
> 
> 题意: 
> 
> 有 $n$ 个点，每个回合里，当 $a_i>a_{i+1}$ ，则 $a_{i+1}$ 被视为死亡。
> 
> 同一个回合内，某个点可能即被其他点杀死，也杀死其他点。

> [!TIP] **思路**
> 
> 推理较显然，每次操作时当下已有的递减区间是不断变长的
> 
> 也即：某个位置被消灭时，其次数至少在前面一个子区间之后
> 
> 考虑单调栈维护子区间
> 
> 细节 **非常好的题目 重复做TODO**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Psychos in a Line
// Contest: Codeforces - Codeforces Round #189 (Div. 1)
// URL: http://codeforces.com/problemset/problem/319/B
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10;

int n;
int a[N], f[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;
    for (int i = 0; i < n; ++i)
        cin >> a[i];

    // 维护一个单调递减栈
    stack<int> st;
    memset(f, 0, sizeof f);
    for (int i = 0; i < n; ++i) {
        int t = 0;
        while (st.size() && a[st.top()] <= a[i]) {
            // f[i] 表示被杀死的回合
            // 此时 i 一定在 st.top() 之后被杀死
            t = max(t, f[st.top()]);
            st.pop();
        }
        // ATTENTION 非空栈说明可以接上前面的，作为下一次杀死的
        if (st.size())
            f[i] = t + 1;
        st.push(i);
    }

    int res = 0;
    for (int i = 0; i < n; ++i)
        res = max(res, f[i]);
    cout << res << endl;

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

> [!NOTE] **[LeetCode 6080. 使数组按非递减顺序排列](https://leetcode.cn/problems/steps-to-make-array-non-decreasing/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **非常经典的单调栈应用 以及理清楚维护的细节**
> 
> 单调栈 + **c数组维护**
> 
> 思想 **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 找到右侧第一个大于等于当前数的位置，中间的都需要消除  ==> 单调递减栈
    //      此时显然需要统计中间的最大消耗               ==> c 数组
    //      重点在于 if (stk.size()) c[i] = t + 1, res = max(res, c[i]);
    //      ==> 理由: 递减栈存储元素及其被删除的时间c 已更新当前的c（临时变量t）
    //          当栈为空：说明前面没有比当前元素大的，c即为0
    int totalSteps_2(vector<int>& nums) {
        int n = nums.size(), res = 0;
        vector<int> stk, c(n);
        for (int i = 0; i < n; ++ i ) {
            int t = 0;
            while (stk.size() && nums[stk.back()] <= nums[i]) {
                t = max(t, c[stk.back()]);
                stk.pop_back();
            }
            // ATTENTION 只有栈不为空 当前位置才需要最终被当前的栈顶所消耗 也即才需要统计当前元素被干掉的时间
            // 否则当前元素可以留下，不需要统计cost
            if (stk.size()) {
                c[i] = t + 1;
                res = max(res, c[i]);
            }
            stk.push_back(i);
        }
        return res;
    }
    
    // 为每个元素找到吃掉它的那个元素 ==> 逆序维护单调递减(可以相等)栈
    //  ATTENTION: 则在 while 循环弹出过程中，弹出的元素都应是整个过程中被当前元素吃掉的
    //      则：当前元素的 cost = max(当前已有的cost+1, 被弹出元素本身的cost) 【重要】
    int totalSteps(vector<int>& nums) {
        int n = nums.size(), res = 0;
        vector<int> stk, c(n);
        for (int i = n - 1; i >= 0; -- i ) {
            while (stk.size() && nums[stk.back()] < nums[i]) {
                res = max(res, c[i] = max(c[i] + 1, c[stk.back()]));
                stk.pop_back();
            }
            stk.push_back(i);
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

### 删数类问题

> [!NOTE] **[Luogu 删数问题](https://www.luogu.com.cn/problem/P1106)**
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

const int N = 260;

int n, k;
string s;

int main() {
    cin >> s >> k;
    n = s.size();
    
    stack<char> st;
    for (int i = 0; i < n; ++ i ) {
        while (st.size() && k && s[st.top()] > s[i]) {
            st.pop();
            k -- ;
        }
        st.push(i);
    }

    while (k -- && st.size())
        st.pop();
    
    string res;
    while (st.size()) {
        res.push_back(s[st.top()]);
        st.pop();
    }
    // zero in head
    while (res.size() > 1 && res.back() == '0')
        res.pop_back();
    reverse(res.begin(), res.end());
    cout << res << endl;
    
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

> [!NOTE] **[LeetCode 316. 去除重复字母](https://leetcode-cn.com/problems/remove-duplicate-letters/)**
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
    string removeDuplicateLetters(string s) {
        string stk;
        unordered_map<char, bool> ins;
        unordered_map<char, int> last;
        for (int i = 0; i < s.size(); i ++ ) last[s[i]] = i;

        for (int i = 0; i < s.size(); i ++ ) {
            if (ins[s[i]]) continue;
            while (stk.size() && stk.back() > s[i] && last[stk.back()] > i) {
                ins[stk.back()] = false;
                stk.pop_back();
            }
            stk += s[i];
            ins[s[i]] = true;
        }

        return stk;
    }
};
```

##### **C++ 标准**

```cpp
class Solution {
public:
    string removeDuplicateLetters(string s) {
        string stk;
        size_t i = 0;
        for (size_t i = 0;i < s.size(); ++ i ) {
            if (stk.find(s[i]) != string::npos) continue;
            // 遇到一个新字符 如果比栈顶小 并且在新字符后面还有和栈顶一样的 就把栈顶的字符抛弃了
            while (!stk.empty() && stk.back() > s[i] && s.find(stk.back(), i) != string::npos)
                stk.pop_back();
            stk.push_back(s[i]);
        }
        return stk;
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

> [!NOTE] **[LeetCode 2030. 含特定字母的最小子序列](https://leetcode-cn.com/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/)**
> 
> [weekly-261](https://github.com/OpenKikCoc/LeetCode/blob/master/Contest/2021-10-03_Weekly-261/)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单题：https://github.com/OpenKikCoc/LeetCode/blob/master/0401-0500/0402/README.md
> 
> 增加条件：其中某个字符 `letter` 至少出现 `repitition` 次
> 
> ==> 在出栈时的限制要求更严格 且后续删除多余字符时有严格的合理性推导
> 
> **本题有多种解题思路 细节看 Contest 全文**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string smallestSubsequence(string s, int k, char letter, int repetition) {
        int n = s.size();
        
        // suf 甚至可以用一个变量统计 在 for-loop 中递减来维护
        vector<int> suf(n);
        for (int i = n - 1; i >= 0; -- i )
            suf[i] = (i < n - 1 ? suf[i + 1] : 0) + (s[i] == letter);
        
        int del = n - k, has = 0;
        stack<int> st;
        for (int i = 0; i < n; ++ i ) {
            // the while-condition
            while (st.size() && del && s[st.top()] > s[i] && 
                  // 把当前栈顶去除letter仍然够用
                  has - (s[st.top()] == letter) + suf[i] >= repetition) {
                if (s[st.top()] == letter)
                    has -- ;
                del -- ;
                st.pop();
            }
            st.push(i);
            if (s[st.top()] == letter)
                has ++ ;
        }
        
        // 只获取前k项
        while (st.size() > k)
            has -= (s[st.top()] == letter), st.pop();
        string res;
        while (st.size())
            res.push_back(s[st.top()]), st.pop();
        reverse(res.begin(), res.end());
        
        // ATTENTION 使用letter向前挤兑其他字符的位置 以满足至少repetition次的要求
        // 重要: 思考为什么这样是可行的? ----> 因为相当于去除其他元素并将后面的直接前移
        for (int i = k - 1; has < repetition; -- i )
            if (res[i] != letter)
                res[i] = letter, has ++ ;
        
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

> [!NOTE] **[LeetCode 321. 拼接最大数](https://leetcode-cn.com/problems/create-maximum-number/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典题 贪心 分治 单调栈 => 综合应用
> 
> dp解法数值会溢出
> 
> 贪心 + 单调栈:
> 
> 原问题直接处理比较困难，我们分成三步来做：
> 
> 1. 先枚举从两个数组中分别选多少个数；
> 2. 然后分别贪心求解每个数组中需要选那些数；
> 3. 将选出的两个数列合并；

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> maxArray(vector<int> & nums, int k) {
        int n = nums.size();
        vector<int> res(k);
        for (int i = 0, j = 0; i < n; ++ i ) {
            while (n - i + j > k && j && res[j - 1] < nums[i]) -- j;
            if (j < k) res[j ++ ] = nums[i];
        }
        return res;
    }
    vector<int> merge(vector<int> & N, vector<int> & M) {
        vector<int> res;
        while (N.size() && M.size())
            if (N > M) res.push_back(N[0]), N.erase(N.begin());
            else res.push_back(M[0]), M.erase(M.begin());
        while (N.size()) res.push_back(N[0]), N.erase(N.begin());
        while (M.size()) res.push_back(M[0]), M.erase(M.begin());
        return res;
    }

    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        int n1 = nums1.size(), n2 = nums2.size();
        vector<int> res(k, INT_MIN);
        // i 是右边界开区间
        for (int i = max(0, k - n2); i <= k && i <= n1; ++ i ) {
            vector<int> N = maxArray(nums1, i);
            vector<int> M = maxArray(nums2, k - i);
            vector<int> t = merge(M, N);
            if (res < t) res = t;
        }
        return res;
    }
};
```

##### **C++ dp溢出**

```cpp
class Solution {
public:
    typedef long long LL;
    // dp
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        int n1 = nums1.size(), n2 = nums2.size();
        vector<int> res;
        vector<vector<vector<LL>>> f(n1 + 1, vector<vector<LL>>(n2 + 1, vector<LL>(k + 1)));
        for (int t = 1; t <= k; ++ t)
            for (int i = 0; i <= n1; ++ i)
                for (int j = 0; j <= n2; ++ j) {
                    if (i > 0)
                        f[i][j][t] = max(f[i][j][t], max(f[i-1][j][t], f[i-1][j][t-1] * 10 + nums1[i-1]));
                    if (j > 0)
                        f[i][j][t] = max(f[i][j][t], max(f[i][j-1][t], f[i][j-1][t-1] * 10 + nums2[j-1]));
                }
        LL val = f[n1][n2][k];
        while (val) res.push_back(val % 10), val /= 10;
        reverse(res.begin(), res.end());
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

> [!NOTE] **[LeetCode 402. 移掉K位数字](https://leetcode-cn.com/problems/remove-k-digits/)**
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
    string removeKdigits(string num, int k) {
        int n = num.size();
        string res;
        for (int i = 0; i < n; ++ i ) {
            while (!res.empty() && k && res.back() > num[i]) res.pop_back(), -- k ;
            res.push_back(num[i]);
        }
        while (k -- ) res.pop_back();
        while (!res.empty() && res[0] == '0') res.erase(res.begin());
        if (res.empty()) res = "0";
        return res;
    }
};
```

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    string removeKdigits(string num, int k) {
        k = min(k, (int)num.size());
        string res;
        for (auto c: num) {
            while (k && res.size() && res.back() > c) {
                k -- ;
                res.pop_back();
            }
            res += c;
        }
        while (k -- ) res.pop_back();
        k = 0;
        while (k < res.size() && res[k] == '0') k ++ ;
        if (k == res.size()) res += '0';
        return res.substr(k);
    }
};
```

##### **Python**

```python
"""
贪心：
题解：如果我们当前遇到的数字比上一个数字要小的话，肯定是删除上一个数字比较划算。我们最多能删除k个字符。所以我们使用一个单调栈来存储每一个字符，如果当前读进来的数字比前一个数字小，我们就将栈顶元素出栈，直至出栈了k个字符或者栈顶元素已经比当前元素还小。这样在我们删除k个元素后，栈中元素就是剩下的数字啦。这时候我们需要考虑的就是删除前导0和空栈的情况啦。字符串有push和pop操作，所以我们可以直接用字符串来模拟栈，效果是一样的。
"""
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        remain = len(num) - k 
        for c in num:
            while stack and stack[-1] > c and k:
                stack.pop()
                k -= 1
            stack.append(c)
        return ''.join(stack[:remain]).lstrip('0') or '0'
      

      
# 以下写法有一个case会过不了：'9',1
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        n = len(num)
        for i in range(n):
            while stack and stack[-1] > num[i] and k > 0:
                stack.pop()
                k -= 1 
            stack.append(num[i])
        res = ''.join(stack).lstrip('0')
        return res if res else '0'
# 上述代码的问题在于：需要注意的是，如果给定的数字是一个单调递增的数字，那么我们的算法会永远选择不丢弃。这个题目中要求的，我们要永远确保丢弃 k 个矛盾。
# 一个简单的思路就是：每次丢弃一次，k 减去 1。当 k 减到 0 ，我们可以提前终止遍历。
# 而当遍历完成，如果 k 仍然大于 0。不妨假设最终还剩下 x 个需要丢弃，那么我们需要选择删除末尾 x 个元素。修改后，可以通过：

class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        n = len(num)
        for i in range(n):
            while stack and stack[-1] > num[i] and k > 0:
                stack.pop()
                k -= 1 
            stack.append(num[i])
        while k > 0:   # 判断：是否已经移除了k位！！！
            stack.pop()
            k -= 1
        res = ''.join(stack).lstrip('0')
        return res if res else '0'
      
# 逆向思维，一定需要舍弃k位，那就是需要保留len(num) - k位。这样就不需要对k是否为0进行校验了, 但是在输出的时候，stack还是只能取前remain个数字输出。
class Solution(object):
    def removeKdigits(self, num, k):
        stack = []
        remain = len(num) - k
        for c in num:
            while k and stack and stackresls[-1] > c:
                stack.pop()
                k -= 1
            stack.append(c)
        return ''.join(stack[:remain]).lstrip('0') or '0' # 踩坑：stack[:remain]
       
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1673. 找出最具竞争力的子序列](https://leetcode-cn.com/problems/find-the-most-competitive-subsequence/)**
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
    vector<int> mostCompetitive(vector<int>& nums, int k) {
        stack<int> st;
        int n = nums.size(), tot = n - k;
        for (auto v : nums) {
            while (!st.empty() && st.top() > v && tot) st.pop(), -- tot;
            st.push(v);
        }
        while (tot -- ) st.pop();
        vector<int> res;
        while (!st.empty()) {
            res.push_back(st.top());
            st.pop();
        }
        reverse(res.begin(), res.end());
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

### 132 模式

> [!NOTE] **[LeetCode 456. 132模式](https://leetcode-cn.com/problems/132-pattern/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> stack-sortable permutation
>
> 思维 trick 重复
>
> 与之前一个判断二叉树序列的题类比
>
> >   注意 维护的是单调递减的栈
> >
> >   -   我们可以从右往左遍历数组考虑每个数作为1或3的情况，同时维护一个次大值，这个次大值满足左边有一个数比它大，即是132模式中的2。
> >
> >       -   假如我们遇到了小于当前次大值的数说明我们就找到了一个1，构成了一个132模式。
> >       -   否则我们就把当前数看成3，从当前数开始向右找到比当前数小的数并且更新次大值，这里我们只用向右找到第一个比当前数大的位置x即可，因为从该位置开始再往右的数如果小于当前数那么它也一定会小于这个比当前数大的数，也就是说他们之前已经在考虑x的时候被作为次大值更新过了，没有必要再重新更新一遍。
> >
> >   -   我们从右往左扫描数组并维护一个单调递减的栈，初始时次大值设为无穷小。
> >
> >       如果当前数小于次大值说明我们找到了一个答案，否则考虑当前数作为3的情况，当当前数大于栈顶时说明我们找到了一个32模式，我们不断的弹出栈顶来更新2，即维护我们当前遇到的次大值，直到栈顶大于当前值为止。
> >
> >       注意这时栈顶的右边可能还有之前被弹出过的小于当前数的值，但他们都会比当前的2小，即在扫描过程中这个2是会单调递增的，原因是如果不是单调递增的，那么这个第一次出现下降的数和当前的栈顶以及栈顶右边的数就构成了一个132模式，会在之前就直接返回。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
/*
这个问题与Knuth所提出来的 stack-sortable permutation 类似，
即判断一个数组是否可以只用一个栈来进行排序，当且仅当它不包含231模式。
而将本问题中的数组逆序，寻找132模式就变成了寻找231模式，
也即判断数组是否可以仅用一个栈来进行排序。
*/
    bool find132pattern(vector<int>& nums) {
        stack<int> stk;
        int right = INT_MIN;
        for (int i = nums.size() - 1; i >= 0; -- i ) {
            // 考虑当前元素作为1
            if (nums[i] < right) return true;
            // 更新2
            while (!stk.empty() && nums[i] > stk.top()) {
                right = max(right, stk.top());
                stk.pop();
            }
            stk.push(nums[i]);
        }
        return false;
    }
};
```

##### **Python**

```python
"""
# 模板
# 当前数向右找第一个比自己大的位置：从左向右维护一个单调递减栈
def nextBiggerElement(nums: list):
    n=len(nums)
    #先初始化所有res，只用改变存在的；
    #小技巧：stack里保存的是数组下标
    res,stack=[-1]*n,[]
    for i in range(n):
        while stack and nums[stack[-1]]<nums[i]:
            #当前数nums[i]大于栈顶元素时，就把栈顶元素对应的res的值更新为当前元素
            #然后把栈顶元素pop出去，继续比较
            res[stack[-1]=nums[i]
            stack.pop()
        #当前数小于或者等于栈顶元素时，直接把当前数的下标append到栈中        
        stack.append(i)
    return res
                   
                
# 当前数向左找第一个比自己的大的位置：从左边向右边维护一个单调递减栈         
def nextBiggerElement(nums: list):
  	n=len(nums)
    res,stack=[-1]*(n),[]
    for i in range(n-1，-1，-1):
      	while stack and nums[stack[-1]]<nums[i]:
            res[stack[-1]]=nums[i]
            stack.pop()
        stack.append(i)
    return res   
"""

class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        #对于每个数，找到右边比它小的数，左边比它小的数，并且左边的数要比右边的数小。
        stack = []
        _MIN = float("-inf")
        for num in reversed(nums):
            #[3, 1, 4, 2]
            if _MIN > num: return True
            while stack and stack[-1] < num:
                _MIN = stack.pop()
            stack.append(num)
        return False     
```

<!-- tabs:end -->
</details>

<br>

* * *

### 单调栈辅助思想

> [!NOTE] **[LeetCode 6202. 使用机器人打印字典序最小的字符串](https://leetcode.cn/problems/using-a-robot-to-print-the-lexicographically-smallest-string/)** [TAG]
> 
> 题意: 
> 
> 从左到右遍历 $s$，在允许用一个辅助栈的前提下，计算能得到的字典序最小的字符串

> [!TIP] **思路**
> 
> 显然单调栈，重点在于理清什么时候弹出
> 
> 贪心地思考，为了让字典序最小，在遍历 $s$ 的过程中：
> 
> - 如果栈顶字符 ≤ 后续字符（未入栈）的最小值，那么应该出栈并加到答案末尾
> 
> - 否则应当继续遍历，取到比栈顶字符小的那个字符

作者：endlesscheng
链接：https://leetcode.cn/problems/using-a-robot-to-print-the-lexicographically-smallest-string/solution/tan-xin-zhan-by-endlesscheng-ldds/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 比赛时**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10, M = 26;
    
    int c[M];  // 方便找后面的串中 最小的字符是谁 ==> 查询过程可以常量优化
    
    string robotWithString(string s) {
        int n = s.size();
        memset(c, 0, sizeof c);
        for (auto x : s)
            c[x - 'a'] ++ ;
        
        string res;
        stack<char> st;
        for (auto x : s) {
            int p = -1;
            for (int i = 0; i < M; ++ i )
                if (c[i]) {
                    p = i;
                    break;
                }
            if (x - 'a' == p) {
                res.push_back(x);
                c[p] -- ;
                
                while (st.size()) {
                    int t = -1;
                    for (int i = 0; i < M; ++ i )
                        if (c[i]) {
                            t = i;
                            break;
                        }
                    if (st.top() - 'a' <= t) {
                        res.push_back(st.top()), st.pop();
                    } else
                        break;
                }
            } else {
                st.push(x);
                c[x - 'a'] -- ;
            }
            // cout << " after x = " << x << " res = " << res << endl;
            // vzho fn 
        }
        while (st.size())
            res.push_back(st.top()), st.pop();
        return res;
    }
};
```

##### **C++ 标准**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10, M = 26;
    
    int c[M];
    
    string robotWithString(string s) {
        int n = s.size();
        memset(c, 0, sizeof c);
        for (auto x : s)
            c[x - 'a'] ++ ;
        
        string res;
        stack<char> st;
        int p = 0;  // 优化查找 p 的过程，干掉一个 for 循环
        for (auto x : s) {
            st.push(x);
            c[x - 'a'] -- ;
            
            // int p = -1;
            // for (int i = 0; i < M; ++ i )
            //     if (c[i]) {
            //         p = i;
            //         break;
            //     }
            while (p < M && c[p] == 0)
                p ++ ;
            
            while (st.size() && st.top() - 'a' <= p)
                res.push_back(st.top()), st.pop();
        }
        while (st.size())
            res.push_back(st.top()), st.pop();
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