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