> [!NOTE] **TODO**
> 
> 整合与 z-func 的比对与更完善的模版

## 描述

给定一个长度为 $n$ 的字符串 $s$，请找到所有对 $(i, j)$ 使得子串 $s[i \dots j]$ 为一个回文串。当 $t = t_{\text{rev}}$ 时，字符串 $t$ 是一个回文串（$t_{\text{rev}}$ 是 $t$ 的反转字符串）。

## 更进一步的描述

对于每个位置 $i = 0 \dots n - 1$，我们找出值 $d_1[i]$ 和 $d_2[i]$。二者分别表示以位置 $i$ 为中心的长度为奇数和长度为偶数的回文串个数。换个角度，二者也表示了以位置 $i$ 为中心的最长回文串的半径长度（半径长度 $d_1[i]$，$d_2[i]$ 均为从位置 $i$ 到回文串最右端位置包含的字符个数）。

举例来说，字符串 $s = \mathtt{abababc}$ 以 $s[3] = b$ 为中心有三个奇数长度的回文串，最长回文串半径为 $3$，也即 $d_1[3] = 3$：

$$
a\ \overbrace{b\ a\ \underset{s_3}{b}\ a\ b}^{d_1[3]=3}\ c
$$

字符串 $s = \mathtt{cbaabd}$ 以 $s[3] = a$ 为中心有两个偶数长度的回文串，最长回文串半径为 $2$，也即 $d_2[3] = 2$：

$$
c\ \overbrace{b\ a\ \underset{s_3}{a}\ b}^{d_2[3]=2}\ d
$$

因此关键思路是，如果以某个位置 $i$ 为中心，我们有一个长度为 $l$ 的回文串，那么我们有以 $i$ 为中心的长度为 $l - 2$，$l - 4$，等等的回文串。所以 $d_1[i]$ 和 $d_2[i]$ 两个数组已经足够表示字符串中所有子回文串的信息。

## 解法

总的来说，该问题具有多种解法：应用字符串哈希，该问题可在 $O(n \log n)$ 时间内解决，而使用后缀数组和快速 LCA 该问题可在 $O(n)$ 时间内解决。

但是这里描述的算法 **压倒性** 的简单，并且在时间和空间复杂度上具有更小的常数。该算法由 **Glenn K. Manacher** 在 1975 年提出。

## Manacher 算法

这里我们将只描述算法中寻找所有奇数长度子回文串的情况，即只计算 $d_1[]$；寻找所有偶数长度子回文串的算法（即计算数组 $d_2[]$）将只需对奇数情况下的算法进行一些小修改。

为了快速计算，我们维护已找到的最靠右的子回文串的 **边界 $(l, r)$**（即具有最大 $r$ 值的回文串，其中 $l$ 和 $r$ 分别为该回文串左右边界的位置）。初始时，我们置 $l = 0$ 和 $r = -1$（*-1*需区别于倒序索引位置，这里可为任意负数，仅为了循环初始时方便）。

现在假设我们要对下一个 $i$ 计算 $d_1[i]$，而之前所有 $d_1[]$ 中的值已计算完毕。我们将通过下列方式计算：

-   如果 $i$ 位于当前子回文串之外，即 $i > r$，那么我们调用朴素算法。

    因此我们将连续地增加 $d_1[i]$，同时在每一步中检查当前的子串 $[i - d_1[i] \dots i + d_1[i]]$（$d_1[i]$ 表示半径长度，下同）是否为一个回文串。如果我们找到了第一处对应字符不同，又或者碰到了 $s$ 的边界，则算法停止。在两种情况下我们均已计算完 $d_1[i]$。此后，仍需记得更新 $(l, r)$。

-   现在考虑 $i \le r$ 的情况。我们将尝试从已计算过的 $d_1[]$ 的值中获取一些信息。首先在子回文串 $(l, r)$ 中反转位置 $i$，即我们得到 $j = l + (r - i)$。现在来考察值 $d_1[j]$。因为位置 $j$ 同位置 $i$ 对称，我们 **几乎总是** 可以置 $d_1[i] = d_1[j]$。该想法的图示如下（可认为以 $j$ 为中心的回文串被“拷贝”至以 $i$ 为中心的位置上）：

    $$
    \ldots\
    \overbrace{
        s_l\ \ldots\
        \underbrace{
            s_{j-d_1[j]+1}\ \ldots\ s_j\ \ldots\ s_{j+d_1[j]-1}
        }_\text{palindrome}\
        \ldots\
        \underbrace{
            s_{i-d_1[j]+1}\ \ldots\ s_i\ \ldots\ s_{i+d_1[j]-1}
        }_\text{palindrome}\
        \ldots\ s_r
    }^\text{palindrome}\
    \ldots
    $$

    然而有一个 **棘手的情况** 需要被正确处理：当“内部”的回文串到达“外部”回文串的边界时，即 $j - d_1[j] + 1 \le l$（或者等价的说，$i + d_1[j] - 1 \ge r$）。因为在“外部”回文串范围以外的对称性没有保证，因此直接置 $d_1[i] = d_1[j]$ 将是不正确的：我们没有足够的信息来断言在位置 $i$ 的回文串具有同样的长度。

    实际上，为了正确处理这种情况，我们应该“截断”回文串的长度，即置 $d_1[i] = r - i$。之后我们将运行朴素算法以尝试尽可能增加 $d_1[i]$ 的值。

    该种情况的图示如下（以 $j$ 为中心的回文串已经被截断以落在“外部”回文串内）：

    $$
    \ldots\
    \overbrace{
        \underbrace{
            s_l\ \ldots\ s_j\ \ldots\ s_{j+(j-l)}
        }_\text{palindrome}\
        \ldots\
        \underbrace{
            s_{i-(r-i)}\ \ldots\ s_i\ \ldots\ s_r
        }_\text{palindrome}
    }^\text{palindrome}\
    \underbrace{
        \ldots \ldots \ldots \ldots \ldots
    }_\text{try moving here}
    $$

    该图示显示出，尽管以 $j$ 为中心的回文串可能更长，以致于超出“外部”回文串，但在位置 $i$，我们只能利用其完全落在“外部”回文串内的部分。然而位置 $i$ 的答案可能比这个值更大，因此接下来我们将运行朴素算法来尝试将其扩展至“外部”回文串之外，也即标识为 **try moving here" 的区域。

最后，仍有必要提醒的是，我们应当记得在计算完每个 $d_1[i]$ 后更新值 $(l, r)$。

同时，再让我们重复一遍：计算偶数长度回文串数组 $d_2[]$ 的算法同上述计算奇数长度回文串数组 $d_1[]$ 的算法十分类似。

## Manacher 算法的复杂度

因为在计算一个特定位置的答案时我们总会运行朴素算法，所以一眼看去该算法的时间复杂度为线性的事实并不显然。

然而更仔细的分析显示出该算法具有线性复杂度。此处我们需要指出，[计算 Z 函数的算法](string/z-func.md) 和该算法较为类似，并同样具有线性时间复杂度。

实际上，注意到朴素算法的每次迭代均会使 $r$ 增加 $1$，以及 $r$ 在算法运行过程中从不减小。这两个观察告诉我们朴素算法总共会进行 $O(n)$ 次迭代。

Manacher 算法的另一部分显然也是线性的，因此总复杂度为 $O(n)$。

## Manacher 算法的实现

### 分类讨论

为了计算 $d_1[]$，我们有以下代码：

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
vector<int> d1(n);
for (int i = 0, l = 0, r = -1; i < n; i++) {
    int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
    while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) { k++; }
    d1[i] = k--;
    if (i + k > r) {
        l = i - k;
        r = i + k;
    }
}
```

###### **Python**

```python
# Python Version
d1 = [0] * n
l, r = 0, -1
for i in range(0, n):
    k = 1 if i > r else min(d1[l + r - i], r - i + 1)
    while 0 <= i - k and i + k < n and s[i - k] == s[i + k]:
        k += 1
    d1[i] = k
    k -= 1
    if i + k > r:
        l = i - k
        r = i + k
```

<!-- tabs:end -->
</details>

<br>

计算 $d_2[]$ 的代码十分类似，但是在算术表达式上有些许不同：

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
vector<int> d2(n);
for (int i = 0, l = 0, r = -1; i < n; i++) {
    int k = (i > r) ? 0 : min(d2[l + r - i + 1], r - i + 1);
    while (0 <= i - k - 1 && i + k < n && s[i - k - 1] == s[i + k]) { k++; }
    d2[i] = k--;
    if (i + k > r) {
        l = i - k - 1;
        r = i + k;
    }
}
```

###### **Python**

```python
# Python Version
d2 = [0] * n
l, r = 0, -1
for i in range(0, n):
    k = 0 if i > r else min(d2[l + r - i + 1], r - i + 1)
    while 0 <= i - k - 1 and i + k < n and s[i - k - 1] == s[i + k]:
        k += 1
    d2[i] = k
    k -= 1
    if i + k > r:
        l = i - k - 1
        r = i + k
```

<!-- tabs:end -->
</details>

<br>

### 统一处理

虽然在讲解过程及上述实现中我们将 $d_1[]$ 和 $d_2[]$ 的计算分开考虑，但实际上可以通过一个技巧将二者的计算统一为 $d_1[]$ 的计算。

给定一个长度为 $n$ 的字符串 $s$，我们在其 $n + 1$ 个空中插入分隔符 $\#$，从而构造一个长度为 $2n + 1$ 的字符串 $s'$。举例来说，对于字符串 $s = \mathtt{abababc}$，其对应的 $s' = \mathtt{\#a\#b\#a\#b\#a\#b\#c\#}$。

对于字母间的 $\#$，其实际意义为 $s$ 中对应的“空”。而两端的 $\#$ 则是为了实现的方便。

注意到，在对 $s'$ 计算 $d_1[]$ 后，对于一个位置 $i$，$d_1[i]$ 所描述的最长的子回文串必定以 $\#$ 结尾（若以字母结尾，由于字母两侧必定各有一个 $\#$，因此可向外扩展一个得到一个更长的）。因此，对于 $s$ 中一个以字母为中心的极大子回文串，设其长度为 $m + 1$，则其在 $s'$ 中对应一个以相应字母为中心，长度为 $2m + 3$ 的极大子回文串；而对于 $s$ 中一个以空为中心的极大子回文串，设其长度为 $m$，则其在 $s'$ 中对应一个以相应表示空的 $\#$ 为中心，长度为 $2m + 1$ 的极大子回文串（上述两种情况下的 $m$ 均为偶数，但该性质成立与否并不影响结论）。综合以上观察及少许计算后易得，在 $s'$ 中，$d_1[i]$ 表示在 $s$ 中以对应位置为中心的极大子回文串的 **总长度加一**。

上述结论建立了 $s'$ 的 $d_1[]$ 同 $s$ 的 $d_1[]$ 和 $d_2[]$ 间的关系。

由于该统一处理本质上即求 $s'$ 的 $d_1[]$，因此在得到 $s'$ 后，代码同上节计算 $d_1[]$ 的一样。

## 练习题目

- [UVA #11475 **Extend to Palindrome"](https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=2470)
- [「国家集训队」最长双回文串](https://www.luogu.com.cn/problem/P4555)

* * *

## 习题

### 一般应用

> [!NOTE] **[AcWing 3188. manacher算法](https://www.acwing.com/problem/content/3190/)**
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

// 原串的任意一个子字符串都可以映射到新串的【长度为奇数】的字符串
// 原串的回文子串的长度x 对应新串的半径r-1  ==> x=r-1

const int N = 2e7 + 10;

int n;
char a[N], b[N];
int p[N];

void init() {
    int k = 0;
    b[k ++ ] = '$', b[k ++ ] = '#';
    for (int i = 0; i < n; ++ i )
        b[k ++ ] = a[i], b[k ++ ] = '#';
    b[k ++ ] = '^';
    n = k;
}

void manacher() {
    int mr = 0, mid;
    for (int i = 1; i < n; ++ i ) {
        if (i < mr)
            p[i] = min(p[mid * 2 - i], mr - i);
        else
            p[i] = 1;
        while (b[i - p[i]] == b[i + p[i]])
            p[i] ++ ;
        if (i + p[i] > mr) {
            mr = i + p[i];
            mid = i;
        }
    }
}

int main() {
    cin >> a;
    n = strlen(a);
    
    init();
    manacher();
    
    int res = 0;
    for (int i = 0; i < n; ++ i )
        res = max(res, p[i]);
    cout << res - 1 << endl;
    
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

> [!NOTE] **[LeetCode 5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)**
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
    string longestPalindrome(string s) {
        string ns = "$#";
        for (auto & c : s) {
            ns.push_back(c);
            ns.push_back('#');
        }
        int l = ns.size(), id = 0, mx = 0, maxidx = 0;
        vector<int> mp(l);
        for (int i = 1; i < l; ++ i ) {
            mp[i] = mx > i ? min(mp[2 * id - i], mx - i) : 1;
            while (ns[i - mp[i]] == ns[i + mp[i]]) ++ mp[i];
            if (i + mp[i] > mx) id = i, mx = i + mp[i];
            if (mp[i] > mp[maxidx]) maxidx = i;
        }
        string res;
        for (int i = maxidx - mp[maxidx] + 1; i <= maxidx + mp[maxidx] - 1; ++ i )
            if (ns[i] != '#') res.push_back(ns[i]);
        return res;
    }
};
```

##### **Python**

```python
# 把每个字母当成回文串的中心;这里要考虑两种情况，回文串的长度为奇数或者偶数情况。

class Solution:
    def longestPalindrome(self, s: str) -> str:
        self.res = ''
        n = len(s)

        def dfs(i, j):
            while i >= 0 and j < n and s[i] == s[j]:
                i -= 1
                j += 1
            self.res = max(self.res, s[i+1:j], key = len)
            
				# 注意：这样写 会timeout：只有其对应的两个位值的字符相等才会使 i j 发生改变，那么 如果 s[i] != s[j] ，是不是就会一直卡死在while循环出不去了呀
        #def dfs(i, j):
        #    while i >= 0 and j < n:
        #        if s[i] == s[j]:
        #            i -= 1
        #            j += 1
        #    self.res = max(self.res, s[i+1:j], key = len)
        
        for i in range(n):
            dfs(i, i)
            dfs(i, i + 1)
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)**
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

    int countSubstrings(string s) {
        string ms = "$#";
        for (auto c : s) ms.push_back(c), ms.push_back('#');

        int n = ms.size();
        vector<int> mp(n);
        int id = 0, mx = 0, maxid = 0;
        int res = 0;
        for (int i = 1; i < n; ++ i ) {
            mp[i] = i < mx ? min(mp[2 * id - i], mx - i) : 1;
            while (ms[i + mp[i]] == ms[i - mp[i]]) ++ mp[i];
            if (i + mp[i] > mx) {
                mx = i + mp[i];
                id = i;
            }
            // diff
            res += mp[i] / 2;
        }
        return res;
    }

    int countSubstrings_2(string s) {
        int n = s.size();
        vector<vector<bool>> f(n + 1, vector<bool>(n + 1));

        int res = n;
        for (int i = 1; i <= n; ++ i ) f[i][i] = 1;
        for (int len = 2; len <= n; ++ len )
            for (int l = 1; l + len - 1 <= n; ++ l ) {
                int r = l + len - 1;
                if (s[l - 1] == s[r - 1] && (l + 1 > r - 1 || f[l + 1][r - 1]))
                    f[l][r] = 1;
                res += f[l][r];
            }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        self.res = 0 
        n = len(s)

        def dfs(i, j):
            while i >= 0 and j < n and s[i] == s[j]:
                self.res += 1 
                i -= 1 
                j += 1 
            
        for i in range(n):
            dfs(i, i)
            dfs(i, i + 1)
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

### 进阶

> [!NOTE] **[LeetCode 1960. 两个回文子字符串长度的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/)**
> 
> [Biweekly-58]()
> 
> 题意: TODO

> [!TIP] **思路**
> 
>  【本题只需要求奇数长度的回文串 所以不用填充字符】
> 
> ```cpp
> p[0] = 1;	// 本题特殊处理
> ...
> while (i >= p[i] ...) // 需要加的特判
> ```
> 
> 结合【前缀和后缀分解】的思路 + 【双指针优化】
> 
> 算法实现来源于 yxc **学习并背过这种写法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;
    
    char b[N];
    int p[N];
    
    void manacher(int n) {
        p[0] = 1;
        int id = 0, mx = 0;
        for (int i = 1; i < n; ++ i ) {
            p[i] = mx > i ? min(p[2 * id - i], mx - i) : 1;
            while (i >= p[i] && b[i - p[i]] == b[i + p[i]])
                p[i] ++ ;
            if (i + p[i] > mx)
                id = i, mx = i + p[i];
        }
    }
    
    long long maxProduct(string s) {
        int n = s.size();
        for (int i = 0; i < n; ++ i )
            b[i] = s[i];
        b[n] = 0;
        
        manacher(n);
        
        vector<int> f(n), g(n);
        // i 指前缀下标
        // j 指当前扫到的中心
        // 非常巧妙的双指针优化
        for (int i = 0, j = 0, mx = 0; i < n; ++ i ) {
            while (j + p[j] - 1 < i) {
                mx = max(mx, p[j]);
                j ++ ;
            }
            mx = max(mx, i - j + 1);
            f[i] = mx;
        }
        for (int i = n - 1, j = n - 1, mx = 0; i >= 0; -- i ) {
            while (j - p[j] + 1 > i) {
                mx = max(mx, p[j]);
                j -- ;
            }
            mx = max(mx, j - i + 1);
            g[i] = mx;
        }
        
        LL res = 0;
        for (int i = 0; i < n - 1; ++ i )
            res = max(res, (LL)(f[i] * 2 - 1) * (g[i + 1] * 2 - 1));
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