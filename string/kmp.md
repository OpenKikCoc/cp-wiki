
### 在字符串中查找子串：Knuth-Morris-Pratt 算法

该算法由 Knuth、Pratt 和 Morris 在 1977 年共同发布<sup>[\[1\]](https://epubs.siam.org/doi/abs/10.1137/0206024)</sup>。

该任务是前缀函数的一个典型应用。

给定一个文本 $t$ 和一个字符串 $s$，我们尝试找到并展示 $s$ 在 $t$ 中的所有出现（occurrence）。

为了简便起见，我们用 $n$ 表示字符串 $s$ 的长度，用 $m$ 表示文本 $t$ 的长度。

我们构造一个字符串 $s + \# + t$，其中 $\#$ 为一个既不出现在 $s$ 中也不出现在 $t$ 中的分隔符。接下来计算该字符串的前缀函数。现在考虑该前缀函数除去最开始 $n + 1$ 个值（即属于字符串 $s$ 和分隔符的函数值）后其余函数值的意义。根据定义，$\pi[i]$ 为右端点在 $i$ 且同时为一个前缀的最长真子串的长度，具体到我们的这种情况下，其值为与 $s$ 的前缀相同且右端点位于 $i$ 的最长子串的长度。由于分隔符的存在，该长度不可能超过 $n$。而如果等式 $\pi[i] = n$ 成立，则意味着 $s$ 完整出现在该位置（即其右端点位于位置 $i$）。注意该位置的下标是对字符串 $s + \# + t$ 而言的。

因此如果在某一位置 $i$ 有 $\pi[i] = n$ 成立，则字符串 $s$ 在字符串 $t$ 的 $i - (n - 1) - (n + 1) = i - 2n$ 处出现。

正如在前缀函数的计算中已经提到的那样，如果我们知道前缀函数的值永远不超过一特定值，那么我们不需要存储整个字符串以及整个前缀函数，而只需要二者开头的一部分。在我们这种情况下这意味着只需要存储字符串 $s + \#$ 以及相应的前缀函数值即可。我们可以一次读入字符串 $t$ 的一个字符并计算当前位置的前缀函数值。

因此 Knuth-Morris-Pratt 算法（简称 KMP 算法）用 $O(n + m)$ 的时间以及 $O(n)$ 的内存解决了该问题。

### 字符串的周期

对字符串 $s$ 和 $0 < p \le |s|$，若 $s[i] = s[i+p]$ 对所有 $i \in [0, |s| - p - 1]$ 成立，则称 $p$ 是 $s$ 的周期。

对字符串 $s$ 和 $0 \le r < |s|$，若 $s$ 长度为 $r$ 的前缀和长度为 $r$ 的后缀相等，就称 $s$ 长度为 $r$ 的前缀是 $s$ 的 border。

由 $s$ 有长度为 $r$ 的 border 可以推导出 $|s|-r$ 是 $s$ 的周期。

根据前缀函数的定义，可以得到 $s$ 所有的 border 长度，即 $\pi[n-1],\pi[\pi[n-1]-1], \ldots$。[^ref1]

所以根据前缀函数可以在 $O(n)$ 的时间内计算出 $s$ 所有的周期。其中，由于 $\pi[n-1]$ 是 $s$ 最长 border 的长度，所以 $n - \pi[n-1]$ 是 $s$ 的最小周期。

### 统计每个前缀的出现次数

在该节我们将同时讨论两个问题。给定一个长度为 $n$ 的字符串 $s$，在问题的第一个变种中我们希望统计每个前缀 $s[0 \dots i]$ 在同一个字符串的出现次数，在问题的第二个变种中我们希望统计每个前缀 $s[0 \dots i]$ 在另一个给定字符串 $t$ 中的出现次数。

首先让我们来解决第一个问题。考虑位置 $i$ 的前缀函数值 $\pi[i]$。根据定义，其意味着字符串 $s$ 一个长度为 $\pi[i]$ 的前缀在位置 $i$ 出现并以 $i$ 为右端点，同时不存在一个更长的前缀满足前述定义。与此同时，更短的前缀可能以该位置为右端点。容易看出，我们遇到了在计算前缀函数时已经回答过的问题：给定一个长度为 $j$ 的前缀，同时其也是一个右端点位于 $i$ 的后缀，下一个更小的前缀长度 $k < j$ 是多少？该长度的前缀需同时也是一个右端点为 $i$ 的后缀。因此以位置 $i$ 为右端点，有长度为 $\pi[i]$ 的前缀，有长度为 $\pi[\pi[i] - 1]$ 的前缀，有长度为 $\pi[\pi[\pi[i] - 1] - 1]$ 的前缀，等等，直到长度变为 $0$。故而我们可以通过下述方式计算答案。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
vector<int> ans(n + 1);
for (int i = 0; i < n; i++) ans[pi[i]]++;
for (int i = n - 1; i > 0; i--) ans[pi[i - 1]] += ans[i];
for (int i = 0; i <= n; i++) ans[i]++;
```

###### **Python**

```python
# Python Version
ans = [0] * (n + 1)
for i in range(0, n):
    ans[pi[i]] += 1
for i in range(n - 1, 0, -1):
    ans[pi[i - 1]] += ans[i]
for i in range(0, n + 1):
    ans[i] += 1
```

<!-- tabs:end -->
</details>

<br>

在上述代码中我们首先统计每个前缀函数值在数组 $\pi$ 中出现了多少次，然后再计算最后答案：如果我们知道长度为 $i$ 的前缀出现了恰好 $\text{ans}[i]$ 次，那么该值必须被叠加至其最长的既是后缀也是前缀的子串的出现次数中。在最后，为了统计原始的前缀，我们对每个结果加 $1$。

现在考虑第二个问题。我们应用来自 Knuth-Morris-Pratt 的技巧：构造一个字符串 $s + \# + t$ 并计算其前缀函数。与第一个问题唯一的不同之处在于，我们只关心与字符串 $t$ 相关的前缀函数值，即 $i \ge n + 1$ 的 $\pi[i]$。有了这些值之后，我们可以同样应用在第一个问题中的算法来解决该问题。

### 一个字符串中本质不同子串的数目

给定一个长度为 $n$ 的字符串 $s$，我们希望计算其本质不同子串的数目。

我们将迭代的解决该问题。换句话说，在知道了当前的本质不同子串的数目的情况下，我们要找出一种在 $s$ 末尾添加一个字符后重新计算该数目的方法。

令 $k$ 为当前 $s$ 的本质不同子串数量。我们添加一个新的字符 $c$ 至 $s$。显然，会有一些新的子串以字符 $c$ 结尾。我们希望对这些以该字符结尾且我们之前未曾遇到的子串计数。

构造字符串 $t = s + c$ 并将其反转得到字符串 $t^{\sim}$。现在我们的任务变为计算有多少 $t^{\sim}$ 的前缀未在 $t^{\sim}$ 的其余任何地方出现。如果我们计算了 $t^{\sim}$ 的前缀函数最大值 $\pi_{\max}$，那么最长的出现在 $s$ 中的前缀其长度为 $\pi_{\max}$。自然的，所有更短的前缀也出现了。

因此，当添加了一个新字符后新出现的子串数目为 $|s| + 1 - \pi_{\max}$。

所以对于每个添加的字符，我们可以在 $O(n)$ 的时间内计算新子串的数目，故最终复杂度为 $O(n^2)$。

值得注意的是，我们也可以重新计算在头部添加一个字符，或者从尾或者头移除一个字符时的本质不同子串数目。

### 字符串压缩

给定一个长度为 $n$ 的字符串 $s$，我们希望找到其最短的“压缩”表示，也即我们希望寻找一个最短的字符串 $t$，使得 $s$ 可以被 $t$ 的一份或多份拷贝的拼接表示。

显然，我们只需要找到 $t$ 的长度即可。知道了该长度，该问题的答案即为长度为该值的 $s$ 的前缀。

让我们计算 $s$ 的前缀函数。通过使用该函数的最后一个值 $\pi[n - 1]$，我们定义值 $k = n - \pi[n - 1]$。我们将证明，如果 $k$ 整除 $n$，那么 $k$ 就是答案，否则不存在一个有效的压缩，故答案为 $n$。

假定 $n$ 可被 $k$ 整除。那么字符串可被划分为长度为 $k$ 的若干块。根据前缀函数的定义，该字符串长度为 $n - k$ 的前缀等于其后缀。但是这意味着最后一个块同倒数第二个块相等，并且倒数第二个块同倒数第三个块相等，等等。作为其结果，所有块都是相等的，因此我们可以将字符串 $s$ 压缩至长度 $k$。

诚然，我们仍需证明该值为最优解。实际上，如果有一个比 $k$ 更小的压缩表示，那么前缀函数的最后一个值 $\pi[n - 1]$ 必定比 $n - k$ 要大。因此 $k$ 就是答案。

现在假设 $n$ 不可以被 $k$ 整除，我们将通过反证法证明这意味着答案为 $n$[^1]。假设其最小压缩表示 $r$ 的长度为 $p$（$p$ 整除 $n$），字符串 $s$ 被划分为 $n / p \ge 2$ 块。那么前缀函数的最后一个值 $\pi[n - 1]$ 必定大于 $n - p$（如果等于则 $n$ 可被 $k$ 整除），也即其所表示的后缀将部分的覆盖第一个块。现在考虑字符串的第二个块。该块有两种解释：第一种为 $r_0 r_1 \dots r_{p - 1}$，另一种为 $r_{p - k} r_{p - k + 1} \dots r_{p - 1} r_0 r_1 \dots r_{p - k - 1}$。由于两种解释对应同一个字符串，因此可得到 $p$ 个方程组成的方程组，该方程组可简写为 $r_{(i + k) \bmod p} = r_{i \bmod p}$，其中 $\cdot \bmod p$ 表示模 $p$ 意义下的最小非负剩余。

$$
\begin{gathered}
\overbrace{r_0 ~ r_1 ~ r_2 ~ r_3 ~ r_4 ~ r_5}^p ~ \overbrace{r_0 ~ r_1 ~ r_2 ~ r_3 ~ r_4 r_5}^p \\
r_0 ~ r_1 ~ r_2 ~ r_3 ~ \underbrace{\overbrace{r_0 ~ r_1 ~ r_2 ~ r_3 ~ r_4 ~ r_5}^p ~ r_0 ~ r_1}_{\pi[11] = 8}
\end{gathered}
$$

根据扩展欧几里得算法我们可以得到一组 $x$ 和 $y$ 使得 $xk + yp = \gcd(k, p)$。通过与等式 $pk - kp = 0$ 适当叠加我们可以得到一组 $x' > 0$ 和 $y' < 0$ 使得 $x'k + y'p = \gcd(k, p)$。这意味着通过不断应用前述方程组中的方程我们可以得到新的方程组 $r_{(i + \gcd(k, p)) \bmod p} = r_{i \bmod p}$。

由于 $\gcd(k, p)$ 整除 $p$，这意味着 $\gcd(k, p)$ 是 $r$ 的一个周期。又因为 $\pi[n - 1] > n - p$，故有 $n - \pi[n - 1] = k < p$，所以 $\gcd(k, p)$ 是一个比 $p$ 更小的 $r$ 的周期。因此字符串 $s$ 有一个长度为 $\gcd(k, p) < p$ 的压缩表示，同 $p$ 的最小性矛盾。

综上所述，不存在一个长度小于 $n$ 的压缩表示，因此答案为 $n$。

[^1]: 在俄文版及英文版中该部分证明均疑似有误。本文章中的该部分证明由作者自行添加。

### 根据前缀函数构建一个自动机

让我们重新回到通过一个分隔符将两个字符串拼接的新字符串。对于字符串 $s$ 和 $t$ 我们计算 $s + \# + t$ 的前缀函数。显然，因为 $\#$ 是一个分隔符，前缀函数值永远不会超过 $|s|$。因此我们只需要存储字符串 $s + \#$ 和其对应的前缀函数值，之后就可以动态计算对于之后所有字符的前缀函数值：

$$
\underbrace{s_0 ~ s_1 ~ \dots ~ s_{n-1} ~ \#}_{\text{need to store}} ~ \underbrace{t_0 ~ t_1 ~ \dots ~ t_{m-1}}_{\text{do not need to store}}
$$

实际上在这种情况下，知道 $t$ 的下一个字符 $c$ 以及之前位置的前缀函数值便足以计算下一个位置的前缀函数值，而不需要用到任何其它 $t$ 的字符和对应的前缀函数值。

换句话说，我们可以构造一个 **自动机**（一个有限状态机）：其状态为当前的前缀函数值，而从一个状态到另一个状态的转移则由下一个字符确定。

因此，即使没有字符串 $t$，我们同样可以应用构造转移表的算法构造一个转移表 $( \text { old } \pi , c ) \rightarrow \text { new } _ { - } \pi$：

```cpp
void compute_automaton(string s, vector<vector<int>>& aut) {
    s += '#';
    int n = s.size();
    vector<int> pi = prefix_function(s);
    aut.assign(n, vector<int>(26));
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 26; c++) {
            int j = i;
            while (j > 0 && 'a' + c != s[j]) j = pi[j - 1];
            if ('a' + c == s[j]) j++;
            aut[i][c] = j;
        }
    }
}
```

然而在这种形式下，对于小写字母表，算法的时间复杂度为 $O(|\Sigma|n^2)$。注意到我们可以应用动态规划来利用表中已计算过的部分。只要我们从值 $j$ 变化到 $\pi[j - 1]$，那么我们实际上在说转移 $(j, c)$ 所到达的状态同转移 $(\pi[j - 1], c)$ 一样，但该答案我们之前已经精确计算过了。

```cpp
void compute_automaton(string s, vector<vector<int>>& aut) {
    s += '#';
    int n = s.size();
    vector<int> pi = prefix_function(s);
    aut.assign(n, vector<int>(26));
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 26; c++) {
            if (i > 0 && 'a' + c != s[i])
                aut[i][c] = aut[pi[i - 1]][c];
            else
                aut[i][c] = i + ('a' + c == s[i]);
        }
    }
}
```

最终我们可在 $O(|\Sigma|n)$ 的时间复杂度内构造该自动机。

该自动机在什么时候有用呢？首先，记得大部分时候我们为了一个目的使用字符串 $s + \# + t$ 的前缀函数：寻找字符串 $s$ 在字符串 $t$ 中的所有出现。

因此使用该自动机的最直接的好处是 **加速计算字符串 $s + \# + t$ 的前缀函数**。

通过构建 $s + \#$ 的自动机，我们不再需要存储字符串 $s$ 以及其对应的前缀函数值。所有转移已经在表中计算过了。

但除此以外，还有第二个不那么直接的应用。我们可以在字符串 $t$ 是 **某些通过一些规则构造的巨型字符串** 时，使用该自动机加速计算。Gray 字符串，或者一个由一些短的输入串的递归组合所构造的字符串都是这种例子。

出于完整性考虑，我们来解决这样一个问题：给定一个数 $k \le 10^5$，以及一个长度 $\le 10^5$ 的字符串 $s$，我们需要计算 $s$ 在第 $k$ 个 Gray 字符串中的出现次数。回想起 Gray 字符串以下述方式定义：

$$
\begin{aligned}
g_1 &= \mathtt{a}\\
g_2 &= \mathtt{aba}\\
g_3 &= \mathtt{abacaba}\\
g_4 &= \mathtt{abacabadabacaba}
\end{aligned}
$$

由于其天文数字般的长度，在这种情况下即使构造字符串 $t$ 都是不可能的：第 $k$ 个 Gray 字符串有 $2^k - 1$ 个字符。然而我们可以在仅仅知道开头若干前缀函数值的情况下，有效计算该字符串末尾的前缀函数值。

除了自动机之外，我们同时需要计算值 $G[i][j]$：在从状态 $j$ 开始处理 $g_i$ 后的自动机的状态，以及值 $K[i][j]$：当从状态 $j$ 开始处理 $g_i$ 后，$s$ 在 $g_i$ 中的出现次数。实际上 $K[i][j]$ 为在执行操作时前缀函数取值为 $|s|$ 的次数。易得问题的答案为 $K[k][0]$。

我们该如何计算这些值呢？首先根据定义，初始条件为 $G[0][j] = j$ 以及 $K[0][j] = 0$。之后所有值可以通过先前的值以及使用自动机计算得到。为了对某个 $i$ 计算相应值，回想起字符串 $g_i$ 由 $g_{i - 1}$，字母表中第 $i$ 个字符，以及 $g_{i - 1}$ 三者拼接而成。因此自动机会途径下列状态：

$$
\begin{gathered}
\text{mid} = \text{aut}[G[i - 1][j]][i] \\
G[i][j] = G[i - 1][\text{mid}]
\end{gathered}
$$

$K[i][j]$ 的值同样可被简单计算。

$$
K[i][j] = K[i - 1][j] + [\text{mid} == |s|] + K[i - 1][\text{mid}]
$$

其中 $[\cdot]$ 当其中表达式取值为真时值为 $1$，否则为 $0$。综上，我们已经可以解决关于 Gray 字符串的问题，以及一大类与之类似的问题。举例来说，应用同样的方法可以解决下列问题：给定一个字符串 $s$ 以及一些模式 $t_i$，其中每个模式以下列方式给出：该模式由普通字符组成，当中可能以 $t_{k}^{\text{cnt}}$ 的形式递归插入先前的字符串，也即在该位置我们必须插入字符串 $t_k$ $\text{cnt}$ 次。以下是这些模式的一个例子：

$$
\begin{aligned}
t_1 &= \mathtt{abdeca} \\
t_2 &= \mathtt{abc} + t_1^{30} + \mathtt{abd} \\
t_3 &= t_2^{50} + t_1^{100} \\
t_4 &= t_2^{10} + t_3^{100}
\end{aligned}
$$

递归代入会使字符串长度爆炸式增长，他们的长度甚至可以达到 $100^{100}$ 的数量级。而我们必须找到字符串 $s$ 在每个字符串中的出现次数。

该问题同样可通过构造前缀函数的自动机解决。同之前一样，我们利用先前计算过的结果对每个模式计算其转移然后相应统计答案即可。

## 练习题目

- [UVA 455 "Periodic Strings"](http://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=396)
- [UVA 11022 "String Factoring"](http://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=1963)
- [UVA 11452 "Dancing the Cheeky-Cheeky"](http://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=2447)
- [UVA 12604 - Caesar Cipher](https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=4282)
- [UVA 12467 - Secret Word](https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=3911)
- [UVA 11019 - Matrix Matcher](https://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=1960)
- [SPOJ - Pattern Find](http://www.spoj.com/problems/NAJPF/)
- [Codeforces - Anthem of Berland](http://codeforces.com/contest/808/problem/G)
- [Codeforces - MUH and Cube Walls](http://codeforces.com/problemset/problem/471/D)

## 参考资料与注释

[^ref1]: [金策 - 字符串算法选讲](https://wenku.baidu.com/view/850f93f4fbb069dc5022aaea998fcc22bcd1433e.html)


## 习题

### 一般 kmp

> [!NOTE] **[AcWing 831. KMP字符串](https://www.acwing.com/problem/content/833/)**
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

int n, m;
char p[N], s[N];
int f[N];

void getnxt() {
    f[0] = f[1] = 0;
    for (int i = 1; i < n; ++ i ) {
        int j = f[i];
        while (j && p[j] != p[i]) j = f[j];
        if (p[j] == p[i]) f[i + 1] = j + 1;
        else f[i + 1] = 0;
    }
}

int main() {
    scanf("%d%s", &n, p);
    
    getnxt();
    
    scanf("%d%s", &m, s);
    
    int j = 0;
    for (int i = 0; i < m; ++ i ) {
        while (j && p[j] != s[i]) j = f[j];
        if (p[j] == s[i]) ++ j ;
        if (j == n) {
            cout << i - n + 1 << ' ';
            j = f[j];
        }
    }
    cout << endl;
    return 0;
}
```

##### **Python**

```python
"""
next数组存的是：字符串的前缀集合与后缀集合的交集中最长元素的长度
KMP 的核心思想：在每次匹配失败时，不是把p串往后移一位，而是把p串往后移动到下一次可以和前面部分匹配的位置的最大的地方，这样就可以跳过大多数的不匹配的步骤；而每次p串移动的步数就是通过查找next数组来确定的。

"""
if __name__ == '__main__':
    n = int(input())
    # 加一个' ', 使得下标从1开始
    p = ' ' + input()
    m = int(input())
    s = ' ' + input()
    # 待匹配字符的next数组
    ne = [0] * 10010

    # 求next数组
    j = 0
    # 注意！！next数组下标从1开始，next[1]=0，所以计算next数组的时候下标从2开始循环！（踩坑）
    for i in range(2, n + 1):
        while j and p[i] != p[j + 1]:
            j = ne[j]
        if p[i] == p[j + 1]:
            j += 1
        ne[i] = j

    # 字符串匹配
    j = 0
    for i in range(1, m + 1):
        while j and s[i] != p[j + 1]:
            j = ne[j]
        if s[i] == p[j + 1]:
            j += 1
        if j == n:
            print(i - j + 1 - 1, end=' ')
            j = ne[j]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)**
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
class Solution {
public:
    int n, m;

    vector<int> get(string & p) {
        vector<int> f(n + 1);
        for (int i = 2, j = 0; i <= n; ++ i ) {
            while (j && p[i] != p[j + 1])
                j = f[j];
            if (p[i] == p[j + 1])
                j ++ ;
            f[i] = j;
        }
        return f;
    }

    int strStr(string haystack, string needle) {
        if (needle.empty())
            return 0;

        // 以下无法处理两个都是空串or第二个是空串的情况
        string p = ' ' + needle, s = ' ' + haystack;
        this->n = p.size() - 1;
        this->m = s.size() - 1;
        auto f = get(p);
        for (int i = 1, j = 0; i <= m; ++ i ) {
            while (j && s[i] != p[j + 1])
                j = f[j];
            if (s[i] == p[j + 1])
                j ++ ;
            if (j == n) {
                return i - n;
                // j = f[j];
            }
        }
        return -1;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int strStr(string s, string p) {
        if (p.empty()) return 0;
        int n = s.size(), m = p.size();
        s = ' ' + s, p = ' ' + p;

        vector<int> next(m + 1);
        for (int i = 2, j = 0; i <= m; i ++ ) {
            while (j && p[i] != p[j + 1]) j = next[j];
            if (p[i] == p[j + 1]) j ++ ;
            next[i] = j;
        }

        for (int i = 1, j = 0; i <= n; i ++ ) {
            while (j && s[i] != p[j + 1]) j = next[j];
            if (s[i] == p[j + 1]) j ++ ;
            if (j == m) return i - m;
        }

        return -1;
    }
};
```

##### **Python**

```python
# # 这竟然是一道KMP的裸题!!!我竟然没想到！！！
class Solution:
    def strStr(self, s: str, p: str) -> int:
        if not p:return 0 # 当输入 s 和 p 都是空字符串时，特判。
        n, m  = len(s), len(p)
        s, p  = ' ' + s, ' ' + p
        ne = [0] * (m + 1)

        j = 0 
        for i in range(2, m + 1):
            while j and p[i] != p[j + 1]:
                j = ne[j]
            if p[i] == p[j + 1]:
                j += 1
            ne[i] = j 
        
        j = 0
        for i in range(1, n + 1):
            while j and s[i] != p[j + 1]:
                j = ne[j]
            if s[i] == p[j + 1]:
                j += 1
            if j == m:
                return i -j
        return -1

# 法2 不用KMP求解
class Solution:
    def strStr(self, s: str, p: str) -> int:
        n, m = len(s), len(p)
        for p1 in range(n - m + 1):
            if s[p1:p1 + m] == p:
                return p1 
        else:return -1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 686. 重复叠加字符串匹配](https://leetcode-cn.com/problems/repeated-string-match/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    int repeatedStringMatch(string a, string p) {
        string s;
        while (s.size() < p.size()) s += a;
        s += a;
        int n = s.size(), m = p.size();
        s = ' ' + s, p = ' ' + p;

        vector<int> next(m + 1);
        for (int i = 2, j = 0; i <= m; i ++ ) {
            while (j && p[i] != p[j + 1]) j = next[j];
            if (p[i] == p[j + 1]) j ++ ;
            next[i] = j;
        }
        for (int i = 1, j = 0; i <= n; i ++ ) {
            while (j && s[i] != p[j + 1]) j = next[j];
            if (s[i] == p[j + 1]) j ++ ;
            if (j == m) return (i + a.size() - 1) / a.size();
        }
        return -1;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    vector<int> get_next(string p, int n) {
        vector<int> f;
        f.push_back(0), f.push_back(0);
        for (int i = 1; i < n; ++ i ) {
            int j = f[i];
            if (j && p[j] != p[i]) j = f[j];
            if (p[j] == p[i])
                f.push_back(j + 1);
            else
                f.push_back(0);
        }
        return f;
    }

    int repeatedStringMatch(string a, string p) {
        string s;
        while (s.size() < p.size()) s += a;
        s += a;

        int n = s.size(), m = p.size();
        auto f = get_next(p, m);

        int j = 0;
        for (int i = 0; i < n; ++ i ) {
            while (j && p[j] != s[i]) j = f[j];
            if (p[j] == s[i])
                ++ j ;
            if (j == m) {
                return (i + a.size()) / a.size();
            }
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

### 循环节

> [!NOTE] **[LeetCode 459. 重复的子字符串]()**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> kmp 循环节问题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        s = ' ' + s;
        vector<int> next(n + 1);
        for (int i = 2, j = 0; i <= n; i ++ ) {
            while (j && s[i] != s[j + 1]) j = next[j];
            if (s[i] == s[j + 1]) j ++ ;
            next[i] = j;
        }
        int t = n - next[n];
        return t < n && n % t == 0;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    // kmp 循环节问题
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        vector<int> f;
        f.push_back(0), f.push_back(0);
        for (int i = 1; i < n; ++ i ) {
            int j = f[i];
            while (j && s[j] != s[i]) j = f[j];
            if (s[j] == s[i]) f.push_back(j + 1);
            else f.push_back(0);
        }
        return f[n] && n % (n - f[n]) == 0;    
    }
};
```
##### **Python**

```python
# 暴力解法1：就是找前缀，看s是否能有几个这样前缀组成。
# class Solution:
#     def repeatedSubstringPattern(self, s: str) -> bool:
#         n = len(s)
#         for i in range(1, len(s) // 2 + 1):
#             a, b = divmod(n, i)
#             if b == 0 and s[:i] * a  == s:
#                 return True
#         return False

# 暴力解法2：我们知道如果s是重复字符串，那么可以由两个子串组成。我们通过ss = s + s就有4个子串组成，
# 删除首尾字母，那么 ss[1:-1]就有应该有2个子串组成，就是说ss[1:-1]是否存在s
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return (s+s)[1:-1].find(s) != -1
      
# 解法3：KMP
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 466. 统计重复个数](https://leetcode-cn.com/problems/count-the-repetitions/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一般类似这样的大规模数据都考虑能否【倍增】或【循环节】
> 
> 本题即为循环节优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 循环节
    int getMaxRepetitions(string s1, int n1, string s2, int n2) {
        vector<int> cnt;
        // 匹配完后是匹配了s2的多少个字符
        unordered_map<int, int> hash;
        for (int i = 0, k = 0; i < n1; ++ i ) {
            for (int j = 0; j < s1.size(); ++ j )
                if (s1[j] == s2[k % s2.size()]) ++ k ;
            cnt.push_back(k);
            // 判断当前余数有没有出现过
            if (hash.count(k % s2.size())) {    // 出现循环节
                int a = i - hash[k % s2.size()];        // 循环节中有多少个s1
                int b = k - cnt[hash[k % s2.size()]];   // 循环节中匹配了多少个字符
                int res = (n1 - i - 1) / a * b;
                // 不完整部分
                for (int u = 0; u < (n1 - i - 1) % a; ++ u )
                    for (int j = 0; j < s1.size(); ++ j )
                        if (s1[j] == s2[k % s2.size()])
                            ++ k ;
                res += k;
                return res / s2.size() / n2;
            }
            hash[k % s2.size()] = i;
        }
        if (cnt.empty()) return 0;
        return cnt.back() / s2.size() / n2;
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

### 构造

> [!NOTE] **[LeetCode 214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本质求 s 的最长回文前缀，所以拼接求 next 数组即可。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10;  // 2 * 5e4

    int f[N];

    int get(string s) {
        memset(f, 0, sizeof f); // f[0] = f[1] = 0;
        int n = s.size();
        s = ' ' + s;
        for (int i = 2, j = 0; i <= n; ++ i ) {
            if (j && s[i] != s[j + 1])
                j = f[j];
            if (s[i] == s[j + 1])
                j ++ ;
            f[i] = j;
        }
        return f[n];
    }

    string shortestPalindrome(string s) {
        int n = s.size();
        string rs(s.rbegin(), s.rend());
        auto p = s + '#' + rs;
        int idx = get(p);
        string res;
        for (int i = n - 1; i >= idx; -- i )
            res.push_back(s[i]);
        return res + s;
    }
};
```

##### **Python**

```python
# 反转拼接用kmp求解相等前后缀，找出了最长回文前缀
"""
(贪心，KMP) O(n)
求字符串 s 的最长回文前缀，然后剩余的部分就可以逆序拼接到 s 的最前边得到一个回文串。例如 abcbabcab 的最长回文前缀是 abcba，则答案就是 bacb + abcba + bcab = bacbabcbabcab。
首先将原串逆序复制一份，得到字符串 t。
将 s + # + t 作为新字符串，求其 next 数组。
假设下标从 0 开始，则最后位置上的 next 值加 1 就是最长回文前缀的长度，假设重合长度为 l。
最终答案为 t[0:l] + s。
"""
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        p = " " + s + "#" + s[::-1]
        ne = [0 for _ in range(len(p))]
        j = 0
        for i in range(2, len(p)):
            while j and p[j + 1] != p[i]:
                j = ne[j]
            if p[j + 1] == p[i]:
                j += 1
            ne[i] = j
        return s[:ne[-1] - 1:-1] + s
```

<!-- tabs:end -->
</details>

<br>

* * *