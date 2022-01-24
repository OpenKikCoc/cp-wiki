

定义一个合法括号序列（balanced bracket sequence）为仅由 $($ 和 $)$ 构成的字符串且：

- 空串 $\varepsilon$ 是一个合法括号序列。
- 如果 $s$ 是合法括号序列，那么 $(s)$ 也是合法括号序列。
- 如果 $s,t$ 都是合法括号序列，那么 $st$ 也是合法括号序列。

例如 $(())()$ 是合法括号序列，而 $)()$ 不是。

有时候会有多种不同的括号，如 $[()]\{\}$。这样的变种括号序列与朴素括号序列有相似的定义。

本文将会介绍与括号序列相关的经典问题。

注：英语中一般称左括号为 opening bracket，而右括号是 closing bracket。

## 判断是否合法

判断 $s$ 是否为合法括号序列的经典方法是贪心思想。该算法同样适用于变种括号序列。

我们维护一个栈，对于 $i=1,2,\ldots,|s|$ 依次考虑：

- 如果 $s_i$ 是右括号且栈非空且栈顶元素是 $s_i$ 对应的左括号，就弹出栈顶元素。
- 若不满足上述条件，则将 $s_i$ 压入栈中。

在遍历整个 $s$ 后，若栈是空的，那么 $s$ 就是合法括号序列，否则就不是。时间复杂度 $O(n)$。

## 合法括号序列计数

考虑求出长度为 $2n$ 的合法括号序列 $s$ 的个数 $f_n$。不妨枚举与 $s_1$ 匹配的括号的位置，假设是 $2i+2$。它将整个序列又分成了两个更短的合法括号序列。因此

$$
f_n=\sum_{i=0}^{n-1}f_if_{n-i-1}
$$

这同样是卡特兰数的递推式。也就是说 $f_n=\frac{1}{n+1}\binom{2n}{n}$。

当然，对于变种合法括号序列的计数，方法是类似的。假设有 $k$ 种不同类型的括号，那么有 $f'_n=\frac{1}{n+1}\binom{2n}{n}k^n$。

## 字典序后继

给出合法的括号序列 $s$，我们要求出按字典序升序排序的长度为 $|s|$ 的所有合法括号序列中，序列 $s$ 的下一个合法括号序列。在本问题中，我们认为左括号的字典序小于右括号，且不考虑变种括号序列。

我们需要找到一个最大的 $i$ 使得 $s_i$ 是左括号。然后，将其变成右括号，并将 $s[i+1,|s|]$ 这部分重构一下。另外，$i$ 必须满足：$s[1,i-1]$ 中左括号的数量 **大于** 右括号的数量。

不妨设当 $s_i$ 变成右括号后，$s[1,i]$ 中左括号比右括号多了 $k$ 个。那么我们就让 $s$ 的最后 $k$ 个字符变成右括号，而 $s[i+1,|s|-k]$ 则用 $((\dots(())\dots))$ 的形式填充即可，因为这样填充的字典序最小。

该算法的时间复杂度是 $O(n)$。


```cpp
bool next_balanced_sequence(string& s) {
    int n = s.size();
    int depth = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (s[i] == '(')
            depth--;
        else
            depth++;

        if (s[i] == '(' && depth > 0) {
            depth--;
            int open = (n - i - 1 - depth) / 2;
            int close = n - i - 1 - open;
            string next =
                s.substr(0, i) + ')' + string(open, '(') + string(close, ')');
            s.swap(next);
            return true;
        }
    }
    return false;
}
```

## 字典序计算

给出合法的括号序列 $s$，我们要求出它的字典序排名。

考虑求出字典序比 $s$ 小的括号序列 $p$ 的个数。

不妨设 $p_i<s_i$ 且 $\forall 1\le j<i,p_j=s_i$。显然 $p_i$ 是左括号而 $s_i$ 是右括号。枚举 $i$（满足 $s_i$ 为右括号），假设 $p[1,i]$ 中左括号比右括号多 $k$ 个，那么相当于我们要统计长度为 $|s|-i$ 且存在 $k$ 个未匹配的右括号且不存在未匹配的左括号的括号序列的个数。

不妨设 $f(i,j)$ 表示长度为 $i$ 且存在 $j$ 个未匹配的右括号且不存在未匹配的左括号的括号序列的个数。

通过枚举括号序列第一个字符是什么，可以得到 $f$ 的转移：$f(i,j) = f(i-1,j-1)+f(i-1,j+1)$。初始时 $f(0,0)=1$。其实 $f$ 是 [OEIS - A053121](http://oeis.org/A053121)。

这样我们就可以 $O(|s|^2)$ 计算字典序了。

对于变种括号序列，方法是类似的，只不过我们需要对每个 $s_i$ 考虑比它小的那些字符进行计算（在上述算法中因为不存在比左括号小的字符，所以我们只考虑了 $s_i$ 为右括号的情况）。

另外，利用 $f$ 数组，我们同样可以求出字典序排名为 $k$ 的合法括号序列。


## 习题

> [!NOTE] **[LeetCode 678. 有效的括号字符串](https://leetcode-cn.com/problems/valid-parenthesis-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trick 包含 `*` 符号的括号匹配
> 
> 【思维 维护上下界】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool checkValidString(string s) {
        // 栈存的都是左括号 所以本质上只用常量存左括号数量即可
        // * 考虑其影响左括号数量的范围
        // low high 存左括号数量范围
        int low = 0, high = 0;
        for (auto c : s) {
            if (c == '(')
                ++ low , ++ high ;
            else if (c == ')')
                -- low , -- high ;
            else
                -- low , ++ high ;
            low = max(low, 0);
            if (low > high)
                return false;
        }
        return !low;
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


> [!NOTE] **[LeetCode 761. 特殊的二进制序列](https://leetcode-cn.com/problems/special-binary-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 递归 思维题 类似括号匹配

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string makeLargestSpecial(string S) {
        if (S.size() <= 2)
            return S;

        vector<string> q;
        string s;
        int cnt = 0;
        for (auto c : S) {
            s.push_back(c);
            if (c == '1')
                cnt ++ ;
            else {
                cnt -- ;
                if (cnt == 0) {
                    q.push_back('1' + makeLargestSpecial(s.substr(1, s.size() - 2)) + '0');
                    s.clear();
                }
            }
        }
        sort(q.begin(), q.end(), [](string & a, string & b) {
            return a + b > b + a;
        });
        string res;
        for (auto s : q)
            res += s;
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