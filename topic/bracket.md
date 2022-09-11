

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

> [!NOTE] **[LeetCode 2116. 判断一个括号字符串是否有效](https://leetcode-cn.com/problems/check-if-a-parentheses-string-can-be-valid/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 678 进阶

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
 public:
     bool canBeValid(string s, string locked) {
         int n = s.size();
         for (int i = 0; i < n; ++ i )
             if (locked[i] == '0')
                 s[i] = '*';
         
         int low = 0, high = 0;
         for (auto c : s) {
             if (c == '(')
                 low ++ , high ++ ;
             else if (c == ')')
                 low -- , high -- ;
             else
                 low -- , high ++ ;
             low = max(low, 0);
             if (low > high)
                 return false;
         }
         return !low && n % 2 == 0;
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

> [!NOTE] **[LeetCode 1963. 使字符串平衡的最小交换次数](https://leetcode-cn.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有点思维 其实分析知 `(stk + 1) / 2` （向上取整）即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    int minSwaps(string s) {
        this->n = s.size();
        int stk = 0, cnt = 0;
        for (auto c : s) {
            if (c == '[')
                stk ++ ;
            else {
                if (stk > 0)
                    stk -- ;
                else {
                    cnt ++ ;
                }
            }
        }
        // return cnt
        return (stk + 1) / 2;
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

> [!NOTE] **[LeetCode 2267. 检查是否有合法括号字符串路径](https://leetcode.cn/problems/check-if-there-is-a-valid-parentheses-string-path/)**
> 
> 题意: 
> 
> 类似摘花生的走法，要求括号序列合法

> [!TIP] **思路**
> 
> 理清思路，显然不需要维护区间，只需要关注是否合法
> 
> 需要有一个维度是左括号比右括号多多少个
> 
> **另有空间压缩 bit 优化版本**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 110, K = 210, INF = 1e9;
    vector<vector<char>> g;
    int n, m;
    
    bool f[N][N][K];    // 到 [i, j] 的位置，此时左括号比右括号多 k 的可能
    
    bool hasValidPath(vector<vector<char>>& grid) {
        this->g = grid, this->n = g.size(), this->m = g[0].size();
        if ((n + m - 1) & 1)
            return false;
        memset(f, 0, sizeof f);
        f[0][1][0] = f[1][0][0] = true;
        
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                int t = (g[i - 1][j - 1] == '(' ? 1 : -1);
                for (int k = 0; k <= n + m; ++ k )
                    if (k - t >= 0 && k - t <= n + m) {
                        f[i][j][k] |= f[i - 1][j][k - t];
                        f[i][j][k] |= f[i][j - 1][k - t];
                    }
            }
        
        return f[n][m][0];
    }
};
```


##### **C++ bit 优化**

```cpp
class Solution {
public:
    const static int N = 110, K = 210, INF = 1e9;
    vector<vector<char>> g;
    int n, m;
    
    // 可以再空间压缩
    // 二进制比特位代表 左括号比右括号多的可能性
    __uint128_t f[N][N];
    
    bool hasValidPath(vector<vector<char>>& grid) {
        this->g = grid, this->n = g.size(), this->m = g[0].size();
        if ((n + m - 1) & 1)
            return false;
        memset(f, 0, sizeof f);
        
        f[0][1] = f[1][0] = 1;
        
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                f[i][j] = f[i][j - 1] | f[i - 1][j];
                // 如果是左括号，所有可能性true的统一加一，即左移
                if (g[i - 1][j - 1] == '(')
                    f[i][j] <<= 1;
                else
                    f[i][j] >>= 1;
            }
        
        // 取最后一位 即差为0
        return f[n][m] & 1;
    }
};
```


##### **C++ bit + 空间压缩**

```cpp
class Solution {
public:
    const static int N = 110, K = 210, INF = 1e9;
    vector<vector<char>> g;
    int n, m;
    
    // 空间压缩
    // 二进制比特位代表 左括号比右括号多的可能性
    __uint128_t f[N];
    
    bool hasValidPath(vector<vector<char>>& grid) {
        this->g = grid, this->n = g.size(), this->m = g[0].size();
        if ((n + m - 1) & 1)
            return false;
        memset(f, 0, sizeof f);
        
        // f[0][1]
        f[1] = 1;
        
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                f[j] |= f[j - 1];
                // 如果是左括号，所有可能性true的统一加一，即左移
                if (g[i - 1][j - 1] == '(')
                    f[j] <<= 1;
                else
                    f[j] >>= 1;
            }
        
        // 取最后一位 即差为0
        return f[m] & 1;
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

> [!NOTE] **[Codeforces Treasure](http://codeforces.com/problemset/problem/494/A)**
> 
> 题意: 

> [!TIP] **思路**
> 
> 重点在贪心推导
> 
> 代码可以更优雅些 TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Treasure
// Contest: Codeforces - Codeforces Round #282 (Div. 1)
// URL: https://codeforces.com/problemset/problem/494/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    cin >> s;

    vector<int> xs;
    int l = 0;
    for (auto c : s)
        if (c == ')' || c == '#') {
            if (l)
                l--;
            else {
                cout << -1 << endl;
                return 0;
            }
            if (c == '#')
                xs.push_back(1);
        } else if (c == '(')
            l++;

    if (l) {
        if (xs.empty()) {
            cout << -1 << endl;
            return 0;
        } else {
            xs.back() += l;
        }
    }

    l = 0;
    int p = 0;
    string t;
    for (auto c : s)
        if (c == ')') {
            if (l)
                l--;
            else {
                cout << -1 << endl;
                return 0;
            }
        } else if (c == '#') {
            int cnt = xs[p++];
            while (cnt && l)
                l--, cnt--;
            if (cnt) {
                cout << -1 << endl;
                return 0;
            }
        } else
            l++;
    if (l) {
        cout << -1 << endl;
        return 0;
    }

    for (auto x : xs)
        cout << x << endl;

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

> [!NOTE] **[Codeforces Sereja and Brackets](http://codeforces.com/problemset/problem/380/C)**
> 
> 题意: 
> 
> 题意转化：求区间内最长有效括号

> [!TIP] **思路**
> 
> 考虑线段树维护两个东西：区间 $[l,r]$ 内不能匹配的左括号数量 $Lsum_k$ 和右括号数量 $Rsum_k$ 。
> 
> 合并计算父节点时，显然需要考虑合并时新增的匹配数目: $min(Lsum_l, Rsum_r)$
> 
> **重点在于思维敏感度 想到线段树维护**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Sereja and Brackets
// Contest: Codeforces - Codeforces Round #223 (Div. 1)
// URL: https://codeforces.com/problemset/problem/380/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e6 + 10, M = N << 2;

struct Node {
    int l, r;
    int l_sum, r_sum;  // 区间内未匹配的左/右括号的数量和
} tr[M];

int n, m;
char s[N];

void pushup(Node& u, Node& l, Node& r) {
    int minus = min(l.l_sum, r.r_sum);  // 整段可以匹配的数量
    u.l_sum = l.l_sum + r.l_sum - minus;
    u.r_sum = l.r_sum + r.r_sum - minus;
}

void pushup(int u) {
    int l = u << 1, r = u << 1 | 1;
    pushup(tr[u], tr[l], tr[r]);
}

void build(int u, int l, int r) {
    if (l == r) {
        tr[u] = {l, r};
        if (s[l] == '(')
            tr[u].l_sum = 1;
        else
            tr[u].r_sum = 1;
    } else {
        tr[u] = {l, r, 0, 0};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

Node query(int u, int l, int r) {
    if (l <= tr[u].l && r >= tr[u].r)
        return tr[u];
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid)
            return query(u << 1, l, r);
        if (l > mid)
            return query(u << 1 | 1, l, r);
        auto nl = query(u << 1, l, r);
        auto nr = query(u << 1 | 1, l, r);
        Node ret;
        pushup(ret, nl, nr);
        return ret;
    }
}

int main() {
    cin >> s + 1;
    n = strlen(s + 1);

    build(1, 1, n);

    cin >> m;
    while (m--) {
        int l, r;
        cin >> l >> r;
        auto t = query(1, l, r);
        cout << r - l + 1 - (t.l_sum + t.r_sum) << endl;
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

> [!NOTE] **[AcWing 1153. 括号树](https://www.acwing.com/problem/content/description/1155/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 转化思维 注意细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 5e5 + 10;

int n;
char s[N];
int fa[N];

vector<int> g[N];

stack<int> st;
// 考虑统计以每个结点作为末尾的合法子串的个数 t[i]，即合法的后缀子串
// 这样最终答案 k[i] 就是当前结点到根结点所有 t[i] 的和
LL k[N], t[N];

void dfs(int u) {
    int top = -1;
    if (s[u] == '(') {
        st.push(u);
    } else {
        if (!st.empty()) {
            top = st.top();
            t[u] = t[fa[top]] + 1;
            st.pop();
        }
    }
    
    k[u] = k[fa[u]] + t[u]; // t[u]
    
    for (auto v : g[u])
        dfs(v);
    
    if (s[u] == '(') {
        st.pop();
    } else {
        if (top != -1)
            st.push(top);
    }
}

int main() {
    cin >> n;
    cin >> s + 1;
    
    for (int i = 2; i <= n; ++ i ) {
        cin >> fa[i];
        g[fa[i]].push_back(i);
    }
    
    dfs(1);
    
    LL res = 0;
    for (int i = 1; i <= n; ++ i )
        res ^= (LL)i * k[i];
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