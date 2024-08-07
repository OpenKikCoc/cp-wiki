## Hash 的思想

Hash 的核心思想在于，将输入映射到一个值域较小、可以方便比较的范围。

> [!WARNING]
> 这里的“值域较小”在不同情况下意义不同。
> 
> 在 [哈希表](ds/hash.md) 中，值域需要小到能够接受线性的空间与时间复杂度。
> 
> 在字符串哈希中，值域需要小到能够快速比较（$10^9$、$10^{18}$ 都是可以快速比较的）。
> 
> 同时，为了降低哈希冲突率，值域也不能太小。


## Hash 的实现

### 多次询问子串哈希


## Hash 的应用

### 字符串匹配

求出模式串的哈希值后，求出文本串每个长度为模式串长度的子串的哈希值，分别与模式串的哈希值比较即可。

### 允许 $k$ 次失配的字符串匹配

问题：给定长为 $n$ 的源串 $s$，以及长度为 $m$ 的模式串 $p$，要求查找源串中有多少子串与模式串匹配。$s'$ 与 $s$ 匹配，当且仅当 $s'$ 与 $s$ 长度相同，且最多有 $k$ 个位置字符不同。其中 $1\leq n,m\leq 10^6$，$0\leq k\leq 5$。

这道题无法使用 KMP 解决，但是可以通过哈希 + 二分来解决。

枚举所有可能匹配的子串，假设现在枚举的子串为 $s'$，通过哈希 + 二分可以快速找到 $s'$ 与 $p$ 第一个不同的位置。之后将 $s'$ 与 $p$ 在这个失配位置及之前的部分删除掉，继续查找下一个失配位置。这样的过程最多发生 $k$ 次。

总的时间复杂度为 $O(m+kn\log_2m)$。

### 最长回文子串

二分答案，判断是否可行时枚举回文中心（对称轴），哈希判断两侧是否相等。需要分别预处理正着和倒着的哈希值。时间复杂度 $O(n\log n)$。

这个问题可以使用 [manacher 算法](./manacher.md) 在 $O(n)$ 的时间内解决。

通过哈希同样可以 $O(n)$ 解决这个问题，具体方法就是记 $R_i$ 表示以 $i$ 作为结尾的最长回文的长度，那么答案就是 $\max_{i=1}^nR_i$。考虑到 $R_i\leq R_{i-1}+2$，因此我们只需要暴力从 $R_{i-1}+2$ 开始递减，直到找到第一个回文即可。记变量 $z$ 表示当前枚举的 $R_i$，初始时为 $0$，则 $z$ 在每次 $i$ 增大的时候都会增大 $2$，之后每次暴力循环都会减少 $1$，故暴力循环最多发生 $2n$ 次，总的时间复杂度为 $O(n)$。

### 最长公共子字符串

问题：给定 $m$ 个总长不超过 $n$ 的非空字符串，查找所有字符串的最长公共子字符串，如果有多个，任意输出其中一个。其中 $1\leq m, n\leq 10^6$。

很显然如果存在长度为 $k$ 的最长公共子字符串，那么 $k-1$ 的公共子字符串也必定存在。因此我们可以二分最长公共子字符串的长度。假设现在的长度为 $k$，`check(k)` 的逻辑为我们将所有所有字符串的长度为 $k$ 的子串分别进行哈希，将哈希值放入 $n$ 个哈希表中存储。之后求交集即可。

时间复杂度为 $O(n\log_2\frac{n}{m})$。

### 确定字符串中不同子字符串的数量

问题：给定长为 $n$ 的字符串，仅由小写英文字母组成，查找该字符串中不同子串的数量。

为了解决这个问题，我们遍历了所有长度为 $l=1,\cdots ,n$ 的子串。对于每个长度为 $l$，我们将其 Hash 值乘以相同的 $b$ 的幂次方，并存入一个数组中。数组中不同元素的数量等于字符串中长度不同的子串的数量，并此数字将添加到最终答案中。

为了方便起见，我们将使用 $h [i]$ 作为 Hash 的前缀字符，并定义 $h[0]=0$。


> ```cpp
>    int count_unique_substrings(string const& s) {
>        int n = s.size();
>     
>        const int b = 31;
>        const int m = 1e9 + 9;
>        vector<long long> b_pow(n);
>        b_pow[0] = 1;
>        for (int i = 1; i < n; i++) b_pow[i] = (b_pow[i - 1] * b) % m;
>     
>        vector<long long> h(n + 1, 0);
>        for (int i = 0; i < n; i++)
>            h[i + 1] = (h[i] + (s[i] - 'a' + 1) * b_pow[i]) % m;
>     
>        int cnt = 0;
>        for (int l = 1; l <= n; l++) {
>            set<long long> hs;
>            for (int i = 0; i <= n - l; i++) {
>                long long cur_h = (h[i + l] + m - h[i]) % m;
>                cur_h = (cur_h * b_pow[n - i - 1]) % m;
>                hs.insert(cur_h);
>            }
>            cnt += hs.size();
>        }
>        return cnt;
>    }
> ```

### 例题

> [!NOTE] **[CF1200E Compress Words](http://codeforces.com/contest/1200/problem/E)**
>
>   给你若干个字符串，答案串初始为空。第 $i$ 步将第 $i$ 个字符串加到答案串的后面，但是尽量地去掉重复部分（即去掉一个最长的、是原答案串的后缀、也是第 $i$ 个串的前缀的字符串），求最后得到的字符串。
>    
>   字符串个数不超过 $10^5$，总长不超过 $10^6$。

> [!TIP]  **题解**
> 
>  每次需要求最长的、是原答案串的后缀、也是第 $i$ 个串的前缀的字符串。枚举这个串的长度，哈希比较即可。


 当然，这道题也可以使用 [KMP 算法](string/kmp.md) 解决。
    
> [!TIP]  **参考代码**

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

### 一般应用

> [!NOTE] **[AcWing 841. 字符串哈希](https://www.acwing.com/problem/content/843/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using ULL = unsigned long long;

const int N = 100010, P = 131;

int n, m;
char str[N];
ULL h[N], p[N];

ULL get(int l, int r) {
    return h[r] - h[l - 1] * p[r - l + 1];
}

int main() {
    scanf("%d%d", &n, &m);
    scanf("%s", str + 1);
    
    p[0] = 1;   // h[0] = 0;
    for (int i = 1; i <= n; ++ i ) {
        h[i] = h[i - 1] * P + str[i];
        p[i] = p[i - 1] * P;
    }
    
    while (m -- ) {
        int l1, r1, l2, r2;a
        cin >> l1 >> r1 >> l2 >> r2;
        
        if (get(l1, r1) == get(l2, r2)) cout << "Yes" << endl;
        else cout << "No" << endl;
    }
    return 0;
}
```

##### **Python**

```python
import sys

def sub(l, r):
    return (h[r] - h[l - 1] * p[r - l + 1]) % Q


if __name__ == '__main__':

    n, m = map(int, input().split())
    s = input()
    N = 100010
    P = 131
    Q=2**64

    h = [0] * N
    p = [0] * N
    p[0] = 1
    for i in range(len(s)):
        #！！结果需要mod一个Q
        # c++里用的h[N]和p[N]都是unsigned long long类型的 
        h[i+1] = (h[i] * P + ord(s[i])) % Q
        p[i+1] = (p[i] * P) % (1<<64)

    for _ in range(m):
        l1, r1, l2, r2 = map(int, sys.stdin.readline().strip().split())
        if sub(l1, r1) == sub(l2, r2):
            print('Yes')
        else:
            print('No')

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1368. 最长前缀](https://www.acwing.com/problem/content/1370/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 变形

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using ULL = unsigned long long;

const int N = 200010, P = 131;

string str;
bool f[N];

int main() {
    unordered_set<ULL> hash;
    while (cin >> str, str != ".") {
        ULL h = 0;
        for (int i = str.size() - 1; i >= 0; -- i )
            h = h * P + str[i];
        hash.insert(h);
    }
    
    str.clear();
    string line;
    while (cin >> line) str += line;    // 可能多行
    
    int res = 0;
    f[0] = true;
    for (int i = 1; i <= str.size(); ++ i ) {
        ULL h = 0;
        // 每个串长度不超过10 倒着枚举
        for (int j = i; j > i - 10 && j > 0; -- j ) {
            h = h * P + str[j - 1];
            if (hash.count(h) && f[j - 1]) {
                f[i] = true;
                break;
            }
        }
        if (f[i]) res = i;
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

> [!NOTE] **[LeetCode 648. 单词替换](https://leetcode.cn/problems/replace-words/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 手动hash**

```cpp
class Solution {
public:
    // 手动字符串hash
    typedef unsigned long long ULL;
    const int P = 131;
    string replaceWords(vector<string>& dictionary, string sentence) {
        unordered_set<ULL> hash;
        for (auto & d : dictionary) {
            ULL h = 0;
            for (auto c : d) h = h * P + c;
            hash.insert(h);
        }

        stringstream ss(sentence);
        string s, res;
        while (ss >> s) {
            string t;
            ULL h = 0;
            for (auto c : s) {
                t += c;
                h = h * P + c;
                if (hash.count(h)) break;
            }
            res += t + ' ';
        }
        res.pop_back();
        return res;
    }
```

##### **C++ 原始hash**

```cpp
    // 原始hash
    string replaceWords(vector<string>& dictionary, string sentence) {
        unordered_map<string, bool> hash;
        for (auto & d : dictionary) hash[d] = true;

        vector<string> ve;
        stringstream ss(sentence);
        string s;
        while (ss >> s) {
            for (int i = 1; i < s.size(); ++ i ) {
                string sub = s.substr(0, i);
                if (hash.count(sub)) {
                    s = sub;
                    break;
                }
            }
            ve.push_back(s);
        }

        string res;
        for (auto & s : ve) {
            res += s;
            res.push_back(' ');
        }
        if (res.size()) res.pop_back();
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

> [!NOTE] **[LeetCode 796. 旋转字符串](https://leetcode.cn/problems/rotate-string/)**
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
    using ULL = unsigned long long;
    const static int N = 210, P = 131;
    ULL h[N], p[N];

    ULL get(int l, int r) {
        return h[r] - h[l - 1] * p[r - l + 1];
    }
    
    bool rotateString(string A, string B) {
        if (A == B) // "" 空串
            return true;
        string s = ' ' + A + B;
        int n = s.size() - 1;
        p[0] = 1;
        for (int i = 1; i <= n; ++ i ) {
            p[i] = p[i - 1] * P;
            h[i] = h[i - 1] * P + s[i];
        }

        for (int k = 1; k < A.size(); ++ k )
            if (get(1, k) == get(n - k + 1, n) && 
                get(k + 1, A.size()) == get(A.size() + 1, n - k))
                return true;
        return false;
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

> [!NOTE] **[LeetCode 1147. 段式回文](https://leetcode.cn/problems/longest-chunked-palindrome-decomposition/) TAG**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 字符串 hash + 双指针
> 
> O(n^2) dp 方法超时，考虑双指针向中间收缩
> 
> l r 维护已合法的两侧外部区间边界即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // O(n^2)的dp会超时
    using ULL = unsigned long long;
    const static int P = 131, N = 1010;
    ULL h[N], p[N];
    int f[N][N];
    
    ULL get(int l, int r) {
        return h[r] - h[l - 1] * p[r - l + 1];
    }
    
    int longestDecomposition(string text) {
        int n = text.size();
        
        p[0] = 1;
        for (int i = 1; i <= n; ++ i ) {
            h[i] = h[i - 1] * P + text[i - 1];
            p[i] = p[i - 1] * P;
        }
        
        int res = 0, l = 1, r = n;;
        for (int i = 1, j = n; i < j; ++ i , -- j )
            if (get(l, i) == get(j, r)) {
                res += 2;
                l = i + 1, r = j - 1;
            }
        if (l <= r)
            ++ res;
        
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

> [!NOTE] **[LeetCode 1316. 不同的循环子字符串](https://leetcode.cn/problems/distinct-echo-substrings/)** [TAG]
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
    using ll = long long;
    int distinctEchoSubstrings(string text) {
        int n = text.size();
        ll mod = 1e9 + 7, B = 31;
        vector<ll> h(n + 1);
        for (int i = 0; i < n; ++i) h[i + 1] = (h[i] * B + text[i]) % mod;
        vector<ll> p(n + 1);
        p[0] = 1;
        for (int i = 1; i <= n; ++i) p[i] = p[i - 1] * B % mod;
        auto get = [&](int x, int y) {
            return (h[y] - h[x - 1] * p[y - x + 1] % mod + mod) % mod;
        };
        unordered_set<ll> H;
        for (int i = 1; i <= n; ++i) {
            for (int j = i + 1; j <= n; j += 2) {
                int l = (j - i + 1) / 2;
                int k = i + l - 1;
                ll h1 = get(i, k), h2 = get(k + 1, j);
                if (h1 == h2) H.insert(h1);
            }
        }
        return H.size();
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

> [!NOTE] **[LeetCode 1392. 最长快乐前缀](https://leetcode.cn/problems/longest-happy-prefix/)** [TAG]
> 
> 题意: 
> 
> 求字符串的最长快乐前缀(既是前缀也是后缀)

> [!TIP] **思路**
> 
> 字符串hash
> 
> **记录这种每次追加到前面来更新后缀的写法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string longestPrefix(string s) {
        int n = s.size();
        int prefix = 0, suffix = 0;
        int base = 31, mod = 1000000007, mul = 1;
        int happy = 0;
        for (int i = 1; i < n; ++i) {
            prefix = ((long long)prefix * base + (s[i - 1] - 97)) % mod;
            suffix = (suffix + (long long)(s[n - i] - 97) * mul) % mod;
            if (prefix == suffix) {
                happy = i;
            }
            mul = (long long)mul * base % mod;
        }
        return s.substr(0, happy);
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


> [!NOTE] **[Codeforces D. Good Substrings](https://codeforces.com/problemset/problem/271/D)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 字符串hash 很好的应用

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Good Substrings
// Contest: Codeforces - Codeforces Round #166 (Div. 2)
// URL: https://codeforces.com/problemset/problem/271/D
// Memory Limit: 512 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 字符串hash OR SAM OR Trie
// 一开始想的是滑动窗口 但这种最好是解决【最长子串要求的】
// 如果只是统计子串选取方案 显然可以再借助单调栈算当前一个
//    good字符后面还有多少个连续good字符（显然单调栈）然后窗口累加
// 但是本题要求对串去重 故不能用滑动窗口
using ULL = unsigned long long;
const int N = 1510, M = 30, P = 131;

int n, k;
string s;
int sum[N];
ULL h[N], p[N];
bool st[M];
unordered_set<ULL> S;

ULL get(int l, int r) { return h[r] - h[l - 1] * p[r - l + 1]; }

int main() {
    cin >> s;
    n = s.size();
    for (int i = 0; i < 26; ++i) {
        char c;
        cin >> c;
        st[i] = c == '0';
    }
    cin >> k;

    // 前缀和
    for (int i = 1; i <= n; ++i)
        sum[i] = sum[i - 1] + st[s[i - 1] - 'a'];

    // 字符串hash
    p[0] = 1;
    for (int i = 1; i <= n; ++i) {
        h[i] = h[i - 1] * P + s[i - 1];
        p[i] = p[i - 1] * P;
    }

    for (int i = 1; i <= n; ++i)
        for (int j = i; j <= n; ++j)
            if (sum[j] - sum[i - 1] <= k)
                S.insert(get(i, j));
            else
                break;

    cout << S.size() << endl;

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

> [!NOTE] **[LeetCode 1044. 最长重复子串](https://leetcode.cn/problems/longest-duplicate-substring/)**
> 
> 题意: 
> 
> 找到字符串中重复出现过的长度最长的子串

> [!TIP] **思路**
> 
> 显然可以 二分+字符串哈希
> 
> TODO: 有复杂度更优的 **后缀数组** 解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using ULL = unsigned long long;
    const static int N = 3e4 + 10, P = 131;

    ULL p[N], h[N];
    void init() {
        p[0] = 1;
        for (int i = 1; i < N; ++ i )
            p[i] = p[i - 1] * P;
    }
    ULL get(int l, int r) {
        return h[r] - h[l - 1] * p[r - l + 1];
    }

    int n, start;
    bool check(int m) {
        unordered_set<ULL> S;
        for (int i = 1; i + m - 1 <= n; ++ i ) {
            ULL t = get(i, i + m - 1);
            if (S.count(t)) {
                start = i;
                return true;
            }
            S.insert(t);
        }
        return false;
    }

    string longestDupSubstring(string s) {
        init();

        this->n = s.size();
        for (int i = 1; i <= n; ++ i )
            h[i] = h[i - 1] * P + s[i - 1];
        
        int l = 0, r = n + 1;
        while (l < r) {
            int m = l + r >> 1;
            if (check(m))
                l = m + 1;
            else
                r = m;
        }

        check(l - 1);
        return s.substr(start - 1, l - 1);
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

### 进阶

> [!NOTE] **[LeetCode 2156. 查找给定哈希值的子串](https://leetcode.cn/problems/find-substring-with-given-hash-value/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常好的非常规【字符串hash】结合**滑动窗口**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 2e4 + 10;
    
    int P, M;
    LL p[N];
    
    string subStrHash(string s, int power, int modulo, int k, int hashValue) {
        this->P = power, this->M = modulo;
        int n = s.size();
        {
            p[0] = 1;
            for (int i = 1; i <= n; ++ i )
                p[i] = p[i - 1] * P % M;
        }
        
        int res = 0;
        LL x = 0;
        for (int i = n - k; i < n; ++ i )
            x = (x + p[i - (n - k)] * (s[i] - 'a' + 1) % M) % M;
        if (x == hashValue)
            res = n - k; // DO NOT return, its the last not the first
        
        for (int i = n - 1; i >= k; -- i ) {
            int j = i - k;
            x = ((x - p[k - 1] * (s[i] - 'a' + 1) % M) % M + M) % M;
            x = (x * P % M + (s[j] - 'a' + 1) % M) % M;
            if (x == hashValue)
                // return s.substr(i - k, k); // DO NOT return, its the last not the first
                res = i - k;
        }
        return s.substr(res, k);
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

> [!NOTE] **[Codeforces Watto and Mechanism](http://codeforces.com/problemset/problem/514/C)**
> 
> 题意: 
> 
> 即有字符串集合，判断输入字符串能否 `恰好只修改一个字符` 成为集合中出现过的字符串
> 
> **卡自然哈希 卡ULL 卡131/9991等**

> [!TIP] **思路**
> 
> 字符串 hash 进阶经典
> 
> **TODO: 多模哈希**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Watto and Mechanism
// Contest: Codeforces - Codeforces Round #291 (Div. 2)
// URL: https://codeforces.com/problemset/problem/514/C
// Memory Limit: 256 MB
// Time Limit: 3000 ms

#include <bits/stdc++.h>
using namespace std;

// using ULL = unsigned long long;
using ULL = long long;
using PII = pair<int, int>;
// P = 131 WA on test-27, luogu题解区也有很多卡在27的代码
// P = 99991 WA
const static int N = 3e5 + 10, P = 257;
// 取模同样不能用 ULL 自然取模，手动设置1e9+7
const static int MOD = 1e9 + 7;

ULL h[N], p[N];

void init() {
    p[0] = 1;
    for (int i = 1; i < N; ++i)
        // p[i] = p[i - 1] * P;
        p[i] = p[i - 1] * P % MOD;
}

ULL get_hash(string& s) {
    int len = s.size();
    ULL x = 0;
    for (int j = 0; j < len; ++j)
        // x = x * P + s[j];
        x = (x * P % MOD + s[j]) % MOD;
    // cout << " s = " << s << " x = " << x << endl;
    return x;
}

int n, m;

unordered_set<ULL> S;
string s;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n >> m;
    for (int i = 0; i < n; ++i) {
        cin >> s;
        S.insert(get_hash(s));
    }

    for (int i = 0; i < m; ++i) {
        cin >> s;
        // 必须只算一次后续单次修改，否则TLE
        ULL x = get_hash(s);
        bool flag = false;
        int len = s.size();
        for (int j = 0; j < len && !flag; ++j) {
            char t = s[j];
            ULL y = (x - t * p[len - j - 1] % MOD + MOD) % MOD;
            for (char c = 'a'; c <= 'c' && !flag; ++c)
                if (c != t) {
                    if (S.count((y + c * p[len - j - 1] % MOD) % MOD))
                        flag = true;
                }
        }

        cout << (flag ? "YES" : "NO") << endl;
    }

    return 0;
}
```

##### **C++ 多模哈希 TODO**

```cpp
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces Petr#](http://codeforces.com/problemset/problem/113/B)**
> 
> 题意: 
> 
> 一个字符串，求以 $a$ 开头 $b$ 结尾的本质不同的字符串数。

> [!TIP] **思路**
> 
> 显然需要计算字符串 hash 并记录每个 $a$ $b$ 出现的位置
> 
> 随后遍历前后缀的位置，再根据 hash 去重即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Petr#
// Contest: Codeforces - Codeforces Beta Round #86 (Div. 1 Only)
// URL: https://codeforces.com/problemset/problem/113/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using ULL = unsigned long long;
using LL = long long;
const static int N = 2e3 + 10, P = 131;

ULL p[N];
void init() {
    p[0] = 1;
    for (int i = 1; i < N; ++i)
        p[i] = p[i - 1] * P;
}

string s, a, b;
int ns, na, nb;

ULL h[N];
void make_hash() {
    h[0] = 0;
    for (int i = 1; i <= ns; ++i)
        h[i] = h[i - 1] * P + s[i - 1];
}
ULL get(int l, int r) { return h[r] - h[l - 1] * p[r - l + 1]; }

ULL get(string& s, int n) {
    ULL ret = 0;
    for (int i = 0; i < n; ++i)
        ret = ret * P + s[i];
    return ret;
}

int l[N], r[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();  // ATTENTION

    cin >> s >> a >> b;
    ns = s.size(), na = a.size(), nb = b.size();
    make_hash();

    ULL ha = get(a, na), hb = get(b, nb);

    memset(l, 0, sizeof l), memset(r, 0, sizeof r);
    vector<int> l, r; // 分别记录【前缀的开始、后缀的结束】位置
    for (int i = 1; i <= ns; ++i) {
        if (i + na - 1 <= ns) {
            if (get(i, i + na - 1) == ha)
                l.push_back(i);
        }
        if (i - nb + 1 >= 1) {
            if (get(i - nb + 1, i) == hb)
                r.push_back(i);
        }
    }
    unordered_set<ULL> S;
    {
        int w = max(na, nb);
        int n1 = l.size(), n2 = r.size();
        for (int i = 0, j = 0; i < n1; ++i) {
            while (j < n2 && r[j] < l[i] + w - 1)  // ATTENTION
                j++;
            for (int k = j; k < n2; ++k) {
                // cout << " l = " << l[i] << " r = " << r[k]
                // << " str = " << s.substr(l[i] - 1, r[k] - l[i] + 1)
                // << endl;
                S.insert(get(l[i], r[k]));
            }
        }
    }
    cout << S.size() << endl;

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
