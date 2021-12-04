数位：把一个数字按照个、十、百、千等等一位一位地拆开，关注它每一位上的数字。如果拆的是十进制数，那么每一位数字都是 0~9，其他进制可类比十进制。

数位 DP 特征：

1. 要求统计满足一定条件的数的数量（即，最终目的为计数）；

2. 这些条件经过转化后可以使用「数位」的思想去理解和判断；

3. 输入会提供一个数字区间（有时也只提供上界）来作为统计的限制；

4. 上界很大（比如 $10^{18}$），暴力枚举验证会超时。

数位 DP 中通常会利用常规计数问题技巧，比如把一个区间内的答案拆成两部分相减（即 $ans_{[l, r]} = ans_{[0, r]}-ans_{[0, l - 1]}$

## 例题

> [!NOTE] **例 1 [Luogu P2602 数字计数](https://www.luogu.com.cn/problem/P2602)**
> 
> 题目大意：给定两个正整数 $a,b$，求在 $[a,b]$ 中的所有整数中，每个数码（digit）各出现了多少次。

> [!TIP] **经典数位 DP**
> 
> TODO@binacs 补充细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **cpp 修改yxc**

```cpp
// 对yxc原版进行修改更好理解(?)
#include <bits/stdc++.h>
using namespace std;

const int N = 10;

int p[N];

void init() {
    p[0] = 1;
    for (int i = 1; i < N; ++ i )
        p[i] = p[i - 1] * 10;
}

int f(int n, int v) {
    if (!n)
        return 0;
    
    vector<int> num;
    {
        int t = n;
        while (t)
            num.push_back(t % 10), t /= 10;
    }
    int d = num.size();
    
    int res = 0;
    // v == 0 则直接从第二位开始枚举 因为第一位必然不能为0
    for (int i = d - 1 - !v; i >= 0; -- i ) {
        int x = num[i];
        int l = n / p[i] / 10, r = n % p[i];
        // 前缀
        if (i < d - 1) {
            res += l * p[i];
            if (!v)
                res -= p[i];    // (l - 1) * p[i];
        }
        
        if (x == v)
            res += r + 1;
        else if (x > v)
            res += p[i];
    }
    return res;
}

int main() {
    init();
    
    int a, b;
    while (cin >> a >> b, a) {
        if (a > b)
            swap(a, b);
        
        for (int i = 0; i < N; ++ i )
            cout << f(b, i) - f(a - 1, i) << ' ';
        cout << endl;
    }
    return 0;
}
```

##### **cpp 1**

```cpp
// https://www.acwing.com/file_system/file/content/whole/index/content/2375416/
#include <bits/stdc++.h>
using namespace std;

int f(int n, int v) {
    if (!n)
        return 0;
    
    vector<int> num;
    {
        int t = n;
        while (t)
            num.push_back(t % 10), t /= 10;
    }
    int d = num.size();
    
    int res = 0, p = 1;
    // 计算从低位到高位考虑 当前位上数字v的出现次数
    for (int j = 0; j < d; ++ j ) {
        int x = num[j], l = n / p / 10, r = n % p; // x = num[j] = n / p % 10
        // 1. 计算左侧整数小于l
        //    1.1 (xxx = 000 ~ abc-1)
        if (v)
            res += l * p;
        //    1.2.如果v = 0,
        else
            // 左侧高位不能全0 (xxx = 001 ~ abc-1)
            // 其实可以直接 (l - 1) * p
            res += (l - 1) * p;
        
        // 左边整数等于 l 的情况 (xxx = abc)
        //    (v || l) 保证 【前缀 + 当前位】不为全0
        if (x > v)
            res += p;
        if (x == v)
            res += r + 1;
        
        // 更新P
        p *= 10;
    }
    return res;
}

int main() {
    int a, b;
    while (cin >> a >> b, a) {
        if (a > b)
            swap(a, b);
        for (int i = 0; i < 10; ++ i )
            cout << f(b, i) - f(a - 1, i) << ' ';
        cout << endl;
    }
    return 0;
}
```

##### **cpp 2**

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> f(int n) {
    vector<int> res(10, 0);
    if (!n)
        return res;
    
    vector<int> num;
    {
        int t = n;
        while (t)
            num.push_back(t % 10), t /= 10;
    }
    int d = num.size();
    
    int p = 1;
    // 计算从低位到高位考虑 当前位上数字v的出现次数
    for (int j = 0; j < d; ++ j ) {
        int x = num[j], l = n / p / 10, r = n % p; // x = num[j] = n / p % 10
        for (int v = 0; v < 10; ++ v ) {
            if (v)
                res[v] += l * p;
            else {
                if (l)
                    res[v] += (l - 1) * p;
            }
            
            if ((x > v) && (v || l))
                res[v] += p;
            if ((x == v) && (v || l))
                res[v] += r + 1;
        }
        // 更新P
        p *= 10;
    }
    return res;
}

int main() {
    int a, b;
    while (cin >> a >> b, a) {
        if (a > b)
            swap(a, b);
        auto va = f(a - 1);
        auto vb = f(b);
        for (int i = 0; i < 10; ++ i )
            cout << vb[i] - va[i] << ' ';
        cout << endl;
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


> [!NOTE] **例 2 [hdu 2089 不要62](https://acm.hdu.edu.cn/showproblem.php?pid=2089)**
> 
> 题面大意：统计一个区间内数位上不能有 4 也不能有连续的 62 的数有多少。

> [!TIP] **思路**
> 
> 没有 4 的话在枚举的时候判断一下，不枚举 4 就可以保证状态合法了，所以这个约束没有记忆化的必要，而对于 62 的话，涉及到两位，当前一位是 6 或者不是 6 这两种不同情况我计数是不相同的，所以要用状态来记录不同的方案数。
> 
> $dp[pos,sta]$ 表示当前第 $\mathit{pos}$ 位，前一位是否是 6 的状态，这里 $\mathit{sta}$ 只需要取 0 和 1 两种状态就可以了，不是 6 的情况可视为同种，不会影响计数。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 35;

int f[N][10];

void init() {
    for (int i = 0; i <= 9; i ++ )
        if (i != 4)
            f[1][i] = 1;

    for (int i = 1; i < N; i ++ )
        for (int j = 0; j <= 9; j ++ ) {
            if (j == 4) continue;
            for (int k = 0; k <= 9; k ++ ) {
                if (k == 4 || j == 6 && k == 2) continue;
                f[i][j] += f[i - 1][k];
            }
        }
}

int dp(int n) {
    if (!n) return 1;

    vector<int> nums;
    while (n) nums.push_back(n % 10), n /= 10;

    int res = 0;
    int last = 0;
    for (int i = nums.size() - 1; i >= 0; i -- ) {
        int x = nums[i];
        for (int j = 0; j < x; j ++ ) {
            if (j == 4 || last == 6 && j == 2) continue;
            res += f[i + 1][j];
        }

        if (x == 4 || last == 6 && x == 2) break;
        last = x;

        if (!i) res ++ ;
    }

    return res;
}

int main() {
    init();

    int l, r;
    while (cin >> l >> r, l || r) {
        cout << dp(r) - dp(l - 1) << endl;
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

> [!NOTE] **例 3 [SCOI2009 windy 数 ](https://loj.ac/problem/10165)**
> 
> 题目大意：给定一个区间 $[l,r]$，求其中满足条件 **不含前导 $0$ 且相邻两个数字相差至少为 $2$** 的数字个数。

> [!TIP] **思路**
> 
> 首先我们将问题转化成更加简单的形式。设 $ans_i$ 表示在区间 $[1,i]$ 中满足条件的数的数量，那么所求的答案就是 $ans_r-ans_{l-1}$。
> 
> 分开求解这两个问题。
> 
> 对于一个小于 $n$ 的数，它从高到低肯定出现某一位，使得这一位上的数值小于 $n$ 这一位上对应的数值。而之前的所有位都和 $n$ 上的位相等。
> 
> 有了这个性质，我们可以定义 $f(i,st,op)$ 表示当前将要考虑的是从高到低的第 $i$ 位，当前该前缀的状态为 $st$ 且前缀和当前求解的数字的大小关系是 $op$（$op=1$ 表示等于，$op=0$ 表示小于）时的数字个数。在本题中，这个前缀的状态就是上一位的值，因为当前将要确定的位不能取哪些数只和上一位有关。在其他题目中，这个值可以是：前缀的数字和，前缀所有数字的 $\gcd$，该前缀取模某个数的余数，也有两种或多种合用的情况。
> 
> 写出 **状态转移方程**：$f(i,st,op)=\sum_{k=1}^{maxx} f(i+1,k,op=1~ \operatorname{and}~ k=maxx )\quad (|st-k|\ge 2)$
> 
> 这里的 $k$ 就是当前枚举的下一位的值，而 $maxx$ 就是当前能取到的最高位。因为如果 $op=1$，那么你在这一位上取的值一定不能大于求解的数字上该位的值，否则则没有限制。
> 
> 我们发现，尽管前缀所选择的状态不同，而 $f$ 的三个参数相同，答案就是一样的。为了防止这个答案被计算多次，可以使用记忆化搜索的方式实现。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 11;

int f[N][10];    // i位最后一个数填j

void init() {
    for (int i = 0; i <= 9; i ++ ) f[1][i] = 1;  // 注意 这里f[1][0] = 1

    for (int i = 2; i < N; i ++ )
        for (int j = 0; j <= 9; j ++ )
            for (int k = 0; k <= 9; k ++ )
                if (abs(j - k) >= 2)
                    f[i][j] += f[i - 1][k];
}

int dp(int n) {
    if (!n) return 0;

    vector<int> nums;
    while (n) nums.push_back(n % 10), n /= 10;

    int res = 0;
    int last = -2;  // or 12
    for (int i = nums.size() - 1; i >= 0; i -- ) {
        int x = nums[i];
        // 最高位则需从1开始枚举
        for (int j = i == nums.size() - 1; j < x; j ++ )
            if (abs(j - last) >= 2)
                res += f[i + 1][j];

        if (abs(x - last) >= 2) last = x;
        else break;

        if (!i) res ++ ;
    }

    // 特殊处理有前导零的数
    for (int i = 1; i < nums.size(); i ++ )
        for (int j = 1; j <= 9; j ++ )
            res += f[i][j];

    return res;
}

int main() {
    init();

    int l, r;
    cin >> l >> r;
    cout << dp(r) - dp(l - 1) << endl;

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

> [!NOTE] **例 4.[SPOJMYQ10](https://www.spoj.com/problems/MYQ10/en/)**
> 
> 题面大意：假如手写下 $[n,m]$ 之间所有整数，会有多少数看起来和在镜子里看起来一模一样？（$n,m<10^{44}, T<10^5$)

注：由于这里考虑到的镜像，只有 $0,1,8$ 的镜像是自己本身。所以，这里的“一模一样”并不是传统意义上的回文串，而是只含有 $0,1,8$ 的回文串。

首先，在数位 DP 过程中，显然只有 $0,1,8$ 不会被 ban。

其次，由于数值超过 long long 范围，所以 $[n,m]=[1,m]-[1,n-1]$ 不再适用（高精度比较繁琐），而是需要对 $n$ 是否合法进行判断，得出：$[n,m]=[1,m]-[1,n]+check(n)$。

镜像解决了，如何判断回文？

我们需要用一个小数组记录一下之前的值。在未超过一半的长度时，只要不超上限就行；在超过一半的长度时，还需要判断是否和与之“镜面对称”的位相等。

需要额外注意的是，这道题的记忆化部分，不能用 `memset`，否则会导致超时。

TODO@binacs

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


> [!NOTE] **例 5. [P3311 数数](https://www.luogu.com.cn/problem/P3311)**
> 
> 题面：我们称一个正整数 $x$ 是幸运数，当且仅当它的十进制表示中不包含数字串集合 $S$ 中任意一个元素作为其子串。例如当 $S = \{22, 333, 0233\}$ 时，$233233$ 是幸运数，$23332333$、$2023320233$、$32233223$ 不是幸运数。给定 $n$ 和 $S$，计算不大于 $n$ 的幸运数个数。答案对 $10^9 + 7$ 取模。
> 
> $1 \leq n<10^{1201}，1 \leq m \leq 100，1 \leq \sum_{i = 1}^m |s_i| \leq 1500，\min_{i = 1}^m |s_i| \geq 1$，其中 $|s_i|$ 表示字符串 $s_i$ 的长度。$n$ 没有前导 $0$，但是 $s_i$ 可能有前导 $0$。

阅读题面发现，如果将数字看成字符串，那么这就是需要完成一个多模匹配，自然而然就想到 AC 自动机。普通数位 DP 中，先从高到低枚举数位，再枚举每一位都填什么，在这道题中，我们也就自然地转化为枚举已经填好的位数，再枚举此时停在 AC 自动机上的哪个节点，然后从当前节点转移到它在 AC 自动机上的子节点。

设 $f[i][j][0/1]$ 表示当前从高到低已经填了 $i$ 位（即在 AC 自动机上走过了 $i$ 条边），此时停在标号为 $j$ 的节点上，当前是否正好贴着上界。

至于题目中的“不包含”条件，只需在 AC 自动机上给每个模式串的结尾节点都打上标记，DP 过程中一旦遇上这些结尾节点就跳过即可。

转移很好想，详见代码 `main` 函数部分。


TODO@binacs

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

此题可以很好地帮助理解数位 DP 的原理。

## 习题

[Ahoi2009 self 同类分布](https://www.luogu.com.cn/problem/P4127)

[洛谷  P3413 SAC#1 - 萌数](https://www.luogu.com.cn/problem/P3413)

[HDU 6148 Valley Number](http://acm.hdu.edu.cn/showproblem.php?pid=6148)

[CF55D Beautiful numbers](http://codeforces.com/problemset/problem/55/D)

[CF628D Magic Numbers](http://codeforces.com/problemset/problem/628/D)
