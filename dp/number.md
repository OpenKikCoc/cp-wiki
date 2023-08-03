> [!TIP] **数位 DP 两大思想**
> 
> - 补集的思想
> 
> - `求区间` 变 `区间减` 思想

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

##### **C++ 3**

```cpp
// 标准写法
#include <bits/stdc++.h>
using namespace std;

int n, x;

int main() {
    cin >> n >> x;
    
    int res = 0;
    vector<int> nums;
    while (n)
        nums.push_back(n % 10), n /= 10;
    n = nums.size();
    // Solve: x = 0
    // NOT `for (int i = n - 1; i >= 0; -- i ) {`
    // BUT
    for (int i = n - 1 - !x; i >= 0; -- i ) {
        int l = 0, r = 0, t = 1;
        for (int j = n - 1; j > i; -- j )
            l = l * 10 + nums[j];
        for (int j = i - 1; j >= 0; -- j )
            r = r * 10 + nums[j], t *= 10;
        
        // NOT `res += l * t;`
        // BUT
        {
            if (i < n - 1) {
                res += l * t;
                if (!x)
                    res -= t;
            }
        }
        if (nums[i] == x)
            res += r + 1;
        else if (nums[i] > x)
            res += t;
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

* * *

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


> [!TIP] **思路**
> 
> f 状态定义
> 
> 共 i 个位置且最近一个位置为 j 的方案数
> 
> 注意 init 中 f[1][0] = 1 dp 中为0
> 
> 注意处理 不重不漏

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

* * *

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

* * *

## 习题

[Ahoi2009 self 同类分布](https://www.luogu.com.cn/problem/P4127)

[洛谷  P3413 SAC#1 - 萌数](https://www.luogu.com.cn/problem/P3413)

[HDU 6148 Valley Number](http://acm.hdu.edu.cn/showproblem.php?pid=6148)

[CF55D Beautiful numbers](http://codeforces.com/problemset/problem/55/D)

[CF628D Magic Numbers](http://codeforces.com/problemset/problem/628/D)


> [!NOTE] **[AcWing 1382. 比特串](https://www.acwing.com/problem/content/1384/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典基础数位dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
数位dp
考虑截至现在第i位
    如果填1后面多少种方案
        填0后面多少种方案
*/
#include <bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 40;

LL n, L, I;
LL c[N][N], f[N][N];

int main() {
    cin >> n >> L >> I;
    
    // 1. 组合数 i的长度恰好有j位的种类数
    for (int i = 0; i < N; ++ i )
        for (int j = 0; j <= i; ++ j )
            if (!j) c[i][j] = 1;
            else c[i][j] = c[i - 1][j] + c[i - 1][j - 1];
    
    // i位长度 后面不超过j个 种类总数
    for (int i = 0; i < N; ++ i )
        for (int j = 0; j < N; ++ j )   // j < N instead of j <= i
            for (int k = 0; k <= j; ++ k )
                f[i][j] += c[i][k];
    
    for (int i = 1, s = 0; i <= n; ++ i ) {
        // 假设当前位填0，看后面还有多少种选择
        LL x = f[n - i][L - s];
        if (I > x) {
            cout << 1;
            I -= x;
            s ++ ;
        } else cout << 0;
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

> [!NOTE] **[AcWing 1081. 度的数量](https://www.acwing.com/problem/content/1083/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 理解记忆 f状态定义
> 
> 共 i 个位置放置 j 个 1 的方案数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// https://www.acwing.com/solution/content/6997/
int x, y, k, b;
vector<vector<int>> c;

void init() {
    // 组合数
    for (int i = 0; i < 35; ++i)
        for (int j = 0; j <= i; ++j)
            if (!j)
                c[i][j] = 1;
            else
                c[i][j] = c[i - 1][j] + c[i - 1][j - 1];
}

int f(int n) {
    if (!n) return 0;
    vector<int> nums;
    while (n) nums.push_back(n % b), n /= b;
    int res = 0;
    int last = 0;  // 表示已经取了多少个 1
    // 从最高位对每一位数讨论
    for (int i = nums.size() - 1; i >= 0; --i) {
        int x = nums[i];
        // if(x) 左侧分支
        if (x) {
            // 加上第i位取0的时候的组合数 即：对于后面i位取k-last个1的数量
            res += c[i][k - last];
            // 第i位取1：
            if (x > 1) {
                // 如果x>1 后面i位随便取k-last-1个1  使用组合数
                if (k - last - 1 >= 0) res += c[i][k - last - 1];
                break;
            } else {
                // 如果x==1 第i位取1时需继续向下取 此时不是组合数
                ++last;
                if (last > k) break;
            }
        }
        if (!i && last == k) ++res;  // 最右侧分支上的方案 对于最后一位特殊考虑
    }
    return res;
}

int main() {
    c = vector<vector<int>>(36, vector<int>(36));
    init();

    cin >> x >> y >> k >> b;
    cout << f(y) - f(x - 1) << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1082. 数字游戏](https://www.acwing.com/problem/content/1084/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> f 状态定义
> 
> 共 i 个位置且最高位为 j 的不递减方案数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 15;

int f[N][N];  // f[i, j] 表示一共有i位 且最高位填j的数的个数

void init() {
    for (int i = 0; i <= 9; ++i) f[1][i] = 1;
    for (int i = 2; i < N; ++i)
        for (int j = 0; j <= 9; ++j)  // j 从0开始
            for (int k = j; k <= 9; ++k) f[i][j] += f[i - 1][k];
}

int dp(int n) {
    if (!n) return 1;
    vector<int> nums;
    while (n) nums.push_back(n % 10), n /= 10;
    int res = 0;
    int last = 0;  // 保留前缀信息 对于本题 前缀信息是：上一个数是多少
    for (int i = nums.size() - 1; i >= 0; --i) {
        int x = nums[i];
        for (int j = last; j < x; ++j)
            res += f[i + 1][j];  // 位置下标为i 后面总共有i+1个数
        if (x < last) break;
        last = x;
        if (!i) ++res;
    }
    return res;
}

int main() {
    init();
    int a, b;
    while (cin >> a >> b) { cout << dp(b) - dp(a - 1) << endl; }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1084. 数字游戏 II](https://www.acwing.com/problem/content/1086/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 同样 init 中n=0 f[1][i][i%c] = 1 dp中对于n=0为0

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 11;
const int M = 110;
int f[N][10][M];

int c;

int mod(int x, int y) { return (x % y + y) % y; }

void init() {
    memset(f, 0, sizeof f);
    for (int i = 0; i <= 9; ++i) f[1][i][i % c] = 1;
    for (int i = 2; i < N; ++i)
        for (int j = 0; j <= 9; ++j)
            for (int k = 0; k < c; ++k)
                for (int x = 0; x <= 9; ++x)
                    f[i][j][k] += f[i - 1][x][mod(k - j, c)];
}

int dp(int n) {
    if (!n) return 1;
    vector<int> nums;
    while (n) nums.push_back(n % 10), n /= 10;
    int res = 0;
    int last = 0;
    for (int i = nums.size() - 1; i >= 0; --i) {
        int x = nums[i];
        for (int j = 0; j < x; ++j) res += f[i + 1][j][mod(-last, c)];
        last += x;
        if (!i && last % c == 0) ++res;
    }
    return res;
}

int main() {
    int a, b;
    while (cin >> a >> b >> c) {
        init();
        cout << dp(b) - dp(a - 1) << endl;
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1086. 恨7不成妻](https://www.acwing.com/problem/content/1088/)**
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
#include <bits/stdc++>

using namespace std;

typedef long long LL;

const int N = 20, P = 1e9 + 7;

struct F {
    int s0, s1, s2;
} f[N][10][7][7];

int power7[N], power9[N];

int mod(LL x, int y) {
    return (x % y + y) % y;
}

void init() {
    for (int i = 0; i <= 9; i ++ ) {
        if (i == 7) continue;
        auto& v = f[1][i][i % 7][i % 7];
        v.s0 ++, v.s1 += i, v.s2 += i * i;
    }

    LL power = 10;
    for (int i = 2; i < N; i ++, power *= 10)
        for (int j = 0; j <= 9; j ++ ) {
            if (j == 7) continue;
            for (int a = 0; a < 7; a ++ )
                for (int b = 0; b < 7; b ++ )
                    for (int k = 0; k <= 9; k ++ ) {
                        if (k == 7) continue;
                        auto &v1 = f[i][j][a][b], &v2 = f[i - 1][k][mod(a - j * power, 7)][mod(b - j, 7)];
                        v1.s0 = mod(v1.s0 + v2.s0, P);
                        v1.s1 = mod(v1.s1 + v2.s1 + j * (power % P) % P * v2.s0, P);
                        v1.s2 = mod(v1.s2 + j * j * (power % P) % P * (power % P) % P * v2.s0 + v2.s2 + 2 * j * power % P * v2.s1, P);
                    }
        }

    power7[0] = 1;
    for (int i = 1; i < N; i ++ ) power7[i] = power7[i - 1] * 10 % 7;

    power9[0] = 1;
    for (int i = 1; i < N; i ++ ) power9[i] = power9[i - 1] * 10ll % P;
}

F get(int i, int j, int a, int b) {
    int s0 = 0, s1 = 0, s2 = 0;
    for (int x = 0; x < 7; x ++ )
        for (int y = 0; y < 7; y ++ )
            if (x != a && y != b) {
                auto v = f[i][j][x][y];
                s0 = (s0 + v.s0) % P;
                s1 = (s1 + v.s1) % P;
                s2 = (s2 + v.s2) % P;
            }
    return {s0, s1, s2};
}

int dp(LL n) {
    if (!n) return 0;

    LL backup_n = n % P;
    vector<int> nums;
    while (n) nums.push_back(n % 10), n /= 10;

    int res = 0;
    LL last_a = 0, last_b = 0;
    for (int i = nums.size() - 1; i >= 0; i -- ) {
        int x = nums[i];
        for (int j = 0; j < x; j ++ ) {
            if (j == 7) continue;
            int a = mod(-last_a * power7[i + 1], 7);
            int b = mod(-last_b, 7);
            auto v = get(i + 1, j, a, b);
            res = mod(
                res + 
                (last_a % P) * (last_a % P) % P * power9[i + 1] % P * power9[i + 1] % P * v.s0 % P + 
                v.s2 + 
                2 * last_a % P * power9[i + 1] % P * v.s1,
            P);
        }

        if (x == 7) break;
        last_a = last_a * 10 + x;
        last_b += x;

        if (!i && last_a % 7 && last_b % 7) res = (res + backup_n * backup_n) % P;
    }

    return res;
}

int main() {
    int T;
    cin >> T;

    init();

    while (T -- ) {
        LL l, r;
        cin >> l >> r;
        cout << mod(dp(r) - dp(l - 1), P) << endl;
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

### TODO

> [!NOTE] **[Luogu P3281 [SCOI2013]数数](https://www.luogu.com.cn/problem/P3281)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO

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

* * *

> [!NOTE] **[Luogu P2518 [HAOI2010]计数](https://www.luogu.com.cn/problem/P2518)**
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

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu P3286 [SCOI2014]方伯伯的商场之旅](https://www.luogu.com.cn/problem/P3286)**
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

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu P4124 [CQOI2016]手机号码](https://www.luogu.com.cn/problem/P4124)**
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

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu P4999 烦人的数学作业](https://www.luogu.com.cn/problem/P4999)**
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

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu P2606 [ZJOI2010]排列计数](https://www.luogu.com.cn/problem/P2606)**
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

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu P4798 [CEOI2015 Day1] 卡尔文球锦标赛](https://www.luogu.com.cn/problem/P4798)**
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

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### LeetCode 基础

> [!NOTE] **[LeetCode 233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)**
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
    int countDigitOne(int n) {
        if (!n)
            return 0;
        vector<int> nums;
        while (n)
            nums.push_back(n % 10), n /= 10;

        int res = 0;
        // i = nums.size() - 1 - !v
        for (int i = nums.size() - 1 - !1; i >= 0; -- i ) {
            int x = nums[i];
            int l = 0, r = 0, p = 1;
            for (int j = nums.size() - 1; j > i; -- j )
                l = l * 10 + nums[j];
            for (int j = i - 1; j >= 0; -- j )
                r = r * 10 + nums[j], p *= 10;
            
            if (i < nums.size() - 1) {
                res += l * p;
                if (!1) // v = 1, will not come in
                    res -= p;
            }

            if (x == 1)
                res += r + 1;
            else if (x > 1)
                res += p;
        }
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int countDigitOne(int n) {
        if (n <= 0) return 0;
        vector<int> nums;
        while (n) nums.push_back(n % 10), n /= 10;
        reverse(nums.begin(), nums.end());
        int res = 0;
        for (int i = 0; i < nums.size(); i ++ ) {
            int d = nums[i];
            int left = 0, right = 0, p = 1;
            for (int j = 0; j < i; j ++ ) left = left * 10 + nums[j];
            for (int j = i + 1; j < nums.size(); j ++ ) {
                right = right * 10 + nums[j];
                p = p * 10;
            }
            if (d == 0) res += left * p;
            else if (d == 1) res += left * p + right + 1;
            else res += (left + 1) * p;
        }
        return res;
    }
};
```

##### **C++ 废弃**

```cpp
class Solution {
public:
    int countDigitOne(int n) {
        int res = 0;
        long long base = 1;
        while (base <= n) {
            int t = (n / base ) %10;
            if (t == 0) res += n / (base * 10) * base; // front
            else if (t == 1) res += n / (base * 10) * base + n % base + 1;
            else res += (n / (base * 10) + 1) * base;
            base *= 10;
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        nums = []
        while n:
            nums.append(n % 10)
            n //= 10 
        n = len(nums)
        res = 0

        for i in range(n - 1, -1, -1):
            l, r, t = 0, 0, 1
            for j in range(n - 1, i, -1):
                l = l * 10 + nums[j]
            for j in range(i - 1, -1, -1):
                r = r * 10 + nums[j]
                t *= 10 

        
            res += l * t
            if nums[i] ==  1:
                res += (r + 1)
            elif nums[i] > 1:
                res += t
        return res


```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 600. 不含连续1的非负整数](https://leetcode-cn.com/problems/non-negative-integers-without-consecutive-ones/)**
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
    const static int N = 35;

    int f[N][2];
    void init() {
        f[1][0] = f[1][1] = 1;
        for (int i = 2; i < N; ++ i ) {
            f[i][0] = f[i - 1][0] + f[i - 1][1];
            f[i][1] = f[i - 1][0];
        }
    }

    int findIntegers(int n) {
        if (!n)
            return 1;
        
        init();
        vector<int> nums;
        while (n)
            nums.push_back(n % 2), n /= 2;
        
        int res = 0;
        int last = 0;   // 上一位是啥
        for (int i = nums.size() - 1; i >= 0; -- i ) {
            int x = nums[i];
            if (x)
                // 假定当前位为0
                res += f[i + 1][0];
            if (last && x)
                break;
            last = x;
            if (!i)
                res ++ ;
        }
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int findIntegers(int num) {
        vector<int> nums;
        while (num) nums.push_back(num % 2), num /= 2;
        vector<vector<int>> f(nums.size() + 1, vector<int>(2));
        f[1][0] = f[1][1] = 1;
        for (int i = 2; i <= nums.size(); ++ i ) {
            f[i][0] = f[i - 1][0] + f[i - 1][1];
            f[i][1] = f[i - 1][0];
        }

        int res = 0;
        for (int i = nums.size(), last = 0; i; -- i ) {
            int x = nums[i - 1];
            if (x) {
                res += f[i][0];
                // 前面已经是1 必然不合法 直接return
                if (last) return res;
            }
            last = x;
        }
        return res + 1;
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

> [!NOTE] **[LeetCode 1397. 找到所有好字符串](https://leetcode-cn.com/problems/find-all-good-strings/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数位DP + KMP
> 
> 重复练习 增加熟练度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int MOD = 1e9 + 7;

    vector<int> get_next(string & s) {
        int m = s.size();
        vector<int> nxt(m + 1);
        nxt[0] = nxt[1] = 0;
        for (int i = 1; i < m; ++ i ) {
            int j = nxt[i];
            while (j && s[j] != s[i])
                j = nxt[j];
            if (s[j] == s[i])
                nxt[i + 1] = j + 1;
            else
                nxt[i + 1] = 0;
        }
        return nxt;
    }

    int n, m;
    vector<vector<int>> g, f;

    int calc(string & s, int flag) {
        int ret = 0, cur = 0;
        for (int i = 0; i < n; ++ i ) {
            int x = s[i] - 'a';
            for (int c = 0; c < x; ++ c )
                ret = (ret + f[n - i - 1][g[cur][c]]) % MOD;
            cur = g[cur][x];    // ATTENTION
            if (cur == m)
                break;
            if (i == n - 1 && flag && cur != m)
                ret = (ret + 1) % MOD;
        }
        return ret;
    }

    int findGoodStrings(int n, string s1, string s2, string evil) {
        this->n = n, this->m = evil.size();

        auto next = get_next(evil);

        // ATTENTION
        this->g = vector<vector<int>>(m + 1, vector<int>(26));
        for (int i = 0; i < m; ++ i )
            for (int c = 0; c < 26; ++ c ) {
                int j = i;
                while (j && evil[j] != 'a' + c)
                    j = next[j];
                if (evil[j] == 'a' + c)
                    g[i][c] = j + 1;
                else
                    g[i][c] = 0;
            }
        
        // 数位 dp 的预处理
        this->f = vector<vector<int>>(n + 1, vector<int>(m + 1));
        for (int j = 0; j < m; ++ j )
            f[0][j] = 1;
        for (int i = 1; i <= n; ++ i )
            for (int j = 0; j < m; ++ j ) {
                f[i][j] = 0;
                for (int c = 0; c < 26; ++ c )
                    f[i][j] = (f[i][j] + f[i - 1][g[j][c]]) % MOD;
            }
        
        return (calc(s2, 1) - calc(s1, 0) + MOD) % MOD;
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

> [!NOTE] **[LeetCode 1012. 至少有 1 位重复的数字](https://leetcode.cn/problems/numbers-with-repeated-digits/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 补集思想的应用

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 排列数
    int P(int a, int b) {
        int ret = 1;
        for (int i = a, j = 0; j < b; -- i , ++ j )
            ret *= i;
        return ret;
    }

    // 本题即使用补集的思想, 算不能重复的数的个数
    int numDupDigitsAtMostN(int n) {
        int res = n;

        vector<int> nums;
        while (n)
            nums.push_back(n % 10), n /= 10;
        
        // 1. 枚举较少位数的情况
        for (int i = 1; i < nums.size(); ++ i )
            // 首位不能为 0, 后面的每个都和前面的不同
            res -= 9 * P(9, i - 1);
        // 2. 枚举相同位数的情况, 但是首位较低, 后续自由选
        res -= (nums.back() - 1) * P(9, nums.size() - 1);

        // 3. 位数相同, 且首位相同的情况, 后续选择会受限制
        vector<bool> st(10);    // 前后有关联关系 对于本题来说就是哪个数有没有被用过
        st[nums.back()] = true;
        for (int i = nums.size() - 2; i >= 0; -- i ) {
            int x = nums[i];
            for (int j = 0; j < x; ++ j )
                if (!st[j])
                    // 用过的数 nums.size()-i ==> 没有用过的数 10-(nums.size()-i)
                    res -= P(10 - (nums.size() - i), i);
            
            if (st[x])
                return res;
            st[x] = true;
        }
        return res - 1;
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

> [!NOTE] **[LeetCode 2376. 统计特殊整数](https://leetcode.cn/problems/count-special-integers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 同上题 **[LeetCode 1012. 至少有 1 位重复的数字](https://leetcode.cn/problems/numbers-with-repeated-digits/)**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 排列数
    int P(int a, int b) {
        int ret = 1;
        for (int i = a, j = 0; j < b; -- i , ++ j )
            ret *= i;
        return ret;
    }
    
    int countSpecialNumbers(int n) {
        vector<int> nums;
        while (n)
            nums.push_back(n % 10), n /= 10;
        
        int res = 0;
        // 1. 枚举较少位数的情况
        for (int i = 1; i < nums.size(); ++ i )
            // 首位不能为 0, 后面的每个都和前面的不同
            res += 9 * P(9, i - 1);
        // 2. 枚举相同位数的情况, 但是首位较低, 后续自由选
        res += (nums.back() - 1) * P(9, nums.size() - 1);

        // 3. 位数相同, 且首位相同的情况, 后续选择会受限制
        vector<bool> st(10);    // 前后有关联关系 对于本题来说就是哪个数有没有被用过
        st[nums.back()] = true;
        for (int i = nums.size() - 2; i >= 0; -- i ) {
            int x = nums[i];
            for (int j = 0; j < x; ++ j )
                if (!st[j])
                    // 用过的数 nums.size()-i ==> 没有用过的数 10-(nums.size()-i)
                    res += P(10 - (nums.size() - i), i);
            
            if (st[x])
                return res;
            st[x] = true;
        }
        return res + 1;
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

> [!NOTE] **[LeetCode 2719. 统计整数数目](https://leetcode.cn/problems/count-of-integers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准数位 dp
> 
> 还得写个高精度减法（也可以直接枚举下 num1 是否是一个合法方案）
> 
> 加快速度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 30, M = 500, MOD = 1e9 + 7;
    
    LL f[N][M]; // 共计 i 个长度的数字，总和不超过 j 的方案数
    void init() {
        memset(f, 0, sizeof f);
        for (int i = 0; i < N; ++ i )
            f[i][0] = 1;
        for (int i = 1; i < N; ++ i ) {
            for (int j = 1; j < M; ++ j )
                for (int k = 0; k < 10; ++ k )  // 最后一位的数值
                    if (j - k >= 0)
                        f[i][j] = (f[i][j] + f[i - 1][j - k]) % MOD;
        }
        for (int i = 0; i < N; ++ i )
            for (int j = 1; j < M; ++ j )
                f[i][j] = (f[i][j] + f[i][j - 1]) % MOD;
        
        // for (int i = 0; i < 5; ++ i ) {
        //     for (int j = 0; j <= 15; ++ j )
        //         cout << f[i][j] << ' ';
        //     cout << endl;
        // }
    }
    
    LL get(string s, int mx) {
        int n = s.size();
        LL res = 0;
        for (int i = 0; i < s.size(); ++ i ) {
            int x = s[i] - '0';
            for (int j = 0; j < x; ++ j )
                if (mx - j >= 0)
                    res = (res + f[n - i - 1][mx - j]);
            mx -= x;
            if (mx < 0)
                break;
            if (i == n - 1)
                res = (res + 1) % MOD;
        }
        return res;
    }
    
    string sub_one(string s) {
        vector<int> xs;
        for (auto c : s)
            xs.push_back(c - '0');
        reverse(xs.begin(), xs.end());
        
        bool has_sub = false;
        
        vector<int> res;
        for (int i = 0, t = 0; i < xs.size(); ++ i ) {
            t = xs[i] - t;
            if (!has_sub)
                t -= 1, has_sub = true;
            res.push_back((t + 10) % 10);
            if (t < 0)
                t = 1;
            else
                t = 0;
        }
        while (res.size() > 1 && res.back() == 0)
            res.pop_back();
        reverse(res.begin(), res.end());
        
        string ns;
        for (auto x : res)
            ns.push_back('0' + x);
        // cout << " origin = " << s << " got ns = " << ns << endl;
        return ns;
    }
    
    int count(string num1, string num2, int min_sum, int max_sum) {
        init();
        
        LL t1 = (get(num2, max_sum) - get(num2, min_sum - 1) + MOD) % MOD;
        {
            num1 = sub_one(num1);
        }
        LL t2 = (get(num1, max_sum) - get(num1, min_sum - 1) + MOD) % MOD;
        // cout << " t1 = " << t1 << " t2 = " << t2 << endl;
        return (t1 - t2 + MOD) % MOD;
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

> [!NOTE] **[LeetCode 2801. 统计范围内的步进数字数目](https://leetcode.cn/problems/count-stepping-numbers-in-range/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准数位dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 显然 是一个标准的数位dp
const static int N = 110, M = 10, MOD = 1e9 + 7;

int f[N][M];    // 长度为 i, 最前面的数字为 j 的所有 “步进数”
bool inited = false;

void init() {
    if (inited)
        return;
    inited = true;
    memset(f, 0, sizeof f);
    for (int i = 0; i < M; ++ i )   // ATTENTION 注意 0 idx
        f[1][i] = 1;
    for (int i = 2; i < N; ++ i )
        for (int j = 0; j < M; ++ j )
            for (int k = 0; k < M; ++ k )
                if (abs(j - k) == 1)
                    f[i][j] = (f[i][j] + f[i - 1][k]) % MOD;
}

class Solution {
public:
    string sub_one(string & s) {
        vector<int> C;
        for (int i = s.size() - 1, t = 1; i >= 0; -- i ) {
            t = s[i] - '0' - t;
            C.push_back((t + 10) % 10);
            if (t < 0)
                t = 1;
            else
                t = 0;
        }
        while (C.size() > 1 && C.back() == 0)
            C.pop_back();
        string t;
        for (int i = C.size() - 1; i >= 0; -- i )
            t += '0' + C[i];
        return t;
    }
    
    int get(string s) {
        if (s == "0")
            return 0;
        
        int t = 0, n = s.size(), last = -1;
        // 长度小于的
        for (int i = 1; i < n; ++ i )
            for (int j = 1; j < M; ++ j )   // 不能 0 起始
                t = (t + f[i][j]) % MOD;
        
        // 长度等于的
        for (int i = 0; i < n; ++ i ) {
            int x = s[i] - '0';
            
            for (int j = (i ? 0 : 1); j < x; ++ j ) // ATTENTION 不能有前导 0
                if (last == -1 || abs(last - j) == 1)
                    t = (t + f[n - i][j]) % MOD;
            
            if (last != - 1 && abs(last - x) != 1)
                break;
            last = x;
            
            if (i == n - 1)
                t = (t + 1) % MOD;
        }
        
        return t;
    }
    
    int countSteppingNumbers(string low, string high) {
        init();
        // 需要让 low -1
        return (get(high) - get(sub_one(low)) + MOD) % MOD;
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

### TODO 汇总

#### 拓展 n以内所有数字 每个数字在各个数位出现多少次

#### 拓展 从 a 到 b 的所有数字 每个数字在各个数位出现多少次

[LibreOJ #10169. 「一本通 5.3 练习 4」数字计数](https://loj.ac/p/10169)

[Luogu P2602 [ZJOI2010]数字计数](https://www.luogu.com.cn/problem/P2602)

[AcWing 338 计数问题](https://www.acwing.com/problem/content/340/)



### 多维状态

> [!NOTE] **[AcWing 1411. 二五](https://www.acwing.com/problem/content/1413/)**
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
// 数位dp
// 5000000复杂度
#include <bits/stdc++.h>
using namespace std;

const int N = 6;

int f[N][N][N][N][N];
int p[30];
bool st[30];
int id[N][N];

bool check(int a, int b) {
    return p[b] == -1 || p[b] == a;
}

// 枚举从小到达每个字母 [k] 放到哪一个位置 每行的个数状态分别abcde
int dp(int a, int b, int c, int d, int e, int k) {
    if (k == 25) return 1;
    auto & v = f[a][b][c][d][e];
    if (v != -1) return v;
    v = 0;
    if (a < 5 && check(k, id[1][a + 1])) v += dp(a + 1, b, c, d, e, k + 1);
    if (b < a && check(k, id[2][b + 1])) v += dp(a, b + 1, c, d, e, k + 1);
    if (c < b && check(k, id[3][c + 1])) v += dp(a, b, c + 1, d, e, k + 1);
    if (d < c && check(k, id[4][d + 1])) v += dp(a, b, c, d + 1, e, k + 1);
    if (e < d && check(k, id[5][e + 1])) v += dp(a, b, c, d, e + 1, k + 1);
    return v;
}

void work1() {
    int n;
    cin >> n;
    memset(p, -1, sizeof p);
    for (int i = 1; i <= 25; ++ i )
        for (int j = 0; j < 25; ++ j ) {
            if (st[j]) continue;
            p[i] = j, st[j] = true;
            memset(f, -1, sizeof f);
            int t = dp(0, 0, 0, 0, 0, 0);
            if (t >= n) break;
            n -= t;
            p[i] = -1, st[j] = false;
        }
    string res;
    for (int i = 1; i <= 25; ++ i ) res += p[i] + 'A';
    cout << res << endl;
}

void work2() {
    char str[30];
    cin >> str + 1;
    int res = 0;
    memset(p, -1, sizeof p);
    for (int i = 1; i <= 25; ++ i ) {
        int u = str[i] - 'A';
        for (int j = 0; j < u; ++ j ) {
            if (st[j]) continue;
            p[i] = j, st[j] = true;
            memset(f, -1, sizeof f);
            res += dp(0, 0, 0, 0, 0, 0);
            p[i] = -1, st[j] = false;
        }
        p[i] = u, st[u] = false;
    }
    cout << res + 1 << endl;
}

int main() {
    for (int i = 1, k = 1; i <= 5; ++ i )
        for (int j = 1; j <= 5; ++ j , ++ k )
            id[i][j] = k;
    char op;
    cin >> op;
    if (op == 'N') work1();
    else work2();
    
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

> [!NOTE] **[Luogu 编码](https://www.luogu.com.cn/problem/P1246)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数位 dp + 组合数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 27;

int c[N][N];

// 组合数
void init() {
    for (int i = 0; i < N; ++ i )
        for (int j = 0; j <= i; ++ j )
            if (!j)
                c[i][j] = 1;
            else
                c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]);
}

int main() {
    init();
    
    string str;
    cin >> str;
    
    for (auto c : str)
        if (c < 'a' || c > 'z') {
            cout << 0 << endl;
            return 0;
        }
    
    int s = 0, n = str.size();
    // 先累加所有长度比其小的单词数
    for (int i = 1; i < n; ++ i )
        s += c[26][i];

    // 数位dp
    int last = 0;
    for (int i = 0; i < n; ++ i ) {
        // 当前位置对应的值
        int x = str[i] - 'a' + 1;
        for (int j = last + 1; j < x; ++ j )
            // 当前这一位填j 后面的n-i-1只要挑比j大的即可
            s += c[26 - j][n - i - 1];

        if (x <= last) {
            s = 0;
            break;
        }
        last = x;

        if (i == n - 1)
            s ++ ;
    }
    
    cout << s << endl;
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