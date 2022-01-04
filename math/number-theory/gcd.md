## 最大公约数

最大公约数即为 Greatest Common Divisor，常缩写为 gcd。

一组整数的公约数，是指同时是这组数中每一个数的约数的数。$\pm 1$ 是任意一组整数的公约数。

一组整数的最大公约数，是指所有公约数里面最大的一个。

那么如何求最大公约数呢？我们先考虑两个数的情况。

### 欧几里得算法

如果我们已知两个数 $a$ 和 $b$，如何求出二者的最大公约数呢？

不妨设 $a > b$

我们发现如果 $b$ 是 $a$ 的约数，那么 $b$ 就是二者的最大公约数。
下面讨论不能整除的情况，即 $a = b \times q + r$，其中 $r < b$。

我们通过证明可以得到 $\gcd(a,b)=\gcd(b,a \bmod b)$，过程如下：

* * *

设 $a=bk+c$，显然有 $c=a \bmod b$。设 $d \mid a,~d \mid b$，则 $c=a-bk, \frac{c}{d}=\frac{a}{d}-\frac{b}{d}k$。

由右边的式子可知 $\frac{c}{d}$ 为整数，即 $d \mid c$ 所以对于 $a,b$ 的公约数，它也会是 $a \bmod b$ 的公约数。

反过来也需要证明：

设 $d \mid b,~\mid (a \bmod b)$，我们还是可以像之前一样得到以下式子 $\frac{a\bmod b}{d}=\frac{a}{d}-\frac{b}{d}k,~\frac{a\bmod b}{d}+\frac{b}{d}k=\frac{a}{d}$。

因为左边式子显然为整数，所以 $\frac{a}{d}$ 也为整数，即 $d \mid a$，所以 $b,a\bmod b$ 的公约数也是 $a,b$ 的公约数。

既然两式公约数都是相同的，那么最大公约数也会相同。

所以得到式子 $\gcd(a,b)=\gcd(b,a\bmod b)$

既然得到了 $\gcd(a, b) = \gcd(b, r)$，这里两个数的大小是不会增大的，那么我们也就得到了关于两个数的最大公约数的一个递归求法。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}
```

###### **Python**

```python
# Python Version
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
```

<!-- tabs:end -->
</details>

<br>

递归至 `b == 0`（即上一步的 `a % b == 0`) 的情况再返回值即可。

上述算法被称作欧几里得算法（Euclidean algorithm）。

如果两个数 $a$ 和 $b$ 满足 $\gcd(a, b) = 1$，我们称 $a$ 和 $b$ 互质。

* * *

欧几里得算法的时间效率如何呢？下面我们证明，欧几里得算法的时间复杂度为 $O(\log n)$。

当我们求 $\gcd(a,b)$ 的时候，会遇到两种情况：

- $a < b$，这时候 $\gcd(a,b)=\gcd(b,a)$；
- $a \geq b$，这时候 $\gcd(a,b)=\gcd(b,a \bmod b)$，而对 $a$ 取模会让 $a$ 至少折半。这意味着这一过程最多发生 $O(\log n)$ 次。

第一种情况发生后一定会发生第二种情况，因此第一种情况的发生次数一定 **不多于** 第二种情况的发生次数。

从而我们最多递归 $O(\log n)$ 次就可以得出结果。

事实上，假如我们试着用欧几里得算法去求 [斐波那契数列](fibonacci.md) 相邻两项的最大公约数，会让该算法达到最坏复杂度。

### 多个数的最大公约数

那怎么求多个数的最大公约数呢？显然答案一定是每个数的约数，那么也一定是每相邻两个数的约数。我们采用归纳法，可以证明，每次取出两个数求出答案后再放回去，不会对所需要的答案造成影响。

## 最小公倍数

接下来我们介绍如何求解最小公倍数（Least Common Multiple, LCM）。

一组整数的公倍数，是指同时是这组数中每一个数的倍数的数。0 是任意一组整数的公倍数。

一组整数的最小公倍数，是指所有正的公倍数里面，最小的一个数。

### 两个数的

设 $a = p_1^{k_{a_1}}p_2^{k_{a_2}} \cdots p_s^{k_{a_s}}$，$b = p_1^{k_{b_1}}p_2^{k_{b_2}} \cdots p_s^{k_{b_s}}$

我们发现，对于 $a$ 和 $b$ 的情况，二者的最大公约数等于

$p_1^{\min(k_{a_1}, k_{b_1})}p_2^{\min(k_{a_2}, k_{b_2})} \cdots p_s^{\min(k_{a_s}, k_{b_s})}$

最小公倍数等于

$p_1^{\max(k_{a_1}, k_{b_1})}p_2^{\max(k_{a_2}, k_{b_2})} \cdots p_s^{\max(k_{a_s}, k_{b_s})}$

由于 $k_a + k_b = \max(k_a, k_b) + \min(k_a, k_b)$

所以得到结论是 $\gcd(a, b) \times \operatorname{lcm}(a, b) = a \times b$

要求两个数的最小公倍数，先求出最大公约数即可。

### 多个数的

可以发现，当我们求出两个数的 $\gcd$ 时，求最小公倍数是 $O(1)$ 的复杂度。那么对于多个数，我们其实没有必要求一个共同的最大公约数再去处理，最直接的方法就是，当我们算出两个数的 $\gcd$，或许在求多个数的 $\gcd$ 时候，我们将它放入序列对后面的数继续求解，那么，我们转换一下，直接将最小公倍数放入序列即可。

## 扩展欧几里得算法

扩展欧几里得算法（Extended Euclidean algorithm, EXGCD），常用于求 $ax+by=\gcd(a,b)$ 的一组可行解。

### 证明

设

$ax_1+by_1=\gcd(a,b)$

$bx_2+(a\bmod b)y_2=\gcd(b,a\bmod b)$

由欧几里得定理可知：$\gcd(a,b)=\gcd(b,a\bmod b)$

所以 $ax_1+by_1=bx_2+(a\bmod b)y_2$

又因为 $a\bmod b=a-(\lfloor\frac{a}{b}\rfloor\times b)$

所以 $ax_1+by_1=bx_2+(a-(\lfloor\frac{a}{b}\rfloor\times b))y_2$

$ax_1+by_1=ay_2+bx_2-\lfloor\frac{a}{b}\rfloor\times by_2=ay_2+b(x_2-\lfloor\frac{a}{b}\rfloor y_2)$

因为 $a=a,b=b$，所以 $x_1=y_2,y_1=x_2-\lfloor\frac{a}{b}\rfloor y_2$

将 $x_2,y_2$ 不断代入递归求解直至 $\gcd$（最大公约数，下同）为 `0` 递归 `x=1,y=0` 回去求解。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int Exgcd(int a, int b, int &x, int &y) {
    if (!b) {
        x = 1;
        y = 0;
        return a;
    }
    int d = Exgcd(b, a % b, x, y);
    int t = x;
    x = y;
    y = t - (a / b) * y;
    return d;
}
```

###### **Python**

```python
# Python Version
def Exgcd(a, b):
    if b == 0:
        return a, 1, 0
    d, x, y = Exgcd(b, a % b)
    return d, y, x - (a // b) * y
```

<!-- tabs:end -->
</details>

<br>

函数返回的值为 $\gcd$，在这个过程中计算 $x,y$ 即可。

### 迭代法编写拓展欧几里得算法

因为迭代的方法避免了递归，所以代码运行速度将比递归代码快一点。

```cpp
int gcd(int a, int b, int& x, int& y) {
    x = 1, y = 0;
    int x1 = 0, y1 = 1, a1 = a, b1 = b;
    while (b1) {
        int q = a1 / b1;
        tie(x, x1) = make_tuple(x1, x - q * x1);
        tie(y, y1) = make_tuple(y1, y - q * y1);
        tie(a1, b1) = make_tuple(b1, a1 - q * b1);
    }
    return a1;
}
```

如果你仔细观察 $a_1$ 和 $b_1$，你会发现，他们在迭代版本的欧几里德算法中取值完全相同，并且以下公式无论何时（在 while 循环之前和每次迭代结束时）都是成立的：$x \cdot a +y \cdot b =a_1$ 和 $x_1 \cdot a +y_1 \cdot b= b_1$。因此，该算法肯定能正确计算出 $\gcd$。

最后我们知道 $a_1$ 就是要求的 $\gcd$，有 $x \cdot a +y \cdot b =g$。

#### 矩阵的解释

对于正整数 $a$ 和 $b$ 的一次辗转相除即 $\gcd(a,b)=\gcd(b,a\bmod b)$ 使用矩阵表示如

$$
\begin{bmatrix}
b\\a\bmod b
\end{bmatrix}
=
\begin{bmatrix}
0&1\\1&-\lfloor a/b\rfloor
\end{bmatrix}
\begin{bmatrix}
a\\b
\end{bmatrix}
$$

其中向下取整符号 $\lfloor c\rfloor$ 表示不大于 $c$ 的最大整数。我们定义变换 $\begin{bmatrix}a\\b\end{bmatrix}\mapsto \begin{bmatrix}0&1\\1&-\lfloor a/b\rfloor\end{bmatrix}\begin{bmatrix}a\\b\end{bmatrix}$。

易发现欧几里得算法即不停应用该变换，有

$$
\begin{bmatrix}
\gcd(a,b)\\0
\end{bmatrix}
=
\left(
\cdots 
\begin{bmatrix}
0&1\\1&-\lfloor a/b\rfloor
\end{bmatrix}
\begin{bmatrix}
1&0\\0&1
\end{bmatrix}
\right)
\begin{bmatrix}
a\\b
\end{bmatrix}
$$

令

$$
\begin{bmatrix}
x_1&x_2\\x_3&x_4
\end{bmatrix}
=
\cdots 
\begin{bmatrix}
0&1\\1&-\lfloor a/b\rfloor
\end{bmatrix}
\begin{bmatrix}
1&0\\0&1
\end{bmatrix}
$$

那么

$$
\begin{bmatrix}
\gcd(a,b)\\0
\end{bmatrix}
=
\begin{bmatrix}
x_1&x_2\\x_3&x_4
\end{bmatrix}
\begin{bmatrix}
a\\b
\end{bmatrix}
$$

满足 $a\cdot x_1+b\cdot x_2=\gcd(a,b)$ 即扩展欧几里得算法，注意在最后乘了一个单位矩阵不会影响结果，提示我们可以在开始时维护一个 $2\times 2$ 的单位矩阵编写更简洁的迭代方法如

```cpp
int exgcd(int a, int b, int &x, int &y) {
    int x1 = 1, x2 = 0, x3 = 0, x4 = 1;
    while (b != 0) {
        int c = a / b;
        std::tie(x1, x2, x3, x4, a, b) =
            std::make_tuple(x3, x4, x1 - x3 * c, x2 - x4 * c, b, a - b * c);
    }
    x = x1, y = x2;
    return a;
}
```

这种表述相较于递归更简单。

## 应用

- [10104 - Euclid Problem](https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=1045)
- [GYM - (J) once upon a time](http://codeforces.com/gym/100963)
- [UVA - 12775 - Gift Dilemma](https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=4628)


## 习题

### 约数

> [!NOTE] **[AcWing 872. 最大公约数](https://www.acwing.com/problem/content/874/)**
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
#include<bits/stdc++.h>
using namespace std;

int gcd(int a, int b) {
    return b ? gcd(b, a % b) : a;
}

int main() {
    int n;
    cin >> n;
    while (n -- ) {
        int a, b;
        cin >> a >> b;
        cout << gcd(a, b) << endl;
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


> [!NOTE] **[AcWing 869. 试除法求约数](https://www.acwing.com/problem/content/871/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

vector<int> get_divisor(int x) {
    vector<int> res;
    for (int i = 1; i <= x / i; ++ i )
        if (x % i == 0) {
            res.push_back(i);
            if (i != x / i) res.push_back(x / i);
        }
    sort(res.begin(), res.end());
    return res;
}

int main() {
    int n;
    cin >> n;
    
    while (n -- ) {
        int x;
        cin >> x;
        auto res = get_divisor(x);
        
        for (auto x : res) cout << x << ' ';
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

* * *

> [!NOTE] **[AcWing 870. 约数个数](https://www.acwing.com/problem/content/872/)**
> 
> 题意: TODO


> [!TIP] **思路**
> 
> 分解质因子

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 求一系列数的乘积的约数个数
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int mod = 1e9 + 7;

int main() {
    int n;
    cin >> n;
    
    unordered_map<int, int> primes;
    
    while (n -- ) {
        int x;
        cin >> x;
        
        for (int i = 2; i <= x / i; ++ i )
            while (x % i == 0)
                x /= i, ++ primes[i];
        if (x > 1) ++ primes[x];
    }
    LL res = 1;
    for (auto [v, c] : primes) res = res * (c + 1) % mod;
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

> [!NOTE] **[AcWing 871. 约数之和](https://www.acwing.com/problem/content/873/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> $f(n)=(p_1^0+p_1^1+…p_1^a1)(p_2^0+p_2^1+…p_2^a2)…(p_k^0+p_k^1+…p_k^ak)$
> 
> 可以乘法逆元加速求积运算

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int mod = 1e9 + 7;

int main() {
    int n;
    cin >> n;
    unordered_map<int, int> primes;
    
    while (n -- ) {
        int x;
        cin >> x;
        for (int i = 2; i <= x / i; ++ i )
            while (x % i == 0)
                x /= i, ++ primes[i];
        if (x > 1) ++ primes[x];
    }
    LL res = 1;
    for (auto [v, c] : primes) {
        LL t = 1;
        // p1^0 + p1^1 + .. + p1^c     [共c+1项之和]
        while (c -- ) t = (t * v + 1) % mod;
        // 累乘积
        res = res * t % mod;
    }
    cout << res << endl;
    return 0;
}
```

##### **C++ 乘法逆元加速**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int MOD = 1e9 + 7;

int n;

int qpow(int a, int b) {
    int ret = 1;
    while (b) {
        if (b & 1)
            ret = (LL)ret * a % MOD;
        a = (LL)a * a % MOD;
        b >>= 1;
    }
    return ret;
}

int main() {
    cin >> n;
    
    unordered_map<int, int> prime;
    while (n -- ) {
        int x;
        cin >> x;
        for (int i = 2; i <= x / i; ++ i )
            while (x % i == 0) {
                x /= i;
                prime[i] ++ ;
            }
        if (x > 1)
            prime[x] ++ ;
    }
    
    LL res = 1;
    for (auto [p, c] : prime) {
        // 使用乘法逆元加速计算
        // s = a0 * (1 - q^n) / (1 - q)
        // n = c + 1;
        // 除q-1等于乘q-1的逆元
        LL t = (LL)(qpow(p, c + 1) - 1 + MOD) % MOD * qpow(p - 1, MOD - 2) % MOD;
        res = res * t % MOD;
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

> [!NOTE] **[AcWing 97. 约数之和](https://www.acwing.com/problem/content/99/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> ![png](https://github.com/OpenKikCoc/AcWing/raw/master/02_senior/97/题解.png)
> 
> 约数个数 & 约数之和 公式
> 
> 本题本身也有公式

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int mod = 9901;

int qmi(int a, int k) {
    int res = 1;
    a %= mod;
    while (k) {
        if (k & 1) res = res * a % mod;
        a = a * a % mod;
        k >>= 1;
    }
    return res;
}

int sum(int p, int k) {
    if (k == 1) return 1;
    if (k % 2 == 0) return (1 + qmi(p, k / 2)) * sum(p, k / 2) % mod;
    return (sum(p, k - 1) + qmi(p, k - 1)) % mod;
}

int main() {
    int a, b;
    cin >> a >> b;
    int res = 1;
    
    // 对a分解质因数
    for (int i = 2; i * i <= a; ++ i )
        if (a % i == 0) {
            int s = 0;
            while (a % i == 0) {
                a /= i, ++ s;
            }
            res = res * sum(i, b * s + 1) % mod;
        }
    if (a > 1) res = res * sum(a, b + 1) % mod;
    if (a == 0) res = 0;
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

> [!NOTE] **[LeetCode 1808. 好因子的最大数目](https://leetcode-cn.com/problems/maximize-number-of-nice-divisors/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 根据题意显然约数个数定理
> 
> 又因定理累乘显然需要积最大，同减绳子
> 
> 忘写快速幂 TLE 1

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 根据约数个数定理 对 p 分解
    // 变为剪绳子的题
    using LL = long long;
    const LL MOD = 1e9 + 7;
    
    LL quick_pow(LL x, LL k, LL p) {
        x %= p;
        LL ret = 1;
        while (k) {
            if (k & 1)
                ret = (ret * x) % p;
            x = (x * x) % p;
            k >>= 1;
        }
        return ret;
    }
    
    int maxNiceDivisors(int p) {
        if (p <= 2)
            return p;
        LL res = 1;
        if (p % 3 == 1)
            res *= 4, p -= 4;
        // 需快速幂
        // while (p >= 3)
        //     res = (res * 3) % MOD, p -= 3;
        if (p >= 3) {
            LL k = p / 3;
            res = (res * quick_pow(3, k, MOD)) % MOD;
            p -= k * 3;
        }
        if (p == 2)
            res = (res * 2) % MOD;
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

> [!NOTE] **[Luogu 因子和](https://www.luogu.com.cn/problem/P1593)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 约束和 等比数列求和优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PII = pair<int, int>;
const int MOD = 9901;

LL a, b;

LL qpow(LL a, LL b) {
    // ATTENTION: DO NOT [b %= MOD]
    a %= MOD;
    LL ret = 1;
    while (b) {
        if (b & 1)
            ret = (ret * a) % MOD;
        a = (a * a) % MOD;
        b >>= 1;
    }
    return ret;
}

// 等比数列求和
// sum = (p^n - 1) / (p - 1)
LL smul(LL p, LL s) {
    s *= b; // a^b 扩大b倍
    s ++ ;  // 此时 s = n - 0 + 1
    
    LL t = 0;
    if (p % MOD == 1)
        t = s % MOD;  // 逆元不存在
    else
        // (p^n-1) * modniv(p-1)
        // ATTENTION -1 需要 + MOD 有个case在这里
        t = (qpow(p, s) - 1 + MOD) % MOD *
            qpow(p - 1, MOD - 2) % MOD;
    return t;
}

int main() {
    cin >> a >> b;
    
    vector<PII> ve;
    for (int i = 2; i <= a / i; ++ i )
        if (a % i == 0) {
            int t = 0;
            while (a % i == 0)
                a /= i, t ++ ;
            ve.push_back({i, t});
        }
    if (a > 1)
        ve.push_back({a, 1});
    
    LL res = 1;
    for (auto [p, s] : ve)
        res = res * smul(p, s) % MOD;
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

> [!NOTE] **[LeetCode 650. 只有两个键的键盘](https://leetcode-cn.com/problems/2-keys-keyboard/)**
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
    // 约数做法
    // [122...22] [122....22] ... 操作分解
    // n = p1 * p2 * ... * pn 
    // res = p1 + p2 + ... + pn
    int minSteps(int n) {
        int res = 0;
        for (int i = 2; i <= n / i; ++ i )
            while (n % i == 0)
                res += i, n /= i;
        if (n > 1) res += n;
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int minSteps(int n) {
        vector<int> f(n + 1, INT_MAX);
        f[1] = 0;
        for (int i = 2; i <= n; ++ i )
            if (n % i == 0)
                // 枚举约数
                for (int j = 1; j * j <= n; ++ j )
                    if (i % j == 0) {
                        f[i] = min(f[i], f[j] + i / j);
                        f[i] = min(f[i], f[i / j] + j);
                    }
        return f[n];
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

> [!NOTE] **[LeetCode 1250. 检查「好数组」](https://leetcode-cn.com/problems/check-if-it-is-a-good-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 类似倒水问题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool isGoodArray(vector<int>& nums) {
        int n = nums.size();
        int res = nums[0];
        for (int i = 1; i < n; ++i) res = __gcd(res, nums[i]);
        return res == 1;
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

> [!NOTE] **[LeetCode 1819. 序列中不同最大公约数的数目](https://leetcode-cn.com/problems/number-of-different-subsequences-gcds/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 逆向考虑枚举每一个 GCD 即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
ass Solution {
public:
    int countDifferentSubsequenceGCDs(vector<int>& nums) {
        vector<bool> st(200010);
        int mxv = 0;
        for (auto v : nums) {
            mxv = max(mxv, v);
            st[v] = true;
        }
        
        int res = 0;
        for (int i = 1; i <= mxv; ++ i ) {
            int g = -1;
            for (int j = i; j <= mxv; j += i)
                if (st[j]) {
                    if (g == -1)
                        g = j;
                    else
                        g = __gcd(j, g); // ATTENTION
                }
            if (g == i)
                res ++ ;
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

> [!NOTE] **[LeetCode 2001. 可互换矩形的组数](https://leetcode-cn.com/problems/number-of-pairs-of-interchangeable-rectangles/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> `double` 统计即可 对于本题只要保证精度即可边统计边更新记录
> 
> 更稳妥的是使用 gcd

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    
    // wi * hj == wj * hi
    long long interchangeableRectangles(vector<vector<int>>& recs) {
        unordered_map<double, int> cnt;
        LL res = 0;
        for (auto rec : recs) {
            double w = rec[0], h = rec[1];
            res += cnt[w / h];
            cnt[w / h] ++ ;
        }
        return res;
    }
};
```

##### **C++ gcd**

```cpp
class Solution {
public:
    using LL = long long;
    LL interchangeableRectangles(vector<vector<int>>& rectangles) {
        const int n = rectangles.size();
        unordered_map<LL, int> seen;

        LL ans = 0;
        for (const auto &r : rectangles) {
            int w = r[0], h = r[1];
            const int g = __gcd(r[0], r[1]);

            w /= g; h /= g;
            LL hv = (LL)(w) * 100001 + h;

            ans += seen[hv];
            seen[hv]++;
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

### 欧几里得

> [!NOTE] **[AcWing 877. 扩展欧几里得算法](https://www.acwing.com/problem/content/879/)**
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
/*
ax + by = d
求出一组可行解 x0 y0后  k为任意整数
x = x0 - b / d * k;
y = y0 + a / d * k
*/
#include<bits/stdc++.h>
using namespace std;

int exgcd(int a, int b, int & x, int & y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int main() {
    int n;
    cin >> n;
    while (n -- ) {
        int a, b;
        cin >> a >> b;
        int x, y;
        exgcd(a, b, x, y);
        cout << x << " " << y << endl;
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

> [!NOTE] **[AcWing 878. 线性同余方程](https://www.acwing.com/problem/content/880/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 拓展欧几里得

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
         a * x === b % m
     ==> ax = my + b
     ==> ax - my = b
     ==> ax + my'= b
         gcd(a, m) | b 则有解
      x = x0 * b / d % m        相当于倍增
*/
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

int exgcd(int a, int b, int & x, int & y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int main() {
    int n;
    cin >> n;
    while (n -- ) {
        int a, b, m;
        cin >> a >> b >> m;
        int x, y;
        int d = exgcd(a, m, x, y);
        if (b % d) cout << "impossible" << endl;
        else cout << (LL)b / d * x % m << endl;
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

> [!NOTE] **[LeetCode 780. 到达终点](https://leetcode-cn.com/problems/reaching-points/)**
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
    bool reachingPoints(int sx, int sy, int tx, int ty) {
        while (tx >= sx && ty >= sy) {
            if (tx == ty) break;
            if (tx > ty) {
                if (ty > sy) tx %= ty;
                else return (tx - sx) % ty == 0;
            } else {
                if (tx > sx) ty %= tx;
                else return (ty - sy) % tx == 0;
            }
        }

        return sx == tx && sy == ty;
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

### 负进制

> [!NOTE] **[Luogu [NOIP2000 提高组] 进制转换](https://www.luogu.com.cn/problem/P1017)**
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

int n, m;

int main() {
    cin >> n >> m;

    cout << n << "=";

    string s;
    while (n) {
        int mod = n % m;
        n /= m;
        // ATTENTION
        // 出现负数 则从上一位借1
        if (mod < 0)
            mod -= m, ++n;

        if (mod < 10)
            s.push_back('0' + mod);
        else
            s.push_back('A' + (mod - 10));
    }
    reverse(s.begin(), s.end());
    cout << s << "(base" << m << ")" << endl;

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

### gcd 思想

> [!NOTE] **[Luogu zzc 种田](https://www.luogu.com.cn/problem/P2660)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> gcd 的思想

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
LL s;

/*
// 1. dfs 直接 MLE

    dfs(x, y);

void dfs(LL x, LL y) {
    if (x == 0 || y == 0)
        return;
    if (x < y)
        swap(x, y);
    // x >= y
    s += y * 4;
    dfs(y, x - y);
}
*/

/*
// 2. 模拟最后一个点超时

    while (x && y) {
        if (x < y)
            swap(x, y);
        s += y * 4;
        {
            LL nx = y, ny = x - y;
            x = nx, y = ny;
        }
    }
*/

int main() {
    LL x, y;
    cin >> x >> y;

    // 由做法2 考虑优化
    // 每次不要只切一个正方形 可以且多个
    // 即切 max(x, y) / min(x, y) 个  对应如下情况
    // 【长10 宽1】这样的 大幅优化
    while (x && y) {
        if (x < y)
            swap(x, y);
        // x >= y
        s += y * 4 * (x / y);
        x %= y;
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

> [!NOTE] **[LeetCode 1354. 多次求和构造目标数组](https://leetcode-cn.com/problems/construct-target-array-with-multiple-sums/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 比赛中好多暴力模拟代码都过了 主要应对两个case
> 
> 对于 [2] left计算得0 所以最好放在第一行特判
> 
> 对于 [1,1000000000] 如果当前取出的maxv在处理后还会是maxv 需要考虑：
> 
> 每次减去的数值都是其他数的值(sum-maxv)=left
> 
> 因此考虑取余 但要-1因为减后的结果不能是0
> 
> 【此办法减少循环中操作优先队列的次数】


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// [2]
// [1,1000000000]
class Solution {
public:
    bool isPossible(vector<int>& target) {
        if (target.size() == 1) return target[0] == 1;
        int n = target.size();
        long long sum = 0, maxv, newv, left, y, k;
        priority_queue<int> pq;
        for (auto v : target) {
            sum += v;
            pq.push(v);
        }
        while (!pq.empty() && pq.top() > 1) {
            maxv = pq.top();
            pq.pop();
            // check 新数字
            if (2 * maxv - sum < 1) return false;
            // 其他数的和为 sum-maxv 新数为 maxv - (sum-maxv) 新sum为 maxv
            left = sum - maxv;
            /*
            if (!pq.empty()) {
                y = pq.top();
                if (y == 1)
                    k = (maxv - y + left - 1) / left;
                else
                    k = (maxv - y) / left + 1;
                maxv -= k * left;
                if (maxv < 1) return false;
                sum -= k * left;
            }
            */
            k = (maxv - 1) / left;
            maxv -= k * left;
            sum -= k * left;

            pq.push(maxv);
        }
        return true;
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