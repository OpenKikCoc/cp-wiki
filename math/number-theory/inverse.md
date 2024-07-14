> [!NOTE] **求解逆元的方式**
> 
> 如本章节所述，不仅有快速幂法，还有拓展欧几里得和线性计算方法


本文介绍模意义下乘法运算的逆元（Modular Multiplicative Inverse），并介绍如何使用扩展欧几里德算法（Extended Euclidean algorithm）求解乘法逆元

## 逆元简介

如果一个线性同余方程 $ax \equiv 1 \pmod b$，则 $x$ 称为 $a \bmod b$ 的逆元，记作 $a^{-1}$。

## 如何求逆元

### 扩展欧几里得法

> [!NOTE] **模板代码**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
void exgcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1, y = 0;
        return;
    }
    exgcd(b, a % b, y, x);
    y -= a / b * x;
}
```

###### **Python**
    
```python
# Python Version
def exgcd(a, b, x, y):
    if b == 0:
        x, y = 1, 0
        return
    exgcd(b, a % b, y, x)
    y = y - (a // b * x)
```

<!-- tabs:end -->
</details>

<br>


扩展欧几里得法和求解 [线性同余方程](./linear-equation.md) 是一个原理，在这里不展开解释。

### 快速幂法

因为 $ax \equiv 1 \pmod b$；

所以 $ax \equiv a^{b-1} \pmod b$（根据 [费马小定理](./fermat.md)）；

所以 $x \equiv a^{b-2} \pmod b$。

然后我们就可以用快速幂来求了。

> [!NOTE] **模板代码**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
inline int qpow(long long a, int b) {
    int ans = 1;
    a = (a % p + p) % p;
    for (; b; b >>= 1) {
        if (b & 1) ans = (a * ans) % p;
        a = (a * a) % p;
    }
    return ans;
}
```

###### **C++ AcWing**

```cpp
// C++ Version
//      a / b === a * x (mod m)
// -->  a / b === a * b^-1 (mod m)
//      b * b^-1 === 1 (mod m)
//      b * x === 1 (mod m)
//      x 是 b 的逆元
//
//      m 为质数 则
//      b^p-1 === 1 (mod p)
//      b * b^p-2 === 1 (mod p)
//      本质要求 b^p-2 mod p

/*
                a / b ≡ a * x (mod n)
两边同乘b可得
                a ≡ a * b * x (mod n)
即
                1 ≡ b * x (mod n)
同
                b * x ≡ 1 (mod n)
由费马小定理可知，当n为质数时
                b ^ (n - 1) ≡ 1 (mod n)
拆一个b出来可得
                b * b ^ (n - 2) ≡ 1 (mod n)
故当n为质数时，b的乘法逆元
                x = b ^ (n - 2)

当n不是质数时，可以用扩展欧几里得算法求逆元：
a有逆元的充要条件是a与p互质，所以gcd(a, p) = 1
假设a的逆元为x，那么有a * x ≡ 1 (mod p)
等价：ax + py = 1
exgcd(a, p, x, y)
*/
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

LL qmi(int a, int b, int p) {
    LL res = 1;
    while (b) {
        if (b & 1) res = res * a % p;
        a = a * (LL)a % p;
        b >>= 1;
    }
    return res;
}

int main() {
    int n;
    cin >> n;
    while (n -- ) {
        int a, p;
        cin >> a >> p;
        // 如果b是p的倍数则无解
        int res = qmi(a, p - 2, p);
        if (a % p) cout << res << endl;
        else cout << "impossible" << endl;
    }
    return 0;
}
```


###### **Python**
    
```python
# Python Version
def qpow(a, b):
    ans = 1
    a = (a % p + p) % p
    while b:
        if b & 1:
            ans = (a * ans) % p
            a = (a * a) % p
        b >>= 1
    return ans
```

<!-- tabs:end -->
</details>

<br>

注意使用 [费马小定理](math/fermat.md) 需要限制 $b$ 是一个素数，而扩展欧几里得算法只要求 $\gcd(a, p) = 1$。

### 线性求逆元

求出 $1,2,...,n$ 中每个数关于 $p$ 的逆元。

如果对于每个数进行单次求解，以上两种方法就显得慢了，很有可能超时，所以下面来讲一下如何线性（$O(n)$）求逆元。

首先，很显然的 $1^{-1} \equiv 1 \pmod p$；

> [!NOTE] **证明**
> 
> 对于 $\forall p \in \mathbf{Z}$，有 $1 \times 1 \equiv 1 \pmod p$ 恒成立，故在 $p$ 下 $1$ 的逆元是 $1$，而这是推算出其他情况的基础。

其次对于递归情况 $i^{-1}$，我们令 $k = \lfloor \frac{p}{i} \rfloor$，$j = p \bmod i$，有 $p = ki + j$。再放到 $\mod p$ 意义下就会得到：$ki+j \equiv 0 \pmod p$；

两边同时乘 $i^{-1} \times j^{-1}$：

$kj^{-1}+i^{-1} \equiv 0 \pmod p$

$i^{-1} \equiv -kj^{-1} \pmod p$

再带入 $j = p \bmod i$，有 $p = ki + j$，有：

$i^{-1} \equiv -\lfloor\frac{p}{i}\rfloor (p \bmod i)^{-1} \pmod p$

我们注意到 $p \bmod i < i$，而在迭代中我们完全可以假设我们已经知道了所有的模 $p$ 下的逆元 $j^{-1}, j < i$。

故我们就可以推出逆元，利用递归的形式，而使用迭代实现：

$$
i^{-1} \equiv \begin{cases}
    1,                                           & \text{if } i = 1, \\
    -\lfloor\frac{p}{i}\rfloor (p \bmod i)^{-1}, & \text{otherwises}.
\end{cases} \pmod p
$$



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
inv[1] = 1;
for (int i = 2; i <= n; ++i) {
    inv[i] = (long long)(p - p / i) * inv[p % i] % p;
}
```

###### **Python**

```python
# Python Version
inv[1] = 1
for i in range(2, n + 1):
    inv[i] = (p - p // i) * inv[p % i] % p
```

<!-- tabs:end -->
</details>

<br>

使用 $p-\lfloor \dfrac{p}{i} \rfloor$ 来防止出现负数。

另外我们注意到我们没有对 `inv[0]` 进行定义却可能会使用它：当 $i | p$ 成立时，我们在代码中会访问 `inv[p % i]`，也就是 `inv[0]`，这是因为当 $i | p$ 时不存在 $i$ 的逆元 $i^{-1}$。[线性同余方程](./linear-equation.md) 中指出，如果 $i$ 与 $p$ 不互素时不存在相应的逆元（当一般而言我们会使用一个大素数，比如 $10^9 + 7$ 来确保它有着有效的逆元）。因此需要指出的是：如果没有相应的逆元的时候，`inv[i]` 的值是未定义的。

另外，根据线性求逆元方法的式子：$i^{-1} \equiv -kj^{-1} \pmod p$

递归求解 $j^{-1}$, 直到 $j=1$ 返回 $1$。

中间优化可以加入一个记忆化来避免多次递归导致的重复，这样求 $1,2,...,n$ 中所有数的逆元的时间复杂度仍是 $O(n)$。

**注意**：如果用以上给出的式子递归进行单个数的逆元求解，目前已知的时间复杂度的上界为 $O(n^{\frac 1 3})$，具体请看 [知乎讨论](https://www.zhihu.com/question/59033693)。算法竞赛中更好地求单个数的逆元的方法有扩展欧几里得法和快速幂法。

### 线性求任意 n 个数的逆元

上面的方法只能求 $1$ 到 $n$ 的逆元，如果需要求任意给定 $n$ 个数（$1 \le a_i < p$）的逆元，就需要下面的方法：

首先计算 $n$ 个数的前缀积，记为 $s_i$，然后使用快速幂或扩展欧几里得法计算 $s_n$ 的逆元，记为 $sv_n$。

因为 $sv_n$ 是 $n$ 个数的积的逆元，所以当我们把它乘上 $a_n$ 时，就会和 $a_n$ 的逆元抵消，于是就得到了 $a_1$ 到 $a_{n-1}$ 的积逆元，记为 $sv_{n-1}$。

同理我们可以依次计算出所有的 $sv_i$，于是 $a_i^{-1}$ 就可以用 $s_{i-1} \times sv_i$ 求得。

所以我们就在 $O(n + \log p)$ 的时间内计算出了 $n$ 个数的逆元。



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
s[0] = 1;
for (int i = 1; i <= n; ++i) s[i] = s[i - 1] * a[i] % p;
sv[n] = qpow(s[n], p - 2);
// 当然这里也可以用 exgcd 来求逆元,视个人喜好而定.
for (int i = n; i >= 1; --i) sv[i - 1] = sv[i] * a[i] % p;
for (int i = 1; i <= n; ++i) inv[i] = sv[i] * s[i - 1] % p;
```

###### **Python**
    
```python
# Python Version
s[0] = 1
for i in range(1, n + 1):
    s[i] = s[i - 1] * a[i] % p
sv[n] = qpow(s[n], p - 2)
# 当然这里也可以用 exgcd 来求逆元,视个人喜好而定.
for i in range(n, 0, -1):
    sv[i - 1] = sv[i] * a[i] % p
for i in range(1, n + 1):
    inv[i] = sv[i] * s[i - 1] % p
```

<!-- tabs:end -->
</details>

<br>

## 逆元练习题

[乘法逆元](https://loj.ac/problem/110)

[乘法逆元 2](https://loj.ac/problem/161)

[「NOIP2012」同余方程](https://loj.ac/problem/2605)

[「AHOI2005」洗牌](https://www.luogu.com.cn/problem/P2054)

[「SDOI2016」排列计数](https://loj.ac/problem/2034)

## 习题

> [!NOTE] **[LeetCode 1622. 奇妙序列](https://leetcode.cn/problems/fancy-sequence/)** [TAG]
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
#define LL long long

class Fancy {
private:
    vector<LL> nums;
    LL add, mul;
    const int mod = 1000000007;
    
    LL power(LL x, int y) {
        LL tot = 1, p = x;
        for (; y; y >>= 1) {
            if (y & 1)
                tot = (tot * p) % mod;
            p = (p * p) % mod;
        }
        return tot;
    }

public:
    Fancy() {
        add = 0;
        mul = 1;
    }
    
    void append(int val) {
        val = ((val - add) % mod + mod) % mod;
        val = (val * power(mul, mod - 2)) % mod;
        nums.push_back(val);
    }
    
    void addAll(int inc) {
        add = (add + inc) % mod;
    }
    
    void multAll(int m) {
        add = add * m % mod;
        mul = mul * m % mod;
    }
    
    int getIndex(int idx) {
        if (idx >= nums.size())
            return -1;
        return (nums[idx] * mul + add) % mod;
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

> [!NOTE] **[LeetCode 2400. 恰好移动 k 步到达某一位置的方法数目](https://leetcode.cn/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> DP 或组合数
> 
> **一种新的线形求逆元的方式**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ DP**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1010, DIFF = 501, MOD = 1e9 + 7;
    
    // [-500, 1500] instead of [-1000, 2000]
    LL f[N * 2], g[N * 2];
    
    int numberOfWays(int startPos, int endPos, int k) {
        memset(f, 0, sizeof f);
        f[startPos + DIFF] = 1;
        
        for (int _ = 0; _ < k; ++ _ ) {
            memcpy(g, f, sizeof g);
            for (int j = -500; j <= 1500; ++ j )
                f[j + DIFF] = (g[j - 1 + DIFF] + g[j + DIFF + 1]) % MOD;
        }
        return f[endPos + DIFF];
    }
};
```

##### **C++ 组合数**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1010, MOD = 1e9 + 7;

    int f[N][N];

    int numberOfWays(int startPos, int endPos, int k) {
        int d = abs(startPos - endPos);
        if ((d + k) % 2 || d > k)
            return 0;
        
        for (int i = 0; i <= k; ++ i )
            for (int j = 0; j <= i; ++ j )
                if (!j)
                    f[i][j] = 1;
                else
                    f[i][j] = (f[i - 1][j] + f[i - 1][j - 1]) % MOD;
        // 假定向正方向走 a 步 反方向 k-a 步
        // 则 a - (k - a) = d => a = (d + k) / 2
        // C[k, (d + k) / 2]
        return f[k][(d + k) / 2];
    }
};
```

##### **C++ 组合数 新的逆元求解方式**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1010, MOD = 1e9 + 7;

    LL f[N], g[N], v[N];

    int C(int n, int m) {
        if (m == 0)
            return 1;
        return f[n] * v[m] % MOD * v[n - m] % MOD;
    }

    int numberOfWays(int startPos, int endPos, int k) {
        int d = abs(startPos - endPos);
        if ((d + k) % 2 || d > k)
            return 0;
        
        // 假定向正方向走 a 步 反方向 k-a 步
        // 则 a - (k - a) = d => a = (d + k) / 2
        // C[k, (d + k) / 2]
        f[0] = g[0] = v[0] = 1;
        f[1] = g[1] = v[1] = 1;
        for (int i = 2; i < N; ++ i ) {
            f[i] = f[i - 1] * i % MOD;
            g[i] = MOD - (LL)MOD / i * g[MOD % i] % MOD;
            v[i] = v[i - 1] * g[i] % MOD;
        }
        return C(k, (d + k) / 2);
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
