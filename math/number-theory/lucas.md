## Lucas 定理

Lucas 定理用于求解大组合数取模的问题，其中模数必须为素数。正常的组合数运算可以通过递推公式求解（详见 [排列组合](combinatorics/combination.md)），但当问题规模很大，而模数是一个不大的质数的时候，就不能简单地通过递推求解来得到答案，需要用到 Lucas 定理。

### 求解方式

Lucas 定理内容如下：对于质数 $p$，有

$$
\binom{n}{m}\bmod p = \binom{\left\lfloor n/p \right\rfloor}{\left\lfloor m/p\right\rfloor}\cdot\binom{n\bmod p}{m\bmod p}\bmod p
$$

观察上述表达式，可知 $n\bmod p$ 和 $m\bmod p$ 一定是小于 $p$ 的数，可以直接求解，$\displaystyle\binom{\left\lfloor n/p \right\rfloor}{\left\lfloor m/p\right\rfloor}$ 可以继续用 Lucas 定理求解。这也就要求 $p$ 的范围不能够太大，一般在 $10^5$ 左右。边界条件：当 $m=0$ 的时候，返回 $1$。

时间复杂度为 $O(f(p) + g(n)\log n)$，其中 $f(n)$ 为预处理组合数的复杂度，$g(n)$ 为单次求组合数的复杂度。



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
long long Lucas(long long n, long long m, long long p) {
    if (m == 0) return 1;
    return (C(n % p, m % p, p) * Lucas(n / p, m / p, p)) % p;
}
```

###### **Python**
    
```python
# Python Version
def Lucas(n, m, p):
    if m == 0:
        return 1
    return (C(n % p, m % p, p) * Lucas(n // p, m // p, p)) % p
```

<!-- tabs:end -->
</details>

<br>

### Lucas 定理的证明

略

## 素数在阶乘中的幂次

Legengre 在 1808 年指出 $n!$ 中含有的素数 $p$ 的幂次为 $\sum_{j\geq 1}\lfloor n/p^j\rfloor$。

证明：将 $n!$ 记为 $1\times 2\times \cdots \times p\times \cdots \times 2p\times \cdots \times \lfloor n/p\rfloor p\times \cdots \times n$ 那么其中 $p$ 的倍数有 $p\times 2p\times \cdots \times \lfloor n/p\rfloor p=p^{\lfloor n/p\rfloor }\lfloor n/p\rfloor !$ 然后在 $\lfloor n/p\rfloor !$ 中继续寻找 $p$ 的倍数即可，这是一个递归的过程。为了方便记 $\nu(n!)=\sum_{j\geq 1}\lfloor n/p^j\rfloor$。

另一种其他地方比较常见的公式，用到了 p 进制下各位数字和：

$v_p(n!)=\frac{n-S_p(n)}{p-1}$

与等比数列求和公式很相似。由于涉及各位数字和，利用数学归纳法可以轻松证明。

特别地，阶乘中 2 的幂次是：

$v_2(n!)=n-S_2(n)$

### 素数在组合数中的幂次

组合数对一个数取模的结果，往往构成分形结构，例如谢尔宾斯基三角形就可以通过组合数模 2 得到。

$v_p(C_m^n)=\frac{S_p(n)+S_p(m-n)-S_p(m)}{p-1}$

如果仔细分析，p 是否整除组合数其实和上下标在 p 进制下减法是否需要借位有关。这就有了 Kummer 定理。

**Kummer 定理：p 在组合数 $C_m^n$ 中的幂次，恰好是 p 进制下 m 减掉 n 需要借位的次数。**

特别地，组合数中 2 的幂次是：

$v_2(C_m^n)=S_2(n)+S_2(m-n)-S_2(m)$

## exLucas 定理

Lucas 定理中对于模数 $p$ 要求必须为素数，那么对于 $p$ 不是素数的情况，就需要用到 exLucas 定理。

### 求解思路

#### 第一部分：中国剩余定理

要求计算二项式系数 $\binom{n}{m}\bmod M$，其中 $M$ 可能为合数。

考虑利用 [中国剩余定理](math/number-theory/crt.md) 合并答案，这种情况下我们只需求出 $\binom{n}{m}\bmod p^q$ 的值即可（其中 $p$ 为素数且 $q$ 为正整数）。

根据 **唯一分解定理**，将 $p$ 质因数分解：

$$
p={q_1}^{\alpha_1}\cdot{q_2}^{\alpha_2}\cdots{q_r}^{\alpha_r}=\prod_{i=1}^{r}{q_i}^{\alpha_i}
$$

对于任意 $i,j$，有 ${q_i}^{\alpha_i}$ 与 ${q_j}^{\alpha_j}$ 互质，所以可以构造如下 $r$ 个同余方程：

$$
\left\{
\begin{aligned}
a_1\equiv \displaystyle\binom{n}{m}&\pmod {{q_1}^{\alpha_1}}\\
a_2\equiv \displaystyle\binom{n}{m}&\pmod {{q_2}^{\alpha_2}}\\
&\cdots\\
a_r\equiv \displaystyle\binom{n}{m}&\pmod {{q_r}^{\alpha_r}}\\
\end{aligned}
\right.
$$

我们发现，在求出 $a_i$ 后，就可以用中国剩余定理求解出 $\displaystyle\binom{n}{m}$。

#### 第二部分：移除分子分母中的素数

根据同余的定义，$\displaystyle a_i=\binom{n}{m}\bmod {q_i}^{\alpha_i}$，问题转化成，求 $\displaystyle \binom{n}{m} \bmod q^k$（$q$ 为质数）的值。

根据组合数定义 $\displaystyle \binom{n}{m} = \frac{n!}{m! (n-m)!}$，$\displaystyle \binom{n}{m} \bmod q^k = \frac{n!}{m! (n-m)!} \bmod q^k$。

由于式子是在模 $q^k$ 意义下，所以分母要算乘法逆元。

同余方程 $ax \equiv 1 \pmod p$（即乘法逆元）**有解** 的充要条件为 $\gcd(a,p)=1$（裴蜀定理），

然而 **无法保证有解**，发现无法直接求 $\operatorname{inv}_{m!}$ 和 $\operatorname{inv}_{(n-m)!}$，

所以将原式转化为：

$$
\frac{\frac{n!}{q^x}}{\frac{m!}{q^y}\frac{(n-m)!}{q^z}}q^{x-y-z} \bmod q^k
$$

$x$ 表示 $n!$ 中包含多少个 $q$ 因子，$y, z$ 同理。

#### 第三部分：Wilson 定理的推论

问题转化成，求形如：

$$
\frac{n!}{q^x}\bmod q^k
$$

的值。这时可以利用 [Wilson 定理的推论](math/number-theory/wilson.md)。如果难以理解，可以看看下面的解释。

#### 一个示例：22! mod 9

先考虑 $n! \bmod q^k$，

比如 $n=22, q=3, k=2$ 时：

$22!=1\times 2\times 3\times 4\times 5\times 6\times 7\times 8\times 9\times 10\times 11\times 12$

$\times 13\times 14\times 15\times 16\times 17\times 18\times 19\times20\times21\times22$

将其中所有 $q$ 的倍数提取，得到：

$22!=3^7 \times (1\times 2\times 3\times 4\times 5\times 6\times 7)$$\times(1\times 2\times 4\times 5\times 7\times 8\times 10 \times 11\times 13\times 14\times 16\times 17\times 19 \times 20 \times 22 )$

可以看到，式子分为三个整式的乘积：

1. 是 $3$ 的幂，次数是 $\lfloor\frac{n}{q}\rfloor$；

2. 是 $7!$，即 $\lfloor\frac{n}{q}\rfloor!$，由于阶乘中仍然可能有 $q$ 的倍数，考虑递归求解；

3.  是 $n!$ 中与 $q$ 互质的部分的乘积，具有如下性质：  
    $1\times 2\times 4\times 5\times 7\times 8\equiv10 \times 11\times 13\times 14\times 16\times 17 \pmod{3^2}$，  
    即：$\displaystyle \prod_{i,(i,q)=1}^{q^k}i\equiv\prod_{i,(i,q)=1}^{q^k}(i+tq^k) \pmod{q^k}$（$t$ 是任意正整数）。  
    $\displaystyle \prod_{i,(i,q)=1}^{q^k}i$ 一共循环了 $\displaystyle \lfloor\frac{n}{q^k}\rfloor$ 次，暴力求出 $\displaystyle \prod_{i,(i,q)=1}^{q^k}i$，然后用快速幂求 $\displaystyle \lfloor\frac{n}{q^k}\rfloor$ 次幂。  
    最后要乘上 $\displaystyle \prod_{i,(i,q)=1}^{n \bmod q^k}i$，即 $19\times 20\times 22$，显然长度小于 $q^k$，暴力乘上去。

..........

下面这种写法，拥有单次询问 $O(plogp)$ 的时间复杂度。其中 `int inverse(int x)` 函数返回 $x$ 在模 $p$ 意义下的逆元。


```cpp
LL calc(LL n, LL x, LL P) {
    if (!n) return 1;
    LL s = 1;
    for (LL i = 1; i <= P; i++)
        if (i % x) s = s * i % P;
    s = Pow(s, n / P, P);
    for (LL i = n / P * P + 1; i <= n; i++)
        if (i % x) s = i % P * s % P;
    return s * calc(n / x, x, P) % P;
}
LL multilucas(LL m, LL n, LL x, LL P) {
    int cnt = 0;
    for (LL i = m; i; i /= x) cnt += i / x;
    for (LL i = n; i; i /= x) cnt -= i / x;
    for (LL i = m - n; i; i /= x) cnt -= i / x;
    return Pow(x, cnt, P) % P * calc(m, x, P) % P * inverse(calc(n, x, P), P) %
           P * inverse(calc(m - n, x, P), P) % P;
}
LL exlucas(LL m, LL n, LL P) {
    int cnt = 0;
    LL p[20], a[20];
    for (LL i = 2; i * i <= P; i++) {
        if (P % i == 0) {
            p[++cnt] = 1;
            while (P % i == 0) p[cnt] = p[cnt] * i, P /= i;
            a[cnt] = multilucas(m, n, i, p[cnt]);
        }
    }
    if (P > 1) p[++cnt] = P, a[cnt] = multilucas(m, n, P, P);
    return CRT(cnt, a, p);
}
```

若不考虑 excrt 的复杂度，通过预处理 $\frac{n!}{n以内的p的所有倍数的乘积}\bmod{p}$，可以使时间复杂度优化至单次 $O(p + \log p)$。而如果 p 是固定的，我们在一开始就可以对 p 进行分解，并进行预处理，可以达到总复杂度 $O(p + T\log p)$。

## 习题

- [Luogu3807【模板】卢卡斯定理](https://www.luogu.com.cn/problem/P3807)
- [SDOI2010 古代猪文  卢卡斯定理](https://loj.ac/problem/10229)
- [Luogu4720【模板】扩展卢卡斯](https://www.luogu.com.cn/problem/P4720)
- [Ceizenpok’s formula](http://codeforces.com/gym/100633/problem/J)
