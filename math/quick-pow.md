## 代码实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

// OlogK
LL qmi(int a, int b, int p) {
    LL res = 1 % p;
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
        int a, b, p;
        cin >> a >> b >> p;
        cout << qmi(a, b, p) << endl;
    }
    return 0;
}
```

###### **Python**

```python
# Python Version
def fastPow(a, b, p):
    res = 1 % p
    while b:
        if b & 1:
            res = (res * a) % p
        b >>= 1
        a = (a * a) % p
    return res


if __name__ == "__main__":
    n = int(input())
    for _ in range(n):
        a, b, p = map(int, input().split())
        print(fastPow(a, b, p))

```

<!-- tabs:end -->
</details>

<br>

模板：[Luogu P1226](https://www.luogu.com.cn/problem/P1226)

## 应用

### 模意义下取幂

> [!NOTE] **问题描述**
> 
> 计算 $x^n\bmod m$。

这是一个非常常见的应用，例如它可以用于计算模意义下的乘法逆元。

既然我们知道取模的运算不会干涉乘法运算，因此我们只需要在计算的过程中取模即可。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1) res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}
```

###### **Python**

```python
# Python Version
def binpow(a, b, m):
    a = a % m
    res = 1
    while b > 0:
        if (b & 1):
            res = res * a % m
        a = a * a % m
        b >>= 1
    return res
```

<!-- tabs:end -->
</details>

<br>



注意：根据费马小定理，如果 $m$ 是一个质数，我们可以计算 $x^{n\bmod (m-1)}$ 来加速算法过程。

### 计算斐波那契数

> [!NOTE] **问题描述**
> 
> 计算斐波那契数列第 $n$ 项 $F_n$。

根据斐波那契数列的递推式 $F_n = F_{n-1} + F_{n-2}$，我们可以构建一个 $2\times 2$ 的矩阵来表示从 $F_i,F_{i+1}$ 到 $F_{i+1},F_{i+2}$ 的变换。于是在计算这个矩阵的 $n$ 次幂的时侯，我们使用快速幂的思想，可以在 $\Theta(\log n)$ 的时间内计算出结果。对于更多的细节参见 [斐波那契数列](math/fibonacci.md)。

### 多次置换

> [!NOTE] **问题描述**
> 
> 给你一个长度为 $n$ 的序列和一个置换，把这个序列置换 $k$ 次。

简单地把这个置换取 $k$ 次幂，然后把它应用到序列 $n$ 上即可。时间复杂度是 $O(n \log k)$ 的。

注意：给这个置换建图，然后在每一个环上分别做 $k$ 次幂（事实上做一下 $k$ 对环长取模的运算即可）可以取得更高效的算法，达到 $O(n)$ 的复杂度。

### 加速几何中对点集的操作

> 三维空间中，$n$ 个点 $p_i$，要求将 $m$ 个操作都应用于这些点。包含 3 种操作：
>
> 1. 沿某个向量移动点的位置（Shift）。
> 2. 按比例缩放这个点的坐标（Scale）。
> 3. 绕某个坐标轴旋转（Rotate）。
>
> 还有一个特殊的操作，就是将一个操作序列重复 $k$ 次（Loop），这个序列中也可能有 Loop 操作（Loop 操作可以嵌套）。现在要求你在低于 $O(n \cdot \textit{length})$ 的时间内将这些变换应用到这个 $n$ 个点，其中 $\textit{length}$ 表示把所有的 Loop 操作展开后的操作序列的长度。

让我们来观察一下这三种操作对坐标的影响：

1. Shift 操作：将每一维的坐标分别加上一个常量；
2. Scale 操作：把每一维坐标分别乘上一个常量；
3. Rotate 操作：这个有点复杂，我们不打算深入探究，不过我们仍然可以使用一个线性组合来表示新的坐标。

可以看到，每一个变换可以被表示为对坐标的线性运算，因此，一个变换可以用一个 $4\times 4$ 的矩阵来表示：

$$
\begin{bmatrix}
a_{11} & a_ {12} & a_ {13} & a_ {14} \\
a_{21} & a_ {22} & a_ {23} & a_ {24} \\
a_{31} & a_ {32} & a_ {33} & a_ {34} \\
a_{41} & a_ {42} & a_ {43} & a_ {44} \\
\end{bmatrix}
$$

使用这个矩阵就可以将一个坐标（向量）进行变换，得到新的坐标（向量）：

$$
\begin{bmatrix}
a_{11} & a_ {12} & a_ {13} & a_ {14} \\
a_{21} & a_ {22} & a_ {23} & a_ {24} \\
a_{31} & a_ {32} & a_ {33} & a_ {34} \\
a_{41} & a_ {42} & a_ {43} & a_ {44} \\
\end{bmatrix}\cdot
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
 = \begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix}
$$

你可能会问，为什么一个三维坐标会多一个 1 出来？原因在于，如果没有这个多出来的 1，我们没法使用矩阵的线性变换来描述 Shift 操作。

接下来举一些简单的例子来说明我们的思路：

1.  Shift 操作：让 $x$ 坐标方向的位移为 $5$，$y$ 坐标的位移为 $7$，$z$ 坐标的位移为 $9$：

    $$
    \begin{bmatrix}
    1 & 0 & 0 & 5 \\
    0 & 1 & 0 & 7 \\
    0 & 0 & 1 & 9 \\
    0 & 0 & 0 & 1 \\
    \end{bmatrix}
    $$

2.  Scale 操作：把 $x$ 坐标拉伸 10 倍，$y,z$ 坐标拉伸 5 倍：

    $$
    \begin{bmatrix}
    10 & 0 & 0 & 0 \\
    0 & 5 & 0 & 0 \\
    0 & 0 & 5 & 0 \\
    0 & 0 & 0 & 1 \\
    \end{bmatrix}
    $$

3.  Rotate 操作：绕 $x$ 轴旋转 $\theta$ 弧度，遵循右手定则（逆时针方向）

    $$
    \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & \cos \theta & \sin \theta & 0 \\
    0 & -\sin \theta & \cos \theta & 0 \\
    0 & 0 & 0 & 1 \\
    \end{bmatrix}
    $$

现在，每一种操作都被表示为了一个矩阵，变换序列可以用矩阵的乘积来表示，而一个 Loop 操作相当于取一个矩阵的 k 次幂。这样可以用 $O(m \log k)$ 计算出整个变换序列最终形成的矩阵。最后将它应用到 $n$ 个点上，总复杂度 $O(n + m \log k)$。

### 定长路径计数

> [!NOTE] **问题描述**
> 
> 给一个有向图（边权为 1），求任意两点 $u,v$ 间从 $u$ 到 $v$，长度为 $k$ 的路径的条数。

我们把该图的邻接矩阵 M 取 k 次幂，那么 $M_{i,j}$ 就表示从 $i$ 到 $j$ 长度为 $k$ 的路径的数目。该算法的复杂度是 $O(n^3 \log k)$。有关该算法的细节请参见 [矩阵](./matrix.md) 页面。

### 模意义下大整数乘法

> 计算 $a\times b\bmod m,\,\,a,b\le m\le 10^{18}$。

与二进制取幂的思想一样，这次我们将其中的一个乘数表示为若干个 2 的整数次幂的和的形式。因为在对一个数做乘 2 并取模的运算的时侯，我们可以转化为加减操作防止溢出。这样仍可以在 $O (\log_2 m)$ 的时内解决问题。递归方法如下：

$$
a \cdot b = \begin{cases}
0 &\text{if }a = 0 \\\\
2 \cdot \frac{a}{2} \cdot b &\text{if }a > 0 \text{ and }a \text{ even} \\\\
2 \cdot \frac{a-1}{2} \cdot b + b &\text{if }a > 0 \text{ and }a \text{ odd}
\end{cases}
$$

#### 快速乘

但是 $O(\log_2 m)$ 的“龟速乘”还是太慢了，这在很多对常数要求比较高的算法比如 Miller_Rabin 和 Pollard-Rho 中，就显得不够用了。所以我们要介绍一种可以处理模数在 `long long` 范围内、不需要使用黑科技 `__int128` 的、复杂度为 $O(1)$ 的“快速乘”。

我们发现：

$$
a\times b\bmod m=a\times b-\left\lfloor \dfrac{ab}m \right\rfloor\times m
$$

我们巧妙运用 `unsigned long long` 的自然溢出：

$$
a\times b\bmod m=a\times b-\left\lfloor \dfrac{ab}m \right\rfloor\times m=\left(a\times b-\left\lfloor \dfrac{ab}m \right\rfloor\times m\right)\bmod 2^{64}
$$

于是在算出 $\left\lfloor\dfrac{ab}m\right\rfloor$ 后，两边的乘法和中间的减法部分都可以使用 `unsigned long long` 直接计算，现在我们只需要解决如何计算 $\left\lfloor\dfrac {ab}m\right\rfloor$。

我们考虑先使用 `long double` 算出 $\dfrac am$ 再乘上 $b$。

既然使用了 `long double`，就无疑会有精度误差。极端情况就是第一个有效数字（二进制下）在小数点后一位。在 `x86-64` 机器下，`long double` 将被解释成 $80$ 位拓展小数（即符号为 $1$ 位，指数为 $15$ 位，尾数为 $64$ 位），所以 `long double` 最多能精确表示的有效位数为 $64$[^note1]。所以 $\dfrac am$ 最差从第 $65$ 位开始出错，误差范围为 $\left(-2^{-64},2^{64}\right)$。乘上 $b$ 这个 $64$ 位整数，误差范围为 $(-0.5,0.5)$，再加上 $0.5$ 误差范围为 $(0,1)$，取整后误差范围位 $\{0,1\}$。于是乘上 $-m$ 后，误差范围变成 $\{0,-m\}$，我们需要判断这两种情况。

因为 $m$ 在 `long long` 范围内，所以如果计算结果 $r$ 在 $[0,m)$ 时，直接返回 $r$，否则返回 $r+m$，当然你也可以直接返回 $(r+m)\bmod m$。

代码实现如下：

```cpp
long long binmul(long long a, long long b, long long m) {
    unsigned long long c =
        (unsigned long long)a * b -
        (unsigned long long)((long double)a / m * b + 0.5L) * m;
    if (c < m) return c;
    return c + m;
}
```

### 高精度快速幂

> [!NOTE] **前置技能**
> 
> [高精度](math/bignum.md)

> [!NOTE] **例题【NOIP2003 普及组改编·麦森数】（[原题在此](https://www.luogu.com.cn/problem/P1045)）**
> 
> 题目大意：从文件中输入 P（1000&lt;P&lt;3100000），计算 $2^P−1$ 的最后 100 位数字（用十进制高精度数表示），不足 100 位时高位补 0。

代码实现如下：


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

### 同一底数与同一模数的预处理快速幂

在同一底数与同一模数的条件下，可以利用分块思想，用一定的时间（一般是 $O(\sqrt n)$）预处理后用 $O(1)$ 的时间回答一次幂询问。

算法的具体步骤是：

1. 选定一个数 $s$，预处理出 $a^0$ 到 $a^s$ 与 $a^{0\cdot s}$ 到 $a^{\lceil\frac ps\rceil\cdot s}$ 的值并存在一个（或两个）数组里；
2. 对于每一次询问 $a^b\bmod p$，将 $b$ 拆分成 $\left\lfloor\dfrac bs\right\rfloor\cdot s+b\bmod s$，则 $a^b=a^{\lfloor\frac bs\rfloor\cdot s}\times a^{b\bmod s}$，可以 $O(1)$ 求出答案。

关于这个数 $s$ 的选择，我们一般选择 $\sqrt p$ 或者一个大小适当的 $2$ 的次幂（选择 $\sqrt p$ 可以使预处理较优，选择 $2$ 的次幂可以使用位运算优化/简化计算）。

> [!NOTE] **参考代码**
```cpp
int pow1[65536], pow2[65536];
void preproc(int a, int mod) {
    pow1[0] = pow2[0] = 1;
    for (int i = 1; i < 65536; i++) pow1[i] = 1LL * pow1[i - 1] * a % mod;
    int pow65536 = 1LL * pow1[65535] * a % mod;
    for (int i = 1; i < 65536; i++)
        pow2[i] = 1LL * pow2[i - 1] * pow65536 % mod;
}
int query(int pows) { return 1LL * pow1[pows & 65535] * pow2[pows >> 16]; }
```

## 习题

- [UVa 1230 - MODEX](http://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&category=24&page=show_problem&problem=3671)
- [UVa 374 - Big Mod](http://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&category=24&page=show_problem&problem=310)
- [UVa 11029 - Leading and Trailing](https://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=1970)
- [Codeforces - Parking Lot](http://codeforces.com/problemset/problem/630/I)
- [SPOJ - The last digit](http://www.spoj.com/problems/LASTDIG/)
- [SPOJ - Locker](http://www.spoj.com/problems/LOCKER/)
- [LA - 3722 Jewel-eating Monsters](https://icpcarchive.ecs.baylor.edu/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=1723)
-   [SPOJ - Just add it](http://www.spoj.com/problems/ZSUM/)

## 习题

> [!NOTE] **[AcWing 876. 快速幂求逆元](https://www.acwing.com/problem/content/878/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO@binacs

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
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

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *