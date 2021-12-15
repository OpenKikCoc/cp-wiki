## 欧拉函数的定义

欧拉函数（Euler's totient function），即 $\varphi(n)$，表示的是小于等于 $n$ 和 $n$ 互质的数的个数。

比如说 $\varphi(1) = 1$。

当 n 是质数的时候，显然有 $\varphi(n) = n - 1$。

## 欧拉函数的一些性质

-   欧拉函数是积性函数。

    积性是什么意思呢？如果有 $\gcd(a, b) = 1$，那么 $\varphi(a \times b) = \varphi(a) \times \varphi(b)$。

    特别地，当 $n$ 是奇数时 $\varphi(2n) = \varphi(n)$。

-   $n = \sum_{d \mid n}{\varphi(d)}$。

    利用 [莫比乌斯反演](math/mobius.md) 相关知识可以得出。

    也可以这样考虑：如果 $\gcd(k, n) = d$，那么 $\gcd(\dfrac{k}{d},\dfrac{n}{d}) = 1, （ k < n ）$。

    如果我们设 $f(x)$ 表示 $\gcd(k, n) = x$ 的数的个数，那么 $n = \sum_{i = 1}^n{f(i)}$。

    根据上面的证明，我们发现，$f(x) = \varphi(\dfrac{n}{x})$，从而 $n = \sum_{d \mid n}\varphi(\dfrac{n}{d})$。注意到约数 $d$ 和 $\dfrac{n}{d}$ 具有对称性，所以上式化为 $n = \sum_{d \mid n}\varphi(d)$。

-   若 $n = p^k$，其中 $p$ 是质数，那么 $\varphi(n) = p^k - p^{k - 1}$。
    （根据定义可知）


-   由唯一分解定理，设 $n = \prod_{i=1}^{s}p_i^{k_i}$，其中 $p_i$ 是质数，有 $\varphi(n) = n \times \prod_{i = 1}^s{\dfrac{p_i - 1}{p_i}}$。

    证明：

    -   引理：设 $p$ 为任意质数，那么 $\varphi(p^k)=p^{k-1}\times(p-1)$。

        证明：显然对于从 1 到 $p^k$ 的所有数中，除了 $p^{k-1}$ 个 $p$ 的倍数以外其它数都与 $p^k$ 互素，故 $\varphi(p^k)=p^k-p^{k-1}=p^{k-1}\times(p-1)$，证毕。

    接下来我们证明 $\varphi(n) = n \times \prod_{i = 1}^s{\dfrac{p_i - 1}{p_i}}$。由唯一分解定理与 $\varphi(x)$ 函数的积性

$$
\begin{aligned}
	\varphi(n) &= \prod_{i=1}^{s} \varphi(p_i^{k_i}) \\
	&= \prod_{i=1}^{s} (p_i-1)\times {p_i}^{k_i-1}\\
	&=\prod_{i=1}^{s} {p_i}^{k_i} \times(1 - \frac{1}{p_i})\\
	&=n~ \prod_{i=1}^{s} (1- \frac{1}{p_i})
	&\square
\end{aligned}
$$

## 如何求欧拉函数值

如果只要求一个数的欧拉函数值，那么直接根据定义质因数分解的同时求就好了。这个过程可以用 [Pollard Rho](./pollard-rho.md) 算法优化。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int euler_phi(int n) {
    int m = int(sqrt(n + 0.5));
    int ans = n;
    for (int i = 2; i <= m; i++)
        if (n % i == 0) {
            ans = ans / i * (i - 1);
            while (n % i == 0) n /= i;
        }
    if (n > 1) ans = ans / n * (n - 1);
    return ans;
}
```

###### **Python**

```python
# Python Version
def euler_phi(n):
    m = int(sqrt(n + 0.5))
    ans = n
    for i in range(2, m + 1):
        if n % i == 0:
            ans = ans // i * (i - 1)
            while n % i == 0:
                n = n // i
    if n > 1:
        ans = ans // n * (n - 1)
    return ans
```

<!-- tabs:end -->
</details>

<br>

注：如果将上面的程序改成如下形式，会提升一点效率：

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int euler_phi(int n) {
    int ans = n;
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0) {
            ans = ans / i * (i - 1);
            while (n % i == 0) n /= i;
        }
    if (n > 1) ans = ans / n * (n - 1);
    return ans;
}
```

###### **Python**

```python
# Python Version
def euler_phi(n):
    ans = n
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            ans = ans // i * (i - 1)
            while n % i == 0:
                n = n / i
    if n > 1:
        ans = ans // n * (n - 1)
    return ans
```

<!-- tabs:end -->
</details>

<br>

如果是多个数的欧拉函数值，可以利用后面会提到的线性筛法来求得。
详见：[筛法求欧拉函数](math/sieve.md#_8)

## 欧拉定理

与欧拉函数紧密相关的一个定理就是欧拉定理。其描述如下：

若 $\gcd(a, m) = 1$，则 $a^{\varphi(m)} \equiv 1 \pmod{m}$。

## 扩展欧拉定理

当然也有扩展欧拉定理

$$
a^b\equiv
\begin{cases}
a^{b\bmod\varphi(p)},\,&\gcd(a,\,p)=1\\
a^b,&\gcd(a,\,p)\ne1,\,b<\varphi(p)\\
a^{b\bmod\varphi(p)+\varphi(p)},&\gcd(a,\,p)\ne1,\,b\ge\varphi(p)
\end{cases}
\pmod p
$$

证明和 **习题** 详见 [欧拉定理](math/fermat.md)


## 习题

> [!NOTE] **[AcWing 873. 欧拉函数](https://www.acwing.com/problem/content/875/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1.  **质数 $i$ 的欧拉函数即为 $phi[i] = i - 1$** ：$[1,i−1]$ 均与 $i$ 互质，共 $i−1$ 个。
>
> 2.  **$phi[primes[j] * i]$ 分为两种情况**：
>
>     ① $ i \bmod primes[j] = 0 $ ：
>
>     $primes[j]$ 是 $i$ 的最小质因子，也是 $primes[j] * i$ 的最小质因子，因此 $1 - 1 / primes[j]$ 这一项在 $phi[i]$ 中计算过了，只需将基数 $N$ 修正为 $primes[j]$ 倍，最终结果为 $phi[i] * primes[j]$ 。
>
>     ② $ i \bmod primes[j] \neq 0 $ ：
>
>     $primes[j]$ 不是 $i$ 的质因子，只是 $primes[j] * i$ 的最小质因子，因此不仅需要将基数 $N$ 修正为 $primes[j]$ 倍，还需要补上 $1 - 1 / primes[j]$ 这一项，因此最终结果 $phi[i] * (primes[j] - 1)$ 。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 记得要先写除法，再写乘法； res = res / j * (j - 1)  避免溢出
#include<bits/stdc++.h>
using namespace std;

int phi(int x) {
    int res = x;
    for (int i = 2; i <= x / i; ++ i )
        if (x % i == 0) {
            // i 为其中一个质因子
            res = res / i * (i - 1);
            while (x % i == 0) x /= i;
        }
    if (x > 1) res = res / x * (x - 1);
    return res;
}

int main() {
    int n;
    cin >> n;
    while (n -- ) {
        int x;
        cin >> x;
        cout << phi(x) << endl;
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

> [!NOTE] **[AcWing 874. 筛法求欧拉函数](https://www.acwing.com/problem/content/876/)**
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

using LL = long long;

const int N = 1000010;

int primes[N], cnt;
int euler[N];
bool st[N];

void get_eulers(int n) {
    euler[1] = 1;
    for (int i = 2; i <= n; ++ i ) {
        if (!st[i]) primes[cnt ++ ] = i, euler[i] = i - 1;
        for (int j = 0; primes[j] <= n / i; ++ j ) {
            int t = primes[j] * i;
            st[t] = true;
            if (i % primes[j] == 0) {
                euler[t] = euler[i] * primes[j];
                break;
            }
            euler[t] = euler[i] * (primes[j] - 1);
        }
    }
}

int main() {
    int n;
    cin >> n;
    
    get_eulers(n);
    
    LL res = 0;
    for (int i = 1; i <= n; ++ i ) res += euler[i];
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