## 素数筛法

### 埃拉托斯特尼筛法

考虑这样一件事情：如果 $x$ 是合数，那么 $x$ 的倍数也一定是合数。利用这个结论，我们可以避免很多次不必要的检测。

如果我们从小到大考虑每个数，然后同时把当前这个数的所有（比自己大的）倍数记为合数，那么运行结束的时候没有被标记的数就是素数了。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int Eratosthenes(int n) {
    int p = 0;
    for (int i = 0; i <= n; ++i) is_prime[i] = 1;
    is_prime[0] = is_prime[1] = 0;
    for (int i = 2; i <= n; ++i) {
        if (is_prime[i]) {
            prime[p++] = i;  // prime[p]是i,后置自增运算代表当前素数数量
            if ((long long)i * i <= n)
                for (int j = i * i; j <= n; j += i)
                    // 因为从 2 到 i - 1 的倍数我们之前筛过了，这里直接从 i
                    // 的倍数开始，提高了运行速度
                    is_prime[j] = 0;  // 是i的倍数的均不是素数
        }
    }
    return p;
}
```

###### **Python**

```python
# Python Version
def Eratosthenes(n):
    p = 0
    for i in range(0, n + 1):
        is_prime[i] = True
    is_prime[0] = is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            prime[p] = i
            p = p + 1
            if i * i <= n:
                j = i * i
                while j <= n:
                    is_prime[j] = False
                    j = j + i
    return p
```

<!-- tabs:end -->
</details>

<br>

以上为 **Eratosthenes 筛法**（埃拉托斯特尼筛法，简称埃氏筛法），时间复杂度是 $O(n\log\log n)$。

现在我们就来看看推导过程：

如果每一次对数组的操作花费 1 个单位时间，则时间复杂度为：

$$
O\left(n\sum_{k=1}^{\pi(n)}{1\over p_k}\right)
$$

其中 $p_k$ 表示第 $k$ 小的素数。根据 Mertens 第二定理，存在常数 $B_1$ 使得：

$$
\sum_{k=1}^{\pi(n)}{1\over p_k}=\log\log n+B_1+O\left(1\over\log n\right)
$$

所以 **Eratosthenes 筛法** 的时间复杂度为 $O(n\log\log n)$。接下来我们证明 Mertens 第二定理的弱化版本 $\sum_{k\le\pi(n)}1/p_k=O(\log\log n)$：

根据 $\pi(n)=\Theta(n/\log n)$，可知第 $n$ 个素数的大小为 $\Theta(n\log n)$。于是就有

$$
\begin{aligned}
\sum_{k=1}^{\pi(n)}{1\over p_k}
&=O\left(\sum_{k=2}^{\pi(n)}{1\over k\log k}\right) \\
&=O\left(\int_2^{\pi(n)}{\mathrm dx\over x\log x}\right) \\
&=O(\log\log\pi(n))=O(\log\log n)
\end{aligned}
$$

当然，上面的做法效率仍然不够高效，应用下面几种方法可以稍微提高算法的执行效率。

#### 筛至平方根

显然，要找到直到 $n$ 为止的所有素数，仅对不超过 $\sqrt n$ 的素数进行筛选就足够了。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int n;
vector<char> is_prime(n + 1, true);
is_prime[0] = is_prime[1] = false;
for (int i = 2; i * i <= n; i++) {
    if (is_prime[i]) {
        for (int j = i * i; j <= n; j += i) is_prime[j] = false;
    }
}
```

###### **Python**

```python
# Python Version
is_prime = [True] * (n + 1)
is_prime[0] = is_prime[1] = False
for i in range(2, int(sqrt(n)) + 1):
    if is_prime[i]:
        j = i * i
        while j <= n:
            is_prime[j] = False
            j += i
```

<!-- tabs:end -->
</details>

<br>

这种优化不会影响渐进时间复杂度，实际上重复以上证明，我们将得到 $n \ln \ln \sqrt n + o(n)$，根据对数的性质，它们的渐进相同，但操作次数会明显减少。

#### 只筛奇数

因为除 2 以外的偶数都是合数，所以我们可以直接跳过它们，只用关心奇数就好。

首先，这样做能让我们内存需求减半；其次，所需的操作大约也减半。

#### 减少内存的占用

我们注意到筛法只需要 $n$ 比特的内存。因此我们可以通过将变量声明为布尔类型，只申请 $n$ 比特而不是 $n$ 字节的内存，来显著的减少内存占用。即仅占用 $\dfrac n 8$ 字节的内存。

但是，这种称为 **位级压缩** 的方法会使这些位的操作复杂化。任何位上的读写操作都需要多次算术运算，最终会使算法变慢。

因此，这种方法只有在 $n$ 特别大，以至于我们不能再分配内存时才合理。在这种情况下，我们将牺牲效率，通过显著降低算法速度以节省内存（减小 8 倍）。

值得一提的是，存在自动执行位级压缩的数据结构，如 C++ 中的 `vector<bool>` 和 `bitset<>`。

#### 分块筛选

由优化“筛至平方根”可知，不需要一直保留整个 `is_prime[1...n]` 数组。为了进行筛选，只保留到 $\sqrt n$ 的素数就足够了，即 `prime[1...sqrt(n)]`。并将整个范围分成块，每个块分别进行筛选。这样，我们就不必同时在内存中保留多个块，而且 CPU 可以更好地处理缓存。

设 $s$ 是一个常数，它决定了块的大小，那么我们就有了 $\lceil {\frac n s} \rceil$ 个块，而块 $k$($k = 0 ... \lfloor {\frac n s} \rfloor$) 包含了区间 $[ks; ks + s - 1]$ 中的数字。我们可以依次处理块，也就是说，对于每个块 $k$，我们将遍历所有质数（从 $1$ 到 $\sqrt n$）并使用它们进行筛选。

值得注意的是，我们在处理第一个数字时需要稍微修改一下策略：首先，应保留 $[1; \sqrt n]$ 中的所有的质数；第二，数字 $0$ 和 $1$ 应该标记为非素数。在处理最后一个块时，不应该忘记最后一个数字 $n$ 并不一定位于块的末尾。

以下实现使用块筛选来计算小于等于 $n$ 的质数数量。

```cpp
int count_primes(int n) {
    const int S = 10000;
    vector<int> primes;
    int nsqrt = sqrt(n);
    vector<char> is_prime(nsqrt + 1, true);
    for (int i = 2; i <= nsqrt; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
            for (int j = i * i; j <= nsqrt; j += i) is_prime[j] = false;
        }
    }
    int result = 0;
    vector<char> block(S);
    for (int k = 0; k * S <= n; k++) {
        fill(block.begin(), block.end(), true);
        int start = k * S;
        for (int p : primes) {
            int start_idx = (start + p - 1) / p;
            int j = max(start_idx, p) * p - start;
            for (; j < S; j += p) block[j] = false;
        }
        if (k == 0) block[0] = block[1] = false;
        for (int i = 0; i < S && start + i <= n; i++) {
            if (block[i]) result++;
        }
    }
    return result;
}
```

分块筛分的渐进时间复杂度与埃氏筛法是一样的（除非块非常小），但是所需的内存将缩小为 $O(\sqrt{n} + S)$，并且有更好的缓存结果。
另一方面，对于每一对块和区间 $[1, \sqrt{n}]$ 中的素数都要进行除法，而对于较小的块来说，这种情况要糟糕得多。
因此，在选择常数 $S$ 时要保持平衡。

块大小 $S$ 取 $10^4$ 到 $10^5$ 之间，可以获得最佳的速度。

### 线性筛法

埃氏筛法仍有优化空间，它会将一个合数重复多次标记。有没有什么办法省掉无意义的步骤呢？答案是肯定的。

如果能让每个合数都只被标记一次，那么时间复杂度就可以降到 $O(n)$ 了。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
void init() {
    phi[1] = 1;
    for (int i = 2; i < MAXN; ++i) {
        if (!vis[i]) {
            phi[i] = i - 1;
            pri[cnt++] = i;
        }
        for (int j = 0; j < cnt; ++j) {
            if (1ll * i * pri[j] >= MAXN) break;
            vis[i * pri[j]] = 1;
            if (i % pri[j]) {
                phi[i * pri[j]] = phi[i] * (pri[j] - 1);
            } else {
                // i % pri[j] == 0
                // 换言之，i 之前被 pri[j] 筛过了
                // 由于 pri 里面质数是从小到大的，所以 i
                // 乘上其他的质数的结果一定也是 pri[j] 的倍数
                // 它们都被筛过了，就不需要再筛了，所以这里直接 break 掉就好了
                phi[i * pri[j]] = phi[i] * pri[j];
                break;
            }
        }
    }
}
```

###### **Python**

```python
# Python Version
def init():
    phi[1] = 1
    for i in range(2, MAXN):
        if vis[i] == False:
             phi[i] = i - 1
             pri[cnt] = i
             cnt = cnt + 1
    for j in range(0, cnt):
        if i * pri[j] >= MAXN:
            break
        vis[i * pri[j]] = 1
        if i % pri[j]:
            phi[i * pri[j]] = phi[i] * (pri[j] - 1)
        else:
            **"**
            i % pri[j] == 0
            换言之，i 之前被 pri[j] 筛过了
            由于 pri 里面质数是从小到大的，所以 i 乘上其他的质数的结果一定也是
            pri[j] 的倍数 它们都被筛过了，就不需要再筛了，所以这里直接 break
            掉就好了
            **"**
            phi[i * pri[j]] = phi[i] * pri[j]
            break
```

<!-- tabs:end -->
</details>

<br>

上面代码中的 $phi$ 数组，会在下面提到。

上面的这种 **线性筛法** 也称为 **Euler 筛法**（欧拉筛法）。

> [!NOTE]
> 
> 注意到筛法求素数的同时也得到了每个数的最小质因子

## 筛法求欧拉函数

注意到在线性筛中，每一个合数都是被最小的质因子筛掉。比如设 $p_1$ 是 $n$ 的最小质因子，$n' = \frac{n}{p_1}$，那么线性筛的过程中 $n$ 通过 $n' \times p_1$ 筛掉。

观察线性筛的过程，我们还需要处理两个部分，下面对 $n' \bmod p_1$ 分情况讨论。

如果 $n' \bmod p_1 = 0$，那么 $n'$ 包含了 $n$ 的所有质因子。

$$
\begin{aligned}
\varphi(n) & = n \times \prod_{i = 1}^s{\frac{p_i - 1}{p_i}} \\\\
& = p_1 \times n' \times \prod_{i = 1}^s{\frac{p_i - 1}{p_i}} \\\\
& = p_1 \times \varphi(n')
\end{aligned}
$$

那如果 $n' \bmod p_1 \neq 0$ 呢，这时 $n'$ 和 $p_1$ 是互质的，根据欧拉函数性质，我们有：

$$
\begin{aligned}
\varphi(n) & = \varphi(p_1) \times \varphi(n') \\\\
& = (p_1 - 1) \times \varphi(n')
\end{aligned}
$$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
void pre() {
    memset(is_prime, 1, sizeof(is_prime));
    int cnt = 0;
    is_prime[1] = 0;
    phi[1] = 1;
    for (int i = 2; i <= 5000000; i++) {
        if (is_prime[i]) {
            prime[++cnt] = i;
            phi[i] = i - 1;
        }
        for (int j = 1; j <= cnt && i * prime[j] <= 5000000; j++) {
            is_prime[i * prime[j]] = 0;
            if (i % prime[j])
                phi[i * prime[j]] = phi[i] * phi[prime[j]];
            else {
                phi[i * prime[j]] = phi[i] * prime[j];
                break;
            }
        }
    }
}
```

###### **Python**

```python
# Python Version
def pre():
    cnt = 0
    is_prime[1] = False
    phi[1] = 1
    for i in range(2, 5000001):
        if is_prime[i]:
            prime[cnt] = i
            cnt = cnt + 1
            phi[i] = i - 1
        j = 1
        while j <= cnt and i * prime[j] <= 5000000:
            is_prime[i * prime[j]] = 0
            if i % prime[j]:
                phi[i * prime[j]] = phi[i] * phi[prime[j]]
            else:
                phi[i * prime[j]] = phi[i] * prime[j]
                break
            j = j + 1
```

<!-- tabs:end -->
</details>

<br>





## 筛法求莫比乌斯函数

### 线性筛

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
void pre() {
    mu[1] = 1;
    for (int i = 2; i <= 1e7; ++i) {
        if (!v[i]) mu[i] = -1, p[++tot] = i;
        for (int j = 1; j <= tot && i <= 1e7 / p[j]; ++j) {
            v[i * p[j]] = 1;
            if (i % p[j] == 0) {
                mu[i * p[j]] = 0;
                break;
            }
            mu[i * p[j]] = -mu[i];
        }
    }
```

###### **Python**

```python
# Python Version
def pre():
    mu[1] = 1
    for i in range(2, int(1e7 + 1)):
        if v[i] == 0:
            mu[i] = -1
            p[tot] = i
            tot = tot + 1
        j = 1
        while j <= tot and i <= 1e7 // p[j]:
            v[i * p[j]] = 1
            if i % p[j] == 0:
                mu[i * p[j]] = 0
                break
            j = j + 1
        mu[i * p[j]] = -mu[i]
```

<!-- tabs:end -->
</details>

<br>



## 筛法求约数个数

用 $d_i$ 表示 $i$ 的约数个数，$num_i$ 表示 $i$ 的最小质因子出现次数。

### 约数个数定理

定理：若 $n=\prod_{i=1}^mp_i^{c_i}$ 则 $d_i=\prod_{i=1}^mc_i+1$.

证明：我们知道 $p_i^{c_i}$ 的约数有 $p_i^0,p_i^1,\dots ,p_i^{c_i}$ 共 $c_i+1$ 个，根据乘法原理，$n$ 的约数个数就是 $\prod_{i=1}^mc_i+1$.

### 实现

因为 $d_i$ 是积性函数，所以可以使用线性筛。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
void pre() {
    d[1] = 1;
    for (int i = 2; i <= n; ++i) {
        if (!v[i]) v[i] = 1, p[++tot] = i, d[i] = 2, num[i] = 1;
        for (int j = 1; j <= tot && i <= n / p[j]; ++j) {
            v[p[j] * i] = 1;
            if (i % p[j] == 0) {
                num[i * p[j]] = num[i] + 1;
                d[i * p[j]] = d[i] / num[i * p[j]] * (num[i * p[j]] + 1);
                break;
            } else {
                num[i * p[j]] = 1;
                d[i * p[j]] = d[i] * 2;
            }
        }
    }
}
```

###### **Python**

```python
# Python Version
def pre():
    d[1] = 1
    for i in range(2, n + 1):
        if v[i] == 0:
            v[i] = 1; p[tot] = i; tot = tot + 1; d[i] = 2; num[i] = 1
        j = 1
        while j <= tot and i <= n // p[j]:
            v[p[j] * i] = 1
            if i % p[j] == 0:
                num[i * p[j]] = num[i] + 1
                d[i * p[j]] = d[i] // num[i * p[j]] * (num[i * p[j]] + 1)
                break
            else:
                num[i * p[j]] = 1
                d[i * p[j]] = d[i] * 2
            j = j + 1
```

<!-- tabs:end -->
</details>

<br>



## 筛法求约数和

$f_i$ 表示 $i$ 的约数和，$g_i$ 表示 $i$ 的最小质因子的 $p+p^1+p^2+\dots p^k$.

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
void pre() {
    g[1] = f[1] = 1;
    for (int i = 2; i <= n; ++i) {
        if (!v[i]) v[i] = 1, p[++tot] = i, g[i] = i + 1, f[i] = i + 1;
        for (int j = 1; j <= tot && i <= n / p[j]; ++j) {
            v[p[j] * i] = 1;
            if (i % p[j] == 0) {
                g[i * p[j]] = g[i] * p[j] + 1;
                f[i * p[j]] = f[i] / g[i] * g[i * p[j]];
                break;
            } else {
                f[i * p[j]] = f[i] * f[p[j]];
                g[i * p[j]] = 1 + p[j];
            }
        }
    }
    for (int i = 1; i <= n; ++i) f[i] = (f[i - 1] + f[i]) % Mod;
}
```

###### **Python**

```python
# Python Version
def pre():
    g[1] = f[1] = 1
    for i in range(2, n + 1):
        if v[i] == 0:
            v[i] = 1; p[tot] = i; tot = tot + 1; g[i] = i + 1; f[i] = i + 1;
        j = 1
        while j <= tot and i <= n // p[j]:
            v[p[j] * i] = 1
            if i % p[j] == 0:
                g[i * p[j]] = g[i] * p[j] + 1
                f[i * p[j]] = f[i] // g[i] * g[i * p[j]]
                break
            else:
                f[i * p[j]] = f[i] * f[p[j]]
                g[i * p[j]] = 1 + p[j]
    for i in range(1, n + 1):
        f[i] = (f[i - 1] + f[i]) % Mod
```

<!-- tabs:end -->
</details>

<br>

## 其他线性函数

**本节部分内容译自博文 [Решето Эратосфена](http://e-maxx.ru/algo/eratosthenes_sieve) 与其英文翻译版 [Sieve of Eratosthenes](https://cp-algorithms.com/algebra/sieve-of-eratosthenes.html)。其中俄文版版权协议为 Public Domain + Leave a Link；英文版版权协议为 CC-BY-SA 4.0。**

## 习题

### 一般筛法

> [!NOTE] **[LeetCode 1390. 四因数](https://leetcode.cn/problems/four-divisors/)**
> 
> 题意: 
> 
> 整数数组 nums，返回该数组中恰有四个因数的这些整数的各因数之和。

> [!TIP] **思路**
> 
> 用预处理+筛

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int sumFourDivisors(vector<int>& nums) {
        // C 是数组 nums 元素的上限，C3 是 C 的立方根
        int C = 100000, C3 = 46;
        
        vector<int> isprime(C + 1, 1);
        vector<int> primes;

        // 埃拉托斯特尼筛法
        for (int i = 2; i <= C; ++i) {
            if (isprime[i]) {
                primes.push_back(i);
            }
            for (int j = i + i; j <= C; j += i) {
                isprime[j] = 0;
            }
        }

        // 欧拉筛法
        /*
        for (int i = 2; i <= C; ++i) {
            if (isprime[i]) {
                primes.push_back(i);
            }
            for (int prime: primes) {
                if (i * prime > C) {
                    break;
                }
                isprime[i * prime] = 0;
                if (i % prime == 0) {
                    break;
                }
            }
        }
        */
        
        // 通过质数表构造出所有的四因数
        unordered_map<int, int> factor4;
        for (int prime: primes) {
            if (prime <= C3) {
                factor4[prime * prime * prime] = 1 + prime + prime * prime + prime * prime * prime;
            }
        }
        for (int i = 0; i < primes.size(); ++i) {
            for (int j = i + 1; j < primes.size(); ++j) {
                if (primes[i] <= C / primes[j]) {
                    factor4[primes[i] * primes[j]] = 1 + primes[i] + primes[j] + primes[i] * primes[j];
                }
                else {
                    break;
                }
            }
        }

        int ans = 0;
        for (int num: nums) {
            if (factor4.count(num)) {
                ans += factor4[num];
            }
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

> [!NOTE] **[Codeforces B. Prime Matrix](https://codeforces.com/problemset/problem/271/B)**
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
// Problem: B. Prime Matrix
// Contest: Codeforces - Codeforces Round #166 (Div. 2)
// URL: https://codeforces.com/problemset/problem/271/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

const int N = 510, M = 100010;

int n, m;
int g[N][N], r[N], c[N], rn[N], cn[N];

int primes[M], cnt;
bool st[M];

unordered_map<int, int> Hash;

void init() {
    // st[1] needed
    st[1] = true;
    for (int i = 2; i < M; ++i) {
        if (!st[i])
            primes[cnt++] = i;
        for (int j = 0; primes[j] <= (M - 1) / i; ++j) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0)
                break;
        }
    }
}

int get_dis(int x) {
    if (Hash.count(x))
        return Hash[x];
    int l = 0, r = cnt;
    while (l < r) {
        int m = l + r >> 1;
        if (x > primes[m])
            l = m + 1;
        else
            r = m;
    }
    return Hash[x] = primes[l] - x;
}

int main() {
    init();

    cin >> n >> m;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j) {
            cin >> g[i][j];
            int x = g[i][j], y;
            if (st[x]) {
                y = get_dis(x);
                rn[i] += y;
                cn[j] += y;
            }
        }

    int res = INT_MAX;
    for (int i = 1; i <= n; ++i)
        res = min(res, rn[i]);
    for (int i = 1; i <= m; ++i)
        res = min(res, cn[i]);
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

> [!NOTE] **[Codeforces C. Bear and Prime Numbers](https://codeforces.com/problemset/problem/385/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **非常好的数论题 素数筛**
> 
> **以及用埃式筛法 而非线形筛 思维**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Bear and Prime Numbers
// Contest: Codeforces - Codeforces Round #226 (Div. 2)
// URL: https://codeforces.com/problemset/problem/385/C
// Memory Limit: 512 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 非常好的数论题
// 考虑统计某数值出现多少次 在筛法中求该数值对和的贡献个数
// TLE 18
//     https://codeforces.com/contest/385/submission/111333415

const int N = 1e7 + 10;

int n, m;
int d[N], s[N];
bool st[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;
    for (int i = 1; i <= n; ++i) {
        int x;
        cin >> x;
        d[x]++;
    }

    // 埃氏筛法 同时求前缀和
    for (int i = 2; i < N; ++i) {
        s[i] = s[i - 1];
        if (st[i])
            continue;
        for (int j = 1; j * i < N; ++j) {
            s[i] += d[j * i];
            st[i * j] = true;
        }
    }

    cin >> m;
    while (m--) {
        int l, r;
        cin >> l >> r;
        l--;
        l = min(l, N - 2);
        r = min(r, N - 2);
        cout << s[r] - s[l] << endl;
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

> [!NOTE] **[Codeforces Soldier and Number Game](http://codeforces.com/problemset/problem/546/D)**
> 
> 题意: 
> 
> 求 $n$ 的质因子个数（分解之后的，非求不同质因子）

> [!TIP] **思路**
> 
> 线性筛应用 **积性性质**
> 
> 设 $f[x]$ 表示 $x$ 的质因子个数，易得 $f[xy]=f[x]+f[y]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Soldier and Number Game
// Contest: Codeforces - Codeforces Round #304 (Div. 2)
// URL: https://codeforces.com/problemset/problem/546/D
// Memory Limit: 256 MB
// Time Limit: 3000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 5e6 + 10;

int primes[N], cnt;
bool st[N];
LL f[N], s[N];

void init() {
    cnt = 0;
    memset(st, 0, sizeof st);
    memset(f, 0, sizeof f);  // f[i] 表示i的所有可分解质因子个数
    f[1] = 0;
    for (int i = 2; i < N; ++i) {
        if (!st[i])
            primes[cnt++] = i, f[i] = 1;
        for (int j = 0; primes[j] <= (N - 1) / i; ++j) {
            int t = primes[j] * i;
            st[t] = true;
            f[t] = f[i] + 1;  // 1 为 primes[j]
            if (i % primes[j] == 0) {
                break;
            }
        }
    }
    memset(s, 0, sizeof s);
    for (int i = 1; i < N; ++i)
        s[i] = s[i - 1] + f[i];
}

int main() {
    // ios::sync_with_stdio(false);
    // cin.tie(nullptr);
    // cout.tie(nullptr);

    init();

    int t;
    // cin >> t;
    scanf("%d", &t);
    while (t--) {
        int l, r;
        // cin >> r >> l;
        scanf("%d %d", &r, &l);
        // cout << s[r] - s[l] << endl;
        printf("%d\n", s[r] - s[l]);
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

> [!NOTE] **[AcWing 1292. 哥德巴赫猜想](https://www.acwing.com/problem/content/1294/)**
> 
> 题意: 
> 
> 哥德巴赫猜想的内容如下：
> 
> 任意一个大于 4 的偶数都可以拆成两个奇素数之和。
> 
> 验证所有小于一百万的偶数能否满足哥德巴赫猜想

> [!TIP] **思路**
> 
> 素数筛

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 1000010;

int primes[N], cnt;
bool st[N];

void init(int n) {
    for (int i = 2; i <= n; ++ i ) {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; ++ j ) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}

int main() {
    init(N - 1);
    
    int n;
    while (cin >> n, n) {
        for (int i = 1; ; ++ i ) {
            int a = primes[i];
            int b = n - a;
            if (!st[b]) {
                printf("%d = %d + %d\n", n, a, b);
                break;
            }
        }
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

> [!NOTE] **[AcWing 1293. 夏洛克和他的女朋友](https://www.acwing.com/problem/content/1295/)**
> 
> 题意: 
> 
> 给这些珠宝染色，使得一件珠宝的价格是另一件珠宝的价格的质因子时，两件珠宝的颜色不同

> [!TIP] **思路**
> 
> 所有质数染成一个色 1 即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 所有质数染成一个色1即可
#include<bits/stdc++.h>
using namespace std;

const int N = 100010;

int primes[N], cnt;
bool st[N];

void init(int n) {
    for (int i = 2; i <= n; ++ i ) {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; ++ j ) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}

int main() {
    int n;
    cin >> n;
    
    init(n + 1);
    
    if (n <= 2) cout << 1 << endl;
    else cout << 2 << endl;
    
    for (int i = 2; i <= n + 1; ++ i )
        if (!st[i]) cout << "1 ";
        else cout << "2 ";
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

> [!NOTE] **[LeetCode 2584. 分割数组使乘积互质](https://leetcode.cn/problems/split-the-array-to-make-coprime-products/) [TAG]**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然的 题意可以转化为求最左侧分界点使得其左右没有 `除了 1 之外的公共质因子`
> 
> 素数筛后对原数组分解即可
>
> 一种可行解为记录每个质因子出现的左右区间 按区间切即可
> 
> - 注意 1 的特判
> 
> - 注意对原数组分解素数时的优化 `x 为质数直接 break`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
 public:
     // 显然不能直接裸着乘，考虑 1e6 数字进行质因数分解 最多有 78499
     //               【ATTENTION】 但是 不同的质因数最多只有 
     using PII = pair<int, int>;
     const static int N = 1e6 + 10, M = 1e5;
     
     int primes[M], cnt;
     bool st[N];
     void init() {
         cnt = 0;
         memset(st, 0, sizeof st);
         for (int i = 2; i < N; ++ i ) {
             if (!st[i])
                 primes[cnt ++ ] = i;
             for (int j = 0; primes[j] <= (N - 1) / i; ++ j ) {
                 st[primes[j] * i] = true;
                 if (i % primes[j] == 0)
                     break;
             }
         }
     }
     
     int l[N], r[N];
     
     int findValidSplit(vector<int>& nums) {
         if (nums.size() <= 1)   // ATTENTION 特判
             return -1;
         
         init();
         
         // unordered_map<int, int> l, r;   // TLE
         memset(l, -1, sizeof l), memset(r, -1, sizeof r);
         int n = nums.size();
         for (int i = 0; i < n; ++ i ) {
             int x = nums[i];
             for (int j = 0; j < cnt && primes[j] <= x; ++ j ) {
                 {
                     // 优化
                     if (!st[x]) {
                         if (l[x] == -1)
                             l[x] = i;
                         r[x] = i;
                         break;
                     }
                 }
                 
                 
                 if (x % primes[j] == 0) {
                     int y = primes[j];
                     {
                         if (l[y] == -1)   // ATTENTION 要使用 primes[j] 而非 j
                             l[y] = i;
                         r[y] = i;
                     }
                     while (x % y == 0)
                         x /= y;
                 }
             }
             if (x > 1) { // 1 不统计
                 if (l[x] == -1)
                     l[x] = i;
                 r[x] = i;
             }
         }
         
         // 直接枚举位置显然会超时，考虑【区间分组】思想
         
         vector<PII> xs;
         for (int i = 0; i < N; ++ i )
             if (l[i] != -1)
                 xs.push_back({l[i], r[i]});
         sort(xs.begin(), xs.end()); // sort by l
         
         int st = -1, ed = -1;
         vector<PII> ys;
         for (auto & s : xs)
             if (ed < s.first) {
                 if (ed != -1)
                     ys.push_back({st, ed});
                 st = s.first, ed = s.second;
             } else
                 ed = max(ed, s.second);
         if (st != -1)
             ys.push_back({st, ed});
         
         if (ys.empty())         // 全是 1 的情况
             return 0;
         if (ys[0].first != 0)   // 前面一段是 1 的情况
             return 0;
         if (ys[0].second != n - 1)
             return ys[0].second;
         return -1;
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

### 区间筛

> [!NOTE] **[AcWing 196. 质数距离](https://www.acwing.com/problem/content/198/)** [TAG]
> 
> 题意: 
> 
> 在闭区间 `[L,U]` 内找到距离最接近的两个相邻质数 `C1` 和 `C2`

> [!TIP] **思路**
> 
> 区间范围给定的最大值是 `2^31 - 1` 
> 
> 1. 若一个数 `n` 是一个合数，其每一对因子中必然存在一个较小的因子且小于 `√ n` 的因子；
> 
> 2.  若 `x` 属于 `[L, R]` 且是合数，则一定存在 `P <= √ 2^31 - 1  (P <= 50000) ` 使得 `P` 能够整除 `X` ，其中 `P < x` 。
> 
> 故
> 
> 1. 先找出 50000 以内的所有质因子；
> 
> 2. 对于每个质数 `P` ，将 `[L, R]` 中所有 `P` 的倍数筛掉 (至少2倍)。
> 
>    找到大于等于 `L` 的最小的 `P` 的倍数 `P0` ，找下一个倍数时只需要 `+= P` 即可。
> 
> >  分数的向上取整： `[ L / P ]` 向上取整 `=`  `(l + p - 1) / p`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 1000010;  // lr差值

int primes[N], cnt;
bool st[N];

void init(int n) {
    memset(st, 0, sizeof st);
    cnt = 0;
    for (int i = 2; i <= n; ++ i ) {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; ++ j ) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}

int main() {
    int l, r;
    while (cin >> l >> r) {
        init(50000);
        
        memset(st, 0, sizeof st);
        for (int i = 0; i < cnt; ++ i ) {
            LL p = primes[i];
            for (LL j = max(p * 2, (l + p - 1) / p * p); j <= r; j += p)
                st[j - l] = true;
        }
        
        cnt = 0;
        for (int i = 0; i <= r - l; ++ i )
            if (!st[i] && i + l >= 2)
                primes[cnt ++ ] = i + l;
        
        if (cnt < 2) cout << "There are no adjacent primes." << endl;
        else {
            int minp = 0, maxp = 0;
            for (int i = 0; i + 1 < cnt; ++ i ) {
                int d = primes[i + 1] - primes[i];
                if (d < primes[minp + 1] - primes[minp]) minp = i;
                if (d > primes[maxp + 1] - primes[maxp]) maxp = i;
            }
            
            printf("%d,%d are closest, %d,%d are most distant.\n",
                primes[minp], primes[minp + 1],
                primes[maxp], primes[maxp + 1]);
        }
        
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