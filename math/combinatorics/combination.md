## 加法 & 乘法原理

### 加法原理

完成一个工程可以有 $n$ 类办法，$a_i(1 \le i \le n)$ 代表第 $i$ 类方法的数目。那么完成这件事共有 $S=a_1+a_2+\cdots +a_n$ 种不同的方法。

### 乘法原理

完成一个工程需要分 $n$ 个步骤，$a_i(1 \le i \le n)$ 代表第 $i$ 个步骤的不同方法数目。那么完成这件事共有 $S = a_1 \times a_2 \times \cdots \times a_n$ 种不同的方法。

## 排列与组合基础

### 排列数

从 $n$ 个不同元素中，任取 $m$（$m\leq n$，$m$ 与 $n$ 均为自然数，下同）个元素按照一定的顺序排成一列，叫做从 $n$ 个不同元素中取出 $m$ 个元素的一个排列；从 $n$ 个不同元素中取出 $m$($m\leq n$) 个元素的所有排列的个数，叫做从 $n$ 个不同元素中取出 $m$ 个元素的排列数，用符号 $\mathrm A_n^m$（或者是 $\mathrm P_n^m$）表示。

排列的计算公式如下：

$$
\mathrm A_n^m = n(n-1)(n-2) \cdots (n-m+1) = \frac{n!}{(n - m)!}
$$

$n!$ 代表 $n$ 的阶乘，即 $6! = 1 \times 2 \times 3 \times 4 \times 5 \times 6$。

公式可以这样理解：$n$ 个人选 $m$ 个来排队 ($m \le n$)。第一个位置可以选 $n$ 个，第二位置可以选 $n-1$ 个，以此类推，第 $m$ 个（最后一个）可以选 $n-m+1$ 个，得：

$$
\mathrm A_n^m = n(n-1)(n-2) \cdots (n-m+1) = \frac{n!}{(n - m)!}
$$

全排列：$n$ 个人全部来排队，队长为 $n$。第一个位置可以选 $n$ 个，第二位置可以选 $n-1$ 个，以此类推得：

$$
\mathrm A_n^n = n(n-1)(n-2) \cdots 3 × 2 × 1 = n!
$$

全排列是排列数的一个特殊情况。

### 组合数

从 $n$ 个不同元素中，任取 $m$($m\leq n$) 个元素组成一个集合，叫做从 $n$ 个不同元素中取出 $m$ 个元素的一个组合；从 $n$ 个不同元素中取出 $m$($m\leq n$) 个元素的所有组合的个数，叫做从 $n$ 个不同元素中取出 $m$ 个元素的组合数。用符号 $\mathrm C_n^m$ 来表示。

组合数计算公式

$$
\mathrm C_n^m = \frac{\mathrm A_n^m}{m!} = \frac{n!}{m!(n - m)!}
$$

如何理解上述公式？我们考虑 $n$ 个人 $m$($m \le n$) 个出来，不排队，不在乎顺序 $C_n^m$。如果在乎排列那么就是 $A_n^m$，如果不在乎那么就要除掉重复，那么重复了多少？同样选出的来的 $m$ 个人，他们还要“全排”得 $A_n^m$，所以得：

$$
\mathrm C_n^m \times m! = \mathrm A_n^m\\
\mathrm C_n^m = \frac{\mathrm A_n^m}{m!} = \frac{n!}{m!(n-m)!}
$$

组合数也常用 $\displaystyle \binom{n}{m}$ 表示，读作「$n$ 选 $m$」，即 $\displaystyle \mathrm C_n^m=\binom{n}{m}$。实际上，后者表意清晰明了，美观简洁，因此现在数学界普遍采用 $\displaystyle \binom{n}{m}$ 的记号而非 $\mathrm C_n^m$。

组合数也被称为「二项式系数」，下文二项式定理将会阐述其中的联系。

特别地，规定当 $m>n$ 时，$\mathrm A_n^m=\mathrm C_n^m=0$。

## 二项式定理

在进入排列组合进阶篇之前，我们先介绍一个与组合数密切相关的定理——二项式定理。

二项式定理阐明了一个展开式的系数：

$$
(a+b)^n=\sum_{i=0}^n\binom{n}{i}a^{n-i}b^i
$$

证明可以采用数学归纳法，利用 $\displaystyle \binom{n}{k}+\binom{n}{k-1}=\binom{n+1}{k}$ 做归纳。

二项式定理也可以很容易扩展为多项式的形式：

设 n 为正整数，$x_i$ 为实数，

$$
(x_1 + x_2 + \cdots + x_t)^n = \sum_{满足 n_1 + \cdots + n_t=n 的非负整数解} \binom{n}{n_1n_2\cdots n_t} x_1^{n_1}x_2^{n_2}\cdots x_t^{n_t}
$$

其中的 $\binom{n}{n_1,n_2,\cdots ,n_t}$ 是多项式系数，它的性质也很相似：

$$
\sum{\binom{n}{n_1n_2\cdots n_t}} = t^n
$$

## 排列与组合进阶篇

接下来我们介绍一些排列组合的变种。

### 多重集的排列数 | 多重组合数

请大家一定要区分 **多重组合数** 与 **多重集的组合数**！两者是完全不同的概念！

多重集是指包含重复元素的广义集合。设 $S=\{n_1\cdot a_1,n_2\cdot a_2,\cdots,n_k\cdot a_k,\}$ 表示由 $n_1$ 个 $a_1$，$n_2$ 个 $a_2$，…，$n_k$ 个 $a_k$ 组成的多重集，$S$ 的全排列个数为

$$
\frac{n!}{\prod_{i=1}^kn_i!}=\frac{n!}{n_1!n_2!\cdots n_k!}
$$

相当于把相同元素的排列数除掉了。具体地，你可以认为你有 $k$ 种不一样的球，每种球的个数分别是 $n_1,n_2,\cdots,n_k$，且 $n=n_1+n_2+\ldots+n_k$。这 $n$ 个球的全排列数就是 **多重集的排列数**。多重集的排列数常被称作 **多重组合数**。我们可以用多重组合数的符号表示上式：

$$
\binom{n}{n_1,n_2,\cdots,n_k}=\frac{n!}{\prod_{i=1}^kn_i!}
$$

可以看出，$\displaystyle \binom{n}{m}$ 等价于 $\displaystyle \binom{n}{m,n-m}$，只不过后者较为繁琐，因而不采用。

### 多重集的组合数 1

设 $S=\{n_1\cdot a_1,n_2\cdot a_2,\cdots,n_k\cdot a_k,\}$ 表示由 $n_1$ 个 $a_1$，$n_2$ 个 $a_2$，…，$n_k$ 个 $a_k$ 组成的多重集。那么对于整数 $r(r<n_i,\forall i\in[1,k])$，从 $S$ 中选择 $r$ 个元素组成一个多重集的方案数就是 **多重集的组合数**。这个问题等价于 $x_1+x_2+\cdots+x_k=r$ 的非负整数解的数目，可以用插板法解决，答案为

$$
\binom{r+k-1}{k-1}
$$

### 多重集的组合数 2

考虑这个问题：设 $S=\{n_1\cdot a_1,n_2\cdot a_2,\cdots,n_k\cdot a_k,\}$ 表示由 $n_1$ 个 $a_1$，$n_2$ 个 $a_2$，…，$n_k$ 个 $a_k$ 组成的多重集。那么对于正整数 $r$，从 $S$ 中选择 $r$ 个元素组成一个多重集的方案数。

这样就限制了每种元素的取的个数。同样的，我们可以把这个问题转化为带限制的线性方程求解：

$$
\forall i\in [1,k],\ x_i\le n_i,\ \sum_{i=1}^kx_i=r
$$

于是很自然地想到了容斥原理。容斥的模型如下：

1. 全集：$\displaystyle \sum_{i=1}^kx_i=r$ 的非负整数解。
2. 属性：$x_i\le n_i$。

于是设满足属性 $i$ 的集合是 $S_i$，$\overline{S_i}$ 表示不满足属性 $i$ 的集合，即满足 $x_i\ge n_i+1$ 的集合。那么答案即为

$$
\left|\bigcap_{i=1}^kS_i\right|=|U|-\left|\bigcup_{i=1}^k\overline{S_i}\right|
$$

根据容斥原理，有：

$$
\begin{aligned}
\left|\bigcup_{i=1}^k\overline{S_i}\right|
=&\sum_i\left|\overline{S_i}\right|
-\sum_{i,j}\left|\overline{S_i}\cap\overline{S_j}\right|
+\sum_{i,j,k}\left|\overline{S_i}\cap\overline{S_j}\cap\overline{S_k}\right|
-\cdots\\
&+(-1)^{k-1}\left|\bigcap_{i=1}^k\overline{S_i}\right|\\
=&\sum_i\binom{k+r-n_i-2}{k-1}
-\sum_{i,j}\binom{k+r-n_i-n_j-3}{k-1}+\sum_{i,j,k}\binom{k+r-n_i-n_j-n_k-4}{k-1}
-\cdots\\
&+(-1)^{k-1}\binom{k+r-\sum_{i=1}^kn_i-k-1}{k-1}
\end{aligned}
$$

拿全集 $\displaystyle |U|=\binom{k+r-1}{k-1}$ 减去上式，得到多重集的组合数

$$
Ans=\sum_{p=0}^k(-1)^p\sum_{A}\binom{k+r-1-\sum_{A} n_{A_i}-p}{k-1}
$$

其中 A 是充当枚举子集的作用，满足 $|A|=p,\ A_i<A_{i+1}$。

### 不相邻的排列

$1 \sim n$ 这 $n$ 个自然数中选 $k$ 个，这 $k$ 个数中任何两个数都不相邻的组合有 $\displaystyle \binom {n-k+1}{k}$ 种。

### 错位排列

我们把错位排列问题具体化，考虑这样一个问题：

$n$ 封不同的信，编号分别是 $1,2,3,4,5$，现在要把这五封信放在编号 $1,2,3,4,5$ 的信封中，要求信封的编号与信的编号不一样。问有多少种不同的放置方法？

假设我们考虑到第 $n$ 个信封，初始时我们暂时把第 $n$ 封信放在第 $n$ 个信封中，然后考虑两种情况的递推：

- 前面 $n-1$ 个信封全部装错；
- 前面 $n-1$ 个信封有一个没有装错其余全部装错。

对于第一种情况，前面 $n-1$ 个信封全部装错：因为前面 $n-1$ 个已经全部装错了，所以第 $n$ 封只需要与前面任一一个位置交换即可，总共有 $f(n-1)\times (n-1)$ 种情况。

对于第二种情况，前面 $n-1$ 个信封有一个没有装错其余全部装错：考虑这种情况的目的在于，若 $n-1$ 个信封中如果有一个没装错，那么我们把那个没装错的与 $n$ 交换，即可得到一个全错位排列情况。

其他情况，我们不可能通过一次操作来把它变成一个长度为 $n$ 的错排。

于是可得错位排列的递推式为 $f(n)=(n-1)(f(n-1)+f(n-2))$。

错位排列数列的前几项为 $0,1,2,9,44,265$。

### 圆排列

$n$ 个人全部来围成一圈，所有的排列数记为 $\mathrm Q_n^n$。考虑其中已经排好的一圈，从不同位置断开，又变成不同的队列。
所以有

$$
\mathrm Q_n^n \times n = \mathrm A_n^n \Longrightarrow \mathrm Q_n = \frac{\mathrm A_n^n}{n} = (n-1)!
$$

由此可知部分圆排列的公式：

$$
\mathrm Q_n^r = \frac{\mathrm A_n^r}{r} = \frac{n!}{r \times (n-r)!}
$$

## 组合数性质 | 二项式推论

由于组合数在 OI 中十分重要，因此在此介绍一些组合数的性质。

$$
\binom{n}{m}=\binom{n}{n-m}\tag{1}
$$

相当于将选出的集合对全集取补集，故数值不变。（对称性）

$$
\binom{n}{k} = \frac{n}{k} \binom{n-1}{k-1}\tag{2}
$$

由定义导出的递推式。

$$
\binom{n}{m}=\binom{n-1}{m}+\binom{n-1}{m-1}\tag{3}
$$

组合数的递推式（杨辉三角的公式表达）。我们可以利用这个式子，在 $O(n^2)$ 的复杂度下推导组合数。

$$
\binom{n}{0}+\binom{n}{1}+\cdots+\binom{n}{n}=\sum_{i=0}^n\binom{n}{i}=2^n\tag{4}
$$

这是二项式定理的特殊情况。取 $a=b=1$ 就得到上式。

$$
\sum_{i=0}^n(-1)^i\binom{n}{i}=[n=0]\tag{5}
$$

二项式定理的另一种特殊情况，可取 $a=1, b=-1$。式子的特殊情况是取 $n=0$ 时答案为 $1$。

$$
\sum_{i=0}^m \binom{n}{i}\binom{m}{m-i} = \binom{m+n}{m}\ \ \ (n \geq m)\tag{6}
$$

拆组合数的式子，在处理某些数据结构题时会用到。

$$
\sum_{i=0}^n\binom{n}{i}^2=\binom{2n}{n}\tag{7}
$$

这是 $(6)$ 的特殊情况，取 $n=m$ 即可。

$$
\sum_{i=0}^ni\binom{n}{i}=n2^{n-1}\tag{8}
$$

带权和的一个式子，通过对 $(3)$ 对应的多项式函数求导可以得证。

$$
\sum_{i=0}^ni^2\binom{n}{i}=n(n+1)2^{n-2}\tag{9}
$$

与上式类似，可以通过对多项式函数求导证明。

$$
\sum_{l=0}^n\binom{l}{k} = \binom{n+1}{k+1}\tag{10}
$$

可以通过组合意义证明，在恒等式证明中较常用。

$$
\binom{n}{r}\binom{r}{k} = \binom{n}{k}\binom{n-k}{r-k}\tag{11}
$$

通过定义可以证明。

$$
\sum_{i=0}^n\binom{n-i}{i}=F_{n+1}\tag{12}
$$

其中 $F$ 是斐波那契数列。

$$
\sum_{l=0}^n \binom{l}{k} = \binom{n+1}{k+1}\tag{13}
$$

通过组合分析——考虑 $S={a_1, a_2, \cdots, a_{n+1}}$ 的 $k+1$ 子集数可以得证。


## 习题

### 求组合数

> [!NOTE] **[AcWing 885. 求组合数 I](https://www.acwing.com/problem/content/887/)**
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

const int N = 2010;
const int mod = 1e9 + 7;

int c[N][N];

void init() {
    for (int i = 0; i < N; ++ i )
        for (int j = 0; j <= i; ++ j )
            if (!j) c[i][j] = 1;
            else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
}

int main() {
    int n;
    init();
    cin >> n;
    
    while (n -- ) {
        int a, b;
        cin >> a >> b;
        cout << c[a][b] << endl;
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

> [!NOTE] **[AcWing 886. 求组合数 II](https://www.acwing.com/problem/content/888/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> $$
> C_{a}^{b} = a! / ((a - b)! b!) \\
> 
> fact[i] = i! mod 1e9+7 \\
> 
> infacr[i] = (i!)^-1 mod 1e9+7
> $$
> 
> ==>
> 
> $$
> C_{a}^{b} = fact[a] * infact[a - b] * infact[b]
> $$
> 
> 核心思想 预处理阶乘以及阶乘模逆元

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 100010, mod = 1e9 + 7;

int fact[N], infact[N];

int qmi(int a, int k, int p) {
    int res = 1;
    while (k) {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

void init() {
    fact[0] = infact[0] = 1;
    for (int i = 1; i < N; ++ i ) {
        fact[i] = (LL)fact[i - 1] * i % mod;
        infact[i] = (LL)infact[i - 1] * qmi(i, mod - 2, mod) % mod;
    }
}

int main() {
    init();
    
    int n;
    cin >> n;
    while (n -- ) {
        int a, b;
        cin >> a >> b;
        cout << (LL)fact[a] * infact[b] % mod * infact[a - b] % mod << endl;
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

> [!NOTE] **[AcWing 887. 求组合数 III](https://www.acwing.com/problem/content/889/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 时间复杂度 接近 10^18 -> 4*10^7   $PlogNlogP$
>
> 基于：
> 
>    $$ [C_{a}^{b}] = [C_{a\bmod p}^{b\bmod p}] * [C_{a/p}^{b/p}]  (\bmod p)$$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

int qmi(int a, int k, int p) {
    int res = 1;
    while (k) {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

// Ca b = a! / ((a- b)! * b!) = a*(a-1)*...*(a-b+1) / b!
int C(int a, int b, int p) {
    if (b > a) return 0;
    int res = 1;
    for (int i = 1, j = a; i <= b; ++ i , -- j ) {
        // res = res * j / i = res * j * i^-1
        res = (LL)res * j % p;
        res = (LL)res * qmi(i, p - 2, p) % p;
    }
    return res;
}

int lucas(LL a, LL b, int p) {
    if (a < p && b < p) return C(a, b, p);
    return (LL)C(a % p, b % p, p) * lucas(a / p, b / p, p) % p;
}

int main() {
    int n;
    cin >> n;
    
    while (n -- ) {
        LL a, b;
        int p;
        cin >> a >> b >> p;
        cout << lucas(a, b, p) << endl;
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

> [!NOTE] **[AcWing 888. 求组合数 IV](https://www.acwing.com/problem/content/890/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> $$
> C_{a}^{b} = [a*(a-1)*...*(a-b+1)] / [b*(b-1)*...*1]
> $$
> 
> 大数运算很难
>
> 故第一步：将 $C_{a}^{b}$ 分解质因数：$$C_{a}^{b} = p1^{a1} ... pk^{ak}$$
>
> 此时 $C_{a}^{b}$ 表示为 $C_{a}^{b} = a! / [b! * (a-b)!]$ 更好
>
> 先求分子的p 再减去分母的p即可
>
> 【求每个质数的次数】
>
> $$a! = a/p + a/p^2 + ... + a/p^k$$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 5010;

int primes[N], cnt;
int sum[N]; // 每个质数的次数
bool st[N];

void get_primes(int n) {
    for (int i = 2; i <= n; ++ i ) {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; ++ j ) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}

// 获取质数次数
int get(int n, int p) {
    int res = 0;
    while (n) {
        res += n / p;
        n /= p;
    }
    return res;
}

vector<int> mul(vector<int> a, int b) {
    vector<int> c;
    int t = 0;
    for (int i = 0; i < a.size(); ++ i ) {
        t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }
    while (t) {
        c.push_back(t % 10);
        t /= 10;
    }
    return c;
}

int main() {
    int a, b;
    cin >> a >> b;
    
    get_primes(a);
    
    for (int i = 0; i < cnt; ++ i ) {
        int p = primes[i];
        sum[i] = get(a, p) - get(a - b, p) - get(b, p);
    }
    
    vector<int> res;
    res.push_back(1);
    
    for (int i = 0; i < cnt; ++ i )
        for (int j = 0; j < sum[i]; ++ j )
            res = mul(res, primes[i]);
            
    for (int i = res.size() - 1; i >= 0; -- i ) cout << res[i];
    cout << endl;
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

### 组合数-隔板法

> [!NOTE] **[AcWing 1308. 方程的解](https://www.acwing.com/problem/content/1310/)**
> 
> 题意: 
> 
> 求非负整数解

> [!TIP] **思路**
> 
> 隔板法 + 高精度加

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const static int N = 150;   // 组合数大数的位数

int k, x;
int f[1000][100][N];    // 大数记录组合数

int qmi(int a, int b, int p) {
    a %= p;
    int ret = 1;
    while (b) {
        if (b & 1)
            ret = ret * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return ret;
}

void add(int c[], int a[], int b[]) {
    for (int i = 0, t = 0; i < N; ++ i ) {
        t += a[i] + b[i];
        c[i] = t % 10;
        t /= 10;
    }
}

int main() {
    cin >> k >> x;
    
    int n = qmi(x, x, 1000);
    
    // C(n - 1, k - 1);
    for (int i = 0; i < n; ++ i )
        for (int j = 0; j <= i && j < k; ++ j )
            if (!j)
                f[i][j][0] = 1;
            else
                add(f[i][j], f[i - 1][j], f[i - 1][j - 1]);
    
    int *g = f[n - 1][k - 1];
    int i = N - 1;
    while (!g[i])
        i -- ;
    while (i >= 0)
        cout << g[i -- ];
    
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

> [!NOTE] **[AcWing 1312. 序列统计](https://www.acwing.com/problem/content/1314/)** [TAG]
> 
> 题意: 
> 
> 给定三个整数 $N,L,R$，统计长度在 $1$ 到 $N$ 之间，元素大小都在 $L$ 到 $R$ 之间的单调不降序列的数量。
> 
> 输出答案对 $10^6+3$ 取模的结果。

> [!TIP] **思路**
> 
> 约定：题目中的 $N,L,R$ 用 $n,l,r$ 表示。
> 
> 数据之间只存在相对关系，则将 $[l,r]$ 区间映射到 $[0,r-l]$，设序列长度为 $k$，题意即求：
> 
> $$
> 满足 \ 0 \le a_1 \le a_2 \le \cdots \le a_k \le r-l,\ 其中 \ a_i \in[0,r-l] \ 的序列个数
> $$
> 
> 令 $x_1 = a_1, x_2 = a_2 - a_1, \cdots, x_k = a_k - a_{k-1}$，则有：
> 
> $$
> 0 \le x_1 + x_2 + \cdots + x_k \le r-l,\ 其中 \ x_i \ge 0 
> $$
> 
> 问题即：用不超过 $r-l$ 个小球放入 $k$ 个盒子，**盒子允许为空**的方案数。
> 
> 等价为：先给每一个盒子放入一个小球（需要令小球的总数 $+k$），即用不超过 $r-l+k$ 个小球放入 $k$ 个盒子，且**盒子不空**的方案数。
> 
> 显然可以「隔板法」。但本题的条件是不等式，对于等式直接用 $k-1$ 个隔板将所有小球分为 $k$ 部分即可；对于不等式要用 $k$ 个隔板，将所有小球分为 $k+1$ 部分，其中最后一部分被舍弃（即不选用），本题最后一部分的个数可以为零。
> 
> 则最终答案为 $\sum_{k=1}^{n}C_{r-l+k}^{k}$
> 
> 数据范围 $10^9$，显然不能枚举长度 $k$ 来累加答案，由两个组合数公式进行推导：
> 
> $$
> C_n^m=C_n^{n-m}, \ C_n^m = C_{n-1}^{m} + C_{n-1}^{m-1}
> $$
> 
> 令 $r-l=m$，则：
> 
> $$
> \begin{aligned}
> 原式= &\sum_{k=1}^{n}C_{m+k}^{k} \\
> = &C_{m+1}^{1} + C_{m+2}^{2} + \cdots + C_{m+n}^{n} \\
> = &C_{m+1}^{m} + C_{m+2}^{m} + \cdots + C_{m+n}^{m} \\
> = &({\color{red}{C_{m+1}^{m+1}}} + C_{m+1}^{m}) + \cdots + C_{m+n}^{m} - {\color{red}{C_{m+1}^{m+1}}} \\
> = &(C_{m+2}^{m+1} + C_{m+2}^{m}) + \cdots + C_{m+n}^{m} - 1 \\
> = &\cdots \\
> = &C_{m+n+1}^{m+1} - 1
> \end{aligned}
> $$
> 
> 然后用 $\rm Lucas$ 定理求解组合数即可。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int MOD = 1000003;

int qmi(int a, int k) {
    int ret = 1;
    while (k) {
        if (k & 1)
            ret = (LL)ret * a % MOD;
        a = (LL)a * a % MOD;
        k >>= 1;
    }
    return ret;
}

int C(int a, int b) {
    if (a < b)
        return 0;
    int down = 1, up = 1;
    for (int i = a, j = 1; j <= b; -- i , ++ j ) {
        up = (LL)up * i % MOD;
        down = (LL)down * j % MOD;
    }
    return (LL)up * qmi(down, MOD - 2) % MOD;
}

int lucas(int a, int b) {
    if (a < MOD && b < MOD)
        return C(a, b);
    return (LL)lucas(a / MOD, b / MOD) * C(a % MOD, b % MOD) % MOD;
}

int main() {
    int T;
    cin >> T;
    while (T -- ) {
        int n, l, r;
        cin >> n >> l >> r;
        cout << (lucas(r - l + n + 1, r - l + 1) + MOD - 1) % MOD << endl;
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

> [!NOTE] **[LeetCode 3154. 到达第 K 级台阶的方案数](https://leetcode.cn/problems/find-number-of-ways-to-reach-the-k-th-stair/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 结合题意 [约束]，结合数据范围求解
>
> - 暴搜 需要记忆化
>
> - 组合数学 trick
>
>     实际上，本题相当于在连续的第二种操作中间插入第一种操作。
>
>     假设两种操作分别做了 $i$ 次 / $j$ 次，数学分析：
>
>     $第二种操作 上升数量\ up1=1+2+...+2^{j-1}=2^j-1$
>
>     $第一种操作上升数量\ up2=i\ (0<=i<=j+1)$
>
>     $总的上升数量={up1 + up2}$
>
>     则 $k-1\ (-1是因为起点为1)$ 必然在此范围内，由此可推导 $j$ 的合法值，随后排列组合计算即可...


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 搜索**

```cpp
class Solution {
public:
    // 考虑到第一种操作无法无限使用 假设每次跳都执行一次
    // 则根据k的数据范围结合第二种操作的限制  最多跳 32*2=64 次
    
    int k, res;
    
    unordered_map<int, unordered_map<int, unordered_map<int, int>>> mem;
    
    int dfs(int u, int p, int j) {
        if (u > k + 2)
            return 0;
        
        if (mem[u][p].count(j))
            return mem[u][p][j];
        
        int ret = 0;
        
        if (u == k)
            ret ++ ;
        
        if (p != 0)
            ret += dfs(u - 1, 0, j);
        ret += dfs(u + (1 << j), 1, j + 1);
        return mem[u][p][j] = ret;
    }
    
    int waysToReachStair(int k) {
        this->k = k, this->res = 0;
        return dfs(1, -1, 0);
    }
};
```

##### **C++ 组合数学**

```cpp
class Solution {
public:
    const static int N = 32;
    
    int c[N][N];
    void init() {
        for (int i = 0; i < N; ++ i )
            for (int j = 0; j <= i; ++ j )
                if (!j)
                    c[i][j] = 1;
                else
                    c[i][j] = c[i - 1][j] + c[i - 1][j - 1];
    }
    
    int comb(int a, int b) {
        return c[a][b];
    }
    
    int waysToReachStair(int k) {
        init();
        k -- ;  // ATTENTION 起点偏移一下
        
        int res = 0;
        for (int j = 0; j < 31; ++ j ) {
            int tot = (1 << j) - 1;
            if (tot >= k && tot - (j + 1) <= k) {
                int i = tot - k;
                // 在 j+1 个缝隙里找到 i 个位置插入
                res += comb(j + 1, i);
            }
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

### 组合数应用

> [!NOTE] **[AcWing 889. 满足条件的01序列](https://www.acwing.com/problem/content/891/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> **卡特兰数**
>
> 转化为：平面图路径 中线绿 非法线红
>
> 在互补的情况下 $C_{12}^{5} = C_{12}^{7}$
>
> 在线下方的个数等于 $C_{2n}^{n} = C_{2n}^{n} - C_{2n}^{n-1} = C_{2n}^{n} / (n+1)$ 成为卡特兰数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 100010, mod = 1e9 + 7;

int qmi(int a, int k, int p) {
    int res = 1;
    while (k) {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

int main() {
    int n;
    cin >> n;
    
    int a = n * 2, b = n;
    int res = 1;
    // C2n n
    for (int i = a; i > a - b; -- i ) res = (LL)res * i % mod;
    
    for (int i = 1; i <= b; ++ i ) res = (LL)res * qmi(i, mod - 2, mod) % mod;
    
    // 除 n + 1
    res = (LL)res * qmi(n + 1, mod - 2, mod) % mod;
    
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

> [!NOTE] **[Luogu NOIP2003 普及组 栈](https://www.luogu.com.cn/problem/P1044)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典组合数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 40;

LL c[N][N];

int main() {
    for (int i = 0; i < N; ++ i )
        for (int j = 0; j <= i; ++ j )
            if (!j)
                c[i][j] = 1;
            else
                c[i][j] = c[i - 1][j - 1] + c[i - 1][j];
    
    int n;
    cin >> n;
    cout << c[n * 2][n] / (n + 1) << endl;
    
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

> [!NOTE] **[LeetCode 3251. 单调数组对的数目 II](https://leetcode.cn/problems/find-the-count-of-monotonic-pairs-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 将数组拆解，每个数分成两个数字分散在不同数组，且一个递增一个递减，求所有可能方案
> 
> - 标准的前缀和优化
> 
> - TODO: revisit 整理组合数学做法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ DP 前缀和优化**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 2010, M = 1010, MOD = 1e9 + 7;
    
    int f[N][M];
    // 假设对于第i个位置 分配给第一个数组的数值大小为j 则第二个数组为 s_i - j
    // f[i][j] = sum{f[i - 1][k]}  j>=k && (s_i-1 - k >= s_i - j)
    //                      => k <= j && k <= s_i-1 - s_i + j
    //                      伴随着j的枚举 和是单调递增的...
    
    void mod_add(int & a, int b) {
        a = ((LL)a + b) % MOD;
    }
    
    int countOfPairs(vector<int>& nums) {
        int n = nums.size();
        
        memset(f, 0, sizeof f);
        for (int j = 0; j <= nums[0]; ++ j )
            f[1][j] = 1;
        
        for (int i = 2; i <= n; ++ i ) {
            int s = nums[i - 1], s_last = nums[i - 2];
            
            // 枚举当前选择的 j
            int sum = 0, gap = max(0, s - s_last);
            for (int j = 0; j <= s; ++ j ) {
                // k <= j && k <= s_last - s + j
                // 那么对于一个具体的 j 来说，可以取的范围是 [0, min(j, j+s_last-s)]
                // 差值的 gap 是 s-s_last
                if (j - gap >= 0)
                    mod_add(sum, f[i - 1][j - gap]);
                f[i][j] = sum;
            }
        }
        
        int res = 0;
        for (int j = 0; j < M; ++ j )
            mod_add(res, f[n][j]);
        return res;
    }
};
```

##### **C++ 组合数学**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 3010, MOD = 1e9 + 7;
    
    LL f[N], g[N];
    
    LL qpow(LL x, int y) {
        LL ret = 1;
        while (y) {
            if (y & 1)
                ret = ret * x % MOD;
            x = x * x % MOD;
            y >>= 1;
        }
        return ret;
    }
    
    void init() {
        f[0] = g[0] = 1;
        for (int i = 1; i < N; ++ i ) {
            f[i] = f[i - 1] * i % MOD;
            g[i] = g[i - 1] * qpow(i, MOD - 2) % MOD;
        }
    }
    LL C(int n, int m) {
        return f[n] * g[m] % MOD * g[n - m] % MOD;
    }
    
    int countOfPairs(vector<int>& nums) {
        init();
        
        int n = nums.size(), m = nums.back();   // ATTENTION m
        for (int i = 1; i < n; ++ i ) {
            m -= max(0, nums[i] - nums[i - 1]); // 减掉必须往上走的强制约束
            if (m < 0)
                return 0;
        }
        return C(n + m, n); // 总共可以自由走 n+m 步
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

### 综合应用

> [!NOTE] **[Luogu [NOIP2016 提高组] 组合数问题](https://www.luogu.com.cn/problem/P2822)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 为了处理多组数据询问 前缀和预处理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 2010;

int c[N][N];
int s[N][N];

int main() {
    int T, k;
    scanf("%d%d", &T, &k);

    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++) {
            if (!j)
                c[i][j] = 1 % k;
            else
                c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % k;

            if (!c[i][j])
                s[i][j] = 1;
        }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (i)
                s[i][j] += s[i - 1][j];
            if (j)
                s[i][j] += s[i][j - 1];
            if (i && j)
                s[i][j] -= s[i - 1][j - 1];
        }

    while (T--) {
        int n, m;
        scanf("%d%d", &n, &m);

        printf("%d\n", s[n][m]);
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

> [!NOTE] **[LeetCode 1569. 将子数组重新排序得到同一个二叉查找树的方案数](https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc 标准**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e3 + 10, MOD = 1e9 + 7;

    int C[N][N];
    void init() {
        for (int i = 0; i < N; ++ i )
            for (int j = 0; j <= i; ++ j )
                if (!j)
                    C[i][j] = 1;
                else
                    C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) % MOD;
    }

    int f(vector<int> nums) {
        if (nums.empty())
            return 1;
        int n = nums.size(), k = nums[0];
        vector<int> l, r;
        // 排列 不会有多个等于 k 的数
        for (auto x : nums)
            if (x < k)
                l.push_back(x);
            else if (x > k)
                // ATTENTION: 必须进行下标转移的映射
                r.push_back(x - k);
        return (LL)C[n - 1][k - 1] * f(l) % MOD * f(r) % MOD;
    }

    int numOfWays(vector<int>& nums) {
        init();
        // 减去最初的一种方案
        return (f(nums) - 1 + MOD) % MOD;
    }
};
```

##### **C++**

```cpp
class Solution {
    int mod = 1000000007;
    vector<int> op;
    long long P[1005], Q[1005];

public:
    long long mul(long long x, long long y) {
        if (y == 0) return 1;
        if (y == 1) return x;
        long long ret = mul(x, y / 2);
        if (y % 2 == 0)
            return ret * ret % mod;
        else
            return ret * ret % mod * x % mod;
    }
    long long C(long long x, long long y) {
        return P[x] * Q[y] % mod * Q[x - y] % mod;
    }
    int dfs(int l, int r) {
        if (l > r) return 1;
        int mid;
        for (int i = 0; i < op.size(); i++) {
            if (l <= op[i] && op[i] <= r) {
                mid = op[i];
                break;
            }
        }
        long long cnt1 = dfs(l, mid - 1);
        long long cnt2 = dfs(mid + 1, r);
        long long ret = C(r - l, (mid - l)) * cnt1 % mod * cnt2 % mod;
        return ret;
    }
    int numOfWays(vector<int>& nums) {
        op = nums;
        P[0] = Q[0] = 1;
        for (int i = 1; i <= 1000; i++) P[i] = P[i - 1] * i % mod;
        for (int i = 1; i <= 1000; i++) Q[i] = mul(P[i], mod - 2);
        long long ans = dfs(1, nums.size());
        ans = (ans - 1 + mod) % mod;
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

> [!NOTE] **[LeetCode 1573. 分割字符串的方案数](https://leetcode.cn/problems/number-of-ways-to-split-a-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分割 组合数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int MOD = 1e9 + 7;
    int numWays(string s) {
        long long count1 = 0, n = s.size();
        unordered_map<long long, long long> mp;  //存储1的位置
        for (int i = 0; i < n; i++)
            if (s[i] == '1') {
                count1++;  //统计1 的个数
                mp[count1] = i;
            }
        if (count1 == 0) return ((n - 1) * (n - 2) / 2) % MOD;
        if (count1 % 3 != 0) return 0;
        long long t = count1 / 3;  //确立3等分1的个数
        return ((mp[t + 1] - mp[t]) * (mp[2 * t + 1] - mp[2 * t])) % MOD;
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

> [!NOTE] **[LeetCode 1735. 生成乘积数组的方案数](https://leetcode.cn/problems/count-ways-to-make-array-with-product/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 对每个质因数的幂次求组合数即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using LL = long long;
const int N = 10010;
const int mod = 1e9 + 7;

class Solution {
public:
    int f[N], g[N];
    
    int qmi(int a, int b) {
        int res = 1;
        while (b) {
            if (b & 1) res = (LL)res * a % mod;
            a = (LL)a * a % mod;
            b >>= 1;
        }
        return res;
    }

    // a!  /  [(a - b)! * (b)!]
    int C(int a, int b) {
        return (LL)f[a] * g[b] % mod * g[a - b] % mod;
    }
    
    vector<int> waysToFillArray(vector<vector<int>>& queries) {
        f[0] = g[0] = 1;
        for (int i = 1; i < N; ++ i ) {
            f[i] = (LL)f[i - 1] * i % mod;
            g[i] = qmi(f[i], mod - 2);
        }
        
        vector<int> res;
        for (auto & q : queries) {
            int n = q[0], k = q[1];
            int ret = 1;
            for (int i = 2; i <= k / i; ++ i )
                if (k % i == 0) {
                    int s = 0;
                    while (k % i == 0)
                        k /= i, ++ s ;
                    ret = (LL)ret * C(n + s - 1, n - 1) % mod;
                }
            if (k > 1) ret = (LL)ret * C(n, n - 1) % mod;
            res.push_back(ret);
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

> [!NOTE] **[LeetCode 建信04. 电学实验课](https://leetcode.cn/contest/ccbft-2021fall/problems/lSjqMF/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常好的组合计数题
> 
> 重复做 TODO
> 
> 1. 由于每一列只允许防止一个插孔，所以将所有目标插孔按照列从小到大排序后，考虑每两个目标插孔之间的方案数，然后求乘积就是答案。
> 
> 2. 设 v 是一个长度为 row 的数组，v(j) 表示到达第 j 行的方案数。设二维转移矩阵 T，其中 T(i,j)=1 表示可以从前一列的第 i 行到达当前列的第 j 行。
所以每移动一列，v 就可以更新为 T⋅v。移动 k 列时，vk=T^k⋅v0。
> 
> 3. 可以提前预处理出来所有二进制位的矩阵乘积结果，即 T1，T2，T4 直到 T29，然后根据两个目标插孔之间的距离进行组合，计算出方案数。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 33, MOD = 1e9 + 7;
    
    int n, A[N][N][N];
    int t[N], g[N];
    
    void mul(int c[N][N], int a[N][N], int b[N][N]){
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ ){
                c[i][j] = 0;
                for (int k = 0; k < n; k ++ )
                    c[i][j] = (c[i][j] + 1ll * a[i][k] * b[k][j]) % MOD;
            }
    }
    
    void mul2(int c[N], int a[N], int b[N][N]){
        for (int i = 0; i < n; i ++ ) c[i] = 0;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j++)
                c[j] = (c[j] + 1ll * a[i] * b[i][j]) % MOD;
    }
    
    int electricityExperiment(int row, int col, vector<vector<int>>& p) {
        // Step 1: sorting
        this->n = row;
        sort(p.begin(), p.end(), [](vector<int> & a, vector<int> & b){
            return a[1] < b[1];
        });
        
        // Step 2: return 0 if impossible
        int len = p.size();
        for (int i = 1; i < len; i ++ )
            if (p[i][1] == p[i-1][1])
                return 0;
        
        // Step 3.1: init A[0]
        for (int i = 0; i < n; i ++ )
            for (int j = -1; j <= 1; j ++ )
                if (i + j >= 0 && i + j < n)
                    A[0][i + j][i] = 1;
        // Step 3.2: calc the A
        for (int k = 1; k <= 30; k ++ )
            mul(A[k], A[k - 1], A[k - 1]);
        
        // Step 4: calc the result
        int res = 1;
        for (int i = 1; i < len; i ++ ){
            memset(t, 0, sizeof t);
            t[p[i - 1][0]] = 1;
            int d = p[i][1] - p[i - 1][1];
            for (int k = 0; (1 << k) <= d; k ++ )
                if ((d >> k) & 1) {
                    mul2(g, t, A[k]);
                    memcpy(t, g, sizeof(g));
                }
            res = 1ll * res * t[p[i][0]] % MOD;
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

> [!NOTE] **[Codeforces C. Beautiful Numbers](https://codeforces.com/problemset/problem/300/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常好的组合数题目 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Beautiful Numbers
// Contest: Codeforces - Codeforces Round #181 (Div. 2)
// URL: https://codeforces.com/problemset/problem/300/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 非常好的组合数题目
// 暴力会遍历跑lucas会超时 考虑记忆化
// https://codeforces.com/contest/300/submission/110875084
// 记忆化仍然超时
// https://codeforces.com/contest/300/submission/110875327
// 参考luogu的做法，这里会频繁用到组合数所以直接递推计算 而不用lucas
// https://www.luogu.com.cn/problem/solution/CF300C
// 预处理阶乘和逆

using LL = long long;
const int N = 1000010, MOD = 1e9 + 7;

int a, b, n;
int fact[N], infact[N];

bool check(int x) {
    while (x) {
        int t = x % 10;
        if (t != a && t != b)
            return false;
        x /= 10;
    }
    return true;
}

int qmi(int a, int k, int p) {
    int res = 1;
    while (k) {
        if (k & 1)
            res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

void init() {
    fact[0] = infact[0] = 1;
    for (int i = 1; i < N; ++i) {
        fact[i] = (LL)fact[i - 1] * i % MOD;
        infact[i] = (LL)infact[i - 1] * qmi(i, MOD - 2, MOD) % MOD;
    }
}

int main() {
    init();

    cin >> a >> b >> n;

    LL res = 0;

    // 枚举 a 的个数
    for (int i = 0; i <= n; ++i) {
        int s = a * i + b * (n - i);
        if (check(s))
            res = (res + (LL)fact[n] * infact[i] % MOD * infact[n - i]) % MOD;
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

> [!NOTE] **[LeetCode 2338. 统计理想数组的数目](https://leetcode.cn/problems/count-the-number-of-ideal-arrays/) [TAG]**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 先考虑数列无重复数字的情况，显然可以 DP 递推来求每个长度下不同结尾数字的方案数
> 
> - 进一步考虑，在无重复数字情况下，每一种方案按隔板法求有重复数字下的方案数
> 
> 求和即可
> 
> > TODO: 积性函数解 https://leetcode.cn/problems/count-the-number-of-ideal-arrays/solution/shu-ju-fan-wei-ge-ju-xiao-liao-by-johnkr-dl63/

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e4 + 10, M = 15, MOD = 1e9 + 7;  // 对于 1e4 在一个数组内最多存在 log1e4 + 1 = 14 个不同的数
    
    int fact[N], infact[N];
    int qmi(int a, int k, int p) {
        int ret = 1;
        while (k) {
            if (k & 1)
                ret = ((LL)ret * a) % MOD;
            a = (LL)a * a % MOD;
            k >>= 1;
        }
        return ret;
    }
    void init() {
        fact[0] = infact[0] = 1;
        for (int i = 1; i < N; ++ i ) {
            fact[i] = (LL)fact[i - 1] * i % MOD;
            infact[i] = (LL)infact[i - 1] * qmi(i, MOD - 2, MOD) % MOD;
        }
    }
    int C(int x, int y) {
        return (LL)fact[x] * infact[x - y] % MOD * infact[y] % MOD;
    }
    
    int f[M][N];    // 长度为 i 末尾数值为 j 的所有不重复数方案数
    
    int idealArrays(int n, int maxValue) {
        init();
        
        memset(f, 0, sizeof f);
        for (int i = 1; i <= maxValue; ++ i )
            f[1][i] = 1;
        for (int i = 1; i < M - 1; ++ i )
            for (int j = 1; j <= maxValue; ++ j )
                for (int k = 2; k * j <= maxValue; ++ k )   // log
                    f[i + 1][k * j] = (f[i + 1][k * j] + f[i][j]) % MOD;
        
        int res = 0;
        for (int i = 1; i <= n && i < M; ++ i )
            for (int j = 1; j <= maxValue; ++ j )
                // C_{n - 1}^{i - 1} 隔板法
                res = (res + (LL)f[i][j] * C(n - 1, i - 1)) % MOD;
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

> [!NOTE] **[LeetCode 2842. 统计一个字符串的 k 子序列美丽值最大的数目](https://leetcode.cn/problems/count-k-subsequences-of-a-string-with-maximum-beauty/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典组合数 加快速度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using LL = long long;
using PII = pair<int, int>;
const static int N = 2e5 + 10, M = 30, MOD = 1e9 + 7;

LL qpow(int x, int y) {
    LL ret = 1;
    while (y) {
        if (y & 1)
            ret = ret * x % MOD;
        x = (LL)x * x % MOD;
        y >>= 1;
    }
    return ret;
}

bool flag = false;
int fact[N], infact[N];
void init() {
    if (flag)
        return;
    flag = true;
    fact[0] = infact[0] = 1;
    for (int i = 1; i < N; ++ i ) {
        fact[i] = (LL)fact[i - 1] * i % MOD;
        infact[i] = (LL)infact[i - 1] * qpow(i, MOD - 2) % MOD;
    }
}

LL C(int x, int y) {
    return (LL)fact[x] * infact[y] % MOD * infact[x - y] % MOD;
}


class Solution {
public:
    int c[M];
    int n;
    
    int countKSubsequencesWithMaxBeauty(string s, int k) {
        init();
        memset(c, 0, sizeof c);
        this->n = s.size();
        for (auto ch : s)
            c[ch - 'a'] ++ ;
        
        {
            int cnt = 0;
            for (int i = 0; i < M; ++ i )
                if (c[i])
                    cnt ++ ;
            // cout << " s = " << s << " cnt = " << cnt << endl;
            if (cnt < k)
                return 0;
        }
        
        unordered_map<int, int> h;
        for (int i = 0; i < 26; ++ i )
            if (c[i])
                h[c[i]] ++ ;        // 出现了 x 次的有 y 个
        
        vector<PII> xs;
        for (auto [x, y] : h)
            xs.push_back({x, y});   // 按 x 次数降序排序
        
        sort(xs.begin(), xs.end());
        reverse(xs.begin(), xs.end());
        
        LL res = 1;
        for (auto [x, y] : xs) {
            if (k >= y) {
                // 这么多个(y)必然全选, 每一个都是 x 种可能
                res = res * qpow(x, y) % MOD;
                k -= y;
            } else {
                // 这么多个(y)只挑 k 个选, 每一个都是 x 种可能
                res = res * qpow(x, k) % MOD * C(y, k) % MOD;
                k = 0;
            }
            if (k == 0)
                break;
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

> [!NOTE] **[LeetCode 2954. 统计感冒序列的数目](https://leetcode.cn/problems/count-the-number-of-infection-sequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 组合计数应用题 注意去重的推导和实现思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using LL = long long;
const static int N = 1e5 + 10, MOD = 1e9 + 7;

LL qpow(LL a, LL b) {
    LL ret = 1;
    while (b) {
        if (b & 1)
            ret = ret * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return ret;
}

LL f[N], g[N];  // 阶乘 & 逆元
bool flag = false;
void init() {
    if (flag)
        return;
    flag = true;
    f[0] = g[0] = 1;
    for (int i = 1; i < N; ++ i ) {
        f[i] = f[i - 1] * i % MOD;
        g[i] = g[i - 1] * qpow(i, MOD - 2) % MOD;
    }
}

class Solution {
public:
    // 直观看没有太好的思路 考虑尝试数学解法 [组合数学]
    //
    // ATTENTION 假设连续 0 的数量为 k
    // - 如果只有一边都可以发生传染，那么只有一种传染序列
    // - 如果两边都可以，那么每一次都可以选择从左或从右传染，除了最后一次，因此有 2^{k−1} 种序列
    //
    // 则 分区间统计累乘即可 ==> 实际上会有重复情况，因为在不同区间里也只能一次选其中的一个
    // ==> 重复元素排列问题 【计算细节】
    
    LL comb(int a, int b) {
        return f[a] * g[a - b] % MOD * g[b] % MOD;
    }
    
    int numberOfSequence(int n, vector<int>& sick) {
        init();
        sort(sick.begin(), sick.end());
        sick.push_back(n);  // 边界哨兵

        LL res = 1;
        int tot = 0, last = -1;
        for (auto x : sick) {
            int k = x - last - 1;
            if (k > 0) {
                LL t = 0;
                if (last == -1 || x == n) {
                    // 特殊情况 只能从某一个方向转移过来
                    t = 1;
                } else {
                    // 可以从 左/右 转移过来 => 2^(k-1)
                    t = qpow(2, k - 1);
                }
                // 【ATTENTION: 计算逻辑】
                // 在总长为 tot+len 的序列中，选择 len 个位置填充 (内部的填充方法有 t 种)
                res = res * comb(tot + k, k) % MOD * t % MOD;
            }
            last = x, tot += k;
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

> [!NOTE] **[LeetCode 2963. 统计好分割方案的数目](https://leetcode.cn/problems/count-the-number-of-good-partitions/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 区间合并 + 组合数 (后者也可转化成 $2^{m-1}$ 思考)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using LL = long long;
const static int N = 1e5 + 10, MOD = 1e9 + 7;

int qmi(int a, int k, int p) {
    int ret = 1;
    while (k) {
        if (k & 1)
            ret = (LL)ret * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return ret;
}

int fact[N], infact[N];
bool inited = false;

void init() {
    if (inited)
        return;
    inited = true;
    
    fact[0] = infact[0] = 1;
    for (int i = 1; i < N; ++ i ) {
        fact[i] = (LL)fact[i - 1] * i % MOD;
        infact[i] = (LL)infact[i - 1] * qmi(i, MOD - 2, MOD) % MOD;
    }
}

int comb(int a, int b) {
    return (LL)fact[a] * infact[b] % MOD * infact[a - b] % MOD;
}

class Solution {
public:
    using PII = pair<int, int>;
    
    void merge(vector<PII> & segs) {
        vector<PII> res;
        sort(segs.begin(), segs.end());
        int st = -2e9, ed = -2e9;
        for (auto seg : segs)
            if (ed < seg.first) {
                if (st != -2e9)
                    res.push_back({st, ed});
                st = seg.first, ed = seg.second;
            } else
                ed = max(ed, seg.second);
        if (st != -2e9)
            res.push_back({st, ed});
        segs = res;
    }
    
    int numberOfGoodPartitions(vector<int>& nums) {
        init();
        
        int n = nums.size();
        
        unordered_set<int> S;
        unordered_map<int, int> l, r;
        {
            // 先找到 最左侧/最右侧 出现的位置
            for (int i = 0; i < n; ++ i ) {
                int x = nums[i];
                if (S.count(x)) {
                    l[x] = min(l[x], i), r[x] = max(r[x], i);
                } else {
                    l[x] = r[x] = i;
                    S.insert(x);
                }
            }
        }
        vector<PII> xs;
        {
            // 根据出现过的元素划分基本区间
            for (auto [k, v] : l)
                xs.push_back({v, r[k]});
            // 区间合并
            merge(xs);
        }
        
        // 留下的都是可以任意合并的区间
        // 最坏情况下 m=1e5
        int m = xs.size(), res = 0;
        for (int i = 0; i < m; ++ i )
            // C{m-1}{i}
            res = (res + comb(m - 1, i)) % MOD;
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

### 排列数

> [!NOTE] **[AcWing 1309. 车的放置](https://www.acwing.com/problem/content/1311/)** [TAG]
> 
> 题意: 
> 
> L 型方格，求放 $k$ 个 `车` 的方案数

> [!TIP] **思路**
> 
> 先考虑简单情况，对于 $n * m$ 的矩阵，放置 $k$ 个 `车` 的方案数为：
> 
> $C_n^k * A_m^k$ 先选 $k$ 行，再有次序的选择 $k$ 列 (**组合数**)
> 
> 对于本题，按照上下两部分进行切分，切分后各自结合符合 **乘法原理**
> 
> 假定上半部分摆放 $i$ 个 ==> $C_b^i * A_a^i$
> 
> 下半部分相应为 $k-i$ 个 ==> $C_d^{k-i} * A_{a+c-i}^{k-i}$
> 
> 枚举 $i$ 相乘累加即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 2010, MOD = 100003;

int fact[N], infact[N];

int qmi(int a, int k) {
    int ret = 1;
    while (k) {
        if (k & 1)
            ret = (LL)ret * a % MOD;
        a = (LL)a * a % MOD;
        k >>= 1;
    }
    return ret;
}

int C(int a, int b) {
    if (a < b)
        return 0;
    return (LL)fact[a] * infact[a - b] % MOD * infact[b] % MOD;
}

int A(int a, int b) {
    if (a < b)
        return 0;
    return (LL)fact[a] * infact[a - b] % MOD;
}

int main() {
    fact[0] = infact[0] = 1;
    for (int i = 1; i < N; ++ i ) {
        fact[i] = (LL)fact[i - 1] * i % MOD;
        infact[i] = (LL)infact[i - 1] * qmi(i, MOD - 2) % MOD;
    }
    
    int a, b, c, d, k;
    cin >> a >> b >> c >> d >> k;
    
    int res = 0;
    for (int i = 0; i <= k; ++ i )
        res = (res + (LL)C(b, i) * A(a, i) % MOD * C(d, k - i) % MOD * A(a + c - i, k - i) % MOD) % MOD;
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

> [!NOTE] **[LeetCode 2514. 统计同位异构字符串数目](https://leetcode.cn/problems/count-anagrams/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准 可重集排列问题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10, MOD = 1e9 + 7;
    
    LL f[N], g[N];
    LL qmi(int a, int k) {
        LL ret = 1;
        while (k) {
            if (k & 1)
                ret = (LL)ret * a % MOD;
            a = (LL)a * a % MOD;
            k >>= 1;
        }
        return ret;
    }
    void init() {
        f[0] = g[0] = 1;
        for (int i = 1; i < N; ++ i ) {
            f[i] = f[i - 1] * (LL)i % MOD;
            g[i] = g[i - 1] * qmi(i, MOD - 2) % MOD;
        }
    }
    
    LL get(string & s) {
        unordered_map<char, int> hash;
        for (auto c : s)
            hash[c] ++ ;
        
        LL ret = f[s.size()] % MOD;
        for (auto [k, v] : hash)
            ret = ret * g[v] % MOD;
        return ret;
    }
    
    int countAnagrams(string s) {
        init();
        
        stringstream ss(s);
        string t;
        LL res = 1;
        while (ss >> t)
            res = (res * get(t)) % MOD;
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