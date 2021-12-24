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
> C_{a}^{b} = a! / ((a - b)! b!)
>   fact[i] = i! mod 1e9+7
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

> [!NOTE] **[AcWing 889. 满足条件的01序列](https://www.acwing.com/problem/content/891/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 卡特兰数
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