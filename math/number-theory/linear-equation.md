## 介绍

形如 $ax \equiv c \pmod b$ 的方程被称为 **线性同余方程**(Congruence Equation)。

[「NOIP2012」同余方程](https://loj.ac/problem/2605)

## 求解方法

根据以下两个定理，我们可以求出同余方程 $ax \equiv c \pmod b$ 的解。

**定理 1**：方程 $ax+by=c$ 与方程 $ax \equiv c \pmod b$ 是等价的，有整数解的充要条件为 $\gcd(a,b) \mid c$。

根据定理 1，方程 $ax+by=c$，我们可以先用扩展欧几里得算法求出一组 $x_0,y_0$，也就是 $ax_0+by_0=\gcd(a,b)$，然后两边同时除以 $\gcd(a,b)$，再乘 $c$。然后就得到了方程 $a\dfrac{c}{\gcd(a,b)}x_0+b\dfrac{c}{\gcd(a,b)}y_0=c$，然后我们就找到了方程的一个解。

**定理 2**：若 $\gcd(a,b)=1$，且 $x_0$、$y_0$ 为方程 $ax+by=c$ 的一组解，则该方程的任意解可表示为：$x=x_0+bt$，$y=y_0-at$, 且对任意整数 $t$ 都成立。

根据定理 2，可以求出方程的所有解。但在实际问题中，我们往往被要求求出一个最小整数解，也就是一个特解 $x=(x \bmod t+t) \bmod t$，其中 $t=\dfrac{b}{\gcd(a,b)}$。



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int ex_gcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    int d = ex_gcd(b, a % b, x, y);
    int temp = x;
    x = y;
    y = temp - a / b * y;
    return d;
}
bool liEu(int a, int b, int c, int& x, int& y) {
    int d = ex_gcd(a, b, x, y);
    if (c % d != 0) return 0;
    int k = c / d;
    x *= k;
    y *= k;
    return 1;
}
```

###### **Python**

```python
# Python Version
def ex_gcd(a, b ,x, y):
    if b == 0:
        x = 1; y = 0
        return a
    d = ex_gcd(b, a % b, x, y)
    temp = x
    x = y
    y = temp - a // b * y
    return d
    
def liEu(a, b, c, x, y):
    d = ex_gcd(a, b, x, y)
    if c % d != 0:
        return 0
    k = c // d
    x = x * k
    y = y * k
    return 1
```

<!-- tabs:end -->
</details>

## 习题

> [!NOTE] **[AcWing 203. 同余方程](https://www.acwing.com/problem/content/205/)**
> 
> 题意: 
> 
> 求线性同余方程的最小正整数解

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
    int a, b;
    cin >> a >> b;
    
    int x, y;
    exgcd(a, b, x, y);
    
    // ax + by = 1
    // d = 1
    // x = k * (b / d)
    cout << (x % b + b) % b << endl;
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

> [!NOTE] **[AcWing 222. 青蛙的约会](https://www.acwing.com/problem/content/224/)**
> 
> 题意: 
> 
> 转化，求最小正整数解

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
x + an == y + am + KL
an - am == y - x + KL
a(n - m) == (y - x) mod L
// 更换a x y后整理得
x(n - m) - yL == distance
*/
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

LL exgcd(LL a, LL b, LL & x, LL & y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int main() {
    LL a, b, m, n, L;
    cin >> a >> b >> m >> n >> L;
    
    LL x, y;
    LL d = exgcd(m - n, L, x, y);
    if ((b - a) % d) cout << "Impossible" << endl;
    else {
        // x是原方程的解 通解为 x0 + k*(L/d)
        // 原方程解需先扩大若干倍
        x *= (b - a) / d;
        // 现在要求最小正整数解 mod L/d
        LL t = abs(L / d);
        cout << (x % t + t) % t << endl;
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

> [!NOTE] **[AcWing 202. 最幸运的数字](https://www.acwing.com/problem/content/204/)** [TAG]
> 
> 题意: 
> 
> 现在给定一个正整数 L ，请问至少多少个 8 连在一起组成的正整数（即最小幸运数字）是 L 的倍数

> [!TIP] **思路**
> 
> TODO: 细节 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
L | 888...888 (x个8)
也即
L | 8 * (111...111)     (x个1)
L | 8 * (999...999) / 9 (x个9)
L | 8 * (10^x - 1) / 9

9*L | 8*(10^x-1)
假设 d = (L, 8) 则
9*L/d | 10^x-1
假设 C = 9*L/d 为常数 则
10^x === 1 mod C
显然：
10^phi(C) 为解

【有结论：可以满足10^x === 1 mod C 的最小正整数解x x一定可以整除phi(C)】
*/
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

// qmi中乘法会爆long long  故再使用慢速乘
LL qmul(LL a, LL k, LL b) {
    LL res = 0;
    while (k) {
        if (k & 1) res = (res + a) % b;
        a = (a + a) % b;
        k >>= 1;
    }
    return res;
}

LL qmi(LL a, LL k, LL b) {
    LL res = 1;
    while (k) {
        if (k & 1) res = qmul(res, a, b);
        a = qmul(a, a, b);
        k >>= 1;
    }
    return res;
}

LL get_euler(LL C) {
    LL res = C;
    for (LL i = 2; i <= C / i; ++ i )
        if (C % i == 0) {
            while (C % i == 0) C /= i;
            res = res / i * (i - 1);
        }
    if (C > 1) res = res / C * (C - 1);
    return res;
}

int main() {
    int T = 1;
    LL L;
    while (cin >> L, L) {
        // int d = 1;
        // while (L % (d * 2) == 0 && d * 2 <= 8) d *= 2;
        int d = __gcd(8ll, L);
        
        LL C = 9 * L / d;
        
        LL phi = get_euler(C);
        
        LL res = 1e18;
        if (C % 2 == 0 || C % 5 == 0) res = 0;
        else {
            // 枚举所有约数
            for (LL d = 1; d * d <= phi; ++ d )
                if (phi % d == 0) {
                    if (qmi(10, d, C) == 1) res = min(res, d);
                    if (qmi(10, phi / d, C) == 1) res = min(res, phi / d);
                }
        }
        printf("Case %d: %lld\n", T ++, res);
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

> [!NOTE] **[AcWing 1298. 曹冲养猪](https://www.acwing.com/problem/content/1300/)**
> 
> 题意: 
> 
> 中国剩余定理

> [!TIP] **思路**
> 
> TODO: 细节 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
中国剩余定理
*/
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 10;

int n;
int A[N], B[N];

LL exgcd(LL a, LL b, LL & x, LL & y) {
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
    
    LL M = 1;
    for (int i = 0; i < n; ++ i ) {
        cin >> A[i] >> B[i];
        M *= A[i];
    }
    
    LL res = 0;
    for (int i = 0; i < n; ++ i ) {
        // x === Bi mod Ai
        // Mi * ti === 1 mod Ai
        // ti 是 Mi 的逆元
        // 答案为 sum BiMiti
        LL Mi = M / A[i];
        LL ti, x;
        exgcd(Mi, A[i], ti, x);
        res += B[i] * Mi * ti;
    }
    cout << (res % M + M) % M << endl;
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