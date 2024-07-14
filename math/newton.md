
具体的任务是，对于在 $[a,b]$ 上连续且单调的函数 $f(x)$，求方程 $f(x)=0$ 的近似解。

## 算法描述

初始时我们从给定的 $f(x)$ 和一个近似解 $x_0$ 开始。（$x_0$ 的值可任意取）

假设我们目前的近似解是 $x_i$，我们画出与 $f(x)$ 切于点 $(x_i,f(x_i))$ 的直线 $l$，将 $l$ 与 $x$ 轴的交点横坐标记为 $x_{i+1}$，那么这就是一个更优的近似解。重复这个迭代的过程。
根据导数的几何意义，可以得到如下关系：

$$
 f^\prime(x_i) = \frac{f(x_i)}{x_{i} - x_{i+1}}
$$

整理后得到如下递推式：

$$
 x_{i+1} = x_i - \frac{f(x_i)}{f^\prime(x_i)}
$$

直观地说，如果 $f(x)$ 比较平滑，那么随着迭代次数的增加，$x_i$ 会越来越逼近方程的解。

牛顿迭代法的收敛率是平方级别的，这意味着每次迭代后近似解的精确数位会翻倍。
关于牛顿迭代法的收敛性证明可参考 [citizendium - Newton method Convergence analysis](http://en.citizendium.org/wiki/Newton%27s_method#Convergence_analysis)

当然牛顿迭代法也同样存在着缺陷，详情参考 [Xiaolin Wu - Roots of Equations 第 18 - 20 页分析](https://www.ece.mcmaster.ca/~xwu/part2.pdf)

## 求解平方根

我们尝试用牛顿迭代法求解平方根。设 $f(x)=x^2-n$，这个方程的近似解就是 $\sqrt{n}$ 的近似值。于是我们得到

$$
x_{i+1}=x_i-\frac{x_i^2-n}{2x_i}=\frac{x_i+\frac{n}{x_i}}{2}
$$

在实现的时候注意设置合适的精度。代码如下

```cpp
// C++ Version
double sqrt_newton(double n) {
    const double eps = 1E-15;
    double x = 1;
    while (true) {
        double nx = (x + n / x) / 2;
        if (abs(x - nx) < eps) break;
        x = nx;
    }
    return x;
}
```

```python
# Python Version
def sqrt_newton(n):
    eps = 1e-15
    x = 1
    while True:
        nx = (x + n / x) / 2
        if abs(x - nx) < eps:
            break
        x = nx
    return x
```

## 求解整数平方根

尽管我们可以调用 `sqrt()` 函数来获取平方根的值，但这里还是讲一下牛顿迭代法的变种算法，用于求不等式 $x^2\le n$ 的最大整数解。我们仍然考虑一个类似于牛顿迭代的过程，但需要在边界条件上稍作修改。如果 $x$ 在迭代的过程中上一次迭代值得近似解变小，而这一次迭代使得近似解变大，那么我们就不进行这次迭代，退出循环。

```cpp
// C++ Version
int isqrt_newton(int n) {
    int x = 1;
    bool decreased = false;
    for (;;) {
        int nx = (x + n / x) >> 1;
        if (x == nx || (nx > x && decreased)) break;
        decreased = nx < x;
        x = nx;
    }
    return x;
}
```

```python
# Python Version
def isqrt_newton(n):
    x = 1
    decreased = False
    while True:
        nx = (x + n // x) // 2
        if x == nx or (nx > x and decreased):
            break
        decreased = nx < x
        x = nx
    return x
```

## 高精度平方根

最后考虑高精度的牛顿迭代法。迭代的方法是不变的，但这次我们需要关注初始时近似解的设置，即 $x_0$ 的值。由于需要应用高精度的数一般都非常大，因此不同的初始值对于算法效率的影响也很大。一个自然的想法就是考虑 $x_0=2^{\left\lfloor\frac{1}{2}\log_2n\right\rfloor}$，这样既可以快速计算出 $x_0$，又可以较为接近平方根的近似解。

给出 Java 代码的实现：

```java
public static BigInteger isqrtNewton(BigInteger n) {
    BigInteger a = BigInteger.ONE.shiftLeft(n.bitLength() / 2);
    boolean p_dec = false;
    for (;;) {
        BigInteger b = n.divide(a).add(a).shiftRight(1);
        if (a.compareTo(b) == 0 || a.compareTo(b) < 0 && p_dec)
            break;
        p_dec = a.compareTo(b) > 0;
        a = b;
    }
    return a;
}
```

实践效果：在 $n=10^{1000}$ 的时候该算法的运行时间是 60 ms，如果我们不优化 $x_0$ 的值，直接从 $x_0=1$ 开始迭代，那么运行时间将增加到 120 ms。

## 习题

- [UVa 10428 - The Roots](https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&category=16&page=show_problem&problem=1369)
-   [LeetCode 69. x 的平方根](https://leetcode.cn/problems/sqrtx/)


> [!NOTE] **[Luogu [NOIP2001 提高组] 一元三次方程求解](https://www.luogu.com.cn/problem/P1024)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 1. 对函数求导并求 $f'(x) = 0$ 的点，即【极点】
> 
> 2. 显然有两哥单峰极值
> 
> 3. 在极值划分的三个区间内牛顿迭代即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const double eps = 1e-4;

double a, b, c, d;

// f(x)
double f(double x) { return a * x * x * x + b * x * x + c * x + d; }

// f'(x)
double df(double x) { return 3 * a * x * x + 2 * b * x + c; }

// 牛顿迭代
double slove(double l, double r) {
    double x, x0 = (l + r) / 2;
    while (abs(x0 - x) > eps)
        x = x0 - f(x0) / df(x0), swap(x0, x);
    return x;
}

int main() {
    cin >> a >> b >> c >> d;

    double p = (-b - sqrt(b * b - 3 * a * c)) / (3 * a);
    double q = (-b + sqrt(b * b - 3 * a * c)) / (3 * a);

    printf("%.2lf %.2lf %.2lf\n", slove(-100, p), slove(p, q), slove(q, 100));

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

> [!NOTE] **[LeetCode 69. x 的平方根](https://leetcode.cn/problems/sqrtx/)**
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
// 牛顿迭代
class Solution {
public:
    int mySqrt(int x) {
        if (x == 0) return 0;
        double C = x, x0 = x;
        while (true) {
            double xi = 0.5 * (x0 + C / x0);
            if (fabs(x0 - xi) < 1e-7) {
                break;
            }
            x0 = xi;
        }
        return int(x0);
    }
};

// 二分略
```

##### **Python**

```python
"""
二分法：
当前数的平方都小于或者等于目标值时，就全部舍弃（因为我们要找的是第一个大于target的整数）
最后return的答案是，第一个大于target的整数减去1 即可。 
"""
class Solution:
    def mySqrt(self, x: int) -> int:
      	if x == 0 or x == 1:return x   # 特殊case判断
        l, r = 0, x
        while l < r:
            m = l + (r - l)//2
            if m * m <= x:
                l = m + 1 
            else:
                r = m 
        return l - 1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 367. 有效的完全平方数](https://leetcode.cn/problems/valid-perfect-square/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    bool isPerfectSquare(int num) {
        if (num < 2) return true;
        long long x = num / 2;
        while (x * x > num) x = (x + num / x) / 2;
        return x * x == num;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    bool isPerfectSquare(int num) {
        int l = 1, r = num;
        while (l < r) {
            int mid = l + 1ll + r >> 1;
            if (mid <= num / mid) l = mid;
            else r = mid - 1;
        }
        return r * r == num;
    }
};
```

##### **Python**

```python
# 也可以用普通的遍历一遍；这里用的是二分查找
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        l, r = 1, num
        while l < r:
            mid = (l + r) // 2
            if mid * mid < num:
                l = mid + 1
            else:
                r = mid
        return l * l == num
```

<!-- tabs:end -->
</details>

<br>

* * *