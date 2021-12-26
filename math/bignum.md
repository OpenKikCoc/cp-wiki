## 前言


高精度计算（Arbitrary-Precision Arithmetic），也被称作大整数（bignum）计算，运用了一些算法结构来支持更大整数间的运算（数字大小超过语言内建整型）。


> [!NOTE] **任务**
> 
> 输入：一个形如 `a <op> b` 的表达式。
> 
> - `a`、`b` 分别是长度不超过 $1000$ 的十进制非负整数；
> 
> - `<op>` 是一个字符（`+`、`-`、`*` 或 `/`），表示运算。
> 
> - 整数与运算符之间由一个空格分隔。
> 
> 输出：运算结果。
> 
> - 对于 `+`、`-`、`*` 运算，输出一行表示结果；
> 
> - 对于 `/` 运算，输出两行分别表示商和余数。
> 
> - 保证结果均为非负整数。



## 四则运算

### 加法


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 不压位写法
vector<int> add(vector<int> & A, vector<int> & B) {
    if (A.size() < B.size()) return add(B, A);
    
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); ++ i ) {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    if (t) C.push_back(t);
    return C;
}

int main() {
    string a, b;
    vector<int> A, B;
    cin >> a >> b;
    for (int i = a.size() - 1; i >= 0; -- i ) A.push_back(a[i] - '0');
    for (int i = b.size() - 1; i >= 0; -- i ) B.push_back(b[i] - '0');
    
    auto C = add(A, B);
    
    for (int i = C.size() - 1; i >= 0; -- i ) cout << C[i];
    cout << endl;
    return 0;
}
```

##### **C++ 压位**

```cpp
#include<bits/stdc++.h>
using namespace std;

// 压位写法
const int base = 1e9;

vector<int> add(vector<int> & A, vector<int> & B) {
    if (A.size() < B.size()) return add(B, A);
    
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); ++ i ) {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % base);  //
        t /= base;              //
    }
    if (t) C.push_back(t);
    return C;
}

int main() {
    string a, b;
    vector<int> A, B;
    cin >> a >> b;
    for (int i = a.size() - 1, s = 0, j = 0, t = 1; i >= 0; -- i ) {
        s += (a[i] - '0') * t;
        j ++ , t *= 10;
        if (j == 9 || i == 0) {
            A.push_back(s);
            s = j = 0;
            t = 1;
        }
    }
    for (int i = b.size() - 1, s = 0, j = 0, t = 1; i >= 0; -- i ) {
        s += (b[i] - '0') * t;
        j ++ , t *= 10;
        if (j == 9 || i == 0) {
            B.push_back(s);
            s = j = 0;
            t = 1;
        }
    }
    
    auto C = add(A, B);
    
    cout << C.back();
    for (int i = C.size() - 2; i >= 0; -- i ) printf("%09d", C[i]);
    cout << endl;
    return 0;
}
```

##### **Python**

```python
#1.大整数在代码里如何保存呢？把每一位存到一个数组中去，大整数的个位存在数组的第0位，以此类推。（因为有进位的原因，直接在数组后加一位就可以）
# ===> 从前向后相加。（浮点数的计算用的很少，不需要掌握）

#在python里，整数是不会溢出（python和java都有大整数），但是有的题目要求不能用；如果用的话，代码如下：
a = input()
b = input()
print(a + b)

#2. 整个运算的过程就是一个模拟人工加法的过程。===> 每一位的计算都需要用三个变量：A[i]+B[i]+上一位的进位t（初始值为0）
# 如果A[i] or B[i]不存在的话，就被当作是0；如果上一位没有进位的话，t也是为0；


def add(A,B):
    res = []
    i, t = 0, 0
    while i < len(A) or i < len(B) or t:
      #如果输入的是字符串，那这里就需要写成：a=int(A[i]) if i<len)A else 0
        a = A[i] if i < len(A) else 0
        b = B[i] if i < len(B) else 0
        t, n = divmod(a + b + t, 10)
        #注意：需要把数字类型转化为str类型
        res.append(str(n))
        i += 1
    return ''.join(res[::-1])

if __name__=='__main__':
    s1 = list(map(int, input()))
    s2 = list(map(int, input()))
    A = s1[::-1]
    B = s2[::-1]
    # print(''.join(list(map(str, add(A, B)))))
    print(add(A, B))
```

<!-- tabs:end -->
</details>

<br>

* * *


### 减法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

bool cmp(vector<int> & A, vector<int> & B) {
    if (A.size() != B.size()) return A.size() > B.size();
    
    for (int i = A.size() - 1; i >= 0; -- i )
        if (A[i] != B[i])
            return A[i] > B[i];
    return true;
}

vector<int> sub(vector<int> & A, vector<int> & B) {
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); ++ i ) {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }
    while (C.size() > 1 && C.back() == 0) C.pop_back(); // 前导0
    return C;
}

int main() {
    string a, b;
    vector<int> A, B;
    cin >> a >> b;
    for (int i = a.size() - 1; i >= 0; -- i ) A.push_back(a[i] - '0');
    for (int i = b.size() - 1; i >= 0; -- i ) B.push_back(b[i] - '0');
    
    vector<int> C;
    if (cmp(A, B)) C = sub(A, B);   // 不能用stl内置比较符号 因为比较时是下标逆序 而内置比较是正序
    else C = sub(B, A), cout << '-';
    
    for (int i = C.size() - 1; i >= 0; -- i ) cout << C[i];
    cout << endl;
    return 0;
}
```

##### **Python**

```python
def compare(A, B):
    if(len(A) != len(B)): return len(A) > len(B)
    for i in range(len(A) - 1, -1, -1):
        if(A[i] != B[i]): return A[i] > B[i]
    return True;

def sub(A, B):
    C = list()
    t = 0
    i = 0
    while(i < len(A)):
        t = A[i] - t
        if(i < len(B)): t -= B[i]
        C.append((t + 10) % 10)
        if(t < 0): t = 1
        else: t = 0
        i += 1
    i = 0
    C = C[::-1]
    while len(C) > 1 and C[0] == 0:
        C.pop(0)
    return C
    
if __name__=='__main__':
    a = list(map(int, input()))
    b = list(map(int, input()))
    #倒序写法一利用python特性
    A = a[::-1]
    B = b[::-1]

    if(compare(A, B)):
        res = sub(A, B)
        print(''.join(map(str, res)))
    else:
        res = sub(B, A)
        print('-' + "".join(map(str, res)))

```

<!-- tabs:end -->
</details>

<br>

* * *

### 乘法

#### 高精度—单精度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

vector<int> mul(vector<int> & A, int b) {
    vector<int> C;
    for (int i = 0, t = 0; i < A.size() || t; ++ i ) {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}

int main() {
    string a;
    int b;
    cin >> a >> b;
    
    vector<int> A;
    for (int i = a.size() - 1; i >= 0; -- i ) A.push_back(a[i] - '0');
    
    auto C = mul(A, b);
    
    for (int i = C.size() - 1; i >= 0; -- i ) cout << C[i];
    cout << endl;
    return 0;
}
```

##### **Python**

```python
# 把B看成一个整体 对每一个A数组中的数相乘
if __name__ == '__main__':
    A = list(map(int,input()))
    A.reverse()
    B = int(input())  #B的数据范围较小，可以直 接用int类型

    res = []
    t = i = 0
    while i < len(A) or t:
        if i < len(A):
            t += A[i] * B
        res.append(t%10)
        t//=10
        i+=1
    str1 = ''.join([str(x) for x in res[::-1]])
    s = str1[:-1].lstrip('0') + str1[-1]
    print(s)
```

<!-- tabs:end -->
</details>

<br>

* * *

#### 高精度—高精度

TODO@binacs

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

* * *

### 除法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

vector<int> div(vector<int> & A, int b, int & r) {
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; -- i ) {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}

int main() {
    string a;
    vector<int> A;
    int B;
    cin >> a >> B;
    for (int i = a.size() - 1; i >= 0; -- i ) A.push_back(a[i] - '0');
    
    int r;
    auto C = div(A, B, r);
    
    for (int i = C.size() - 1; i >= 0; -- i ) cout << C[i];
    cout << endl << r << endl;
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

## 压位高精度

在一般的高精度加法，减法，乘法运算中，我们都是将参与运算的数拆分成一个个单独的数码进行运算。

例如计算 $8192\times 42$ 时，如果按照高精度乘高精度的计算方式，我们实际上算的是 $(8000+100+90+2)\times(40+2)$。

在位数较多的时候，拆分出的数也很多，高精度运算的效率就会下降。

有没有办法作出一些优化呢？

注意到拆分数字的方式并不影响最终的结果，因此我们可以将若干个数码进行合并。

还是以上面这个例子为例，如果我们每两位拆分一个数，我们可以拆分成 $(8100+92)\times 42$。

这样的拆分不影响最终结果，但是因为拆分出的数字变少了，计算效率也就提升了。

从 [进位制](./base.md) 的角度理解这一过程，我们通过在较大的进位制（上面每两位拆分一个数，可以认为是在 $100$ 进制下进行运算）下进行运算，从而达到减少参与运算的数字的位数，提升运算效率的目的。

这就是 **压位高精度** 的思想。

下面我们给出压位高精度的加法代码，用于进一步阐述其实现方法：

> [!NOTE] **压位高精度加法参考实现**
```cpp
//这里的 a,b,c 数组均为 p 进制下的数
//最终输出答案时需要将数字转为十进制
void add(int a[], int b[], int c[]) {
    clear(c);

    for (int i = 0; i < LEN - 1; ++i) {
        c[i] += a[i] + b[i];
        if (c[i] >= p) {  //在普通高精度运算下，p=10
            c[i + 1] += 1;
            c[i] -= p;
        }
    }
}
```

### 压位高精下的高效竖式除法

在使用压位高精时，如果试商时仍然使用上文介绍的方法，由于试商次数会很多，计算常数会非常大。例如在万进制下，平均每个位需要试商 5000 次，这个巨大的常数是不可接受的。因此我们需要一个更高效的试商办法。

我们可以把 double 作为媒介。假设被除数有 4 位，是 $a_4,a_3,a_2,a_1$，除数有 3 位，是 $b_3,b_2,b_1$，那么我们只要试一位的商：使用 $base$ 进制，用式子 $\dfrac{a_4 base + a_3}{b_3 + b_2 base^{-1} + (b_1+1)base^{-2}}$ 来估商。而对于多个位的情况，就是一位的写法加个循环。由于除数使用 3 位的精度来参与估商，能保证估的商 q' 与实际商 q 的关系满足 $q-1 \le q' \le q$，这样每个位在最坏的情况下也只需要两次试商。但与此同时要求 $base^3$ 在 double 的有效精度内，即 $base^3 < 2^{53}$，所以在运用这个方法时建议不要超过 32768 进制，否则很容易因精度不足产生误差从而导致错误。

另外，由于估的商总是小于等于实际商，所以还有再进一步优化的空间。绝大多数情况下每个位只估商一次，这样在下一个位估商时，虽然得到的商有可能因为前一位的误差造成试商结果大于等于 base，但这没有关系，只要在最后再最后做统一进位便可。举个例子，假设 base 是 10，求 $395081/9876$，试商计算步骤如下：

1. 首先试商计算得到 $3950/988=3$，于是 $395081-(9876 \times 3 \times 10^1) = 98801$，这一步出现了误差，但不用管，继续下一步计算。
2. 对余数 98801 继续试商计算得到 $9880/988=10$，于是 $98801-(9876 \times 10 \times 10^0) = 41$，这就是最终余数。
3. 把试商过程的结果加起来并处理进位，即 $3 \times 10^1 + 10 \times 10^0 = 40$ 便是准确的商。

方法虽然看着简单，但具体实现上很容易进坑，所以以下提供一个经过多番验证确认没有问题的实现供大家参考，要注意的细节也写在注释当中。

> [!NOTE] **压位高精度高效竖式除法参考实现**
```cpp
// 完整模板和实现 https://baobaobear.github.io/post/20210228-bigint1/
// 对b乘以mul再左移offset的结果相减，为除法服务
BigIntSimple &sub_mul(const BigIntSimple &b, int mul, int offset) {
    if (mul == 0) return *this;
    int borrow = 0;
    // 与减法不同的是，borrow可能很大，不能使用减法的写法
    for (size_t i = 0; i < b.v.size(); ++i) {
        borrow += v[i + offset] - b.v[i] * mul - BIGINT_BASE + 1;
        v[i + offset] = borrow % BIGINT_BASE + BIGINT_BASE - 1;
        borrow /= BIGINT_BASE;
    }
    // 如果还有借位就继续处理
    for (size_t i = b.v.size(); borrow; ++i) {
        borrow += v[i + offset] - BIGINT_BASE + 1;
        v[i + offset] = borrow % BIGINT_BASE + BIGINT_BASE - 1;
        borrow /= BIGINT_BASE;
    }
    return *this;
}
BigIntSimple div_mod(const BigIntSimple &b, BigIntSimple &r) const {
    BigIntSimple d;
    r = *this;
    if (absless(b)) return d;
    d.v.resize(v.size() - b.v.size() + 1);
    // 提前算好除数的最高三位+1的倒数，若最高三位是a3,a2,a1
    // 那么db是a3+a2/base+(a1+1)/base^2的倒数，最后用乘法估商的每一位
    // 此法在BIGINT_BASE<=32768时可在int32范围内用
    // 但即使使用int64，那么也只有BIGINT_BASE<=131072时可用（受double的精度限制）
    // 能保证估计结果q'与实际结果q的关系满足q'<=q<=q'+1
    // 所以每一位的试商平均只需要一次，只要后面再统一处理进位即可
    // 如果要使用更大的base，那么需要更换其它试商方案
    double t = (b.get((unsigned)b.v.size() - 2) +
                (b.get((unsigned)b.v.size() - 3) + 1.0) / BIGINT_BASE);
    double db = 1.0 / (b.v.back() + t / BIGINT_BASE);
    for (size_t i = v.size() - 1, j = d.v.size() - 1; j <= v.size();) {
        int rm = r.get(i + 1) * BIGINT_BASE + r.get(i);
        int m = std::max((int)(db * rm), r.get(i + 1));
        r.sub_mul(b, m, j);
        d.v[j] += m;
        if (!r.get(i + 1))  // 检查最高位是否已为0，避免极端情况
            --i, --j;
    }
    r.trim();
    // 修正结果的个位
    int carry = 0;
    while (!r.absless(b)) {
        r.subtract(b);
        ++carry;
    }
    // 修正每一位的进位
    for (size_t i = 0; i < d.v.size(); ++i) {
        carry += d.v[i];
        d.v[i] = carry % BIGINT_BASE;
        carry /= BIGINT_BASE;
    }
    d.trim();
    d.sign = sign * b.sign;
    return d;
}

BigIntSimple operator/(const BigIntSimple &b) const {
    BigIntSimple r;
    return div_mod(b, r);
}

BigIntSimple operator%(const BigIntSimple &b) const {
    BigIntSimple r;
    div_mod(b, r);
    return r;
}
```

## Karatsuba 乘法

记高精度数字的位数为 $n$，那么高精度—高精度竖式乘法需要花费 $O(n^2)$ 的时间。本节介绍一个时间复杂度更为优秀的算法，由前苏联（俄罗斯）数学家 Anatoly Karatsuba 提出，是一种分治算法。

考虑两个十进制大整数 $x$ 和 $y$，均包含 $n$ 个数码（可以有前导零）。任取 $0 < m < n$，记

$$
\begin{aligned}
x &= x_1 \cdot 10^m + x_0, \\
y &= y_1 \cdot 10^m + y_0, \\
x \cdot y &= z_2 \cdot 10^{2m} + z_1 \cdot 10^m + z_0,
\end{aligned}
$$

其中 $x_0, y_0, z_0, z_1 < 10^m$。可得

$$
\begin{aligned}
z_2 &= x_1 \cdot y_1, \\
z_1 &= x_1 \cdot y_0 + x_0 \cdot y_1, \\
z_0 &= x_0 \cdot y_0.
\end{aligned}
$$

观察知

$$
z_1 = (x_1 + x_0) \cdot (y_1 + y_0) - z_2 - z_0,
$$

于是要计算 $z_1$，只需计算 $(x_1 + x_0) \cdot (y_1 + y_0)$，再与 $z_0$、$z_2$ 相减即可。

上式实际上是 Karatsuba 算法的核心，它将长度为 $n$ 的乘法问题转化为了 $3$ 个长度更小的子问题。若令 $m = \left\lceil \dfrac n 2 \right\rceil$，记 Karatsuba 算法计算两个 $n$ 位整数乘法的耗时为 $T(n)$，则有 $T(n) = 3 \cdot T \left(\left\lceil \dfrac n 2 \right\rceil\right) + O(n)$，由主定理可得 $T(n) = \Theta(n^{\log_2 3}) \approx \Theta(n^{1.585})$。

整个过程可以递归实现。为清晰起见，下面的代码通过 Karatsuba 算法实现了多项式乘法，最后再处理所有的进位问题。

> [!TIP] **karatsuba_mulc.cpp**
```cpp
int *karatsuba_polymul(int n, int *a, int *b) {
    if (n <= 32) {
        // 规模较小时直接计算，避免继续递归带来的效率损失
        int *r = new int[n * 2 + 1]();
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j) r[i + j] += a[i] * b[j];
        return r;
    }

    int m = n / 2 + 1;
    int *r = new int[m * 4 + 1]();
    int *z0, *z1, *z2;

    z0 = karatsuba_polymul(m - 1, a, b);
    z2 = karatsuba_polymul(n - m, a + m, b + m);

    // 计算 z1
    // 临时更改，计算完毕后恢复
    for (int i = 0; i + m <= n; ++i) a[i] += a[i + m];
    for (int i = 0; i + m <= n; ++i) b[i] += b[i + m];
    z1 = karatsuba_polymul(m - 1, a, b);
    for (int i = 0; i + m <= n; ++i) a[i] -= a[i + m];
    for (int i = 0; i + m <= n; ++i) b[i] -= b[i + m];
    for (int i = 0; i <= (m - 1) * 2; ++i) z1[i] -= z0[i];
    for (int i = 0; i <= (n - m) * 2; ++i) z1[i] -= z2[i];

    // 由 z0、z1、z2 组合获得结果
    for (int i = 0; i <= (m - 1) * 2; ++i) r[i] += z0[i];
    for (int i = 0; i <= (m - 1) * 2; ++i) r[i + m] += z1[i];
    for (int i = 0; i <= (n - m) * 2; ++i) r[i + m * 2] += z2[i];

    delete[] z0;
    delete[] z1;
    delete[] z2;
    return r;
}

void karatsuba_mul(int a[], int b[], int c[]) {
    int *r = karatsuba_polymul(LEN - 1, a, b);
    memcpy(c, r, sizeof(int) * LEN);
    for (int i = 0; i < LEN - 1; ++i)
        if (c[i] >= 10) {
            c[i + 1] += c[i] / 10;
            c[i] %= 10;
        }
    delete[] r;
}
```

但是这样的实现存在一个问题：在 $b$ 进制下，多项式的每一个系数都有可能达到 $n \cdot b^2$ 量级，在压位高精度实现中可能造成整数溢出；而若在多项式乘法的过程中处理进位问题，则 $x_1 + x_0$ 与 $y_1 + y_0$ 的结果可能达到 $2 \cdot b^m$，增加一个位（如果采用 $x_1 - x_0$ 的计算方式，则不得不特殊处理负数的情况）。因此，需要依照实际的应用场景来决定采用何种实现方式。

## 习题

- [NOIP 2012 国王游戏](https://loj.ac/problem/2603)
- [SPOJ - Fast Multiplication](http://www.spoj.com/problems/MUL/en/)
- [SPOJ - GCD2](http://www.spoj.com/problems/GCD2/)
- [UVA - Division](https://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=1024)
- [UVA - Fibonacci Freeze](https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=436)
- [Codeforces - Notepad](http://codeforces.com/contest/17/problem/D)
