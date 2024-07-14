

格雷码是一个二进制数系，其中两个相邻数的二进制位只有一位不同。举个例子，$3$ 位二进制数的格雷码序列为

$$
000,001,011,010,110,111,101,100
$$

注意序列的下标我们以 $0$ 为起点，也就是说 $G(0)=000,G(4)=110$。

格雷码由贝尔实验室的 Frank Gray 于 1940 年代提出，并于 1953 年获得专利。

## 构造格雷码（变换）

格雷码的构造方法很多。我们首先介绍手动构造方法，然后会给出构造的代码以及正确性证明。

### 手动构造

$k$ 位的格雷码可以通过以下方法构造。我们从全 $0$ 格雷码开始，按照下面策略：

1. 翻转最低位得到下一个格雷码，（例如 $000\to 001$）；
2. 把最右边的 $1$ 的左边的位翻转得到下一个格雷码，（例如 $001\to 011$）；

交替按照上述策略生成 $2^k-1$ 次，可得到 $k$ 位的格雷码序列。

### 镜像构造

$k$ 位的格雷码可以从 $k-1$ 位的格雷码以上下镜射后加上新位的方式快速得到，如下图：

$$
\begin{matrix}
k=1\\
0\\ 1\\\\\\\\\\\\\\
\end{matrix}
\to \begin{matrix}\\
\color{Red}0\\\color{Red}1\\\color{Blue}1\\\color{Blue}0\\\\\\\\\\
\end{matrix}
\to \begin{matrix}
k=2\\
{\color{Red}0}0\\{\color{Red}0}1\\{\color{Blue}1}1\\{\color{Blue}1}0\\\\\\\\\\
\end{matrix}
\to \begin{matrix}\\
\color{Red}00\\\color{Red}01\\\color{Red}11\\\color{Red}10\\\color{Blue}10\\\color{Blue}11\\\color{Blue}01\\\color{Blue}00
\end{matrix}
\to \begin{matrix}
k=3\\
{\color{Red}0}00\\{\color{Red}0}01\\{\color{Red}0}11\\{\color{Red}0}10\\{\color{Blue}1}10\\{\color{Blue}1}11\\{\color{Blue}1}01\\{\color{Blue}1}00
\end{matrix}
$$

### 计算方法

我们观察一下 $n$ 的二进制和 $G(n)$。可以发现，如果 $G(n)$ 的二进制第 $i$ 位为 $1$，仅当 $n$ 的二进制第 $i$ 位为 $1$，第 $i+1$ 位为 $0$ 或者第 $i$ 位为 $0$，第 $i+1$ 位为 $1$。于是我们可以当成一个异或的运算，即

$$
G(n)=n\oplus \left\lfloor\frac{n}{2}\right\rfloor
$$

```cpp
int g(int n) { return n ^ (n >> 1); }
```

### 正确性证明

接下来我们证明一下，按照上述公式生成的格雷码序列，相邻两个格雷码的二进制位有且仅有一位不同。

我们考虑 $n$ 和 $n+1$ 的区别。把 $n$ 加 $1$，相当于把 $n$ 的二进制下末位的连续的 $1$ 全部变成取反，然后把最低位的 $0$ 变成 $1$。我们这样表示 $n$ 和 $n+1$ 的二进制位：

$$
\begin{array}{rll}
(n)_2&=&\cdots0\underbrace{11\cdots11}_{k\text{个}}\\
(n+1)_2&=&\cdots1\underbrace{00\cdots00}_{k\text{个}}
\end{array}
$$

于是我们在计算 $g(n)$ 和 $g(n+1)$ 的时侯，后 $k$ 位都会变成 $\displaystyle\underbrace{100\cdots00}_{k\text{个}}$ 的形式，而第 $k+1$ 位是不同的，因为 $n$ 和 $n+1$ 除了后 $k+1$ 位，其他位都是相同的。因此第 $k+1$ 位要么同时异或 $1$，要么同时异或 $0$。两种情况，第 $k+1$ 位都是不同的。而除了后 $k+1$ 位以外的二进制位也是做相同的异或运算，结果是相同的。

证毕。

## 通过格雷码构造原数（逆变换）

接下来我们考虑格雷码的逆变换，即给你一个格雷码 $g$，要求你找到原数 $n$。我们考虑从二进制最高位遍历到最低位（最低位下标为 $1$，即个位；最高位下标为 $k$）。则 $n$ 的二进制第 $i$ 位与 $g$ 的二进制第 $i$ 位 $g_i$ 的关系如下：

$$
\begin{array}{rll}
n_k &= g_k \\
n_{k-1} &= g_{k-1} \oplus n_k &= g_k \oplus g_{k-1} \\
n_{k-2} &= g_{k-2} \oplus n_{k-1} &= g_k \oplus g_{k-1} \oplus g_{k-2} \\
n_{k-3} &= g_{k-3} \oplus n_{k-2} &= g_k \oplus g_{k-1} \oplus g_{k-2} \oplus g_{k-3} \\
&\vdots\\
n_{k-i} &=\displaystyle\bigoplus_{j=0}^ig_{k-j}
\end{array}
$$

```cpp
int rev_g(int g) {
    int n = 0;
    for (; g; g >>= 1) n ^= g;
    return n;
}
```

## 实际应用

格雷码有一些十分有用的应用，有些应用让人意想不到：

- $k$ 位二进制数的格雷码序列可以当作 $k$ 维空间中的一个超立方体（二维里的正方形，一维里的单位向量）顶点的哈密尔顿回路，其中格雷码的每一位代表一个维度的坐标。

- 格雷码被用于最小化数字模拟转换器（比如传感器）的信号传输中出现的错误，因为它每次只改变一个位。

-   格雷码可以用来解决汉诺塔的问题。

    设盘的数量为 $n$。我们从 $n$ 位全 $0$ 的格雷码 $G(0)$ 开始，依次移向下一个格雷码（$G(i)$ 移向 $G(i+1)$）。当前格雷码的二进制第 $i$ 位表示从小到大第 $i$ 个盘子。

    由于每一次只有一个二进制位会改变，因此当第 $i$ 位改变时，我们移动第 $i$ 个盘子。在移动盘子的过程中，除了最小的盘子，其他任意一个盘子在移动的时侯，只能有一个放置选择。在移动第一个盘子的时侯，我们总是有两个放置选择。于是我们的策略如下：

    如果 $n$ 是一个奇数，那么盘子的移动路径为 $f\to t\to r\to f\to t\to r\to\cdots$，其中 $f$ 是最开始的柱子，$t$ 是最终我们把所有盘子放到的柱子，$r$ 是中间的柱子。

    如果 $n$ 是偶数：$f \to r \to t \to f \to r \to t \to \cdots$

- 格雷码也在遗传算法理论中得到应用。

## 习题

- [CSP S2 2019 D1T1](https://www.luogu.com.cn/problem/P5657) Difficulty: easy

- [SGU #249 Matrix](http://codeforces.com/problemsets/acmsguru/problem/99999/249) Difficulty: medium

> [!NOTE] **[LeetCode 89. 格雷编码](https://leetcode.cn/problems/gray-code/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> ![acwing题解](https://github.com/OpenKikCoc/LeetCode/raw/master/0001-0100/0089/acwing-lc-89.jpeg)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    vector<int> grayCode(int n) {
        int top = pow(2, n);
        vector<int> res;
        res.push_back(0);
        for (int i = 1; i <= n; i ++ ){
            int e = 1 << (i - 1);                        // i - 1位结果前增加一位1
            for (int j = res.size() - 1; j >= 0; j -- )  // 镜像排列
                res.push_back(e + res[j]);
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    vector<int> grayCode(int n) {
        vector<int> res(1, 0);
        while (n -- ) {
            for (int i = res.size() - 1; i >= 0; i -- ) {
                res[i] *= 2;
                res.push_back(res[i] + 1);
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# 有规律的一道题！
# 第k个格雷编码 是 k-1个格雷编码 进行对称翻转，然后前半部分后面补0，后半部分后面补1

class Solution:
    def grayCode(self, n: int) -> List[int]:
        res = [0]
        while n:
            for i in range(len(res) - 1, -1, -1):
                res[i] *= 2  # 前半部分 补0 
                res.append(res[i] + 1) #后半部分 补1，就是在前半部分的基础上加上1就可
            n -= 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1238. 循环码排列](https://leetcode.cn/problems/circular-permutation-in-binary-representation/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二进制码转格雷码：`G = B ^ B >> 1` ；
> 
> 格雷码转二进制码：`B = ^(G >> i), i = 0 .. n - 1` , n 为格雷码二进制位数。
> 
> 题解区有先生成 n 位格雷码再旋转数组的操作 也可 TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准 trick**

```cpp
class Solution {
public:
    vector<int> circularPermutation(int n, int start) {
        vector<int> res(1 << n);
        for (int i = 0; i < 1 << n; ++ i )
            // i ^ (i >> 1) 得到格雷码
            res[i] = i ^ (i >> 1) ^ start;
        return res;
    }
};
```

##### **C++ 构造坐标偏移**

```cpp
class Solution {
public:
    vector<int> circularPermutation(int n, int start) {
        vector<int> res = {start};
        int b = start;
        while (start >>= 1) b ^= start;
        n = (1 << n) - 1;
        for (int i = 1; i <= n; ++i)
            res.push_back(b + i & n ^ (b + i & n) >> 1);
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

> [!NOTE] **[LeetCode 1611. 使整数变为 0 的最少操作次数](https://leetcode.cn/problems/minimum-one-bit-operations-to-make-integers-zero/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 递归TODO
> 
> 题目要求的处理规则其实对应格雷码解码，故直接向零方向解码即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
 * 其实这种变为全0的操作就是格雷码为n的解码
 * [格雷码百度百科](https://baike.baidu.com/item/%E6%A0%BC%E9%9B%B7%E7%A0%81)
 * 格雷码→二进制码（解码）：
 * 从左边第二位起，将每位与左边一位解码后的值异或，作为该位解码后的值（最左边一位依然不变）。
 * 依次异或，直到最低位。依次异或转换后的值（二进制数）就是格雷码转换后二进制码的值。
 * eg:n=1110
   1. n的左边第二位：1，与前一位已经解码：1，进行异或，1^1=0，所以数字变成10xx
   2. n的左边第三位：1，与前一位已经解码：0，进行异或，1^0=1，所以数字变成101x
   3. n的左边第四位：0，与前一位已经解码：1，进行异或，0^1=1，所以数字变成1011
   所以答案就是1011B=11D
 */
class Solution {
public:
    int minimumOneBitOperations(int n) {
        int res = 0;
        while (n) {
            res ^= n;
            n /= 2;
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