## 与、或、异或

这三者都是两数间的运算，因此在这里一起讲解。

它们都是将两个整数作为二进制数，对二进制表示中的每一位逐一运算。

| 运算 | 运算符 | 数学符号表示                   |                 解释                  |
| ---- | :----: | ------------------------------ | :-----------------------------------: |
| 与   |  `&`   | $\&$、$\operatorname{and}$     |   只有两个对应位都为 $1$ 时才为 $1$   |
| 或   |  `\|`  | $\mid$、$\operatorname{or}$    | 只要两个对应位中有一个 $1$ 时就为 $1$ |
| 异或 |  `^`   | $\oplus$、$\operatorname{xor}$ |     只有两个对应位不同时才为 $1$      |

注意区分逻辑与（对应的数学符号为 $\wedge$）和按位与、逻辑或（$\vee$）和按位或的区别。

## 取反

取反暂无默认的数学符号表示，其对应的运算符为 `~`。它的作用是把 $num$ 的二进制补码中的 $0$ 和 $1$ 全部取反（$0$ 变为 $1$，$1$ 变为 $0$）。有符号整数的符号位在 `~` 运算中同样会取反。

补码：在二进制表示下，正数和 $0$ 的补码为其本身，负数的补码是将其对应正数按位取反后加一。

举例（有符号整数）：

$$
\begin{aligned}
5&=(00000101)_2\\
\text{~}5&=(11111010)_2=-6\\
-5\text{ 的补码}&=(11111011)_2\\
\text{~}(-5)&=(00000100)_2=4
\end{aligned}
$$

## 左移和右移

 `num << i` 表示将 $num$ 的二进制表示向左移动 $i$ 位所得的值。

 `num >> i` 表示将 $num$ 的二进制表示向右移动 $i$ 位所得的值。

举例：

$$
\begin{aligned}
11&=(00001011)_2\\
11<<3&=(01011000)_2=88\\
11>>2&=(00000010)_2=2
\end{aligned}
$$

移位运算中如果出现如下情况，则其行为未定义：

1. 右操作数（即移位数）为负值；
2. 右操作数大于等于左操作数的位数；

例如，对于 `int` 类型的变量 `a` ， `a << -1` 和 `a << 32` 都是未定义的。

对于左移操作，需要确保移位后的结果能被原数的类型容纳，否则行为也是未定义的。[^note1]对一个负数执行左移操作也未定义。[^note2]

对于右移操作，右侧多余的位将会被舍弃，而左侧较为复杂：对于无符号数，会在左侧补 $0$；而对于有符号数，则会用最高位的数（其实就是符号位，非负数为 $0$，负数为 $1$）补齐。[^note3]

## 复合赋值位运算符

和 `+=` , `-=` 等运算符类似，位运算也有复合赋值运算符： `&=` , `|=` , `^=` , `<<=` , `>>=` 。

## 关于优先级

位运算的优先级低于算术运算符（除了取反），而按位与、按位或及异或低于比较运算符，在必要时添加括号。

## 位运算的应用

位运算一般有三种作用：

1. 高效地进行某些运算，代替其它低效的方式。

2. 表示集合。（常用于 [状压 DP](dp/state.md) 。）

3. 题目本来就要求进行位运算。

需要注意的是，用位运算代替其它运算方式（即第一种应用）在很多时候并不能带来太大的优化，反而会使代码变得复杂，使用时需要斟酌。（但像“乘 2 的非负整数次幂”和“除以 2 的非负整数次幂”就最好使用位运算，因为此时使用位运算可以优化复杂度。）

### 有关 2 的幂的应用

由于位运算针对的是变量的二进制位，因此可以推广出许多与 2 的整数次幂有关的应用。

将一个数乘（除） 2 的非负整数次幂：

```cpp
// C++ Version
int mulPowerOfTwo(int n, int m) {  // 计算 n*(2^m)
    return n << m;
}
int divPowerOfTwo(int n, int m) {  // 计算 n/(2^m)
    return n >> m;
}
```

```python
# Python Version
def mulPowerOfTwo(n, m): # 计算 n*(2^m)
    return n << m
def divPowerOfTwo(n, m): # 计算 n/(2^m)
    return n >> m
```

> [!WARNING]
> 
> 我们平常写的除法是向 $0$ 取整，而这里的右移是向下取整（注意这里的区别），即当数大于等于 $0$ 时两种方法等价，当数小于 $0$ 时会有区别，如： `-1 / 2` 的值为 $0$ ，而 `-1 >> 1` 的值为 $-1$ 。

判断一个数是不是 $2$ 的非负整数次幂：

```cpp
// C++ Version
bool isPowerOfTwo(int n) { return n > 0 && (n & (n - 1)) == 0; }
```

```python
# Python Version
def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0
```

对 $2$ 的非负整数次幂取模：

```cpp
// C++ Version
int modPowerOfTwo(int x, int mod) { return x & (mod - 1); }
```

```python
# Python Version
def modPowerOfTwo(x, mod):
    return x & (mod - 1)
```

### 判断两非零数符号是否相同

```cpp
// C++ Version
bool isSameSign(int x, int y) {  // 有 0 的情况例外
    return (x ^ y) >= 0;
}
```

```python
# Python Version
# 有 0 的情况例外
def isSameSign(x, y):
    return (x ^ y) >= 0
```

### 模拟集合操作

一个数的二进制表示可以看作是一个集合（$0$ 表示不在集合中，$1$ 表示在集合中）。比如集合 $\{1,3,4,8\}$ ，可以表示成 $(100011010)_2$ 。而对应的位运算也就可以看作是对集合进行的操作。

| 操作   |    集合表示     |         位运算语句          |
| ------ | :-------------: | :-------------------------: |
| 交集   |   $a \cap b$    |           `a & b`           |
| 并集   |   $a \cup b$    |            `a|b`            |
| 补集   |    $\bar{a}$    | `~a` （全集为二进制都是 1） |
| 差集   | $a \setminus b$ |         `a & (~b)`          |
| 对称差 | $a\triangle b$  |           `a ^ b`           |

子集遍历：

```cpp
// 遍历 u 的非空子集
for (int s = u; s; s = (s - 1) & u) {
    // s 是 u 的一个非空子集
}
```

用这种方法可以在 $O(2^{\text{popcount}(u)})$ （ $\text{popcount}(u)$ 表示 $u$ 二进制中 1 的个数）的时间复杂度内遍历 $u$ 的子集，进而可以在 $O(3^n)$ 的时间复杂度内遍历大小为 $n$ 的集合的每个子集的子集。（复杂度为 $O(3^n)$ 是因为每个元素都有 不在大子集中/只在大子集中/同时在大小子集中 三种状态。）

## 内建函数

GCC 中还有一些用于位运算的内建函数：

1.  `int __builtin_ffs(int x)` ：返回 $x$ 的二进制末尾最后一个 $1$ 的位置，位置的编号从 $1$ 开始（最低位编号为 $1$ ）。当 $x$ 为 $0$ 时返回 $0$ 。

2.  `int __builtin_clz(unsigned int x)` ：返回 $x$ 的二进制的前导 $0$ 的个数。当 $x$ 为 $0$ 时，结果未定义。

3.  `int __builtin_ctz(unsigned int x)` ：返回 $x$ 的二进制末尾连续 $0$ 的个数。当 $x$ 为 $0$ 时，结果未定义。

4.  `int __builtin_clrsb(int x)` ：当 $x$ 的符号位为 $0$ 时返回 $x$ 的二进制的前导 $0$ 的个数减一，否则返回 $x$ 的二进制的前导 $1$ 的个数减一。

5.  `int __builtin_popcount(unsigned int x)` ：返回 $x$ 的二进制中 $1$ 的个数。

6.  `int __builtin_parity(unsigned int x)` ：判断 $x$ 的二进制中 $1$ 的个数的奇偶性。

这些函数都可以在函数名末尾添加 `l` 或 `ll` （如 `__builtin_popcountll` ）来使参数类型变为 ( `unsigned` ) `long` 或 ( `unsigned` ) `long long` （返回值仍然是 `int` 类型）。
例如，我们有时候希望求出一个数以二为底的对数，如果不考虑 `0` 的特殊情况，就相当于这个数二进制的位数 `-1` ，而一个数 `n` 的二进制表示的位数可以使用 `32-__builtin_clz(n)` 表示，因此 `31-__builtin_clz(n)` 就可以求出 `n` 以二为底的对数。

由于这些函数是内建函数，经过了编译器的高度优化，运行速度十分快（有些甚至只需要一条指令）。

## 更多位数

如果需要操作的集合非常大，可以使用 [bitset](lang/bitset.md) 。

## 题目推荐

[Luogu P1225 黑白棋游戏](https://www.luogu.com.cn/problem/P1225) 

## 参考资料与注释

1. 位运算技巧： <https://graphics.stanford.edu/~seander/bithacks.html> 
2. Other Builtins of GCC： <https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html> 

[^note1]: 适用于 C++14 以前的标准。在 C++14 和 C++17 标准中，若原值为带符号类型，且移位后的结果能被原类型的无符号版本容纳，则将该结果 [转换](lang/var.md#variable-conversion) 为相应的带符号值，否则行为未定义。在 C++20 标准中，规定了无论是带符号数还是无符号数，左移均直接舍弃移出结果类型的位。

[^note2]: 适用于 C++20 以前的标准。

[^note3]: 这种右移方式称为算术右移。在 C++20 以前的标准中，并没有规定带符号数右移运算的实现方式，大多数平台均采用算术右移。在 C++20 标准中，规定了带符号数右移运算是算术右移。

## 习题

### 基础应用

> [!NOTE] **[AcWing 801. 二进制中1的个数](https://www.acwing.com/problem/content/803/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, v;

int get(int v) {
    int cnt = 0;
    while (v) {
        v = v & (v - 1);
        ++cnt;
    }
    return cnt;
}

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; ++i) {
        scanf("%d", &v);
        printf("%d", get(v));
        if (i != n - 1)
            printf(" ");
        else
            printf("\n");
    }
}
```

##### **C++ 2**

```cpp
#include <iostream>

using namespace std;

int main() {
    int n;
    scanf("%d", &n);
    while (n -- ) {
        int x, s = 0;
        scanf("%d", &x);

        for (int i = x; i; i -= i & -i) s ++ ;

        printf("%d ", s);
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

> [!NOTE] **[Luogu 高低位交换](https://www.luogu.com.cn/problem/P1100)**
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
#include <bits/stdc++.h>
using namespace std;

using namespace std;
int main() {
    unsigned long long x;
    cin >> x;
    cout << ((x & 0x0000ffff) << 16 | (x & 0xffff0000) >> 16)
         << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 190. 颠倒二进制位](https://leetcode-cn.com/problems/reverse-bits/)**
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
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        int res = 0;
        for (int i = 0; i < 32; ++ i )
            res = (res << 1) + (n >> i & 1);
        return res;
    }
};
```

##### **Python**

```python
# 使用位运算 n >> i & 1 可以取出 n 的第 i 位二进制数。我们从小到大依次取出 n 的所有二进制位，然后逆序累加到另一个无符号整数中。
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(32):
            res = (res << 1) + (n >> i & 1)
            # res = (res * 2) + (n >> i & 1)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/)**
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
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int res = 0;
        while (n) n &= (n - 1), res ++ ;
        return res;
    }
};

class Solution {
public:
    int hammingWeight(uint32_t n) {
        int res = 0;
        while (n) n -= n & -n, res ++ ;
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        def lowbit(x):
            return x & (-x)

        cnt = 0
        if n < 0:
            n = n & (1 << 32 - 1)
        while n:
            n -= lowbit(n)
            cnt += 1
        return cnt
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 231. 2的幂](https://leetcode-cn.com/problems/power-of-two/)**
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
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n > 0 && !(n & (n - 1));
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

> [!NOTE] **[LeetCode 326. 3的幂](https://leetcode-cn.com/problems/power-of-three/)**
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
class Solution {
public:
    bool isPowerOfThree(int n) {
        return n > 0 && 1162261467 % n == 0;
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

> [!NOTE] **[LeetCode 342. 4的幂](https://leetcode-cn.com/problems/power-of-four/)**
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
class Solution {
public:
    bool isPowerOfFour(int num) {
        if (num <= 0) return false;
        // 判断是否是2的幂
        if (num & num - 1) return false;
        // 与运算之后是本身 则是4的幂
        if ((num & 0x55555555) == num) return true;
        return false;
    }
};

// yxc
// n 是4的整数次幂，等价于 n 是平方数，且 n 的质因子只有2。
// n 的质因子只有2，等价于 n 能整除 2^30 。
class Solution {
public:
    bool isPowerOfFour(int num) {
        if (num <= 0) return false;
        int t = sqrt(num);
        return t * t == num && ((1 << 30) % num) == 0;
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

> [!NOTE] **[LeetCode 693. 交替位二进制数](https://leetcode-cn.com/problems/binary-number-with-alternating-bits/)**
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
class Solution {
public:
    bool hasAlternatingBits(int n) {
        for (int i = 1; 1ll << i <= n; i ++ ) {
            int a = n >> i - 1 & 1;
            int b = n >> i & 1;
            if (a == b) return false;
        }
        return true;
    }
};
```

##### **C++ 旧**

```cpp
class Solution {
public:
    bool hasAlternatingBits(int n) {
        n = (n ^ (n >> 1));               // 若合法 经过本操作变为全1
        return (n & ((long)n + 1)) == 0;  // +1 首位为1后面全0
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

### 数值操作

> [!NOTE] **[LeetCode 371. 两整数之和](https://leetcode-cn.com/problems/sum-of-two-integers/)**
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
class Solution {
public:
    int getSum(int a, int b) {
        while (b) {
            int t = a ^ b;
            // 处理负数 unsigned 形如 a = -1, b = 1
            int carry = (unsigned)(a & b) << 1;
            a = t, b = carry;
        }
        return a;
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

### 位维护状态

> [!NOTE] **[LeetCode 289. 生命游戏](https://leetcode-cn.com/problems/game-of-life/)**
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
class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) {
        int dx[] = {-1,  0,  1, -1, 1, -1, 0, 1};
        int dy[] = {-1, -1, -1,  0, 0,  1, 1, 1};

        for (int i = 0; i < board.size(); ++ i) {
            for (int j = 0 ; j < board[0].size(); j++) {
                int sum = 0;
                for (int k = 0; k < 8; k++) {
                    int nx = i + dx[k], ny = j + dy[k];
                    if (nx >= 0 && nx < board.size() && ny >= 0 && ny < board[0].size())
                        sum += (board[nx][ny]&1);   // 只累加最低位
                }
                if (board[i][j] == 1) {
                    if (sum == 2 || sum == 3)
                        board[i][j] |= 2;   // 使用第二个bit标记是否存活
                } else {
                    if (sum == 3)
                        board[i][j] |= 2;   // 使用第二个bit标记是否存活
                }
            }
        }
        for (int i = 0; i < board.size(); ++ i )
            for (int j = 0; j < board[i].size(); ++ j )
                board[i][j] >>= 1;          //右移一位，用第二bit覆盖第一个bit。
    }
};
```

##### **Python**

```python
"""
(原地算法+位运算)
如何做到不用额外的空间，且把所有位置细胞的状态同时改变呢？因为想到只有0和1两个状态，可以用二进制中的第二位来存储变化后的状态。
0000：一开始是死细胞，后来还是死细胞
0101：一开始是活细胞，后来变成死细胞
1010：一开始是死细胞，后来变成活细胞
1111：一开始是活细胞，后来还是活细胞
最后把第二位全部右移一位就是结果数组了

"""
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])

        for i in range(m):
            for j in range(n):
                live = 0
                for x in range(max(0, i-1), min(i+1, m-1)+1):
                    for y in range(max(0, j-1), min(j+1, n-1)+1):
                        if i == x and j == y:
                            continue
                        if board[x][y] & 1:
                            live += 1

                next = 0
                if board[i][j] & 1:
                    if 2 <= live <= 3:
                        next = 1
                else:
                    if live == 3:
                        next = 1
                board[i][j] |= next << 1

        for i in range(m):
            for j in range(n):
                board[i][j] = board[i][j] >> 1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1178. 猜字谜](https://leetcode-cn.com/problems/number-of-valid-words-for-each-puzzle/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 位运算即可 注意细节
> 
> 需要枚举子集

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    unordered_map<int, int> hash;
    vector<int> findNumOfValidWords(vector<string>& words, vector<string>& puzzles) {
        int n = words.size();
        for (int i = 0; i < n; ++ i ) {
            int v = 0;
            for (auto & c : words[i])
                v |= 1 << (c - 'a');
            hash[v] ++ ;
        }
        
        vector<int> res;
        for (auto & p : puzzles) {
            int base = 1 << (p[0] - 'a');
            int v = 0, cnt = hash.count(base) ? hash[base] : 0;
            for (int i = 1; i < 7; ++ i )
                v |= 1 << (p[i] - 'a');
            
            for (int i = v; i; i = i - 1 & v)
                if (hash.count(i | base))
                    cnt += hash[i | base];
            
            res.push_back(cnt);
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

> [!NOTE] **[LeetCode 1371. 每个元音包含偶数次的最长子字符串](https://leetcode-cn.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 奇偶次数为一个状态 状态dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findTheLongestSubstring(string s) {
        vector<int> last(32, -2);
        int v = 0, n = s.size();
        int res = 0;
        last[0] = -1;
        for (int i = 0; i < n; ++i) {
            if (s[i] == 'a')
                v ^= (1 << 0);
            else if (s[i] == 'e')
                v ^= (1 << 1);
            else if (s[i] == 'i')
                v ^= (1 << 2);
            else if (s[i] == 'o')
                v ^= (1 << 3);
            else if (s[i] == 'u')
                v ^= (1 << 4);
            if (last[v] == -2)
                last[v] = i;
            else
                res = max(res, i - last[v]);
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

> [!NOTE] **[LeetCode 1386. 安排电影院座位](https://leetcode-cn.com/problems/cinema-seat-allocation/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 2-9位 连续4个0的情况统计

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    // 2-5 4-7 6-9
    int maxNumberOfFamilies(int n, vector<vector<int>>& reservedSeats) {
        int l = 0b11110000;
        int m = 0b11000011;
        int r = 0b00001111;

        unordered_map<int, int> occupied;
        for (auto v : reservedSeats)
            if (v[1] > 1 && v[1] < 10) occupied[v[0]] |= 1 << (v[1] - 2);

        int res = (n - occupied.size()) * 2;
        // cout <<res<<endl;
        for (auto v : occupied) {
            if (((v.second | l) == l) || ((v.second | r) == r) ||
                ((v.second | m) == m))
                ++res;
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int maxNumberOfFamilies(int n, vector<vector<int>>& re) {
        unordered_map<int, unordered_set<int>> s;
        int cnt = 0;
        bool f = 0;
        for (auto it : re) { s[it[0]].insert(it[1]); }
        for (auto it : s) {
            f = 0;
            if (!it.second.count(2) && !it.second.count(3) &&
                !it.second.count(4) && !it.second.count(5))
                ++cnt, f = 1;
            if (!it.second.count(6) && !it.second.count(7) &&
                !it.second.count(8) && !it.second.count(9))
                ++cnt, f = 1;
            if (!it.second.count(4) && !it.second.count(5) &&
                !it.second.count(6) && !it.second.count(7) && !f)
                ++cnt;
        }
        return cnt + 2 * (n - s.size());
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

> [!NOTE] **[LeetCode 1442. 形成两个异或相等数组的三元组数目](https://leetcode-cn.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 前缀异或数组 然后遍历即可
>
> 评论区有O(n^2)解法 本质是只要固定了左右两端 则这段内部如何划分k都是一样的
>
> >   a = arr[i] ^ arr[i + 1] ^ ... ^ arr[j - 1],
> >
> >   b = arr[j] ^ arr[j + 1] ^ ... ^ arr[k],
> >
> >   那么根据位运算的规则， a^b=arr[i]^ arr[i + 1] ^ ... ^ arr[k]，即i到k。 
> > 
> >   此外，根据位运算，如果 a^b==0, 那么 a==b，这是逆否命题，即反推也成立。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ O(n^3)**

```cpp
class Solution {
public:
    int countTriplets(vector<int>& arr) {
        int n = arr.size();
        if (!n) return 0;
        vector<int> xorv(n + 1);
        // xorv[0] = arr[0];
        for (int i = 1; i <= n; ++i) xorv[i] = xorv[i - 1] ^ arr[i - 1];
        int res = 0;
        for (int i = 1; i < n; ++i)
            for (int j = i + 1; j <= n; ++j)
                for (int k = j; k <= n; ++k)
                    if ((xorv[j - 1] ^ xorv[i - 1]) == (xorv[k] ^ xorv[j - 1]))
                        ++res;
        return res;
    }
};
```

##### **C++ O(n^2)**

```cpp
class Solution {
public:
    int countTriplets(vector<int>& arr) {
        int n = arr.size(), res = 0;
        for (int i = 0; i < n; ++i) {
            int t = arr[i];
            for (int j = i + 1; j < n; ++j) {
                t ^= arr[j];
                if (t == 0) { res += j - i; }
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

> [!NOTE] **[LeetCode 1461. 检查一个字符串是否包含所有长度为 K 的二进制子串](https://leetcode-cn.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> bit 维护窗口

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool f[1 << 20];
    bool hasAllCodes(string s, int k) {
        int limit = 1 << k, mask = limit - 1, now = 0;
        int ls = s.size();
        fill(f, f + limit, false);
        for (int i = 0; i < ls; i++) {
            now <<= 1;
            now |= s[i] - '0';
            now &= mask;
            if (i >= k - 1) { f[now] = true; }
        }
        for (int i = 0; i < limit; i++)
            if (!f[i]) return false;
        return true;
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

> [!NOTE] **[LeetCode 1542. 找出最长的超赞子字符串](https://leetcode-cn.com/problems/find-longest-awesome-substring/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 定义超赞子字符串：可以通过重排使其回文，也即出现奇数次的字母个数小于等于1。
> 
> 显然状态压缩 **记录截止目前位置每个数字的奇偶状态**。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int longestAwesome(string s) {
        int n = s.size(), res = 0;
        // 记录某状态第一次出现的位置 对于1～n 默认-1
        vector<int> m(1024, -1);
        m[0] = 0;
        int st = 0, l;

        for (int i = 1; i <= n; ++i) {
            st = st ^ (1 << (s[i - 1] - '0'));
            l = m[st];
            // 和出现过的状态相同的情况
            if (l != -1) res = max(res, i - l);
            // 检查只相差 1 位的情况 只差一位得到的状态需是全0或只有一个1
            for (int j = 0; j < 10; ++j) {
                int v = st ^ (1 << j);
                if (m[v] == -1) continue;
                res = max(res, i - m[v]);
            }
            if (l == -1) m[st] = i;
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

### 进阶

> [!NOTE] **[AcWing 90. 64位整数乘法](https://www.acwing.com/problem/content/92/)**
> 
> 题意: 慢速乘

> [!TIP] **思路**
> 
> 核心思想 不能用乘法 考虑加法
> 
> a * b % p  ----> (a + a + a + ... + a) % p 【b个a】
> 
> ```text
> 计算    a  p
>       2*a % p
>       4*a % p
>       ...
>    2^62*a % p
> ```
> 
> $logb$ 位 最多加 $logb$ 次

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

LL a, b, p;

LL qadd(LL a, LL b, LL p) {
    LL res = 0;
    while (b) {
        if (b & 1) res = (res + a) % p;
        a = (a + a) % p;
        b >>= 1;
    }
    return res;
}

int main() {
    scanf("%lld%lld%lld", &a, &b, &p);
    printf("%lld\n", qadd(a, b, p));
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

> [!NOTE] **[Luogu 开灯](https://www.luogu.com.cn/problem/P1161)**
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
#include <bits/stdc++.h>
using namespace std;

double a;
int n, t;

int main() {
    cin >> n;
    
    int res = 0;
    for (int i = 0; i < n; ++ i ) {
        cin >> a >> t;
        for (int j = 1; j <= t; ++ j )
            res ^= (int)(a * j);
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

> [!NOTE] **[LeetCode 458. 可怜的小猪](https://leetcode-cn.com/problems/poor-pigs/)**
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
class Solution {
public:
    int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
        // 轮次： minutesToTest / minutesToDie
        // 第 x 轮死或一直不死 则可表达如下 states 种状态
        int states = minutesToTest / minutesToDie + 1;
        // (k + 1) ^ x >= buckets
        // k+1 即为测试时间除以中毒检验时间再加一（之前说过可以通过排除法确定最后一列）
        // 把 buckets 转化为 k+1 位进制数
        return ceil(log(buckets) / log(states));
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

> [!NOTE] **[LeetCode 1734. 解码异或后的排列](https://leetcode-cn.com/problems/decode-xored-permutation/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单版:
> 
> - 扫一遍即可
> 
> 扩展:
> 
> - n 为奇数
> - n 不一定奇数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 简单版**

```cpp
class Solution {
public:
    vector<int> decode(vector<int>& encoded) {
        int m = encoded.size(), n = m + 1;
        int v1 = 0, v2 = 0;
        for (int i = 1; i <= n; ++ i ) v1 ^= i;
        for (int i = 0; i < m; ++ i ) if (i % 2 == 1) v2 ^= encoded[i];
        vector<int> res;
        v1 ^= v2;
        res.push_back(v1);
        for (auto & v : encoded) {
            v1 ^= v;
            res.push_back(v1);
        }
        return res;
    }
};
```

##### **C++ 奇数**

```cpp
class Solution {
public:
    vector<int> decode(vector<int>& b) {
        int sum = 0;
        int n = b.size() + 1;
        for (int i = 1; i <= n; i ++ ) sum ^= i;  // a1 ^ ... ^ an
        for (int i = b.size() - 1; i >= 0; i -= 2)
            sum ^= b[i];
        vector<int> a(1, sum);
        for (int i = 0; i < b.size(); i ++ )
            a.push_back(a.back() ^ b[i]);
        return a;
    }
};
```

##### **C++ 不一定奇数**

```cpp
int son[2100000][2];

class Solution {
public:
    int idx;

    void insert(int x) {
        int p = 0;
        for (int i = 20; i >= 0; i -- ) {
            int u = x >> i & 1;
            if (!son[p][u]) son[p][u] = ++ idx;
            p = son[p][u];
        }
    }

    int query(int x) {
        int res = 0, p = 0;
        for (int i = 20; i >= 0; i -- ) {
            int u = x >> i & 1;
            if (son[p][!u]) p = son[p][!u], res = res * 2 + 1;
            else p = son[p][u], res *= 2;
        }
        return res;
    }

    vector<int> decode(vector<int>& b) {
        int n = b.size() + 1;
        idx = 0;
        memset(son, 0, sizeof son);
        for (int i = 1; i < b.size(); i ++ ) b[i] ^= b[i - 1];
        unordered_set<int> hash;
        for (auto x: b) hash.insert(x), insert(x);

        vector<int> res;
        for (int i = 1; i <= n; i ++ ) {
            if (!hash.count(i)) {
                if (query(i) <= n) {
                    res.push_back(i);
                    for (int j = 0; j < b.size(); j ++ )
                        res.push_back(i ^ b[j]);
                    break;
                }
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

### 思想

> [!NOTE] **[LeetCode 201. 数字范围按位与](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO: 思考如果**按位或**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int rangeBitwiseAnd(int m, int n) {
        int cnt = 0;
        while (m != n) {
            m >>= 1;
            n >>= 1;
            cnt ++ ;
        }
        return m <<= cnt; 
    }
    
    int rangeBitwiseAnd(int m, int n) {
        while (m < n) {
            n = n & (n - 1);
        }
        return n;
    }
};
```

##### **Python**

```python
class Solution {
public:
    int rangeBitwiseAnd(int m, int n) {
        int res = 0;
        for (int i = 30; i >= 0; i -- ) {
            if ((m >> i & 1) != (n >> i & 1)) break;
            if (m >> i & 1) res += 1 << i;
        }
        return res;
    }
};
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1558. 得到目标数组的最少函数调用次数](https://leetcode-cn.com/problems/minimum-numbers-of-function-calls-to-make-target-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 类似快速幂的思想
> 
> 可以理解为，在全体不断乘二的过程中，选择是否给某一个数单独加一。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minOperations(vector<int>& nums) {
        int best = 0;
        int ans = 0;
        for (int num : nums) {
            ans += __builtin_popcount(num);
            best = max(best, num);
        }
        for (int i = 29; i >= 0; --i) {
            if (best & (1 << i)) {
                ans += i;
                break;
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

> [!NOTE] **[LeetCode 2527. 查询数组 Xor 美丽值](https://leetcode.cn/problems/find-xor-beauty-of-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 首先题目中 $i,j$ 具有很强的对称性，而 $i,j$ 互换不改变式子的取值，因此，在 $i,j$ 不相等的情况下，`((nums[i] | nums[j]) & nums[k])` 取值会和 `((nums[j] | nums[i]) & nums[k])` 一致，而相等的两个数异或值为 $0$，因此对于所有的三元组而言 $i,j$ 不同的项会相互之间抵消
>
> 只需要考虑 $i=j$ 的情况，此时 `((nums[i] | nums[j]) & nums[k]) = nums[i] & nums[k]`，$i,k$ 又具有了对称性，`nums[i] & nums[k]` 与 `nums[k] & nums[i]` 也发生了抵消，只需要考虑 $i=k$ 的情形
>
> 最终只需要考虑中数组中所有数的异或和即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int xorBeauty(vector<int>& nums) {
        int res = 0;
        for (auto x : nums)
            res ^= x;
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

### 经典位划分

> [!NOTE] **[LeetCode 136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)**
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
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (auto v : nums) res ^= v;
        return res;
    }
};
```

##### **Python**

```python
# 1 ^ 1 = 0; 0 ^ 0 = 0; 1 ^ 0 = 1; 0 ^ 1 = 1
# 两个相同的数字异或值为0 所以res初始化为0 对结果不会产生影响

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for x in nums:
            res ^= x 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 137. 只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)**
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
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            int p = 1 << i;
            int c = 0;
            for (auto v : nums)
                if (v & p)
                    ++ c ;
            if (c % 3)
                res |= p;
        }
        return res;
    }
};
```

##### **C++ trick**

```cpp
// 希望看到 1 的个数是模三余几
//
// 状态机模型
// https://www.acwing.com/video/2853/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int two = 0, one = 0;
        for (auto x: nums) {
            one = (one ^ x) & ~two;
            two = (two ^ x) & ~one;
        }
        return one;
    }
};
```

##### **Python**

```python
# 特别注意 负数的情况：Python是动态类型语言，在这种情况下其会将符号位置的1看成了值，而不是当作符号“负数”
# 【只有一个元素出现一次，其他元素都出现三次。】
# 【位运算】
# 方法比较跳跃性，需要多思考回顾
# 1. 建立一个长度为 32 的数组counts ，记录所有数字的各二进制位的 11 的出现次数。
# 2. 将 counts各元素对 33 求余，则结果为 “只出现一次的数字” 的各二进制位。
# 3. 利用 左移操作 和 或运算 ，可将 counts数组中各二进位的值恢复到数字 res 上，最后返回 res
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0 
        for i in range(32):
            p = 1 << i 
            c = 0 
            for x in nums:
                if x & p:
                    c += 1
            if c % 3:
                res |= p
            if i == 31:
                if c % 3 == 0:
                    flag = True 
                else:
                    flag = False
        return res if flag else ~(res ^ 0xffffffff)


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        c = [0] * 32
        for num in nums:
            for j in range(32):
                c[j] += num & 1
                num >>= 1
        res = 0 
        for i in range(32):
            res <<= 1 
            res |= c[31 - i] % 3
        return res if c[31] % 3 == 0 else ~(res ^ 0xffffffff) 
      # 由于 Python 的存储负数的特殊性，需要先将0 - 32位取反（即res ^ 0xffffffff ），再将所有位取反（即 ~ ）; 两个组合操作实质上是将数字 32 以上位取反， 0 - 32 位不变。
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/)**
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
    using LL = long long;
    vector<int> singleNumber(vector<int>& nums) {
        LL x = 0;
        for (auto y : nums)
            x ^= y;
        // 2022 test case: [1,1,0,-2147483648]
        // 会导致 -x 溢出
        int t = x & -x;
        
        vector<int> res(2);
        for (auto y : nums)
            if (y & t)
                res[0] ^= y;
            else
                res[1] ^= y;
        return res;
    }
};
```

##### **C++ 2**

```cpp
// yxc
class Solution {
public:
    int get(vector<int>& nums, int k, int t) {
        int res = 0;
        for (auto x: nums)
            if ((x >> k & 1) == t)
                res ^= x;
        return res;
    }

    vector<int> singleNumber(vector<int>& nums) {
        int ab = 0;
        for (auto x: nums) ab ^= x;
        int k = 0;
        while ((ab >> k & 1) == 0) k ++ ;
        return {get(nums, k, 0), get(nums, k, 1)};
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

> [!NOTE] **[LeetCode 268. 丢失的数字](https://leetcode-cn.com/problems/missing-number/)**
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
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int res = n;
        for (int i = 0; i < n; ++ i )
            res ^= i ^ nums[i];
        return res;
    }

    int missingNumber(vector<int>& nums) {
        int n = nums.size(), sum = (n + 1) * n / 2, res = 0;
        for (int i = 0; i < n; ++ i )
            res += nums[i];
        return sum - res;
    }
};
```

##### **Python**

```python
// yxc
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int res = n * (n + 1) / 2;
        for (auto x: nums) res -= x;
        return res;
    }
};
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 477. 汉明距离总和](https://leetcode-cn.com/problems/total-hamming-distance/)**
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
class Solution {
public:
/*
将所有数对距离的计算过程按位分离，固定第 i 个二进制位，
统计数组中数字 i 位为 1 的个数 ones，则第 i 位贡献的答案为 ones∗(n−ones)
*/
    int totalHammingDistance(vector<int>& nums) {
        int res = 0;
        for (int i = 0; i <= 30; ++ i ) {
            int x = 0, y = 0;
            for (auto v : nums)
                if (v >> i & 1) ++ y ;
                else ++ x ;
            res += x * y;
        }
        return res;
    }
};
```

##### **Python**

```python
# 如果去枚举每个数，再进行计算汉明距离，那根据数据范围，会超时；所以换一种思路：
# 1. 枚举每个数字的每一位，一共32位
# 2. 遍历数组，统计每个数字当前位为0个数总和为x，为1的个数总和为y；res += x * y 
class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        res = 0 
        for i in range(31):
            x, y = 0, 0 
            for c in nums:
                if c >> i & 1:y += 1
                else:x += 1
            res += x * y 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1835. 所有数对按位与结果的异或和](https://leetcode-cn.com/problems/find-xor-sum-of-all-pairs-bitwise-and/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数学 注意判断条件的思考

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    int getXORSum(vector<int>& arr1, vector<int>& arr2) {
        int n1 = arr1.size(), n2 = arr2.size();
        int res = 0;
        for (int i = 0; i < 32; ++ i ) {
            LL c1 = 0, c2 = 0;
            for (auto v : arr1)
                c1 += (v >> i & 1);
            for (auto v : arr2)
                c2 += (v >> i & 1);
            LL t = c1 * c2;
            // 判断条件: 为奇数个则最终AND结果中该位1的有奇数个
            if (t & 1)
                res += 1 << i;
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