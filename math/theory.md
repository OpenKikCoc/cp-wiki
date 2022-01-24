## 习题

### 拉格朗日四平方和

> [!NOTE] **[LeetCode 279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> [拉格朗日四平方和定理](https://blog.csdn.net/l_mark/article/details/89044137)
>
> * 定理内容：每个正整数均可表示成不超过四个整数的平方之和，即答案只有1、2、3、4
> * 重要的推论：
>   1. 数n如果只能表示成四个整数的平方和，不能表示成更少的数的平方之和，必定满足n=(4^a)*(8b+7)
>   2. 如果 n%4==0，k=n/4，n 和 k 可由相同个数的整数表示
> * 如何利用推论求一个正整数最少需要多少个数的平方和表示：
>   1. 先判断这个数是否满足 n=(4^a)*(8b+7)，如果满足，那么这个数就至少需要 4 个数的平方和表示，即答案为4。
>   2. 如果不满足，再在上面除以 4 之后的结果上暴力尝试只需要 1 个数就能表示和只需要 2 个数就能表示的情况。
>   3. 如果这个数本来就是某个数的平方，那么答案就是1
>   4. 如果答案是2的话，那么n=a^2+b^2，枚举a即可
>   5. 如果还不满足，那么就只需要 3 个数就能表示

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool check(int x) {
        int r = sqrt(x);
        return r * r == x;
    }

    int numSquares(int n) {
        if (check(n)) return 1;

        for (int a = 1; a <= n / a; a ++ )
            if (check(n - a * a))
                return 2;

        while (n % 4 == 0) n /= 4;
        if (n % 8 != 7) return 3;
        return 4;
    }
};
```

##### **C++ dp**

```cpp
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n+1);
        for (int i = 1; i <= n; ++ i ) {
            dp[i] = i;
            for (int j = 1; j <= i / j; ++ j ) {
                dp[i] = min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }
};
```

##### **Python**

```python
"""
完全背包问题

(动态规划) O(nn‾√)
设 f(i) 表示通过平方数组成 i 所需要完全平方数的最少数量。
初始时，f(0)=0其余待定。
转移时，对于一个 i，枚举 j，f(i)=min(f(i−j∗j)+1) ，其中 1≤j≤√i。
最终答案为 f(n)。
"""

import math
class Solution:
    def numSquares(self, n: int) -> int:

        goods = [i * i for i in range(1, int(math.sqrt(n))+1)]

        f = [float('inf')] * (n+1)
        f[0] = 0

        for good in goods:
            for j in range(good, n+1):
                f[j] = min(f[j], f[j-good]+1)

        return f[-1]
```

<!-- tabs:end -->
</details>

<br>

* * *

### 约数与完全平方数

> [!NOTE] **[LeetCode 319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 每个灯泡开关被按的次数等于它的编号的约数个数。
> 
> 最终灯泡是亮的，说明编号有奇数个约数。
> 
> 下面我们证明：一个数有奇数个约数，等价于它是平方数。
> 
> 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int bulbSwitch(int n) {
        return sqrt(n);
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

> [!NOTE] **[LeetCode 672. 灯泡开关 Ⅱ](https://leetcode-cn.com/problems/bulb-switcher-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典状态机 重复
> 
> ![img](https://camo.githubusercontent.com/4409d53a7cd8a786780c8a21238eca8628efb3a0c011c632778946078b64eeec/68747470733a2f2f7069632e6c656574636f64652d636e2e636f6d2f633330306532626435373332396337343634353661333339316231346632306265333035656261623336316436633866323666633266656262393735333534302d2545362539372541302545362541302538372545392541322539382e706e67)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 推导**

```cpp
class Solution {
public:
    int flipLights(int n, int m) {
        n = min(n, 3);
        if (m == 0) return 1;
        if (m == 1) return n == 1 ? 2 : n == 2 ? 3 : 4;
        if (m == 2) return n == 1 ? 2 : n == 2 ? 4 : 7;
        return n == 1 ? 2 : n == 2 ? 4 : 8;
    }
};
```

##### **C++ 状态机**

```cpp
class Solution {
public:
    int state[8][6] = {
        {1, 1, 1, 1, 1, 1},  // 不按
        {0, 0, 0, 0, 0, 0},  // 1
        {1, 0, 1, 0, 1, 0},  // 2
        {0, 1, 0, 1, 0, 1},  // 3
        {0, 1, 1, 0, 1, 1},  // 4
        {1, 0, 0, 1, 0, 0},  // 14
        {0, 0, 1, 1, 1, 0},  // 24
        {1, 1, 0, 0, 0, 1},  // 34
    };

    int work(int n, vector<int> ops) {
        set<int> S;
        for (auto op: ops) {
            int t = 0;
            for (int i = 0; i < n; i ++ )
                t = t * 2 + state[op][i];
            S.insert(t);
        }
        return S.size();
    }

    int flipLights(int n, int m) {
        n = min(n, 6);
        if (m == 0) return work(n, {0});
        else if (m == 1) return work(n, {1, 2, 3, 4});
        else if (m == 2) return work(n, {0, 1, 2, 3, 5, 6, 7});
        else return work(n, {0, 1, 2, 3, 4, 5, 6, 7});
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

> [!NOTE] **[LeetCode 1375. 灯泡开关 III](https://leetcode-cn.com/problems/bulb-switcher-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 左边连续亮 则左边的都变成蓝色 求所有灯变成蓝色的时刻的数目
> 
> 其实就是求 亮的个数 == 最靠右的灯的序号 的数目

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int numTimesAllBlue(vector<int>& light) {
        int res = 0;
        int maxv = 0, cnt = 0;
        for (auto k : light) {
            maxv = max(maxv, k);
            ++cnt;
            if (maxv == cnt) ++res;
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

> [!NOTE] **[LeetCode 1529. 灯泡开关 IV](https://leetcode-cn.com/problems/bulb-switcher-iv/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 显然需要从最左侧考虑 边处理边记录整个右侧的状态 扫一遍即可
> 
> - 赛榜有别的做法 都是从最右侧考虑得到：从左向右扫相邻数值不等则加1

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int minFlips(string target) {
        int n = target.size();
        int now = 0, res = 0;  // now表示整个处理点右侧的状态 res为改动次数
        for (int i = 0; i < n; ++i) {
            if (target[i] == '1' && now & 1)
                continue;
            else if (target[i] == '0' && !(now & 1))
                continue;
            now ^= 1, ++res;
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int minFlips(string s) {
        s = "0" + s;
        int res = 0;
        for (int i = 1; i < s.size(); ++i)
            if (s[i] != s[i - 1]) ++res;
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

> [!NOTE] **[LeetCode 365. 水壶问题](https://leetcode-cn.com/problems/water-and-jug-problem/)**
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
    bool canMeasureWater(int x, int y, int z) {
        if (x + y < z) return false;
        return !z || z % __gcd(x, y) == 0;
    }
};



class Solution {
public:
    bool canMeasureWater(int x, int y, int z) {
        if (x + y < z) return false;
        if (x == 0 || y == 0) return z == 0 || x + y == z;
        return z % __gcd(x, y) == 0;
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