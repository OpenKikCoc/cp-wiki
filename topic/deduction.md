## 习题

> [!NOTE] **[Luogu NOIP2001 普及组 数的计算](https://www.luogu.com.cn/problem/P1028)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 递推公式

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 递推公式：
// f_{2n} = f_{2n-1} + f_{n}
// f_{2n+1} = f_{2n}

using LL = long long;
const int N = 1010;

int n;

LL f[N];

void init() {
    f[0] = f[1] = 1;
    for (int i = 2; i < N; ++ i )
        if (i & 1)
            f[i] = f[i - 1];
        else
            f[i] = f[i - 1] + f[i / 2];
}

int main() {
    init();
    
    cin >> n;
    cout << f[n] << endl;
    
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

> [!NOTE] **[Luogu USACO17JAN Secret Cow Code S](https://www.luogu.com.cn/problem/P3612)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 递推 公式推导部分

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;

LL n;
// string s; // TLE
char s[55];

int main() {
    scanf("%s%lld", s, &n);
    
    // LL m = s.size(), t = m;
    LL m = strlen(s), t = m;
    while (t < n)
        t <<= 1;
    while (t != m) {
        t >>= 1;
        if (n <= t)         // the front half
            continue;
        
        if (t + 1 == n)     // special case
            n = t;
        else
            n -= 1 + t;     // n - 1 - ori_t / 2
    }
    
    putchar(s[n - 1]);
    
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

> [!NOTE] **[Luogu 直线交点数](https://www.luogu.com.cn/problem/P2789)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二维平面经典问题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// https://www.zhihu.com/question/362149679/answer/1560733589
// 若求交点数 显然 n*(n-1)/2
// 若求划分为多少个平面 有(n^2+n+2)/2
//
// 本题求能有多少不同的交点数
//  m条直线的交点方案 = r条平行线与(m-r)条直线交叉的交点数
//                    + (m-r)条直线本身的交点方案
//                    = r*(m-r)+已有的个数k

const int N = 1e4 + 10;

int n, res;
bool st[N];

void f(int m, int k) {
    if (!m) {
        if (!st[k])
            res ++ ;
        st[k] = true;
    } else
        for (int r = m; r; -- r )
            f(m - r, r * (m - r) + k);
}

int main() {
    cin >> n;
    f(n, 0);
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

> [!NOTE] **[LeetCode 338. 比特位计数](https://leetcode.cn/problems/counting-bits/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 令 $f[i]$ 表示 i 的二进制表示中 1 的个数。 则 $f[i]$ 可以由 $f[i/2]$ 转移过来， i 的二进制表示和 ⌊i/2⌋ 的二进制表示除了最后一位都一样
> 
> 所以 $ f[i] = f[i/2] + (i\&1) $ ;

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> f(num + 1);
        for (int i = 1; i <= num; i ++ )
            f[i] = f[i >> 1] + (i & 1);
        return f;
    }
};

class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> dp(num + 1, 0);
        for (int i = 1; i <= num; ++ i )
            dp[i] = i & 1 ? dp[i-1] + 1 : dp[i >> 1];
        return dp;
    }
};
```

##### **Python**

```python
# lowbit方法
class Solution:
    def countBits(self, num: int) -> List[int]:
        def lowbit(i):
            return i & -i
        res = [0]
        for i in range(1, num + 1):
            res.append(res[i - lowbit(i)] + 1)
        return res


# dp 
"""
令f[i]表示 i 的二进制表示中1的个数。
则f[i]可以由f[i/2]转移过来，ii 的二进制表示和 ⌊i/2⌋的二进制表示除了最后一位都一样，所以f[i] = f[i/2] + (i&1);

时间复杂度分析：总共有 n 个状态，每个状态进行转移的计算量是 O(1)，所以总时间复杂度是 O(n)。
"""
class Solution:
    def countBits(self, n: int) -> List[int]:
        f = (n + 1) * [0] 
        for i in range(1, n + 1):
            # 例如要看 1101
            # 我们只用看 110有多少个1 + 1101上的个位是不是1
            f[i] = f[i >> 1] + (i & 1)
        return f
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 357. 计算各个位数不同的数字个数](https://leetcode.cn/problems/count-numbers-with-unique-digits/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 首先总共有0-9十个不同的数字，所以我们只需要考虑 n≤10 的情况。
>
> 然后我们从最高位开始计数，为了方便，我们先不考虑 x=0 的情况：
>
> 1. 最高位不能选0，只有9种选法；
> 2. 次高位不能和最高位相同，但可以选0，有9种选法；
> 3. 下一位不能和前两位相同，有8种选法；
> 4. 以此类推，枚举 n 位；
>
> 最后根据乘法原理，把每一位的选法数量相乘就是总方案数。 最后不要忘记加上 x=0 的情况，让答案加1。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countNumbersWithUniqueDigits(int n) {
        if (!n) return 1;
        n = min(n, 10);
        vector<int> f(n);
        f[0] = 9;
        for (int i = 1; i < n; ++ i ) f[i] = f[i - 1] * (10 - i);
        int res = 0;
        for (int i = 0; i < n; ++ i ) res += f[i];
        return res + 1;
    }
};

// yxc
class Solution {
public:
    int countNumbersWithUniqueDigits(int n) {
        n = min(n, 10);
        if (!n) return 1;
        vector<int> f(n + 1);
        f[1] = 9;
        for (int i = 2; i <= n; i ++ )
            f[i] = f[i - 1] * (11 - i);

        int res = 1;
        for (int i = 1; i <= n; i ++ ) res += f[i];
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

> [!NOTE] **[LeetCode 396. 旋转函数](https://leetcode.cn/problems/rotate-function/)**
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
    int maxRotateFunction(vector<int>& A) {
        typedef long long LL;
        LL sum = 0, cur = 0;
        for (auto v : A) sum += v;
        int n = A.size();
        for (int i = 0; i < n; ++ i ) cur += i * A[i];
        LL res = cur;
        for (int i = n - 1; i >= 0; -- i ) {
            cur += sum - (LL)n * A[i];
            res = max(res, cur);
        }
        return res;
    }
    int maxRotateFunction_2(vector<int>& A) {
        int n = A.size();
        long long tot = 0;
        for (int i = 0; i < n; ++ i ) A.push_back(A[i]), tot += A[i];
        
        long long sum = 0;
        for (int i = 1; i <= n; ++ i ) sum += (i - 1) * A[i - 1];
        long long res = sum;
        for (int i = 2; i <= n; ++ i ) {
            // 上次的开头为 i , 末尾为 i + n - 1
            // 对于当前 i , 上次的末尾为 i + n - 2
            //cout << " - : " <<  A[n - i + 1] << endl;
            sum += tot;
            sum -= (long long)n * A[n - i + 1];
            //cout << "get sum at " << i - 1 << " = " << sum << endl;
            res = max(res, sum);
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