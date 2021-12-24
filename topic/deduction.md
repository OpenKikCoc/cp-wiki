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