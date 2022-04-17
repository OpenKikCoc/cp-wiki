
> [!TIP] **求最大值最小值初始化总结**
> 
> **二维情况**
> 
> 1. 体积至多 $j$ （只会求价值的最大值）:
>    - $$f[i, k] = 0 , 0 <= i <= n , 0 <= k <= m$$ 
> 
> 2. 体积恰好 $j$ :
>    - 当求价值的最小值： 
>      $$f[0][0] = 0 , 其余是 INF$$
>    - 当求价值的最大值： 
>      $$f[0][0] = 0 , 其余是 -INF$$
> 
> 3. 体积至少 $j$ （只会求价值的最小值）: 
>    - $$f[0][0] = 0, 其余是INF$$

## 0-1 背包

> [!TIP] 
>
> **思路**
>
> **朴素写法**
>
> 1. 状态表示：$f[i, j]$ 表示的是放 $i$ 个物品，总体积不超过 $j$ 的价值；属性：$max$
>
> 3. 状态转移：(曲线救国==> 第 $i$ 个物品先去掉，看前 $i-1$ 个物品的最大值怎么求）
> 	 以第 $i$ 个物品"选"还是"不选"作为区分来进行转移。
> 	 
> 	 1）选： $f[i, j] = f[i - 1, j - v[i]] + w[i]$
> 	 
> 	 2）不选： $f[i, j] = f[i - 1, j]$
>
> **滚动数组优化**
>
> $f[i]$ 在计算的时候只用到了 $f[i - 1]$ 的值，所以可以用滚动数组进行优化；
> ==> 滚动数组可以将第二维的 $n$ 降到 $2$ ，做法：第 $i$ 项和 $i - 1$ 项都和 $1$ 做位与运算（也相当于模 $2$）
>
> 如果$f[i]$ 在计算的时候只用到了 $f[i - 1]$ 和 $f[i - 2]$ 的值，那也可以用滚动数组来优化，这个时候就不是直接和1做与运算，而是每一项模 $3$（本质上是做一个映射）
>
> **空间压缩优化**
>
> 1. $f[i， j],  f[i， j - v[i]]$ 这两个中的 $j$ 这一维都是小于等于 $j$ 的（都是在 $j$ 的一侧），可以进一步进行空间优化：压缩成一维
>
> 2. 注意枚举背包容量 $j$ 必须从 $m$ 开始，从大到小遍历
> 	
> 	$f[i， j]$ 依赖 $f[i - 1, j - v[i]] 和  f[i - 1, j]$， 如果从小到大遍历，那么 $f[i - 1, j - v[i]]$ 已经被 $f[i, j - v[i]]$ 更新了。简单俩说，就是一维情况下正序更新状态 $f[j]$ 需要用到前面一行计算的状态已经被当前行[更新污染]了，逆序就不会存在这样的问题。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N];

int main() {
    cin >> n >> m;

    for (int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];

    for (int i = 1; i <= n; i ++ )
        for (int j = m; j >= v[i]; j -- )
            f[j] = max(f[j], f[j - v[i]] + w[i]);

    cout << f[m] << endl;

    return 0;
}
```

##### **Python-朴素**

```python
if __name__ == '__main__':
    N = 1010
    v = [0] * N
    w = [0] * N
    f = [[0] * N for _ in range(N)]

    n, m = map(int, input().split())
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        v[i] = a
        w[i] = b
    for i in range(1, n + 1):
        for j in range(m + 1):
            f[i][j] = f[i - 1][j]
            if j >= v[i]:
                f[i][j] = max(f[i][j], f[i - 1][j - v[i]] + w[i])
    print(f[n][m])
```

##### **Python-滚动数组**

```python
if __name__ == '__main__':
    N = 1010
    v = [0] * N
    w = [0] * N
    f = [[0] * N for _ in range(2)]

    n, m = map(int, input().split())
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        v[i] = a
        w[i] = b
    for i in range(1, n + 1):
        for j in range(m + 1):
            f[i & 1][j] = f[(i - 1) & 1][j]
            if j >= v[i]:
                f[i & 1][j] = max(f[i & 1][j], f[(i - 1) & 1][j - v[i]] + w[i])
    print(f[n & 1][m])
```

##### **Python-空间压缩**

```python
if __name__ == '__main__':
    N = 1010
    v = [0] * N
    w = [0] * N
    f = [0] * N

    n, m = map(int, input().split())
    for i in range(1, n + 1):
        v[i], w[i] = map(int, input().split())
    for i in range(1, n + 1):
        for j in range(m, v[i] - 1, -1):
            f[j] = max(f[j], f[j - v[i]] + w[i])
    print(f[m])
```

##### **en-us**

1.  State define:

	Let us assume $f[i][j]$ means the max value which picks from the first i numbers and the sum of volume $<= j$; 

2.  Transition function:

	For each number, we can pick it or not.
	1) If we don't pick it: $f[i][j] = f[i-1][j]$, which means if the first $i-1$ element has made it to $j$, $f[i][j]$ would als make it to $j$, and we can just ignore $nums[i]$
	2) If we pick nums[i]: $f[i][j] = f[i-1][j-nums[i]]$, which represents that $j$ is composed of the current value $nums[i]$ and the remaining composed of other previous numbers. 

3.  Base case: 

	$f[0][0] = 0$ ; (zero number consists of volumen $0$ is $0$, which reprents it's validaing）

-----------------------------------------------------------------------------------------------------

It seems that we cannot optimize it in time. But we can optimize in space.

1.  Optimize to $O(2 * n)$

	You can see that f[i][j] only depends on previous row, so we can optimize the space by only using two rows instead of the matrix. Let's say $arr1$ and $arr2$. Every time we finish updating $arr2$, $arr1$ have no value, you can copy $arr2$ to $arr1$ as the previous row of the next new row.

2.  Optimize to $O(n)$
  
	You can also see that, the column indices of $f[i - 1][j - nums[i]]$ and $f[i - 1][j]$ are $<= j$. 
	
	The conclusion you can get is: the elements of previous row whose column index is > j will not affect the update of $f[i][j]$ since we will not touch them.
	
	Thus, if you merge $arr1$ and $arr2$ to a single array, if you update array backwards(从后面开始更新), all dependencies are not touched!
	However if you update array forwards(从前面开始更新), $f[j - nums[i - 1]]$ is updated already, you cannot use it.

<!-- tabs:end -->
</details>

<br>

* * *

## 完全背包

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N];

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];

    for (int i = 1; i <= n; i ++ )
        for (int j = v[i]; j <= m; j ++ )
            f[j] = max(f[j], f[j - v[i]] + w[i]);

    cout << f[m] << endl;

    return 0;
}
```

##### **Python-朴素dp**

```python
if __name__ == '__main__':
    N = 1010
    f = [[0] * N for _ in range(N)]
    v = [0] * N
    w = [0] * N
    
    n, m = map(int, input().split())
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        v[i], w[i] = a, b
        
    for i in range(1, n + 1):
        for j in range(m + 1):
            k = 0 
            while k * v[i] <= j:
                f[i][j] = max(f[i][j], f[i - 1][j - k * v[i]] + k * w[i])
                k += 1
    print(f[n][m])
```

##### **Python-优化**

```python
"""
列举一下更新状态的内部关系：
f[i, j] = max(f[i-1,j], f[i-1][j-v]+w, f[i-1][j-2*v]+2*w, ...,f[i-1][j-k*v]+k*w+...)
f[i, j-v] = max(        f[i-1,j-v],    f[i-1][j-2*v]+w, ...,  f[i-1][j-k*v]+(k-1)*w+...)
由上两式子可得出:
f[i, j] = max(f[i-1,j], f[i][j-v]+w)
所以可以省掉k那层循环，对代码进行等价变化
"""

if __name__ == '__main__':
    N = 1010
    f = [[0] * N for _ in range(N)]
    v = [0] * N
    w = [0] * N

    n, m = map(int, input().split())
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        v[i] = a
        w[i] = b

    for i in range(1, n + 1):
        for j in range(m + 1):
            f[i][j] = f[i - 1][j]
            if j >= v[i]:
                f[i][j] = max(f[i - 1][j], f[i][j - v[i]] + w[i])
    print(f[n][m])
```

##### **Python-空间压缩**

```python
"""
f[i, j] = max(f[i-1,j], f[i][j-v]+w), 
当优化掉i维时，为了保证状态转移的正确，j是从小到大遍历计算的
如果像0/1从大到小遍历j，计算f[i, j]中的f[i-1,j]，f[i-1,j]已经被第i行的值更新了，无法再用。
"""
if __name__ == '__main__':
    N = 1010
    f = [0] * N
    v = [0] * N
    w = [0] * N

    n, m = map(int, input().split())
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        v[i] = a
        w[i] = b

    for i in range(1, n + 1):
        for j in range(v[i], m + 1):
            f[j] = max(f[j], f[j - v[i]] + w[i])
    print(f[m])
```

<!-- tabs:end -->
</details>

<br>

* * *

## 多重背包

> [!TIP] 
>
> **思路**
>
> 完全背包问题可以用多重背包问题来优化，其实时间复杂度会变慢。
> 为什么多重背包问题就需要用单调队列来进行优化，但是完全背包问题就不需要呢？
> 核心原因在于：多重背包问题求的是滑动窗口内的最大值，那滑动窗口的最大值必须要用单调队列来进行优化
>
> 1. 状态表示 $f[i,j]$: 所有只从前 $i$ 个物品中选，并且总体积不超过 $j$ 的选法；属性：$max$
> 2. 状态转移：按照第 $i$ 个物品选几个，把所有分发分成若干类。
> 	 $f[i,j] = max(f[i - 1， j - k * v[i]] + w[i] * k); k = 0, 1, 2..., s[i]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n, m;
int v[N], w[N], s[N];
int f[N][N];

int main() {
    cin >> n >> m;

    for (int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i] >> s[i];

    for (int i = 1; i <= n; i ++ )
        for (int j = 0; j <= m; j ++ )
            for (int k = 0; k <= s[i] && k * v[i] <= j; k ++ )
                f[i][j] = max(f[i][j], f[i - 1][j - v[i] * k] + w[i] * k);

    cout << f[n][m] << endl;
    return 0;
}
```

##### **Python**

```python
N = 110
f = [[0] * N for _ in range(N)]
v = [0] * N
w = [0] * N
s = [0] * N

if __name__=='__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):
        v[i], w[i], s[i] = map(int, input().split())
    for i in range(1, n + 1):
        for j in range(m + 1):
            for j in range(s[i] + 1):
                if j >= k * v[i]:
                     # 这里已经包含了不选第i个物品的选法了。
                     f[i][j] = max(f[i][j],f[i - 1][j - k * v[i]] + k * w[i]) 
    print(f[n][m])
		#本题数据范围比较小，所以可以不用优化就可以ac
```

<!-- tabs:end -->
</details>

<br>

* * *

### 二进制分组优化

> [!TIP] 
>
> **思路**
>
> 本题数据范围更大 $N=1000$ ；所以需要优化上述暴力算法。
> 需要用经典优化：**二进制的优化方式**：
> 把若干个第 $i$ 个物品打包分组，每组最多选一次，看是否能凑出所有的选法：$1，2，4，8，..., 512$ ===> 可以凑出 $0-1023$
>
> 如果 $s = 200$， 用 $1，2，4，8，16，32，64，73$ 就可以凑出 $0-200$ 
>
> ===> 就是把 200 个物品 $i$ 分成8组， 每组物品都只能选一次，可以凑出来 200 个物品。转化为 $0/1$ 背包问题。
>
> （这里不能要 128，因为从 1加到 128等于 255，我们没有那么多物品）
> 对于一般数来抽象出规律，再用代码实现。
> 1）把第 $i$ 个物品的 $s$ 个，拆成 $logs$ 组的新物品：新物品只能用一次。
> 2）再对新物品做一遍 0/1 背包问题 就可以。时间复杂度从 $O(NVS)$ ===> $O(NVlogS)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 12010, M = 2010;

int n, m;
int v[N], w[N];
int f[M];

int main() {
    cin >> n >> m;

    int cnt = 0;
    for (int i = 1; i <= n; i ++ ) {
        int a, b, s;
        cin >> a >> b >> s;
        int k = 1;
        while (k <= s) {
            cnt ++ ;
            v[cnt] = a * k;
            w[cnt] = b * k;
            s -= k;
            k *= 2;
        }
        if (s > 0) {
            cnt ++ ;
            v[cnt] = a * s;
            w[cnt] = b * s;
        }
    }

    n = cnt;

    for (int i = 1; i <= n; i ++ )
        for (int j = m; j >= v[i]; j -- )
            f[j] = max(f[j], f[j - v[i]] + w[i]);

    cout << f[m] << endl;

    return 0;
}
```

##### **C++ 习惯写法**

```cpp
#include <algorithm>
#include<bits/stdc++.h>
using namespace std;

int main() {
    int n, v, c, w, s;
    cin >> n >> v;
    vector<int> f(v+1);
    for(int i = 0; i < n; ++i) {
        cin >> c >> w >> s;
        for(int k = 1; k <= s; k *= 2) {
            for(int j = v; j >= k*c; --j)
                f[j] = max(f[j], f[j-k*c] + k*w);
            s -= k;
        }
        if(s) for(int j = v; j >= s*c; --j) f[j] = max(f[j], f[j-s*c] + s*w);
    }
    cout << f[v] << endl;
}
```

##### **Python**

```python
if __name__ == '__main__':
    N = 12010
    M = 2010
    v = [0] * N
    w = [0] * N
    f = [0] * M

    n, m = map(int, input().split())
    cnt = 0
    for i in range(1, n + 1):
        k = 1
        a, b, s = map(int, input().split())
        while k <= s:
            cnt += 1
            v[cnt] = a * k
            w[cnt] = b * k
            s -= k
            k *= 2
        if s > 0:
            cnt += 1
            v[cnt] = a * s
            w[cnt] = b * s
    n = cnt
    for i in range(1, n + 1):
        for j in range(m, v[i] - 1, -1):
            f[j] = max(f[j], f[j - v[i]] + w[i])
    print(f[m])
```

<!-- tabs:end -->
</details>

<br>

* * *

### 单调队列优化

见 [单调队列/单调栈优化](dp/opt/monotonous-queue-stack.md)。

习题：[「Luogu P1776」宝物筛选\_NOI 导刊 2010 提高（02）](https://www.luogu.com.cn/problem/P1776)

TODO@binacs 一大批题目

## 混合背包

> [!NOTE] **[「Luogu P1833」樱花](https://www.luogu.com.cn/problem/P1833)**
> 
> 题意概要：有 $n$ 种樱花树和长度为 $T$ 的时间，有的樱花树只能看一遍，有的樱花树最多看 $A_{i}$ 遍，有的樱花树可以看无数遍。每棵樱花树都有一个美学值 $C_{i}$，求在 $T$ 的时间内看哪些樱花树能使美学值最高。

> [!TIP]
> 
> `01 背包` 算 `多重背包` 特殊情况，对 for 循环优化（二进制、单调队列）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 1e3 + 5;
int q[maxn];

int main() {
    int n, m, v, w, s;
    cin >> n >> m;
    vector<int> f(m + 1);
    // 3.
    vector<int> pre(m + 1);
    for (int i = 0; i < n; ++i) {
        cin >> v >> w >> s;
        // 3.
        pre = f;
        // 完全背包
        if (!s) {
            for (int j = v; j <= m; ++j) f[j] = max(f[j], f[j - v] + w);
        } else {
            // 01背包 算多重背包的特殊情况
            if (s == -1) s = 1;
            //  随后多重背包二进制优化
            // 两种写法：
            // 1.
            for (int k = 1; k <= s; k *= 2) {
                for (int j = m; j >= k * v; --j)
                    f[j] = max(f[j], f[j - k * v] + k * w);
                s -= k;
            }
            if (s)
                for (int j = m; j >= s * v; --j)
                    f[j] = max(f[j], f[j - s * v] + s * w);

            // 2.
            int num = min(s, m / v);
            for (int k = 1; num > 0; k *= 2) {
                if (k > num) k = num;
                num -= k;
                for (int j = m; j >= k * v; --j)
                    f[j] = max(f[j], f[j - k * v] + k * w);
            }

            // 或用单调队列优化
            // 3.
            for (int j = 0; j < v; ++j) {
                int head = 0, tail = -1;
                for (int k = j; k <= m; k += v) {
                    while (head <= tail && q[head] < k - s * v) ++head;
                    while (head <= tail &&
                           pre[q[tail]] - (q[tail] - j) / v * w <=
                               pre[k] - (k - j) / v * w)
                        --tail;
                    q[++tail] = k;
                    f[k] = pre[q[head]] + (k - q[head]) / v * w;
                }
            }
        }
    }
    cout << f[m] << endl;
}
```

##### **Python**

```python
# 本题数据范围比较大，需要对多重背包问题进行二进制优化。（优化成一维）
# 分析的时候，永远是事实实际情况出发！

N = 10100
f = [0] * N

if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(n):
        v, w, s = map(int, input().split())
        if s == 0:  # 完全背包
            for j in range(v, m + 1):
                f[j] = max(f[j], f[j - v] + w)
        else:
            if s == -1:  # 0/1背包 :  其实是特殊的 物品只有一件的 多重背包问题 （就当作多重背包问题来处理即可）
                s = 1
            k = 1
            while k <= s:
                for j in range(m, v * k - 1, -1):
                    f[j] = max(f[j], f[j - k * v] + k * w)
                s -= k
                k *= 2
            if s > 0:  # 如果s有剩余的话，还需要计算完全。
                for j in range(m, v * s - 1, -1):
                    f[j] = max(f[j], f[j - s * v] + s * w)
    print(f[m])
    
# 多重背包问题单调队列优化参见：https://www.acwing.com/activity/content/code/content/446901/
```

<!-- tabs:end -->
</details>

<br>

* * *

## 二维费用背包

> [!TIP] 
>
> **思路**
>
> 二维费用背包问题====可以和所有背包问题都划分在一块。
> 1. 状态表示：$f[i,j,k]$ 表示所有只从前 $i$ 个物品中选，并且总体积不超过 $j$, 总重量不超过 $k$ 的选法；属性：$max$
> 2. 状态转移：以最后一个物品 放 还是 不放。
> 不包含最后一个物品：$f[i - 1, j, k]$
> 包含最后一个物品：$f[i - 1, j - v[i], k - m[i]) + w[i]$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>

using namespace std;

const int N = 110;

int n, V, M;
int f[N][N];

int main() {
    cin >> n >> V >> M;

    for (int i = 0; i < n; i ++ ) {
        int v, m, w;
        cin >> v >> m >> w;
        for (int j = V; j >= v; j -- )
            for (int k = M; k >= m; k -- )
                f[j][k] = max(f[j][k], f[j - v][k - m] + w);
    }

    cout << f[V][M] << endl;

    return 0;
}
```

##### **Python**

```python
N = 1010
f = [[0] * N for _ in range(N)]
v = [0] * N
w = [0] * N
g = [0] * N

if __name__ == '__main__':
    n, m, k = map(int, input().split())
    for i in range(1, n + 1):
        v[i], g[i], w[i] = map(int, input().split())
    for i in range(1, n + 1):
        for j in range(m, v[i] - 1, -1):
            for u in range(k, g[i] - 1, -1):
                f[j][u] = max(f[j][u], f[j - v[i]][u - g[i]] + w[i])
    print(f[m][k])
```

<!-- tabs:end -->
</details>

<br>

* * *

## 分组背包

> [!TIP] 
>
> **思路**
>
> 这里要注意：**一定不能搞错循环顺序**，这样才能保证正确性。
>
> 1. 状态表示$：f[i, j]$ 表示只能从前 $i$ 组物品中选，且总体积不大于 $j$ 的所有选法；属性：$max$
>
> 2. 状态转移：枚举第 $i$ 组物品 选哪个？或者不选 ==> 1) 不选 2）选第 $i$ 组的第 $1$ 个物品,...,第 $k$ 个物品...
> 	（上述转移的过程很像一根热狗，就叫 “热狗划分法”）
> 	
> 	1）第 $i$ 组物品不选：$f[i - 1,j]$
> 	2）第 $i$ 组物品选第 k 个物品： $f[i - 1,j - v[k]] + w[i,k]$ 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n, m;
int v[N][N], w[N][N], s[N];
int f[N];

int main() {
    cin >> n >> m;

    for (int i = 1; i <= n; i ++ ) {
        cin >> s[i];
        for (int j = 0; j < s[i]; j ++ )
            cin >> v[i][j] >> w[i][j];
    }

    for (int i = 1; i <= n; i ++ )
        for (int j = m; j >= 0; j -- )
            for (int k = 0; k < s[i]; k ++ )
                if (v[i][k] <= j)
                    f[j] = max(f[j], f[j - v[i][k]] + w[i][k]);

    cout << f[m] << endl;

    return 0;
}
```

##### **Python**

```python
# 空间优化的时候：
# 如果用的是上一层的状态的话，就从大到小枚举；如果用的是本层的状态，就直接从小到大枚举就可以。
# 统一组别和物品都是从下标1开始的

if __name__ == '__main__':
    N = 110
    f = [0] * N
    v = [[0] * N for _ in range(N)]
    w = [[0] * N for _ in range(N)]
    s = [0] * N

    n, m = map(int, input().split())
    for i in range(1, n + 1):  # 组别下标从1开始
        s[i] = int(input())
        for j in range(1, s[i] + 1):  # 物品下标从1开始
            v[i][j], w[i][j] = map(int, input().split())

    for i in range(1, n + 1):
        for j in range(m, -1, -1):
            for k in range(1, s[i] + 1):  # 遍历组内物品的时候，也应该从1开始！
                if v[i][k] <= j:
                    f[j] = max(f[j], f[j - v[i][k]] + w[i][k])
    print(f[m])

```

<!-- tabs:end -->
</details>

<br>

* * *

## 有依赖的背包

> [!TIP]
>
> **要选 y 必选其依赖 x** 如选课模型等 结合树形DP。
>
> 【分组背包是循环体积再循环选择组内物品（本质是选择子体积），选择只选一个】
>
> 对于每一个节点选择可能有 $2^x$ 种可能，简化为体积，每个体积作为一件物品 => 每个节点作为一个分组。
>
> 
>
> 用递归的思路考虑，框架是树形 $dp$ 的模板
>
> 1. 状态表示：$f[u,j]$:表示所有以 $u$ 为根的子树中选，体积不超过 $j$ 的方案；属性：$max$
>
> 2. 状态转移：用新的划分方式：以体积来进行划分!!!（从每个子树选 or 不选，优化成体积为 $0-m$ 的划分）
>
>    $k$ 个子树，选or不选，时间复杂度 $2^k$ 
>
> 每棵子树都分为 $0-m$ 种情况，分别表示：体积是 $0,...m$ 的价值是多少（一共有 $m+1$ 类，求每一类里取最大价值就可以）
>
> 这道题转化为了：一共有 $n$ 颗子树，每个子树内可以选择用多大的体积，问总体积不超过 $j$ 的最大价值是多少。
>
> 3. 总结：可以把每个子树看成一个物品组，每个组内部有 $m+1$ 个物品，第 $0$ 个物品表示体积是 $0$，价值 $f[0]$；第 $m$ 个类表示体积是 $m$,价值是 $f[m]$。每个物品组只能选一个出来。
>
> dp分析精髓： 用某一个数表示一类；这里就是用不同的体积表示一大类，
>
> 分组背包问题：一定要记住：先循环组数，再循环体积，再循环决策（选某个组的哪个物品）



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> v, w;
vector<vector<int>> f, es;

int N, V;

void dfs(int x) {
    // 点 x 必选 故初始化包含 x 的价值
    for (int i = v[x]; i <= V; ++i) f[x][i] = w[x];
    for (auto& y : es[x]) {
        dfs(y);
        // j 范围   小于 v[x] 无法选择节点x
        for (int j = V; j >= v[x]; --j)
            // 分给子树y的空间不能大于 j-v[x] 否则无法选择物品x
            for (int k = 0; k <= j - v[x]; ++k)
                f[x][j] = max(f[x][j], f[x][j - k] + f[y][k]);
    }
}

int main() {
    int p, root;
    cin >> N >> V;
    v.resize(N + 1);
    w.resize(N + 1);
    f.resize(N + 1, vector<int>(V + 1));
    es.resize(N + 1);

    for (int i = 1; i <= N; ++i) {
        cin >> v[i] >> w[i] >> p;
        if (p == -1)
            root = i;
        else
            es[p].push_back(i);
    }
    dfs(root);
    cout << f[root][V] << endl;
}
```

##### **Python**

```python
# 如果会爆栈的话，可以加入以下代码段
# import sys
# limit = 10000
# sys.setrecursionlimit(limit)

N = 110
h = [-1] * N
ev = [0] * N
ne = [0] * N
idx = 0
v = [0] * N
w = [0] * N
f = [[0] * N for _ in range(N)]


def dfs1(u):
    global m
    i = h[u]
    while i != -1:  # 1 go thru every sub-tree
        son = ev[i]
        dfs(ev[i])
        # group knapsack template
        # 1 travel group == every sub-tree
        # 2 travel volumn of knapsack
        for j in range(m - v[u], -1, -1):  # 必须要加上根节点,所有为根节点留出空间
            # 3 travel options 枚举当前用到的背包容量
            for k in range(0, j + 1):
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])
        i = ne[i]
    # 将当前点root加进来
    for i in range(m, v[u] - 1, -1):
        f[u][i] = f[u][i - v[u]] + w[u]
    # 当容量小于根节点v[root],则必然放不下，最大值都是0
    for i in range(v[u]):
        f[u][i] = 0


def dfs(u):
    global m  # 递归时 m需要随着变化，所以需要加global关键字
    for i in range(v[u], m + 1):  # 点u必须选，才能选子节点，所以初始化f[u][v[u]~m] = w[u]
        f[u][i] = w[u]
    i = h[u]
    while i != -1:
        son = ev[i]
        dfs(ev[i])

        for j in range(m, v[u] - 1, -1):  # j的范围为v[x]~m, 小于v[x]无法选择以x为子树的物品
            for k in range(0, j - v[u] + 1):  # 分给子树y的空间不能大于j-v[x],不然都无法选根物品x
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])
        i = ne[i]  # 踩坑：一定要记得写啊！！！


def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


if __name__ == '__main__':
    n, m = map(int, input().split())
    root = 0
    for i in range(1, n + 1):
        v[i], w[i], p = map(int, input().split())
        if p == -1:
            root = i
        else:
            add_edge(p, i)
    dfs(root)
    print(f[root][m])  # f[i][m] 前i个物品，体积<=m，的最大价值

```

<!-- tabs:end -->
</details>

<br>

* * *

## 泛化物品的背包

这种背包，没有固定的费用和价值，它的价值是随着分配给它的费用而定。在背包容量为 $V$ 的背包问题中，当分配给它的费用为 $v_i$ 时，能得到的价值就是 $h\left(v_i\right)$。这时，将固定的价值换成函数的引用即可。

TODO@binacs

## 杂项 TODO@binacs

### 小优化

根据贪心原理，当费用相同时，只需保留价值最高的；当价值一定时，只需保留费用最低的；当有两件物品 $i,j$ 且 $i$ 的价值大于 $j$ 的价值并且 $i$ 的费用小于 $j$ 的费用是，只需保留 $j$。

### 背包问题变种

#### 输出方案

输出方案其实就是记录下来背包中的某一个状态是怎么推出来的。我们可以用 $g_{i,v}$ 表示第 $i$ 件物品占用空间为 $v$ 的时候是否选择了此物品。然后在转移时记录是选用了哪一种策略（选或不选）。输出时的伪代码：

```cpp
int v = V;  // 记录当前的存储空间

// 因为最后一件物品存储的是最终状态，所以从最后一件物品进行循环
for (从最后一件循环至第一件) {
    if (g[i][v]) {
        选了第 i 项物品;
        v -= 第 i 项物品的价值;
    } else
        未选第 i 项物品;
}
```

#### 求方案数

对于给定的一个背包容量、物品费用、其他关系等的问题，求装到一定容量的方案总数。

这种问题就是把求最大值换成求和即可。

例如 0-1 背包问题的转移方程就变成了：

$$
\mathit{dp}_i=\sum(\mathit{dp}_i,\mathit{dp}_{i-c_i})
$$

初始条件：$\mathit{dp}_0=1$

因为当容量为 $0$ 时也有一个方案，即什么都不装。

#### 求最优方案总数

要求最优方案总数，我们要对 0-1 背包里的 $\mathit{dp}$ 数组的定义稍作修改，DP 状态 $f_{i,j}$ 为在只能放前 $i$ 个物品的情况下，容量为 $j$ 的背包“正好装满”所能达到的最大总价值。

这样修改之后，每一种 DP 状态都可以用一个 $g_{i,j}$ 来表示方案数。

$f_{i,j}$ 表示只考虑前 $i$ 个物品时背包体积“正好”是 $j$ 时的最大价值。

$g_{i,j}$ 表示只考虑前 $i$ 个物品时背包体积“正好”是 $j$ 时的方案数。

转移方程：

如果 $f_{i,j} = f_{i-1,j}$ 且 $f_{i,j} \neq f_{i-1,j-v}+w$ 说明我们此时不选择把物品放入背包更优，方案数由 $g_{i-1,j}$ 转移过来，

如果 $f_{i,j} \neq f_{i-1,j}$ 且 $f_{i,j} = f_{i-1,j-v}+w$ 说明我们此时选择把物品放入背包更优，方案数由 $g_{i-1,j-v}$ 转移过来，

如果 $f_{i,j} = f_{i-1,j}$ 且 $f_{i,j} = f_{i-1,j-v}+w$ 说明放入或不放入都能取得最优解，方案数由 $g_{i-1,j}$ 和 $g_{i-1,j-v}$ 转移过来。

初始条件：

```cpp
memset(f, 0x3f3f, sizeof(f));  // 避免没有装满而进行了转移
f[0] = 0;
g[0] = 1;  // 什么都不装是一种方案
```

因为背包体积最大值有可能装不满，所以最优解不一定是 $f_{m}$。

最后我们通过找到最优解的价值，把 $g_{j}$ 数组里取到最优解的所有方案数相加即可。

核心代码：

```cpp
for (int i = 0; i < N; i++) {
    for (int j = V; j >= v[i]; j--) {
        int tmp = max(dp[j], dp[j - v[i]] + w[i]);
        int c = 0;
        if (tmp == dp[j]) c += cnt[j];  // 如果从dp[j]转移
        if (tmp == dp[j - v[i]] + w[i])
            c += cnt[j - v[i]];  // 如果从dp[j-v[i]]转移
        dp[j] = tmp;
        cnt[j] = c;
    }
}
int max = 0;  // 寻找最优解
for (int i = 0; i <= V; i++) { max = max(max, dp[i]); }
int res = 0;
for (int i = 0; i <= V; i++) {
    if (dp[i] == max) {
        res += cnt[i];  // 求和最优解方案数
    }
}
```

#### 背包的第 k 优解

普通的 0-1 背包是要求最优解，在普通的背包 DP 方法上稍作改动，增加一维用于记录当前状态下的前 k 优解，即可得到求 0-1 背包第 $k$ 优解的算法。

具体来讲：$\mathit{dp_{i,j,k}}$ 记录了前 $i$ 个物品中，选择的物品总体积为 $j$ 时，能够得到的第 $k$ 大的价值和。这个状态可以理解为将普通 0-1 背包只用记录一个数据的 $\mathit{dp_{i,j}}$ 扩展为记录一个有序的优解序列。转移时，普通背包最优解的求法是 $\mathit{dp_{i,j}}=\max(\mathit{dp_{i-1,j}},\mathit{dp_{i-1,j-v_{i}}}+w_{i})$，现在我们则是要合并 $\mathit{dp_{i-1,j}}$，$\mathit{dp_{i-1,j-v_{i}}}+w_{i}$ 这两个大小为 $k$ 的递减序列，并保留合并后前 $k$ 大的价值记在 $\mathit{dp_{i,j}}$ 里，这一步利用双指针法，复杂度是 $O(k)$ 的，整体时间复杂度为 $O(nmk)$。空间上，此方法与普通背包一样可以压缩掉第一维，复杂度是 $O(mk)$ 的。

> [!NOTE] **[例题 hdu 2639 Bone Collector II](http://acm.hdu.edu.cn/showproblem.php?pid=2639)**
> 
> 求 0-1 背包的严格第 $k$ 优解。$n \leq 100,v \leq 1000,k \leq 30$

> [!NOTE] **核心代码**
```cpp
memset(dp, 0, sizeof(dp));
int i, j, p, x, y, z;
scanf("%d%d%d", &n, &m, &K);
for (i = 0; i < n; i++) scanf("%d", &w[i]);
for (i = 0; i < n; i++) scanf("%d", &c[i]);
for (i = 0; i < n; i++) {
    for (j = m; j >= c[i]; j--) {
        for (p = 1; p <= K; p++) {
            a[p] = dp[j - c[i]][p] + w[i];
            b[p] = dp[j][p];
        }
        a[p] = b[p] = -1;
        x = y = z = 1;
        while (z <= K && (a[x] != -1 || b[y] != -1)) {
            if (a[x] > b[y])
                dp[j][z] = a[x++];
            else
                dp[j][z] = b[y++];
            if (dp[j][z] != dp[j][z - 1]) z++;
        }
    }
}
printf("%d\n", dp[m][K]);
```


## 习题

### 一般背包

> [!NOTE] **[AcWing 1024. 装箱问题](https://www.acwing.com/problem/content/1026/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可以把体积当成价值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int V, n, v;
    cin >> V >> n;
    vector<int> f(V + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> v;
        for (int j = V; j >= v; --j) f[j] = max(f[j], f[j - v] + v);
    }
    cout << V - f[V] << endl;
}
```

##### **Python**

```python
N = 20010
f = [0] * N
v = [0] * N

if __name__ == '__main__':
    m = int(input())
    n = int(input())
    for i in range(1, n + 1):
        v[i] = int(input())
    for i in range(1, n + 1):
        for j in range(m, v[i] - 1, -1):
            f[j] = max(f[j], f[j - v[i]] + v[i])
    print(m - f[m])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1022. 宠物小精灵之收服](https://www.acwing.com/problem/content/1024/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 需要注意的是：
> 
> $体积 w 与价值 v 是可逆的$ 
> $f[i] 表示体积为 i 时能装的最大价值，f[j] 表示价值为 j 所需的最小体积$ 
> $二者等价，借此优化时空复杂度$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 经典01背包思维**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;

int main() {
    int N, M, K;
    cin >> N >> M >> K;
    // 剩余 i 个球 j 体力 所能收服的最大小精灵个数
    vector<vector<int>> f(N + 1, vector<int>(M + 1));
    int c, w;
    for (int k = 1; k <= K; ++k) {
        // cost weight
        cin >> c >> w;
        for (int i = N; i >= c; --i)
            for (int j = M - 1; j >= w; --j)
                f[i][j] = max(f[i][j], f[i - c][j - w] +
                                           1);  // 【题目要求不能为0 故上界-1】
    }
    int res = -1, t;
    for (int i = 0; i <= N; ++i)
        for (int j = 0; j <= M; ++j)
            if (f[i][j] > res || (f[i][j] == res && j < t))
                res = f[i][j], t = j;
    cout << res << " " << M - t << endl;
}
```

##### **C++ 转换定义**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;

int main() {
    int N, M, K;
    cin >> N >> M >> K;
    int c, w;
    // 体力为 i 收集 j 个小精灵消耗的最小精灵球数量
    vector<vector<int>> f(M + 1, vector<int>(K + 1, inf));
    f[0][0] = 0;
    for (int k = 1; k <= K; ++k) {
        cin >> c >> w;
        for (int i = M; i >= w; --i)
            for (int j = k; j >= 1; --j)
                // if 判断
                if (f[i - w][j - 1] + c <= N)
                    f[i][j] = min(f[i][j], f[i - w][j - 1] + c);
    }
    for (int k = K; k >= 0; --k) {
        int t = inf;
        for (int i = 0; i < M; ++i)
            if (f[i][k] != inf) {
                t = i;
                break;
            }  // i < M 因为不能达到上界
        if (t != inf) {
            cout << k << " " << M - t << endl;
            break;
        }
    }
}
```

##### **Python**

```python
N = 1010
M = 510
K = 110
f = [[0] * N for _ in range(N)]
v1 = [0] * N
v2 = [0] * N

if __name__ == '__main__':
    V1, V2, n = map(int, input().split())
    for i in range(1, n + 1):
        v1[i], v2[i] = map(int, input().split())
    for i in range(1, n + 1):
        for j in range(V1, v1[i] - 1, -1):
            for k in range(V2 - 1, v2[i] - 1, -1):
                f[j][k] = max(f[j][k], f[j - v1[i]][k - v2[i]] + 1)
    print(f[V1][V2 - 1], end=' ')
    # 皮卡丘的体力值
    w = V2 - 1
    while w and f[V1][w - 1] == f[V1][V2 - 1]:
        w -= 1
    print(V2 - w)
```

##### **Python 压缩空间**

```python
n, m, k = map(int, input().split())

dp = [[0] * (n + 1) for _ in range(m + 1)]

while k:
    b, h = map(int, input().split())
    for j in range(n, b - 1, -1):
        for i in range(m - 1, h - 1, -1):
            dp[i][j] = max(dp[i][j], dp[i - h][j - b] + 1)
    k -= 1

print(dp[m - 1][n], end=' ')
k = m - 1
while (k > 0 and dp[k - 1][n] == dp[m - 1][n]):
    k -= 1
print(m - k, end=' ')
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 278. 数字组合](https://www.acwing.com/problem/content/description/280/)**
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
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 10010;

int n, m;
int f[N];

int main() {
    cin >> n >> m;

    f[0] = 1;
    for (int i = 0; i < n; i++) {
        int v;
        cin >> v;
        for (int j = m; j >= v; j--) f[j] += f[j - v];
    }

    cout << f[m] << endl;

    return 0;
}
```

##### **Python**

```python
N = 110
M = 10010
f = [0] * M  # 涉及到方案数这一类求解的，不可达状态 就设置为 0.
a = [0] * N

if __name__ == '__main__':
    n, m = map(int, input().split())
    a[1:] = list(map(int, input().split()))
    f[0] = 1
    for i in range(1, n + 1):
        for j in range(m, a[i] - 1, -1):
            f[j] += f[j - a[i]]
    print(f[m])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1023. 买书](https://www.acwing.com/problem/content/1025/)**
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
#include <iostream>

using namespace std;

const int N = 1010;

int n;
int v[4] = {10, 20, 50, 100};
int f[N];

int main() {
    cin >> n;

    // 总钱为 i 有多少种买书方案
    f[0] = 1;
    for (int i = 0; i < 4; i++)
        for (int j = v[i]; j <= n; j++) f[j] += f[j - v[i]];

    cout << f[n] << endl;

    return 0;
}
```

##### **Python**

```python
N = 1010
f = [0] * N  # 注意：这里不能初始化为f=[1]*N  这道题求解的是方案数 
v = [0, 10, 20, 50, 100]

if __name__ == '__main__':
    n = int(input())
    f[0] = 1  # 需要初始化
    for i in range(1, len(v)):
        for j in range(v[i], n + 1):
            f[j] = f[j] + f[j - v[i]]
    print(f[n])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 532. 货币系统](https://www.acwing.com/problem/content/534/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 这道题 要先去分析挖掘性质：
>
> 1）每一个 $a_i$ 都可以被 $b_i$ 表示出来
>
> 2）猜测：在最优解中，$b_1,b_2...,b_m$ 一定都是从 $a_1,a_2,...,a_n$ 中选择出来的（用反证法证明）
>
> 3）$b_1,b_2...,b_m$ 一定不能被其他 $b_i$ 表示出来
>
> 先排序，大的数一定是由前面小的数表示出来的（本题没有负数）
>
> 对于任何一个 $a_i$, 如果他不能被前面的数表示出来，如果他可以被表示出来，那这个数一定不能选；如果不能被表示，就一定要选。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 25010;

int n;
int a[N];
bool f[N];

int main() {
    int T;
    cin >> T;
    while (T--) {
        cin >> n;
        for (int i = 0; i < n; i++) cin >> a[i];
        sort(a, a + n);

        int m = a[n - 1];
        memset(f, 0, sizeof f);
        f[0] = true;

        int k = 0;
        for (int i = 0; i < n; i++) {
            if (!f[a[i]]) k++;
            for (int j = a[i]; j <= m; j++) f[j] |= f[j - a[i]];
        }

        cout << k << endl;
    }

    return 0;
}
```

##### **Python**

```python
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        n = int(input())
        a = list(map(int, input().split()))

        a.sort()
        m = a[-1]
        f = [False] * (m + 1)
        f[0] = True

        k = 0
        for c in a:
            if not f[c]:
                k += 1
            for j in range(c, m + 1):
                f[j] += f[j - c]
        print(k)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu kkksc03考前临时抱佛脚](https://www.luogu.com.cn/problem/P2392)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 均分

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 22;

int s[4];
int a[4][N], t[4];
int f[4][1500];

int main() {
    for (int i = 0; i < 4; ++ i )
        cin >> s[i];
    
    for (int i = 0; i < 4; ++ i )
        for (int j = 0; j < s[i]; ++ j )
            cin >> a[i][j], t[i] += a[i][j];
    
    int res = 0;
    for (int i = 0; i < 4; ++ i ) {
        // 01 背包
        for (int j = 0; j < s[i]; ++ j )
            for (int k = t[i] >> 1; k >= a[i][j]; -- k )
                f[i][k] = max(f[i][k], f[i][k - a[i][j]] + a[i][j]);
        res += t[i] - f[i][t[i] >> 1];
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

> [!NOTE] **[Luogu 5倍经验日](https://www.luogu.com.cn/problem/P1802)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **01背包简单变形**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 0/1背包变形

const int N = 1010;

int n, m;
int a[N], b[N], c[N], f[N];

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; ++ i )
        cin >> a[i] >> b[i] >> c[i];
    
    for (int i = 0; i < n; ++ i ) {
        // 打赢或打输
        // =
        for (int j = m; j >= c[i]; -- j )
            f[j] = max(f[j] + a[i], f[j - c[i]] + b[i]);
        // 打输
        // +
        for (int j = c[i] - 1; j >= 0; -- j )
            f[j] += a[i];
            // f[j] = max(f[j], f[j] + a[i]);
    }
    cout << 5ll * f[m] << endl;
    
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

> [!NOTE] **[Luogu 尼克的任务](https://www.luogu.com.cn/problem/P1280)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 排序 贪心 背包

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e4 + 10;

int n, k;
struct Ks {
    int k, l;
} ks[N];
int s[N], f[N];

int main() {
    cin >> n >> k;
    
    for (int i = 0; i < k; ++ i ) {
        cin >> ks[i].k >> ks[i].l;
        s[ks[i].k] ++ ;    // ATTENTION
    }
    
    // ATTENTION
    sort(ks, ks + k, [](const Ks & a, const Ks & b) {
        return a.k > b.k;
    });
    
    int p = 0;  // 任务从晚到早遍历
    for (int i = n; i > 0; -- i ) {
        if (s[i] == 0)
            f[i] = f[i + 1] + 1;
        else
            // 找出选择哪一个本时刻的任务使空闲时间最大化
            for (int j = 0; j < s[i]; ++ j) {
                if (f[i + ks[p].l] > f[i])
                    f[i] = f[i + ks[p].l];
                p ++ ;
            }
    }

    cout << f[1] << endl;
    
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

> [!NOTE] **[AcWing 1023. 买书](https://www.acwing.com/problem/content/1025/)**
>
> 题意: TODO

> [!TIP] **思路**
>
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```C++ 
```

##### **Python-朴素dp**

```python
# 朴素dp
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # 剪枝
        if sum(nums) % 2 == 1 or len(nums) == 1:return False
        n, v = len(nums), int(sum(nums) / 2)

        f = [[False] * (v + 1) for _ in range(n + 1)]
        f[i][0] = True
        for i in range(1, n + 1):
            for j in range(v + 1):
                if j < nums[i - 1]:
                    f[i][j] = f[i - 1][j]
                else:
                    f[i][j] = f[i - 1][j - nums[i - 1]] or f[i - 1][j]
        return f[n][v]
```

##### **Python-dp优化**

```python
# 滚动数组优化
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # 剪枝
        if sum(nums) % 2 == 1 or len(nums) == 1:return False
        n, v = len(nums), int(sum(nums) / 2)

        f = [[False] * (v + 1) for _ in range(2)]
        f[0][0] = True
        for i in range(1, n + 1):
            for j in range(v + 1):
                if j < nums[i - 1]:
                    f[i & 1][j] = f[(i - 1) & 1][j]
                else:
                    f[i & 1][j] = f[(i - 1) & 1][j - nums[i - 1]] or f[(i - 1) & 1][j]
        return f[n & 1][v]
   
 # 空间压缩
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
      	# 剪枝
        if sum(nums) % 2 == 1 or len(nums) == 1:return False
        n, v = len(nums), int(sum(nums) / 2)
        
        f = [False] * (v + 1)
        f[0] = True

        for i in range(n):
            for j in range(v, nums[i - 1] - 1, -1):
                f[j] = f[j - nums[i - 1]] or f[j]
        return f[v]
```

##### en-us

This problem is essentially let us to select several numbers in a set which are able to sum to a specific value(in this problem, the value is $sum/2$).

Actually, this is a 0/1 knapsack problem. 

1. State define:
  
   Let us assume $f[i][j]$ means whether the specific sum $j$ can be gotten from the first $i$ numbers. 

   If we can pick such a series of numbers from $0-i$ whose sum is $j$, $fi][j]$ is true, otherwise it is false.

2. Transition function:
  
   For each number, we can pick it or not.

   1) If we don't pick it: $f[i][j] = f[i-1][j]$, which means if the first $i-1$ element has made it to $j$, $f[i][j]$ would als make it to $j$, and we can just ignore $nums[i]$
   2) If we pick nums[i]: $f[i][j] = f[i-1][j-nums[i]]$, which represents that $j$ is composed of the current value $nums[i]$ and the remaining composed of other previous numbers. 

   Thus, the transition function is $f[i][j] = f[i-1][j] || f[i-1][j-nums[i]]$

3. Base case: 
  
   $f[0][0] = True$ ; (zero number consists of sum 0 is true)

* * *

> [!NOTE] **[LeetCode 1449. 数位成本和为目标值的最大数字](https://leetcode-cn.com/problems/form-largest-integer-with-digits-that-add-up-to-target/)**
> 
> 题意: 
> 
> 每一个数字有其消耗，在总消耗为target的情况下选出【最大整数】（其实就是选出的数字个数最多 且这些数字组合起来最大）

> [!TIP] **思路**
> 
> 完全背包，实现的时候选择数字最多的即可。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string largestNumber(vector<int>& cost, int target) {
        int n = cost.size();
        vector<int> dp(target + 1, -1);
        vector<int> fa(target + 1);
        dp[0] = 0;
        for (int i = 0; i < n; ++i) {
            int w = cost[i];
            for (int j = w; j <= target; ++j) {
                if (dp[j - w] != -1 && dp[j - w] + 1 >= dp[j]) {
                    dp[j] = dp[j - w] + 1;
                    fa[j] = i;
                }
            }
        }
        if (dp[target] == -1) return "0";
        string s;
        for (int i = target; i; i -= cost[fa[i]]) s += '1' + fa[i];
        sort(s.begin(), s.end(), greater<char>());
        return s;
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

> [!NOTE] **[Codeforces B. Color the Fence](https://codeforces.com/problemset/problem/349/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 自己写背包写过
> 
> 官方题解和 luogu 都是模拟的做法
> 
> **模拟思路有技巧**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Color the Fence
// Contest: Codeforces - Codeforces Round #202 (Div. 2)
// URL: https://codeforces.com/problemset/problem/349/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 显然 首先要求个数最多 再求最多的时候都有哪些数
// 个数相同时大的数越多越好 最后排序输出即可
//
// luogu 有模拟的做法 优先选最多的 然后挨个替换
// https://www.luogu.com.cn/problem/solution/CF349B

const int N = 1000010;

int v;
int a[11];
int f[N];
int p[N];

int main() {
    cin >> v;
    for (int i = 1; i <= 9; ++i)
        cin >> a[i];

    // 完全背包
    for (int i = 1; i <= 9; ++i)
        for (int j = a[i]; j <= v; ++j) {
            // f[j] = max(f[j], f[j - a[i]] + 1);
            int t = f[j - a[i]] + 1;
            if (t >= f[j]) {
                p[j] = i;
                f[j] = t;
            }
        }

    vector<int> ve;
    int id = v, x = p[id];
    while (x) {
        ve.push_back(x);
        id -= a[x];
        x = p[id];
    }

    if (ve.empty()) {
        cout << -1 << endl;
    } else {
        sort(ve.begin(), ve.end());
        for (int i = ve.size() - 1; i >= 0; --i)
            cout << ve[i];
        cout << endl;
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

### 二维背包

> [!NOTE] **[AcWing 1020. 潜水员](https://www.acwing.com/problem/content/description/1022/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> [关于解释和一些初始化的讨论](https://www.acwing.com/solution/content/7438/)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <cstring>
#include <iostream>

using namespace std;

const int N = 22, M = 80;

int n, m, K;
int f[N][M];

int main() {
    cin >> n >> m >> K;

    memset(f, 0x3f, sizeof f);
    f[0][0] = 0;

    while (K--) {
        int v1, v2, w;
        cin >> v1 >> v2 >> w;
        for (int i = n; i >= 0; i--)
            for (int j = m; j >= 0; j--)
                f[i][j] = min(f[i][j], f[max(0, i - v1)][max(0, j - v2)] + w);
    }

    cout << f[n][m] << endl;

    return 0;
}
```

##### **Python**

```python
# 二维费用背包问题：题意转换为：如何选第一个>=m, 第二个>=n的最小值
# 状态表示：f[i,j,k]：从前i个物品选，使得氧气含量至少是j, 氮气含量至少是k的所有选法；属性：Min
# 状态转移：找第i个物品的不同: a. 包含物品i & b.不包含物品i
# a:就是从1-(i-1)里选，f[i-1,j,k]
# b:包含i,分成两个步骤：1）去掉第i个物品 2）先求前面的最小值 3）再加上物品i的重量
# 左边： f[i-1,j-v1,k-v2] + 右边i的重量 wi
# 初始化：f[0,0,0]=0, f[0,j,k]=+无穷（和之前的“恰好”的初始化和状态转移 一模一样，但是上节课的代码过不去。。所以是哪里不一样？！ 当前方法才可以ac)
# 不同点：算状态的时候，一点要保证j>=v1 并且 k>=v2


# 背包状态表示：1）体积最多是j； 2）体积恰好是j； 3)体积至少是j
# 所有的背包问题 都可以从“不超过”转化为“恰好是”===> 对于代码的唯一的区别就是在初始化的不同。 所有的动态规划方程  都需要从实际含义出发，尤其是初始化的时候，什么都不选的时候的方案


N = 22
M = 80
K = 1010
f = [[float('inf')] * M for _ in range(M)]
a = [0] * K
b = [0] * K
c = [0] * K

if __name__ == '__main__':
    n, m = map(int, input().split())
    k = int(input())
    for i in range(1, k + 1):
        a[i], b[i], c[i] = map(int, input().split())
    f[0][0] = 0
    for i in range(1, k + 1):
        for j in range(n, -1, -1):
            for k in range(m, -1, -1):
                f[j][k] = min(f[j][k], f[max(0, j - a[i])][max(0, k - b[i])] + c[i])
    print(f[n][m])

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)**
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
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        for (auto& str: strs) {
            int a = 0, b = 0;
            for (auto c: str)
                if (c == '0') a ++ ;
                else b ++ ;
            for (int i = m; i >= a; i -- )
                for (int j = n; j >= b; j -- )
                    f[i][j] = max(f[i][j], f[i - a][j - b] + 1);
        }
        return f[m][n];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> dp(m+1, vector<int>(n+1));
        for (auto s : strs) {
            int zero = 0, one = 0;
            for (auto c : s) {
                if (c == '0') ++ zero ;
                else ++one;
            }
            for (int i = m; i >= zero; -- i )
                for (int j = n; j >= one; -- j )
                    dp[i][j] = max(dp[i][j], dp[i - zero][j - one] + 1);
        }
        return dp[m][n];
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

### 分组背包

> [!NOTE] **[AcWing 487. 金明的预算方案](https://www.acwing.com/problem/content/description/489/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二进制优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ dfs TLE**

dfs 形式在深度较多且数值较大时 TLE

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    cin >> m >> n;
    vector<int> v(n + 1), w(n + 1), q(n + 1);
    vector<vector<int>> es(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> v[i] >> w[i] >> q[i];
        // 必须有一个统一的树根，所以把所有不依赖别的节点的父节点都设置为0
        es[q[i]].push_back(i);
    }
    vector<vector<int>> f(n + 1, vector<int>(m + 1));
    function<void(int)> dfs = [&](int x) {
        for (int j = m; j >= v[x]; --j) f[x][j] = w[x] * v[x];
        for (auto& y : es[x]) {
            dfs(y);
            for (int j = m; j >= v[x]; --j)
                for (int k = 0; k <= j - v[x]; ++k)
                    f[x][j] = max(f[x][j], f[x][j - k] + f[y][k]);
        }
    };
    dfs(0);
    cout << f[0][m] << endl;
}
```


##### **C++ 倍缩优化**

倍缩可过【首先除10必定可行，再判断能否再除10（即所有w[i]是否全是100的倍数）】：

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    cin >> m >> n;
    vector<int> v(n + 1), w(n + 1), q(n + 1);
    vector<vector<int>> es(n + 1);
    m /= 10;  // 倍缩 降低dfs复杂度
    bool flag = true;
    for (int i = 1; i <= n; ++i) {
        cin >> v[i] >> w[i] >> q[i];
        if (v[i] % 100) flag = false;
        v[i] /= 10;
        // 必须有一个统一的树根，所以把所有不依赖别的节点的父节点都设置为0
        es[q[i]].push_back(i);
    }
    if (flag) {
        // 都是100的倍数 还可以再除10
        for (int i = 1; i <= n; ++i) v[i] /= 10;
        m /= 10;
    }
    vector<vector<int>> f(n + 1, vector<int>(m + 1));
    function<void(int)> dfs = [&](int x) {
        for (int j = m; j >= v[x]; --j) f[x][j] = w[x] * v[x];
        for (auto& y : es[x]) {
            dfs(y);
            for (int j = m; j >= v[x]; --j)
                for (int k = 0; k <= j - v[x]; ++k)
                    f[x][j] = max(f[x][j], f[x][j - k] + f[y][k]);
        }
    };
    dfs(0);
    if (flag)
        cout << f[0][m] * 100 << endl;
    else
        cout << f[0][m] * 10 << endl;
}
```

##### **C++**

 yxc 的枚举子集的办法，在每个物品的依赖数量较少以内没问题，但假设数量有 x 个，则对每个体积都需枚举 2^x 次方中子集方案，复杂度较高。

 **需关注数据范围**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#define v first
#define w second

using namespace std;

typedef pair<int, int> PII;

const int N = 60, M = 32010;

int n, m;
PII master[N];
vector<PII> servent[N];
int f[M];

int main() {
    cin >> m >> n;

    for (int i = 1; i <= n; i++) {
        int v, p, q;
        cin >> v >> p >> q;
        p *= v;
        if (!q)
            master[i] = {v, p};
        else
            servent[q].push_back({v, p});
    }

    for (int i = 1; i <= n; i++)
        for (int u = m; u >= 0; u--) {
            for (int j = 0; j < 1 << servent[i].size(); j++) {
                int v = master[i].v, w = master[i].w;
                for (int k = 0; k < servent[i].size(); k++)
                    if (j >> k & 1) {
                        v += servent[i][k].v;
                        w += servent[i][k].w;
                    }
                if (u >= v) f[u] = max(f[u], f[u - v] + w);
            }
        }

    cout << f[m] << endl;

    return 0;
}
```

##### **Python**

```python
# 分组背包问题：主件就是一个大组，组与组之间互斥。

if __name__ == '__main__':
    n, m = map(int, input().split())
    master, servent = [0 for _ in range(m + 1)], [[] for _ in range(m + 1)]
    for i in range(1, m + 1):
        v, p, q = map(int, input().split())
        if q == 0:
            master[i] = [v, v * p]
        else:
            servent[q].append([v, v * p])
    f = [0 for _ in range(n + 1)]

    for i in range(1, m + 1):
        if master[i] == 0:
            continue
        for j in range(n, -1, -1):
            # 使用二进制的思想枚举四种组合， 可以简化代码
            length = len(servent[i])
            for k in range(1 << length):  # 坑1:枚举2**n方
                v, w = master[i][0], master[i][1]  # 坑2:记得加上mater的体积和价值
                for u in range(length):
                    if k >> u & 1:  # 第u个物品被选了 才会进入到下面累加
                        v += servent[i][u][0]
                        w += servent[i][u][1]
                if j >= v:
                    f[j] = max(f[j], f[j - v] + w)
    print(f[n])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1013. 机器分配](https://www.acwing.com/problem/content/description/1015/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分组

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 非递归**

```cpp
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 11, M = 16;

int n, m;
int w[N][M];
int f[N][M];
int way[N];

int main() {
    // 体积相当于总数M
    cin >> n >> m;

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) cin >> w[i][j];

    for (int i = 1; i <= n; i++)
        // j 表示体积， w[i][j] 表示价值
        for (int j = 0; j <= m; j++)
            // 决策 选拿几个
            for (int k = 0; k <= j; k++)
                f[i][j] = max(f[i][j], f[i - 1][j - k] + w[i][k]);

    cout << f[n][m] << endl;

    int j = m;
    for (int i = n; i; i--)
        for (int k = 0; k <= j; k++)
            if (f[i][j] == f[i - 1][j - k] + w[i][k]) {
                way[i] = k;
                j -= k;
                break;
            }

    for (int i = 1; i <= n; i++) cout << i << ' ' << way[i] << endl;

    return 0;
}
```

##### **C++ 递归**

```cpp
#include <bits/stdc++.h>
using namespace std;

void dprint(vector<vector<int>>& g, int x, int y) {
    if (!x) return;
    int k = g[x][y];
    dprint(g, x - 1, y - k);
    cout << x << " " << k << endl;
}

int main() {
    // 体积相当于总数M
    int N, M;
    cin >> N >> M;
    vector<int> f(M + 1), w(M + 1);
    vector<vector<int>> g(N + 1, vector<int>(M + 1));
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= M; ++j) cin >> w[j];
        // j 表示体积， w[j] 表示价值
        for (int j = M; j >= 1; --j)
            // 决策 选拿几个
            for (int k = 1; k <= j; ++k) {
                int v = f[j - k] + w[k];
                if (v > f[j]) f[j] = v, g[i][j] = k;
            }
    }
    cout << f[M] << endl;
    dprint(g, N, M);
}
```

##### **Python**

```python
# 分组背包以及背包方案问题的应用
# 分组背包问题的实质：本身是一个组合数的问题（有些限制的组合数：物品之间有很多种互斥的选法）
# 逻辑转化：每个公司当成物品组；公司1:3个物品，第一个物品：分配1台（v:1,w1:30)；第二个物品：分配2台（v:2,w2:40)；第三个物品：分配3台（v:2,w3:50)
N = 20
f = [0] * N
w = [[0] * N for _ in range(N)]
s = [0] * N
p = [[0] * N for _ in range(N)]

if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):
        nums = list(map(int, input().split()))
        for j, val in enumerate(nums):
            w[i][j + 1] = val
    for i in range(1, n + 1):
        for j in range(m, -1, -1):
            for k in range(j + 1):
                t = f[j - k] + w[i][k]
                if t > f[j]:
                    f[j] = t
                    p[i][j] = k

    print(f[m])


    # 递归求解：（当数据过多，存在爆栈的风险）
    def pri(i, j):
        if (i == 0):
            return
        k = p[i][j]
        pri(i - 1, j - k)
        print(i, k)


    pri(n, m)

    # 推荐用while循环来打印方案，如果需要正序输出，就用一个数组保存，最后再输出
    i = n;
    v = m
    res = []
    while i >= 1:
        k = p[i][v]
        if k >= 0:
            res.append([i, k])
            v -= k
        i -= 1
    for i in range(len(res) - 1, -1, -1):
        print(res[i][0], res[i][1])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu [NOIP2012 普及组] 摆花](https://www.luogu.com.cn/problem/P1077)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分组背包即可
> 
> 重点在于部分情况下分组背包可以前缀和优化
> 
> 以及【生成函数】解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 110, MOD = 1000007;

int n, m;
int a[N];
int f1[N], s[N];

// 分组背包即可
void func1() {
    f1[0] = 1;
    for (int i = 1; i <= n; ++ i )
        for (int j = m; j >= 0; -- j )
            for (int k = 1; k <= a[i]; ++ k )
                if (k <= j)
                    f1[j] = (f1[j] + f1[j - k]) % MOD;
    cout << f1[m] << endl;
}

// func1 的前缀和优化
void func1_more() {
    s[0] = 1;   // think
    for (int i = 1; i <= m; ++ i )
        s[i] += s[i - 1];   // always 1
    f1[0] = 1;
    for (int i = 1; i <= n; ++ i ) {
        for (int j = m; j >= 1; -- j ) {    // 需要修改为 [1, m]
            int bound = j - a[i] - 1;       // 左侧
            if (bound >= 0)
                f1[j] = (f1[j] + s[j - 1] - s[bound] + MOD) % MOD;
            else
                f1[j] = (f1[j] + s[j - 1]) % MOD;
        }
        for (int j = 1; j <= m; ++ j )
            s[j] = (s[j - 1] + f1[j]) % MOD;
    }
    cout << f1[m] << endl;
}

// 生成函数
// https://www.luogu.com.cn/blog/76228/ti-xie-p1077-bai-hua-post
void func2() {
    // TODO
}

int main() {
    cin >> n >> m;
    
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    // func1();
    func1_more();
    // func2();

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

> [!NOTE] **[LeetCode 1155. 掷骰子的N种方法](https://leetcode-cn.com/problems/number-of-dice-rolls-with-target-sum/)** TAG
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
    // 分组背包
    const static int MOD = 1e9 + 7;
    int numRollsToTarget(int d, int f, int target) {
        vector<int> ff(target + 1);
        ff[0] = 1;
        for (int i = 1; i <= d; ++ i )
            for (int j = target; j >= 0; -- j ) {   // j >= 0 因为后面可能用到 ff[0]
                ff[j] = 0;                          // ATTENTION 一维状态下必须初始化为 0
                for (int k = 1; k <= f; ++ k )
                    if (j - k >= 0)
                        ff[j] = (ff[j] + ff[j - k]) % MOD;
            }
                
        return ff[target];
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

### 有依赖背包 树

> [!NOTE] **[AcWing 10. 有依赖的背包问题](https://www.acwing.com/problem/content/description/10/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 【要选 $y$ 必选其依赖 $x$ 】如选课模型等 结合树形DP。
> 
> 【分组背包是循环体积再循环选择组内物品（本质是选择子体积），选择只选一个】
> 
> 对于每一个节点选择可能有 $2^X$ 种可能，简化为体积，每个体积作为一件物品 =》每个节点作为一个分组。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> v, w;
vector<vector<int>> f, es;

int N, V;

void dfs(int x) {
    // 点 x 必选 故初始化包含 x 的价值
    for (int i = v[x]; i <= V; ++i) f[x][i] = w[x];
    for (auto& y : es[x]) {
        dfs(y);
        // j 范围   小于 v[x] 无法选择节点x
        for (int j = V; j >= v[x]; --j)
            // 分给子树y的空间不能大于 j-v[x] 否则无法选择物品x
            for (int k = 0; k <= j - v[x]; ++k)
                f[x][j] = max(f[x][j], f[x][j - k] + f[y][k]);
    }
}

int main() {
    int p, root;
    cin >> N >> V;
    v.resize(N + 1);
    w.resize(N + 1);
    f.resize(N + 1, vector<int>(V + 1));
    es.resize(N + 1);

    for (int i = 1; i <= N; ++i) {
        cin >> v[i] >> w[i] >> p;
        if (p == -1)
            root = i;
        else
            es[p].push_back(i);
    }
    dfs(root);
    cout << f[root][V] << endl;
}
```

##### **Python**

```python
# 用递归的思路考虑，框架是树形dp的模板
# 状态表示：f[u,j]:表示所有以u为根的子树中选，体积不超过j的方案；属性：Max
# 状态转移：用新的划分方式：以体积来进行划分!!!（从 每个子树选or不选，优化成 体积为0-m的划分）k个子树，选or不选，时间复杂度2**k 
# 每棵子树都分为0-m种情况，分别表示：体积是0,...m的价值是多少（一共有m+1类，求每一类里取最大价值就可以）
# 这道题转化为了：一共有n颗子树，每个子树内 可以选择用多大的体积，问总体积不超过j的最大价值是多少。
# 总结：可以把每个子树看成一个物品组，每个组内部有m+1个物品，第0个物品表示体积是0，价值f[0],第m个类表示体积是m,价值是f[m]。每个物品组只能选一个出来。
# dp分析精髓： 用某一个数表示一类；这里就是用不同的体积表示一大类，

# 分组背包问题：一定要记住：先循环组数，再循环体积，再循环决策（选某个组的哪个物品）

# 如果会爆栈的话，可以加入以下代码段
# import sys
# limit = 10000
# sys.setrecursionlimit(limit)

N = 110
h = [-1] * N
ev = [0] * N
ne = [0] * N
idx = 0
v = [0] * N
w = [0] * N
f = [[0] * N for _ in range(N)]


def dfs1(u):
    global m
    i = h[u]
    while i != -1:  # 1 go thru every sub-tree
        son = ev[i]
        dfs(ev[i])
        # group knapsack template
        # 1 travel group == every sub-tree
        # 2 travel volumn of knapsack
        for j in range(m - v[u], -1, -1):  # 必须要加上根节点,所有为根节点留出空间
            # 3 travel options 枚举当前用到的背包容量
            for k in range(0, j + 1):
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])
        i = ne[i]
    # 将当前点root加进来
    for i in range(m, v[u] - 1, -1):
        f[u][i] = f[u][i - v[u]] + w[u]
    # 当容量小于根节点v[root],则必然放不下，最大值都是0
    for i in range(v[u]):
        f[u][i] = 0


def dfs(u):
    global m  # 递归时 m需要随着变化，所以需要加global关键字
    for i in range(v[u], m + 1):  # 点u必须选，才能选子节点，所以初始化f[u][v[u]~m] = w[u]
        f[u][i] = w[u]
    i = h[u]
    while i != -1:
        son = ev[i]
        dfs(ev[i])

        for j in range(m, v[u] - 1, -1):  # j的范围为v[x]~m, 小于v[x]无法选择以x为子树的物品
            for k in range(0, j - v[u] + 1):  # 分给子树y的空间不能大于j-v[x],不然都无法选根物品x
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])
        i = ne[i]  # 踩坑：一定要记得写啊！！！


def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


if __name__ == '__main__':
    n, m = map(int, input().split())
    root = 0
    for i in range(1, n + 1):
        v[i], w[i], p = map(int, input().split())
        if p == -1:
            root = i
        else:
            add_edge(p, i)
    dfs(root)
    print(f[root][m])  # f[i][m] 前i个物品，体积<=m，的最大价值
```

<!-- tabs:end -->
</details>

<br>

* * *

### 背包求方案

> [!NOTE] **[AcWing 11. 背包问题求方案数](https://www.acwing.com/problem/content/description/11/)**
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
#include <cstring>
#include <iostream>

using namespace std;

const int N = 1010, mod = 1e9 + 7;

int n, m;
int f[N], g[N];

int main() {
    cin >> n >> m;

    memset(f, -0x3f, sizeof f);
    f[0] = 0;
    g[0] = 1;

    for (int i = 0; i < n; i++) {
        int v, w;
        cin >> v >> w;
        for (int j = m; j >= v; j--) {
            int maxv = max(f[j], f[j - v] + w);
            int s = 0;
            if (f[j] == maxv) s = g[j];
            if (f[j - v] + w == maxv) s = (s + g[j - v]) % mod;
            f[j] = maxv, g[j] = s;
        }
    }

    int res = 0;
    for (int i = 1; i <= m; i++)
        if (f[i] > f[res]) res = i;

    int sum = 0;
    for (int i = 0; i <= m; i++)
        if (f[i] == f[res]) sum = (sum + g[i]) % mod;

    cout << sum << endl;

    return 0;
}
```

##### **Python**

```python
# 所有的背包问题 都可以转化为 最短路问题。
# 起点是[0,0] 终点是[n,m] 
# 求最优解的方案数 等价于 求解 最短路径的条数。

# 第一种状态表示：
# g[i,j]表示 f[i,j]对应的方案数。(f[i,j]表示的是体积 【恰好】 是j的方案数)
# f[i,j]: 左边：不选i ,f[i-1,j]; 右边：选i, f[i-1,j-v[i]]+w[i]
# 对应的g[i][j]: 如果f[i][j]计算的时候 左边大：g[i][j]=g[i-1][j]；
#               如果右边大，那g[i][j]=g[i-1][j-v[i];
#               如果相等：g[i-1][j]+g[i-1][j-v[i]
# g[i,j]可以优化成一维。
if __name__ == "__main__":
    MOD = int(1e9) + 7
    n, m = map(int, input().split())

    f = [float('-inf')] * (m + 1)
    g = [0] * (m + 1)
    f[0] = 0
    g[0] = 1

    for i in range(n):
        v, w = map(int, input().split())
        for j in range(m, v - 1, -1):
            if f[j] < f[j - v] + w:
                f[j] = f[j - v] + w
                g[j] = g[j - v]
            elif f[j] == f[j - v] + w:
                g[j] = (g[j] + g[j - v]) % MOD
    # 注意：这里不能直接输出g[m], 因为比如 f[9]的值可能是最佳方案，但f[7]也可能是最佳方案

    v = 0  # 先求出最优方案的值是多少。
    for i in range(m + 1):
        v = max(v, f[i])

    res = 0
    for i in range(m + 1):
        if (f[i] == v):
            res = (res + g[i]) % MOD
    print(res)

# 第二种状态表示：
# f[j]: 表示体积 【不超过】j时的最大价值；和第一种状态表示的区别在于：初始化的不同。
if __name__ == "__main__":
    MOD = int(1e9) + 7
    n, m = map(int, input().split())

    f = [0] * (m + 1)  # f全部初始化为0
    g = [1] * (m + 1)  # g全部初始化为1，g[0][0]:表示啥都不放，是一种方案 初始化为1；g[0][1]：体积不超过1，是包含g[0][0]的，所以也可以初始化为1

    for i in range(n):
        v, w = map(int, input().split())
        for j in range(m, v - 1, -1):
            if f[j] < f[j - v] + w:
                f[j] = f[j - v] + w
                g[j] = g[j - v]
            elif f[j] == f[j - v] + w:
                g[j] = (g[j] + g[j - v]) % MOD
    # 在输出最优方案数时，直接输出即可         
    print(g[m])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 12. 背包问题求具体方案](https://www.acwing.com/problem/content/description/12/)**
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
#include <iostream>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N][N];

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++) cin >> v[i] >> w[i];

    for (int i = n; i >= 1; i--)
        for (int j = 0; j <= m; j++) {
            f[i][j] = f[i + 1][j];
            if (j >= v[i]) f[i][j] = max(f[i][j], f[i + 1][j - v[i]] + w[i]);
        }

    int j = m;
    for (int i = 1; i <= n; i++)
        if (j >= v[i] && f[i][j] == f[i + 1][j - v[i]] + w[i]) {
            cout << i << ' ';
            j -= v[i];
        }

    return 0;
}
```

##### **Python**

```python
#f[i,j]=max(f[i-1][j],f[i-1][j-v[i]+w[i])   f[n,m]
#求具体方案 其实：是判断出每个物品是否被选。需要倒推出来每一次的抉择 是第一项 还是第二项
#dp问题求方案对应的是：最短路问题。（从哪条路径走到f[n][m])

#注意如果不用数组存储状态的话：1）不能用状态压缩 2）去倒推如何走到f[n][m]
#看到“字典序最小方案” 不要害怕===> 这是因为出题人懒，只想有一个答案。（不是想难为大家）

#本题考点：1）f[i][j]状态定义需要倒序：之前的f(i,j)记录的都是前i个物品总容量为j的最优解，那么我们现在将f(i,j)定义为从第i个元素到最后一个元素总容量为j的最优解。接下来考虑状态转移：
#f(i,j)=max(f(i+1,j),f(i+1,j−v[i])+w[i])：两种情况，第一种是不选第i个物品，那么最优解等同于从第i+1个物品到最后一个元素总容量为j的最优解；第二种是选了第i个物品，那么最优解等于当前物品的价值w[i]加上从第i+1个物品到最后一个元素总容量为j−v[i]的最优解。
#2）用数组存储路径，可以压缩空间；3）如果是要求字典序最大，在判断的时候t>f[j]:f[j]=t （需要理解）

N = 1010
v = [0] * N
w = [0] * N
f = [0] * N  #用一个数组来存储状态，可以压缩空间
p = [[0] * N for _ in range(N)]  #用数组来存储状态

if __name__ == '__main__':
    n, m = map(int,input().split())
    for i in range(1, n + 1):
        v[i],w[i] = map(int,input().split())
    for i in range(n, 0, -1):
        for j in range(m, v[i] - 1, -1):
            t = f[j - v[i]] + w[i]
            if t >= f[j]:
                f[j] = t
                p[i][j] = True
   	j = m 
    for i in range(1, n + 1):
      if p[i][j]:
        	print(i, end = ' ')
          val -= v[i]
#    while i <= n:
#        if p[i][val]:
#            print(i, end = ' ')
#            val -= v[i]
#        i += 1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces B. Maximum Absurdity](https://codeforces.com/problemset/problem/332/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 记录路径方法 熟练度
> 
> 二维数组大小定义反了导致卡了数个小时

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Maximum Absurdity
// Contest: Codeforces - Codeforces Round #193 (Div. 2)
// URL: https://codeforces.com/problemset/problem/332/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

#define x first
#define y second

using LL = long long;
using PII = pair<int, int>;
const int N = 200010, M = 3;

// ATTENTION f g 二维数组大小定义反了，导致卡了数个小时
// 路径转移仍然依赖 g
// 有几个关键条件：
// 1. 状态转移依赖 【从 0 开始截至某个位置】的区间最值
//    因此需要记录【截至某个位置】的最值，使得减少一维循环
//    ==> 新开一个 g 数组
// 2. 需要记录方案，
//    使用 g 数组记录  同时借助 g 的更新条件
//    因为它的更新是真正新数值的更新
// 3. 记录更新路径 以及输出的写法
int n, k;
LL s[N];
LL f[M][N], g[M][N];
unordered_map<int, PII> p[M];

int main() {
    cin >> n >> k;
    for (int i = 1; i <= n; ++i)
        cin >> s[i], s[i] += s[i - 1];

    for (int i = 1; i <= 2; ++i) {
        for (int j = k * i; j <= n; ++j) {
            LL t = g[i - 1][j - k] + s[j] - s[j - k];
            if (t > f[i][j]) {
                f[i][j] = t;
            }
            // g[i][j] = max(f[i][j], g[i][j - 1]);
            if (f[i][j] > g[i][j - 1]) {
                g[i][j] = f[i][j];
                p[i][g[i][j]] = {i, j};
            } else
                g[i][j] = g[i][j - 1];
        }
    }

    vector<int> ve;
    LL v = g[2][n];
    for (int c = 2; c > 0; --c) {
        auto [x, y] = p[c][v];
        // cout << "v = " << v << " x = " << x << " y = " << y << endl;
        ve.push_back(y - k + 1);
        v -= (s[y] - s[y - k]);
    }
    for (int i = ve.size() - 1; i >= 0; --i)
        cout << ve[i] << " \n"[i == 0];

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

> [!NOTE] **[Codeforces C. Color Stripe](https://codeforces.com/problemset/problem/219/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单dp 记录方案

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Color Stripe
// Contest: Codeforces - Codeforces Round #135 (Div. 2)
// URL: https://codeforces.com/problemset/problem/219/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const int N = 500010, M = 27;

int n, k;
string s;
int f[N][M];
PII p[N][M];

int main() {
    cin >> n >> k >> s;

    memset(f, 0x3f, sizeof f);
    for (int i = 0; i < k; ++i)
        f[0][i] = 0;

    for (int i = 1; i <= n; ++i)
        for (int j = 0; j < k; ++j)
            for (int u = 0; u < k; ++u)
                if (j != u) {
                    int t = f[i - 1][u] + (s[i - 1] - 'A' != j);
                    if (t < f[i][j]) {
                        f[i][j] = t;
                        p[i][j] = {i - 1, u};
                    }
                }

    int res = 1e9, pj;
    for (int i = 0; i < k; ++i)
        if (f[n][i] < res) {
            res = f[n][i];
            pj = i;
        }
    cout << res << endl;

    string ss;
    int x = n, y = pj;
    while (x) {
        ss += 'A' + y;
        auto [nx, ny] = p[x][y];
        x = nx, y = ny;
    }
    reverse(ss.begin(), ss.end());
    cout << ss << endl;

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