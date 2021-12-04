

## 0-1 背包

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

##### **Python 朴素**

```python
#朴素写法：
#1. 状态表示：f[i][j]: 表示的是放i个物品，总体积不超过j的价值；
#2. 属性：f[i][j]的属性是：最大价值
#3. 状态转移：(曲线救国==> 第i个物品先去掉，看前i-1个物品的最大值怎么求）以第i个物品 “放” 还是 “不放” 作为区分来进行转移。
if __name__=='__main__':
    N = 1010
    v = [0] * N
    w = [0] * N
    f = [[0] * N for _ in range(N)]
    
    n,m = map(int,input().split())
    for i in range(1, n + 1):
        a, b = map(int,input().split())
        v[i] = a
        w[i] = b
    for i in range(1,n + 1):
        for j in range(m + 1):
            f[i][j] = f[i - 1][j]
            if j >= v[i]:
                f[i][j] = max(f[i][j],f[i - 1][j - v[i]] + w[i])
    print(f[n][m])
```

##### **Python 空间压缩**

```python
#空间压缩
#1. f[i]这一层在计算的时候只用到了f[i-1]这一层，所以可以用滚动数组来做；
#2. f[i][j],f[i][j-v[i]]这两个中的j这一维都是小于等于j的，在一侧（而不是在两侧），所以可以用一维来做
#3. 1) 直接删掉一维，然后对代码做等价变形 2）直接删掉i那一维度；j-v[i]<j 已经第i层计算过了，循环顺序变成从大到小枚举，就可以解决。那f[j-v[i]]存的就是i-1层的值。
if __name__=='__main__':
    N = 1010
    v = [0] * N
    w = [0] * N
    f = [0] * N
    
    n, m = map(int,input().split())
    for i in range(1, n + 1):
        v[i], w[i] = map(int,input().split())
    for i in range(1, n + 1):
        for j in range(m,v[i] - 1, -1):
                f[j] = max(f[j], f[j - v[i]] + w[i])
    print(f[m])
```

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

##### **Python 朴素优化**

```python
#朴素写法+优化代码
if __name__=='__main__':
    N=1010
    f=[[0] * N for _ in range(N)]
    v=[0] * N
    w=[0] * N
    
    n,m=map(int,input().split())
    for i in range(1, n + 1):
        a, b = map(int,input().split())
        v[i] = a
        w[i] = b
    
    for i in range(1, n + 1):
        for j in range(1 , m + 1):
            f[i][j] = f[i - 1][j]
            if j >= v[i]:
                f[i][j] = max(f[i - 1][j], f[i][j - v[i]] + w[i])
    print(f[n][m])
```

##### **Python 空间压缩**

```python
#空间压缩：
if __name__=='__main__':
    N=1010
    f=[0] * N
    v=[0] * N
    w=[0] * N
    
    n, m = map(int,input().split())
    for i in range(1, n + 1):
        a,b = map(int,input().split())
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
#完全背包问题可以用多重背包问题来优化，其实时间复杂度会变慢。
#为什么多重背包问题就需要用单调队列来进行优化，但是完全背包问题就不需要呢？
#核心原因在于：多重背包问题求的是滑动窗口内的最大值，那滑动窗口的最大值必须要用单调队列来进行优化
#感兴趣可以看 背包9讲（会讲得更详细）

# 状态表示f[i,j]: 1）集合：所有只从前i个物品中选，并且总体积不超过j的选法；2）属性：Max
# 状态计算：按照第i个物品选几个，把所有分发分成若干类。
# f[i,j]=max(f[i-1][j-k*v[i]]+w[i]*k)   k=0,1,2...,s[i]


#朴素写法；
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
                     f[i][j]=max(f[i][j],f[i - 1][j - k * v[i]] + k * w[i]) #这里已经包含了不选第i个物品的选法了。
    print(f[n][m])
		#本题数据范围比较小，所以可以不用优化就可以ac
```

<!-- tabs:end -->
</details>

<br>

* * *

### 二进制分组优化

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
# 第4题的优化，数据范围更大 N=1000；所以需要优化上述暴力算法。
# 不能用完全背包问题来进行优化，需要用经典优化：二进制的优化方式：
# 把若干个第i个物品打包分组，每组最多选一次，看是否能凑出所有的选法：1，2，4，8，..., 512 ===> 可以凑出0-1023

# 如果s=200， 用1，2，4，8，16，32，64，73就可以凑出0-200===> 就是把200个物品i 分成8组， 每组物品都只能选一次，可以凑出来200个物品。转化为0/1背包问题。（这里不能要128，因为从1加到128，等于255，我们没有那么多物品）
# 对于一般数来抽象出规律，再用代码实现。
# 1）把第i个物品的s个，拆成logs组的新物品：新物品只能用一次。
# 2）再对新物品 做一遍0/1背包问题就可以。时间复杂度从O(nvs)===> O(nvlogs)

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
# #本题数据范围比较大，需要对多重背包问题进行二进制优化。（优化成一维）
# #分析的时候，永远是事实实际情况出发！

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
# 二维费用背包问题====可以和所有背包问题都划分在一块。
# 状态表示：f[i,j,k]:所有只从前i个物品中选，并且总体积不超过j,总重量不超过k的选法；属性：最大值
# 状态转移：以最后一个物品 放 还是 不放。
# 不包含最后一个物品：f[i-1,j,k]
# 包含最后一个物品：f[i-1,j-v[i],k-m[i])+w[i]

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

这里要注意：**一定不能搞错循环顺序**，这样才能保证正确性。

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
# 状态表示f[i,j] ---1)集合：只能从前i组物品中选，且总体积不大于j的所有选法；2）属性：最大价值
# 状态转移（计算）：枚举第i组物品 选哪个？或者不选==> 1) 不选 2）选第i组的第1个物品,...,第k个物品..
# （上述转移的过程很像一根热狗，就叫 “热狗划分法”）
# 1)第i组物品不选：f[i-1,j]; 2）从第i组物品选第k个物品： f[i-1,j-v[k]]+w[i,k]


# 空间优化的时候：
# 如果用的是上一层的状态的话，就从大到小枚举；如果用的是本层的状态，就直接从小到大枚举就可以。
if __name__ == '__main__':
    N = 110
    f = [0] * N
    v = [[0] * N for _ in range(N)]
    w = [[0] * N for _ in range(N)]
    s = [0] * N

    n, m = map(int, input().split())
    for i in range(1, n + 1):  # 组别是从1开始的
        s[i] = int(input())
        for j in range(s[i]):  # 第i组物品里的物品 是从第0个开始的
            v[i][j], w[i][j] = map(int, input().split())

    for i in range(1, n + 1):
        for j in range(m, -1, -1):
            for k in range(s[i]):
                if v[i][k] <= j:
                    f[j] = max(f[j], f[j - v[i][k]] + w[i][k])
    print(f[m])

# 统一组别和物品都是从下标1开始的：
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