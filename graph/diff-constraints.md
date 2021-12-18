> [!TIP] **总结**
> 
> 差分约束：
> 
> （1）求不等式组的可行解
>  
>    源点需要满足的条件：从源点出发，一定可以走到所有的边
> 
> 步骤：
> 
> [1] 先将每一个不等式 xi <= xj + ck 转化成一条从 xj 走到 xi，长度为 ck 的一条边
> 
> [2] 找到一个超级源点，使得该源点一定可以遍历到所有的边
> 
> [3] 从源点求一遍单元最短路
> 
>     结果1: 如果存在负环，则原不等式一定无解
> 
>     结果2: 如果没有负环，则 dist[i] 就是原不等式组的一个可行解
> 
> （2）如何求最大值或最小值，这里的最值指的是每个变量的最值
> 
> 结论：如果求的是最小值，则应该求最长路；如果求得是最大值，则应该求最短路
> 
> 问题：如何转化 xi <= c ，其中 c 是一个常数，这类的不等式。
> 
> 方法：建立一个超级源点 0 ，然后建立 0 -> i，长度是 c 的边即可。
> 
> 以求 xi 的最大值为例：求所有从 xi 触发，
> 
> 【构成的不等式链 xi <= xj + c1 <= xk + c2 + c1 <= ... <= c1+c2...】
> 
> 所计算出的上界，最终 xi 的最大值等于所有上界的最小值.


**差分约束系统** 是一种特殊的 $n$ 元一次不等式组，它包含 $n$ 个变量 $x_1,x_2,...,x_n$ 以及 $m$ 个约束条件，每个约束条件是由两个其中的变量做差构成的，形如 $x_i-x_j\leq c_k$，其中 $1 \leq i, j \leq n, i \neq j, 1 \leq k \leq m$ 并且 $c_k$ 是常数（可以是非负数，也可以是负数）。我们要解决的问题是：求一组解 $x_1=a_1,x_2=a_2,...,x_n=a_n$，使得所有的约束条件得到满足，否则判断出无解。

差分约束系统中的每个约束条件 $x_i-x_j\leq c_k$ 都可以变形成 $x_i\leq x_j+c_k$，这与单源最短路中的三角形不等式 $dist[y]\leq dist[x]+z$ 非常相似。因此，我们可以把每个变量 $x_i$ 看做图中的一个结点，对于每个约束条件 $x_i-x_j\leq c_k$，从结点 $j$ 向结点 $i$ 连一条长度为 $c_k$ 的有向边。

注意到，如果 $\{a_1,a_2,...,a_n\}$ 是该差分约束系统的一组解，那么对于任意的常数 $d$，$\{a_1+d,a_2+d,...,a_n+d\}$ 显然也是该差分约束系统的一组解，因为这样做差后 $d$ 刚好被消掉。

设 $dist[0]=0$ 并向每一个点连一条权重为 $0$ 边，跑单源最短路，若图中存在负环，则给定的差分约束系统无解，否则，$x_i=dist[i]$ 为该差分约束系统的一组解。

一般使用 Bellman-Ford 或队列优化的 Bellman-Ford（俗称 SPFA，在某些随机图跑得很快）判断图中是否存在负环，最坏时间复杂度为 $O(nm)$。


## 关键

### 问题解的存在性

1. 存在负环：路径中出现负环，就表示最短路可以无限小，即不存在最短路

   在SPFA实现过程中体现为某一点的入队次数大于节点数。（貌似可以用sqrt(num_node)来代替减少运行时间）

2. 终点不可达：没有约束关系
   
   代码实现过程中体现为 `dis[n - 1] = INF`


### 不等式组转化：

做题时可能会遇到不等式中的符号不相同的情况，但我们可以对它们进行适当的转化

（1）方程给出：$X[n-1]-X[0]>=T$ ,可以进行移项转化为： $X[0]-X[n-1]<=-T$。

（2）方程给出：$X[n-1]-X[0]<T$, 可以转化为 $X[n-1]-X[0]<=T-1$。

（3）方程给出：$X[n-1]-X[0]=T$，可以转化为 $X[n-1]-X[0]<=T$ && $X[n-1]-X[0]>=T$ ，再利用（1）进行转化即可


### 建模

对于不同的题目，给出的条件都不一样，我们首先需要关注问题是什么，

- 如果需要求的是两个变量差的最大值，那么需要将所有不等式转变成 $<=$ 的形式，建图后求最短路；
  
- 相反，如果需要求的是两个变量差的最小值，那么需要将所有不等式转化成 $>=$ ，建图后求最长路。


## 常用变形技巧

### 例题 [luogu P1993 小 K 的农场](https://www.luogu.com.cn/problem/P1993)

题目大意：求解差分约束系统，有 $m$ 条约束条件，每条都为形如 $x_a-x_b\geq c_k$，$x_a-x_b\leq c_k$ 或 $x_a=x_b$ 的形式，判断该差分约束系统有没有解。

|         题意         |                      转化                     |               连边              |
| :----------------: | :-----------------------------------------: | :---------------------------: |
| $x_a - x_b \geq c$ |             $x_b - x_a \leq -c$             |        `add(a, b, -c);`       |
| $x_a - x_b \leq c$ |              $x_a - x_b \leq c$             |        `add(b, a, c);`        |
|     $x_a = x_b$    | $x_a - x_b \leq 0, \space x_b - x_a \leq 0$ | `add(b, a, 0), add(a, b, 0);` |

跑判断负环，如果不存在负环，输出 `Yes`，否则输出 `No`。


```cpp

```

### 例题 [P4926\[1007\]倍杀测量者](https://www.luogu.com.cn/problem/P4926)

不考虑二分等其他的东西，这里只论述差分系统 $\frac{x_i}{x_j}\leq c_k$ 的求解方法。

对每个 $x_i,x_j$ 和 $c_k$ 取一个 $\log$ 就可以把乘法变成加法运算，即 $\log x_i-\log x_j \leq \log c_k$，这样就可以用差分约束解决了。

## Bellman-Ford 判负环代码实现

下面是用 Bellman-Ford 算法判断图中是否存在负环的代码实现，请在调用前先保证图是连通的。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
bool Bellman_Ford() {
  for (int i = 0; i < n; i++) {
    bool jud = false;
    for (int j = 1; j <= n; j++)
      for (int k = h[j]; ~k; k = nxt[k])
        if (dist[j] > dist[p[k]] + w[k])
          dist[j] = dist[p[k]] + w[k], jud = true;
    if (!jud) break;
  }
  for (int i = 1; i <= n; i++)
    for (int j = h[i]; ~j; j = nxt[j])
      if (dist[i] > dist[p[j]] + w[j]) return false;
  return true;
}
```

###### **Python**

```python
# Python Version
def Bellman_Ford():
    for i in range(0, n):
        jud = False
        for j in range(1, n + 1):
            while ~k:
                k = h[j]
                if dist[j] > dist[p[k]] + w[k]:
                    dist[j] = dist[p[k]] + w[k]; jud = True
                k = nxt[k]
        if jud == False:
            break
    for i in range(1, n + 1):
        while ~j:
            j = h[i]
            if dist[i] > dist[p[j]] + w[j]:
                return False
            j = nxt[j]
    return True
```

<!-- tabs:end -->
</details>

<br>

## 习题

[Usaco2006 Dec Wormholes 虫洞](https://loj.ac/problem/10085)

[「SCOI2011」糖果](https://loj.ac/problem/2436)

[POJ 1364 King](http://poj.org/problem?id=1364)

[POJ 2983 Is the Information Reliable?](http://poj.org/problem?id=2983)

> [!NOTE] **[AcWing 1169. 糖果](https://www.acwing.com/problem/content/1171/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 
> 不等式关系
> $$
>     w[a] >= 0   \\
>                 \\
> x == 1          \\
>     w[a] = w[b] \\
> x == 2          \\
>     w[a] < w[b] \\
> x == 3          \\
>     w[a] >= w[b]\\
> x == 4          \\
>     w[a] > w[b] \\
> x == 5          \\
>     w[a] <= w[b]\\
> $$
> 
> 求最长路
> 
> **之所以建边方向和 oi-wiki 不同，是因为这里求的最长路 而非最短路**
> **【一定要理解记忆】**


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 100010, M = 300010;

int n, m;
int h[N], e[M], w[M], ne[M], idx;
LL dist[N];
int q[N], cnt[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

bool spfa() {
    memset(dist, 0xcf, sizeof dist);
    
    dist[0] = 0;
    st[0] = true;
    cnt[0] = 0;
    int tt = 0;
    q[tt ++ ] = 0;
    
    while (tt) {
        int t = q[ -- tt];
        st[t] = false;
        
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] < dist[t] + w[i]) {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= n + 1)
                    return false;
                if (!st[j]) {
                    q[tt ++ ] = j;
                    st[j] = true;
                }
            }
        }
    }
    return true;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    
    for (int i = 1; i <= n; ++ i )
        add(0, i, 1);   // 边长为1

    while (m -- ) {
        int x, a, b;
        cin >> x >> a >> b;
        if (x == 1) {
            add(a, b, 0);
            add(b, a, 0);
        } else if (x == 2) {
            add(a, b, 1);
        } else if (x == 3) {
            add(b, a, 0);
        } else if (x == 4) {
            add(b, a, 1);
        } else {
            add(a, b, 0);
        }
    }
    
    if (!spfa())
        cout << -1 << endl;
    else {
        LL res = 0;
        for (int i = 1; i <= n; ++ i )
            res += dist[i];
        cout << res << endl;
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

> [!NOTE] **[AcWing 362. 区间](https://www.acwing.com/problem/content/364/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **寻找不等式**
> 
> 不等式关系
> $$
> s[i] <= s[i + 1]
> s[i + 1] - s[i] <= 1
> s[b] - s[a - 1] >= c
> $$
> 
> 最长路
> 
> ==>
> 
> $$
> s[i + 1] >= s[i]
> s[i] >= s[i + 1] - 1
> s[b] >= s[a - 1] + c
> $$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 50010, M = 150010;

int n;
int h[N], e[M], w[M], ne[M], idx;
int dist[N];
int q[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void spfa() {
    memset(dist, 0xcf, sizeof dist);
    dist[0] = 0;
    st[0] = true;
    int hh = 0, tt = 0; // 0, tt ++ 
    q[tt ++ ] = 0;
    while (hh != tt) {
        int t = q[hh ++ ];
        if (hh == N)
            hh = 0;
        st[t] = false;
        
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i], c = w[i];
            if (dist[j] < dist[t] + c) {
                dist[j] = dist[t] + c;
                if (!st[j]) {
                    q[tt ++ ] = j;
                    if (tt == N)
                        tt = 0;
                    st[j] = true;
                }
            }
        }
    }
}

int main() {
    memset(h, -1, sizeof h);
    for (int i = 0; i < N - 1; ++ i ) {
        add(i, i + 1, 0);
        add(i + 1, i, -1);
    }
    
    cin >> n;
    for (int i = 0; i < n; ++ i ) {
        int a, b, c;
        cin >> a >> b >> c;
        add(a - 1, b, c);
    }
    
    spfa();
    cout << dist[50001] << endl;
    
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

> [!NOTE] **[AcWing 1170. 排队布局](https://www.acwing.com/problem/content/1172/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 判断条件
> 
> spfa(n), spfa(1)
> 
> 最短路
> 
> $$
> s[b] - s[a] <= l
> s[b] - s[a] >= d
> s[b] - s[a] >= 0
> $$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1010, M = 21010, INF = 0x3f3f3f3f;

int n, ml, md;
int h[N], e[M], w[M], ne[M], idx;
int dist[N], q[N], cnt[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

bool spfa(int size) {
    int hh = 0, tt = 0;
    memset(dist, 0x3f, sizeof dist);
    memset(st, 0, sizeof st);
    memset(cnt, 0, sizeof cnt);
    
    for (int i = 1; i <= size; ++ i ) {
        q[tt ++ ] = i;
        dist[i] = 0;
        st[i] = true;
    }
    
    while (hh != tt) {
        int t = q[hh ++ ];
        if (hh == N)
            hh = 0;
        st[t] = false;
        
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[t] + w[i]) {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= n)
                    return true;
                if (!st[j]) {
                    q[tt ++ ] = j;
                    if (tt == N)
                        tt = 0;
                    st[j] = true;
                }
            }
        }
    }
    return false;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> ml >> md;
    
    for (int i = 1; i < n; ++ i )
        add(i + 1, i, 0);
    while (ml -- ) {
        int a, b, l;
        cin >> a >> b >> l;
        if (a > b)
            swap(a, b);
        add(a, b, l);
    }
    while (md -- ) {
        int a, b, d;
        cin >> a >> b >> d;
        if (a > b)
            swap(a, b);
        add(b, a, -d);
    }
    
    if (spfa(n))
        cout << -1 << endl;
    else {
        spfa(1);
        cout << (dist[n] == INF ? -2 : dist[n]) << endl;
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

> [!NOTE] **[AcWing 393. 雇佣收银员](https://www.acwing.com/problem/content/395/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **环状 复杂不等式转化**
> 
> 求最长路
> 
> $$
> s[i] - s[i - 1] >= 0
> s[i] - s[i - 1] <= num[i]
> 
> if i >= 8:
>     s[i] - s[i - 8] >= r[i]
> else:
>     s[i] + s[24] - s[24 - i] >= r[i]
>     ===>
>     s[i] + c - s[16 + i] >= r[i]
> $$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 30, M = 100, INF = 0x3f3f3f3f;

int n;
int h[N], e[M], w[M], ne[M], idx;
int r[N], num[N];   // r需求 num某个时刻开始工作的人
int dist[N], q[N], cnt[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void build(int c) {
    memset(h, -1, sizeof h);
    idx = 0;
    
    for (int i = 1; i <= 24; ++ i ) {
        add(i - 1, i, 0);
        add(i, i - 1, -num[i]);
    }
    for (int i = 1; i <= 7; ++ i )
        add(i + 16, i, r[i] - c);   // ATTENTION
    for (int i = 8; i <= 24; ++ i )
        add(i - 8, i, r[i]);
    // s[0] + c == s[24]
    add(0, 24, c);  // s[24] >= s[0] + c
    add(24, 0, -c); // s[0] >= s[24] - c
}

bool spfa(int c) {
    build(c);
    
    memset(dist, 0xcf, sizeof dist);
    memset(cnt, 0, sizeof cnt);
    memset(st, 0, sizeof st);
    
    int hh = 0, tt = 0;
    q[tt ++ ] = 0;
    dist[0] = 0;
    st[0] = true;
    
    while (hh != tt) {
        int t = q[hh ++ ];
        if (hh == N)
            hh = 0;
        st[t] = false;
        
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] < dist[t] + w[i]) {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= 25)
                    return false;
                if (!st[j]) {
                    q[tt ++ ] = j;
                    if (tt == N)
                        tt = 0;
                    st[j] = true;
                }
            }
        }
    }
    return true;
}

int main() {
    int T;
    cin >> T;
    while (T -- ) {
        for (int i = 1; i <= 24; ++ i )
            cin >> r[i];
        cin >> n;
        memset(num, 0, sizeof num);
        for (int i = 0; i < n; ++ i ) {
            int t;
            cin >> t;
            num[t + 1] ++ ;
        }
        
        // 枚举最后需要多少人 也即dist[24]
        bool success = false;
        for (int i = 0; i <= 1000; ++ i )
            if (spfa(i)) {
                cout << i << endl;
                success = true;
                break;
            }
        if (!success)
            cout << "No Solution" << endl;
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