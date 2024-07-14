> [!NOTE] **ATTENTION**
> 
> - 返回值为是否成功进行拓扑排序，也即是否存在环。也就是说拓扑排序是可以用来简单地判环的。
> 
> - 有时会要求输出字典序最小的方案，这时把 `queue` 改成 `priority_queue` 即可，复杂度会多一个 $log$ 。

## 定义

拓扑排序的英文名是 Topological sorting。

拓扑排序要解决的问题是给一个图的所有节点排序。

我们可以拿大学选课的例子来描述这个过程，比如学习大学课程中有：单变量微积分，线性代数，离散数学概述，概率论与统计学概述，语言基础，算法导论，机器学习。当我们想要学习 算法导论 的时候，就必须先学会 离散数学概述 和 概率论与统计学概述，不然在课堂就会听的一脸懵逼。当然还有一个更加前的课程 单变量微积分。这些课程就相当于几个顶点 $u$, 顶点之间的有向边 $(u,v)$ 就相当于学习课程的顺序。显然拓扑排序不是那么的麻烦，不然你是如何选出合适的学习顺序。下面将介绍如何将这个过程抽象出来，用算法来实现。

但是如果某一天排课的老师打瞌睡了，说想要学习 算法导论，还得先学 机器学习，而 机器学习 的前置课程又是 算法导论，然后你就一万脸懵逼了，我到底应该先学哪一个？当然我们在这里不考虑什么同时学几个课程的情况。在这里，算法导论 和 机器学习 间就出现了一个环，显然你现在没办法弄清楚你需要学什么了，于是你也没办法进行拓扑排序了。因而如果有向图中存在环路，那么我们就没办法进行 拓扑排序 了。

因此我们可以说 在一个 [DAG（有向无环图）](./dag.md) 中，我们将图中的顶点以线性方式进行排序，使得对于任何的顶点 $u$ 到 $v$ 的有向边 $(u,v)$, 都可以有 $u$ 在 $v$ 的前面。

还有给定一个 DAG，如果从 $i$ 到 $j$ 有边，则认为 $j$ 依赖于 $i$。如果 $i$ 到 $j$ 有路径（$i$ 可达 $j$），则称 $j$ 间接依赖于 $i$。

拓扑排序的目标是将所有节点排序，使得排在前面的节点不能依赖于排在后面的节点。

## Kahn 算法

初始状态下，集合 $S$ 装着所有入度为 $0$ 的点，$L$ 是一个空列表。

每次从 $S$ 中取出一个点 $u$（可以随便取）放入 $L$, 然后将 $u$ 的所有边 $(u, v_1), (u, v_2), (u, v_3) \cdots$ 删除。对于边 $(u, v)$，若将该边删除后点 $v$ 的入度变为 $0$，则将 $v$ 放入 $S$ 中。

不断重复以上过程，直到集合 $S$ 为空。检查图中是否存在任何边，如果有，那么这个图一定有环路，否则返回 $L$，$L$ 中顶点的顺序就是拓扑排序的结果。

首先看来自 [Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm) 的伪代码

```text
L← Empty list that will contain the sorted elements
S ← Set of all nodes with no incoming edges
while S is non-empty do
    remove a node n from S
    insert n into L
    for each node m with an edge e from n to m do
        remove edge e from the graph
        if m has no other incoming edges then
            insert m into S
if graph has edges then
    return error (graph has at least onecycle)
else
    return L (a topologically sortedorder)
```

代码的核心是维持一个入度为 0 的顶点的集合。

可以参考该图

![topo](images/topo-example.svg)

对其排序的结果就是：2 -> 8 -> 0 -> 3 -> 7 -> 1 -> 5 -> 6 -> 9 -> 4 -> 11 -> 10 -> 12

### 时间复杂度

假设这个图 $G = (V, E)$ 在初始化入度为 $0$ 的集合 $S$ 的时候就需要遍历整个图，并检查每一条边，因而有 $O(E+V)$ 的复杂度。然后对该集合进行操作，显然也是需要 $O(E+V)$ 的时间复杂度。

因而总的时间复杂度就有 $O(E+V)$

### 实现

伪代码：

```text
bool toposort() {
    q = new queue();
    for (i = 0; i < n; i++)
        if (in_deg[i] == 0) q.push(i);
    ans = new vector();
    while (!q.empty()) {
        u = q.pop();
        ans.push_back(u);
    for each
        edge(u, v) {
            if (--in_deg[v] == 0) q.push(v);
        }
    }
    if (ans.size() == n) {
        for (i = 0; i < n; i++) std::cout << ans[i] << std::endl;
        return true;
    } else {
        return false;
    }
}
```

## DFS 算法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, m, u, v, t;

void add(int u, int v, vector<vector<int>>& G) { G[u].push_back(v); }

bool dfs(int u, vector<vector<int>>& G, vector<int>& c, vector<int>& topo) {
    c[u] = -1;
    for (auto v : G[u]) {
        if (c[v] < 0)
            return false;
        else if (!c[v] && !dfs(v, G, c, topo))
            return false;
    }
    c[u] = 1;
    // cout <<"u="<<u<<" topo[u]="<<t-1<<endl;
    topo[--t] = u;
    return true;
}

int main() {
    cin >> n >> m;
    vector<vector<int>> G(n + 1, vector<int>());
    while (m--) {
        cin >> u >> v;
        add(u, v, G);
    }
    t = n;
    vector<int> topo(n + 1);
    vector<int> c(n + 1, 0);
    bool f = true;
    for (int u = 1; u <= n; ++u)
        if (!c[u]) {
            if (!dfs(u, G, c, topo)) {
                f = false;
                break;
            }
        }

    if (f)
        for (int i = 0; i < n; ++i) cout << topo[i] << " ";
    else
        cout << -1 << endl;
}
```

##### **C++ 2**

```cpp
// C++ Version
vector<int> G[MAXN];  // vector 实现的邻接表
int c[MAXN];          // 标志数组
vector<int> topo;     // 拓扑排序后的节点

bool dfs(int u) {
    c[u] = -1;
    for (int v : G[u]) {
        if (c[v] < 0)
            return false;
        else if (!c[v])
            if (!dfs(v)) return false;
    }
    c[u] = 1;
    topo.push_back(u);
    return true;
}

bool toposort() {
    topo.clear();
    memset(c, 0, sizeof(c));
    for (int u = 0; u < n; u++)
        if (!c[u])
            if (!dfs(u)) return false;
    reverse(topo.begin(), topo.end());
    return true;
}
```

##### **Python**

```python
# Python Version
G = [] * MAXN
c = [0] * MAXN
topo = []

def dfs(u):
    c[u] = -1
    for v in G[u]:
        if c[v] < 0:
            return False
        elif c[v] == False:
            if dfs(v) == False:
                return False
    c[u] = 1
    topo.append(u)
    return True

def toposort():
    topo = []
    while u < n:
        if c[u] == 0:
            if dfs(u) == False:
                return False
        u = u + 1
    topo.reverse()
    return True
```

<!-- tabs:end -->
</details>

<br>

* * *


时间复杂度：$O(E+V)$ 空间复杂度：$O(V)$

### 合理性证明

考虑一个图，删掉某个入度为 $0$ 的节点之后，如果新图可以拓扑排序，那么原图一定也可以。反过来，如果原图可以拓扑排序，那么删掉后也可以。

### 应用

拓扑排序可以用来判断图中是否有环，

还可以用来判断图是否是一条链。

### 求字典序最大/最小的拓扑排序

将 Kahn 算法中的队列替换成最大堆/最小堆实现的优先队列即可，此时总的时间复杂度为 $O(E+V \log{V})$。

## 习题

[CF 1385E](https://codeforces.com/problemset/problem/1385/E)：需要通过拓扑排序构造。

## 参考

1. 离散数学及其应用。ISBN:9787111555391
2. <https://blog.csdn.net/dm_vincent/article/details/7714519>
3. Topological sorting,<https://en.wikipedia.org/w/index.php?title=Topological_sorting&oldid=854351542>


## 习题

### 一般应用

> [!NOTE] **[AcWing 848. 有向图的拓扑序列](https://www.acwing.com/problem/content/850/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100010;

int n, m;
int h[N], e[N], ne[N], idx;
int d[N];
int q[N];

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

bool topsort() {
    int hh = 0, tt = -1;

    for (int i = 1; i <= n; i++)
        if (!d[i]) q[++tt] = i;

    while (hh <= tt) {
        int t = q[hh++];

        for (int i = h[t]; i != -1; i = ne[i]) {
            int j = e[i];
            if (--d[j] == 0) q[++tt] = j;
        }
    }

    return tt == n - 1;
}

int main() {
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);

    for (int i = 0; i < m; i++) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);

        d[b]++;
    }

    if (!topsort())
        puts("-1");
    else {
        for (int i = 0; i < n; i++) printf("%d ", q[i]);
        puts("");
    }

    return 0;
}
```

##### **Python**

```python
"""
> 什么叫拓扑序列？ 针对有向图的才有拓扑序列（无向图没有）
- 简而言之，拓扑序列：对于每条边 都是起点在终点前面；
- 当把一张图按照拓扑序列排好后，所有的序列都是从前指向后的。这样就是形成了拓扑序列
- 并不是所有有向图 都有拓扑序列，比如存在环的图 一定不是拓扑序列!!!

- 可以证明：一个有向无环图，一定存在拓扑序列；有向无环图被称为拓扑图。
  - 基本概念：
    - 入度：对于一个节点，有多少边指向自己
    - 出度：对于一个节点，有多少边从自己这边指出去
- 一个无环图，一定至少存在一个入度位0的点（反证法 可以很好证明）；所有入度位0的点，都可以排在当前最前面的位置。

  - 拓扑排序算法思想：
    - 把所有入度位0的点 入队
    - 然后就是BFS的过程：拿出队头t，枚举t的所有出边t->j ， 删掉t-> j，删掉后，那j点的入度数要减少1：d[j]-=1；如果d[j]==0, 那j前面所有的点都已经放好了，那就让j入队。
    - 就结束了。如果 存在环的，那一定有点不会入队；如果一个图没有环的话，那所有的点都可以被突破掉，都可以入队。
"""


def add(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


def topsort():
    from collections import deque
    q = deque()
    for i in range(1, n + 1):
        if d[i] == 0:
            q.append(i)
    while q:
        t = q.popleft()
        res.append(t)
        i = h[t]
        while i != -1:
            j = ev[i]
            d[j] -= 1
            if d[j] == 0:
                q.append(j)
            i = ne[i]  # 踩坑：不要忘了往后遍历！！！
    return len(res) == n


if __name__ == '__main__':
    N = 10 ** 5 + 10
    h = [-1] * N
    ev = [0] * N
    ne = [0] * N
    idx = 0
    d = [0] * N

    n, m = map(int, input().split())
    for _ in range(m):
        a, b = map(int, input().split())
        add(a, b)
        d[b] += 1
    res = []
    if topsort():
        for i in range(n):
            print(res[i], end=' ')
    else:
        print('-1')
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1192. 奖金](https://www.acwing.com/problem/content/1194/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思考建边方向

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 10010, M = 20010;

int n, m;
int h[N], e[M], ne[M], idx;
int q[N], d[N];
int dist[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

bool topsort() {
    int hh = 0, tt = -1;
    for (int i = 1; i <= n; ++ i )
        if (!d[i]) q[ ++ tt] = i;
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (-- d[j] == 0) q[ ++ tt] = j;
        }
    }
    return tt == n - 1;
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        // b -> a 而不是 a -> b
        add(b, a);
        ++ d[a];
    }
    if (topsort()) {
        for (int i = 1; i <= n; ++ i ) dist[i] = 100;
        for (int i = 0; i < n; ++ i ) {
            int j = q[i];
            for (int k = h[j]; ~k; k = ne[k])
                dist[e[k]] = max(dist[e[k]], dist[j] + 1);
        }
        int res = 0;
        for (int i = 1; i <= n; ++ i ) res += dist[i];
        cout << res << endl;
    } else cout << "Poor Xed" << endl;
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

> [!NOTE] **[AcWing 164. 可达性统计](https://www.acwing.com/problem/content/166/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 30010, M = 30010;

int n, m;
int h[N], e[M], ne[M], idx;
int d[N], q[N];
// f[i] 所有从点i可以到达的点的集合 拓扑排序逆序求一遍集合并即可
bitset<N> f[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void topsort() {
    int hh = 0, tt = -1;
    for (int i = 1; i <= n; ++ i )
        if (!d[i]) q[ ++ tt] = i;
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (-- d[j] == 0) q[ ++ tt] = j;
        }
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        add(a, b);
        ++ d[b];
    }
    topsort();
    
    // 注意 逆序才能这样遍历K
    for (int i = n - 1; i >= 0; -- i ) {
        int j = q[i];
        f[j][i] = 1;
        for (int k = h[j]; ~k; k = ne[k])
            f[j] |= f[e[k]];
    }
    for (int i = 1; i <= n; ++ i ) cout << f[i].count() << endl;    // 1的个数
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

> [!NOTE] **[AcWing 1400. 堆叠相框](https://www.acwing.com/problem/content/1402/)**
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
// 拓扑排序
#include <bits/stdc++.h>
using namespace std;

const int N = 40;

int n, m;
char str[N][N];
bool g[N][N], st[N];
int d[N];
string path;

// 检查【边上】是否有覆盖 建边
void work(int x1, int y1, int x2, int y2, char c) {
    for (int i = x1; i <= x2; ++ i )
        for (int j = y1; j <= y2; ++ j )
            if (str[i][j] != c && str[i][j] != '.') {
                int a = c - 'A', b = str[i][j] - 'A';
                if (!g[a][b]) {
                    g[a][b] = true;
                    d[b] ++ ;
                }
            }
}

// s 所有入度为0的点的集合
void dfs(string s) {
    if (s.empty()) {
        cout << path << endl;
        return;
    }
    
    sort(s.begin(), s.end());
    for (int i = 0; i < s.size(); ++ i ) {
        char c = s[i];
        path += c;
        string w = s.substr(0, i) + s.substr(i + 1);
        // 加入删完i后新的入度为0的点
        for (int j = 0; j < 26; ++ j )
            if (g[c - 'A'][j] && -- d[j] == 0)
                w += j + 'A';
        dfs(w);
        // 恢复
        for (int j = 0; j < 26; ++ j )
            if (g[c - 'A'][j])
                d[j] ++ ;
        path.pop_back();
    }
}

void topsort() {
    string s;
    for (int i = 0; i < 26; ++ i )
        if (st[i] && !d[i])
            s += i + 'A';
    dfs(s);
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; ++ i ) cin >> str[i];
    for (char i = 'A'; i <= 'Z'; ++ i ) {
        int x1 = N, x2 = -N, y1 = N, y2 = -N;
        for (int j = 0; j < n; ++ j )
            for (int k = 0; k < m; ++ k )
                if (str[j][k] == i) {
                    x1 = min(x1, j), x2 = max(x2, j);
                    y1 = min(y1, k), y2 = max(y2, k);
                }
        if (x1 == N) continue;
        // 检查四条边 看是否有覆盖
        work(x1, y1, x1, y2, i);
        work(x1, y1, x2, y1, i);
        work(x1, y2, x2, y2, i);
        work(x2, y1, x2, y2, i);
        st[i - 'A'] = true; // 出现过 标记一下
    }
    
    topsort();
    
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

> [!NOTE] **[AcWing 456. 车站分级](https://www.acwing.com/problem/content/458/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 某趟车如果停了某个车站，必然要在区间内部所有大于该车站值的位置停靠
> 
> 区间内：所有大于等于某值的全停 小于的全不停 【某值即停靠站的最小值】
>
> 如 1 3 5 6是停靠站 且合法 则其最小等级必然大于 2 4 的等级
> 
> ==> 可以由未停靠的位置向所有停靠位置连边 表示小于停靠位置
> 
>     【但这样建边 数量太大 且复杂度过高 故考虑虚拟点】
> 
> ------------------------ 虚拟点 ------------------------
> 
> 如果左右两个集合：
> 
> 每个【左侧集合的每个点】都需要建立连接【右侧每一个点】的边
> 
> 也即 N^2 图
> 
> ==> 则在集合中间创建一个虚拟节点即可
> 
>     左侧全连虚拟节点【对于本题 权重0】 虚拟节点连右侧【对于本题 权重1】
> 
>     【N^2 图转变为 n+m】
> 
> -------------------------- end -------------------------
> 
> **核心：1.建图思路 2.虚拟节点优化边数**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 2010, M = 1000010;

int n, m;
int h[N], e[M], ne[M], w[M], idx;
int q[N], d[N];
int dist[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    d[b] ++ ;
}

void topsort() {
    int hh = 0, tt = -1;
    for (int i = 1; i <= n + m; ++ i )
        if (!d[i]) q[ ++ tt] = i;
    while (hh <= tt) {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (-- d[j] == 0) q[ ++ tt] = j;
        }
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> n >> m;
    for (int i = 1; i <= m; ++ i ) {
        memset(st, 0, sizeof st);
        int cnt;
        cin >> cnt;
        int start = n, end = 1;
        while (cnt -- ) {
            int stop;
            cin >> stop;
            start = min(start, stop);
            end = max(end, stop);
            st[stop] = true;
        }
        
        // ver 虚拟源点
        int ver = n + i;
        for (int j = start; j <= end; ++ j )
            if (!st[j]) add(j, ver, 0);
            else add(ver, j, 1);
    }
    topsort();
    
    // 随后求最长路【题目所求结果】
    
    // 初始化一个点 等价于dist最小为1
    for (int i = 1; i <= n; ++ i ) dist[i] = 1;
    // 注意 遍历顺序与所求结果有关 内部更新顺序与边定义有关
    //      求到起点的最长距离 故从前向后遍历
    for (int i = 0; i < n + m; ++ i ) {
        int j = q[i];
        for (int k = h[j]; ~k; k = ne[k])
            dist[e[k]] = max(dist[e[k]], dist[j] + w[k]);
    }
    int res = 0;
    for (int i = 1; i <= n; ++ i ) res = max(res, dist[i]);
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

> [!NOTE] **[Luogu 【XR-3】核心城市](https://www.luogu.com.cn/problem/P5536)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **类似拓扑排序的思路即可**
> 
> 题解区有很多用树直径 + 推理贪心的思路 也可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 拓扑的方案显然就可做
// 题解区很多用树直径的做法 TODO

const int N = 1e5 + 10, M = N << 1;

int n, k;
int h[N], e[M], ne[M], idx;
int c[N], d[N], q[N];

void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int main() {
    init();

    cin >> n >> k;
    
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
        c[a] ++ , c[b] ++ ;
    }
    
    int hh = 0, tt = -1;
    for (int i = 1; i <= n; ++ i )
        if (c[i] == 1)  // 叶子节点
            q[ ++ tt] = i, d[i] = 1;
    
    int cnt = n - k;
    while (hh <= tt) {
        int t = q[hh ++ ];
        cnt -- ;
        if (!cnt) {
            cout << d[t] << endl;
            return 0;
        }

        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if ( -- c[j] == 1)  // 1
                d[j] = d[t] + 1, q[ ++ tt] = j;
        }
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

> [!NOTE] **[LeetCode 1136. 平行课程](https://leetcode.cn/problems/parallel-courses/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 拓扑排序板子题 略
> 
> **这类转移较简单的可以直接在 topo 中计算**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 5010;
    
    int h[N], e[N], ne[N], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n;
    int d[N], q[N];
    int f[N];
    
    int topo() {
        int hh = 0, tt = -1;
        for (int i = 1; i <= n; ++ i )
            if (!d[i])
                q[ ++ tt] = i;
        while (hh <= tt) {
            int t = q[hh ++ ];
            for (int i = h[t]; ~i; i = ne[i]) {
                int j = e[i];
                if ( -- d[j] == 0)
                    q[ ++ tt] = j;
            }
        }
        if (hh != n)
            return -1;
        
        int res = 0;
        for (int i = n - 1; i >= 0; -- i ) {
            int t = q[i];
            f[t] = 1;   // min
            for (int j = h[t]; ~j; j = ne[j]) {
                int k = e[j];
                f[t] = max(f[t], f[k] + 1);
            }
            res = max(res, f[t]);
        }
        return res;
    }
    
    int minimumSemesters(int n, vector<vector<int>>& relations) {
        init();
        memset(f, 0, sizeof f);
        this->n = n;
        for (auto & re : relations)
            add(re[0], re[1]), d[re[1]] ++ ;
        return topo();
    }
};
```

##### **C++ topo中计算**

```cpp
class Solution {
public:
    int minimumSemesters(int N, vector<vector<int>>& relations) {
        vector<int> v[N + 1];
        vector<int> d(N + 1), f(N + 1, N + 1);
        for (auto e : relations) {
            v[e[0]].push_back(e[1]);
            d[e[1]] ++;
        }
        queue<int> Q;
        for (int i = 1; i <= N; ++ i)
            if (d[i] == 0) {
                Q.push(i);
                f[i] = 1;
            }
        while (!Q.empty()) {
            int x = Q.front();
            Q.pop();
            for (auto y : v[x])
                if (! -- d[y]) {
                    Q.push(y);
                    f[y] = f[x]+1;
                }
        }
        int ans = 0;
        for (int i = 1; i <= N; ++ i)
            ans = max(ans, f[i]);
        if (ans == N + 1) return -1;
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

> [!NOTE] **[LeetCode 1857. 有向图中最大颜色值](https://leetcode.cn/problems/largest-color-value-in-a-directed-graph/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始写了判环 然后跑拓扑 其实直接拓扑就可以
> 
> 随后正序递推即可 （正序dfs会TLE, 当然拓扑序性质注定无需搜索, 可以直接递推）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 1e5 + 10, M = 2e5 + 10;

int h[N], e[M], ne[M], idx;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int q[N], d[N];
int f[N][26];

void init() {
    memset(h, -1, sizeof h);
    idx = 0;
    memset(d, 0, sizeof d);
    memset(f, 0, sizeof f);
}

class Solution {
public:
    int n;
    bool topsort() {
        int hh = 0, tt = -1;
        for (int i = 0; i < n; i ++ )
            if (!d[i])
                q[ ++ tt] = i;

        while (hh <= tt) {
            int t = q[hh ++ ];
            for (int i = h[t]; ~i; i = ne[i]) {
                int j = e[i];
                if (-- d[j] == 0)
                    q[ ++ tt] = j;
            }
        }
        return tt == n - 1;
    }
    
    int largestPathValue(string colors, vector<vector<int>>& edges) {
        n = colors.size();
        init();
        for (auto & es : edges) {
            int a = es[0], b = es[1];
            add(a, b);
            d[b] ++ ;
        }
        if (!topsort())
            return -1;
        
        int res = 0;
        // 正序就是拓扑序列
        for (int i = 0; i < n; ++ i ) {
            int j = q[i], ci = colors[j] - 'a';  // 这里写成了colors[i] - 'a'比赛的时候WA
            f[j][ci] = max(f[j][ci], 1);     // 细节
            for (int k = h[j]; ~k; k = ne[k]) {
                int v = e[k], cv = colors[v] - 'a';
                for (int c = 0; c < 26; ++ c )
                    f[v][c] = max(f[v][c], f[j][c] + (c == cv));
            }
        }
        
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < 26; ++ j )
                res = max(res, f[i][j]);
        
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

> [!NOTE] **[LeetCode 2050. 并行课程 III](https://leetcode.cn/problems/parallel-courses-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典求拓扑序再根据拓扑序计算值 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 5e4 + 10, M = 5e4 + 10;
    
    int n;
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int din[N];
    vector<int> topology() {
        vector<int> q(n);
        int hh = 0, tt = -1;
        for (int i = 1; i <= n; ++ i )
            if (!din[i])
                q[ ++ tt] = i;
        
        while (hh <= tt) {
            int t = q[hh ++ ];
            for (int i = h[t]; ~i; i = ne[i]) {
                int j = e[i];
                if ( -- din[j] == 0)
                    q[ ++ tt] = j;
            }
        }
        
        return q;
    }
    
    int minimumTime(int n, vector<vector<int>>& relations, vector<int>& time) {
        this->n = n;
        init();
        
        memset(din, 0, sizeof din);
        for (auto & r : relations)
            add(r[0], r[1]), din[r[1]] ++ ;
        
        auto q = topology();
        vector<int> f(n);
        int res = 0;
        for (int i = n - 1; i >= 0; -- i ) {
            int u = q[i], t = 0;
            for (int j = h[u]; ~j; j = ne[j])
                t = max(t, f[e[j] - 1]);
            f[u - 1] = t + time[u - 1];
            res = max(res, f[u - 1]);
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

> [!NOTE] **[LeetCode 207. 课程表](https://leetcode.cn/problems/course-schedule/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 旧**

```cpp
class Solution {
public:
    bool dfs(int u, vector<vector<int>>& es, vector<int>& c) {
        c[u] = -1;
        for (auto v : es[u]) {
            if (c[v] < 0) return false;
            else if(!c[v] && !dfs(v, es, c)) return false;
        }
        c[u] = 1;
        return true;
    }
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> es(numCourses);
        for (auto e : prerequisites) {
            es[e[1]].push_back(e[0]);
        }
        vector<int> c(numCourses);
        for (int u = 0; u < numCourses; ++ u )
            if (!c[u])
                if(!dfs(u, es, c))
                    return false;
        
        return true;
    }
};
```

##### **Python**

```python
# topsort 排序
# 在图论中，拓扑排序（Topological Sorting）是一个有向无环图（DAG, Directed Acyclic Graph）的所有顶点的线性序列。且该序列必须满足下面两个条件：
# 每个顶点出现且只出现一次。
# 若存在一条从顶点 A 到顶点 B 的路径，那么在序列中顶点 A 出现在顶点 B 的前面。



class Solution:
    def canFinish(self, n: int, pre: List[List[int]]) -> bool:
        N=10**5+10
        h=[-1]*N
        ev=[0]*N
        ne=[0]*N
        idx=0
        d=[0]*N 
        res=[]

        def add(a,b):
            nonlocal idx
            ev[idx]=b 
            ne[idx]=h[a]
            h[a]=idx
            idx+=1

        def topsort():
            from collections import deque
            q=deque()
            for i in range(n):
                if d[i]==0:
                    q.append(i)
            while q:
                t=q.popleft()
                res.append(t)
                i=h[t]
                while i!=-1:
                    j=ev[i]
                    d[j]-=1
                    if d[j]==0:
                        q.append(j)
                    i=ne[i]
            return len(res)==n

        for i in range(len(pre)):
            a,b=pre[i][0],pre[i][1]
            add(a,b)
            d[b]+=1
        
        if topsort():
            return True
        else:
            return False
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/)**
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
    // 1. 迭代
    vector<int> findOrder(int n, vector<vector<int>>& edges) {
        vector<vector<int>> g(n);
        vector<int> d(n);
        for (auto& e: edges) {
            int b = e[0], a = e[1];
            g[a].push_back(b);
            d[b] ++ ;
        }
        queue<int> q;
        for (int i = 0; i < n; i ++ )
            if (d[i] == 0)
                q.push(i);

        vector<int> res;
        while (q.size()) {
            auto t = q.front();
            q.pop();
            res.push_back(t);
            for (int i: g[t])
                if ( -- d[i] == 0)
                    q.push(i);
        }
        if (res.size() < n) res = {};
        return res;
    }

    // 2. 递归
    int t;
    bool dfs(int u, vector<vector<int>>& G, vector<int>& c, vector<int>& topo) {
        c[u] = -1;
        for (auto v : G[u]) {
            if (c[v] < 0) return false;
            else if (!c[v] && !dfs(v, G, c, topo)) return false;
        }
        c[u] = 1;
        topo[ -- t] = u;
        return true;
    }
    vector<int> findOrder_2(int n, vector<vector<int>>& prerequisites) {
        t = n;
        vector<vector<int>> G(n);
        for (auto v : prerequisites) G[v[1]].push_back(v[0]);
        vector<int> c(n);
        vector<int> topo(n);

        for (int u = 0; u < n; ++ u )
            if (!c[u])
                if(!dfs(u, G, c, topo)) return vector<int>{};
        
        return topo;
    }
};
```

##### **Python**

```python
# topsort 排序
# 在图论中，拓扑排序（Topological Sorting）是一个有向无环图（DAG, Directed Acyclic Graph）的所有顶点的线性序列。且该序列必须满足下面两个条件：
# 每个顶点出现且只出现一次。
# 若存在一条从顶点 A 到顶点 B 的路径，那么在序列中顶点 A 出现在顶点 B 的前面。



class Solution:
    def findOrder(self, n: int, pre: List[List[int]]) -> List[int]:
        N=10**5+10
        h=[-1]*N
        ev=[0]*N
        ne=[0]*N
        idx=0
        d=[0]*N 
        res=[]

        def add(a,b):
            nonlocal idx
            ev[idx]=b 
            ne[idx]=h[a]
            h[a]=idx
            idx+=1

        def topsort():
            from collections import deque
            q=deque()
            for i in range(n):
                if d[i]==0:
                    q.append(i)
            while q:
                t=q.popleft()
                res.append(t)
                i=h[t]
                while i!=-1:
                    j=ev[i]
                    d[j]-=1
                    if d[j]==0:
                        q.append(j)
                    i=ne[i]
            return len(res)==n

        for i in range(len(pre)):
            a,b=pre[i][1],pre[i][0]
            add(a,b)
            d[b]+=1
        
        if topsort():
            return res
        else:
            return []
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 269. 火星词典](https://leetcode.cn/problems/alien-dictionary)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 拓扑排序即可 注意特殊 case

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 26, M = 700;

    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }

    int d[N], q[M];
    string topo() {
        int hh = 0, tt = -1;
        for (int i = 0; i < 26; ++ i )
            cout << " i = " << i << " d = " << d[i] << endl;
        for (int i = 0; i < 26; ++ i )
            if (!d[i])
                q[ ++ tt] = i;
        for (int i = 0; i <= tt; ++ i )
            cout << char('a' + q[i]) << ' ';
        cout << endl;
        while (hh <= tt) {
            int t = q[hh ++ ];
            for (int i = h[t]; ~i; i = ne[i]) {
                int j = e[i];
                if ( -- d[j] == 0)
                    q[ ++ tt] = j;
            }
        }
        if (tt != 26 - 1)
            return "";

        string res;
        for (int i = 0; i <= tt; ++ i )
            if (st[q[i]])
                res.push_back('a' + q[i]);
        return res;
    }

    bool g[N][N], st[N];

    string alienOrder(vector<string>& words) {
        int n = words.size();
        memset(g, 0, sizeof g);
        for (int i = 0; i < n; ++ i )
            for (int j = i + 1; j < n; ++ j ) {
                auto a = words[i], b = words[j];
                bool flag = false;  // ATTENTION
                for (int k = 0; k < a.size() && k < b.size(); ++ k )
                    if (a[k] != b[k]) {
                        flag = true;
                        g[a[k] - 'a'][b[k] - 'a'] = true;
                        break;
                    }
                // ["abc","ab"]
                if (!flag) {
                    if (a.size() > b.size())
                        return "";
                }
            }
        for (auto & x : words)
            for (auto & y : x)
                st[y - 'a'] = true;
        
        init();
        memset(d, 0, sizeof d);
        for (int i = 0; i < N; ++ i )
            for (int j = 0; j < N; ++ j )
                if (g[i][j])
                    add(i, j), d[j] ++ ;

        return topo();
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

### 拓扑排序方案数（可重集排序问题）

> [!NOTE] **[LeetCode 1916. 统计为蚁群构筑房间的不同顺序](https://leetcode.cn/problems/count-ways-to-build-rooms-in-an-ant-colony/)**
> 
> [weekly-247](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-06-27_Weekly-247)
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
    using LL = long long;
    const static int N = 1e5 + 10, MOD = 1e9 + 7;
    int h[N], e[N], ne[N], idx;
    int f[N], g[N];
    int s[N], sz[N];
    int n;
    
    int qmi(int a, int k) {
        int ret = 1;
        while (k) {
            if (k & 1)
                ret = (LL)ret * a % MOD;
            a = (LL)a * a % MOD;
            k >>= 1;
        }
        return ret;
    }
    
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
        f[0] = g[0] = 1;
        for (int i = 1; i <= n; ++ i ) {
            f[i] = f[i - 1] * (LL)i % MOD;
            g[i] = g[i - 1] * (LL)qmi(i, MOD - 2) % MOD;
        }
    }
    
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    // 单向边 不需要记录fa
    int dfs(int u) {
        sz[u] = 0;  // 初始时不包括跟节点
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            dfs(j);
            sz[u] += sz[j];
        }
        // 所有子树的和的阶乘
        s[u] = f[sz[u]];
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            // 子树数量逆元 子树方案数
            s[u] = (LL)s[u] * g[sz[j]] % MOD;
            s[u] = (LL)s[u] * s[j] % MOD;
        }
        sz[u] ++ ;
        return s[u];
    }
    
    int waysToBuildRooms(vector<int>& prevRoom) {
        this->n = prevRoom.size();
        init();
        for (int i = 1; i < n; ++ i )
            add(prevRoom[i], i);
        return dfs(0);
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

### 进阶应用

> [!NOTE] **[LeetCode 1203. 项目管理](https://leetcode.cn/problems/sort-items-by-groups-respecting-dependencies/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 双层拓扑排序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    //const static int N = 30010;
    const static int N = 34010; // 33010 wa 35010 ac
    int e[N], ne[N], h[N], idx;
    int din[N], q[N];
    void init() {
        idx = 0;
        memset(h, -1, sizeof h);
        memset(din, 0, sizeof din);
    }
    void addedge(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    bool topo(int n) {
        int hh = 0, tt = -1;
        for (int i = 0; i < n; ++ i )
            if (!din[i])
                q[ ++ tt] = i;
        while (hh <= tt) {
            int t = q[hh ++ ];
            for (int i = h[t]; ~i; i = ne[i]) {
                int j = e[i];
                if (! --din[j])
                    q[ ++ tt] = j;
            }
        }
        // 注意判断
        return tt == n - 1;
    }
    
    vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
        // 1. 项目之间 拓扑排序
        init();
        for (int i = 0; i < n; ++ i )
            for (auto x : beforeItems[i])
                addedge(x, i), din[i] ++ ;
        if (!topo(n)) return {};
        
        // 2. 为不属于某个小组的项目新建虚拟小组
        int tmp_m = m;
        for (int i = 0; i < n; ++ i )
            if (group[i] == -1)
                group[i] = m ++ ;
        // 3. 所有本小组的项目加入小组内 保存在v数组
        vector<vector<int>> v(m);
        for (int i = 0; i < n; ++ i )
            v[group[q[i]]].push_back(q[i]);
        
        // 4. 小组之间建边
        init();
        for (int i = 0; i < n; ++ i )
            for (auto x : beforeItems[i]) {
                if (group[x] == group[i]) continue;
                addedge(group[x], group[i]), din[group[i]] ++ ;
            }
        if (!topo(m)) return {};
        
        // 5. 结果
        vector<int> res;
        for (int i = 0; i < m; ++ i )
            for (auto x : v[q[i]])
                res.push_back(x);
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

> [!NOTE] **[LeetCode 1591. 奇怪的打印机 II](https://leetcode.cn/problems/strange-printer-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 每个颜色染一个矩形 颜色不能重复使用 问能否达成目标染色情况
> 
> 显然先统计颜色并求每个颜色的左上与右下边界
> 
> 随后：枚举每个矩阵内的所有其他颜色，建图，跑拓扑排序
> 
> 判断是否有环，原理是：颜色只能用一次 则前述有向图不能有环

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 考虑统计每个颜色出现的左上、右下坐标
    int m, n;
    vector<vector<int>> G;
    vector<unordered_set<int>> es;
    vector<int> vis;
    bool dfs(int u) {
        vis[u] = -1;
        for (auto& v : es[u]) {
            if (vis[v] < 0)
                return false;
            else if (!vis[v] && !dfs(v))
                return false;
        }
        vis[u] = 1;
        return true;
    }
    bool isPrintable(vector<vector<int>>& targetGrid) {
        this->G = targetGrid;
        m = G.size(), n = G[0].size();
        vector<int> l(61, 60), u(61, 60), r(61, -1), d(61, -1);
        es = vector<unordered_set<int>>(61);
        vis = vector<int>(61);
        unordered_set<int> color;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                int c = G[i][j];
                color.insert(c);
                l[c] = min(l[c], j);
                u[c] = min(u[c], i);
                r[c] = max(r[c], j);
                d[c] = max(d[c], i);
            }
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                int c = G[i][j];
                for (auto& uc : color)
                    if (c != uc && l[uc] <= j && u[uc] <= i && r[uc] >= j &&
                        d[uc] >= i)
                        es[uc].insert(c);
            }
        for (auto& c : color)
            if (!vis[c])
                if (!dfs(c)) return false;
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

> [!NOTE] **[LeetCode 1632. 矩阵转换后的秩](https://leetcode.cn/problems/rank-transform-of-a-matrix/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思路：并查集+拓扑排序
> 
> 比赛想到了拓扑排序【二维点转化一维，每行、每列排序建边】，节点数太多感觉要爆
> 
> **需要想得到结合并查集减少节点数**
> 
> liuzhou 的优雅模版

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 记录 liuzhou_101 聚聚的写法
template<class T>
inline bool freshmin(T& a, const T& b) {
    if (a > b) {
        a = b;
        return true;
    } else return false;
}

template<class T>
inline bool freshmax(T& a, const T& b) {
    if (a < b) {
        a = b;
        return true;
    } else return false;
}

struct directed_graph {
    int n;
    vector<vector<int>> v;
    
    directed_graph (int n = 0) {
        init(n);
    }
    void init(int n) {
        assert(n >= 0);
        this->n = n;
        v = vector<vector<int>>(n+1);
    }
    void addedge(int x, int y) {
        assert(1 <= x && x <= n);
        assert(1 <= y && y <= n);
        v[x].push_back(y);
    }
    void erase_multiedges() {
        for (int i = 1; i <= n; ++ i) {
            sort(v[i].begin(), v[i].end());
            v[i].erase(unique(v[i].begin(), v[i].end()), v[i].end());
        }
    }
    
    vector<int> in_degree() {
        vector<int> ret(n+1);
        for (int x = 1; x <= n; ++ x)
            for (auto y : v[x])
                ret[y] ++;
        return ret;
    }
    vector<int> topsort(int bound) {
        vector<int> deg = in_degree();
        vector<int> dp = vector<int>(n + 1);
        
        vector<int> ret;
        queue<int> q;
        for (int i = 1; i <= n; ++ i)
            if (!deg[i]) q.push(i);
        while (!q.empty()) {
            int x = q.front();
            q.pop();
            ret.push_back(x);
            for (auto y : v[x]) {
                dp[y] = max(dp[y], dp[x] + (x <= bound));
                if (!-- deg[y]) q.push(y);
            }
        }
        return dp;
    }
    
    int times;
    vector<int> dfn, low, in;
    vector<int> st;
    int scc;
    vector<int> belong;
    void tarjan(int x) {
        dfn[x] = low[x] = ++ times;
        st.push_back(x);
        in[x] = 1;
        for (auto y : v[x]) {
            if (!dfn[y]) {
                tarjan(y);
                freshmin(low[x], low[y]);
            }
            else if (in[y])
                freshmin(low[x], dfn[y]);
        }
        if (dfn[x] == low[x]) {
            scc ++;
            while (1) {
                int y = st.back();
                st.pop_back();
                in[y] = 0;
                belong[y] = scc;
                if (x == y) break;
            }
        }
    }
    directed_graph strong_connected_component() {
        times = 0;
        dfn = vector<int>(n+1);
        low = vector<int>(n+1);
        in = vector<int>(n+1);
        st = vector<int>();
        scc = 0;
        belong = vector<int>(n+1);
        for (int i = 1; i <= n; ++ i)
            if (!dfn[i]) tarjan(i);
        directed_graph ret(scc);
        for (int x = 1; x <= n; ++ x)
            for (auto y : v[x]) if (belong[x] != belong[y])
                ret.addedge(belong[x], belong[y]);
        // ret.erase_multiedges();
        return ret;
    }
};

template<class T>
auto arr(int n = 0) {
    return vector<T>(n);
}

template<class T, class... Args>
auto arr(int n, Args... args) {
    return vector<decltype(arr<T>(args...))>(n, arr<T>(args...));
}

class Solution {
public:
    vector<vector<int>> matrixRankTransform(vector<vector<int>>& a) {
        int n = a.size(), m = a[0].size();
        
        int N = n*m;
        auto place = [&](int x, int y) {
            return x * m + y + 1;
        };
        
        vector<int> f(N + 1);
        for (int i = 0; i <= N; ++i) f[i] = i;
        function<int(int)> father = [&](int x) {
            return f[x] == x ? x : f[x] = father(f[x]);
        };
        
        for (int i = 0; i < n; ++i) {
            vector<int> p(m);
            for (int j = 0; j < m; ++j) p[j] = j;
            sort(p.begin(), p.end(), [&](int x, int y) {
                    return a[i][x] < a[i][y];
                });
            for (int k = 1; k < m; ++k)
                if (a[i][p[k - 1]] == a[i][p[k]])
                    f[father(place(i, p[k - 1]))] = father(place(i, p[k]));
        }
        for (int j = 0; j < m; ++j) {
            vector<int> p(n);
            for (int i = 0; i < n; ++i) p[i] = i;
            sort(p.begin(), p.end(), [&](int x, int y) {
                    return a[x][j] < a[y][j];
                });
            for (int k = 1; k < n; ++k)
                if (a[p[k - 1]][j] == a[p[k]][j])
                    f[father(place(p[k - 1], j))] = father(place(p[k], j));
        }
        
        directed_graph G(N + 1);
        
        for (int i = 0; i < n; ++i) {
            vector<int> p(m);
            for (int j = 0; j < m; ++j) p[j] = j;
            sort(p.begin(), p.end(), [&](int x, int y) {
                    return a[i][x] < a[i][y];
                });
            for (int k = 1; k < m; ++k)
                if (a[i][p[k - 1]] < a[i][p[k]])
                    G.addedge(father(place(i, p[k - 1])), father(place(i, p[k])));
        }
        
        for (int j = 0; j < m; ++j) {
            vector<int> p(n);
            for (int i = 0; i < n; ++i) p[i] = i;
            sort(p.begin(), p.end(), [&](int x, int y) {
                    return a[x][j] < a[y][j];
                });
            for (int k = 0; k < n; ++k)
                if (k - 1 >= 0 && a[p[k - 1]][j] < a[p[k]][j])
                    G.addedge(father(place(p[k - 1], j)), father(place(p[k], j)));
        }
        
        auto dp = G.topsort(N);
        auto res = arr<int>(n, m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                res[i][j] = dp[father(place(i, j))] + 1;
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

> [!NOTE] **[LeetCode 2115. 从给定原材料中找到所有可以做出的菜](https://leetcode.cn/problems/find-all-possible-recipes-from-given-supplies/)** [TAG]
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
     // 标程 topo
     using US = unordered_set<string>;
     using USM = unordered_map<string, US>;
     
     USM g;
     unordered_map<string, int> deg;
     
     vector<string> findAllRecipes(vector<string>& recipes, vector<vector<string>>& ingredients, vector<string>& supplies) {
         int n = recipes.size();
         for (int i = 0; i < n; ++ i ) {
             auto & pa = recipes[i];
             auto & sons = ingredients[i];
             for (auto & son : sons)
                 g[son].insert(pa), deg[pa] ++ ;
         }
         
         queue<string> q;
         for (auto & s : supplies)
             q.push(s);
         
         vector<string> res;
         while (!q.empty()) {
             auto t = q.front(); q.pop();
             for (auto & pa : g[t])
                 if ( -- deg[pa] == 0)
                     q.push(pa), res.push_back(pa);
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

> [!NOTE] **[Codeforces C. Fox And Names](http://codeforces.com/problemset/problem/510/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 拓扑序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Fox And Names
// Contest: Codeforces - Codeforces Round #290 (Div. 2)
// URL: http://codeforces.com/problemset/problem/510/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const int N = 26, M = 110;

int n;
string ss[M];
int sz[M];
bool g[N][N], tg[N][N];
int d[N], q[N];
int hh, tt;

void topo() {
    hh = 0, tt = -1;
    for (int i = 0; i < N; ++i)
        if (!d[i])
            q[++tt] = i;

    while (hh <= tt) {
        int t = q[hh++];

        for (int i = 0; i < N; ++i)
            if (g[t][i] && --d[i] == 0)
                q[++tt] = i;
    }
}

int main() {
    cin >> n;

    bool impossible = false;
    for (int i = 0; i < n; ++i) {
        cin >> ss[i];
        sz[i] = ss[i].size();
        // 注意脑袋理清楚 这里只需要和上一个字符串作比较即可
        // https://codeforces.com/contest/510/submission/110623880
        if (i) {
            int j = i - 1;
            bool f = false;
            for (int k = 0; k < sz[i] && k < sz[j]; ++k)
                if (ss[i][k] != ss[j][k]) {
                    int a = ss[j][k] - 'a', b = ss[i][k] - 'a';
                    // 防止重复建边 导致拓扑出错
                    // http://codeforces.com/contest/510/submission/110623880
                    // https://codeforces.com/contest/510/submission/110694636
                    if (!g[a][b]) {
                        g[a][b] = true;
                        d[b]++;
                    }

                    f = true;
                    break;
                }
            if (!f && sz[i] < sz[j])
                impossible = true;
        }
    }

    memcpy(tg, g, sizeof g);
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                tg[i][j] |= tg[i][k] && tg[k][j];
    for (int i = 0; i < N && !impossible; ++i)
        if (tg[i][i])
            impossible = true;

    if (!impossible) {
        topo();
        for (int i = 0; i < N; ++i)
            cout << char('a' + q[i]);
        cout << endl;
    } else
        cout << "Impossible" << endl;

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

> [!NOTE] **[Codeforces C. Misha and Forest](https://codeforces.com/problemset/problem/501/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 论算法思想应用的千变万化
> 
> 考虑一棵树必然存在度为 $1$ 的点，且该点的 $s$ 值即为与其相连的点的编号
> 
> 故拓扑排序做即可
> 
> 非常好的图论思想应用题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Misha and Forest
// Contest: Codeforces - Codeforces Round #285 (Div. 2)
// URL: https://codeforces.com/problemset/problem/501/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 初看没思路 需加强图论训练和图论敏感度
// 考虑一棵树必然存在度为 1 的点，且该点的 s 值即为与其相连的点的编号
// 故拓扑排序做即可 很好的图论题

const int N = 66000;  // > 65535

int n;
int d[N], s[N];
int q[N];

void topo() {
    int hh = 0, tt = -1;
    for (int i = 0; i < n; ++i)
        if (d[i] == 1)
            q[++tt] = i;

    while (hh <= tt) {
        int t = q[hh++];
        if (d[t] != 1)
            continue;

        // 当前 t 相连的只有一个点 v
        int v = s[t];
        // 这里是 s[v] d[v] 而非 t WA
        // https://codeforces.com/contest/501/submission/110866847
        s[v] ^= t, d[v]--;
        if (d[v] == 1)
            q[++tt] = v;
        cout << t << ' ' << v << endl;
    }
}

int main() {
    cin >> n;

    int m = 0;
    for (int i = 0; i < n; ++i) {
        cin >> d[i] >> s[i];
        m += d[i];
    }
    cout << m / 2 << endl;

    topo();

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