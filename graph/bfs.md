
BFS 全称是 [Breadth First Search](https://en.wikipedia.org/wiki/Breadth-first_search)，中文名是宽度优先搜索，也叫广度优先搜索。


## 双端队列 BFS

如果你不了解双端队列 `deque` 的话，请参阅 [deque 相关章节](lang/csl/sequence-container/#deque)。

双端队列 BFS 又称 0-1 BFS。

### 适用范围

边权值为可能有，也可能没有（由于 BFS 适用于权值为 1 的图，所以一般权值是 0 或 1），或者能够转化为这种边权值的最短路问题。

例如在走迷宫问题中，你可以花 1 个金币走 5 步，也可以不花金币走 1 步，这就可以用 0-1 BFS 解决。

### 实现

一般情况下，我们把没有权值的边扩展到的点放到队首，有权值的边扩展到的点放到队尾。这样即可保证像普通 BFS 一样整个队列队首到队尾权值单调不下降。

下面是伪代码：

```cpp
while (队列不为空) {
    int u = 队首;
    弹出队首;
    for (枚举 u 的邻居) {
        更新数据
        if (...)
            添加到队首;
        else
            添加到队尾;
    }
}
```

### 例题

### [Codeforces 173B](http://codeforces.com/problemset/problem/173/B)

一个 $n \times m$ 的图，现在有一束激光从左上角往右边射出，每遇到 '#'，你可以选择光线往四个方向射出，或者什么都不做，问最少需要多少个 '#' 往四个方向射出才能使光线在第 $n$ 行往右边射出。

此题目正解不是 0-1 BFS，但是适用 0-1 BFS，减小思维强度，赛时许多大佬都是这么做的。

做法很简单，一个方向射出不需要花费（0），而往四个方向射出需要花费（1），然后直接来就可以了。

#### 代码


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

## 优先队列 BFS

优先队列，相当于一个二叉堆，STL 中提供了 [`std::priority_queue`](lang/csl/container-adapter.md)，可以方便我们使用优先队列。

在基于优先队列的 BFS 中，我们每次从队首取出代价最小的结点进行进一步搜索。容易证明这个贪心思想是正确的，因为从这个结点开始扩展的搜索，一定不会更新原来那些代价更高的结点。换句话说，其余那些代价更高的结点，我们不回去考虑更新它。

当然，每个结点可能会被入队多次，只是每次入队的代价不同。当该结点第一次从优先队列中取出，以后便无需再在该结点进行搜索，直接忽略即可。所以，优先队列的 BFS 当中，每个结点只会被处理一次。

相对于普通队列的 BFS，时间复杂度多了一个 $\log n$，毕竟要维护这个优先队列嘛。不过普通 BFS 有可能每个结点入队、出队多次，时间复杂度会达到 $O(n^2)$，不是 $O(n)$。所以优先队列 BFS 通常还是快的。

诶？这怎么听起来这么像堆优化的 [Dijkstra](./shortest-path.md#dijkstra) 算法呢？事实上，堆优化 Dijkstra 就是优先队列 BFS。

## 习题

- [「NOIP2017」奶酪](https://uoj.ac/problem/332)

## 参考

<https://cp-algorithms.com/graph/breadth-first-search.html>


## 习题

> [!NOTE] **[AcWing 844. 走迷宫](https://www.acwing.com/problem/content/846/)**
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
#include <queue>

using namespace std;

typedef pair<int, int> PII;

const int N = 110;

int n, m;
int g[N][N], d[N][N];

int bfs() {
    queue<PII> q;

    memset(d, -1, sizeof d);
    d[0][0] = 0;
    q.push({0, 0});

    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

    while (q.size()) {
        auto t = q.front();
        q.pop();

        for (int i = 0; i < 4; i++) {
            int x = t.first + dx[i], y = t.second + dy[i];

            if (x >= 0 && x < n && y >= 0 && y < m && g[x][y] == 0 &&
                d[x][y] == -1) {
                d[x][y] = d[t.first][t.second] + 1;
                q.push({x, y});
            }
        }
    }

    return d[n - 1][m - 1];
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) cin >> g[i][j];

    cout << bfs() << endl;

    return 0;
}
```

##### **Python**

```python
def bfs():
    from collections import deque
    q = deque()
    q.append([0, 0])
    d[0][0] = 0  # 该点 已经被走过了。
    idx = [-1, 1, 0, 0]
    idy = [0, 0, -1, 1]
    while q:
        cur = q.popleft()
        for i in range(4):
            x = cur[0] + idx[i]
            y = cur[1] + idy[i]
            if 0 <= x < n and 0 <= y < m and not arr[x][y] and d[x][y] == -1:
                q.append([x, y])
                d[x][y] = d[cur[0]][cur[1]] + 1
    return d[n - 1][m - 1]


if __name__ == '__main__':
    n, m = map(int, input().split())
    arr = [list(map(int, input().split())) for _ in range(n)]
    d = [[-1] * m for _ in range(n)]  # 所有的距离初始化为-1，表示这个点 没有走过。
    print(bfs())
```

<!-- tabs:end -->
</details>

<br>

* * *> [!NOTE] **[AcWing 845. 八数码](https://www.acwing.com/problem/content/847/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;
string s;
string tar = "12345678x";
int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};

int bfs() {
    queue<string> q;
    unordered_map<string, int> d;
    q.push(s);
    d[s] = 0;
    while (!q.empty()) {
        auto ss = q.front();
        q.pop();
        if (ss == tar) return d[ss];
        int dis = d[ss];
        int k = ss.find('x');
        int x = k / 3, y = k % 3;
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= 3 || ny < 0 || ny >= 3) continue;
            swap(ss[nx * 3 + ny], ss[k]);
            if (!d.count(ss)) {
                d[ss] = dis + 1;
                q.push(ss);
            }
            swap(ss[nx * 3 + ny], ss[k]);
        }
    }
    return -1;
}

int main() {
    char c;
    for (int i = 0; i < 9; ++i) {
        cin >> c;
        s.push_back(c);
    }
    cout << bfs() << endl;
    return 0;
}
```

##### **Python**

```python
# 这道题可以转化为：状态a转移到状态b,需要走一步，那最后走到终点，最少需要走多少步
# 难点：1. 状态表示复杂（导致有两个问题，如何 把状态存入队列中）；2. 如何记录每个状态的距离 dist
# 比较简单的存储方式是：用一个字符串表示一个状态；比如"1234x5678"
# 如何判断一个状态 可以 转移成 哪几种状态？ 需要思考一下。1）将字符串想象成3*3的矩阵 2）进行转移 3）3*3的矩阵恢复成字符串

def bfs(start):
    end = "12345678x"
    d = {start: 0}  # 该字典存放每个状态到初始状态的距离

    from collections import deque
    q = deque()
    q.append(start)
    dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
    while q:
        t = q.popleft()
        distance = d[t]
        if t == end:
            return distance
        k = t.index('x')  # 找到x的下标
        x, y = k // 3, k % 3  # 将字符串的索引 转换为3*3二维数组中对应的坐标

        tl = list(t)  # 由于字符串不是不可变量，因而把字符串变为数组，用于交换字符串中的字符位置
        for i in range(4):  # 'x'上下左右移动
            a, b = x + dx[i], y + dy[i]
            if 0 <= a < 3 and 0 <= b < 3:
                nk = a * 3 + b  # 将3*3矩阵的坐标转移为字符串的索引
                tl[nk], tl[k] = tl[k], tl[nk]
                t = ''.join(tl)
                if t not in d:  # 如果t这个状态是新状态（之前没有出现过）
                    q.append(t)
                    d[t] = distance + 1
                tl[nk], tl[k] = tl[k], tl[nk]  # 还原现场
    return -1


if __name__ == '__main__':
    start = input().replace(" ", "")
    ans = bfs(start)
    print(ans)
```

<!-- tabs:end -->
</details>

<br>

* * *

### 01 bfs

> [!NOTE] **[AcWing 175. 电路维修](https://www.acwing.com/problem/content/177/)**
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

const int inf = 0x3f3f3f3f;
const int N = 510, M = N * N;
int n, m;
char g[N][N];
int d[N][N];
bool vis[N][N];

int bfs() {
    memset(d, 0x3f, sizeof d);
    memset(vis, false, sizeof vis);

    char cs[] = "\\/\\/";  // 包含两个转义字符
    int dx[4] = {-1, -1, 1, 1}, dy[4] = {-1, 1, 1, -1};
    int ix[4] = {-1, -1, 0, 0},
        iy[4] = {-1, 0, 0, -1};  // 获取某个点对应方向位置保存的字符

    deque<pair<int, int>> dq;
    d[0][0] = 0;
    dq.push_back({0, 0});

    while (!dq.empty()) {
        auto t = dq.front();
        dq.pop_front();
        int x = t.first, y = t.second;
        if (vis[x][y]) continue;
        vis[x][y] = true;
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx > n || ny < 0 || ny > m)
                continue;  // 注意 这里比n m 大 1

            int cx = x + ix[i], cy = y + iy[i];
            int dis = d[x][y] + (g[cx][cy] != cs[i]);
            if (dis < d[nx][ny]) {
                d[nx][ny] = dis;

                if (g[cx][cy] != cs[i])
                    dq.push_back({nx, ny});
                else
                    dq.push_front({nx, ny});
            }
        }
    }
    return d[n][m];
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        cin >> n >> m;
        for (int i = 0; i < n; ++i) cin >> g[i];
        int t = bfs();
        if (t == inf)
            cout << "NO SOLUTION" << endl;
        else
            cout << t << endl;
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *