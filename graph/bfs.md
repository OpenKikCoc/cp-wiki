
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

### 一般 bfs

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

* * *

> [!NOTE] **[AcWing 845. 八数码](https://www.acwing.com/problem/content/847/)**
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

> [!NOTE] **[LeetCode 1210. 穿过迷宫的最少移动次数](https://leetcode-cn.com/problems/minimum-moves-to-reach-target-with-rotations/)**
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
    const int INF = 0x3f3f3f3f;
    using TIII = tuple<int, int, int>;
    using PTI = pair<TIII, int>;
    int n;
    vector<vector<int>> g;
    set<TIII> vis;
    
    int minimumMoves(vector<vector<int>>& grid) {
        n = grid.size();
        if (!n) return 0;
        g = grid;
        
        queue<PTI> q;
        // x, y, st, dis
        q.push({{0, 0, 0}, 0});
        while (!q.empty()) {
            auto [t, d] = q.front(); q.pop();
            if (vis.count(t)) continue;
            vis.insert(t);
            
            auto [x, y, st] = t;
            //cout << " x = " << x << "  y = " << y << " st = " << st << "  d = " << d << endl;
            if (x == n - 1 && y == n - 2) return d;
            if (st == 0 && y + 2 < n && g[x][y + 2] == 0) q.push({{x, y + 1, st}, d + 1});
            if (st == 1 && y + 1 < n && g[x][y + 1] == 0 && x + 1 < n && g[x + 1][y + 1] == 0) q.push({{x, y + 1, st}, d + 1});
            if (st == 0 && x + 1 < n && g[x + 1][y] == 0 && g[x + 1][y + 1] == 0) q.push({{x + 1, y, st}, d + 1});
            if (st == 1 && x + 2 < n && g[x + 2][y] == 0) q.push({{x + 1, y, st}, d + 1});
            if (st == 0 && x + 1 < n && g[x + 1][y] == 0 && g[x + 1][y + 1] == 0) q.push({{x, y, !st}, d + 1});
            if (st == 1 && y + 1 < n && g[x][y + 1] == 0 && g[x + 1][y + 1] == 0) q.push({{x, y, !st}, d + 1});
        }
        //cout << endl;
        return -1;
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

> [!NOTE] **[LeetCode 1926. 迷宫中离入口最近的出口](https://leetcode-cn.com/problems/nearest-exit-from-entrance-in-maze/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 多源 bfs 即可
> 
> 加点的细节写错WA了几发。。。
> 
> 也可以单源最短路 trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 110;
    const int INF = 0x3f3f3f3f;
    
    vector<vector<char>> g;
    vector<vector<int>> d;
    int n, m;
    PII q[N * N];
    
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    
    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        this->g = maze;
        this->n = g.size(), m = g[0].size();
        d = vector<vector<int>>(n, vector<int>(m, INF));
        
        int sx = entrance[0], sy = entrance[1];
        
        int hh = 0, tt = -1;
        for (int i = 0; i < n; ++ i ) {
            if (!(i == sx && 0 == sy) && g[i][0] == '.')
                q[ ++ tt] = {i, 0}, d[i][0] = 0;
            if (!(i == sx && m - 1 == sy) && g[i][m - 1] == '.')
                q[ ++ tt] = {i, m - 1}, d[i][m - 1] = 0;
        }
        for (int i = 0; i < m; ++ i ) {
            if (!(0 == sx && i == sy) && g[0][i] == '.')
                q[ ++ tt] = {0, i}, d[0][i] = 0;
            if (!(n - 1 == sx && i == sy) && g[n - 1][i] == '.')
                q[ ++ tt] = {n - 1, i}, d[n - 1][i] = 0;
        }
        
        while (hh <= tt) {
            auto [x, y] = q[hh ++ ];
            
            if (g[x][y] == '+')
                continue;
            g[x][y] = '+';
            if (x == sx && y == sy)
                break;
            
            for (int i = 0; i < 4; ++ i ) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx >= 0 && nx < n && ny >= 0 && ny < m && g[nx][ny] == '.') {
                    // ATTENTION if 一定要写 否则加太多点 数组越界
                    if (d[x][y] + 1 < d[nx][ny])
                        q[ ++ tt] = {nx, ny}, d[nx][ny] = d[x][y] + 1;
                }
            }
        }
        
        if (d[sx][sy] > INF / 2)
            return -1;
        return d[sx][sy];
    }
};
```

##### **C++ 单源最短路**

```cpp
#define x first
#define y second

typedef pair<int, int> PII;

class Solution {
public:
    int nearestExit(vector<vector<char>>& g, vector<int>& s) {
        int n = g.size(), m = g[0].size(), INF = 1e8;
        int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
        vector<vector<int>> dist(n, vector<int>(m, INF));
        queue<PII> q;
        q.push({s[0], s[1]});
        dist[s[0]][s[1]] = 0;
        while (q.size()) {
            auto t = q.front();
            q.pop();
            
            for (int i = 0; i < 4; i ++ ) {
                int x = t.x + dx[i], y = t.y + dy[i];
                if (x >= 0 && x < n && y >= 0 && y < m && g[x][y] == '.' && dist[x][y] > dist[t.x][t.y] + 1) {
                    dist[x][y] = dist[t.x][t.y] + 1;
                    if (x == 0 || x == n - 1 || y == 0 || y == m - 1) return dist[x][y];
                    q.push({x, y});
                }
            }
        }
        return -1;
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

### trick bfs

> [!NOTE] **[LeetCode 675. 为高尔夫比赛砍树](https://leetcode-cn.com/problems/cut-off-trees-for-golf-event/)**
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
    using PII = pair<int, int>;
    using TIII = tuple<int, int, int>;
    int n, m;
    vector<vector<int>> g;
    int dx[4] = {1, 0, 0, -1}, dy[4] = {0, -1, 1, 0};

    int bfs(int sx, int sy, int tx, int ty) {
        if (sx == tx && sy == ty)
            return 0;
        vector<vector<int>> dist(n, vector<int>(m, 1e9));
        dist[sx][sy] = 0;
        queue<PII> q;
        q.push({sx, sy});
        while (q.size()) {
            auto [x, y] = q.front(); q.pop();
            for (int i = 0; i < 4; ++ i ) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                if (!g[nx][ny]) continue;
                if (dist[nx][ny] > dist[x][y] + 1) {
                    dist[nx][ny] = dist[x][y] + 1;
                    if (nx == tx && ny == ty)
                        return dist[nx][ny];
                    q.push({nx, ny});
                }
            }
        }
        return -1;
    }

    int cutOffTree(vector<vector<int>>& forest) {
        g = forest;
        n = g.size(), m = g[0].size();
        vector<TIII> trs;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (g[i][j] > 1)
                    trs.push_back({g[i][j], i, j});
        sort(trs.begin(), trs.end());

        int x = 0, y = 0, res = 0;
        for (auto [tmp, nx, ny] : trs) {
            int t = bfs(x, y, nx, ny);
            if (t == -1)
                return -1;
            res += t;
            x = nx, y = ny;
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

> [!NOTE] **[LeetCode 1263. 推箱子](https://leetcode-cn.com/problems/minimum-moves-to-move-a-box-to-their-target-location/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本质是bfs **双端队列确保遍历的层次性**
> 
> 经典 理解记忆

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    int minPushBox(vector<vector<char>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int sx, sy, bx, by, tx, ty;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 'S') sx = i, sy = j;
                if (grid[i][j] == 'B') bx = i, by = j;
                if (grid[i][j] == 'T') tx = i, ty = j;
            }
        vector<vector<vector<vector<int>>>> f(
            m, vector<vector<vector<int>>>(
                   n, vector<vector<int>>(m, vector<int>(n, inf))));
        // 初始位置状态
        f[sx][sy][bx][by] = 0;
        deque<tuple<int, int, int, int>> dq;
        dq.push_back({sx, sy, bx, by});
        auto go_back = [&](int sx, int sy, int bx, int by, int d) {
            if (f[sx][sy][bx][by] > d) {
                f[sx][sy][bx][by] = d;
                dq.push_back({sx, sy, bx, by});
            }
        };
        auto go_front = [&](int sx, int sy, int bx, int by, int d) {
            if (f[sx][sy][bx][by] > d) {
                f[sx][sy][bx][by] = d;
                dq.push_front({sx, sy, bx, by});
            }
        };
        auto check = [&](int x, int y) {
            return x >= 0 && x < m && y >= 0 && y < n && grid[x][y] != '#';
        };
        while (!dq.empty()) {
            auto [sx, sy, bx, by] = dq.front();
            dq.pop_front();
            for (int i = 0; i < 4; ++i) {
                int x = sx + dx[i], y = sy + dy[i];
                if (check(x, y)) {
                    if (x == bx && y == by) {
                        int px = bx + dx[i], py = by + dy[i];
                        if (check(px, py))
                            go_back(x, y, px, py, f[sx][sy][bx][by] + 1);
                    } else
                        go_front(x, y, bx, by, f[sx][sy][bx][by]);
                }
            }
        }
        int res = inf;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) res = min(res, f[i][j][tx][ty]);
        return res == inf ? -1 : res;
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

> [!NOTE] **[LeetCode 1293. 网格中的最短路径](https://leetcode-cn.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 网格最短路 bfs
>
> >   **题目类型扩展：**
> >
> >   1.  若题目要求求解最小层级搜索（节点间距离固定为1），通过统计层级计数，遇到终止条件终止即可。
> >   2.  若节点间有加权值，求解最短路径时可以在Node中增加cost记录，比较获取最佳值
> >   3.  若需要求解最短路径，可以逆向根据visited访问记录情况回溯

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    struct node {
        int x, y, k;
        node(){};
        node(int _x, int _y, int _k) : x(_x), y(_y), k(_k){};
    };
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    int shortestPath(vector<vector<int>>& grid, int k) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<vector<int>>> dis(
            m + 1, vector<vector<int>>(n + 1, vector<int>(k + 1, -1)));
        dis[0][0][0] = 0;
        queue<node> q;
        q.push({0, 0, 0});
        while (!q.empty()) {
            auto [x, y, d] = q.front();
            q.pop();
            for (int i = 0; i < 4; ++i) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
                int nd = d + grid[nx][ny];
                if (nd > k) continue;
                if (dis[nx][ny][nd] != -1) continue;
                dis[nx][ny][nd] = dis[x][y][d] + 1;
                q.push({nx, ny, nd});
            }
        }
        int res = 0x3f3f3f3f;
        for (int i = 0; i <= k; ++i)
            if (dis[m - 1][n - 1][i] != -1)
                res = min(res, dis[m - 1][n - 1][i]);
        if (res == 0x3f3f3f3f) return -1;
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

> [!NOTE] **[LeetCode 1298. 你能从盒子里获得的最大糖果数](https://leetcode-cn.com/problems/maximum-candies-you-can-get-from-boxes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 每次循环校验即可 要敢于暴力
> 
> 经典

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxCandies(vector<int>& status, vector<int>& candies,
                   vector<vector<int>>& keys,
                   vector<vector<int>>& containedBoxes,
                   vector<int>& initialBoxes) {
        int n = status.size();
        vector<bool> open(n), haskey(n), hasbox(n), used(n);
        int res = 0;
        queue<int> q;
        for (int i = 0; i < n; ++i) open[i] = (status[i] == 1);
        for (auto v : initialBoxes) {
            hasbox[v] = true;
            if (open[v]) {
                q.push(v);
                used[v] = true;
                res += candies[v];
            }
        }
        while (!q.empty()) {
            int b = q.front();
            q.pop();
            for (auto k : keys[b]) haskey[k] = true;
            for (auto k : containedBoxes[b]) hasbox[k] = true;
            for (auto k : keys[b]) {
                if (!used[k] && hasbox[k] && (open[k] || haskey[k])) {
                    q.push(k);
                    used[k] = true;
                    res += candies[k];
                }
            }
            for (auto k : containedBoxes[b]) {
                if (!used[k] && hasbox[k] && (open[k] || haskey[k])) {
                    q.push(k);
                    used[k] = true;
                    res += candies[k];
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

> [!NOTE] **[LeetCode 2092. 找出知晓秘密的所有专家](https://leetcode-cn.com/problems/find-all-people-with-secret/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分析 显然需要根据会议时间 每次建图并跑连通图
> 
> - 寻常思路 set 维护, 每次找会发生变动的部分去跑连通图
> 
>   **踩坑：set 不能边迭代边修改**
> 
> - bfs trick 稳定且极低的复杂度 ==> **反复做**
> 
>   非常有技巧性的 bfs 实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ trick bfs**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 1e5 + 10, M = 2e5 + 10;
    
    bool st[N];
    int d[N];

    int h[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
        memset(st, 0, sizeof st);
        memset(d, 0x3f, sizeof d);
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }

    // [ trick 非常简单且复杂度更低的写法 ]
    // 1. 维护一个优先队列，内部为即将了解信息的人以及时间，按照后者升序排序。
    // 2. 随后我们一个个取出优先队列的元素，并且记录元素的时间为当前时间。
    //    如果这个人是首次知道这个秘密，那么枚举所有连接的边，如果时间不比当前时间早，那么加入到优先队列当中。
    // 3. 这样子写可以摆脱每次都建一次图的问题，而且似乎会让代码更好写。
    vector<int> findAllPeople(int n, vector<vector<int>>& meetings, int firstPerson) {
        init();
        for (auto & m : meetings)
            add(m[0], m[1], m[2]), add(m[1], m[0], m[2]);
        
        // 时间从小到大排  [time_he_know, person]
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        heap.push({d[0] = 0, 0}), heap.push({d[firstPerson] = 0, firstPerson});

        vector<int> res;
        while (heap.size()) {
            auto [_, u] = heap.top(); heap.pop();
            if (st[u])
                continue;
            st[u] = true;
            res.push_back(u);
            for (int i = h[u]; ~i; i = ne[i]) {
                int v = e[i], t = w[i];
                // 如果开会比当前u知道的晚 ==> 说明可以对v发挥作用
                // 且当前开会会使得v更早知道 ==> 可以优化v知道的时间
                if (t >= d[u] && t < d[v])
                    heap.push({d[v] = t, v});
            }
        }
        
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10, M = 2e5 + 10;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
        
        memset(vis, 0, sizeof vis);
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    bool vis[N];
    void dfs(int u) {
        if (vis[u])
            return;
        vis[u] = true;
        S.insert(u);
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            dfs(j);
        }
    }
    
    unordered_set<int> S;
    
    vector<int> findAllPeople(int n, vector<vector<int>>& meetings, int firstPerson) {
        sort(meetings.begin(), meetings.end(), [](const vector<int> & m1, const vector<int> & m2) {
            return m1[2] < m2[2];
        });
        vector<int> ts;
        for (auto & m : meetings)
            ts.push_back(m[2]);
        ts.erase(unique(ts.begin(), ts.end()), ts.end());
        
        S.clear();
        S.insert(0), S.insert(firstPerson);
        
        int m = meetings.size();
        for (int i = 0, p = 0; i < ts.size(); ++ i ) {
            init();
            vector<int> t;
            while (p < meetings.size() && meetings[p][2] <= ts[i]) {
                auto & m = meetings[p];
                int a = m[0], b = m[1];
                // 优化: 只关心变动的部分，而非遍历 S 中所有的点
                // 不优化则 TLE
                if ((S.count(a) == 0) ^ (S.count(b) == 0))
                    t.push_back(a);
                add(a, b), add(b, a);
                p ++ ;
            }
            // dfs 中会修改 S, 如果直接迭代 S 会 WA 37 / 60
            // for (auto v : S)
            //    dfs(v);
            for (auto v : t)
                dfs(v);
        }
        vector<int> res;
        for (int i = 0; i < n; ++ i )
            if (S.count(i))
                res.push_back(i);
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

> [!NOTE] **[LeetCode 1368. 使网格图至少有一条有效路径的最小代价](https://leetcode-cn.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 最短路 可以直接到达的边权为0 需要翻转的点边权为1

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int dx[5] = {0, 1, -1, 0, 0};
const int dy[5] = {0, 0, 0, 1, -1};
typedef pair<int, int> pii;

class Solution {
public:
    int minCost(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        deque<pii> pq;
        pq.push_back(make_pair(0, 0));
        vector<vector<bool>> vis(n, vector<bool>(m));
        while (!pq.empty()) {
            pii f = pq.front();
            pq.pop_front();
            int y = f.second / m, x = f.second % m;
            if (vis[y][x]) continue;
            vis[y][x] = true;
            if (y == n - 1 && x == m - 1)
                return f.first;
            for (int k = 1; k <= 4; ++k) {
                int nx = x + dx[k], ny = y + dy[k];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n)
                    continue;
                if (grid[y][x] == k) 
                    pq.push_front(make_pair(f.first, ny * m + nx));
                else
                    pq.push_back(make_pair(f.first + 1, ny * m + nx));
            }
        }
        return 0;
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

### USACO Training 杂项

> [!NOTE] **[AcWing 1355. 母亲的牛奶](https://www.acwing.com/problem/content/1357/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> bfs 注意**写法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 21;

int A, B, C;
bool st[N][N][N];
struct Node {
    int a, b, c;
}q[N * N * N];
int hh, tt;

void insert(int a, int b, int c) {
    if (!st[a][b][c]) {
        q[ ++ tt] = {a, b, c};
        st[a][b][c] = true;
    }
}

void bfs() {
    q[0] = {0, 0, C};
    st[0][0][C] = true;
    
    while (hh <= tt) {
        auto [a, b, c] = q[hh ++ ];
        insert(a - min(a, B - b), min(a + b, B), c);
        insert(a - min(a, C - c), b, min(a + c, C));
        insert(min(a + b, A), b - min(b, A - a), c);
        insert(a, b - min(b, C - c), min(b + c, C));
        insert(min(a + c, A), b, c - min(A - a, c));
        insert(a, min(b + c, B), c - min(B - b, c));
    }
}

int main() {
    cin >> A >> B >> C;
    
    bfs();
    
    for (int c = 0; c <= C; ++ c )
        for (int b = 0; b <= B; ++ b )
            if (st[0][b][c]) {
                cout << c << ' ';
                break;
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

> [!NOTE] **[AcWing 1374. 穿越栅栏](https://www.acwing.com/problem/content/1376/)**
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

using PII = pair<int, int>;

const int N = 210, M = 100;

int n, m;
string g[N];
int dist[2][N][M];
int dx[] = {-1, 0, 1, 0}, dy[] = {0, 1, 0, -1};

PII q[N * M];
int hh, tt;

// 是空格 则可以走
bool check(int x, int y, int d) {
    return g[x * 2 - 1 + dx[d]][y * 2 - 1 + dy[d]] == ' ';
}

void bfs(int sx, int sy, int dist[][M]) {
    memset(dist, 0x3f, N * M * 4);   // attention
    
    int hh = 0, tt = -1;
    q[ ++ tt] = {sx, sy};
    dist[sx][sy] = 1;
    while (hh <= tt) {
        auto [x, y] = q[hh ++ ];
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 1 || nx > n || ny < 1 || ny > m) continue;
            if (!check(x, y, i)) continue; // 
            if (dist[nx][ny] > dist[x][y] + 1) {
                dist[nx][ny] = dist[x][y] + 1;
                q[ ++ tt] = {nx, ny};
            }
        }
    }
}

int main() {
    cin >> m >> n;
    getchar();
    for (int i = 0; i < n * 2 + 1; ++ i ) getline(cin, g[i]);
    
    int k = 0;
    for (int i = 1; i <= n; ++ i ) {
        if (check(i, 1, 3)) bfs(i, 1, dist[k ++ ]);
        if (check(i, m, 1)) bfs(i, m, dist[k ++ ]);
    }
    for (int i = 1; i <= m; ++ i ) {
        if (check(1, i, 0)) bfs(1, i, dist[k ++ ]);
        if (check(n, i, 2)) bfs(n, i, dist[k ++ ]);
    }
    
    int res = 0;
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            res = max(res, min(dist[0][i][j], dist[1][i][j]));
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