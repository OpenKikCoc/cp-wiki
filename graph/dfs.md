
DFS 全称是 Depth First Search，中文名是深度优先搜索，是一种用于遍历或搜索树或图的算法。所谓深度优先，就是说每次都尝试向更深的节点走。

### DFS 序列

DFS 序列是指 DFS 调用过程中访问的节点编号的序列。

我们发现，每个子树都对应 DFS 序列中的连续一段（一段区间）。

### 括号序列

DFS 进入某个节点的时候记录一个左括号 `(`，退出某个节点的时候记录一个右括号 `)`。

每个节点会出现两次。相邻两个节点的深度相差 1。

### 一般图上 DFS

对于非连通图，只能访问到起点所在的连通分量。

对于连通图，DFS 序列通常不唯一。

注：树的 DFS 序列也是不唯一的。

在 DFS 过程中，通过记录每个节点从哪个点访问而来，可以建立一个树结构，称为 DFS 树。DFS 树是原图的一个生成树。

[DFS 树](./scc.md#dfs) 有很多性质，比如可以用来求 [强连通分量](./scc.md)。

## 习题

> [!NOTE] **[AcWing 842. 排列数字](https://www.acwing.com/problem/content/844/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>

using namespace std;

const int N = 10;

int n;
int path[N];

void dfs(int u, int state) {
    if (u == n) {
        for (int i = 0; i < n; i++) printf("%d ", path[i]);
        puts("");

        return;
    }

    for (int i = 0; i < n; i++)
        if (!(state >> i & 1)) {
            path[u] = i + 1;
            dfs(u + 1, state + (1 << i));
        }
}

int main() {
    scanf("%d", &n);

    dfs(0, 0);

    return 0;
}
```

##### **Python**

```python
def dfs(path, length):
    if length == n:
        res.append(path[:])
        return
    for i in range(1, n + 1):
        if not used[i]:
            used[i] = True
            path.append(i)
            dfs(path, length + 1)
            used[i] = False
            path.pop()


if __name__ == '__main__':
    n = int(input())
    res = []
    used = [False] * (n + 1)

    dfs([], 0)
    for i in range(len(res)):
        for j in range(n):
            print(res[i][j], end=' ')
        print()
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 843. n-皇后问题](https://www.acwing.com/problem/content/845/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
// 第一种搜索顺序
#include <iostream>

using namespace std;

const int N = 10;

int n;
bool row[N], col[N], dg[N * 2], udg[N * 2];
char g[N][N];

void dfs(int x, int y, int s) {
    if (s > n) return;
    if (y == n) y = 0, x++;

    if (x == n) {
        if (s == n) {
            for (int i = 0; i < n; i++) puts(g[i]);
            puts("");
        }
        return;
    }

    g[x][y] = '.';
    dfs(x, y + 1, s);

    if (!row[x] && !col[y] && !dg[x + y] && !udg[x - y + n]) {
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = true;
        g[x][y] = 'Q';
        dfs(x, y + 1, s + 1);
        g[x][y] = '.';
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = false;
    }
}

int main() {
    cin >> n;

    dfs(0, 0, 0);

    return 0;
}
```

##### **C++ 2**

```cpp
// 第二种搜索顺序
#include <iostream>

using namespace std;

const int N = 20;

int n;
char g[N][N];
bool col[N], dg[N], udg[N];

void dfs(int u) {
    if (u == n) {
        for (int i = 0; i < n; i++) puts(g[i]);
        puts("");
        return;
    }

    for (int i = 0; i < n; i++)
        if (!col[i] && !dg[u + i] && !udg[n - u + i]) {
            g[u][i] = 'Q';
            col[i] = dg[u + i] = udg[n - u + i] = true;
            dfs(u + 1);
            col[i] = dg[u + i] = udg[n - u + i] = false;
            g[u][i] = '.';
        }
}

int main() {
    cin >> n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) g[i][j] = '.';

    dfs(0);

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

> [!NOTE] **[AcWing 1357. 优质牛肋骨](https://www.acwing.com/problem/content/1359/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 搜索 注意写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;

bool is_prime(int x) {
    for (int i = 2; i <= x / i; ++ i )
        if (x % i == 0)
            return false;
    return true;
}

void dfs(int x, int k) {
    if (!is_prime(x)) return;
    if (k == n) cout << x << endl;
    else {
        int d[] = {1, 3, 7, 9};
        for (int i : d)
            dfs(x * 10 + i, k + 1);
    }
}

int main() {
    cin >> n;
    dfs(2, 1), dfs(3, 1), dfs(5, 1), dfs(7, 1);
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


> [!NOTE] **[AcWing 1363. 汉明码](https://www.acwing.com/problem/content/1365/)**
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

const int N = 256;

int n, b, d;
bool g[N][N];
int path[N];

int get_ones(int x) {
    int res = 0;
    while (x) res += x & 1, x >>= 1;
    return res;
}

bool dfs(int u, int start) {
    if (u == n) {
        for (int i = 0; i < n; ++ i ) {
            cout << path[i];
            if ((i + 1) % 10) cout << ' ';
            else cout << endl;
        }
        return true;
    }
    for (int i = start; i < 1 << b; ++ i ) {
        bool flag = true;
        for (int j = 0; j < u; ++ j )
            if (!g[i][path[j]]) {
                flag = false;
                break;
            }
        if (flag) {
            path[u] = i;
            if (dfs(u + 1, i + 1)) return true;
        }
    }
    return false;
}

int main() {
    cin >> n >> b >> d;
    for (int i = 0; i < 1 << b; ++ i )
        for (int j = 0; j < 1 << b; ++ j )
            // if (__builtin_popcount(i ^ j) >= d)
            if (get_ones(i ^ j) >= d)
                g[i][j] = true;
    // 显然0必选 再选后面其他的    
    dfs(1, 1);
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

> [!NOTE] **[AcWing 1372. 控股公司](https://www.acwing.com/problem/content/1374/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **分析** 更新图的思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 110;

int n = 100, m;
int w[N][N];
bool g[N][N];

void dfs(int x, int y) {
    // return 避免无限递归
    if (g[x][y]) return;
    g[x][y] = true;
    
    for (int i = 1; i <= n; ++ i ) w[x][i] += w[y][i];
    for (int i = 1; i <= n; ++ i )
        if (g[i][x])
            dfs(i, y);
    for (int i = 1; i <= n; ++ i )
        if (w[x][i] > 50)
            dfs(x, i);
}

int main() {
    for (int i = 1; i <= n; ++ i ) g[i][i] = true;
    
    cin >> m;
    while (m -- ) {
        int a, b, c;
        cin >> a >> b >> c;
        for (int i = 1; i <= n; ++ i )
            if (g[i][a]) {
                w[i][b] += c;
                if (w[i][b] > 50) dfs(i, b);    // 虚边变为实边
            }
    }
    
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= n; ++ j )
            if (i != j && g[i][j])
                cout << i << ' ' << j << endl;
                
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

> [!NOTE] **[AcWing 1404. 蜗牛漫步](https://www.acwing.com/problem/content/1406/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **经典回溯思想**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 爆搜即可
// 回溯精髓题
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const int N = 130;

int n, m;
int g[N][N];
int dx[] = {-1, 0, 1, 0}, dy[] = {0, 1, 0, -1};
int ans;

void dfs(int x, int y, int d, int k) {
    if (!x || x > n || !y || y > n || g[x][y]) return;
    // 栈存储本次走一行/一列所走过的点 方便恢复
    stack<PII> stk;
    while (x && x <= n && y && y <= n && !g[x][y]) {
        g[x][y] = 2;
        stk.push({x, y});
        x += dx[d], y += dy[d];
        k ++ ;
    }
    ans = max(ans, k);
    // 转弯
    if (!x || x > n || !y || y > n || g[x][y] == 1) {
        x -= dx[d], y -= dy[d];
        for (int i = 0; i < 4; ++ i )
            if ((i + d) % 2)    // 90度方向的 相加都是奇数
                dfs(x + dx[i], y + dy[i], i, k);
    }
    
    while (stk.size()) {
        auto [x, y] = stk.top(); stk.pop();
        g[x][y] = 0;
    }
}

int main() {
    cin >> n >> m;
    while (m -- ) {
        char a; int b;
        cin >> a >> b;
        a = a - 'A' + 1;
        g[b][a] = 1;
    }
    dfs(1, 1, 1, 0);
    dfs(1, 1, 2, 0);
    cout << ans << endl;
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

> [!NOTE] **[AcWing 1421. 威斯康星方形牧场](https://www.acwing.com/problem/content/1423/)**
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

const int N = 5;

char g[N][N];
int num[] = {3, 3, 3, 4, 3};
bool st[N][N];
int ans;
struct Node {
    char c;
    int a, b;
} path[N * N];

bool check(int a, int b, int c) {
    for (int i = a - 1; i <= a + 1; ++ i )
        for (int j = b - 1; j <= b + 1; ++ j )
            if (i >= 0 && i < 4 && j >= 0 && j < 4 && g[i][j] == c)
                return false;
    return true;
}

void dfs(char c, int u) {
    if (u == 16) {
        ans ++ ;
        if (ans == 1) {
            for (int i = 0; i < 16; ++ i )
                cout << path[i].c << ' ' << path[i].a + 1 << ' ' << path[i].b + 1 << endl;
        }
        return;
    }
    
    for (int i = 0; i < 4; ++ i )
        for (int j = 0; j < 4; ++ j )
            // 没放过 且可以填c
            if (!st[i][j] && check(i, j, c)) {
                st[i][j] = true;
                char t = g[i][j];
                g[i][j] = c;
                path[u] = {c, i, j};
                num[c - 'A'] -- ;
                
                // u == 15 时任意点搜下一层 否则会多次重复计数ans++
                if (u == 15) dfs('A', u + 1);
                else {
                    for (int k = 0; k < 5; ++ k )
                        if (num[k] > 0)
                            dfs(k + 'A', u + 1);
                }
                
                num[c - 'A'] ++ ;
                g[i][j] = t;
                st[i][j] = false;
            }
}

int main() {
    for (int i = 0; i < 4; ++ i ) cin >> g[i];
    dfs('D', 0);
    cout << ans << endl;
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

> [!NOTE] **[LeetCode 133. 克隆图](https://leetcode-cn.com/problems/clone-graph/)**
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
    Node* vis[101];
    Node* cloneGraph(Node* node) {
        if (!node)
            return nullptr;
        if (vis[node->val])
            return vis[node->val];

        Node * p = new Node(node->val);
        vis[node->val] = p;
        
        vector<Node*> nb = node->neighbors;
        for (int i = 0; i < nb.size(); ++ i )
            p->neighbors.push_back(cloneGraph(nb[i]));
        return p;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    unordered_map<Node*, Node*> hash;

    Node* cloneGraph(Node* node) {
        if (!node) return NULL;
        dfs(node);  // 复制所有点

        for (auto [s, d]: hash) {
            for (auto ver: s->neighbors)
                d->neighbors.push_back(hash[ver]);
        }

        return hash[node];
    }

    void dfs(Node* node) {
        hash[node] = new Node(node->val);

        for (auto ver: node->neighbors)
            if (hash.count(ver) == 0)
                dfs(ver);
    }
};
```

##### **Python**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

# 递归地访问(复制)邻居节点，用一个map来存储节点访问情况
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        my_dict = dict()
        if not node:return None

        def dfs(node):
            my_dict[node]=Node(node.val)
            for ver in node.neighbors:
                if ver not in my_dict:
                    dfs(ver)

        dfs(node)
        for k,v in my_dict.items():
            for ver in k.neighbors:
                v.neighbors.append(my_dict[ver])
        return my_dict[node]
```

<!-- tabs:end -->
</details>

<br>

* * *

### 遍历问题

> [!NOTE] **[LeetCode 130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)**
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
    int n, m;
    vector<vector<char>> board;
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    
    void dfs(int x, int y) {
        board[x][y] = '#';
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && board[a][b] == 'O')
                dfs(a, b);
        }
    }

    void solve(vector<vector<char>>& _board) {
        board = _board;
        n = board.size();
        if (!n) return;
        m = board[0].size();

        for (int i = 0; i < n; i ++ ) {
            if (board[i][0] == 'O') dfs(i, 0);
            if (board[i][m - 1] == 'O') dfs(i, m - 1);
        }

        for (int i = 0; i < m; i ++ ) {
            if (board[0][i] == 'O') dfs(0, i);
            if (board[n - 1][i] == 'O') dfs(n - 1, i);
        }

        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                if (board[i][j] == '#') board[i][j] = 'O';
                else board[i][j] = 'X';

        _board = board;
    }
};
```

##### **Python**

```python
# 方法：dfs
# 1. 遍历grid的四周，当有O时， 进行dfs， 把与边界相连的O，并且也为O的点 置为‘#’。
# 2. 然后 再遍历整个grid，把O 变为 1，把 # 变为 O即可
class Solution:
    def solve(self, arr: List[List[str]]) -> None:
        if not arr:return 
        n, m = len(arr), len(arr[0])

        def dfs(x, y):
            arr[x][y] = '#'  # 踩坑：这里误写成 arr[x][y] == '#'过！！！
            dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and arr[nx][ny] == 'O':
                    dfs(nx, ny)

        for i in range(m):
            if arr[0][i] == 'O':
                dfs(0, i)
            if arr[n-1][i] =='O':
                dfs(n-1, i)
        for i in range(n):
            if arr[i][0] == 'O':
                dfs(i, 0)
            if arr[i][m-1] == 'O':
                dfs(i, m-1)
        
        for i in range(n):
            for j in range(m):
                if arr[i][j] == 'O':
                    arr[i][j] = 'X'
                elif arr[i][j] == '#':
                    arr[i][j] = 'O'
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)**
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
    int m, n;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    void dfs(int x, int y, vector<vector<char>>& grid) {
        grid[x][y] = '0';
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < m && nx >= 0 && ny < n && ny >= 0 && grid[nx][ny] == '1') {
                dfs(nx, ny, grid);
            }
        }
    }
    int numIslands(vector<vector<char>>& grid) {
        m = grid.size();
        if (!m) return 0;
        n = grid[0].size();
        int res = 0;
        for (int i = 0; i < m; ++ i ) {
            for (int j = 0;  j < n; ++ j ) {
                if (grid[i][j] == '1') {
                    dfs(i, j, grid);
                    res ++ ;
                }
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# Flood Fill算法-- DFS模版

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n, m = len(grid), len(grid[0])
        res = 0

        def dfs(x, y):
            grid[x][y] = '0'
            dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == '1':
                    dfs(nx, ny)
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res += 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    int n, m;
    vector<vector<int>> w;
    vector<vector<int>> st;

    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

    void dfs(int x, int y, int t) {
        if (st[x][y] & t) return;
        st[x][y] |= t;
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && w[a][b] >= w[x][y])
                dfs(a, b, t);
        }
    }

    vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
        w = matrix;
        if (w.empty() || w[0].empty()) return {};
        n = w.size(), m = w[0].size();
        st = vector<vector<int>>(n, vector<int>(m));

        for (int i = 0; i < n; i ++ ) dfs(i, 0, 1);
        for (int i = 0; i < m; i ++ ) dfs(0, i, 1);
        for (int i = 0; i < n; i ++ ) dfs(i, m - 1, 2);
        for (int i = 0; i < m; i ++ ) dfs(n - 1, i, 2);

        vector<vector<int>> res;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                if (st[i][j] == 3)
                    res.push_back({i, j});
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int m, n;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    void dfs(int x, int y, vector<vector<int>>& matrix, vector<vector<bool>>& vis) {
        vis[x][y] = true;
        int nx, ny;
        for (int i = 0; i < 4; ++ i ) {
            nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !vis[nx][ny] && matrix[x][y] <= matrix[nx][ny])
                dfs(nx, ny, matrix, vis);
        }
    }
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
        vector<vector<int>> res;
        if (matrix.empty()) return res;
        m = matrix.size();
        n = matrix[0].size();
        vector<vector<bool>> canp(m, vector<bool>(n)), cana(m, vector<bool>(n));
        for (int i = 0; i < n; ++ i ) dfs(0, i, matrix, canp);
        for (int i = 0; i < m; ++ i ) dfs(i, 0, matrix, canp);
        for (int i = 0; i < n; ++ i ) dfs(m - 1, i, matrix, cana);
        for (int i = 0; i < m; ++ i ) dfs(i, n - 1, matrix, cana);
        for (int i = 0; i < m; ++ i ) {
            for (int j = 0; j < n; ++ j ) {
                if (cana[i][j] && canp[i][j]) res.push_back({i, j});
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# 1. 分别从太平洋和大西洋边界位置出发遍历，能同时被他们遍历到的 就是满足条件的
# 2. 用二进制为表示是否合法。第一位表示太平洋 为1合法，第二位表示大西洋 为1合法；所以加起来当 当前格子的值为3时 就是满足条件的

class Solution:
    def pacificAtlantic(self, arr: List[List[int]]) -> List[List[int]]:
        n, m = len(arr), len(arr[0])
        st = [[0] * m for _ in range(n)]
        
        def dfs(x, y, t):
            if st[x][y] & t:return # 如果被遍历过，就return（不是continue!!!)
            st[x][y] |= t   # 如果没有被遍历过，又符合条件，就改变当前格子的st的值
            dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and arr[x][y] <= arr[nx][ny]:
                    dfs(nx, ny, t)
        
        for i in range(m):dfs(0, i, 1)
        for i in range(n):dfs(i, 0, 1)
        for i in range(m):dfs(n - 1, i, 2)
        for i in range(n):dfs(i, m - 1, 2)

        res = []
        for i in range(n):
            for j in range(m):
                if st[i][j] == 3:
                    res.append([i, j])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 463. 岛屿的周长](https://leetcode-cn.com/problems/island-perimeter/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ trick**

```cpp
class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
        int res = 0;
        for (int i = 0; i < grid.size(); i ++ )
            for (int j = 0; j < grid[i].size(); j ++ )
                if (grid[i][j] == 1) {
                    for (int k = 0; k < 4; k ++ ) {
                        int x = i + dx[k], y = j + dy[k];
                        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size())
                            res ++ ;
                        else if (grid[x][y] == 0) res ++ ;
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
    int n, m, res;
    vector<vector<int>> g;
    vector<vector<bool>> vis;
    vector<int> dx = {-1, 0, 0, 1}, dy = {0, -1, 1, 0};
    void dfs(int x, int y) {
        // cout << x << " " << y << endl;
        vis[x][y] = true;
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m || !g[nx][ny]) {
                ++ res ;
                continue;
            }
            if (vis[nx][ny]) continue;
            dfs(nx, ny);
        }
    }
    int islandPerimeter(vector<vector<int>>& grid) {
        n = grid.size(), m = grid[0].size();
        res = 0;
        g = grid;
        vis = vector<vector<bool>>(n, vector<bool>(m));
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (g[i][j] && !vis[i][j])
                    dfs(i, j);
        return res;
    }
};
```

##### **Python**

```python
# 不需要BFS/DFS 直接暴力遍历整个矩阵 发现岛屿块 检测上下左右四个方向即可(只要直接数一下边界数)
# 当发现1时，枚举四个方向：1）当出界  or  2) 从1 变成 0的话，说明就是岛屿的边界

class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        res = 0
        dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    for k in range(4):
                        nx, ny = i + dx[k], j + dy[k]
                        if nx < 0 or nx >= n or ny < 0 or ny >= m:
                            res += 1
                        elif grid[nx][ny] == 0:
                            res += 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)**
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
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    vector<vector<int>> updateMatrix(vector<vector<int>> & matrix) {
        if (matrix.empty() || matrix[0].empty()) return matrix;
        int n = matrix.size(), m = matrix[0].size();
        vector<vector<int>> dist(n, vector<int>(m, -1));
        queue<PII> q;

        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (matrix[i][j] == 0)
                    dist[i][j] = 0, q.push({i, j});
        while (!q.empty()) {
            auto [x, y] = q.front(); q.pop();
            for (int i = 0; i < 4; ++ i ) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m || dist[nx][ny] != -1) continue;
                dist[nx][ny] = dist[x][y] + 1;
                q.push({nx, ny});
            }
        }
        return dist;
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

> [!NOTE] **[LeetCode 1254. 统计封闭岛屿的数目](https://leetcode-cn.com/problems/number-of-closed-islands/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dfs 注意写法和判断
> 
> 如果发现 f 为 true 直接返回的话显然没有完全遍历 小细节[不能有true直接返回]

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, m;
    bool f;
    vector<vector<int>> g;
    vector<vector<bool>> v;
    // 能否走到边界 不能说明被水包围
    vector<int> dx = {-1, 0, 0, 1}, dy = {0, -1, 1, 0};
    void dfs(int x, int y) {
        v[x][y] = true;
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m) {
                f = true;
                continue;
            }
            if (v[nx][ny] || g[nx][ny] == 1) continue;
            dfs(nx, ny);
        }
    }
    int closedIsland(vector<vector<int>>& grid) {
        n = grid.size(), m = grid[0].size();
        g = grid;
        v = vector<vector<bool>>(n, vector<bool>(m));
        int res = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                if (!v[i][j] && g[i][j] == 0) {
                    f = false;
                    dfs(i, j);
                    if (!f) ++res;
                }
        return res;
    }
}
```

##### **C++ trick**

```cpp
class Solution {
public:
    int check(int x, int y, vector<vector<int>>& grid,
              vector<vector<int>>& vis) {
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size()) return 0;
        if (grid[x][y] || vis[x][y]) return 1;
        vis[x][y] = 1;
        return check(x + 1, y, grid, vis) & check(x - 1, y, grid, vis) &
               check(x, y + 1, grid, vis) & check(x, y - 1, grid, vis);
    }
    int closedIsland(vector<vector<int>>& grid) {
        vector<vector<int>> vis(grid.size(), vector<int>(grid[0].size(), 0));
        int res = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[i].size(); ++j) {
                if (vis[i][j] || grid[i][j]) continue;
                res += check(i, j, grid, vis);
            }
        }
        return res;
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

> [!NOTE] **[LeetCode 1559. 二维网格图中探测环](https://leetcode-cn.com/problems/detect-cycles-in-2d-grid/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dfs 记录写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int move[5] = {-1, 0, 1, 0, -1};
    int m, n;
    bool containsCycle(vector<vector<char>>& grid) {
        n = grid.size();
        m = grid[0].size();
        vector<vector<bool>> visited(n, vector<bool>(m));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                if (!visited[i][j])
                    if (dfs(grid, i, j, -1, -1, visited)) return true;
        return false;
    }

    bool dfs(vector<vector<char>>& grid, int r, int c, int fr, int fc,
             vector<vector<bool>>& visited) {
        visited[r][c] = true;
        for (int i = 0; i < 4; i++) {
            int nr = r + move[i];
            int nc = c + move[i + 1];
            if (nr < 0 || nr >= n || nc < 0 || nc >= m ||
                grid[nr][nc] != grid[r][c])
                continue;
            if (visited[nr][nc]) {
                if (nr != fr || nc != fc)
                    return true;
                else
                    continue;
            }
            if (dfs(grid, nr, nc, r, c, visited)) return true;
        }
        return false;
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

> [!NOTE] **[LeetCode 1905. 统计子岛屿](https://leetcode-cn.com/problems/count-sub-islands/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单 dfs 注意判断子岛屿的全局标记方式即可
> 
> 重点在于简化的原因：
> 
> > dfs g2过程中不会出现碰到两个g1岛屿的情况（否则g1这两块必然联通或出现了非岛屿的点）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> g1, g2, num, st;
    int n, m;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    
    bool f;
    
    void dfs1(int x, int y, int k) {
        num[x][y] = k;
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && g1[nx][ny] && !num[nx][ny])
                dfs1(nx, ny, k);
        }
    }
    
    void dfs2(int x, int y, int k) {
        if (num[x][y] != k)
            f = false;
        
        st[x][y] = k;
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && g2[nx][ny] && !st[nx][ny])
                dfs2(nx, ny, k);
        }
    }
    
    int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2) {
        this->g1 = grid1, this->g2 = grid2;
        n = g1.size(), m = g1[0].size();
        num = st = vector<vector<int>>(n, vector<int>(m, 0));
        
        {
            int cnt = 0;
            for (int i = 0; i < n; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (g1[i][j] && !num[i][j])
                        dfs1(i, j, ++ cnt);
        }
        
        int res = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (g2[i][j] && !st[i][j] && num[i][j]) {
                    f = true;
                    dfs2(i, j, num[i][j]);
                    
                    if (f)
                        res ++ ;
                }
        
        return res;
    }
};
```

##### **C++ 简化**

```cpp
class Solution {
public:
    int n, m;
    vector<vector<int>> g1, g2;
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    
    bool dfs(int x, int y) {
        bool f = true;
        if (!g1[x][y]) f = false;
        g2[x][y] = 0;
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && g2[a][b]) {
                if (!dfs(a, b)) f = false;
            }
        }
        return f;
    }
    
    int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2) {
        g1 = grid1, g2 = grid2;
        n = g1.size(), m = g1[0].size();
        int res = 0;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                if (g2[i][j]) {
                    if (dfs(i, j)) res ++ ;
                }
        
        return res;
    }
};
```

##### **Python**

```python
# python
class Solution:
    def countSubIslands(self, g1: List[List[int]], g2: List[List[int]]) -> int:
        n, m = len(g1), len(g1[0])
        dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
        num = [[0] * m for _ in range(n)]
        st = [[0] * m for _ in range(n)]
        
        self.f = False
        
        def dfs1(x, y, k):
            num[x][y] = k 
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and g1[nx][ny] and not num[nx][ny]:
                    dfs1(nx, ny, k)
                    
        def dfs2(x, y, k):
            if num[x][y] != k:
                self.f = False 
            
            st[x][y] = k
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and g2[nx][ny] and not st[nx][ny]:
                    dfs2(nx, ny, k)
            
                    
        cnt = 0
        for i in range(n):
            for j in range(m):
                if g1[i][j] and not num[i][j]:
                    cnt += 1
                    dfs1(i, j, cnt)
        
        res = 0 
        for i in range(n):
            for j in range(m):
                if g2[i][j] and not st[i][j] and num[i][j]:
                    self.f = True 
                    dfs2(i, j, num[i][j])
                    
                    if (self.f):
                        res += 1
        return res       
```

<!-- tabs:end -->
</details>

<br>

* * *

### 进阶

> [!NOTE] **[LeetCode 749. 隔离病毒](https://leetcode-cn.com/problems/contain-virus/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 勇于暴力 find + dfs + 标记
> 
> 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 本质是模拟题
    using PII = pair<int, int>;
    #define x first
    #define y second

    vector<vector<int>> g;
    vector<vector<bool>> st;
    vector<PII> path;
    int n, m;
    set<PII> S;
    int dx[4] = {0, -1, 1, 0}, dy[4] = {1, 0, 0, -1};

    int dfs(int x, int y) {
        st[x][y] = true;
        path.push_back({x, y});
        int res = 0;
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                continue;
            if (!g[nx][ny])
                S.insert({nx, ny}), res ++ ;
            else if (g[nx][ny] == 1 && !st[nx][ny])
                res += dfs(nx, ny);
        }
        return res;
    }

    // 找到即将扩散最多的那个区域
    int find() {
        st = vector<vector<bool>>(n, vector<bool>(m));
        int cnt = 0, res = 0;
        // 保存该连通块内所有的点 方便后面标记
        vector<set<PII>> ss;
        vector<PII> ps;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (g[i][j] == 1 && !st[i][j]) {
                    path.clear(), S.clear();
                    int t = dfs(i, j);
                    if (S.size() > cnt) {
                        cnt = S.size();
                        res = t;
                        ps = path;
                    }
                    ss.push_back(S);
                }
        // 设置
        for (auto & p : ps)
            g[p.x][p.y] = -1;
        // 恢复其他
        for (auto & s : ss)
            if (s.size() != cnt)
                for (auto & p : s)
                    g[p.x][p.y] = 1;
        return res;
    }

    int containVirus(vector<vector<int>>& grid) {
        g = grid;
        n = g.size(), m = g[0].size();
        int res = 0;
        for (;;) {
            auto cnt = find();
            if (!cnt)
                break;
            res += cnt;
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

> [!NOTE] **[LeetCode 1568. 使陆地分离的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-disconnect-island/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> floodfill + tarjan
> 
> 关键在于想通 在只有一个岛屿的情况下，最多经过两天一定可以分割成两个岛屿。tarjan 求割点，存在则为 1 否则 2
> 
> 如果不用 tarjan 由上述结论可以每次替换一个 1->0 如果成功就返回1 否则结束时返回2
> 
> 复杂度 O(n^4)
> 
> 这题本质只需改一个点，所以不用tarjan的复杂度可以接受，直接改即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, m;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    void dfs(vector<vector<int>>& g, int x, int y) {
        g[x][y] = 0;
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || g[nx][ny] == 0)
                continue;
            g[nx][ny] = 0;
            dfs(g, nx, ny);
        }
    }
    int cal(vector<vector<int>> g) {
        int cnt = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (g[i][j] == 1) {
                    ++cnt;
                    if (cnt > 1) return 2;
                    dfs(g, i, j);
                }
        return cnt;
    }
    int minDays(vector<vector<int>>& grid) {
        m = grid.size(), n = grid[0].size();
        int t = cal(grid);
        if (t == 0 || t == 2) return 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j] == 1) {
                    grid[i][j] = 0;
                    t = cal(grid);
                    if (t == 2) return 1;
                    grid[i][j] = 1;
                }
        return 2;
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