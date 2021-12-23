
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
