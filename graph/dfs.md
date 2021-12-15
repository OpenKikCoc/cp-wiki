
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