

## 习题

> [!NOTE] **[AcWing 901. 滑雪](https://www.acwing.com/problem/content/903/)**
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

const int maxn = 305;
int n, m;
int h[maxn][maxn];
int f[maxn][maxn];

int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
int dp(int x, int y) {
    if (f[x][y] != -1) return f[x][y];
    f[x][y] = 1;
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx < 1 || nx > n || ny < 1 || ny > m || h[nx][ny] >= h[x][y])
            continue;
        f[x][y] = max(f[x][y], dp(nx, ny) + 1);
    }
    return f[x][y];
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j) cin >> h[i][j];
    int res = 0;
    memset(f, -1, sizeof f);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j) res = max(res, dp(i, j));
    cout << res << endl;
}
```

##### **Python**

```python
# 这个题常见的应该是dfs和bfs，dp的思想是对以及访问过的点算出来最大路径，从而每次不用重复计算，这个优化dfs本身的方法其实是一样的

# 状态表示：f[i,j] 所有从(i,j)开始滑的路径
# 状态计算：按照第一步往哪个方向滑的，可以把所有情况分成四类：向上，下，左，右。
# 比如向右滑的最大值：(i,j)-->(i,j+1) ，那就是求(i,j+1)开始滑的最大值，那就是f[i,j+1],有一个前提条件：那就是h[i,j]>h[i,j+1]

def dp(x, y):
    if f[x][y] != -1:
        return f[x][y]  # 踩坑：判断是否已经计算过了，计算过了 直接返回，这就是记忆化搜索
    f[x][y] = 1  # 如果这个点没有被计算过，那初始值是为1的。（自己也算一步）
    dx, dy = [0, 0, -1, 1], [1, -1, -0, 0]
    for i in range(4):
        a, b = x + dx[i], y + dy[i]
        if 1 <= a <= n and 1 <= b <= m and g[x][y] > g[a][b]:
            f[x][y] = max(f[x][y], dp(a, b) + 1)
    return f[x][y]


if __name__ == '__main__':
    N = 310
    g = [[0] * N for _ in range(N)]
    f = [[-1] * N for _ in range(N)]
    dx = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    n, m = map(int, input().split())
    for i in range(1, n + 1):
        nums = map(int, input().split())
        for j, val in enumerate(nums):
            g[i][j + 1] = val
    # 这样处理也可以哦         
    # g1=[[0]*N]
    # for i in range(n):
    #   g1.append([0] + list(map(int, input().split())))

    # 更简约的处理输入：==> 就让下标从0开始，下面遍历也需要改成i,j从0到n/m，dp函数判断也是[0,n/m]
    # g = []
    # for i in range(n):
    #    g.append(list(map(int, input().split())))
    res = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            res = max(res, dp(i, j))
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *