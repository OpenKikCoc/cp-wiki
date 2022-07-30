## 习题

> [!NOTE] **[AcWing 1402. 星空之夜](https://www.acwing.com/problem/content/1404/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 考虑设计一个 hash 使得图形翻转 hash 值不变
> 
> 本题 计算每个小格子之间的距离之和

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;

const int N = 110;

int n, m;
char g[N][N];
PII q[N * N];
int top;

double get_dist(PII a, PII b) {
    double dx = a.first - b.first;
    double dy = a.second - b.second;
    return sqrt(dx * dx + dy * dy);
}

double get_hash() {
    double res = 0;
    for (int i = 0; i < top; ++ i )
        for (int j = i + 1; j < top; ++ j )
            res += get_dist(q[i], q[j]);
    return res;
}

char get_id(double s) {
    static double hash[N];
    static int cnt = 0;
    for (int i = 0; i < cnt; ++ i )
        if (fabs(s - hash[i]) < 1e-6)
            return i + 'a';
    hash[cnt ++ ] = s;
    return cnt - 1 + 'a';
}

void dfs(int a, int b) {
    g[a][b] = '0';
    q[top ++ ] = {a, b};
    for (int x = a - 1; x <= a + 1; ++ x )
        for (int y = b - 1; y <= b + 1; ++ y )
            if (x != a || y != b)
                if (x >= 0 && x < n && y >= 0 && y < m && g[x][y] == '1')
                    dfs(x, y);
}

int main() {
    cin >> m >> n;
    for (int i = 0; i < n; ++ i ) cin >> g[i];
    
    for (int i = 0; i < n; ++ i )
        for (int j = 0; j < m; ++ j )
            if (g[i][j] == '1') {
                top = 0;
                dfs(i, j);
                auto s = get_hash();
                char c = get_id(s);
                for (int k = 0; k < top; ++ k )
                    g[q[k].first][q[k].second] = c;
            }
    
    for (int i = 0; i < n; ++ i ) cout << g[i] << endl;
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

> [!NOTE] **[LeetCode 959. 由斜杠划分区域](https://leetcode.cn/problems/regions-cut-by-slashes/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一个格子按对角线拆分为四个格子
> 
> **注意 `dx` / `dy` 与 `k` 的设置是对应的**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 容易想到把一个格子分成四个小三角（能够被交叉的斜杠分开）
    const static int N = 3610;

    int pa[N];
    void init() {
        for (int i = 0; i < N; ++ i )
            pa[i] = i;
    }
    int find(int x) {
        if (pa[x] != x)
            pa[x] = find(pa[x]);
        return pa[x];
    }

    int n;
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};   // ATTENTION 对应
    int get(int x, int y, int k) {
        return (x * n + y) * 4 + k;
    }

    int regionsBySlashes(vector<string>& grid) {
        init();
        this->n = grid.size();

        for (int x = 0; x < n; ++ x )
            for (int y = 0; y < n; ++ y ) {
                // 与周围其他方块连接
                for (int k = 0; k < 4; ++ k ) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= n)
                        continue;
                    pa[find(get(x, y, k))] = find(get(nx, ny, k ^ 2));   // ATTENTION k ^ 2
                }

                // 与内部其他三角相连
                if (grid[x][y] != '/') {
                    pa[find(get(x, y, 0))] = find(get(x, y, 1));
                    pa[find(get(x, y, 2))] = find(get(x, y, 3));
                }
                if (grid[x][y] != '\\') {
                    pa[find(get(x, y, 0))] = find(get(x, y, 3));
                    pa[find(get(x, y, 1))] = find(get(x, y, 2));
                }
            }
        
        int res = 0;
        for (int i = 0; i < n * n * 4; ++ i )
            if (find(i) == i)
                res ++ ;
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