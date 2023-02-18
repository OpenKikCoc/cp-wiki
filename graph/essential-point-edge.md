# TODO 待整理


## 习题

> [!NOTE] **[LeetCode 2556. 二进制矩阵中翻转最多一次使路径不连通](https://leetcode.cn/problems/disconnect-path-in-a-binary-matrix-by-at-most-one-flip/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有向图的必经点
>
> DAG 条件下的简易解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 思考与转化：
    //      如果建立有向图，显然求【必经点】
    //      又因本题特殊，有 DAG 性质，可以直接
    //      => https://www.cnblogs.com/PPXppx/p/11590709.html
    using LL = long long;
    
    bool isPossibleToCutPath(vector<vector<int>>& grid) {
        const int MOD = rand() % 1000000 + 1e9 + 7; // ATTENTION 添加随机种子
        int n = grid.size(), m = grid[0].size();
        
        LL f[n + 2][m + 2], g[n + 2][m + 2];
        memset(f, 0, sizeof f), memset(g, 0, sizeof g);
        
        // 计算方案数
        f[1][1] = g[n][m] = 1;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                if (i == 1 && j == 1)
                    continue;
                if (grid[i - 1][j - 1] == 1)
                    f[i][j] = (f[i - 1][j] + f[i][j - 1]) % MOD;
            }
        for (int i = n; i > 0; -- i )
            for (int j = m; j > 0; -- j ) {
                if (i == n && j == m)
                    continue;
                if (grid[i - 1][j - 1] == 1)
                    g[i][j] = (g[i + 1][j] + g[i][j + 1]) % MOD;
            }
        
        if (f[n][m] == 0)
            return true;
        
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                if (i == 1 && j == 1)
                    continue;
                if (i == n && j == m)
                    continue;
                LL t = f[i][j] * g[i][j] % MOD;
                if (t == f[n][m])
                    return true;
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