## 简介

迭代加深是一种 **每次限制搜索深度的** 深度优先搜索。

它的本质还是深度优先搜索，只不过在搜索的同时带上了一个深度 $d$，当 $d$ 达到设定的深度时就返回，一般用于找最优解。如果一次搜索没有找到合法的解，就让设定的深度加一，重新从根开始。

## 步骤

首先设定一个较小的深度作为全局变量，进行 DFS。每进入一次 DFS，将当前深度加一，当发现 $d$ 大于设定的深度 $\textit{limit}$ 就返回。如果在搜索的途中发现了答案就可以回溯，同时在回溯的过程中可以记录路径。如果没有发现答案，就返回到函数入口，增加设定深度，继续搜索。

## 注意事项

在大多数的题目中，广度优先搜索还是比较方便的，而且容易判重。当发现广度优先搜索在空间上不够优秀，而且要找最优解的问题时，就应该考虑迭代加深。

## 习题

> [!NOTE] **[AcWing 170. 加成序列](https://www.acwing.com/problem/content/172/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>

const int N = 110;

int n;
int path[N];

bool dfs(int u, int k) {
    if (u == k) return path[u - 1] == n;

    bool st[N] = {0};
    for (int i = u - 1; i >= 0; i -- )
        for (int j = i; j >= 0; j -- ) {
            int s = path[i] + path[j];
            // 不合法 || 达不到目标 || 本层已经dfs过 ： continue
            if (s > n || s <= path[u - 1] || st[s]) continue;
            st[s] = true;
            path[u] = s;
            if (dfs(u + 1, k)) return true;
        }

    return false;
}

int main() {
    path[0] = 1;
    while (cin >> n, n) {
        int k = 1;
        while (!dfs(1, k)) k ++ ;

        for (int i = 0; i < k; i ++ ) cout << path[i] << ' ';
        cout << endl;
    }

    return 0;
}
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

* * *