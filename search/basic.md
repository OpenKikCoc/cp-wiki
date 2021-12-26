## 习题

> [!WARNING] **一般指 dfs**

### 子集


### 排列


### dfs 分组

> [!NOTE] **[AcWing 1118. 分成互质组](https://www.acwing.com/problem/content/1120/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **标准dfs分组**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, res, len;
vector<int> c;
vector<vector<int>> g;

int gcd(int a, int b) {
    if (b) return gcd(b, a % b);
    return a;
}

bool check(int group, int x) {
    // for(int i = 0; i < g[group].size(); ++i)
    for (auto& v : g[group])
        if (gcd(v, x) > 1) return false;
    return true;
}

void dfs(int u) {
    if (u == n) {
        res = min(res, len);
        return;
    }
    // 使用已有的组
    for (int i = 0; i < len; ++i)
        if (check(i, c[u])) {
            g[i].push_back(c[u]);
            dfs(u + 1);
            g[i].pop_back();
        }
    // 单独放一组　开辟新的组
    g[len++].push_back(c[u]);
    dfs(u + 1);
    g[--len].pop_back();
}

int main() {
    cin >> n;
    c = vector<int>(n);
    g = vector<vector<int>>(n);  // 最多用ｎ组
    for (int i = 0; i < n; ++i) cin >> c[i];
    // sort(c.begin(), c.end());

    // res = inf, len = 0;
    res = 10;
    dfs(0);

    cout << res << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *