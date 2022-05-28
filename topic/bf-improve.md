## 习题

> [!NOTE] **[Codeforces The Brand New Function](http://codeforces.com/problemset/problem/243/A)**
> 
> 题意: TODO
> 
> 定义函数 $f(l,r)$ $(1 \le l,r \le n)$ ，表示序列的子串 $[l,r]$ 各项的 `或` 和: 
> 
> $f(l,r)=a_l|a_{l+1}|⋯|a_r$
> 
> 求整个数组有多少个不同的 `或` 和

> [!TIP] **思路**
> 
> **非常经典的暴力优化**
> 
> 需要严格数学推导

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. The Brand New Function
// Contest: Codeforces - Codeforces Round #150 (Div. 1)
// URL: https://codeforces.com/problemset/problem/243/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10;

int n;
int a[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;

    unordered_set<int> S;
    for (int i = 1, x; i <= n; ++i) {
        cin >> a[i];
        S.insert(a[i]);
        // TRICK: 经过严谨数学证明的剪枝与实现方式
        for (int j = i - 1; j; --j) {
            // ATTENTION: trick
            // if-condition 满足时必然此前已计算过同样值的了，直接break
            if ((a[j] | a[i]) == a[j])
                break;
            // 为什么可以直接或 ？ 更改后是否影响正确性 ？
            // 1. 区间具有包含性质
            // 2. 由 1 后续使用的必然包含上一次使用的，正确性不变
            a[j] |= a[i];
            S.insert(a[j]);
        }
    }

    cout << S.size() << endl;

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

> [!NOTE] **[LeetCode 2257. 统计网格图中没有被保卫的格子数](https://leetcode.cn/problems/count-unguarded-cells-in-the-grid/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂度可以接受直接把坐标存下来再二分，复杂度 $O(nmlog)$
> 
> 实际上，我们可以直接用 bit 位表示某个位置的 `上下左右` 是否有守卫，进而把时间复杂度优化到 $O(nm)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 二分**

```cpp
class Solution {
public:
    // m * n <= 1e5
    using PII = pair<int, int>;
    
    vector<vector<bool>> st;
    
    int countUnguarded(int m, int n, vector<vector<int>>& guards, vector<vector<int>>& walls) {
        this->st = vector<vector<bool>>(m, vector<bool>(n));
        vector<vector<PII>> r(m), c(n);
        for (int i = 0; i < m; ++ i )
            r[i].push_back({-1, 0}), r[i].push_back({n, 0});
        for (int i = 0; i < n; ++ i )
            c[i].push_back({-1, 0}), c[i].push_back({m, 0});
        for (auto & g : guards) {
            int x = g[0], y = g[1];
            st[x][y] = true;
            r[x].push_back({y, 1});
            c[y].push_back({x, 1});
        }
        for (auto & w : walls) {
            int x= w[0], y = w[1];
            st[x][y] = true;
            r[x].push_back({y, -1});
            c[y].push_back({x, -1});
        }
        for (int i = 0; i < m; ++ i )
            sort(r[i].begin(), r[i].end());
        for (int i = 0; i < n; ++ i )
            sort(c[i].begin(), c[i].end());
        
        int res = 0;
        for (int i = 0; i < m; ++ i )
            for (int j = 0; j < n; ++ j ) {
                if (st[i][j])
                    continue;
                // cout << " I = " << i << " j = " << j << endl;
                
                {
                    auto it = lower_bound(r[i].begin(), r[i].end(), PII{j, 0});
                    if ((*it).second == 1) {
                        continue;
                    }
                    it -- ;
                    if ((*it).second == 1) {
                        continue;
                    }
                }
                {
                    auto it = lower_bound(c[j].begin(), c[j].end(), PII{i, 0});
                    if ((*it).second == 1) {
                        continue;
                    }
                    it -- ;
                    if ((*it).second == 1) {
                        continue;
                    }
                }
                res ++ ;
            }
        return res;
    }
};
```

##### **C++ 优化**

```cpp
class Solution {
public:
    // m * n <= 1e5
    using PII = pair<int, int>;
    
    vector<vector<int>> st;
    
    // 入参把 n m swap 了
    int countUnguarded(int n, int m, vector<vector<int>>& guards, vector<vector<int>>& walls) {
        st = vector<vector<int>>(n, vector<int>(m));
        for (auto & g : guards)
            st[g[0]][g[1]] = 15;
        for (auto & w : walls)
            st[w[0]][w[1]] = 16;    // 只要后四bit位全0即可
        
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (st[i][j] < 16) {
                    if (i)
                        st[i][j] |= st[i - 1][j] & 1;
                    if (j)
                        st[i][j] |= st[i][j - 1] & 2;
                }
        for (int i = n - 1; i >= 0; -- i )
            for (int j = m - 1; j >= 0; -- j )
                if (st[i][j] < 16) {
                    if (i < n - 1)
                        st[i][j] |= st[i + 1][j] & 4;
                    if (j < m - 1)
                        st[i][j] |= st[i][j + 1] & 8;
                }
        
        int res = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                res += !st[i][j];
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