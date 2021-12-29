## 习题

> [!NOTE] **[AcWing 1369. 牛之关系谱](https://www.acwing.com/problem/content/1371/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 所有 i 个节点，高度**小于等于** j 的所有方案 这样好求一些

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 210, M = 110, MOD = 9901;

int n, m;
int f[N][N];
// 所有 i 个节点，高度小于等于 j 的所有方案 这样好求一些

int main() {
    cin >> n >> m;
    for (int i = 1; i <= m; ++ i ) f[1][i] = 1;
    for (int i = 2; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            for (int k = 1; k <= i - 2; ++ k )
                (f[i][j] += f[k][j - 1] * f[i - 1 - k][j - 1]) %= MOD;
    cout << (f[n][m] - f[n][m - 1] + MOD) % MOD << endl;
    
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

> [!NOTE] **[AcWing 1380. 邮票](https://www.acwing.com/problem/content/1382/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 状态定义 转化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 2000010;

int n, m = N - 1, k;
int f[N];   // 【重要 状态定义】 求某个面值 最少需要多少个物品

int main() {
    cin >> k >> n;
    memset(f, 0x3f, sizeof f);
    
    f[0] = 0;
    for (int i = 0; i < n; ++ i ) {
        int v;
        cin >> v;
        for (int j = v; j <= m; ++ j )
            f[j] = min(f[j], f[j - v] + 1);
    }
    
    int x = 0;
    while (f[x] <= k) ++ x ;
    cout << x - 1 << endl;
    
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

> [!NOTE] **[AcWing 1385. 购物优惠](https://www.acwing.com/problem/content/1387/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 转化 当作多维背包

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 6;

int n, m, p, b;

int cnt;
int f[N][N][N][N][N];
unordered_map<int, int> ids;

int get(int x) {
    if (ids.count(x) == 0) ids[x] = cnt ++ ;
    return ids[x];
}

// 5维费用的完全背包问题
void update(int v[], int p) {
    for (int a = v[0]; a < 6; ++ a )
        for (int b = v[1]; b < 6; ++ b )
            for (int c = v[2]; c < 6; ++ c )
                for (int d = v[3]; d < 6; ++ d )
                    for (int e = v[4]; e < 6; ++ e )
                        f[a][b][c][d][e] = min(f[a][b][c][d][e],
                            f[a - v[0]][b - v[1]][c - v[2]][d - v[3]][e - v[4]] + p);
}

int main() {
    cin >> n;
    memset(f, 0x3f, sizeof f);
    f[0][0][0][0][0] = 0;
    while (n -- ) {
        int v[5] = {0};
        cin >> m;
        while (m -- ) {
            int c, k;
            cin >> c >> k;
            v[get(c)] += k;
        }
        
        cin >> p;
        update(v, p);
    }
    
    cin >> b;
    int m[5] = {0};
    while (b -- ) {
        int c, k, p;
        cin >> c >> k >> p;
        m[get(c)] = k;
        int v[5] = {0};
        v[get(c)] = 1;
        update(v, p);
    }
    
    cout << f[m[0]][m[1]][m[2]][m[3]][m[4]] << endl;
    
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

> [!NOTE] **[AcWing 1391. 喧嚣摇滚乐队](https://www.acwing.com/problem/content/1393/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 类似于背包问题的dp 但略有不同
> 
> 需要按顺序挑
> 
> f[i][j][k] 所有从1~i中挑 且正在挑第j个唱片 且总时长不超过k的所有方案的集合
> 
> 爆搜也可以 正解是dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 25;

int n, t, m;
int f[N][N][N];

int main() {
    cin >> n >> t >> m;
    for (int i = 1; i <= n; ++ i ) {
        int v;
        cin >> v;
        for (int j = 1; j <= m; ++ j )
            for (int k = 0; k <= t; ++ k ) {
                f[i][j][k] = f[i - 1][j][k];
                if (k >= v) {
                    // 放入之前的唱片
                    f[i][j][k] = max(f[i][j][k], f[i - 1][j][k - v] + 1);
                    // 新开一个唱片
                    f[i][j][k] = max(f[i][j][k], f[i - 1][j - 1][t] + 1);   // attention t
                }
            }
    }
    cout << f[n][m][t] << endl;
    
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

> [!NOTE] **[AcWing 1403. 音乐主题](https://www.acwing.com/problem/content/1405/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp **状态定义**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// BIT 但太复杂 数据范围可以考虑直接dp
#include <bits/stdc++.h>
using namespace std;

const int N = 5010;

int n;
int w[N], f[N][N];
// f[i][j]开始的可匹配的最大长度

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i ) cin >> w[i];
    
    int res = 0;
    for (int i = n; i; -- i )
        for (int j = n; j > i; -- j ) {
            if (j == n || w[j] - w[i] != w[j + 1] - w[i + 1])
                f[i][j] = 1;
            else f[i][j] = f[i + 1][j + 1] + 1;
            res = max(res, min(f[i][j], j - i));  // 去除重叠部分
        }
    if (res < 5) res = 0;
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


> [!NOTE] **[AcWing 1405. 牛奶量取](https://www.acwing.com/problem/content/1407/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 挑出若干【种】数（而不是个数）可以凑成某个数值
> 
> 包含
> 
> 1. dp推导
> 
> 2. 优化
> 
> 3. 方案 [不唯一 所以需要爆搜]

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 110, M = 20010, INF = 0x3f3f3f3f;

int n, m;
int v[N], f[N][M];
vector<int> ans, path;

void dfs(int i, int j) {
    if (!i) {
        if (ans.empty() || ans > path) ans = path;
        return ;
    }
    if (f[i][j] == f[i - 1][j]) dfs(i - 1, j);
    
    path.push_back(v[i]);
    for (int k = j - v[i]; k >= 0; k -= v[i])
        if (f[i][j] == f[i - 1][k] + 1)
            dfs(i - 1, k);
    path.pop_back();
}

int main() {
    cin >> m >> n;
    for (int i = 1; i <= n; ++ i ) cin >> v[i];
    sort(v + 1, v + n + 1, greater<int>()); // 翻转排序 方便爆搜方案
    memset(f, 0x3f, sizeof f);
    f[0][0] = 0;
    for (int i = 1; i <= n; ++ i )
        // 枚举 v[i] 组
        for (int j = 0; j < v[i]; ++ j ) {
            int minf = INF;
            for (int k = j; k <= m; k += v[i]) {
                f[i][k] = min(f[i - 1][k], minf + 1);
                minf = min(minf, f[i - 1][k]);
            }
        }
    cout << f[n][m] << ' ';
    dfs(n, m);
    for (auto x : ans) cout << x << ' ';
    cout << endl;
    
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

> [!NOTE] **[AcWing 1408. 加拿大之旅](https://www.acwing.com/problem/content/1410/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 若不限定从左至右 则费用流
> 
> 本题dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 110;

int n, m;
unordered_map<string, int> id;
// f[i][j] 所有 1~i 1~j 除起点外只用一次的所有方案
int f[N][N];
bool g[N][N];

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) {
        string name;
        cin >> name;
        id[name] = i;
    }
    while (m -- ) {
        string a, b;
        cin >> a >> b;
        g[id[a]][id[b]] = g[id[b]][id[a]] = true;
    }
    
    // dp
    // f[i][j] = f[j][i] 故只算 i < j 的情况即可
    memset(f, -1, sizeof f);
    f[1][1] = 1;
    for (int i = 1; i <= n; ++ i )
        for (int j = i + 1; j <= n; ++ j )
            // 枚举倒数第2个点的位置
            for (int k = 1; k < j; ++ k )
                // 需 k->j 有边
                if (f[i][k] > 0 && g[k][j])
                    f[i][j] = f[j][i] = max(f[i][j], f[i][k] + 1);
    
    int res = 1;
    for (int i = 1; i <= n; ++ i )
        if (f[i][n] > 0 && g[i][n])
            res = max(res, f[i][n]);
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

> [!NOTE] **[Luogu 大师](https://www.luogu.com.cn/problem/P4933)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 求形成等差序列的子序列方案数
> 
> 状态定义和转移 实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e3 + 10, M = 4e4 + 10, MOD = 998244353;

int n, res;
int h[N];
int f1[N][M];
int f2[N][M];

// O(n*m)
// f1[i][j] 表示以i结尾公差为j的等差数列的数量
// 删去

// O(n^2)
// f2[i][j] 同样表示以i结尾公差为j的等差数列的数量
void work2() {
    int offset = 2e4;
    for (int i = 1; i <= n; ++ i )
        //    枚举上一个数是哪个
        // 以i结尾且上一个数是j的公差为d的等差数列数量
        // 是以j结尾公差为d的等差数列数+1
        for (int j = i - 1; j; -- j ) {
            int d = h[i] - h[j] + offset;
            int add = f2[j][d] + 1;
            
            f2[i][d] = (f2[i][d] + add) % MOD;
            res = (res + add) % MOD;
        }
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> h[i];
    
    work2();
    
    // 加上每个数字自己单独作为等差数列
    cout << res + n << endl;
    
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

> [!NOTE] **[LeetCode 494. 目标和](https://leetcode-cn.com/problems/target-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 搜索 2^20 会超时 考虑dp
> 
> dp思路的思维 => 转化为取一些数其和达到 S / 2

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int n = nums.size(); long sum = S;
        // 求和 优化 转化为取一些数达到 sum / 2 的值
        for (auto& v : nums) sum += v;
        if (sum & 1 || sum < 2 * S) return 0;
        sum /= 2;
        vector<int> f(sum + 1);
        f[0] = 1;
        for (int i = 1; i <= n; ++ i )
            for (int j = sum; j >= nums[i - 1]; -- j )
                f[j] = f[j] + f[j - nums[i - 1]];
        return f[sum];
    }
};


// yxc 较慢
class Solution {
public:
    int findTargetSumWays(vector<int>& a, int S) {
        if (S < -1000 || S > 1000) return 0;
        int n = a.size(), Offset = 1000;
        vector<vector<int>> f(n + 1, vector<int>(2001));
        f[0][Offset] = 1;
        for (int i = 1; i <= n; i ++ )
            for (int j = -1000; j <= 1000; j ++ ) {
                if (j - a[i - 1] >= -1000)
                    f[i][j + Offset] += f[i - 1][j - a[i - 1] + Offset];
                if (j + a[i - 1] <= 1000)
                    f[i][j + Offset] += f[i - 1][j + a[i - 1] + Offset];
            }
        return f[n][S + Offset];
    }
};

// wzc
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int n = nums.size();
        int sum = accumulate(nums.begin(), nums.end(), 0);

        if (!(-sum <= S && S <= sum))
            return 0;

        vector<vector<int>> f(n + 1, vector<int>(2 * sum + 1, 0));
        f[0][0 + sum] = 1;

        for (int i = 1; i <= n; i++)
            for (int j = -sum; j <= sum; j++) {
                if (-sum <= j - nums[i - 1])
                    f[i][j + sum] += f[i - 1][j - nums[i - 1] + sum];
                if (j + nums[i - 1] <= sum)
                    f[i][j + sum] += f[i - 1][j + nums[i - 1] + sum];
            }

        return f[n][S + sum];
    }
};
```

##### **Python**

```python
#dp : 0/1背包问题：选 就是 正数；不选 就是负数
# f[i][j] 前i个数 总和为j的所有方案的集合；属性：数量
# 状态计算：分为两个子集：1) a[i]取正；f[i - 1][j - a[i]]
#                     2）a[i]取负 :f[i - 1][j + a[i]]
# f[0][0] = 1

class Solution:
    def findTargetSumWays(self, a: List[int], S: int) -> int:
        if S > 1000 or S < -1000:return 0
        n = len(a)
        offset = 1000 
        f = [[0] * 2001 for _ in range(n + 1)]
        f[0][offset] = 1  

        for i in range(1, n + 1):
            for j in range(-1000, 1001):
                if j - a[i - 1] >= -1000:
                    f[i][j + offset] += f[i - 1][j - a[i - 1] + offset]
                if j + a[i - 1] <= 1000:
                    f[i][j + offset] += f[i - 1][j + a[i - 1] + offset]

        return f[n][S + offset]
```

<!-- tabs:end -->
</details>

<br>

* * *