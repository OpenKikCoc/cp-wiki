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
>
> （类似01背包问题（选或不选），这题换成是了选+ 还是选-）
>
> **暴力DP**
>
> 1. 状态表示：f[i, j] 前i个数，总和为j的所有方案的集合；属性：数量
> 2. 状态转移：分为两个子集 1) a[i]取正；f[i - 1, j - a[i]]； 2）a[i]取负 :f[i - 1， j + a[i]]
> 3. 初始化，f[0, 0] = 1， 由于S 可能为负数，所以在坐标上做了一个映射（整体往后走offset = 1000）所以: f[0, offset] = 1
>
> **DP优化**
>
> 1. 假设所有符号为`+`的元素和为x，符号为`-`的元素和的绝对值是y。
> 2. 那么 S = x - y; 并且已知 x + y = sum ; ==> x = (S + sum) // 2
> 3. 上述数学计算后，也就是需要从 nums 中选出几个数，使得他们的和为  (S + sum) // 2
> 4. 上述优化后：可以转化为选出体积为 (S + sum) // 2的 01背包问题
> 5. 01背包可以压缩一维空间进行优化
>
> **记忆化搜索**
>
> Python通过装饰器实现记忆化



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

##### **Python-暴力dp**

```python
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



**Python-dp优化**

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        sumn = sum(nums)
        S = abs(S)
        if S > sumn or (S + sumn) % 2:
            return 0
        k = (S + sumn) // 2
        f = [0] * (k + 1)
        f[0] = 1

        for i in range(len(nums)):
            for j in range(k, nums[i] - 1, -1):
                f[j] = f[j] + f[j - nums[i]]
        return f[-1]
```



**Python-DFS**

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        import functools

        @functools.lru_cache(None)
        def dfs(i, S):
            if i == len(nums):
                # 如果此时S为0，说明找到了一种方案数
                if S == 0:
                    return 1
                else:return 0
            return dfs(i + 1, S - nums[i]) + dfs(i + 1, S + nums[i])

        return dfs(0, S)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 887. 鸡蛋掉落](https://leetcode.cn/problems/super-egg-drop/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **状态定义与转移**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e4 + 10, M = 110;

    // 扔了 i 次用了 j 个鸡蛋可以探测到的最大高度
    int f[N][M];

    int superEggDrop(int k, int n) {
        for (int i = 1; i <= n; ++ i ) {
            for (int j = 1; j <= k; ++ j )
                // - 没碎 f[i-1,j]
                // - 碎了 f[i-1,j-1]+1
                f[i][j] = f[i - 1][j - 1] + 1 + f[i - 1][j];
            if (f[i][k] >= n)
                return i;
        }
        return -1;
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

> [!NOTE] **[Codeforces Antimatter](http://codeforces.com/problemset/problem/383/D)**
> 
> 题意: 
> 
> Luogu 题意有很大问题
> 
> 每个数可取正负数，问对于每一个前缀有多少种方式使得其和为 $0$ ，求出方案总和

> [!TIP] **思路**
> 
> 记 $f[i][j]$ 代表前 $i$ 个数，前缀和为 $j$ 的方案数
> 
> 随后递推即可，需要注意加偏移量（前缀和有负数）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Antimatter
// Contest: Codeforces - Codeforces Round #225 (Div. 1)
// URL: https://codeforces.com/problemset/problem/383/D
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e3 + 10, OFFSET = 1e4, M = 2e4 + 10, MOD = 1e9 + 7;

int n;
int a[N];
int f[N][M];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    memset(f, 0, sizeof f);
    for (int i = 1; i <= n; ++i) {
        int t = a[i];
        f[i][t + OFFSET]++, f[i][-t + OFFSET]++;
        for (int j = 0; j < M; ++j) {
            if (j + t < M)
                f[i][j] = (f[i][j] + f[i - 1][j + t]) % MOD;  // -
            if (j - t >= 0)
                f[i][j] = (f[i][j] + f[i - 1][j - t]) % MOD;  // +
        }
    }
    int res = 0;
    for (int i = 1; i <= n; ++i)
        res = (res + f[i][OFFSET]) % MOD;
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

> [!NOTE] **[Codeforces C. k-Tree](https://codeforces.com/problemset/problem/431/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp 好题
> 
> 状态定义和转移

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. k-Tree
// Contest: Codeforces - Codeforces Round #247 (Div. 2)
// URL: https://codeforces.com/problemset/problem/431/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 题意分析和状态定义
// dp 定义和转移
const int N = 110, M = 2, MOD = 1e9 + 7;

int f[N][M];

void mod_add(int& a, int b) { a = (a + b) % MOD; }

int main() {
    int n, k, d;
    cin >> n >> k >> d;

    f[0][0] = 1, f[0][1] = 0;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= k && j <= i; ++j)
            if (j < d) {
                mod_add(f[i][0], f[i - j][0]);
                mod_add(f[i][1], f[i - j][1]);
            } else {
                mod_add(f[i][1], f[i - j][0]);
                mod_add(f[i][1], f[i - j][1]);
            }
    }

    cout << f[n][1] << endl;

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

> [!NOTE] **[Codeforces C. Barcode](http://codeforces.com/problemset/problem/225/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 这题竟然是dp？
> 
> 状态定义 转移 前缀和技巧 **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Barcode
// Contest: Codeforces - Codeforces Round #139 (Div. 2)
// URL: http://codeforces.com/problemset/problem/225/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// 给出一个矩阵，包含两种颜色，问最少修改多少个点才能让每一列的颜色一致
// 且颜色一致的连续的列的长度在x和y之间。
//
// ATTENTION
// 可以将 m 列分成一些长度为 n 宽度为 [x, y] 的矩形
// 相邻矩形颜色不同 矩形内部颜色相同
//   dp[i][j] 表示考虑到第i位 本位为j
// 从下一位开始颜色不同且前面均合法的最小修改数

const int N = 1010;

int n, m, x, y;
int s[N], dp[N][2];

int main() {
    cin >> n >> m >> x >> y;

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            char c = 0;
            while (c != '#' && c != '.')
                c = getchar();
            if (c == '#')
                s[j]++;
        }

    for (int i = 1; i <= m; i++)
        s[i] += s[i - 1];

    memset(dp, 0x3f, sizeof(dp));
    dp[0][0] = 0;
    dp[0][1] = 0;
    for (int i = 0; i < m; i++)
        for (int j = x; j <= y; j++) {
            if (i + j > m)
                break;

            dp[i + j][0] = min(dp[i + j][0], dp[i][1] + s[i + j] - s[i]);
            dp[i + j][1] =
                min(dp[i + j][1], dp[i][0] + n * j - s[i + j] + s[i]);
        }

    cout << min(dp[m][0], dp[m][1]);

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

> [!NOTE] **[LeetCode 920. 播放列表的数量](https://leetcode.cn/problems/number-of-music-playlists/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典问题
> 
> 状态定义与转移

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 一个线段每个点可以随意填，但是相同数值必须至少间隔 k ===> 经典问题
    using LL = long long;
    const static int N = 210, MOD = 1e9 + 7;

    // 前 i 首歌，共 j 种不同的所有合法方案的集合
    LL f[N][N];

    int numMusicPlaylists(int n, int goal, int k) {
        f[0][0] = 1;
        for (int i = 1; i <= goal; ++ i )
            for (int j = 1; j <= n && j <= i; ++ j )
                // - 加个新歌 则前面 f[i-1][j-1] * 新歌种类数 (n-j+1)
                // - 加个老歌 则可以播放 【j-k】 中的某一首
                f[i][j] = (f[i - 1][j - 1] * (n - j + 1)
                        + f[i - 1][j] * max(j - k, 0)) % MOD;
        return f[goal][n];
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

> [!NOTE] **[LeetCode 1866. 恰有 K 根木棍可以看到的排列数目](https://leetcode-cn.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典题目 dp时考虑当前枚举的是所有当中最小的即可
> 
> 和另一题假定枚举的是当前 `最大/最高` 的类似
> 
> > 另一种思路是 园排列
> > 
> > 本质是第一类斯特林数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    const static int N = 1010;
    LL f[N][N];    // 用了高度1-i 左侧可以看到j个 最终 f[n][k]
    
    int rearrangeSticks(int n, int k) {
        f[1][1] = 1;
        for (int i = 2; i <= n; ++ i )
            for (int j = 1; j <= i; ++ j )
                f[i][j] = (f[i - 1][j - 1] + f[i - 1][j] * (i - 1) % MOD) % MOD;
        
        return f[n][k];
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

> [!NOTE] **[LeetCode 903. DI 序列的有效排列](https://leetcode.cn/problems/valid-permutations-for-di-sequence/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 类似 [木棍 / 整数拆分] 状态定义与转移非常难想
> 
> **重点在于集合映射**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 210, MOD = 1e9 + 7;

    int f[N][N];    // 0~i 的一个满足要求的排列 且最后一位为 j 的所有排列的集合
    // 状态转移非常有跳跃性 其变换思想类似于 【左边能看到的木棍的数量】

    int numPermsDISequence(string s) {
        int n = s.size();
        memset(f, 0, sizeof f);
        f[0][0] = 1;
        for (int i = 1; i <= n; ++ i )
            if (s[i - 1] == 'D')
                for (int j = i - 1; j >= 0; -- j )
                    // 类似完全背包的优化 直接使用 f[i][j + 1]
                    f[i][j] = (f[i - 1][j] + f[i][j + 1]) % MOD;
            else
                for (int j = 1; j <= i; ++ j )
                    f[i][j] = (f[i - 1][j - 1] + f[i][j - 1]) % MOD;
        
        int res = 0;
        for (int i = 0; i <= n; ++ i )
            res = (res + f[n][i]) % MOD;
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
