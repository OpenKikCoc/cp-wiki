> [!NOTE] **经典递推优化**
>
> **经典线性优化 状压递推复杂度 $O(n^2)$ -> $O(n)$**
> 
> 例: **[LeetCode 1723. 完成所有工作的最短时间](https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/)**
>
> ```cpp
> // 写法1
> // 【学习记忆 这种写法更好】
> for (int i = 1; i < (1 << n); ++ i ) {
>     // o = 最右侧0的数量
>     int o = __builtin_ctz(i & (-i));
>     tot[i] = tot[i ^ (i & (-i))] + jobs[o];
> }
>         
> // 写法2
> for (int i = 1; i < (1 << n); ++ i )
>     for (int j = 0; j < n; ++ j )
>         if ((i & (1 << j))) {
>             int left = i - (1 << j);
>             tot[i] = tot[left] + jobs[j];
>             break;
>         }
> ```
>

> [!NOTE] **枚举子集**
> 
> **部分情况下**：有时我们只需枚举子集的一半
> 
> 因为另一半可以通过 `^` 得到，枚举全集相对于有一半的重复计算
> 
> ```cpp
> for (int i = 0; i < 1 << n; ++ i ) {
>     // 枚举i的一半 240ms
>     int t = i & (i - 1);
>     for (int j = t; j; j = (j - 1) & i)
>         res = max(res, get(j) * get(i ^ j));
> 
>     // 枚举全局  时间多一点点 差别不大
>     // for (int j = i; j; j = (j - 1) & i)
>     //    res = max(res, get(j) * get(i ^ j));
> }
> ```

> [!NOTE] **子集递推计数 去重**
>
> **枚举子集时依靠某位bit位将所有子集分为两类，从而实现去重计数**
>
> ```cpp
> for (int i = 1; i < 1 << N; ++ i ) {
>     int f = i & (-i);  // bit划分
>     for (int j = i; j; j = (j - 1) & i)
>         // 去重
>         if (j & f)
>             sum[i] = (sum[i] + (LL)cnt[j] * sum[i ^ j]) % MOD;
> }
> ```

> [!NOTE] **$Gosper's Hack$ 算法**
>
> 可以在 $O(1)$ 的时间复杂度内找到下一个大小为固定值的集合
>
> https://zhuanlan.zhihu.com/p/360512296
>
> ```cpp
> void GospersHack(int k, int n) {
>     int cur = (1 << k) - 1;
>     int limit = 1 << n;
>     while (cur < limit) {
>         // do something
> 
>         // algorithm
>         int lb = cur & -cur;
>         int r = cur + lb;
>         cur = ((r ^ cur) >> __builtin_ctz(lb) + 2) | r;
>         // 或: cur = (((r ^ cur) >> 2) / lb) | r;
>     }
> }
> ```


## 例题

> [!NOTE] **[「SCOI2005」互不侵犯](https://loj.ac/problem/2153)**
> 
> 在 $N\times N$ 的棋盘里面放 $K$ 个国王，使他们互不攻击，共有多少种摆放方案。国王能攻击到它上下左右，以及左上左下右上右下八个方向上附近的各一个格子，共 $8$ 个格子。

> [!TIP] 思路
> 
> 我们用 $f(i,j,l)$ 表示只考虑前 $i$ 行，第 $i$ 行按照编号为 $j$ 的状态放置国王，且已经放置 $l$ 个国王时的方案数。
> 
> 对于编号为 $j$ 的状态，我们用二进制整数 $sit(j)$ 表示国王的放置情况，$sit(j)$ 的某个二进制位为 $0$ 表示对应位置不放国王，为 $1$ 表示在对应位置上放置国王；
> 
> 用 $sta(j)$ 表示该状态的国王个数，即二进制数 $sit(j)$ 中 $1$ 的个数。例如，如下图所示的状态可用二进制数 $100101$ 来表示（棋盘左边对应二进制低位），则有 $sit(j)=100101_{(2)}=37, sta(j)=3$。
> 
> ![](./images/SCOI2005-互不侵犯.png)
> 
> 我们需要在刚开始的时候枚举出所有的合法状态（即排除同一行内两个国王相邻的不合法情况），并计算这些状态的 $sit(j)$ 和 $sta(j)$。
> 
> 设上一行的状态编号为 $x$，在保证当前行和上一行不冲突的前提下，枚举所有可能的 $x$ 进行转移，转移方程：
> 
> $$
> f(i,j,l) = \sum f(i-1,x,l-sta(j))
> $$

TODO@binacs

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

## 习题

[NOI2001 炮兵阵地](https://loj.ac/problem/10173)

[「USACO06NOV」玉米田 Corn Fields](https://www.luogu.com.cn/problem/P1879)

[九省联考 2018 一双木棋](https://loj.ac/problem/2471)

[UVA10817 校长的烦恼 Headmaster's Headache](https://www.luogu.com.cn/problem/UVA10817)

[UVA1252 20 个问题 Twenty Questions](https://www.luogu.com.cn/problem/UVA1252)

### 状压枚举

> [!NOTE] **[AcWing 1362. 健康的荷斯坦奶牛](https://www.acwing.com/problem/content/1364/)**
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

const int N = 15, M = 25;

int m, n;
int need[M], s[N][M], sum[M];

int main() {
    cin >> m;
    for (int i = 0; i < m; ++ i ) cin >> need[i];
    
    cin >> n;
    for (int i = 0; i < n; ++ i )
        for (int j = 0; j < m; ++ j )
            cin >> s[i][j];
    
    vector<int> res;
    // 2^15 = 30000
    for (int i = 0; i < 1 << n; ++ i ) {
        vector<int> t;
        memset(sum, 0, sizeof sum);
        for (int j = 0; j < n; ++ j )
            if (i >> j & 1) {
                t.push_back(j);
                for (int k = 0; k < m; ++ k )
                    sum[k] += s[j][k];
            }
        
        bool flag = true;
        for (int j = 0; j < m; ++ j )
            if (sum[j] < need[j]) {
                flag = false;
                break;
            }
        if (flag) {
            if (res.empty() || res.size() > t.size() || res.size() == t.size() && res > t)
                res = t;
        }
    }
    cout << res.size() << ' ';
    for (auto x : res) cout << x + 1 << ' ';
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

> [!NOTE] **[LeetCode 1255. 得分最高的单词集合](https://leetcode.cn/problems/maximum-score-words-formed-by-letters/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 状压dp
> 
> 一开始在想线性dp 这样写的话每次统计字母容量较麻烦
>
> 考虑单词数量不多 直接状压遍历所有选择状态

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxScoreWords(vector<string>& words, vector<char>& letters,
                      vector<int>& score) {
        int n = words.size();
        vector<int> cnt(26);
        for (auto& c : letters) ++cnt[c - 'a'];
        int lim = 1 << n;
        int res = 0;
        for (int st = 0; st < lim; ++st) {
            vector<int> ct(26);
            for (int i = 0; i < n; ++i)
                if (st & (1 << i))
                    for (auto& c : words[i]) ++ct[c - 'a'];
            bool f = true;
            int tot = 0;
            for (int i = 0; i < 26; ++i) {
                if (ct[i] > cnt[i]) {
                    f = false;
                    break;
                }
                tot += ct[i] * score[i];
            }
            if (f) res = max(res, tot);
        }
        return res;
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1601. 最多可达成的换楼请求数目](https://leetcode.cn/problems/maximum-number-of-achievable-transfer-requests/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数据范围可以暴力枚举
> 
> 使用状态压缩 遍历所有情况
> 
> 注意判断合法的方法仅仅只是 in==out 不需要考虑图以及流

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maximumRequests(int n, vector<vector<int>>& requests) {
        int sz = requests.size(), res = 0;
        for (int st = 0; st < 1 << sz; ++st) {
            int c = 0;
            vector<int> in(n), out(n);
            for (int i = 0; i < sz; ++i)
                if (st & (1 << i))
                    ++out[requests[i][0]], ++in[requests[i][1]], ++c;
            for (int i = 0; i < n; ++i)
                if (in[i] != out[i]) {
                    c = -1;
                    break;
                }
            res = max(res, c);
        }
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

> [!NOTE] **[LeetCode 1617. 统计子树中城市之间最大距离](https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然需要暴力遍历所有状态，对每个状态检查是否合法：
> 
> 子集所使用的边数 = 子集点数 - 1
> 
> 随后计算当前状态下的城市间最大距离：
> 
> 树 dp
> 
> floyd 预处理
> 
> 注意写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    vector<int> countSubgraphsForEachDiameter(int n,
                                              vector<vector<int>>& edges) {
        // 预处理所有点的距离
        vector<vector<int>> dis(n + 1, vector<int>(n + 1, inf));
        for (int i = 0; i <= n; ++i) dis[i][i] = 0;
        for (auto& e : edges) dis[e[0]][e[1]] = dis[e[1]][e[0]] = 1;
        for (int k = 1; k <= n; ++k)
            for (int i = 1; i <= n; ++i)
                for (int j = 1; j <= n; ++j)
                    dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);
        // 状压dp
        vector<int> res(n - 1, 0);
        for (int i = 0; i < (1 << n); ++i) {
            vector<int> now;
            vector<bool> vis(n + 1, 0);
            int cnt = 0;
            for (int j = 0; j < n; ++j)
                if (i & (1 << j)) {
                    now.push_back(j + 1);
                    vis[j + 1] = 1;
                    ++cnt;
                }
            // 计算用的边数 为了判断子集是否合法（是否是连通的子树）
            int ss = 0;
            for (auto& e : edges)
                if (vis[e[0]] && vis[e[1]]) ++ss;
            if (ss != cnt - 1) continue;
            // 计算该合法子树任意两点间距离最大值
            int cur = 0;
            for (int j = 0; j < now.size(); ++j)
                for (int k = j + 1; k < now.size(); ++k)
                    cur = max(cur, dis[now[j]][now[k]]);
            if (cur > 0 && cur < n) ++res[cur - 1];
        }
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

> [!NOTE] **[LeetCode 1681. 最小不兼容性](https://leetcode.cn/problems/minimum-incompatibility/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 核心在枚举子集的时间复杂度 3 ^ n 以及枚举枚举思路：
> 
> ```cpp
> for (int i = 1; i < 1 << n; ++ i )
>     for (int j = i; j; j = j - 1 & i)
>         if (g[j] != -1)
>             f[i] = min(f[i], f[i - j] + g[j]);
> ```

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int INF = 1e8;
    int minimumIncompatibility(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> f(1 << n, INF);     // f[i]表示选取下标组合是i的二进制表示的最小兼容性和
        vector<int> g(1 << n);          // g[i]表示一种合法组合的兼容性
        // 预处理 g  复杂度 2^16 * (n + nlogn) = 2^16 * 5n = 5 * 2^20 = 5e6
        int d[16];
        for (int i = 1; i < 1 << n; ++ i ) {
            g[i] = -1;
            if (__builtin_popcount(i) == n / k) {
                int cnt = 0;
                for (int j = 0; j < n; ++ j )
                    if (i >> j & 1)
                        d[cnt ++ ] = nums[j];
                sort(d, d + cnt);
                int flag = 1;
                for (int j = 1; j < cnt; ++ j )
                    // 存在相同的两个数则不合法
                    if (d[j] == d[j - 1]) {
                        flag = 0;
                        break;
                    }
                if (flag)
                    g[i] = d[cnt - 1] - d[0];
            }
        }

        // 枚举所有子集的时间是3^n次  通过二项式定理能够证明出来
        f[0] = 0;
        for (int i = 1; i < 1 << n; ++ i )
            for (int j = i; j; j = j - 1 & i)   // 枚举 i 的所有子集
                if (g[j] != -1)                 // i 的子集 j 是合法的【包含n/k个】 则转移
                    f[i] = min(f[i], f[i - j] + g[j]);
        int res = f[(1 << n) - 1];
        if (res == INF) res = -1;
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

> [!NOTE] **[LeetCode 2002. 两个回文子序列长度的最大乘积](https://leetcode.cn/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二进制枚举即可 ==> 枚举子集

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 13;
    
    string s;
    int n;
    int hash[1 << N];
    
    int get(int st) {
        if (hash[st] != -1)
            return hash[st];
        string t;
        for (int i = 0; i < n; ++ i )
            if (st >> i & 1)
                t.push_back(s[i]);
        int len = t.size();
        for (int i = 0, j = len - 1; i < j; ++ i , -- j )
            if (t[i] != t[j])
                return hash[st] = 0;
        return hash[st] = len;
    }
    
    int maxProduct(string s) {
        this->s = s;
        this->n = s.size();
        
        memset(hash, -1, sizeof hash);
        
        int res = 0;
        
        for (int i = 0; i < 1 << n; ++ i ) {
            // 枚举i的一半 240ms
            int t = i & (i - 1);
            for (int j = t; j; j = (j - 1) & i)
                res = max(res, get(j) * get(i ^ j));
            
            // 枚举全局  时间多一点点 差别不大
            // for (int j = i; j; j = (j - 1) & i)
            //    res = max(res, get(j) * get(i ^ j));
        }
        
        return res;
    }
};
```

##### **C++ 更优**

显然判断是否回文有较多的重复操作 可以把判断回文放到前面来

```cpp
class Solution {
public:
    int maxProduct(string s) {
        const int n = s.size();

        vector<int> st(1 << n);
        for (int mask = 0; mask < (1 << n); mask++) {
            string t;
            for (int i = 0; i < n; i++)
                if ((mask >> i) & 1)
                    t += s[i];

            bool ok = true;
            for (int i = 0, j = t.size() - 1; i < j; i++, j--)
                if (t[i] != t[j]) {
                    ok = false;
                    break;
                }

            st[mask] = ok ? t.size() : 0;
        }

        int ans = 0;
        for (int mask = 0; mask < (1 << n); mask++)
            for (int m1 = mask; m1 > 0; m1 = (m1 - 1) & mask) {
                int m2 = mask ^ m1;
                ans = max(ans, st[m1] * st[m2]);
            }

        return ans;
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

> [!NOTE] **[LeetCode 2305. 公平分发饼干](https://leetcode.cn/problems/fair-distribution-of-cookies/)**
> 
> 题意: 
> 
> 把所有饼干分组发放，求最大值最小是多少

> [!TIP] **思路**
> 
> 显然枚举当前处理第 $i$ 个人，当前分发组数状态为 $j$ 时的值
> 
> 枚举子集递推维护即可 + 滚动数组

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 9;
    
    int f[2][1 << N], c[1 << N];
    
    int distributeCookies(vector<int>& cs, int k) {
        memset(c, 0, sizeof c);
        int n = cs.size();
        for (int i = 0; i < 1 << n; ++ i )
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    c[i] += cs[j];
            
        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= k; ++ i )
            for (int j = 0; j < 1 << n; ++ j )
                for (int k = j; k; k = (k - 1) & j)
                    f[i & 1][j] = min(f[i & 1][j], max(f[(i - 1) & 1][j ^ k], c[k]));
        
        return f[k & 1][(1 << n) - 1];
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

> [!NOTE] **[LeetCode 943. 最短超级串](https://leetcode.cn/problems/find-the-shortest-superstring/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 类似哈密顿路径，**状压枚举 + 记录方案** 即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 13;

    int d[N][N];
    int f[1 << N][N], g[1 << N][N];

    int get_dis(string & a, string & b) {   // 记录 a 串末尾与 b 串开头的公共长度
        int m = min(a.size(), b.size());
        for (int i = m; i >= 1; -- i ) {
            bool flag = true;
            for (int j = a.size() - i, k = 0; k < i; ++ j , ++ k )
                if (a[j] != b[k]) {
                    flag = false;
                    break;
                }
            if (flag)
                return i;
        }
        return 0;
    }

    string shortestSuperstring(vector<string>& words) {
        memset(d, 0x3f, sizeof d);
        int n = words.size();
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < n; ++ j )
                if (i != j)
                    d[i][j] = get_dis(words[i], words[j]);
                else
                    d[i][j] = words[i].size();
        
        memset(f, 0x3f, sizeof f), memset(g, -1, sizeof g);
        for (int i = 0; i < n; ++ i )
            f[1 << i][i] = words[i].size();
        
        for (int i = 0; i < 1 << n; ++ i )
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    for (int k = 0; k < n; ++ k )
                        if (j != k && i >> k & 1) {
                            int t = f[i ^ (1 << j)][k] + (int)words[j].size() - d[k][j];
                            if (f[i][j] > t) {
                                f[i][j] = t;
                                g[i][j] = k;
                            }
                        }
        
        string res;
        {
            int p = -1, cap = (1 << n) - 1;
            for (int i = 0; i < n; ++ i )
                if (p == -1 || f[cap][p] > f[cap][i])
                    p = i;
            // cout << " minv = " << f[cap][p] << endl;
            
            vector<int> t;
            for (int i = cap, j = p; j != -1; ) {
                // cout << " i = " << i << " j = " << j << endl;
                t.push_back(j);
                int tmp = j;    // 这里需要一个临时变量
                j = g[i][tmp], i = i ^ (1 << tmp);
            }
            reverse(t.begin(), t.end());

            for (int i = 0; i < t.size() - 1; ++ i ) {
                int a = t[i], b = t[i + 1];
                res += words[a].substr(0, words[a].size() - d[a][b]);
            }
            res += words[t.back()];
        }
        // cout << " res.size() = " << res.size() << endl;
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

> [!NOTE] **[LeetCode 2397. 被列覆盖的最多行数](https://leetcode.cn/problems/maximum-rows-covered-by-columns/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 比较简单的状压枚举 DP
> 
> 重点在于学习 $Gosper's Hack$ 算法 (可以在 $O(1)$ 的时间复杂度内找到下一个大小为固定值的集合)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 显然状压
    const static int N = 13, M = 1 << 12;
    
    int g[N];
    int f[N][M];

    vector<int> GospersHack(int k, int n) {
        vector<int> xs;

        int cur = (1 << k) - 1;
        int limit = 1 << n;
        while (cur < limit) {
            // do something
            xs.push_back(cur);

            // algorithm
            int lb = cur & -cur;
            int r = cur + lb;
            cur = ((r ^ cur) >> __builtin_ctz(lb) + 2) | r;
            // 或: cur = (((r ^ cur) >> 2) / lb) | r;
        }
        return xs;
    }
    
    int maximumRows(vector<vector<int>>& matrix, int numSelect) {
        int n = matrix.size(), m = matrix[0].size();
        
        for (int i = 1; i <= n; ++ i ) {
            int st = 0;
            auto & line = matrix[i - 1];
            for (int j = 0; j < m; ++ j )
                if (line[j])
                    st += 1 << j;
            g[i] = st;
        }
        
        vector<int> xs;
        // for (int i = 0; i < 1 << m; ++ i )
        //     if (__builtin_popcount(i) == numSelect)
        //         xs.push_back(i);
        xs = GospersHack(numSelect, m);

        memset(f, 0, sizeof f);
        for (int i = 1; i <= n; ++ i ) {
            int st = g[i];
            for (auto j : xs)
                f[i][j] = f[i - 1][j] + ((st & j) == st ? 1 : 0);
        }
        int res = 0;
        for (auto i : xs)
            res = max(res, f[n][i]);
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

> [!NOTE] **[LeetCode 3283. 吃掉所有兵需要的最多移动次数](https://leetcode.cn/problems/maximum-number-of-moves-to-kill-all-pawns/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然可以状压枚举
> 
> 注意优雅的实现处理细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 看到 position.length <= 15 考虑状压枚举
    // f[i][j] 当前已经访问过的兵二进制状态为i 当前在第j个兵 的总距离
    // 其中 由i的bit数量可以推断是 alice/bob 操作，并影响其转移来源取 max/min
    //
    using PII = pair<int, int>;
    const static int N = 50, M = 16;    // M = 15+1 把起始点也作为一个兵
    
    int dx[8] = {-1, -2, -2, -1, 1, 2, 2, 1}, dy[8] = {-2, -1, 1, 2, 2, 1, -1, -2};
    int dis[M][N][N];   // 记录每个兵到任意位置到距离
    
    vector<vector<int>> ps;
    int n;
    
    void calc_dis() {
        memset(dis, -1, sizeof dis);
        
        auto bfs = [&](int s) {
            queue<PII> q;
            q.push({ps[s][0], ps[s][1]});
            dis[s][ps[s][0]][ps[s][1]] = 0;
            while (!q.empty()) {
                auto [x, y] = q.front();
                q.pop();
                for (int i = 0; i < 8; ++ i ) {
                    int nx = x + dx[i], ny = y + dy[i];
                    if (nx < 0 || nx >= 50 || ny < 0 || ny >= 50)
                        continue;
                    if (dis[s][nx][ny] != -1)
                        continue;
                    dis[s][nx][ny] = dis[s][x][y] + 1;
                    q.push({nx, ny});
                }
            }
        };
        for (int i = 0; i < n; ++ i )
            bfs(i);
    }
    
    int f[1 << M][M];   // 状态定义
    
    int dp(int st, int last) {
        if (st == (1 << n) - 1)
            return 0;
        
        if (f[st][last] != -1)
            return f[st][last];
        
        // 奇数alice先走,求最大(初始化最小)
        int bit_count = __builtin_popcount(st);
        int ret = (bit_count & 1) ? -1e9 : 1e9;
        
        // 枚举下一位往哪里去
        // ATTENTION: 如果要枚举上一位从哪里来 初始化状态会更为复杂
        for (int i = 0; i < n; ++ i ) {
            if (st >> i & 1)
                continue;
            int t = dp(st | 1 << i, i) + dis[last][ps[i][0]][ps[i][1]];
            ret = (bit_count & 1) ? max(ret, t) : min(ret, t);
        }
        return f[st][last] = ret;
    }
    
    int maxMoves(int kx, int ky, vector<vector<int>>& positions) {
        positions.push_back({kx, ky});  // ATTENTION: 加入起始点 简化后续流程
        this->ps = positions;
        this->n = ps.size();
        
        calc_dis();
        
        memset(f, -1, sizeof f);
        return dp(1 << (n - 1), n - 1);
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

### 递推计算

> [!NOTE] **[AcWing 291. 蒙德里安的梦想](https://www.acwing.com/problem/content/293/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 先求每个状态是否合法 再状态转移累加合法数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 12, M = 1 << N;
int st[M];
long long f[N][M];

int main() {
    int n, m;
    while (cin >> n >> m && (n || m)) {
        for (int i = 0; i < 1 << n; ++i) {
            int cnt = 0;  // 连续空格的数量
            st[i] = true;
            for (int j = 0; j < n; ++j)
                if (i >> j & 1) {
                    if (cnt & 1)
                        break;  // 为啥不break?==> st[i]=false 可以换成break
                    cnt = 0;
                } else
                    ++cnt;
            if (cnt & 1) st[i] = false;
        }
        memset(f, 0, sizeof f);
        f[0][0] = 1;
        for (int i = 1; i <= m; ++i)
            for (int j = 0; j < 1 << n; ++j)
                for (int k = 0; k < 1 << n; ++k)
                    if ((j & k) == 0 && st[j | k])
                        // j & k == 0 表示 i 列和 i - 1列同一行不同时捅出来
                        // st[j | k] == 1 表示 在 i 列状态 j， i - 1 列状态 k
                        // 的情况下是合法的.
                        f[i][j] += f[i - 1][k];
        cout << f[m][0] << endl;
    }
}
```

##### **Python**

```python
# 1. 核心：先放横着的，再放竖着的。
# 2. 总的方案数：如果只放横着的小方块的话，所有的合法的方案数有多少种。
# 3.如何判断当前方式是不是合法的？==> 当摆放完横的小方块后，所有剩余的位置能够填充满竖的小方块。（可以的话，就是合法的）
# 如何判断 是否能填满竖的小方块呢？ ==> 可以按列看，每一列内部所有连续的空着的小方块的数量需要是偶数个。

# 状态表示（化零为整）：f[i,j]表示已经将前i-1列摆好，且从第i-1列伸出到第i列的状态为j的所有方案数；
# 状态转移（化整为零）：把f[i,j]做分割。

# 化零为整(状态转移): 分割的时候，一般是在找最后一个不同点，以此作为分割。
# f[i,j]表示第i-1列伸到第i列的方案已经固定，那最后没有固定的就是从第i-2列伸到第i-1列的状态，以此来分割为若干种。可以分割成pow(2,n)种，每一行 都有两种选择-伸/不伸。===> f[i,j]：最多会被划分成为pow(2,n)的子集

# 每个子集都表示一个具体的状态，比如k(也是一个二进制数表示的)，比如00100（只有第三行是伸出来的）
# f[i,j]所有的集合的倒数第二步一定是属于这个子集之一，这种划分方案是一定是不重不漏；f[i,j]总共方案数，就是每一个子集的方案数之和。

# 那每个子集的方案数怎么求？
# 假设第i-2列伸到第i-1列的状态是k，第i-1列伸到第i列的状态是j，那方案 数是f[i-1,k]，但能不能转移过来是一个问题？什么情况下j和k可以拼在一起构成合法方案呢？
# 1）j和k不能在同一行都有1 ：需要满足j&k==0
# 2）对于i-1列来说，空的小方块的位置是固定的：空着的小方块可以被竖着的1*2小方块塞满，那就是所有空着的连续的位置的长度必须是偶数。

# 最后返回的方案数怎么表示呢？f[m][0]：m其实是m+1列（下标从0开始），指的是前m列已经伸好了，且从m列伸到m+1列的状态是0的所有方案。恰好就是摆满n*m的所有方案。


if __name__ == "__main__":
    N = 12
    M = 1 << N  # pow(2,n)
    f = [[0] * M for _ in range(N)]
    st = [False] * M  # 判断某个状态是否合法，也就是判断当前这一列所有空着的连续位置是否是偶数。

    n, m = map(int, input().split())
    # 首先预处理st数组，
    while n or m:
        # 清空还原初始值数组
        f = [[0] * M for _ in range(N)]
        st = [False] * M
        # 预处理:1)cnt记录连续0的个数
        # 2）对于i 循环遍历每次右移1位，判断当前数 是否为1:a.如果为1，那么就判断之前连续的0是否是偶数个，如果是偶数个，就直接将cnt置为0，如果是奇数个 说明当前状态i不满足要求，直接将st[i]只为Fasle；b.如果当前数为0，那么cnt++即可。f
        for i in range(1 << n):
            cnt = 0
            st[i] = True
            for j in range(n):
                if i >> j & 1:
                    if cnt & 1:
                        st[i] = False
                    cnt = 0
                else:
                    cnt += 1
            # 最后还要判断一下 最后一段如果也是奇数，那也是不合法的:这里容易忘    
            if cnt & 1: st[i] = False

        # base case很重要
        f[0][0] = 1
        for i in range(1, m + 1):
            for j in range(1 << n):
                for k in range(1 << n):
                    if j & k == 0 and st[j | k]:
                        f[i][j] += f[i - 1][k]

        print(f[m][0])
        n, m = map(int, input().split())

# 写代码的时候，加一下优化，不容易受时间限制卡住：预处理对于每个状态k而言，有哪些状态可以更新到j
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 91. 最短Hamilton路径](https://www.acwing.com/problem/content/93/)**
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

const int N = 21, M = 1 << N;
int d[N][N];
int st[M];
int f[M][N];

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) cin >> d[i][j];
    memset(f, 0x3f, sizeof f);
    f[1][0] = 0;
    for (int i = 0; i < 1 << n; ++i) {  // 走过的状态
        for (int j = 0; j < n; ++j)     // 枚举该状态的最后一个点
            if ((i >> j & 1) == 1)      // 状态包含该点
                for (int k = 0; k < n; ++k)  // 又哪个点转移至j点
                    if (i ^ 1 << j >> k & 1)
                        // 如果从当前状态经过点集 state 中，去掉点 j 后，state
                        // 仍然包含点 k，那么才能从点 k 转移到点 j。
                        if (f[i ^ 1 << j][k] + d[k][j] < f[i][j])
                            f[i][j] = f[i ^ 1 << j][k] + d[k][j];
    }
    cout << f[(1 << n) - 1][n - 1] << endl;
}
```

##### **Python**

```python
#如何走的过程不关心，只关心哪些点被用过，怎么走是最短的：1.哪些点被用过；2. 目前最后停在哪个点上
#用二进制用来表示要走的所有情况的路径，这里用state比作，比如 0,1,4 ==>  state=10011
#状态表示：f[i][j]: 所有从0走点j，走过的所有点的路径集合是i；属性：min（i：就是哪些点被用过了）
#状态转移： 枚举从哪个点转移到点j上来===> 找一个中间点k，将已经走过点的集合i中去掉j(表示j不在经过的点的集合中)，然后再加上从k到j的权值。
#f[state][j]=f[state_k][k]+weight[k][j], state_k= state除掉j之外的集合，state_k要包含k===> f[i][j]=min(f[i][j],f[i-(1<<j)][k]+w[k][j])

N = 22
M = 1 << N
f = [[float('inf')] * N for _ in range(M)]  #要求的是最小距离，所以初始化的时候 要初始化为max
w = []   # w表示的是无权图 

if __name__=='__main__':
    n = int(input())
    for i in range(n):
        w.append(list(map(int,input().split())))
    f[1][0] = 0  #因为零是起点，所以f[1][0]=0,第一个点是不需要任何费用的
    for i in range(1 << n):   #i表示所有的情况,一个方案集合
        for j in range(n):    #j表示走到哪一个点
            #判断状态是否是合法的：状态i里的第j位是否为1
            if i >> j & 1: 
                for k in range(n):  #k表示走到j这个点之前，以k为终点的最短距离
                    if i >> k & 1:  #判断状态是否合法：第j位是否为1
                    #也可以写成：if (i - (1 << j) >> k & 1）:  
                        f[i][j] = min(f[i][j], f[i-(1 << j)][k] + w[k][j])  #更新最短距离
    print(f[(1 << n)-1][n-1])  #表示把所有点都遍历过了，并且停在n-1点上。
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu 吃奶酪](https://www.luogu.com.cn/problem/P1433)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典状压

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 16, M = (1 << N) + 10;

int n;
double dis[N][N];
struct Point {
    double x, y;
} ps[N];

double f[N][M];

double get_dist(int i, int j) {
    double dx = ps[i].x - ps[j].x, dy = ps[i].y - ps[j].y;
    return sqrt(dx * dx + dy * dy);
}

int main() {
    cin >> n;
    // 0 能否直接作为集合表示中的点 ? 代价：数组空间扩大一倍
    
    ps[0].x = ps[0].y = 0;
    for (int i = 1; i <= n; ++ i )
        cin >> ps[i].x >> ps[i].y;
    
    for (int i = 0; i <= n; ++ i )
        for (int j = 0; j <= i; ++ j )
            dis[i][j] = dis[j][i] = get_dist(i, j);
    
    // double init INF
    memset(f, 127, sizeof f);
    for (int i = 1; i <= n; ++ i )
        f[i][1 << (i - 1)] = dis[0][i];

    for (int i = 1; i < 1 << n; ++ i )
        for (int j = 1; j <= n; ++ j )
            if (i >> (j - 1) & 1)
                // 当前位于第j个点 从k转移过来
                for (int k = 1; k <= n; ++ k )
                    if ((i >> (k - 1) & 1) && k != j)
                        f[j][i] = min(f[j][i], f[k][i ^ (1 << (j - 1))] + dis[k][j]);

    double res = 2e18;
    // end with i != 0
    for (int i = 1; i <= n; ++ i )
        res = min(res, f[i][(1 << n) - 1]);
    printf("%.2lf\n", res);
    
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

> [!NOTE] **[AcWing 1064. 小国王](https://www.acwing.com/problem/content/description/1066/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **记录写法**
> 
> 提前处理合法的所有状态和状态转移路径，定义三维状态。**四维状态转移写法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

typedef long long LL;

const int N = 12, M = 1 << 10, K = 110;

int n, m;
vector<int> state;
int cnt[M];
vector<int> head[M];
LL f[N][K][M];

// 同一行内的状态 是否合法
bool check(int state) {
    // O(1) 检查是否有相邻的两个 1
    // return !(x & x >> 1);
    for (int i = 0; i < n; i++)
        if ((state >> i & 1) && (state >> i + 1 & 1)) return false;
    return true;
}

int count(int state) {
    int res = 0;
    for (int i = 0; i < n; i++) res += state >> i & 1;
    // while (state) x = x & (x - 1), c ++ ;    // 非负数
    return res;
}

int main() {
    cin >> n >> m;

    for (int i = 0; i < 1 << n; i++)
        if (check(i)) {
            state.push_back(i);
            cnt[i] = count(i);
        }

    // 这里全部遍历 其实可以利用矩阵对称只计算一半
    for (int i = 0; i < state.size(); i++)
        for (int j = 0; j < state.size(); j++) {
            int a = state[i], b = state[j];
            if ((a & b) == 0 && check(a | b)) head[i].push_back(j);
        }

    // f[i][j][s] 为所有只摆在前i行，目前摆了j个国王，而且第i行的摆放状态为s
    // f[n + 1] 仅仅是为了好统计答案
    f[0][0][0] = 1;
    for (int i = 1; i <= n + 1; i++)
        for (int j = 0; j <= m; j++)
            for (int a = 0; a < state.size(); a++)
                for (int b : head[a]) {
                    int c = cnt[state[a]];
                    if (j >= c) f[i][j][a] += f[i - 1][j - c][b];
                    // 上面本质计算时使用的是合法状态的下标，也可以像下面用状态本身
                    // if (j >= c) f[i][j][st[si]] += f[i - 1][j - c][st[ti]];
                }

    cout << f[n + 1][m][0] << endl;

    return 0;
}
```

##### **Python**

```python
# # 状态压缩dp分为两个类：1. 棋盘式的dp（基于连通性的dp) 2. 集合式的dp
# # 状态压缩dp 直接算时间复杂度会很高，但是符合要求 可以并入计算的合法状态少，所以实际上时间复杂度要低一些

# # 状态表示：f[i, j, s]: 所有只摆在前i行，已经摆正了j个国王，并且第i行摆放的状态是s的所有方案的集合；属性：Count
# # 状态计算：分析发现，第i行能摆的状态 只和 第i-1行相关；
# # 已经摆完前i排，并且第i排的状态是a, 第i-1排的状态是b, 并且已经摆放了j个国王的所有方案；==> f[i, j, a]
# # 已经摆完前i-1行，并且第i-1排的状态是b, 并且已经摆了j-count(a)个国王的所有方案；==> f[i - 1, j - count(a), b]

# # 如何判断方案是否合法？直接枚举第i-1行和第i行的所有状态所需循环次数太多，所以这里进行预处理，需要满足以下条件：
# # 1. 第i-1行内部不能有两个1相邻；2. 第i-1行和第i行之间也不能相互攻击到：1）(a & b) ==0 2) (a|b)不能有两个相邻的1。
# # 用state数组来存储不存在连续个1的合法方案数,用head数组来存储与合法方案数进行’|’和’&’运算后的合法方案数


N = 12
M = 1 << 10
K = 110

f = [[[0] * M for _ in range(K)] for _ in range(N)]
state = []
cnt = [0] * M


def check(st):
    for i in range(n):
        if (st >> i & 1) and (st >> i + 1 & 1):
            return False
    return True


def count(st):
    res = 0
    for i in range(n):
        res += st >> i & 1
    return res


if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1 << n):  # 预处理，遍历所有状态，先筛选出来所有合法的
        if (check(i)):
            state.append(i)
            cnt[i] = count(i)

            # head = collections.defaultdict(list)
    head = [[] for _ in range(1 << n)]
    for a in state:  # 找出所有可以相邻的状态对，并进行存储
        for b in state:
            if (a & b) == 0 and check(a | b):
                head[a].append(b)  # 将合法方案的下标存储到数组中,这样可以简化状态的表示

    # 开始进行dp，先初始化: 前0行，只有0个国王的状态的方案数是1
    f[0][0][0] = 1
    for i in range(1, n + 2):  # 枚举行数
        for j in range(m + 1):  # 枚举国王的数量
            for a in state:  # 枚举合法的方案(st数组中的下标)
                for b in head[a]:
                    c = cnt[a]
                    if j >= c:
                        f[i][j][a] += f[i - 1][j - c][b]
    print(f[n + 1][m][0])  # 在第n+1行中,摆放了m个国王,并且第n+1行的状态为0000时的状态即为总方案数,因为此时m个国王在1~i中
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Luogu 邦邦的大合唱站队](https://www.luogu.com.cn/problem/P3694)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典
> 
> 排列类问题 只要考虑已为前面的某些排列好 再考虑当前即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 考虑枚举最终的排列 计算有多少个位值不同(需出列)

const int N = 1e5 + 10, M = (1 << 20) + 10;

int n, m;
int s[N][22], sz[M];
int f[M], len[M];

int main() {
    cin >> n >> m;
    
    for (int i = 1; i <= n; ++ i ) {
        int x;
        cin >> x;
        for (int j = 1; j <= m; ++ j )
            s[i][j] = s[i - 1][j] + (x == j);
        sz[x] ++ ;
    }
    
    memset(f, 0x3f, sizeof f);
    f[0] = 0; len[0] = 0;
    for (int i = 0; i < 1 << m; ++ i )
        for (int j = 0; j < m; ++ j )
            if (i >> j & 1) {
                int old = i ^ (1 << j);
                int l = len[old];
                int r = l + sz[j + 1];
                len[i] = r;
                
                int has = s[r][j + 1] - s[l][j + 1];
                f[i] = min(f[i], f[i ^ (1 << j)] + sz[j + 1] - has);
            }
        
    cout << f[(1 << m) - 1] << endl;
    
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

> [!NOTE] **[LeetCode 1434. 每个人戴不同帽子的方案数](https://leetcode.cn/problems/number-of-ways-to-wear-different-hats-to-each-other/)** [TAG]
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
class Solution {
public:
    typedef long long ll;
    const ll MOD = 1e9 + 7;
    int numberWays(vector<vector<int>>& hats) {
        int n = hats.size();
        vector<ll> dp(1 << n);
        dp[0] = 1;
        vector<set<int>> s(41);
        for (int i = 0; i < n; ++i)
            for (int hat : hats[i]) s[hat].insert(i);
        for (int i = 1; i <= 40; ++i) {
            for (int state = (1 << n) - 1; state >= 0; --state) {
                for (int person : s[i]) {
                    if (state & (1 << person)) continue;
                    int next = state + (1 << person);
                    dp[next] += dp[state];
                    dp[next] %= MOD;
                }
            }
        }
        return dp[(1 << n) - 1];
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

> [!NOTE] **[LeetCode 351. 安卓系统手势解锁](https://leetcode.cn/problems/android-unlock-patterns/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 快速标记非法情况的细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 10;

    int g[N][N];
    void init() {
        g[0][2] = g[2][0] = 1;
        g[3][5] = g[5][3] = 4;
        g[6][8] = g[8][6] = 7;
        g[0][6] = g[6][0] = 3;
        g[1][7] = g[7][1] = 4;
        g[2][8] = g[8][2] = 5;
        g[0][8] = g[8][0] = 4;
        g[2][6] = g[6][2] = 4;
    }

    int f[N][1 << N][N];   // 当前 画了 i 个, 状态是 j, 最后位置为 k
    // ATTENTION 注意状态j中1的数量并不等于i，因为它有可能越过其他点

    bool hasempty(int st, int a, int b) {
        int x = g[a][b];
        if (!x) // ATTENTION
            return false;
        return (st >> x & 1) == 0;
    }

    int numberOfPatterns(int m, int n) {
        init();
        memset(f, 0, sizeof f);
        for (int i = 0; i < 9; ++ i )
            f[1][1 << i][i] = 1;
        for (int i = 2; i <= n; ++ i )
            for (int j = 0; j < 1 << 9; ++ j )
                for (int k = 0; k < 9; ++ k )
                    if (j >> k & 1) {
                        // 上一个点为 x
                        for (int x = 0; x < 9; ++ x ) {
                            if (!(j >> x & 1) || x == k)
                                continue;
                            if (hasempty(j, x, k))
                                continue;
                            f[i][j][k] += f[i - 1][j ^ 1 << k][x];
                        }
                    }

        int res = 0;
        for (int i = m; i <= n; ++ i )
            for (int j = 0; j < 1 << 9; ++ j )
                for (int k = 0; k < 9; ++ k )
                    res += f[i][j][k];
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

> [!NOTE] **[LeetCode 1655. 分配重复整数](https://leetcode.cn/problems/distribute-repeating-integers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 枚举子集加速
> 
> ```cpp
> for (int k = t; k; k = (k - 1) & t)
> ```

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool canDistribute(vector<int>& nums, vector<int>& q) {
        unordered_map<int, int> hash;
        for (auto x: nums) hash[x] ++ ;
        vector<int> w(1);
        for (auto [x, y]: hash) w.push_back(y);
        int n = w.size() - 1, m = q.size();
        vector<vector<int>> f(n + 1, vector<int>(1 << m));
        vector<int> s(1 << m);
        for (int i = 0; i < 1 << m; i ++ )
            for (int j = 0; j < m; j ++ )
                if (i >> j & 1)
                    s[i] += q[j];
        
        f[0][0] = 1;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < 1 << m; j ++ )
                if (f[i][j]) {
                    f[i + 1][j] = 1;
                    for (int t = j ^ ((1 << m) - 1), k = t; k; k = (k - 1) & t) {
                        if (s[k] <= w[i + 1])
                            f[i + 1][j | k] = 1;
                    }
                }
        
        return f[n][(1 << m) - 1];        
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

> [!NOTE] **[LeetCode 1799. N 次操作后的最大分数和](https://leetcode.cn/problems/maximize-score-after-n-operations/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然状压 dp 注意实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxScore(vector<int>& nums) {
        int n = nums.size();
        vector<int> f(1 << n);
        for (int i = 0; i < 1 << n; ++ i ) {
            int cnt = n - __builtin_popcount(i);
            cnt = cnt / 2 + 1;
            
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    for (int k = j + 1; k < n; ++ k )
                        if (i >> k & 1)
                            f[i] = max(f[i], f[i - (1 << j) - (1 << k)] + __gcd(nums[j], nums[k]) * cnt);
        }
        return f[(1 << n) - 1];
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

> [!NOTE] **[LeetCode 1947. 最大兼容性评分和](https://leetcode.cn/problems/maximum-compatibility-score-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 爆搜即可 也可状压DP 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 10, M = 260;
    
    int m, n;
    vector<int> st, mt;
    int g[N][N], f[M];
    
    int maxCompatibilitySum(vector<vector<int>>& students, vector<vector<int>>& mentors) {
        this->m = students.size(), this->n = students[0].size();
        st.clear(); mt.clear();
        for (auto & sts : students) {
            int s = 0;
            for (int i = 0; i < n; ++ i )
                if (sts[i] & 1)
                    s ^= 1 << i;
            st.push_back(s);
        }
        for (auto mts : mentors) {
            int s = 0;
            for (int i = 0; i < n; ++ i )
                if (mts[i] & 1)
                    s ^= 1 << i;
            mt.push_back(s);
        }
        
        memset(g, 0, sizeof g);
        for (int i = 0; i < m; ++ i )
            for (int j = 0; j < m; ++ j ) {
                int a = st[i], b = mt[j];
                int s = 0;
                for (int k = 0; k < n; ++ k )
                    if ((a >> k & 1) == (b >> k & 1))
                        s ++ ;
                g[i][j] = s;
            }
        
        memset(f, 0, sizeof f);
        for (int i = 0; i < 1 << m; ++ i ) {
            int sz = __builtin_popcount(i);
            for (int j = 0; j < m; ++ j )
                if (i >> j & 1)
                    f[i] = max(f[i], f[i ^ (1 << j)] + g[j][sz - 1]);
        }
        
        return f[(1 << m) - 1];
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

> [!NOTE] **[LeetCode 1986. 完成任务的最少工作时间段](https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然需枚举方案 以及【枚举子集】
> 
> 注意需要提前预处理消耗 否则 TLE

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int INF = 1e9;
    int minSessions(vector<int>& tasks, int sessionTime) {
        int n = tasks.size();
        vector<int> f(1 << n, INF), cost(1 << n, 0);
        for (int i = 0; i < 1 << n; ++ i ) {
            int s = 0;
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    s += tasks[j];
            cost[i] = s;
        }
        
        f[0] = 0;
        for (int i = 1; i < 1 << n; ++ i )
            for (int j = i; j; j = (j - 1) & i)
                if (cost[j] <= sessionTime)
                    f[i] = min(f[i], f[i ^ j] + 1);
        return f[(1 << n) - 1];
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

> [!NOTE] **[Codeforces Roman and Numbers](http://codeforces.com/problemset/problem/401/D)**
> 
> 题意: 
> 
> 将 n(n<=10^18) 的各位数字重新排列（不允许有前导零）
> 
> 求：可以构造几个 mod m 等于 0 的数字

> [!TIP] **思路**
> 
> 显然定义是已选哪些数，当前模数
> 
> 需要对接下来选的数字去重 => 增加 st 数组
> 
> 状压dp + **细节 `st` 标记**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Roman and Numbers
// Contest: Codeforces - Codeforces Round #235 (Div. 2)
// URL: https://codeforces.com/problemset/problem/401/D
// Memory Limit: 512 MB
// Time Limit: 4000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1 << 18, M = 110;

LL n, m;
LL f[N][M];  // 选取数为i 模数为j的所有方案数和

int main() {
    cin >> n >> m;

    vector<int> nums;
    while (n)
        nums.push_back(n % 10), n /= 10;
    n = nums.size();

    memset(f, 0, sizeof f);
    f[0][0] = 1;
    for (int i = 0; i < 1 << n; ++i)
        for (int j = 0; j < m; ++j)
            if (f[i][j]) {
                // ATTENTION 必须标记在当前这个顺序下哪些值已经被使用过，否则WA
                static bool st[10];
                memset(st, 0, sizeof st);
                for (int k = 0; k < n; ++k)

                    // 当前最后一位是k ==> 这种做法右侧取模涉及除法会比较难做
                    // 改为：接下来选一个k
                    if ((i >> k & 1) == 0 && !st[nums[k]]) {
                        int next = i ^ (1 << k);
                        if (i || nums[k])
                            // 非前导0
                            f[next][(j * 10 + nums[k]) % m] += f[i][j];

                        st[nums[k]] = true;
                    }
            }

    cout << f[(1 << n) - 1][0] << endl;

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

> [!NOTE] **[LeetCode 1125. 最小的必要团队](https://leetcode.cn/problems/smallest-sufficient-team/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常典型的题目 注意状压定义与转移的 `for-loop` 顺序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 61, M = 16;

    int f[1 << M];
    PII pre[1 << M];

    vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
        int n = req_skills.size(), m = people.size();
        unordered_map<string, int> hash;
        for (int i = 0; i < n; ++ i )
            hash[req_skills[i]] = i;
        
        vector<int> sts;
        for (int i = 0; i < people.size(); ++ i ) {
            int st = 0;
            for (auto & x : people[i])
                st |= 1 << hash[x];
            sts.push_back(st);
        }
        
        memset(f, 0x3f, sizeof f);
        
        f[0] = 0;
        // ATTENTION 先遍历状态
        for (int i = 0; i < 1 << n; ++ i )
            for (int j = 0; j < m; ++ j ) {
                int x = sts[j];
                // 因为 或 的性质，比较难推导或之前的状态，故反过来计算或之后的状态
                if (f[i | x] > f[i] + 1) {
                    f[i | x] = f[i] + 1;
                    pre[i | x] = {i, j};
                }
            }

        vector<int> res;
        for (int st = (1 << n) - 1; st; ) {
            res.push_back({pre[st].second});
            st = pre[st].first;
        }

        return res;
    }
};
```

##### **C++ 最初的二维写法**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 61, M = 16;

    int f[N][1 << M];
    PII pre[N][1 << M];

    vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
        int n = req_skills.size(), m = people.size();
        unordered_map<string, int> hash;
        for (int i = 0; i < n; ++ i )
            hash[req_skills[i]] = i;
        
        vector<int> sts;
        for (int i = 0; i < people.size(); ++ i ) {
            int st = 0;
            for (auto & x : people[i])
                st |= 1 << hash[x];
            sts.push_back(st);
        }
        
        memset(f, 0x3f, sizeof f);
        
        f[0][0] = 0;
        // ATTENTION 先遍历状态
        for (int i = 0; i < m; ++ i )
            for (int j = 0; j < 1 << n; ++ j ) {
                int x = sts[i];
                if (f[i + 1][j | x] > f[i][j] + 1) {
                    f[i + 1][j | x] = f[i][j] + 1;
                    pre[i + 1][j | x] = {i, j};
                }
                // 用二维 以及这个状态定义的情况下 必须加下面这一段
                if (f[i + 1][j] > f[i][j]) {
                    f[i + 1][j] = f[i][j];
                    pre[i + 1][j] = {i, j};
                }
            }
        // cout << " f = " << f[m][(1 << n) - 1] << endl;

        vector<int> res;
        for (int i = m, j = (1 << n) - 1; j; -- i ) {
            auto [li, lj] = pre[i][j];
            if (f[li][lj] != f[i][j])
                res.push_back(i - 1);
            j = lj;
        }

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

### 线性递推

> [!NOTE] **[AcWing 327. 玉米田](https://www.acwing.com/problem/content/description/329/)**
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
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int N = 14, M = 1 << 12, mod = 1e8;

int n, m;
int w[N];
vector<int> state;
vector<int> head[M];
int f[N][M];

bool check(int state) {
    for (int i = 0; i + 1 < m; i++)
        if ((state >> i & 1) && (state >> i + 1 & 1)) return false;
    return true;
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < m; j++) {
            int t;
            cin >> t;
            w[i] += !t * (1 << j);
        }

    // 所有单行合法状态列表（不相邻）
    for (int i = 0; i < 1 << m; i++)
        if (check(i)) state.push_back(i); // 后面用n+1的重要原因：0状态必然合法 会加进去
    // n 列  处理上一列和本列合法的状态转移
    for (int i = 0; i < state.size(); i++)
        for (int j = 0; j < state.size(); j++) {
            int a = state[i], b = state[j];
            if (!(a & b)) head[i].push_back(j);
        }

    // f[i][j] 到第i行，种植状态为j的所有方案数
    // 本来第二维是 1<<n 这里也可以用sz+1 只要后面对应 类似小国王那题
    f[0][0] = 1;
    for (int i = 1; i <= n + 1; i++)
        for (int j = 0; j < state.size(); j++)
            if (!(state[j] & w[i]))
                for (int k : head[j]) f[i][j] = (f[i][j] + f[i - 1][k]) % mod;

    cout << f[n + 1][0] << endl;

    return 0;
}
```

##### **Python**

```python
N = 14
M = 1 << 12
mod = int(1e8)
f = [[0] * M for _ in range(N)]
state = []
head = [[] for _ in range(M)]
w = [0] * N


def check(st):
    for i in range(m):
        if (st >> i & 1) and (st >> i + 1 & 1):
            return False
    return True


# def check(x):
#     return x & (x >> 1) == 0 


if __name__ == '__main__':
    n, m = map(int, input().split())
    # mod = int(1e8)
    # M = 1 << m 
    # state = []
    # w = [0] * (n + 2)

    # head = [[] for _ in range(M)]
    # f = [[0] * M for _ in range(n+2)]

    for i in range(1, n + 1):  # 用二进制表示当前行的土地是否具备种植条件：当前位为1表示可以种植，0表示没有办法种植
        a = list(map(int, input().split()))
        for j in range(m):
            w[i] += (1 if a[m - 1 - j] == 0 else 0) << j

    for i in range(1 << m):
        if check(i):
            state.append(i)

    for a in state:  # 只要没有交集，i就是转移到j(a就可以转移到b)
        for b in state:
            if (a & b) == 0:
                head[a].append(b)

    f[0][0] = 1  # 初始化，表示一行都没有的情况下，一根玉米都不能种，方案数是1
    for i in range(1, n + 2):  # 小技巧，枚举到n+1 行，这样s输出的时候就不需要枚举最后一行的状态
        for j in state:
            if not (j & w[i]):
                for k in head[j]:
                    f[i][j] = (f[i][j] + f[i - 1][k]) % mod
    print(f[n + 1][0])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 292. 炮兵阵地](https://www.acwing.com/problem/content/description/294/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 进阶: 第 i 行与 i-1 and i-2 行状态都相关。
> 
> 重点在状态定义：f[i][j][k] 表示第i行状态k第i-1行状态j的最大摆放数量

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int N = 10, M = 1 << 10;

int n, m;
int g[1010];
int f[2][M][M];
vector<int> state;
int cnt[M];

bool check(int state) {
    for (int i = 0; i < m; i++)
        if ((state >> i & 1) && ((state >> i + 1 & 1) || (state >> i + 2 & 1)))
            return false;
    return true;
}

int count(int state) {
    int res = 0;
    for (int i = 0; i < m; i++)
        if (state >> i & 1) res++;
    return res;
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < m; j++) {
            char c;
            cin >> c;
            g[i] += (c == 'H') << j;
        }

    for (int i = 0; i < 1 << m; i++)
        if (check(i)) {
            state.push_back(i);
            cnt[i] = count(i);
        }

    for (int i = 1; i <= n; i++)
        for (int j = 0; j < state.size(); j++)
            for (int k = 0; k < state.size(); k++)
                for (int u = 0; u < state.size(); u++) {
                    int a = state[j], b = state[k], c = state[u];
                    if (a & b | a & c | b & c) continue;
                    if (g[i] & b | g[i - 1] & a) continue;
                    f[i & 1][j][k] =
                        max(f[i & 1][j][k], f[i - 1 & 1][u][j] + cnt[b]);
                }

    int res = 0;
    for (int i = 0; i < state.size(); i++)
        for (int j = 0; j < state.size(); j++) res = max(res, f[n & 1][i][j]);

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

> [!NOTE] **[LeetCode 526. 优美的排列](https://leetcode.cn/problems/beautiful-arrangement/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 优雅暴力写法
> 
> 以及更优的状压写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countArrangement(int N) {
        vector<int> f(1 << N);
        f[0] = 1;
        for (int i = 0; i < 1 << N; ++ i ) {
            // 共有 k 个数
            int k = 0;
            for (int j = 0; j < N; ++ j )
                if (i >> j & 1) ++ k ;
            // 加上第 j 个数
            for (int j = 0; j < N; ++ j )
                if (!(i >> j & 1))
                    if ((k + 1) % (j + 1) == 0 || (j + 1) % (k + 1) == 0)
                        f[i | (1 << j)] += f[i];
        }
        return f[(1 << N) - 1];
    }
};


// 远古代码
class Solution {
public:
    int n, tot, res;
    map<pair<int, int>, int> m;
    void dfs(int state, int sum, int pos) {
        if (sum >= tot) {    // if (pos >= n)
            res ++ ;
            return;
        }
        for (int i = 1; i <= n; ++ i ) {
            if (((state & (1 << i)) == 0) && ( i % pos == 0 || pos % i == 0)) {
                dfs(state | 1 << i, sum + i, pos + 1);
            }
        }
    }
    int countArrangement(int N) {
        n = N, tot = (1 + n) * n / 2, res = 0;
        for (int i = 1; i <= n; ++ i ) {
            // if ((i % 1) == 0 || (1 % i) == 0) // 隐含条件
            dfs(1 << i, i, 2);
        }
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

> [!NOTE] **[LeetCode 1349. 参加考试的最大学生数](https://leetcode.cn/problems/maximum-students-taking-exam/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然放了和没放的可以转化为状态 每一行2^8 = 256
> 
> 状压dp
> 
> $dp[row][state]=max(dp[row−1][last]+state.count())$
> 
> 以及检查合法性 最后结果为 $max(dp[m][state])$
> 
> 也可以网络流

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, m;
    int mat[8];
    int dp[8][1 << 8];
    int sum(int x) {  // 计算该行有多少个1 即多少个人
        int s = 0;
        while (x > 0) s += 1 & x, x >>= 1;
        return s;
    }
    bool judge(int s1, int s2) { return (s1 & (s2 << 1)) || (s1 & (s2 >> 1)); }
    int maxStudents(vector<vector<char>>& seats) {
        m = seats.size(), n = seats[0].size();
        for (int i = 0; i < m; i++) {
            int s = 0;
            for (int j = 0; j < n; j++) {
                if (seats[i][j] == '#') s |= 1 << j;
            }
            mat[i] = s;
        }

        memset(dp, 0, sizeof(dp));
        // 第0行
        for (int s = 0; s < 1 << n; s++) {
            if (!(s & mat[0] || s & (s >> 1))) dp[0][s] = sum(s);
            // if(!(s & mat[0] || s & (s >> 1) || s & (s << 1))) dp[0][s] =
            // sum(s);
        }
        for (int i = 1; i < m; i++) {
            for (int s1 = 0; s1 < 1 << n; s1++) {
                // 判断状态是否合法【i 有人或右边有人 -> 不合法 continue】
                if (s1 & mat[i] || (s1 & (s1 >> 1))) continue;
                // if(s1 & mat[i] || (s1 & (s1 >> 1)) || (s1 & (s1 << 1)))
                // continue;
                for (int s2 = 0; s2 < 1 << n; s2++) {
                    // 如果合法 更新dp[i][s1]; 其初值自然是0
                    if (!judge(s1, s2))
                        dp[i][s1] = max(dp[i][s1], dp[i - 1][s2]);
                }
                // 加上本行的人数 经过判断 这里的s1必定不包含落在#上的情况
                dp[i][s1] += sum(s1);
            }
        }
        return *max_element(dp[m - 1], dp[m - 1] + (1 << n));
    }
};
```

##### **C++ 网络流**

首先我们只关心可以坐人的座位，我们把作为按照列下标的奇偶建二分图，S向奇数下标的座位连边，偶数下标的座位向T连边，有冲突的座位奇数座位向偶数座位连边。

图中所有边流量都是1。直接跑二分图最大点独立集就行，即可以坐人的座位数-最大流。 时间复杂度：O(n^3) 空间复杂度：O(n^2)

作者：LighTcml

```cpp
class Solution {
    const int INF = 1 << 29;
    int tot, head[110];
    struct Edge {
        int to, net, v;
    } E[10010];
    void addedge(int x, int y, int v) {
        E[++tot].to = y;
        E[tot].net = head[x];
        head[x] = tot;
        E[tot].v = v;
        E[++tot].to = x;
        E[tot].net = head[y];
        head[y] = tot;
        E[tot].v = 0;
    }
    int n, m, S, T, Q[110], depth[110];
    int getp(int x, int y) { return x * m + y + 1; }
    bool bfs() {
        for (int i = S; i <= T; ++i) depth[i] = -1;
        int L = 0, R = 1;
        Q[1] = S;
        depth[S] = 0;
        while (L < R) {
            int x = Q[++L];
            for (int i = head[x]; i; i = E[i].net)
                if (E[i].v > 0 && depth[E[i].to] == -1) {
                    depth[E[i].to] = depth[x] + 1;
                    Q[++R] = E[i].to;
                }
        }
        return depth[T] != -1;
    }
    int dfs(int x, int flow) {
        if (x == T || !flow) return flow;
        int w = 0;
        for (int i = head[x]; i; i = E[i].net)
            if (E[i].v > 0 && depth[E[i].to] == depth[x] + 1) {
                int v = dfs(E[i].to, min(flow - w, E[i].v));
                E[i].v -= v;
                E[i ^ 1].v += v;
                w += v;
            }
        if (!w) depth[x] = -1;
        return w;
    }
    int Dinic() {
        int sum = 0;
        while (bfs()) sum += dfs(S, INF);
        return sum;
    }

public:
    int maxStudents(vector<vector<char>>& seats) {
        tot = 1;
        n = seats.size();
        m = seats[0].size();
        S = 0;
        T = n * m + 1;
        int cnt = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                if (seats[i][j] == '.') {
                    ++cnt;
                    int x = i * m + j + 1;
                    if (j & 1)
                        addedge(S, x, 1);
                    else
                        addedge(x, T, 1);
                    if (j - 1 >= 0 && seats[i][j - 1] == '.') {
                        if (j & 1)
                            addedge(x, getp(i, j - 1), 1);
                        else
                            addedge(getp(i, j - 1), x, 1);
                    }
                    if (j + 1 < m && seats[i][j + 1] == '.') {
                        if (j & 1)
                            addedge(x, getp(i, j + 1), 1);
                        else
                            addedge(getp(i, j + 1), x, 1);
                    }
                    if (i && j + 1 < m && seats[i - 1][j + 1] == '.') {
                        if (j & 1)
                            addedge(x, getp(i - 1, j + 1), 1);
                        else
                            addedge(getp(i - 1, j + 1), x, 1);
                    }
                    if (i && j && seats[i - 1][j - 1] == '.') {
                        if (j & 1)
                            addedge(x, getp(i - 1, j - 1), 1);
                        else
                            addedge(getp(i - 1, j - 1), x, 1);
                    }
                }
        return cnt - Dinic();
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

> [!NOTE] **[LeetCode 1494. 并行课程 II](https://leetcode.cn/problems/parallel-courses-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 状压dp
> 
> 逆序拓扑的思路是对的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
int f[1 << 15];
class Solution {
public:
    const int inf = 1e9;
    int minNumberOfSemesters(int n, vector<vector<int>>& dependencies, int k) {
        vector<vector<int>> es(n);
        for (auto e : dependencies) es[e[1] - 1].push_back(e[0] - 1);
        for (int s = 0; s < 1 << n; ++s) f[s] = inf;
        f[0] = 0;
        for (int s = 0; s < 1 << n; ++s) {
            if (f[s] == inf) continue;
            int can = 0;
            for (int v = 0; v < n; ++v) {
                if (s & 1 << v) continue;
                bool f = 0;
                for (auto u : es[v])
                    if (!(s & 1 << u)) f = 1;
                if (!f) can |= 1 << v;
            }
            for (int t = can; t; t = (t - 1) & can) {
                if (__builtin_popcount(t) > k) continue;
                f[s | t] = min(f[s | t], f[s] + 1);
            }
        }
        return f[(1 << n) - 1];
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

> [!NOTE] **[LeetCode 3276. 选择矩阵中单元格的最大得分](https://leetcode.cn/problems/select-cells-in-grid-with-maximum-score/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 重点在于思路拆解
> 
> 分析题意
> 
> -   选取元素不同行
> -   选取元素值不同
> 
> 显然可以按值统计 对行状压

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 显然无法贪心 考虑暴搜或DP
    //
    // DP 无非两种
    // 1. 考虑到第i行 已经选了哪些元素 复杂度 O(n 2^(nm) nm)
    // 2. 考虑到第i个元素 已经用了哪些行 O(nm 2^n n)          => choosed
    //
    // [行数和值都不能重复，行数的大小明显小于值，所以对行数进行状态压缩，统计每个值所在的行]
    
    const static int N = 110, M = (1 << 10) + 1;
    
    int f[N][M];    // 考虑前i个数字 目前选取行数的状压表示为j的情况下 的最大值
    
    int maxScore(vector<vector<int>>& grid) {
        // 1. 将行状压 记录每个数字所在的行(可能多个行)
        unordered_map<int, int> pos;
        int n = grid.size();
        for (int i = 0; i < n; ++ i )
            for (int x : grid[i])
                pos[x] |= 1 << i;
        
        vector<int> all_nums;
        for (auto & [k, _] : pos)
            all_nums.push_back(k);
        
        int m = all_nums.size();
        int cap = 1 << n;
        
        memset(f, 0, sizeof f);
        
        for (int i = 1; i <= m; ++ i ) {
            int x = all_nums[i - 1];    // 当前数字为 x
            for (int j = 0; j < cap; ++ j ) {
                // 不选x
                f[i][j] = f[i - 1][j];
                // 选x 则之前的行不能冲突 (因为已经按数值做了聚合 可以保证数字一定不重复)
                for (int k = 0; k < n; ++ k )
                    if ((pos[x] >> k & 1) && (j >> k & 1) == 0)
                        f[i][j] = max(f[i][j], f[i - 1][j ^ (1 << k)] + x);
            }
        }
        
        int res = 0;
        for (int j = 0; j < cap; ++ j )
            res = max(res, f[m][j]);
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

### 进阶

> [!NOTE] **[AcWing 524. 愤怒的小鸟](https://www.acwing.com/problem/content/526/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO
> 
> > 经典的 “重复覆盖问题”，即给定01矩阵，要求选择尽量少的行，将所有列覆盖住。这里标准做法是使用 Dancing Links。
> > 
> > 但由于 n<=18 ，因此可以直接使用状态压缩DP求解，代码更简单。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 状压AC**

```cpp
// 状态压缩做法，AC
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#define x first
#define y second

using namespace std;

typedef pair<double, double> PDD;

const int N = 18, M = 1 << 18;
const double eps = 1e-8;

int n, m;
PDD q[N];
int path[N][N];
int f[M];

int cmp(double x, double y) {
    if (fabs(x - y) < eps) return 0;
    if (x < y) return -1;
    return 1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        cin >> n >> m;
        for (int i = 0; i < n; i++) cin >> q[i].x >> q[i].y;

        memset(path, 0, sizeof path);
        for (int i = 0; i < n; i++) {
            path[i][i] = 1 << i;
            for (int j = 0; j < n; j++) {
                double x1 = q[i].x, y1 = q[i].y;
                double x2 = q[j].x, y2 = q[j].y;
                if (!cmp(x1, x2)) continue;
                double a = (y1 / x1 - y2 / x2) / (x1 - x2);
                double b = y1 / x1 - a * x1;

                if (cmp(a, 0) >= 0) continue;
                int state = 0;
                for (int k = 0; k < n; k++) {
                    double x = q[k].x, y = q[k].y;
                    if (!cmp(a * x * x + b * x, y)) state += 1 << k;
                }
                path[i][j] = state;
            }
        }

        memset(f, 0x3f, sizeof f);
        f[0] = 0;
        for (int i = 0; i + 1 < 1 << n; i++) {
            int x = 0;
            for (int j = 0; j < n; j++)
                if (!(i >> j & 1)) {
                    x = j;
                    break;
                }

            for (int j = 0; j < n; j++)
                f[i | path[x][j]] = min(f[i | path[x][j]], f[i] + 1);
        }

        cout << f[(1 << n) - 1] << endl;
    }

    return 0;
}
```

##### **C++ 跳舞链TLE**

```cpp
// Dancing Links，TLE，能过(20/21)
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>

#define x first
#define y second

using namespace std;

typedef pair<double, double> PDD;
const int N = 10000;
const double eps = 1e-8;

int n, m;
int l[N], r[N], u[N], d[N], col[N], row[N], s[N], idx;
bool st[20];
int path[20][20];
PDD q[20];

int cmp(double x) {
    if (abs(x) < eps) return 0;
    if (x > 0) return 1;
    return -1;
}

void init() {
    for (int i = 0; i <= m; i++) {
        l[i] = i - 1, r[i] = i + 1;
        col[i] = u[i] = d[i] = i;
        s[i] = 0;
    }
    l[0] = m, r[m] = 0;
    idx = m + 1;
}

void add(int& hh, int& tt, int x, int y) {
    row[idx] = x, col[idx] = y, s[y]++;
    u[idx] = y, d[idx] = d[y], u[d[y]] = idx, d[y] = idx;
    r[hh] = l[tt] = idx, r[idx] = tt, l[idx] = hh;
    tt = idx++;
}

int h() {
    int res = 0;
    memset(st, 0, sizeof st);
    for (int i = r[0]; i; i = r[i]) {
        if (st[i]) continue;
        st[i] = true;
        res++;
        for (int j = d[i]; j != i; j = d[j])
            for (int k = r[j]; k != j; k = r[k]) st[col[k]] = true;
    }
    return res;
}

void remove(int p) {
    for (int i = d[p]; i != p; i = d[i]) {
        r[l[i]] = r[i];
        l[r[i]] = l[i];
    }
}

void resume(int p) {
    for (int i = u[p]; i != p; i = u[i]) {
        r[l[i]] = i;
        l[r[i]] = i;
    }
}

bool dfs(int k, int depth) {
    if (k + h() > depth) return false;
    if (!r[0]) return true;
    int p = r[0];
    for (int i = r[0]; i; i = r[i])
        if (s[p] > s[i]) p = i;

    for (int i = d[p]; i != p; i = d[i]) {
        remove(i);
        for (int j = r[i]; j != i; j = r[j]) remove(j);
        if (dfs(k + 1, depth)) return true;
        for (int j = l[i]; j != i; j = l[j]) resume(j);
        resume(i);
    }
    return false;
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        scanf("%d%*d", &n);
        for (int i = 0; i < n; i++) scanf("%lf%lf", &q[i].x, &q[i].y);
        memset(path, 0, sizeof path);
        for (int i = 0; i < n; i++) {
            path[i][i] = 1 << i;
            for (int j = i + 1; j < n; j++) {
                double x1 = q[i].x, y1 = q[i].y;
                double x2 = q[j].x, y2 = q[j].y;
                if (!cmp(x1 - x2)) continue;
                double a = (y1 / x1 - y2 / x2) / (x1 - x2);
                if (cmp(a) >= 0) continue;
                double b = y1 / x1 - a * x1;
                for (int k = 0; k < n; k++) {
                    double x = q[k].x, y = q[k].y;
                    if (!cmp(a * x * x + b * x - y)) path[i][j] += 1 << k;
                }
            }
        }

        m = n;
        init();

        vector<int> S;
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++) S.push_back(path[i][j]);

        for (int i = 0; i < S.size(); i++)
            for (int j = 0; j < S.size(); j++)
                if (i != j && S[i] != -1 && S[j] != -1 &&
                    (S[i] & S[j]) == S[i]) {
                    S[i] = -1;
                    break;
                }

        int k = 0;
        for (auto x : S)
            if (x != -1) S[k++] = x;
        S.erase(S.begin() + k, S.end());

        sort(S.begin(), S.end());

        for (auto x : S) {
            int hh = idx, tt = idx;
            for (int i = 0; i < n; i++)
                if (x >> i & 1) add(hh, tt, 0, i + 1);
        }

        int depth = 0;
        while (!dfs(0, depth)) depth++;
        printf("%d\n", depth);
    }

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

> [!NOTE] **[AcWing 529. 宝藏](https://www.acwing.com/problem/content/531/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> TODO
> 
> 实现细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 12, M = 1 << 12, INF = 0x3f3f3f3f;

int n, m;
int d[N][N];
int f[M][N], g[M];

int main() {
    scanf("%d%d", &n, &m);

    memset(d, 0x3f, sizeof d);
    for (int i = 0; i < n; i++) d[i][i] = 0;

    while (m--) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        a--, b--;
        d[a][b] = d[b][a] = min(d[a][b], c);
    }

    for (int i = 1; i < 1 << n; i++)
        for (int j = 0; j < n; j++)
            if (i >> j & 1) {
                for (int k = 0; k < n; k++)
                    if (d[j][k] != INF) g[i] |= 1 << k;
            }

    memset(f, 0x3f, sizeof f);
    for (int i = 0; i < n; i++) f[1 << i][0] = 0;

    for (int i = 1; i < 1 << n; i++)
        for (int j = (i - 1); j; j = (j - 1) & i)
            if ((g[j] & i) == i) {
                int remain = i ^ j;
                int cost = 0;
                for (int k = 0; k < n; k++)
                    if (remain >> k & 1) {
                        int t = INF;
                        for (int u = 0; u < n; u++)
                            if (j >> u & 1) t = min(t, d[k][u]);
                        cost += t;
                    }

                for (int k = 1; k < n; k++)
                    f[i][k] = min(f[i][k], f[j][k - 1] + cost * k);
            }

    int res = INF;
    for (int i = 0; i < n; i++) res = min(res, f[(1 << n) - 1][i]);

    printf("%d\n", res);
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

> [!NOTE] **[LeetCode 691. 贴纸拼词](https://leetcode.cn/problems/stickers-to-spell-word/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 背包进阶+ 双重记忆化搜索
> 
>   注意不是按次数是按位
> 
>   1. 状态表示字母有没有满足
> 
>   2. 剪枝：枚举下一个单词选哪个 加了某个字母之后回到哪个状态【记忆化】
> 
>   3. 爆搜已满足的所有字母【同样记忆化】
> 
> - 直接状压dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 1 << 15;
int f[N], g[N][26];

class Solution {
public:
    const int INF = 20;
    int n;
    string target;
    vector<string> strs;

    int fill(int state, char c) {
        auto& v = g[state][c - 'a'];
        if (v != -1) return v;
        v = state;
        for (int i = 0; i < n; i ++ )
            if (!(state >> i & 1) && target[i] == c) {
                v += 1 << i;
                break;
            }
        return v;
    }

    int dfs(int state) {
        auto& v = f[state];
        if (v != -1) return v;
        if (state == (1 << n) - 1) return v = 0;
        v = INF;
        for (auto& str: strs) {
            int cur = state;
            for (auto c: str)
                cur = fill(cur, c);
            if (cur != state)
                v = min(v, dfs(cur) + 1);
        }
        return v;
    }

    int minStickers(vector<string>& stickers, string _target) {
        memset(f, -1, sizeof f);
        memset(g, -1, sizeof g);
        target = _target;
        strs = stickers;
        n = target.size();
        int res = dfs(0);
        if (res == INF) res = -1;
        return res;
    }
};
```

##### **C++ 状压**

```cpp
class Solution {
private:
    const static int N = 15, M = 50, INF = 0x3f3f3f3f;
    int f[1 << N], seen[M][26];

public:
    int minStickers(vector<string>& stickers, string target) {
        const int n = target.size();
        vector<vector<int>> pos(26);

        for (int i = 0; i < n; i ++ )
            pos[target[i] - 'a'].push_back(i);

        vector<int> used;
        for (int c = 0; c < 26; c ++ )
            if (!pos[c].empty())
                used.push_back(c);

        int m = stickers.size();
        for (int i = 0; i < m; i ++ ) {
            memset(seen[i], 0, sizeof(seen[i]));
            for (char c : stickers[i])
                seen[i][c - 'a'] ++ ;
        }

        memset(f, 0x3f, sizeof f);
        f[0] = 0;
        for (int s = 0; s < (1 << n) - 1; s ++ ) {
            if (f[s] > INF / 2) continue;

            // 枚举再选某一个字符串
            for (int i = 0; i < m; i ++ ) {
                int t = s;
                // 只关注有用的字符
                for (char c : used)
                    for (int j = 0, k = 0; j < pos[c].size() && k < seen[i][c]; j ++ , k ++ )
                        // 已有 不需要消耗 k --
                        if ((s >> pos[c][j]) & 1) k -- ;
                        else t |= 1 << pos[c][j];

                f[t] = min(f[t], f[s] + 1);
            }
        }

        if (f[(1 << n) - 1] > INF / 2)
            return -1;

        return f[(1 << n) - 1];
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

> [!NOTE] **[LeetCode 1595. 连通两组点的最小成本](https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 最简单直接的显然状压 DP + 枚举子集
> 
>   因为已知第二组点的数量较少，所以对第二组点的连通状态进行状态压缩，然后依次处理第一组中的点即可。
> 
>   对于第一组中的每个点，第一种做法是直接连一条边，第二种做法是连接若干个第二组中当前还没有连通的点。
> 
>   - 对于第一种做法，直接枚举 $M$ 个点；
> 
>   - 对于第二种做法，假设当前未连通的点为 $mask$ ，我们需要枚举它的子集，这里可以用位运算枚举子集的方法来进行优化。
> 
> - 也可以二分图过 参考 [2020-09-20_Weekly-207](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2020-09-20_Weekly-207) 但是太麻烦了实现略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
    int f[15][5005], sum[5005];

public:
    int connectTwoGroups(vector<vector<int>>& cost) {
        memset(f, 0x3f, sizeof(f));
        f[0][0] = 0;
        int n = cost.size(), m = cost[0].size();
        int lim = (1 << m) - 1;
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j <= lim; ++j) {
                sum[j] = 0;
                for (int u = 1, v = 1; u <= m; ++u, v <<= 1) {
                    if (v & j) sum[j] += cost[i - 1][u - 1];
                }
            }
            for (int j = 0; j <= lim; ++j) {
                // with a connected
                for (int u = 1, v = 1; u <= m; ++u, v <<= 1) {
                    if (!(v & j)) continue;
                    f[i][j] = min(f[i][j], f[i - 1][j] + cost[i - 1][u - 1]);
                }
                // with unconnected
                int jj = (lim ^ j);
                for (int u = jj; u > 0; u = (u - 1) & jj) {
                    f[i][j | u] = min(f[i][j | u], f[i - 1][j] + sum[u]);
                }
            }
        }
        return f[n][lim];
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

> [!NOTE] **[LeetCode 1723. 完成所有工作的最短时间](https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 爆搜优化即可 [斯特林数 TODO](https://www.acwing.com/problem/content/3169/)
> 
> 赛榜有 DP 做法学习下
> 
> **经典线性优化 状压递推复杂度 $O(n^2)$ -> $O(n)$**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 暴搜**

```cpp
class Solution {
public:
    vector<int> js, sum;
    int n, m, k, cnt, res;
    
    
    void dfs(int p) {
        if (cnt > res) return;
        if (p == n) {
            if (cnt <= k) {
                res = cnt;
            }
            return;
        }
        
        for (int i = 0; i < cnt; ++ i ) {
            if (sum[i] + js[p] <= m) {
                sum[i] += js[p];
                dfs(p + 1);
                sum[i] -= js[p];
            }
        }
        // 写的时候超时 加一行 if cnt < k 就AC
        if (cnt < k) {
            sum[cnt ++ ] = js[p];
            dfs(p + 1);
            sum[ -- cnt] = 0;
        }
        
    }
    
    // 最大工作时间m
    int check() {
        sum = vector<int>(15);
        cnt = 0;
        res = 1e9;
        dfs(0);
        return res <= k;
    }
    
    int minimumTimeRequired(vector<int>& jobs, int k) {
        js = jobs;
        n = js.size();
        this->k = k;
        sort(js.begin(), js.end());
        reverse(js.begin(), js.end());
        
        int l = 0, r = 0;
        for (auto & v : js) r += v, l = max(l, v);
        while (l < r) {
            m = l + r >> 1;
            if (check()) r = m;
            else l = m + 1;
        }
        return l;
    }
};
```

##### **C++ 暴搜 yxc**

```cpp
class Solution {
public:
    vector<int> s, js;
    int res = 1e9;
    
    // a位置 b使用的工人 c最大值
    void dfs(int a, int b, int c) {
        if (c > res) return;
        if (a == js.size()) {
            res = c;
            return;
        }
        
        for (int i = 0; i < b; ++ i ) {
            s[i] += js[a];
            dfs(a + 1, b, max(c, s[i]));
            s[i] -= js[a];
        }
        
        if (b < s.size()) {
            s[b] = js[a];
            dfs(a + 1, b + 1, max(c, s[b]));
            s[b] = 0;
        }
    }
    
    int minimumTimeRequired(vector<int>& jobs, int k) {
        js = jobs, s.resize(k);
        dfs(0, 0, 0);
        return res;
    }
};
```

##### **C++ dp**

```cpp
class Solution {
public:
    int minimumTimeRequired(vector<int>& jobs, int k) {
        int n = jobs.size();
        // 一个任务子集的总时间
        vector<int> tot(1 << n);
        
        // 写法1
        // 【学习记忆 这种写法更好】
        for (int i = 1; i < (1 << n); ++ i ) {
            // o = 最右侧0的数量
            int o = __builtin_ctz(i & (-i));
            tot[i] = tot[i ^ (i & (-i))] + jobs[o];
        }
        
        /* 写法2
        for (int i = 1; i < (1 << n); ++ i )
            for (int j = 0; j < n; ++ j )
                if ((i & (1 << j))) {
                    int left = i - (1 << j);
                    tot[i] = tot[left] + jobs[j];
                    break;
                }
        */
        
        vector<int> f(1 << n, INT_MAX / 2);
        f[0] = 0;
        for (int j = 1; j <= k; ++ j )
            for (int mask = (1 << n) - 1; mask; -- mask)
                // 枚举子集
                for (int sub = mask; sub; sub = (sub - 1) & mask)
                    f[mask] = min(f[mask], max(f[mask ^ sub], tot[sub]));
        return f[(1 << n) - 1];
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

> [!NOTE] **[LeetCode 1879. 两个数组最小的异或值之和](https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数据范围 n = 14
> 
> dp 或 爆搜
> 
> > 标准的 n! => 2^n 模型
> > 
> > 每一步只关心前面用过哪些数&最小和多少 不关心具体顺序 ==> 考虑状态压缩
> > 
> > a[s] 对应第 s+1 个数字
> > 
> > a[s-1] 对应第 s 个数字

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int minimumXORSum(vector<int>& a, vector<int>& b) {
        int n = a.size();
        vector<int> f(1 << n, 1e9);
        f[0] = 0;
        for (int mask = 0; mask < (1 << n); ++ mask ) {
            int s = 0;
            for (int i = 0; i < n; ++ i )
                if (mask >> i & 1)
                    s ++ ;
            // 枚举下一步走哪
            for (int i = 0; i < n; ++ i )
                if (!(mask >> i & 1))
                    f[mask | (1 << i)] = min(f[mask | (1 << i)], f[mask] + (a[s] ^ b[i]));
        }
        return f[(1 << n) - 1];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int minimumXORSum(vector<int>& a, vector<int>& b) {
        int n = a.size();
        vector<int> f(1 << n, 1e9);
        f[0] = 0;
        for (int mask = 0; mask < (1 << n); ++ mask ) {
            int s = 0;
            for (int i = 0; i < n; ++ i )
                if (mask >> i & 1)
                    s ++ ;
            // 枚举上一步从哪走来
            for (int i = 0; i < n; ++ i )
                if (mask >> i & 1)
                    f[mask] = min(f[mask], f[mask ^ (1 << i)] + (a[s - 1] ^ b[i]));
        }
        return f[(1 << n) - 1];
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

> [!NOTE] **[Codeforces Little Pony and Harmony Chest](http://codeforces.com/problemset/problem/453/B)**
> 
> 题意: 
> 
> 求满足题意要求的和谐序列 $b$

> [!TIP] **思路**
> 
> 转化：$b$ 中任意元素无公共质因子 （而非每个数都是质数）
> 
> 这题较 trick 的点在于，枚举状态本身表示因子的使用与否，而非具体的数值
> 
> **非常好的状态定义与转移练习**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Little Pony and Harmony Chest
// Contest: Codeforces - Codeforces Round #259 (Div. 1)
// URL: https://codeforces.com/problemset/problem/453/B
// Memory Limit: 256 MB
// Time Limit: 4000 ms

#include <bits/stdc++.h>
using namespace std;

// 转化：b中每个数之间都没有公共质因子（而非每个数都是质数）
// ps 30 的数据范围说明所有数字不会超过59 否则必然不如放1
// 也因此最多只有16个质因子 最大为53

const static int N = 110, M = 16;

int primes[M], cnt = 0;
bool st[M << 2];
int table[M << 2];  // ATTENTION 能整除i的质数组成的集合
void init() {
    cnt = 0;
    memset(st, 0, sizeof st);
    int top = 53;
    for (int i = 2; i <= top; ++i) {
        if (!st[i])
            primes[cnt++] = i;
        for (int j = 0; primes[j] <= top / i; ++j) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0)
                break;
        }
    }
    // cnt = 16
    // 预处理table 以在转移时o1判断
    for (int i = 1; i < M << 2; ++i)
        for (int j = 0; j < cnt; ++j)
            if (i % primes[j] == 0)
                table[i] += (1 << j);
}

int n, a[N];
int f[N][1 << M], pre[N][1 << M];

void output(int x, int y) {
    if (!x)
        return;
    output(x - 1, y ^ table[pre[x][y]]);
    cout << pre[x][y] << ' ';
}

int main() {
    init();

    cin >> n;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    memset(pre, 0, sizeof pre);
    memset(f, 0x3f, sizeof f);
    for (int i = 0; i < 1 << M; ++i)
        f[0][i] = 0;	// ATTENTION 这样写方便最后output直接用(1<<M)-1
    for (int i = 1; i <= n; ++i)
        for (int j = 0; j < 1 << M; ++j)
            for (int k = 1; k <= 58; ++k) {
                if ((table[k] | j) != j)
                    continue;
                int t = abs(k - a[i]) + f[i - 1][j ^ table[k]];
                if (t < f[i][j])
                    f[i][j] = t, pre[i][j] = k;
            }
    output(n, (1 << M) - 1);
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

> [!NOTE] **[LeetCode 847. 访问所有节点的最短路径](https://leetcode.cn/problems/shortest-path-visiting-all-nodes/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始在想图论，实际上这个数据范围直接状压枚举再跑 BFS 即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 12;

    int f[1 << N][N];

    int shortestPathLength(vector<vector<int>>& graph) {
        memset(f, 0x3f, sizeof f);
        int n = graph.size();
        queue<PII> q;
        for (int i = 0; i < n; ++ i )
            f[1 << i][i] = 0, q.push({1 << i, i});
        
        while (q.size()) {
            auto [st, u] = q.front(); q.pop();

            for (auto v : graph[u]) {
                int nst = st | 1 << v;
                if (f[nst][v] > f[st][u] + 1) {
                    f[nst][v] = f[st][u] + 1;
                    q.push({nst, v});
                }
            }
        }
        
        int res = 0x3f3f3f3f;
        for (int i = 0; i < n; ++ i )
            res = min(res, f[(1 << n) - 1][i]);
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

> [!NOTE] **[LeetCode 3149. 找出分数最低的排列](https://leetcode.cn/problems/find-the-minimum-cost-array-permutation/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 状压DP+具体方案
> 
> 具体方案/记录路径 trick: 当值相同时 记录字典序最小路径的方式 (路径编码为 LL)
> 
> 以及合理推导下的代码简化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 原版**

```cpp
class Solution {
public:
    // 根据数据范围 猜测是状态压缩(通过状压枚举操作顺序)
    using LL = long long;
    const static int N = 17000, M = 14, INF = 0x3f3f3f3f;
    
    int f[N][M]; // 2^14 = 1024*16
    LL p[N][M];
    
    int n;
    
    vector<int> get(int st, int i) {
        LL x = p[st][i];
        vector<int> ret;
        for (int i = 0; i < n; ++ i )
            ret.push_back(x % n), x /= n;
        reverse(ret.begin(), ret.end());
        return ret;
    }
    
    vector<int> findPermutation(vector<int>& nums) {
        this->n = nums.size();
        
        int val = INF;
        vector<int> res;
        for (int start = 0; start < n; ++ start ) {
            memset(f, 0x3f, sizeof f);
            memset(p, 0, sizeof p);
            f[1 << start][start] = 0;
            
            for (int i = 1; i < 1 << n; ++ i ) {
                if (!(i >> start & 1))
                    continue;
                
                if (i == 1 << start) {
                    for (int k = 0; k < n; ++ k )
                        if (k != start) {
                            int t = f[i][start] + abs(start - nums[k]);
                            int st = i | (1 << k);
                            if (f[st][k] > t) {
                                f[st][k] = t;
                                p[st][k] = start * n + k;
                            }
                        }
                    continue;
                }

                for (int j = 0; j < n; ++ j )
                    if (j != start && i >> j & 1)
                        for (int k = 0; k < n; ++ k )
                            if (!(i >> k & 1)) {
                                int t = f[i][j] + abs(j - nums[k]);
                                LL last = p[i][j];
                                int st = i | (1 << k);
                                if (f[st][k] > t || f[st][k] == t && p[st][k] > last * n + k) {
                                    f[st][k] = t;
                                    p[st][k] = last * n + k;
                                }
                            }
            }
            
            // 默认字典序最小
            for (int i = 0; i < n; ++ i ) {
                int t = f[(1 << n) - 1][i] + abs(i - nums[start]);
                auto perm = get((1 << n) - 1, i);
                if (val > t || val == t && res > perm)
                    res = perm, val = t;
            }
            
            // break;
        }
        return res;
    }
};
```

##### **C++ 简化**

```cpp
class Solution {
public:
    // 根据数据范围 猜测是状态压缩(通过状压枚举操作顺序)
    using LL = long long;
    const static int N = 17000, M = 14, INF = 0x3f3f3f3f;
    
    int f[N][M]; // 2^14 = 1024*16
    LL p[N][M];
    
    int n;
    
    vector<int> get(int st, int i) {
        LL x = p[st][i];
        vector<int> ret;
        for (int i = 0; i < n; ++ i )
            ret.push_back(x % n), x /= n;
        reverse(ret.begin(), ret.end());
        return ret;
    }
    
    vector<int> findPermutation(vector<int>& nums) {
        this->n = nums.size();
        
        int val = INF;
        vector<int> res;
        for (int start = 0; start < n; ++ start ) {
            memset(f, 0x3f, sizeof f);
            memset(p, 0, sizeof p);
            f[1 << start][start] = 0;
            
            for (int i = 1; i < 1 << n; ++ i ) {
                if (!(i >> start & 1))
                    continue;
                
                for (int j = 0; j < n; ++ j )
                    if (i >> j & 1)
                        for (int k = 0; k < n; ++ k )
                            if (!(i >> k & 1)) {
                                int t = f[i][j] + abs(j - nums[k]);
                                LL last = p[i][j];
                                int st = i | (1 << k);
                                if (f[st][k] > t || f[st][k] == t && p[st][k] > last * n + k) {
                                    f[st][k] = t;
                                    p[st][k] = last * n + k;
                                }
                            }
            }
            
            // 默认字典序最小
            for (int i = 0; i < n; ++ i ) {
                int t = f[(1 << n) - 1][i] + abs(i - nums[start]);
                auto perm = get((1 << n) - 1, i);
                if (val > t || val == t && res > perm)
                    res = perm, val = t;
            }
            
            // 进一步简化 (结合题意 与首位无关 则答案一定是以 0 为起点开始的)
            break;
        }
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

### 三进制状压

**TODO: 多进制统一整理代码风格 fix**

对于三进制的状态压缩题目，可以考虑使用 `limit << 1` 的形式也可以使用 `pow(3, M)` 的形式

> [!NOTE] **[LeetCode 1774. 最接近目标价格的甜点成本](https://leetcode.cn/problems/closest-dessert-cost/)**
> 
> [weekly-230](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-02-28_Weekly-230)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 每种基料必选，配料最多选两个，用四进制来表示三进制 00 / 01 / 10，四进制枚举时 11 为不合法状态

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    int closestCost(vector<int>& baseCosts, vector<int>& toppingCosts, int target) {
        int res = INT_MAX;
        int n = baseCosts.size(), m = toppingCosts.size();
        for (int i = 0; i < n; ++ i ) {
            // 尝试以 i 为基料
            int s = baseCosts[i];

            // 状压枚举基料的选择
            int lim = pow(3, m);
            for (int j = 0; j < lim; ++ j ) {
                int t = s;
                // 对应的 j 下每种配料的选择情况 (0/1/2)
                for (int k = 0, p = 1; k < m; ++ k , p *= 3 ) {
                    // ATTENTION 先获取前缀 再对三取模 来得到对应位置的值
                    int prefix = j / p; // ATTENTION 除法
                    int x = prefix % 3;
                    t += toppingCosts[k] * x;
                }
                if (abs(t - target) < abs(res - target) || abs(t - target) == abs(res - target) && t < res)
                    res = t;
            }
        }
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int closestCost(vector<int>& a, vector<int>& b, int T) {
        int res = INT_MAX;
        int n = a.size(), m = b.size();
        for (int i = 0; i < n; ++ i ) {
            int s = a[i];
            // 四进制来表示三进制
            for (int j = 0; j < 1 << m * 2; ++ j ) {
                int r = s;
                bool flag = false;
                for (int k = 0; k < m; ++ k ) {
                    int t = j >> k * 2 & 3;
                    if (t == 3) {
                        flag = true;
                        break;
                    }
                    r += b[k] * t;
                }
                if (flag) continue;
                if (abs(r - T) < abs(res - T) || abs(r - T) == abs(res - T) && r < res)
                    res = r;
            }
        }
        return res;
    }
};
```

##### **C++ 转化01背包**

```cpp
class Solution {
public:
    int closestCost(vector<int>& baseCosts, vector<int>& toppingCosts, int target) {
        int tt = 20000;
        vector<bool> f(tt + 1, false);
        for (int x: baseCosts)
            f[x] = true;
        for (int x: toppingCosts)
            for (int j = tt; j >= x; --j)
                if (f[j - x])
                    f[j] = true;
        for (int x: toppingCosts)
            for (int j = tt; j >= x; --j)
                if (f[j - x])
                    f[j] = true;
        int ans = tt;
        for (int i = 0; i <= tt; ++i)
            if (f[i])
                if (abs(i - target) < abs(ans - target))
                    ans = i;
        return ans;
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

> [!NOTE] **[LeetCode 1931. 用三种不同颜色为网格涂色](https://leetcode.cn/problems/painting-a-grid-with-three-different-colors/)**
> 
> [weekly-249](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-07-11_Weekly-249)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp思路简单版 [1411. 给 N x 3 网格图涂色的方案数](https://leetcode.cn/problems/number-of-ways-to-paint-n-3-grid/)
> 
> 之前想用 01 10 11 表示三进制状态，但实现起来非常麻烦
> 
> 直接用 pow(3, k) 来处理此类三进制经典问题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    // 3^5 = 81 * 3 = 243
    const static int N = 1010, M = 250, MOD = 1e9 + 7;

    int f[N][M];

    int colorTheGrid(int m, int n) {
        int lim = pow(3, m);
        vector<int> valid;
        for (int i = 0; i < lim; ++ i ) {
            int last = -1;
            bool flag = true;
            for (int j = 0, p = 1; j < m; ++ j, p *= 3 ) {
                int t = i / p % 3;
                if (t == last) {
                    flag = false;
                    break;
                }
                last = t;
            }
            if (flag)
                valid.push_back(i);
        }

        int sz = valid.size();
        unordered_map<int, vector<int>> es;
        for (int i = 0; i < sz; ++ i ) {
            vector<int> e;
            for (int j = 0; j < sz; ++ j ) {
                bool flag = true;
                for (int k = 0, p = 1; k < m; ++ k, p *= 3 ) {
                    int t1 = valid[i] / p % 3, t2 = valid[j] / p % 3;
                    if (t1 == t2) {
                        flag = false;
                        break;
                    }
                }
                if (flag)
                    e.push_back(valid[j]);
            }
            es[valid[i]] = e;
        }

        memset(f, 0, sizeof f);
        for (auto x : valid)
            f[1][x] = 1;
        
        for (int i = 2; i <= n; ++ i )
            for (auto j : valid)
                for (auto k : es[j])
                    f[i][j] = (f[i][j] + f[i - 1][k]) % MOD;

        int res = 0;
        for (auto x : valid)
            res = (res + f[n][x]) % MOD;
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1010, M = 250;
    const int MOD = 1e9 + 7;

    int n, m;
    vector<int> st;
    unordered_map<int, vector<int>> g;
    LL f[N][M];

    bool check(int x) {
        int last = -1;
        for (int i = 0; i < m; ++ i ) {
            if (x % 3 == last)
                return false;
            last = x % 3;
            x /= 3;
        }
        return true;
    }

    bool match(int a, int b) {
        for (int i = 0; i < m; ++ i ) {
            if (a % 3 == b % 3)
                return false;
            a /= 3, b /= 3;
        }
        return true;
    }

    int colorTheGrid(int m, int n) {
        this->n = n, this->m = m;

        int lim = pow(3, m);

        for (int i = 0; i < lim; ++ i )
            if (check(i))
                st.push_back(i);
        for (auto a : st)
            for (auto b : st)
                if (match(a, b))
                    g[a].push_back(b);

        for (auto v : st)
            f[1][v] = 1;
        for (int i = 2; i <= n; ++ i )
            for (auto j : st)
                for (auto k : g[j])
                    f[i][j] = (f[i][j] + f[i - 1][k]) % MOD;

        int res = 0;
        for (auto v : st)
            res = (res + f[n][v]) % MOD;
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

> [!NOTE] **[LeetCode 1659. 最大化网格幸福感](https://leetcode.cn/problems/maximize-grid-happiness/)** [TAG]
> 
> [weekly-215](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2020-11-15_Weekly-215)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> m 行 n 列的网格，选一部分人放进去。n 列的状态可用三进制枚举表述。pow 随后计算递推即可。
> 
> 经典题，类似于下面的不同颜色涂色
> 
> 单个位置三进制表示
> 
> TODO 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// newhar
class Solution {
public:
    int getMaxGridHappiness(int m, int n, int a, int b) {
        // 0- 不放人 1-放内向 2-放外向 3^n
        int cases = pow(3, n);

        int f[cases][5];
        memset(f, 0, sizeof(f));
        for (int i = 0; i < cases; ++i) {
            for (int t = i, p = 0; t; t /= 3, p++) { f[i][n - 1 - p] = t % 3; }
        }

        int M = cases - 1;
        int dp[m + 1][n][a + 1][b + 1][cases];
        memset(dp, 0, sizeof(dp));

        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                int nei = i, nej = j + 1;
                if (j == n) { nei = i + 1, nej = 0; }
                for (int x = 0; x <= a; ++x) {
                    for (int y = 0; y <= b; ++y) {
                        for (int pre = 0; pre < cases; ++pre) {
                            int nem = (pre * 3) % cases;
                            if (x > 0) {
                                int diff = 120;
                                if (j != 0 && f[pre][n - 1] == 1) {
                                    diff -= 30;
                                    diff -= 30;
                                }
                                if (j != 0 && f[pre][n - 1] == 2) {
                                    diff += 20;
                                    diff -= 30;
                                }
                                if (f[pre][0] == 1) {
                                    diff -= 30;
                                    diff -= 30;
                                }
                                if (f[pre][0] == 2) {
                                    diff += 20;
                                    diff -= 30;
                                }
                                dp[i][j][x][y][pre] =
                                    max(dp[i][j][x][y][pre],
                                        diff + dp[nei][nej][x - 1][y][nem + 1]);
                            }
                            if (y > 0) {
                                int diff = 40;
                                if (j != 0 && f[pre][n - 1] == 1) {
                                    diff -= 30;
                                    diff += 20;
                                }
                                if (j != 0 && f[pre][n - 1] == 2) {
                                    diff += 20;
                                    diff += 20;
                                }
                                if (f[pre][0] == 1) {
                                    diff -= 30;
                                    diff += 20;
                                }
                                if (f[pre][0] == 2) {
                                    diff += 20;
                                    diff += 20;
                                }
                                dp[i][j][x][y][pre] =
                                    max(dp[i][j][x][y][pre],
                                        diff + dp[nei][nej][x][y - 1][nem + 2]);
                            }
                            dp[i][j][x][y][pre] = max(dp[i][j][x][y][pre],
                                                      dp[nei][nej][x][y][nem]);
                        }
                    }
                }
            }
        }
        return dp[0][0][a][b][0];
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

> [!NOTE] **[LeetCode 2172. 数组的最大与和](https://leetcode.cn/problems/maximum-and-sum-of-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **`f[i - 1][mask - 3^(k-1)]`, 后者表示将第k个篮子对应三进制减一**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    // ATTENTION 并非考虑每一个篮子 而是考虑当前每一个数
    // => 压缩篮子的状态
    const static int N = 19, M = 2e4;   // 3^9 < 2e4

    int f[N][M];    // 前i个放完后 所有篮子的【剩余可放】状态为 j 的最大与和

    int maximumANDSum(vector<int>& nums, int numSlots) {
        int n = nums.size(), m = numSlots;
        memset(f, 0, sizeof f);

        int lim = pow(3, m);
        for (int i = 1; i <= n; ++ i )
            for (int j = 0; j < lim; ++ j )
                // 第 i 个数放到第 k 个篮子
                for (int k = 1, p = 1; k <= m; ++ k , p *= 3 ) {
                    int t = j / p % 3;
                    if (t != 0) {   // 还可以放
                        // ATTENTION j-p 意味着消耗掉当前篮子的一个空位
                        f[i][j] = max(f[i][j], f[i - 1][j - p] + (k & nums[i - 1]));    // nums[i - 1]
                    }
                }
        return f[n][lim - 1];
    }
};
```


##### **C++**

```cpp
class Solution {
public:
    const static int N = 19, M = 2e4;   // 2e4 -> 3^9
    
    // 考虑前 i 个整数, 篮子可用状态是 j 的最大与和
    int f[N][M];
    
    int maximumANDSum(vector<int>& nums, int numSlots) {
        int n = nums.size(), m = pow(3, numSlots);
        memset(f, 0, sizeof f);
        for (int i = 1; i <= n; ++ i )
            for (int j = 0; j < M; ++ j )
                // 第 i 个数放到第 k 个篮子里, 对应状态范围为 j / w % 3
                for (int k = 1, w = 1; k <= numSlots; ++ k , w *= 3 )
                    // j / w % 3 != 0 说明还可以放
                    if (j / w % 3)
                        // ATTENTION: f[i - 1][mask - 3^(k-1)], 后者表示将第k个篮子对应三进制减一
                        f[i][j] = max(f[i][j], f[i - 1][j - w] + (k & nums[i - 1]));
        return f[n][m - 1];
    }
};
```

##### **C++ 空间压缩**

```cpp
// 空间压缩
class Solution {
public:
    const static int N = 19, M = 2e4;   // 2e4 -> 3^9
    
    int f[M];
    
    int maximumANDSum(vector<int>& nums, int numSlots) {
        int n = nums.size(), m = pow(3, numSlots);
        memset(f, 0, sizeof f);
        for (int i = 1; i <= n; ++ i )
            for (int j = M - 1; j >= 0; -- j )
                for (int k = 1, w = 1; k <= numSlots; ++ k , w *= 3 )
                    if ((j / w % 3) && j - w >= 0)
                        f[j] = max(f[j], f[j - w] + (k & nums[i - 1]));
        return f[m - 1];
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

### 八进制状压

> [!NOTE] **[LeetCode 638. 大礼包](https://leetcode.cn/problems/shopping-offers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 7进制背包 记忆化
> 
> 枚举大礼包和当前剩余需求

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 本质是一个完全背包问题
    // 最多有六个维度 大礼包个物品
    // 直接写 7*10^7 容易超时

    // needs 改为八进制数 方便位运算 避免高维数组
    
    int n;
    vector<vector<int>> f;
    vector<int> price;
    vector<vector<int>> special;

    int dp(int x, int y) {
        if (f[x][y] != -1) return f[x][y];
        if (!x) {
            // 剩余物品散装买
            f[x][y] = 0;
            for (int i = 0; i < n; ++ i ) {
                int c = y >> i * 3 & 7;
                f[x][y] += c * price[i];
            }
            return f[x][y];
        }
        // 不买当前大礼包
        f[x][y] = dp(x - 1, y);
        // 买当前大礼包
        int state = 0;
        // 当前大礼包是 s 
        auto s = special[x - 1];
        for (int i = n - 1; i >= 0; -- i ) {
            int c = y >> i * 3 & 7;
            if (c < s[i]) {
                // 个数超
                state = -1;
                break;
            }
            state = state * 8 + c - s[i];
        }
        if (state != -1)
            // 买后 dp(x, state) + s.back() 其中s.back()大礼包价格
            f[x][y] = min(f[x][y], dp(x, state) + s.back());
        return f[x][y];
    }

    int shoppingOffers(vector<int>& price, vector<vector<int>>& special, vector<int>& needs) {
        this->price = price;
        this->special = special;
        n = price.size();
        f = vector<vector<int>>(special.size() + 1, vector<int>(1 << n * 3, -1));
        int state = 0;
        for (int i = needs.size() - 1; i >= 0; -- i )
            state = state * 8 + needs[i];
        return dp(special.size(), state);
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

### 状压 + 记忆化搜索

> [!NOTE] **[LeetCode 1900. 最佳运动员的比拼回合](https://leetcode.cn/problems/the-earliest-and-latest-rounds-where-players-compete/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二进制枚举状态类似博弈
> 
> 显然二进制枚举搜索即可
> 
> 比赛时忘了加记忆化 TLE 数次... 加行记忆化就过...
> 
> 加强搜索敏感度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, p1, p2;
    int minr, maxr;
    
    unordered_set<int> S;
    
    void dfs(int st, int d) {
        if (S.count(st))
            return;
        S.insert(st);
        
        int sz = __builtin_popcount(st);
        if (sz < 2)
            return;
        
        int cp = sz / 2;
        
        vector<int> ve;
        for (int i = 0; i < n; ++ i )
            if (st >> i & 1)
                ve.emplace_back(i);
        
        for (int i = 0; i < cp; ++ i )
            if (ve[i] + 1 == p1 && ve[sz - i - 1] + 1 == p2 || ve[i] + 1 == p2 && ve[sz - i - 1] + 1 == p1) {
                minr = min(minr, d), maxr = max(maxr, d);
                return;
            }
        
        // 某位为1则对应前半部分被淘汰
        for (int i = 0; i < 1 << cp; ++ i ) {
            int t = st;
            for (int j = 0; j < cp; ++ j )
                if (i >> j & 1)
                    t ^= 1 << ve[j];
                else
                    t ^= 1 << ve[sz - j - 1];
            if ((t >> (p1 - 1) & 1) == 0 || (t >> (p2 - 1) & 1) == 0)
                continue;
            
            dfs(t, d + 1);
        }
    }
    
    vector<int> earliestAndLatest(int n, int firstPlayer, int secondPlayer) {
        this->n = n, this->p1 = firstPlayer, this->p2 = secondPlayer;
        minr = 2e9, maxr = -2e9;
        
        dfs((1 << n) - 1, 1);
        
        return {minr, maxr};
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

### 状压 + 枚举子集 + 组合计数

> [!NOTE] **[LeetCode 1994. 好子集的数目](https://leetcode.cn/problems/the-number-of-good-subsets/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> [biweekly-60](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-09-04_Biweekly-60)
> 
> 非常好的状态压缩 + 组合计数应用题
> 
> 按比特位划分是为了去重，详情参考第三份代码
> 
> **枚举子集时依靠某位bit位将所有子集分为两类，从而实现去重计数**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 10;
    const int MOD = 1e9 + 7;
    
    vector<int> ps = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    
    int numberOfGoodSubsets(vector<int>& nums) {
        vector<int> cnt(1 << N), sum(1 << N);
        // 1. 求每个数分解得到全素数集合，该全素数集合对应的原始数的个数
        // 本质上是将原始数组中合法的部分重新统计一遍
        for (auto x : nums) {
            int t = 0;
            for (int i = 0; i < N; ++ i )
                if (x % ps[i] == 0) {
                    t |= 1 << i;
                    // ATTENTION: 如果x包含两个p[i]，必然无效
                    if (x / ps[i] % ps[i] == 0) {
                        t = -1;
                        break;
                    }
                }
        
            if (t != -1)
                cnt[t] ++ ;
        }
        
        // 2. 遍历统计
        // 为什么可以遍历到 1 << N ? 因为按照合法性规则，最多包含ps中所有的元素 即2^10
        int res = 0;
        for (int i = 1; i < 1 << N; ++ i ) {
            // 该集合本身作为一个数
            sum[i] = cnt[i];
            // 集合拆分
            // ATTENTION 思考细节：【为什么实现需找第一个存在位来划分】
            // 【目的：去重】如果直接用子集去算会有重合部分
            for (int j = 0; j < N; ++ j )
                if (i >> j & 1) {
                    int k = i ^ (1 << j);
                    // 枚举k的子集（必然包含j的子集）作为第一部分【cnt[(1 << j) | x]】
                    // 其补集作为第二部分【sum[k ^ x]】
                    for (int x = (k - 1) & k; true; x = (x - 1) & k) {
                        sum[i] = (sum[i] + (LL)cnt[(1 << j) | x] * sum[k ^ x]) % MOD;
                        if (x == 0) // 注意 x 可以为空集，即第一部分只包含一个j
                            break;
                    }
                    // ATTENTION 能够根据一位不同去划分为两类即可
                    break;
                }
            // 统计
            res = (res + (LL)sum[i]) % MOD;
        }
        // 1的数量 任意选 2^cnt[0]
        for (int i = 0; i < cnt[0]; ++ i )
            res = (res + (LL)res) % MOD;
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 10;
    const int MOD = 1e9 + 7;
    
    vector<int> ps = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    
    int qpow(int a, int b) {
        LL res = 1;
        while (b) {
            if (b & 1)
                res = res * a % MOD;
            a = (LL)a * a % MOD;
            b >>= 1;
        }
        return res;
    }
    
    int numberOfGoodSubsets(vector<int>& nums) {
        vector<int> cnt(1 << N), sum(1 << N);
        // 1. 求每个数分解得到全素数集合，该全素数集合对应的原始数的个数
        // 本质上是将原始数组中合法的部分重新统计一遍
        for (auto x : nums) {
            int t = 0;
            for (int i = 0; i < N; ++ i )
                if (x % ps[i] == 0) {
                    t |= 1 << i;
                    // ATTENTION: 如果x包含两个p[i]，必然无效
                    if (x / ps[i] % ps[i] == 0) {
                        t = -1;
                        break;
                    }
                }
        
            if (t != -1)
                cnt[t] ++ ;
        }
        
        // 2. 遍历统计
        sum[0] = qpow(2, cnt[0]);
        
        for (int i = 1; i < 1 << N; ++ i ) {
            // ATTENTION 1: 只用sum不用nxt也可以过，因为是从小到大计算的
            auto nxt = sum;
            for (int j = 0; j < 1 << N; ++ j )
                if (!(i & j)) {
                    // i 与 j 无交叉
                    nxt[j | i] = (nxt[j | i] + (LL)sum[j] * cnt[i]) % MOD;
                }
            sum = nxt;
        }
        
        // ATTENTION 2: 这样res统计必须放在最后，因为前面for-loop未将当前结果统计完成
        int res = 0;
        for (int i = 1; i < 1 << N; ++ i )
            res = (res + (LL)sum[i]) % MOD;
        
        return res;
    }
};
```

##### **C++ 3**

```cpp
class Solution {
public:
    // ...
    int numberOfGoodSubsets(vector<int>& nums) {
        vector<int> cnt(1 << N), sum(1 << N);
        for (auto x : nums) {
            // ...
            if (t != -1)
                cnt[t] ++ ;
        }
        
        sum[0] = qpow(2, cnt[0]);
        
        for (int i = 1; i < 1 << N; ++ i ) {
            int f = i & (-i);  // bit划分
            
            for (int j = i; j; j = (j - 1) & i) {
                // 会计算重复
                // DEBUG
                // if (cnt[j] && sum[i ^ j]) {
                //     cout << " i = " << i << " j = " << j << " cnt[j] = " << cnt[j] << " sum[i ^ j] = " << sum[i ^ j] << endl;
                // }
                // 去重
                if (j & f)
                    sum[i] = (sum[i] + (LL)cnt[j] * sum[i ^ j]) % MOD;
            }
            
            // if (sum[i])
            //     cout << " sum[" << i << "] = " << sum[i] << endl;
        }
        
        int res = 0;
        for (int i = 1; i < 1 << N; ++ i )
            res = (res + (LL)sum[i]) % MOD;
        
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

> [!NOTE] **[LeetCode 合作开发](https://leetcode.cn/contest/cnunionpay-2022spring/problems/lCh58I/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 状压 + 数学
> 
> 注意 **枚举非空非全真子集 + 数学计算细节TODO**
> 
> 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int MOD = 1e9 + 7;
    
    LL p[4] = {1, 1000, 1000000, 1000000000};
    
    int coopDevelop(vector<vector<int>>& skills) {
        int n = skills.size();
        map<LL, LL> hash;
        for (auto & s : skills) {
            sort(s.begin(), s.end());
            // encode
            LL st = 0;
            for (int i = 0; i < s.size(); ++ i )
                st += s[i] * p[i];
            hash[st] ++ ;
        }
        
        LL res = (LL)n * (n - 1) / 2;
        for (auto [k, v] : hash) {
            // decode
            vector<int> ve;
            for (LL i = 0, j = k; j; ++ i , j /= 1000 )
                ve.push_back(j % 1000);
            
            int m = ve.size();
            // 非空非全 真子集
            for (int i = 1; i < (1 << m) - 1; ++ i ) {
                LL st = 0, c = 0;
                for (LL j = 0; j < m; ++ j )
                    if (i >> j & 1)
                        st += ve[j] * p[c ++ ];
                
                if (hash.count(st))
                    res -= hash[st] * v;
            }
            res -= v * (v - 1) / 2;
        }
        return res % MOD;
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

### 状压 + meet in the middle

> [!NOTE] **[LeetCode 2035. 将数组分成两个数组并最小化数组和的差](https://leetcode.cn/problems/partition-array-into-two-arrays-to-minimize-sum-difference/)**
> 
> [weekly-262](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-10-10_Weekly-262)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典的数据范围较大 如果直接对全局状压TLE 故【拆分成两个部分】再【扫描一半并二分另一半】
> 
> 理应想到 折半枚举（形如折半双向搜索） 的做法
> 
> - 增强对 meet in the middle 的敏感度
> - 学习优雅的 STL 用法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 16;
    vector<vector<int>> s1, s2;
    
    void work(vector<int> & t, vector<vector<int>> & r) {
        int n = t.size();
        for (int i = 0; i < 1 << n; ++ i ) {
            int c = 0, s = 0;
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    c ++ , s += t[j];
                // else
                else
                    s -= t[j];
            r[c].push_back(s);
        }
        for (int i = 0; i < N; ++ i )
            sort(r[i].begin(), r[i].end());
    }
    
    // 折半拆分的思想!
    int minimumDifference(vector<int>& nums) {
        s1.resize(N), s2.resize(N);
        int n = nums.size() / 2;
        {
            vector<int> t;
            for (int i = 0; i < n; ++ i )
                t.push_back(nums[i]);
            work(t, s1);
        }
        {
            vector<int> t;
            for (int i = n; i < n << 1; ++ i )
                t.push_back(nums[i]);
            work(t, s2);
        }
        
        int res = 2e9;
        for (int i = 0; i < n; ++ i ) {
            // ls rs 分别 i n-i 个元素的所有集合
            auto ls = s1[i];
            auto rs = s2[n - i];
            for (auto v : ls) {
                auto it = lower_bound(rs.begin(), rs.end(), -v);
                // ATTENTION 学习这种STL用法
                if (it != rs.end())
                    res = min(res, abs(v + *it));
                if (it != rs.begin())
                    res = min(res, abs(v + *prev(it)));
            }
        }
        
        
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

> [!NOTE] **[LeetCode 1755. 最接近目标值的子序列和](https://leetcode.cn/problems/closest-subsequence-sum/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> `n = 40` 的复杂度考虑拆半再二进制枚举
> 
> 可以分别计算两个部分的可能和，再双指针
> 
> 也可以先计算第一部分，再计算第二个部分的时候二分查找

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minAbsDifference(vector<int>& nums, int goal) {
        int n = nums.size();
        int m = n >> 1;
        
        vector<int> h;
        for (int s = 0; s < 1 << m; ++ s ) {
            int t = 0;
            for (int i = 0; i < m; ++ i )
                if (s >> i & 1) 
                    t += nums[i];
            h.push_back(t);
        }
        sort(h.begin(), h.end());
        
        int ans = INT_MAX;
        for (int s = 0; s < (1 << n - m); ++ s ) {
            int t = goal;
            for (int i = 0; i < n - m; ++ i )
                if (s >> i & 1)
                    t -= nums[i + m];
            
            int l = 0, r = h.size() - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (h[mid] < t) l = mid + 1;
                else r = mid;
            }
            
            if (ans > abs(h[l] - t)) ans = abs(h[l] - t);
            if (l >= 1 && ans > abs(h[l - 1] - t)) ans = abs(h[l - 1] - t);
        }
        return ans;
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

> [!NOTE] **[LeetCode 982. 按位与为零的三元组](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始想的是按值去重分类，然后预处理当前值会对那些集合产生影响。需要考虑去重等非常繁琐。
> 
> 放弃，其实根据数据范围可以直接两两与
> 
> **类似 meet in the middle** 的思想

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countTriplets(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> hash;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < n; ++ j )
                hash[nums[i] & nums[j]] ++ ;
        int res = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < 1 << 16; ++ j )
                if (!(j & nums[i]))
                    res += hash[j];
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

### bitset

> [!NOTE] **[Luogu 砝码称重](https://www.luogu.com.cn/problem/P1441)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典
> 
> 状压 + bitset优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// bitset 加速简化

const int N = 22, M = 2010;

int n, m;
int w[N];

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; ++ i )
        cin >> w[i];
    
    int res = 0;
    for (int i = 0; i < 1 << n; ++ i )
        if (__builtin_popcount(i) == n - m) {
            bitset<M> s;
            s[0] = 1;
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    // ATTENTION
                    s |= s << w[j];
            // 把第0位的1去掉
            res = max(res, (int)s.count() - 1);
        }
    
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

### 复杂状态定义

> [!NOTE] **[Luogu P2157 [SDOI2009] 学校食堂](https://www.luogu.com.cn/problem/P2157)** [TAG]
> 
> 题意: 
> 
> $n$ 个人每人有属性值 $a[i]$ 和对后向的约束 $b[i]$
> 
> 顺序可调，但不能违背后向约束（即后面不能有多少个人排在当前人前面），同时消耗为相邻属性异或
> 
> 求最小总消耗

> [!TIP] **思路**
> 
> 考虑当前位置，则既和前面的有关系也和后面的有关系。。。
> 
> 状压: **把当前候选人 $i$ 和其后 7 个人的吃饭状态压缩为 $j$，再加一维表示最后一个干饭的是谁**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const static int N = 1010, M = 1 << 8, K = 18, D = 8, INF = 0x3f3f3f3f;

int n, a[N], b[N];

// 前i-1个人都打完了 第i以及后面7个人打状态为j 当前最后一个打饭的是i+k
//  => k [-8, 7] 加个偏移量 D=8
int f[N][M][K];

void freshmin(int& a, int b) { a = min(a, b); }

int cost(int x, int y) { return x ? a[x] ^ a[y] : 0; }

int main() {
    int T;
    cin >> T;
    while (T--) {
        cin >> n;
        for (int i = 1; i <= n; ++i)
            cin >> a[i] >> b[i];

        memset(f, 0x3f, sizeof f);
        f[1][0][-1 + D] = 0;

        for (int i = 1; i <= n; ++i)
            for (int j = 0; j < M; ++j)
                for (int k = -8; k <= 7; ++k) {
                    if (f[i][j][k + D] >= INF / 2)
                        continue;

                    if (j & 1)
                        // 第i个人打饭完成 可以直接转移
                        freshmin(f[i + 1][j >> 1][k + D - 1], f[i][j][k + D]);
                    else {
                        int lim =
                            INF;  // ATTENTION: 记录当前没有吃饭的人可以容忍的最后位置
                        for (int x = 0; x <= 7; ++x)
                            if (!(j >> x & 1)) {
                                if (i + x > lim)
                                    break;
                                freshmin(lim, i + x + b[i + x]);
                                freshmin(f[i][j | (1 << x)][x + D],
                                         f[i][j][k + D] + cost(i + k, i + x));
                            }
                    }
                }

        int res = INF;
        for (int k = -8; k <= 0; ++k)
            res = min(res, f[n + 1][0][k + D]);
        cout << res << endl;
    }

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

> [!NOTE] **[Luogu P2167 [SDOI2009]Bill的挑战](https://www.luogu.com.cn/problem/P2167)** [TAG]
> 
> 题意: 
> 
> 给出 $N$ 个长度相同的字符串（由小写英文字母和 `?` 组成），$S_1,S_2,\dots,S_N$，求与这 $N$ 个串中的刚好 $K$ 个串匹配的字符串 $T$ 的个数，答案对 $1000003$ 取模。

> [!TIP] **思路**
> 
> 由题意数据范围推测对 `所选取的串的编号` 进行状压
> 
> 显然需要数据预处理，随后递推计算即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const static int N = 55, M = 1 << 15, MOD = 1e6 + 3;

string s[N];
int g[N][N];
int f[N][M];

void modadd(int& a, int b) { a = (a + b % MOD) % MOD; }

int main() {
    int T;
    cin >> T;
    while (T--) {
        int n, k, m;
        cin >> n >> k;
        for (int i = 0; i < n; ++i)
            cin >> s[i];
        m = s[0].size();

        // ATTENTION 状态设计
        memset(g, 0, sizeof g);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < 26; ++j)
                for (int x = 0; x < n; ++x)
                    if (s[x][i] == '?' || s[x][i] == 'a' + j)
                        g[i][j] |= 1 << x;

        memset(f, 0, sizeof f);
        f[0][(1 << n) - 1] = 1;  // 长度0 方案数1
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < 1 << n; ++j)
                if (f[i][j])
                    for (int x = 0; x < 26; ++x)
                        modadd(f[i + 1][j & g[i][x]], f[i][j]);
        int res = 0;
        for (int i = 0; i < 1 << n; ++i)
            if (__builtin_popcount(i) == k)
                modadd(res, f[m][i]);
        cout << res << endl;
    }

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

> [!NOTE] **[Luogu P2150 [NOI2015] 寿司晚宴](https://www.luogu.com.cn/problem/P2150)**
> 
> 题意: 
> 
> 有 $[2,n]$ 一共 $n−1$ 个数，两个人分别取子集 $S$ 和 $T$，要求不存在 $x∈S,y∈T$，使得 $gcd(x,y)!=1$
> 
> $2 <= n <= 500$

> [!TIP] **思路**
> 
> TODO: 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// https://www.luogu.com.cn/blog/iamzxy/solution-p2150

using LL = long long;
const static int N = 510, M = (1 << 8) + 10;

const int primes[8] = {2, 3, 5, 7, 11, 13, 17, 19};

struct Node {
    LL x, big, S;
    bool operator<(const Node& t) const { return big < t.big; }
    void init() {
        LL t = x;
        big = -1, S = 0;
        for (int i = 0; i < 8; ++i) {
            if (t % primes[i] == 0) {
                S |= 1 << i;
                while (t % primes[i] == 0)
                    t /= primes[i];
            }
        }
        // 唯一大质数 sqrt(500) > 20
        if (t > 1)
            big = t;
    }
} nums[N];

LL n, p;
void modadd(LL& a, LL b) { a = (a + b % p) % p; }

LL f[M][M];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> p;
    for (int i = 2; i <= n; ++i)
        nums[i].x = i, nums[i].init();

    // 把大质数相同的放在一起 则可以分成两大块
    sort(nums + 2, nums + n + 1);

    memset(f, 0, sizeof f);
    f[0][0] = 1;

    static LL f1[M][M], f2[M][M];
    for (int i = 2; i <= n; ++i) {
        // 如果都没有大质数或大质数不一样，就可以继承上一层合并的答案。（特判初始化）
        if (i == 1 || nums[i].big ^ nums[i - 1].big || nums[i].big == -1)
            memcpy(f1, f, sizeof f), memcpy(f2, f, sizeof f);

        // 对第一维进行了压缩 所以必须逆序
        for (int j = (1 << 8) - 1; j >= 0; --j)
            for (int k = (1 << 8) - 1; k >= 0; --k)
                if ((j & k) == 0) {
                    int s = nums[i].S;
                    // 集合
                    if ((s & j) == 0)
                        modadd(f1[j][k | s], f1[j][k]);
                    if ((s & k) == 0)
                        modadd(f2[j | s][k], f2[j][k]);
                }

        // 如果都没有大质数或大质数不一样（跟下一层比较），则可以合并答案（特判结束）
        if (i == n || nums[i].big ^ nums[i + 1].big || nums[i].big == -1)
            for (int j = 0; j <= (1 << 8) - 1; ++j)
                for (int k = 0; k <= (1 << 8) - 1; ++k)
                    if ((j & k) == 0)
                        // 减去重复情况
                        f[j][k] = (f1[j][k] + f2[j][k] - f[j][k] + p) % p;
    }

    LL res = 0;
    for (int i = 0; i <= (1 << 8) - 1; ++i)
        for (int j = 0; j <= (1 << 8) - 1; ++j)
            if ((i & j) == 0)
                modadd(res, f[i][j]);
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

> [!NOTE] **[LeetCode 2850. 将石头分散到网格图的最少移动次数](https://leetcode.cn/problems/minimum-moves-to-spread-stones-over-grid/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 状压 重点在于**状态定义**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 3*3 的图是固定的，就极大降低了复杂度，且保证有解
    const static int N = 10, M = 1 << N;
    
    int d[N][N];
    int f[M];
    
    int minimumMoves(vector<vector<int>>& grid) {
        // 考虑枚举: 每一个格子从另一个位置搬过来一个方块，方案是否合法，合法的话记录开销
        // 如何表示状态: 全排列 => no 因为可能是一对多
        // 
        // [ATTENTION] 分别记录缺少的位置 ls 与多出的位置 rs（重要：后者多出几个就作为几个位置）
        // 则 ls.size() == rs.size() 随后即是一个匹配问题 => 状压
        
        vector<int> ls, rs;
        memset(d, 0, sizeof d);
        for (int i = 0; i < 9; ++ i ) {
            int x1 = i / 3, y1 = i % 3;
            for (int j = 0; j < 9; ++ j ) {
                int x2 = j / 3, y2 = j % 3;
                d[i][j] = abs(x1 - x2) + abs(y1 - y2);
            }
            if (grid[x1][y1] == 0)
                ls.push_back(i);
            if (grid[x1][y1] > 1)
                for (int j = 0; j < grid[x1][y1] - 1; ++ j )    // ATTENTION
                    rs.push_back(i);
        }
        
        int n = ls.size();          // ls.size() == rs.size()
        memset(f, 0x3f, sizeof f);  // +inf
        f[0] = 0;
        for (int i = 1; i < 1 << n; ++ i ) {
            int tot = __builtin_popcount(i);
            int a = ls[tot - 1];
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1) {
                    int b = rs[j];
                    f[i] = min(f[i], f[i ^ (1 << j)] + d[a][b]);
                }
        }
        
        return f[(1 << n) - 1];
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