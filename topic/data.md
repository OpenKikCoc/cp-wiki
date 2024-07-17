## 习题

> [!NOTE] **[LeetCode 1198. 找出所有行中最小公共元素](https://leetcode.cn/problems/find-smallest-common-element-in-all-rows/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 并非使用堆。
> 
> 正解：统计每行出现的元素，若某元素数量达 n 则为答案。
> 
> 【关键在于题中所给的数据范围: 所有数值小于 10000 】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int smallestCommonElement(vector<vector<int>>& mat) {
        int n = mat.size(), m = mat[0].size();
        vector<int> cnt(10010);
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (j == 0 || mat[i][j] != mat[i][j - 1])
                    ++ cnt[mat[i][j]];
        for (int i = 1; i <= 10000; ++ i)
            if (cnt[i] == n)
                return i;
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

> [!NOTE] **[LeetCode 1326. 灌溉花园的最少水龙头数目](https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本题限制 `ranges[i] <= 100`
> 
> -   直观做法
>
>     超时 34/35个测试用例 `[输入10000]`
>
>     这种显然在每个区间都很短的情况下很慢 比如超时的用例 [0,1,....1,1,1,0,...]
>
> -   trick 做法
>
> -   dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ TLE**

```cpp
class Solution {
public:
    // 贪心 优先右边界大 右边界相同左边界越小越好
    // （这个位置本身范围为0的水龙头不可用
    int minTaps(int n, vector<int>& ranges) {
        vector<pair<int, int>> v;  // 需要排序 直接用pair<r, 0-l>
        for (int i = 0; i <= n; ++i) {
            if (!ranges[i]) continue;  // 跳过0
            v.push_back({min(ranges[i] + i, n), ranges[i] - i});
        }
        sort(v.begin(), v.end());
        // 随后从后向前
        int p = v.size() - 1, mx = n, res = 0;
        while (p >= 0) {
            ++res;
            int nmx = mx;
            // 从 r >= mx 的选出最靠左的一个来用
            while (p >= 0 && v[p].first >= mx) {
                nmx = min(nmx, -v[p].second);
                --p;
            }
            if (p < 0) {
                if (nmx <= 0)
                    return res;
                else
                    return -1;
            }
            mx = nmx;
            if (mx <= 0) return res;
        }
        return -1;
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    // 考虑：每一个range都很大的情况【当然这道题限定了ranges<=100】
    // 复杂度不会高于 n*100 = 1e6
    int minTaps(int n, vector<int>& ranges) {
        vector<int> v(n + 1);
        for (int i = 0; i <= n; ++i) {
            if (!ranges[i]) continue;  // 跳过0
            int l = max(0, i - ranges[i]), r = min(n, i + ranges[i]);
            while (l < r && v[l] < r) v[l++] = r;
        }
        int res = 0;
        int cur = 0;
        while (cur < n) {
            if (!v[cur]) return -1;
            cur = v[cur];
            ++res;
        }
        return res;
    }
};
```

##### **C++ dp**

```cpp
class Solution {
public:
    int minTaps(int n, vector<int>& ranges) {
        ++n;
        const int INF = 1e9;
        vector<int> f(n + 1, INF);
        f[0] = 0;
        vector<vector<int>> v(n);
        for (int i = 0; i < n; ++i) {
            if (!ranges[i]) continue;
            int l = max(0, i - ranges[i]), r = min(n - 1, i + ranges[i]);
            v[r - 1].push_back(l);
        }
        for (int i = 0; i < n - 1; ++i)
            for (auto j : v[i])
                for (int k = j; k <= i; ++k) f[i + 1] = min(f[i + 1], f[k] + 1);
        return f[n - 1] >= INF ? -1 : f[n - 1];
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

> [!NOTE] **[LeetCode 1766. 互质树](https://leetcode.cn/problems/tree-of-coprimes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 如果用蛮力检查一个节点的所有的祖先节点 那么一个节点的祖先节点最多能有 n−1 个 显然会超时
> 
> 原先首先 dfs 找序并反向建图 随后依据反向图向上搜索
> 
> 这样显然会在祖先极多的时候爆掉
> 
> 需要充分利用值域 ( 50 ) 的条件
> 
> 以及注意 **双向边建图源点遍历时加 fa 即可 无需反向建图**
> 
> TODO 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> g, st;
    vector<int> nums, res, depth;
    
    int gcd(int a, int b) {
        return b ? gcd(b, a % b) : a;
    }
    
    void dfs(int u, int fa) {
        int v = nums[u];
        for (int i = 1; i <= 50; ++ i )
            if (st[i].size() && gcd(v, i) == 1)
                if (res[u] == -1 || depth[res[u]] < depth[st[i].back()])
                    res[u] = st[i].back();
        
        st[v].push_back(u);
        for (auto v : g[u]) {
            if (v == fa) continue;
            depth[v] = depth[u] + 1;
            dfs(v, u);
        }
            
        st[v].pop_back();
    }
    
    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        this->nums = nums;
        int n = nums.size();
        g.resize(n), st.resize(55), res.resize(n, -1), depth.resize(n, 0);
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            g[a].push_back(b), g[b].push_back(a);
        }
        
        dfs(0, -1);
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

> [!NOTE] **[LeetCode 1906. 查询差绝对值的最小值](https://leetcode.cn/problems/minimum-absolute-difference-queries/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 根据数据集范围推测直接使用前缀和统计查询区间中出现的所有数字，去重遍历即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // ATTENTION num[i] <= 100
    const static int N = 1e5 + 10, M = 110;
    int cnt[N][M], t[M], n;
    
    vector<int> minDifference(vector<int>& nums, vector<vector<int>>& queries) {
        n = nums.size();
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= 100; ++ j ) {
                cnt[i][j] = cnt[i - 1][j];
                if (nums[i - 1] == j)
                    cnt[i][j] ++ ;
            }
        
        vector<int> res;
        for (auto & q : queries) {
            int l = q[0], r = q[1];
            vector<int> t;
            for (int i = 1; i <= 100; ++ i )
                if (cnt[r + 1][i] > cnt[l][i])
                    t.push_back(i);
            sort(t.begin(), t.end());
            int sz = t.size(), ret = 2e9;
            for (int i = 1; i < sz; ++ i )
                ret = min(ret, t[i] - t[i - 1]);
            if (ret == 2e9)
                ret = -1;
            res.push_back(ret);
        }
        return res;
    }
};
```

##### **Python**

```python
# python
# 前缀和
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1981. 最小化目标值与所选元素的差](https://leetcode.cn/problems/minimize-the-difference-between-target-and-chosen-elements/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 较显然的是一个分组背包问题
> 
> 直接分组写显然 MLE ，可以滚动数组来做
> 
> 鉴于题目所述数据范围，直接标记某值是否可达即可（背包特性较弱）
> 
> 可 bitset 优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 5000; // 70*70
    
    int n, m;
    vector<vector<int>> g;
    vector<bool> st;
    
    int minimizeTheDifference(vector<vector<int>>& mat, int target) {
        this->g = mat;
        this->n = g.size(), this->m = g[0].size();
        this->st = vector<bool>(N);
        
        st[0] = true;
        for (int i = 0; i < n; ++ i ) {
            auto t = vector<bool>(N);
            for (int j = 0; j < m; ++ j ) {
                int v = g[i][j];
                for (int k = 0; k < N; ++ k )
                    if (st[k] && k + v < N)
                        t[k + v] = true;
            }
            st = t;
        }
        
        // 这一坨显然可以直接一个for-loop... 略
        int res = INT_MAX;
        {
            int p = target;
            while (p && !st[p])
                p -- ;
            if (p)
                res = min(res, target - p);
        }
        {
            int p = target;
            while (p < N && !st[p])
                p ++ ;
            if (p < N)
                res = min(res, p - target);
        }
        return res;
    }
};
```

##### **C++ bitset**

```cpp
// bitset优化
class Solution {
public:
    const static int M = 5000;

    int minimizeTheDifference(vector<vector<int>>& mat, int target) {
        int n = mat.size(), m = mat[0].size();
        bitset<M> b;
        b.set(0);
        for (int i = 0; i < n; ++ i ) {
            bitset<M> t;
            for (auto x : mat[i])
                t |= b << x;
            b = t;
        }
        int res = M;
        for (int i = 0; i < M; ++ i )
            if (b[i])
                res = min(res, abs(i - target));
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

> [!NOTE] **[LeetCode 2188 完成比赛的最少时间](https://leetcode.cn/problems/minimum-time-to-finish-the-race/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 2^10 = 1024 2^14
    // 一个轮胎最多使用 1 * 2^14 跑15圈, 耗时 1e5 * 14 + 2 + 
    using LL = long long;
    const static int N = 1e3 + 10, M = 16;
    
    LL dis[M], f[N];
    
    int minimumFinishTime(vector<vector<int>>& tires, int changeTime, int numLaps) {
        memset(dis, 0x3f, sizeof dis);
        int n = tires.size();
        for (auto & t : tires) {
            LL d[N];
            d[0] = 0;
            LL f = t[0], r = t[1];
            for (int i = 1, j = 1; i < M; ++ i , j *= r ) {
                d[i] = d[i - 1] + f * j;
                if (d[i] > 2e5)
                    break;
                dis[i] = min(dis[i], d[i]);
            }
        }
        
        for (int i = 0; i < N; ++ i )
            f[i] = 1e18;
        f[0] = 0;
        for (int i = 1; i <= numLaps; ++ i ) {
            for (int j = 1; j < M; ++ j )
                if (i - j >= 0)
                    f[i] = min(f[i], f[i - j] + dis[j] + changeTime);
        }
        
        return f[numLaps] - changeTime;
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

> [!NOTE] **[Codeforces B. Little Dima and Equation](https://codeforces.com/problemset/problem/460/B)**
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
// Problem: B. Little Dima and Equation
// Contest: Codeforces - Codeforces Round #262 (Div. 2)
// URL: https://codeforces.com/problemset/problem/460/B
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// note
// 和最大为81 故枚举和即可
using LL = long long;

LL S(LL x) {
    // < 0 needed
    if (x < 0)
        return -1;
    string s = to_string(x);
    int ret = 0;
    for (auto c : s)
        ret += c - '0';
    return ret;
}

int main() {
    LL a, b, c;
    cin >> a >> b >> c;

    vector<LL> ve;
    for (int i = 1; i <= 81; ++i) {
        LL x = b * pow(i, a) + c;
        if (S(x) == i && x < 1e9)
            ve.push_back(x);
    }
    cout << ve.size() << endl;
    for (auto v : ve)
        cout << v << ' ';
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

> [!NOTE] **[Codeforces C. Mr. Kitayuta, the Treasure Hunter](https://codeforces.com/problemset/problem/505/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **根据推断优化空间**
> 
> 记忆化搜索 数组比map快很多

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Mr. Kitayuta, the Treasure Hunter
// Contest: Codeforces - Codeforces Round #286 (Div. 2)
// URL: https://codeforces.com/problemset/problem/505/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// d 范围 30000 直接申请空间显然TLE
// 重点在于分析出 l 从始至终变化不会超过 [d-300, d+300]
//
// 或者直接像自己最初的做法 直接map
// 错误原因： 【dp 没有返回值导致子状态无法累加上去就直接 return 了】
// https://codeforces.com/contest/505/submission/110856257
//
// 1. 修改 map
// 用例就会TLE超时
// https://codeforces.com/contest/505/submission/110857387
// 2. 使用正常数组 AC
// https://codeforces.com/contest/505/submission/110857161

using PII = pair<int, int>;
const int N = 30010, OFFSET = 300;

int n, d;
int w[N];
int f[N][OFFSET << 1];

int dp(int u, int l) {
    int nl = l - d + OFFSET;
    if (f[u][nl] != -1)
        return f[u][nl];

    int& t = f[u][nl];
    // t = 0 的初始化很重要 否则TLE
    t = 0;

    int l1 = l - 1, l2 = l, l3 = l + 1;
    int x1 = u + l1, x2 = u + l2, x3 = u + l3;

    if (x1 > u && x1 < N && l1 > d - OFFSET && l1 < d + OFFSET)
        t = max(t, dp(x1, l1));
    if (x2 > u && x2 < N && l1 > d - OFFSET && l1 < d + OFFSET)
        t = max(t, dp(x2, l2));
    if (x3 > u && x3 < N && l1 > d - OFFSET && l1 < d + OFFSET)
        t = max(t, dp(x3, l3));
    t += w[u];
    return t;
}

int main() {
    cin >> n >> d;
    for (int i = 0; i < n; ++i) {
        int x;
        cin >> x;
        w[x]++;
    }

    memset(f, -1, sizeof f);
    // 已累积w[0], 现在在d且l为d
    cout << dp(d, d) << endl;

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

> [!NOTE] **[Codeforces Matrix](http://codeforces.com/problemset/problem/364/A)** [TAG]
> 
> 题意: 
> 
> 有一个十进制字符串 $s$ 。
> 
> 现定义 $b_{i,j}=s_i*s_j$ ，请找出在矩阵 $b$ 中有几个顶点在格点上的矩形所有元素之和等于 $a$

> [!TIP] **思路**
> 
> 错了非常多遍，甚至想错了优化方向
> 
> 实际上最终求解的时候不需要关注 map 保存的值，只需要 for-loop 一遍即可
> 
> 以及 【题意抽象与转化 数值计算细节】
> 
> **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Matrix
// Contest: Codeforces - Codeforces Round #213 (Div. 1)
// URL: https://codeforces.com/problemset/problem/364/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 可以推出矩形的和即为 [长和]*[宽和]
// 故先 cnt 统计所有可能的长和宽

using LL = long long;
using PLL = pair<LL, LL>;
const static int N = 4e3 + 10, M = 4e4 + 10;  // ATTENTION M 9*abs(s)

LL s[N];
unordered_map<LL, LL> cnt;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    LL tar;
    string str;
    cin >> tar >> str;

    int n = str.size();
    for (int i = 1; i <= n; ++i) {
        s[i] = s[i - 1] + (str[i - 1] - '0');
        // 统计所有 sum(j, i)
        for (int j = 1; j <= i; ++j) {
            LL d = s[i] - s[j - 1];
            cnt[d]++;
        }
    }

    LL res = 0;
    if (tar == 0) {
        for (int i = 0; i < M; ++i)
            res += cnt[0] * cnt[i] * 2;
        res -= cnt[0] * cnt[0];
        // ATTENTION
        // sum_of_cnt{0...N-1} = n * (n + 1) / 2
        // res = cnt[0] * n * (n + 1) - cnt[0] * cnt[0];
        cout << res << endl;
    } else {
        // ATTENTION: for-loop N 而不是 cnt-map
        for (int i = 1; i < M; ++i) {
            if (tar / i >= M)
                continue;
            if (tar % i == 0)
                res += cnt[i] * cnt[tar / i];  // no need to *2
        }
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

> [!NOTE] **[Codeforces Little Elephant and Cards](http://codeforces.com/problemset/problem/204/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 根据数据范围，有点小技巧

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Little Elephant and Cards
// Contest: Codeforces - Codeforces Round #129 (Div. 1)
// URL: https://codeforces.com/problemset/problem/204/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const static int N = 1e5 + 10, M = 2e5 + 10;

unordered_map<int, int> H;
int idx = 0;
int get(int x) {
    if (H.count(x))
        return H[x];
    return H[x] = ++idx;
}

int n, tar;
PII a[N];
int f[M], b[M], s[M];

int main() {
    cin >> n;
    tar = (n + 1) / 2;

    for (int i = 0; i < n; ++i) {
        int l, r;
        cin >> l >> r;
        l = get(l), r = get(r);
        a[i] = {l, r};
        f[l]++, b[r]++;
        if (l == r)
            s[l]++;  // 前后颜色相同，对于本个无法通过翻转获得收益
    }

    int res = 1e9;
    for (int i = 1; i <= idx; ++i) {
        int need = max(tar - (f[i]), 0);
        if (need <= b[i] - s[i])
            res = min(res, need);
    }
    if (res == 1e9)
        cout << -1 << endl;
    else
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

> [!NOTE] **[LeetCode 2242. 节点序列的最大得分](https://leetcode.cn/problems/maximum-score-of-a-node-sequence/)**
> 
> 题意: 
> 
> 一个无向图，沿着边连续选择4个点，求最大权值和

> [!TIP] **思路**
> 
> 刚开始想着建图 DFS 限定递归层数为 4 ==> TLE
> 
> 其实显然可以枚举一个必选的边，随后看该边上的两个点相连的所有点可以选哪些即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 5e4 + 10;
    
    int n;
    vector<int> sc;
    int mx[N][3];
    
    int maximumScore(vector<int>& scores, vector<vector<int>>& edges) {
        this->sc = scores, this->n = sc.size();
        memset(mx, -1, sizeof mx);  // ATTENTION 必须 因为 0 节点有意义
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            {
                int x = sc[b];
                if (mx[a][0] == -1 || x >= sc[mx[a][0]])
                    mx[a][2] = mx[a][1], mx[a][1] = mx[a][0], mx[a][0] = b;
                else if (mx[a][1] == -1 || x >= sc[mx[a][1]])
                    mx[a][2] = mx[a][1], mx[a][1] = b;
                else if (mx[a][2] == -1 || x > sc[mx[a][2]])
                    mx[a][2] = b;
            }
            
            {
                int x = sc[a];
                if (mx[b][0] == -1 || x >= sc[mx[b][0]])
                    mx[b][2] = mx[b][1], mx[b][1] = mx[b][0], mx[b][0] = a;
                else if (mx[b][1] == -1 || x >= sc[mx[b][1]])
                    mx[b][2] = mx[b][1], mx[b][1] = a;
                else if (mx[b][2] == -1 || x > sc[mx[b][2]])
                    mx[b][2] = a;
            }
            
        }
        
        int res = -1;
        for (auto & e : edges) {
            int a = e[0], b = e[1];
            vector<int> as, bs;
            for (int i = 0; i < 3; ++ i ) {
                if (mx[a][i] != -1 && mx[a][i] != b)
                    as.push_back(mx[a][i]);
                if (mx[b][i] != -1 && mx[b][i] != a)
                    bs.push_back(mx[b][i]);
            }
            int t = sc[a] + sc[b];
            for (auto & x : as)
                for (auto & y : bs)
                    if (x != y) {
                        // cout << x << " " << y << endl;
                        res = max(res, t + sc[x] + sc[y]);
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

> [!NOTE] **[LeetCode 2250. 统计包含每个点的矩形数目](https://leetcode.cn/problems/count-number-of-rectangles-containing-each-point/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经验:
> 
> - 离散化映射时直接 lower_bound 比建立个 unordered_map 更快
> 
> - **深刻理解此类处理思路，加快速度**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 110;

    vector<int> v[N];  // 按列统计（横向）

    vector<int> countRectangles(vector<vector<int>>& rectangles, vector<vector<int>>& points) {
        for (auto & r : rectangles)
            v[r[1]].push_back(r[0]);
        for (int i = 1; i < N; ++ i )
            sort(v[i].begin(), v[i].end());
        
        vector<int> res;
        for (auto & p : points) {
            int t = 0;
            for (int i = p[1]; i < N; ++ i )
                t += v[i].end() - lower_bound(v[i].begin(), v[i].end(), p[0]);
            res.push_back(t);
        }
        return res;
    }
};
```

##### **C++ 直观离散化并前缀和 差点TLE**

```cpp
class Solution {
public:
    // using LL = long long;
    // 本质对每个点求 在其右上方的矩阵右上顶点的数量
    const static int N = 1e5 + 10, M = 110;
    
    // int s[N][M];
    
    vector<int> countRectangles(vector<vector<int>>& rectangles, vector<vector<int>>& points) {
        int n = 0, m = 0;
        vector<int> ids;
        {
            // 离散化
            ids.push_back(0);
            for (auto & r : rectangles)
                ids.push_back(r[0]), m = max(m, r[1]);
            for (auto & p : points)
                ids.push_back(p[0]), m = max(m, p[1]);
            sort(ids.begin(), ids.end());
            ids.erase(unique(ids.begin(), ids.end()), ids.end());
            
            // 转换映射
            for (auto & r : rectangles)
                r[0] = lower_bound(ids.begin(), ids.end(), r[0]) - ids.begin();
            for (auto & p : points)
                p[0] = lower_bound(ids.begin(), ids.end(), p[0]) - ids.begin();
        }
        n = ids.size();
        vector<vector<int>> s(n + 1, vector<int>(m + 1));
        {
            for (auto & r : rectangles)
                s[r[0]][r[1]] ++ ;
            for (int i = 1; i < n + 1; ++ i )
                for (int j = 1; j < m + 1; ++ j )
                    s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
        }
        
        vector<int> res;
        for (auto & p : points) {
            int t = s[n][m] - s[p[0] - 1][m] - s[n][p[1] - 1] + s[p[0] - 1][p[1] - 1];
            res.push_back(t);
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

> [!NOTE] **[LeetCode 2440. 创建价值相同的连通块](https://leetcode.cn/problems/create-components-with-same-value/) [TAG]**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然的有根据数据范围求解
> 
> 另外还有 dfs 内部的实现思维 => **反复思考细节**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 观察数据，1 <= nums[i] <= 50
    // ===> 树总值最大为 1e6, 因子数 240, 考虑枚举每个联通块的价值【这个价值必须整除总值】
    using LL = long long;
    const static int N = 2e4 + 10, M = 4e4 + 10;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n;
    vector<int> ns;
    
    // 求以某个点为根的子树的价值和
    int f[N];
    void dfs_1(int u, int fa) {
        f[u] = ns[u];
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            dfs_1(j, u);
            f[u] += f[j];
        }
    }
    
    // 每块联通块值都为 k
    bool dfs_2(int u, int fa, int k) {
        int s = ns[u];
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            if (!dfs_2(j, u, k))
                return false;
            // ATTENTION 把这部分加在一起
            if (f[j] % k)
                s += f[j] % k;
        }
        // ATTENTION 思考 只有 <= k 才是合法的，否则说明这一坨一定无法合理分割
        return s <= k;
    }
    
    int componentValue(vector<int>& nums, vector<vector<int>>& edges) {
        init();
        this->ns = nums;
        this->n = ns.size();
        
        for (auto & e : edges)
            add(e[0], e[1]), add(e[1], e[0]);
        
        dfs_1(0, -1);
        
        // 枚举值
        for (int i = 1; i <= f[0]; ++ i )
            if (f[0] % i == 0 && dfs_2(0, -1, i))
                return f[0] / i - 1;
        return 0;
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

> [!NOTE] **[LeetCode 2518. 好分区的数目](https://leetcode.cn/problems/number-of-great-partitions/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 题意范围显然无法直接写 01 背包
> 
> 想到 **容斥** 的基础上逆向计算
> 
> 加快速度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 题目要求: 子数组和为 [k, sum - k]
    // 容斥 考虑: 所有选择方案，减去当前和 [0, k - 1] + [sum - k + 1, sum]
    using LL = long long;
    const static int N = 1010, MOD = 1e9 + 7;
    
    LL qpow(LL a, LL b) {
        LL ret = 1;
        while (b) {
            if (b & 1)
                ret = ret * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return ret;
    }
    
    LL f[N];
    
    int countPartitions(vector<int>& nums, int k) {
        LL s = 0;
        for (auto x : nums)
            s += (LL)x;
        if (s < 2 * k)
            return 0;
        
        memset(f, 0, sizeof f);
        f[0] = 1;
        for (auto x : nums)
            for (int i = N - 1; i >= 0 && i >= x; -- i )
                f[i] = (f[i] + f[i - x]) % MOD;
        
        LL res = 0;
        for (int i = 0; i < k; ++ i )
            res = (res + f[i]) % MOD;
        
        return (qpow(2, nums.size()) - res * 2ll % MOD + MOD) % MOD;
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

> [!NOTE] **[LeetCode 2493. 将节点分成尽可能多的组](https://leetcode.cn/problems/divide-nodes-into-the-maximum-number-of-groups/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数据范围 $n <= 500$
>
> 显然可以按联通性分组，单组内枚举起点求合法的最大值，多分组间累加即可
> 
> **加快速度**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 数据范围 n <= 500
    // 尝试遍历起点，检查是否能遍历所有节点且距离符合要求
    // ==> 对于有多个联通分量的情况 不同分量之间需要累加
    
    const static int N = 510, M = 2e4 + 10, INF = 0x3f3f3f3f;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int id[N];
    vector<int> xs[N];
    void dfs(int u, int pa, int t) {
        id[u] = t;
        xs[t].push_back(u);
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            if (!id[j])
                dfs(j, u, t);
        }
    }
    
    int q[N], from[N], d[N];
    
    int check(int u, int i) {
        memset(d, 0x3f, sizeof d);
        memset(from, -1, sizeof from);
        int hh = 0, tt = -1;
        d[u] = 0, q[ ++ tt] = u;
        
        while (hh <= tt) {
            int t = q[hh ++ ];
            for (int i = h[t]; ~i; i = ne[i]) {
                int j = e[i];
                if (d[j] >= INF / 2)
                    d[j] = d[t] + 1, q[ ++ tt] = j;
                else if (abs(d[j] - d[t]) != 1) {
                    return -1;
                }
            }
        }
        
        int maxd = -1;
        for (auto x : xs[i])
            maxd = max(maxd, d[x] + 1);
        return maxd;
    }
    
    int magnificentSets(int n, vector<vector<int>>& edges) {
        init();
        for (auto & e : edges)
            add(e[0], e[1]), add(e[1], e[0]);
        
        memset(id, 0, sizeof id);
        for (int i = 0; i <= n; ++ i )
            xs[i].clear();
        for (int i = 1; i <= n; ++ i )
            if (!id[i])
                dfs(i, -1, i);
        
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            if (id[i] == i) {
                int v = -1;
                // 从起点开始 找到可行的距离最远的方案
                for (auto x : xs[i])
                    v = max(v, check(x, i));
                
                if (v == -1)
                    return -1;
                res += v;
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

> [!NOTE] **[LeetCode 2552. 统计上升四元组](https://leetcode.cn/problems/count-increasing-quadruplets/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思想上与 [LeetCode 2242. 节点序列的最大得分](https://leetcode.cn/problems/maximum-score-of-a-node-sequence/) 类似，都是勇于暴力枚举中间两个点即可
>
> 1. $4e3$ 的数据范围可以接受 $O(n^2)$ 预处理后暴力计算，需要注意的是：
>
> - 状态定义与边界处理
>
> - 二维数组放在 class 里会 stack overflow
>
> - 二维数组直接开 LL 会 TLE
>
> 2. 显然可以 BIT 维护，同时可以通过改变遍历顺序进一步减少开销

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 二维前缀和预处理**

```cpp
// 二维数组放 class 内部会 stack overflow
using LL = long long;
const static int N = 4010;

// LL l[N][N], r[N][N];    // 左侧较小的 右侧较大的
int l[N][N], r[N][N];    // 【ATTENTION】 int 超时

class Solution {
public:
    long long countQuadruplets(vector<int>& nums) {
        memset(l, 0, sizeof l);
        memset(r, 0, sizeof r);
        int n = nums.size();

        {
            for (int i = 1; i <= n; ++ i ) {
                int x = nums[i - 1];
                memcpy(l[i], l[i - 1], sizeof l[i]);
                for (int j = x + 1; j < N; ++ j )   // ATTENTION j=x+1 细节
                    l[i][j] ++ ;
            }
        }
        {
            for (int i = n; i >= 1; -- i ) {
                int x = nums[i - 1];
                memcpy(r[i], r[i + 1], sizeof r[i]);
                for (int j = x - 1; j >= 0; -- j )
                    r[i][j] ++ ;
            }
        }

        LL res = 0;
        // j
        for (int i = 1; i <= n; ++ i ) {
            int x = nums[i - 1];
            // k
            for (int j = i + 1; j <= n; ++ j ) {
                int y = nums[j - 1];
                if (x > y) {
                    // 左侧要比 y 小, 右侧要比 x 大
                    // ATTENTION 细节 【数组下标数值】
                    int left = l[i][y], right = r[j][x];
                    res += (LL)left * right;
                }
            }
        }
        return res;
    }
};
```

##### **C++ BIT**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 4010;

    int a[N], b[N]; // l, r
    int lowbit(int x) {
        return x & -x;
    }
    void add(int tr[], int x, int c) {
        for (int i = x; i < N; i += lowbit(i))
            tr[i] += c;
    }
    int sum(int tr[], int x) {
        int ret = 0;
        for (int i = x; i; i -= lowbit(i))
            ret += tr[i];
        return ret;
    }

    long long countQuadruplets(vector<int>& nums) {
        int n = nums.size();

        // 初始化右侧的都存在
        for (int i = 1; i <= n; ++ i )
            add(b, nums[i - 1], 1);

        LL res = 0;
        // j
        for (int i = 1; i <= n; ++ i ) {
            int x = nums[i - 1];
            // 移除当前位置作为右侧的可能
            add(b, x, -1);
            // 拷贝至 c 并在随后的过程动态修改
            static int c[N];
            memcpy(c, b, sizeof c);
            // k
            for (int j = i + 1; j <= n; ++ j ) {
                int y = nums[j - 1];
                add(c, y, -1);
                if (x > y) {
                    // 左侧要比 y 小, 右侧要比 x 大
                    // ATTENTION 细节 【数组下标数值】
                    int left = sum(a, y - 1), right = sum(c, N - 1) - sum(c, x);
                    res += (LL)left * right;
                }
            }
            add(a, x, 1);
        }
        return res;
    }
};
```

##### **C++ BIT 优化**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 4010;

    int a[N], b[N]; // l, r
    int lowbit(int x) {
        return x & -x;
    }
    void add(int tr[], int x, int c) {
        for (int i = x; i < N; i += lowbit(i))
            tr[i] += c;
    }
    int sum(int tr[], int x) {
        int ret = 0;
        for (int i = x; i; i -= lowbit(i))
            ret += tr[i];
        return ret;
    }

    long long countQuadruplets(vector<int>& nums) {
        int n = nums.size();

        LL res = 0;
        // j
        for (int i = 1; i <= n; ++ i ) {
            int x = nums[i - 1];
            // 逆序扫描同时维护 b
            memset(b, 0, sizeof b);
            // k
            for (int j = n; j > i; -- j ) {
                int y = nums[j - 1];
                if (x > y) {
                    // 左侧要比 y 小, 右侧要比 x 大
                    // ATTENTION 细节 【数组下标数值】
                    int left = sum(a, y - 1), right = sum(b, N - 1) - sum(b, x);
                    res += (LL)left * right;
                }
                add(b, y, 1);
            }
            add(a, x, 1);
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

> [!NOTE] **[LeetCode 2572. 无平方子集计数](https://leetcode.cn/problems/count-the-number-of-square-free-subsets/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 容易想到枚举质因子出现的情况 **重点在于想到按顺序枚举数字来实现去重**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 较显然的，需要根据数据范围找思路
    //      对于 30 以内的值域，最终的集合的因子必然满足下列质数因子至多出现一次
    //          [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] ==> 只有10个
    // 显然可以枚举最终的集合的情况 (2^10)
    // 问题在于，因为最终结果顺序无关，【如何避免重复计数】？
    //
    // -> 最外层增加一个维度 按照数值域顺序枚举
    using LL = long long;
    const static int N = 1100, M = 31, K = 10, MOD = 1e9 + 7;
    
    int p[K] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    int valid(int x) {
        int ori = x;
        int y = 0;
        for (int i = 0; i < K; ++ i ) {
            if (x % p[i] == 0) {
                int c = 0;
                while (x % p[i] == 0)
                    x /= p[i], c ++ ;
                if (c > 1)
                    return -1;
                y |= 1 << i;
            }
        }
        return y;
    }
    
    int c[M], st[M];
    LL f[N];
    
    int squareFreeSubsets(vector<int>& nums) {
        memset(c, 0, sizeof c), memset(st, 0, sizeof st);
        for (auto x : nums)
            c[x] ++ ;
        for (int i = 1; i <= 30; ++ i )
            st[i] = valid(i);
        
        int TOP = 1 << K;
        f[0] = 1;
        for (int i = 2; i <= 30; ++ i ) {
            if (st[i] == -1)
                continue;
            
            // 枚举当前使用 i 的情况下，最终集合的情况
            int x = st[i], y = (TOP - 1) ^ x;
            // ATTENTION 枚举对应补集的子集
            for (int j = y; ; j = (j - 1) & y) {    // 不能写 j = y; [j]; j = (j - 1) & y   => 因为 0 也需要考虑
                f[x | j] = (f[x | j] + f[j] * c[i] % MOD) % MOD;
                if (j == 0)
                    break;
            }
        }
        
        LL a = 1, b = 0;
        {
            for (int i = 0; i < c[1]; ++ i )
                a = (a << 1) % MOD;
        }
        {
            for (int i = 0; i < TOP; ++ i )
                b = (b + f[i]) % MOD;
        }
        return (a * b % MOD - 1 + MOD) % MOD;
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

> [!NOTE] **[LeetCode 2768. 黑格子的数目](https://leetcode.cn/problems/number-of-black-blocks/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然要从 coordinates
> 
> 一种思路是枚举方块 需要注意细节的去重操作 代码略
> 
> 另一种是计算对哪些方块产生影响 全局统计即可 更好写

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 换个思路 计算当前点对哪些方格产生影响
    using LL = long long;
    using PII = pair<int, int>;
    
    vector<long long> countBlackBlocks(int m, int n, vector<vector<int>>& coordinates) {
        map<PII, int> h;  // 某个右下角的方格 有多少个黑点
        for (auto & cs : coordinates) {
            int x = cs[0], y = cs[1];
            {
                // 当前在左上角
                if (x < m - 1 && y < n - 1)
                    h[{x + 1, y + 1}] ++ ;
            }
            {
                // 当前在左下角
                if (x && y < n - 1)
                    h[{x, y + 1}] ++ ;
            }
            {
                // 右上角
                if (x < m - 1 && y)
                    h[{x + 1, y}] ++ ;
            }
            {
                // 右下角
                if (x && y)
                    h[{x, y}] ++ ;
            }
        }
        
        vector<LL> res(5);
        for (auto & [k, v] : h)
            res[v] ++ ;
        
        LL sum = 1ll * (m - 1) * (n - 1);
        res[0] = sum - res[1] - res[2] - res[3] - res[4];
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

> [!NOTE] **[LeetCode 2857. 统计距离为 k 的点对](https://leetcode.cn/problems/count-pairs-of-points-with-distance-k/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 核心在于想到枚举思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    using PII = pair<int, int>;
    
    // 思考: 如何转化 x1^x2 + y1^y2 ?
    //  直觉比较困难 考虑结合 k<=100
    // =>   x1^x2 = 0, y1^y2 = k
    //      x1^x2 = 1, y1^y2 = k - 1
    //      ...
    //      x1^x2 = k, y1^y2 = 0

    int countPairs(vector<vector<int>>& coordinates, int k) {
        int n = coordinates.size();
        map<PII, int> S;
        
        int res = 0;
        // 考虑 i<j 的顺序关系，需要外层循环枚举元素，内层枚举异或值
        for (int i = 0; i < n; ++ i ) {
            // 枚举 x 的异或值
            int x1 = coordinates[i][0], y1 = coordinates[i][1];
            for (int l = 0; l <= k; ++ l ) {
                int r = k - l;
                PII other = {x1 ^ l, y1 ^ r};
                if (S.count(other))
                    res += S[other];
            }
            // 边遍历边维护
            S[{x1, y1}] ++ ;
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

> [!NOTE] **[LeetCode 3067. 在带权树网络中统计可连接服务器对数目 ](https://leetcode.cn/problems/count-pairs-of-connectable-servers-in-a-weighted-tree-network/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 题目数据范围接受 $n^2$ 暴力
> 
> 边维护边统计

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 显然不能枚举两个端点，这样复杂度爆炸...因为需要遍历所有中间点来统计答案
    // 考虑枚举每个点作为中间点 可以有多少个数对 => 需要知道每个点到其他点的距离汇总信息
    // ... 又因为 题目要求的本质是两个不同子树的信息统计，之前的代码只适用于同一节点的固定两侧 无法细拆
    //
    // 考虑直接暴力
    
    const static int N = 1010, M = 2010;
    
    int h[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int ss;
    
    // 求某个节点
    int dfs(int u, int pa, int sum) {
        int cnt = sum % ss == 0;        // 当前节点自己 如果符合要求则+1
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i], c = w[i];
            if (j == pa)
                continue;
            
            cnt += dfs(j, u, sum + c);  // 往子树延伸 统计总数
        }
        return cnt;
    }
    
    
    vector<int> countPairsOfConnectableServers(vector<vector<int>>& edges, int signalSpeed) {
        init();
        for (auto & e : edges)
            add(e[0], e[1], e[2]), add(e[1], e[0], e[2]);
        
        this->ss = signalSpeed;
        
        vector<int> res;
        for (int i = 0; i < edges.size() + 1; ++ i ) {
            int sum = 0, tot = 0;       // 边遍历边统计 不需要/2
            for (int j = h[i]; ~j; j = ne[j]) {
                int v = e[j], c = w[j];
                // cnt 为该独立子树下的、到当前节点的距离为 ss 倍数的节点总数...
                int cnt = dfs(v, i, c);
                tot += cnt * sum;   // ATTENTION 本颗子树的数量和 * 前面所有子树的数量和
                sum += cnt;         // 维护
            }
            res.push_back(tot);
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

> [!NOTE] **[LeetCode 2920. 收集所有金币可获得的最大积分](https://leetcode.cn/problems/maximum-points-after-collecting-coins-from-all-nodes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 树形 dp + 根据数据范围求解

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 题意: 要想收集节点上的金币，必须先收集该节点的祖先节点上的金币,
    // => 则某一个节点的当前值计算应当与这条链上的选择方案相关
    // 注意值域 1e4 则可以缓存每个值被压多少次
    
    using PII = pair<int, int>;
    const static int N = 1e5 + 10, M = 2e5 + 10;
    
    int h[N], e[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int k;
    vector<int> cs;
    // map<PII, int> hash;
    int hash[N][18];    // ATTENTION 追加 c<17 的判断条件后仍然超时，考虑将 map 转为静态树组 => AC
    
    int dfs(int u, int fa, int c) {
        // if (hash.count({u, c}))
        //     return hash[{u, c}];
        if (hash[u][c] != -1)
            return hash[u][c];
        
        int original = cs[u];
        // RE: Line 28: Char 32: runtime error: shift exponent 32 is too large for 32-bit type 'int' (solution.cpp)
        // => 把 c 缩小
        // c = min(c, 31);
        //
        // => TLE
        // 考虑数据范围，在 c > 17 时直接不进入后续递归
        int updated = original >> c;
        
        int a = updated - k, b = updated >> 1;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            a += dfs(j, u, c);
            
            // ATTENTION: 追加 c<17 的判断条件
            if (c < 17)
                b += dfs(j, u, c + 1);
        }
        // return hash[{u, c}] = max(a, b);
        return hash[u][c] = max(a, b);
    }
    
    int maximumPoints(vector<vector<int>>& edges, vector<int>& coins, int k) {
        init();
        this->k = k, this->cs = coins;
        for (auto & e : edges)
            add(e[0], e[1]), add(e[1], e[0]);
        
        memset(hash, -1, sizeof hash);

        return dfs(0, -1, 0);
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

> [!NOTE] **[LeetCode 3082. 求出所有子序列的能量和](https://leetcode.cn/problems/find-the-sum-of-the-power-of-all-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然无法枚举 考虑每个数字组合的贡献 两种思路
> 
> -   依次考虑每个数字 0-1 背包
> -   分类所有数字 多重背包
> 
> 多重背包需要结合数据范围 分两次统计得到正确的值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 110, M = 10010, MOD = 1e9 + 7;
    
    LL C[N][N];
    void init() {
        for (int i = 0; i < N; ++ i )
            for (int j = 0; j <= i; ++ j )
                if (!j)
                    C[i][j] = 1;
                else
                    C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) % MOD;
    }
    
    LL qpow(int a, int b) {
        LL ret = 1;
        while (b) {
            if (b & 1)
                ret = (ret * a) % MOD;
            a = LL(a) * a % MOD;
            b >>= 1;
        }
        return ret;
    }
    
    int cnt[M];
    LL f[N][M];
    
    int sumOfPower(vector<int>& nums, int k) {
        init();
        
        memset(cnt, 0, sizeof cnt);
        for (auto x : nums)
            cnt[x] ++ ;
        
        memset(f, 0, sizeof f);
        f[0][0] = 1;
        for (int i = 1; i < N; ++ i ) {     // ATTENTION: N
            int count = cnt[i];
            
            for (int j = 0; j < N; ++ j )   // ATTENTION: N
                for (int c = 0; c <= count && c * i <= j; ++ c )
                    f[i][j] = (f[i][j] + f[i - 1][j - c * i] * C[count][c] % MOD * qpow(2, count - c) % MOD) % MOD;
        }
        
        LL res = f[N - 1][k];
        for (int i = N; i < M; ++ i ) {
            int count = cnt[i];
            res = (res * qpow(2, count)) % MOD;
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