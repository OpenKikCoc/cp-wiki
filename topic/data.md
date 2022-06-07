## 习题

> [!NOTE] **[LeetCode 1198. 找出所有行中最小公共元素](https://leetcode-cn.com/problems/find-smallest-common-element-in-all-rows/)**
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

> [!NOTE] **[LeetCode 1326. 灌溉花园的最少水龙头数目](https://leetcode-cn.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/)** [TAG]
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

> [!NOTE] **[LeetCode 1766. 互质树](https://leetcode-cn.com/problems/tree-of-coprimes/)** [TAG]
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

> [!NOTE] **[LeetCode 1906. 查询差绝对值的最小值](https://leetcode-cn.com/problems/minimum-absolute-difference-queries/)**
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

> [!NOTE] **[LeetCode 1981. 最小化目标值与所选元素的差](https://leetcode-cn.com/problems/minimize-the-difference-between-target-and-chosen-elements/)**
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

> [!NOTE] **[LeetCode 2188 完成比赛的最少时间](https://leetcode-cn.com/problems/minimum-time-to-finish-the-race/)**
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

> [!NOTE] **[LeetCode 6043. 统计包含每个点的矩形数目](https://leetcode.cn/problems/count-number-of-rectangles-containing-each-point/)**
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