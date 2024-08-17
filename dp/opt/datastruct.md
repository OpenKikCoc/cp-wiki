# 数据结构优化DP

## 习题

### 线段树优化 / 树状数组优化

> [!NOTE] **[LeetCode 2407. 最长递增子序列 II](https://leetcode.cn/problems/longest-increasing-subsequence-ii/)**
> 
> 题意: 
> 
> 严格最长上升子序列，但要求相邻两个数字之间的值之差不超过 $k$

> [!TIP] **思路**
> 
> $f[i] = max{f[j]} + 1$
> 
> 其中 $max(1, i - k) <= j < i$
> 
> 显然涉及到区间查询，以及后续的单点修改
> 
> 线段树 / 树状数组【树状数组新姿势】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 线段树**

```cpp
const static int N = 1e5 + 10;

int w[N];

struct Node {
    int l, r;
    int maxv;
} tr[N << 2];

void pushup(int u) {
    tr[u].maxv = max(tr[u << 1].maxv, tr[u << 1 | 1].maxv);
}

void eval(Node & t, int v) {
    t.maxv = v;
}

void build(int u, int l, int r) {
    if (l == r)
        tr[u] = {l, r, w[l]};
    else {
        tr[u] = {l, r, 0};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

void modify(int u, int l, int r, int k) {
    if (tr[u].l >= l && tr[u].r <= r)
        eval(tr[u], k);
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)
            modify(u << 1, l, r, k);
        if (r > mid)
            modify(u << 1 | 1, l, r, k);
        pushup(u);
    }
}

int query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)
        return tr[u].maxv;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        
        int x = 0;
        if (l <= mid)
            x = max(x, query(u << 1, l, r));
        if (r > mid)
            x = max(x, query(u << 1 | 1, l, r));
        return x;
    }
}

class Solution {
public:
    int lengthOfLIS(vector<int>& nums, int k) {
        memset(w, 0, sizeof w);
        build(1, 1, N - 1);
        
        for (auto x : nums) {
            int t = query(1, x - k, x - 1);
            modify(1, x, x, t + 1);
        }
        return query(1, 1, N - 1);
    }
};
```

##### **C++ 树状数组**

```cpp
const static int N = 1e5 + 10;

int w[N];   // 必须有两个数组
int tr[N];

int lowbit(int x) {
    return x & -x;
}
void modify(int x, int y) {
    w[x] = y;
    for (int i = x; i < N; i += lowbit(i))
        tr[i] = max(tr[i], y);
}
int query(int l, int r) {
    int res = 0;
    for (; l <= r;) {
        res = max(res, w[r]);   // ATTENTION
        for ( -- r ; r >= l + lowbit(r); r -= lowbit(r))
            res = max(res, tr[r]);
    }
    return res;
}

class Solution {
public:
    int lengthOfLIS(vector<int>& nums, int k) {
        memset(w, 0, sizeof w), memset(tr, 0, sizeof tr);
        for (auto y : nums) {
            int t = query(max(1, y - k), y - 1);
            modify(y, t + 1);
        }
        return query(1, N - 1);
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

> [!NOTE] **[LeetCode 3117. 划分数组得到最小的值之和](https://leetcode.cn/problems/minimum-sum-of-values-by-dividing-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数据结构优化大杂烩
> 
> - `[prefix_sum / RMQ] + [BIT / 单调队列]`
> 
> - `X + LogTrick`
> 
>   LogTrick: **给定一个数组，以某个右端点为结尾的所有子数组，其中不同的 或/与/lcm/gcd 值至多只有 logU 个**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ prefix_sum + BIT**

```cpp
class Solution {
public:
    // 搜索TLE (TODO: revisit 剪枝策略)
    //
    // 考虑 dp
    //  f[i][j] 表示考虑前i个元素切分成j段的最小值和
    //  => f[i][j] = min(f[x][j-1]) + nums[i]
    //     其中x需满足一定条件: and[x+1,j] = andVlues[j]
    //  由与操作性质容易推导 x必然锁定到一个连续区间 接下来就是查询这个区间的最小值
    // 
    // 拆解子问题:
    // 1. 如何计算得到x的区间 => 二分 => 需能快速获取特定区间的 and 值 => 前缀和/rmq
    // 2. 如何维护区间最小值 => BIT/单调队列(更优)
    //
    // 【数据结构优化DP】

    const static int N = 1e4 + 10, M = 11, K = 18, INF = 0x3f3f3f3f;

    // prefix sum
    int sum[N][K];
    int get_by_sum(int l, int r) {
        int ret = 0;
        for (int j = 0; j < K; ++ j )
            if (sum[r][j] - sum[l - 1][j] == r - l + 1)
                ret += 1 << j;
        return ret;
    }
    
    // BIT
    // - 区间最值
    // - 需要nxt数组记录对于第j段下次应当加入bit的位置
    int w[M][N], tr[M][N], nxt[M];
    int lowbit(int x) {
        return x & -x;
    }
    void bit_modify(int tr[], int w[], int x, int y) {
        w[x] = y;
        for (int i = x; i < N; i += lowbit(i))
            tr[i] = min(tr[i], y);  // min
    }
    int bit_query(int tr[], int w[], int l, int r) {
        int ret = INF;  // ATTENTION debug
        for (; l <= r; ) {
            ret = min(ret, w[r]);
            for ( -- r ; r >= l + lowbit(r); r -= lowbit(r))
                ret = min(ret, tr[r]);
        }
        return ret;
    }

    void init() {
        {
            memset(sum, 0, sizeof sum);
            for (int i = 1; i <= n; ++ i )
                for (int j = 0; j < K; ++ j )
                    sum[i][j] = sum[i - 1][j] + (nums[i - 1] >> j & 1);
        }
        {
            memset(w, 0x3f, sizeof w);
            memset(tr, 0x3f, sizeof tr);
            for (int i = 0; i < M; ++ i )
                nxt[i] = 1;
        }
    }

    int search(int L, int R, int x) {
        int l = L, r = R + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (get_by_sum(mid, R) >= x)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    int f[N][M];

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();

        init();

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                int target = andValues[j - 1];

                // ATTENTION 减少无效计算
                if (get_by_sum(1, i) > target)
                    continue;
                
                // ATTENTION: 思考
                // 找到 第一个 >= target 的位置  &  第一个 > target 的位置
                //  l,r 本质都是上一个轮次的下标
                int l = search(1, i, target), r = search(1, i, target + 1);
                if (l == i + 1 || get_by_sum(l, i) != target)
                    continue;

                r -- ;  // 实际上 r 的位置刚好比最右侧的分界点大 1
                
                auto & local_w = w[j - 1];      // 上一个轮次的
                auto & local_tr = tr[j - 1];
                for (; nxt[j - 1] < r; ++ nxt[j - 1]) {
                    int x = nxt[j - 1], y = f[x][j - 1];
                    bit_modify(local_tr, local_w, x, y);
                }

                // ATTENTION l-1: 思考 上一个区间的结束位置自l左侧一个位置开始
                int min_val = bit_query(local_tr, local_w, max(l - 1, 1), r);
                if (j == 1 && get_by_sum(1, i) == avs[j - 1])
                    min_val = 0;

                f[i][j] = min_val + nums[i - 1];
            }
        
        return f[n][m] < INF / 2 ? f[n][m] : -1;
    }
};
```

##### **C++ RMQ(学习记录&的方法) + BIT**

```cpp
class Solution {
public:
    // 搜索TLE (TODO: revisit 剪枝策略)
    //
    // 考虑 dp
    //  f[i][j] 表示考虑前i个元素切分成j段的最小值和
    //  => f[i][j] = min(f[x][j-1]) + nums[i]
    //     其中x需满足一定条件: and[x+1,j] = andVlues[j]
    //  由与操作性质容易推导 x必然锁定到一个连续区间 接下来就是查询这个区间的最小值
    // 
    // 拆解子问题:
    // 1. 如何计算得到x的区间 => 二分 => 需能快速获取特定区间的 and 值 => 前缀和/rmq
    // 2. 如何维护区间最小值 => BIT/单调队列(更优)
    //
    // 【数据结构优化DP】

    const static int N = 1e4 + 10, M = 11, K = 18, INF = 0x3f3f3f3f;

    // [no] prefix sum => RMQ
    int sum[N][K];
    int get_by_sum(int l, int r) {
        int len = r - l + 1;
        int k = log(len) / log(2);
        return sum[l][k] & sum[r - (1 << k) + 1][k];
    }
    
    // BIT
    // - 区间最值
    // - 需要nxt数组记录对于第j段下次应当加入bit的位置
    int w[M][N], tr[M][N], nxt[M];
    int lowbit(int x) {
        return x & -x;
    }
    void bit_modify(int tr[], int w[], int x, int y) {
        w[x] = y;
        for (int i = x; i < N; i += lowbit(i))
            tr[i] = min(tr[i], y);  // min
    }
    int bit_query(int tr[], int w[], int l, int r) {
        int ret = INF;  // ATTENTION debug
        for (; l <= r; ) {
            ret = min(ret, w[r]);
            for ( -- r ; r >= l + lowbit(r); r -= lowbit(r))
                ret = min(ret, tr[r]);
        }
        return ret;
    }

    void init() {
        {
            memset(sum, 0, sizeof sum);
            // 倍增RMQ
            for (int j = 0; j < K; ++ j )
                for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                    if (!j)
                        sum[i][j] = nums[i - 1];
                    else
                        // ATTENTION RMQ 维护性质变种
                        //  下标非常容易写疵...
                        sum[i][j] = sum[i][j - 1] & sum[i + (1 << j - 1)][j - 1];
        }
        {
            memset(w, 0x3f, sizeof w);
            memset(tr, 0x3f, sizeof tr);
            for (int i = 0; i < M; ++ i )
                nxt[i] = 1;
        }
    }

    int search(int L, int R, int x) {
        int l = L, r = R + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (get_by_sum(mid, R) >= x)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    int f[N][M];

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();

        init();

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                int target = andValues[j - 1];

                // ATTENTION 减少无效计算
                if (get_by_sum(1, i) > target)
                    continue;
                
                // ATTENTION: 思考
                // 找到 第一个 >= target 的位置  &  第一个 > target 的位置
                //  l,r 本质都是上一个轮次的下标
                int l = search(1, i, target), r = search(1, i, target + 1);
                if (l == i + 1 || get_by_sum(l, i) != target)
                    continue;

                r -- ;  // 实际上 r 的位置刚好比最右侧的分界点大 1
                
                auto & local_w = w[j - 1];      // 上一个轮次的
                auto & local_tr = tr[j - 1];
                for (; nxt[j - 1] < r; ++ nxt[j - 1]) {
                    int x = nxt[j - 1], y = f[x][j - 1];
                    bit_modify(local_tr, local_w, x, y);
                }

                // ATTENTION l-1: 思考 上一个区间的结束位置自l左侧一个位置开始
                int min_val = bit_query(local_tr, local_w, max(l - 1, 1), r);
                if (j == 1 && get_by_sum(1, i) == avs[j - 1])
                    min_val = 0;

                f[i][j] = min_val + nums[i - 1];
            }
        
        return f[n][m] < INF / 2 ? f[n][m] : -1;
    }
};
```

##### **C++ RMQ(前缀和写法略) + 单调队列**

```cpp
class Solution {
public:
    // 搜索TLE (TODO: revisit 剪枝策略)
    //
    // 考虑 dp
    //  f[i][j] 表示考虑前i个元素切分成j段的最小值和
    //  => f[i][j] = min(f[x][j-1]) + nums[i]
    //     其中x需满足一定条件: and[x+1,j] = andVlues[j]
    //  由与操作性质容易推导 x必然锁定到一个连续区间 接下来就是查询这个区间的最小值
    // 
    // 拆解子问题:
    // 1. 如何计算得到x的区间 => 二分 => 需能快速获取特定区间的 and 值 => 前缀和/rmq
    // 2. 如何维护区间最小值 => BIT/单调队列(更优)
    //
    // 【数据结构优化DP】

    const static int N = 1e4 + 10, M = 11, K = 18, INF = 0x3f3f3f3f;

    // [no] prefix sum => RMQ
    int sum[N][K];
    int get_by_sum(int l, int r) {
        int len = r - l + 1;
        int k = log(len) / log(2);
        return sum[l][k] & sum[r - (1 << k) + 1][k];
    }
    
    // [no] BIT => deque
    // - 区间最值
    // - 需要nxt数组记录对于第j段下次应当加入bit的位置
    deque<int> q[M];
    int nxt[M];

    void init() {
        {
            memset(sum, 0, sizeof sum);
            // 倍增RMQ
            for (int j = 0; j < K; ++ j )
                for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                    if (!j)
                        sum[i][j] = nums[i - 1];
                    else
                        // ATTENTION RMQ 维护性质变种
                        //  下标非常容易写疵...
                        sum[i][j] = sum[i][j - 1] & sum[i + (1 << j - 1)][j - 1];
        }
        {
            for (int i = 0; i < M; ++ i )
                nxt[i] = 1, q[i].clear();
        }
    }

    int search(int L, int R, int x) {
        int l = L, r = R + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (get_by_sum(mid, R) >= x)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    int f[N][M];

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();

        init();

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j ) {
                int target = andValues[j - 1];

                // ATTENTION 减少无效计算
                if (get_by_sum(1, i) > target)
                    continue;
                
                // ATTENTION: 思考
                // 找到 第一个 >= target 的位置  &  第一个 > target 的位置
                //  l,r 本质都是上一个轮次的下标
                int l = search(1, i, target), r = search(1, i, target + 1);
                if (l == i + 1 || get_by_sum(l, i) != target)
                    continue;

                r -- ;  // 实际上 r 的位置刚好比最右侧的分界点大 1
                
                auto & local_q = q[j - 1];      // 上一个轮次的

                for (; nxt[j - 1] < r; ++ nxt[j - 1]) {
                    while (!local_q.empty() && f[local_q.back()][j -1] >= f[nxt[j - 1]][j - 1])
                        local_q.pop_back();
                    local_q.push_back(nxt[j - 1]);
                }
                // ATTENTION: 因为前面可能会加进去小于l-1的，所以pop_front要放在后面
                // ATTENTION l-1: 思考 上一个区间的结束位置自l左侧一个位置开始
                while (!local_q.empty() && local_q.front() < l - 1)
                    local_q.pop_front();

                int min_val = INF;
                if (!local_q.empty())
                    min_val = f[local_q.front()][j - 1];
                if (j == 1 && get_by_sum(1, i) == avs[j - 1])
                    min_val = 0;

                f[i][j] = min_val + nums[i - 1];
            }
        
        return f[n][m] < INF / 2 ? f[n][m] : -1;
    }
};
```

##### **C++ LogTrick (打掉一个log) + 单调队列(BIT略)**

```cpp
class Solution {
public:
    using PII = pair<int, int>;

    template <typename T, typename F = function<T(const T &, const T &)>>
    struct SparseTable {
        int n;
        vector<vector<T>> mat;
        F func;

        SparseTable() {}    // For vector
        SparseTable(const vector<T> & a, const F & f) : func(f) {
            n = a.size();
            if (n == 0)
                return;
            int maxLog = 32 - __builtin_clz(n);
            mat.resize(maxLog);
            mat[0] = a;
            mat[0].insert(mat[0].begin(), 0);
            for (int j = 1; j < maxLog; ++ j ) {
                mat[j].resize(n - (1 << j) + 1 + 1);
                // i = n - (1 << j) + 1, 故申请内存需要再+1
                for (int i = 1; i + (1 << j) - 1 <= n; ++ i )
                    mat[j][i] = func(mat[j - 1][i], mat[j - 1][i + (1 << j - 1)]);
            }
        }
        T query(int l, int r) const {
            assert(0 <= l && l <= r && r <= n - 1);
            int lg = 31 - __builtin_clz(r - l + 1);
            return func(mat[lg][l], mat[lg][r - (1 << lg) + 1]);
        }
    };

    vector<int> nums;
    vector<int> avs;
    int n, m;

    int minimumValueSum(vector<int>& nums, vector<int>& andValues) {
        this->nums = nums, this->avs = andValues;
        this->n = nums.size(), this->m = avs.size();
        
        const int INF = 0x3f3f3f3f;
        vector<int> dp(n + 1, INF);
        dp[0] = 0;
        for (int parts = 1; parts <= m; ++ parts ) {
            SparseTable st(dp, [&](int i, int j) {return min(i, j);});
            vector<int> ndp(n + 1, INF);

            vector<PII> G;  // (and, left endpoint)
            for (int i = 1; i <= n; ++ i ) {
                int x = nums[i - 1];
                G.push_back({x, i});
                for (auto & g : G)
                    g.first &= x;
                G.erase(unique(G.begin(), G.end(), [](const PII & g1, const PII & g2) {
                    // ATTENTION trick: 对于相同的保留第一个
                    return g1.first == g2.first;
                }), G.end());

                // ATTENTION: G.size() <= log(bit_width)
                for (int j = 0; j < G.size(); ++ j ) {
                    int l = G[j].second, r = (j == G.size() - 1 ? i : G[j + 1].second - 1);
                    if (G[j].first == avs[parts - 1])
                        ndp[i] = min(ndp[i], x + st.query(l, r));
                }
            }

            swap(dp, ndp);
        }
        
        return dp[n] >= INF / 2 ? -1 : dp[n];
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

### Trie 优化

> [!NOTE] **[LeetCode 2977. 转换字符串的最小成本 II](https://leetcode.cn/problems/minimum-cost-to-convert-string-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> Trie + DP

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 2e5 + 10, M = 26, K = 2e3 + 10;  // trie树节点上限: 1000 * 100 * 2(两个串) = 2e5
    const static LL INF = 1e16;

    int son[N][M], idx, id;
    unordered_map<string, int> hash;
    unordered_map<int, int> mem;

    void init() {
        memset(son, 0, sizeof son);
        idx = 0, id = 0;
        hash.clear(), mem.clear();
    }
    void insert(string & s) {
        if (hash.count(s))
            return;
        int p = 0;
        for (auto c : s) {
            int u = c - 'a';
            if (!son[p][u])
                son[p][u] = ++ idx ;
            p = son[p][u];
        }
        hash[s] = ++ id ;   // 从 1 开始
        mem[p] = id;        // 顺便记录树节点与对应值
    }

    LL w[K][K], f[K / 2];

    long long minimumCost(string source, string target, vector<string>& original, vector<string>& changed, vector<int>& cost) {
        init();

        {
            // 初始化 w
            memset(w, 0x3f, sizeof w);
            for (int i = 0; i < K; ++ i )
                w[i][i] = 0;
            for (int i = 0; i < original.size(); ++ i ) {
                string a = original[i], b = changed[i];
                // 插入 trie 同时记录离散化
                insert(a), insert(b);
                int x = hash[a], y = hash[b];
                w[x][y] = min(w[x][y], (LL)cost[i]);
            }

            // floyd
            for (int k = 1; k <= id; ++ k )
                for (int i = 1; i <= id; ++ i )
                    for (int j = 1; j <= id; ++ j )
                        w[i][j] = min(w[i][j], w[i][k] + w[k][j]);
        }

        int n = source.size();
        f[n] = 0;   // 边界值
        for (int i = n - 1; i >= 0; -- i ) {
            f[i] = INF;
            
            if (source[i] == target[i])
                f[i] = f[i + 1];

            // 从当前位置往后 找到可行串与对应消耗
            for (int j = i, p1 = 0, p2 = 0; j < n; ++ j ) {
                int u1 = source[j] - 'a', u2 = target[j] - 'a';
                p1 = son[p1][u1], p2 = son[p2][u2];
                if (p1 == 0 || p2 == 0)
                    break;
                // 如果存在两个对应的串
                if (mem[p1] && mem[p2])
                    // ATTENTION
                    f[i] = min(f[i], f[j + 1] + w[mem[p1]][mem[p2]]);
            }
        }

        if (f[0] >= INF / 2)
            return -1;
        return f[0];
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

### bitset 优化

> [!NOTE] **[LeetCode 3181. 执行操作可获得的最大总奖励 II](https://leetcode.cn/problems/maximum-total-reward-using-operations-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 根据题意及数据范围推导 值一定不超过 1e5 => 想到 bitset
> 
> 消除 bitset 部分 0 的姿势

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 根据题意理解及数据范围 总和一定不会超过 max(rewardValues)*2 => 思考
    // 可以用 bitset 快速维护
    const static int N = 5e4 + 10, M = 1e5 + 10;
    
    bitset<M> f;
    
    int maxTotalReward(vector<int>& rewardValues) {
        int n = rewardValues.size();
        sort(rewardValues.begin(), rewardValues.end());
        
        f[0] = 1;
        for (int i = 0; i < n; ++ i ) {
            int x = rewardValues[i];
            // ATTENTION 重点在于 约束只看较低的 x 位
            // for (int j = 0; j < x; ++ j )
            //     f[j + x] = f[j + x] || f[j];
            
            // ATTENTION: trick
            // 先左移 d 消除前面的 0, 再右移 d 恢复, 最后左移 x 实现状态转移
            int d = M - x;
            // f |= f << d >> d << x;
        }
        
        for (int j = M - 1; j >= 0; -- j )
            if (f[j])
                return j;
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