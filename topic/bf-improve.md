## 习题

### 数学分析

> [!NOTE] **[LeetCode 2791. 树中可以形成回文的路径数](https://leetcode.cn/problems/count-paths-that-can-form-a-palindrome-in-a-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典的暴力优化思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 关键点：可以重新排列
    // 则 只需要关心两个点之间的不同字符的奇偶性即可
    //
    // 又因为 1e5 数据范围显然不能两两枚举
    // 考虑直接统计从根到某个节点的所有字符的奇偶状态 并全局计数
    // 最后遍历节点即可
    using LL = long long;
    const static int N = 1e5 + 10, M = N;
    
    int h[N], e[M], w[M], ne[M], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    unordered_map<int, int> count;
    int st[N];
    void dfs(int u, int s) {
        st[u] = s;
        count[s] ++ ;
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            int t = w[i];
            dfs(j, s ^ (1 << t));
        }
    }
    
    long long countPalindromePaths(vector<int>& parent, string s) {
        init();
        int n = parent.size();
        for (int i = 1; i < n; ++ i )
            add(parent[i], i, s[i] - 'a');
        
        count.clear();
        memset(st, 0, sizeof st);
        dfs(0, 0);
        
        LL res = 0;
        for (int i = 0; i < n; ++ i ) {
            int a = st[i];
            // 奇偶性相同的情况 要排除自身
            res = res + ((LL)count[a] - 1);
            // 奇偶性不同的情况
            for (int j = 0; j < 26; ++ j ) {
                int b = a ^ (1 << j);
                res = res + (LL)count[b];
            }
        }
        return res / 2;
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

> [!NOTE] **[LeetCode 2949. 统计美丽子字符串 II](https://leetcode.cn/problems/count-beautiful-substrings-ii/) [TAG]**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 难点在于思路梳理 以及索引组织
> 
> 结合数学分析推导最高效的做法
> 
> 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 基础 O(n*sqrt(k))**

```cpp
class Solution {
public:
    // 考虑记录元音字母的数量及对应的位置 则辅音可求
    //
    // 考虑: 以每一个位置为结尾 往前数能找到的合法区间
    //   => 问题在于 合法区间怎么确定
    //  假定 [i, j] 合法, 则
    //  - 长度为偶数 j-i+1 % 2 == 0                      => 下标奇偶分类可解 => 在后面2.的条件下不需要分类
    //  - 元音个数与偶数相同
    //       -> 1. 按个数是一半 则需要遍历校验
    //       -> 2. 按个数相同【记录diff 取模】 可以快速找到所有符合要求的 再判断下能否整除就好做了
    
    using LL = long long;
    const static int N = 5e4 + 10;
    
    // unordered_set<char> S = {'a', 'e', 'i', 'o', 'u'};
    bool check(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
    
    // vector<LL> f[N + N];                         // 某个diff下，下标是多少
    unordered_map<int, unordered_map<int, int>> f;  // 某个diff下，记录元音个数对k取模的次数【后面推导】
                                                    // unordered_map<int, int> f[N + N] => 会超时 改成两层嵌套
    int sum[N];
    
    long long beautifulSubstrings(string s, int k) {
        // for (int i = 0; i < N + N; ++ i )
        //     f[i].clear();
        // f[0 + N].push_back(0);
        f[0][0 % k] = 1;
        memset(sum, 0, sizeof sum);
        
        vector<int> wd2; // w divide 2
        for (int i = 0; i < k; ++ i ) {    // ATTENTION: 考虑只枚举到 k, 但是需要从 0 开始枚举
            if (i * i % k == 0)
                wd2.push_back(i);
        }
        
        LL res = 0;
        int n = s.size();
        for (int i = 1, d = 0; i <= n; ++ i ) {
            // bool flag = S.count(s[i - 1]);
            bool flag = check(s[i - 1]);
            d += flag ? 1 : -1;
            sum[i] = sum[i - 1] + flag;
            
            // xs 中都是 diff 相同的，满足个数与长度要求
            auto & xs = f[d];
            
            // 【ATTENTION 循环对于该数据范围太过重型 考虑是否有更快速索引办法】
            // for (auto x : xs) {
            //     // l: x+1, r: i
            //     int w = (i - x) / 2;
            //     if (w * w % k == 0)
            //         res ++ ;
            // }
            //
            // 考虑 w*w%k == 0    <=> (w/2%k * w/2%k) % k == 0
            //                    ==> 思考【w/2必然是k的倍数?】不 => 因为 /2 的影响
            //                    ==> 又因为k不定 求逆元比较困难 考虑直接枚举可行的 w/2 记为wd2
            // 此时: 如果使用 i-2*x 去枚举左端点，复杂度仍然很高
            //
            // 考虑如何避免枚举:    重新定义 w = s[r]-s[l] 其也是长度的一半 同时 w*w%k == 0 (w%k==0)
            //                    则 所有可行的l 都能使得 (s[l]-s[r])%k == 0
            //                    => s[l]%k == s[r]%k
            //                    => 同余 则 f 的第二维按照模数维护即可
            for (auto x : wd2) {
                int t = (sum[i] - x + k) % k;
                if (xs.count(t))
                    res += xs[t];
            }
            
            xs[sum[i] % k] ++ ;
        }
        
        return res;
    }
};
```

##### **C++ 数学 O(n)**

```cpp
class Solution {
public:
    // 前面已经推导知: 假定长度为 w, 有 (w/2)*(w/2)%k == 0
    //
    // 0x3f 的思路:  不是转换逆元, 而是把 /2 提取出来 => w*w % (4*k) == 0
    //  TRICK  => 数学思维: 假定 K=4*k, 而 K 由 p1^e1 * p2^e2 ... 组成
    //                 则: w = p1^((e1+1)/2) * ...  【幂次除二上取整即可】
    //
    // 接下来 w 必须是新的 k 的倍数
    //        (r - l) % k == 0 => r % k == l % k
    // 按照下标进行分组计数   且需要保证 diff 相同(使用状态压缩实现)
    //
    // 代码实现: 把 {下标分组, 状态表示} 合并成一维表示了
    
    using LL = long long;
    const static int N = 5e4 + 10;
    
    int p_sqrt(int x) { // x 不会超过 4000
        int res = 1;
        for (int i = 2; i * i <= x; ++ i ) {
            // 计算 x 包含多少个 i 的幂次
            int t = 0;
            while (x % i == 0)
                x /= i, t ++ ;
            
            for (int j = 0; j < (t + 1) / 2; ++ j )
                res *= i;
        }
        if (x > 1)
            res *= x;   // 细节
        return res;
    }

    //                              a + e + i + o + u
    constexpr static int AEIOU_MASK = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 14) + (1 << 20);
    
    long long beautifulSubstrings(string s, int k) {
        int n = s.size();
        k = p_sqrt(k * 4);
        
        unordered_map<int, int> f;
        LL res = 0, sum = N;        // sum=N而非sum=0 是因为其值可能为负数 为了能用int表示而做的偏移
        f[0 << 17 | sum] ++ ;       // 插入 f[{0, N}] N是偏移量
        for (int i = 1; i <= n; ++ i ) {
            int bit = (AEIOU_MASK >> (s[i - 1] - 'a')) & 1;
            sum += bit ? 1 : -1;
            res += f[(i % k) << 17 | sum] ++ ;
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

> [!NOTE] **[LeetCode 2973. 树中每个节点放置的金币数目](https://leetcode.cn/problems/find-number-of-coins-to-place-in-tree-nodes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数学分析 优化数据范围

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
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
    vector<int> cs;
    vector<LL> res;

    vector<int> dfs(int u, int fa) {
        vector<int> ret = {cs[u]};
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == fa)
                continue;
            auto t = dfs(j, u);
            ret.insert(ret.end(), t.begin(), t.end());
        }

        sort(ret.begin(), ret.end());
        int m = ret.size();
        // 求最值
        if (m >= 3) {
            res[u] = max({(LL)ret[m - 3] * ret[m - 2] * ret[m - 1],
                        (LL)ret[0] * ret[1] * ret[m - 1], 0ll});
        }
        // ATTENTION 收缩不必要的区间
        if (m > 5)
            ret = {ret[0], ret[1], ret[m - 3], ret[m - 2], ret[m - 1]};
        return ret;
    }

    vector<long long> placedCoins(vector<vector<int>>& edges, vector<int>& cost) {
        init();
        {
            for (auto & e : edges)
                add(e[0], e[1]), add(e[1], e[0]);
        }

        this-> cs = cost;
        this->n = cs.size();

        res = vector<LL>(n, 1);
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

> [!NOTE] **[LeetCode 3027. 人员站位的方案数 II](https://leetcode.cn/problems/find-the-number-of-ways-to-place-people-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 充要条件推导 以简化暴力遍历过程

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 最初比较糙的版本**

```cpp
class Solution {
public:
    // ATTENTION 题意是 内部或边缘...而非只排排除边缘 显然需要二维前缀合
    const static int N = 1010;      // ATTENTION 1: 只需要 1010 不需要 2010，因为 xy 彼此独立
    
    int g[N][N];
    void init() {
        memset(g, 0, sizeof g);
    }
    
    vector<int> xs, ys;
    int get(vector<int> s, int x) {
        return lower_bound(s.begin(), s.end(), x) - s.begin();
    }
    
    bool check(int x1, int y1, int x2, int y2) {
        return g[x2][y1] - g[x2][y2 - 1] - g[x1 - 1][y1] + g[x1 - 1][y2 - 1] == 2;
    };
    
    int numberOfPairs(vector<vector<int>>& points) {
        // sorting...
        sort(points.begin(), points.end(), [](const vector<int> & a, const vector<int> & b) {
            if (a[0] == b[0])
                return a[1] > b[1];
            return a[0] < b[0];
        });
        
        {
            for (auto & p : points) {
                int x = p[0], y = p[1];
                xs.push_back(x), ys.push_back(y);
            }
            sort(xs.begin(), xs.end());
            sort(ys.begin(), ys.end());
        }
        {
            init();
            for (auto & p : points) {
                int x = get(xs, p[0]) + 1, y = get(ys, p[1]) + 1;
                g[x][y] ++ ;
                p[0] = x, p[1] = y;
            }
            for (int i = 1; i < N; ++ i )
                for (int j = 1; j < N; ++ j )
                    g[i][j] += g[i - 1][j] + g[i][j - 1] - g[i - 1][j - 1];
        }
        
        int n = points.size(), res = 0;
        for (int i = 0; i < n; ++ i ) {
            int x1 = points[i][0], y1 = points[i][1], cap = 0;  // ATTENTION 2: trick 优化 => 理论上如果某个点符合要求，则后续必须比这个点更高
            for (int j = i + 1; j < n; ++ j ) {
                int x2 = points[j][0], y2 = points[j][1];
                if (y2 > y1)
                    continue;
                
                // ATTENTION 2.1
                if (y2 <= cap)
                    continue;
                
                if (check(x1, y1, x2, y2)) {
                    res ++ ;
                    
                    // ATTENTION 2.2
                    cap = max(cap, y2);
                }
            }
        }
        return res;
    }
};
```

##### **C++ 简化**

```cpp
class Solution {
public:
    // ATTENTION 更进一步 只要保证该上限即可满足条件 【思考 充要条件】
    
    int numberOfPairs(vector<vector<int>>& points) {
        // sorting...
        sort(points.begin(), points.end(), [](const vector<int> & a, const vector<int> & b) {
            if (a[0] == b[0])
                return a[1] > b[1];
            return a[0] < b[0];
        });
        
        int n = points.size(), res = 0;
        for (int i = 0; i < n; ++ i ) {
            int x1 = points[i][0], y1 = points[i][1], cap = -2e9;  // ATTENTION 2: trick 优化 => 理论上如果某个点符合要求，则后续必须比这个点更高
            for (int j = i + 1; j < n; ++ j ) {
                int x2 = points[j][0], y2 = points[j][1];
                if (y2 > y1)
                    continue;
                
                // ATTENTION 2.1
                if (y2 <= cap)
                    continue;
                
                res ++ ;
                // ATTENTION 2.2
                cap = max(cap, y2);
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

### 数学 - LogTrick

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

> [!NOTE] **[LeetCode 898. 子数组按位或操作](https://leetcode.cn/problems/bitwise-ors-of-subarrays/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 和 [Codeforces The Brand New Function](http://codeforces.com/problemset/problem/243/A) 一个意思
> 
> 非常经典的通过数学推导优化暴力

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    int subarrayBitwiseORs(vector<int>& arr) {
        unordered_set<int> S;
        int n = arr.size();
        for (int i = 0; i < n; ++ i ) {
            S.insert(arr[i]);
            for (int j = i - 1; j >= 0; -- j ) {
                // ATTENTION 注意括号 (x | y) == z
                if ((arr[j] | arr[i]) == arr[j])
                    break;
                arr[j] |= arr[i];
                S.insert(arr[j]);
            }
        }
        return S.size();
    }
};
```

##### **C++ 类 Vector 写法**

```cpp
class Solution {
public:
    int subarrayBitwiseORs(vector<int>& arr) {
        unordered_set<int> S, pre;
        for (auto x : arr) {
            unordered_set<int> next;
            next.insert(x);
            for (auto y : pre)
                next.insert(x | y);
            for (auto y : next)
                S.insert(y);
            pre = next;
        }
        return S.size();
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

> [!NOTE] **[LeetCode 3171. 找到按位或最接近 K 的子数组](https://leetcode.cn/problems/find-subarray-with-bitwise-or-closest-to-k/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 LogTrick
> 
> 子数组按位 `或`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minimumDifference(vector<int>& nums, int k) {
        int res = 2e9;
        int n = nums.size();
        for (int i = 0; i < n; ++ i ) {
            res = min(res, abs(nums[i] - k));
            for (int j = i - 1; j >= 0; -- j ) {
                if ((nums[j] | nums[i]) == nums[j])
                    break;
                nums[j] |= nums[i];
                res = min(res, abs(nums[j] - k));
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

> [!NOTE] **[LeetCode 1521. 找到最接近目标值的函数值](https://leetcode.cn/problems/find-a-value-of-a-mysterious-function-closest-to-target/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可以部分参考 [898. 子数组按位或操作](https://leetcode.cn/problems/bitwise-ors-of-subarrays/)
> 
> - 双指针 + 前缀和
> 
> - 动态维护 TODO clear => LogTrick
> 
> - 模拟退火 TODO clear

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    int closestToTarget(vector<int>& arr, int target) {
        int res = 1e9;
        int n = arr.size();
        for (int i = 0; i < arr.size(); ++ i ) {
            res = min(res, abs(arr[i] - target));
            for (int j = i - 1; j >= 0; -- j ) {
                if ((arr[j] & arr[i]) == arr[j])
                    break;
                arr[j] &= arr[i];
                res = min(res, abs(arr[j] - target));
            }
        }
        return res;
    }
};
```

##### **C++ 动态维护 非原地修改**

```cpp
class Solution {
public:
    int closestToTarget(vector<int>& arr, int target) {
        int ans = abs(arr[0] - target);
        vector<int> valid = {arr[0]};
        for (int num : arr) {
            vector<int> validNew = {num};
            ans = min(ans, abs(num - target));
            for (int prev : valid) {
                validNew.push_back(prev & num);
                ans = min(ans, abs((prev & num) - target));
            }
            validNew.erase(unique(validNew.begin(), validNew.end()),
                           validNew.end());
            valid = validNew;
        }
        return ans;
    }
};
```

##### **C++ 双指针+前缀和**

```cpp
class Solution {
public:
    // 1e6 数据范围 则最多不超过 20 个不同的值 => 思考
    const static int N = 1e5 + 10, M = 20;

    int s[N][M];    // 逆序思维: 不记录有没有 1 而是记录某一位有没有 0

    int get_sum(int l, int r) {
        int res = 0;
        for (int i = 0; i < M; ++ i )
            if (s[r][i] - s[l - 1][i] == 0) // 没有 0 存在
                res += 1 << i;
        return res;
    }

    int closestToTarget(vector<int>& arr, int target) {
        int n = arr.size();

        memset(s, 0, sizeof s);
        for (int i = 1; i <= n; ++ i )
            for (int j = 0; j < M; ++ j ) {
                s[i][j] = s[i - 1][j];
                if (!(arr[i - 1] >> j & 1))
                    s[i][j] ++ ;
            }
        
        int res = INT_MAX;
        for (int l = 1, r = 1; r <= n; ++ r ) {
            while (l < r && abs(get_sum(l + 1, r)) <= target)
                l ++ ;
            res = min(res, abs(get_sum(l, r) - target));
            if (l < r)
                res = min(res, abs(get_sum(l + 1, r) - target));
        }
        return res;
    }
};
```

##### **C++ 模拟退火**

```cpp
class Solution {
public:
    //通过预处理，快速求解arr[L..R]的与值
    int pre[100001][20] = {0};

    int get(int L, int R, int target) {
        int val = 0;
        for (int i = 0, bit = 1; i < 20; i++, bit <<= 1)
            // 如果第 i 个bit 在 [L,R] 中全为 1，那么与值的该bit也必然为 1。
            if (pre[R][i] - pre[L - 1][i] == R - L + 1) { val |= bit; }
        return abs(val - target);
    }

    // 用模拟退火求解关于 L 的局部最优解
    int query(int L, int n, int target) {
        int dir[2] = {-1, 1};  // 两个方向
        int step = 1000;       // 初始步长
        int now = L;           // R 的起始位置
        int best = 100000000;  // 局部最优解

        while (step > 0) {
            int Lpos = now + step * dir[0];
            if (Lpos < L) Lpos = L;
            int Rpos = now + step * dir[1];
            if (Rpos > n) Rpos = n;
            // 向左右两个方向各走一步，求值
            int ldis = get(L, Lpos, target);
            int rdis = get(L, Rpos, target);
            int pbest = best;

            //更新位置及最优解
            if (ldis < best) {
                now = Lpos;
                best = ldis;
            }
            if (rdis < best) {
                now = Rpos;
                best = rdis;
            }

            //如果没有找到更优解，那就缩小步长
            if (pbest == best) { step /= 2; }
        }
        return best;
    }

    int closestToTarget(vector<int>& arr, int target) {
        int anw = 100000000;

        //统计前 i 个数字中，第 j 个bit 为 1 的数量。
        for (int i = 0; i < arr.size(); i++)
            for (int j = 0, bit = 1; j < 20; j++, bit <<= 1)
                pre[i + 1][j] = pre[i][j] + ((bit & arr[i]) ? 1 : 0);

        for (int i = 1; i <= arr.size(); i++)
            anw = min(anw, query(i, arr.size(), target));

        return anw;
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

> [!NOTE] **[LeetCode 3209. 子数组按位与值为 K 的数目](https://leetcode.cn/problems/number-of-subarrays-with-and-value-of-k/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> LogTrick 综合应用 (不只是求值域集合 还有求数量)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    
    long long countSubarrays(vector<int>& nums, int k) {
        int n = nums.size();
        LL res = 0;
        for (int i = 0, l = 0, r = 0; i < n; ++ i ) {
            int x = nums[i];
            for (int j = i - 1; j >= 0; -- j ) {
                if ((nums[j] & x)/*ATTENTION 括号*/ == nums[j])
                    break;
                nums[j] &= x;
            }
            res += lower_bound(nums.begin(), nums.begin() + i + 1, k + 1) -
                   lower_bound(nums.begin(), nums.begin() + i + 1, k);
            // while (l <= i && nums[l] < k)
            //     l ++ ;
            // while (r <= i && nums[r] <= k)
            //     r ++ ;
            // res += r - l;
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

### 双指针

> [!NOTE] **[LeetCode 1521. 找到最接近目标值的函数值](https://leetcode.cn/problems/find-a-value-of-a-mysterious-function-closest-to-target/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可以部分参考 [898. 子数组按位或操作](https://leetcode.cn/problems/bitwise-ors-of-subarrays/)
> 
> - 双指针 + 前缀和
> 
> - 动态维护 TODO clear
> 
> - 模拟退火 TODO clear

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 双指针+前缀和**

```cpp
class Solution {
public:
    // 1e6 数据范围 则最多不超过 20 个不同的值 => 思考
    const static int N = 1e5 + 10, M = 20;

    int s[N][M];    // 逆序思维: 不记录有没有 1 而是记录某一位有没有 0

    int get_sum(int l, int r) {
        int res = 0;
        for (int i = 0; i < M; ++ i )
            if (s[r][i] - s[l - 1][i] == 0) // 没有 0 存在
                res += 1 << i;
        return res;
    }

    int closestToTarget(vector<int>& arr, int target) {
        int n = arr.size();

        memset(s, 0, sizeof s);
        for (int i = 1; i <= n; ++ i )
            for (int j = 0; j < M; ++ j ) {
                s[i][j] = s[i - 1][j];
                if (!(arr[i - 1] >> j & 1))
                    s[i][j] ++ ;
            }
        
        int res = INT_MAX;
        for (int l = 1, r = 1; r <= n; ++ r ) {
            while (l < r && abs(get_sum(l + 1, r)) <= target)
                l ++ ;
            res = min(res, abs(get_sum(l, r) - target));
            if (l < r)
                res = min(res, abs(get_sum(l + 1, r) - target));
        }
        return res;
    }
};
```

##### **C++ 动态维护**

```cpp
class Solution {
public:
    int closestToTarget(vector<int>& arr, int target) {
        int ans = abs(arr[0] - target);
        vector<int> valid = {arr[0]};
        for (int num : arr) {
            vector<int> validNew = {num};
            ans = min(ans, abs(num - target));
            for (int prev : valid) {
                validNew.push_back(prev & num);
                ans = min(ans, abs((prev & num) - target));
            }
            validNew.erase(unique(validNew.begin(), validNew.end()),
                           validNew.end());
            valid = validNew;
        }
        return ans;
    }
};
```

##### **C++ 模拟退火**

```cpp
class Solution {
public:
    //通过预处理，快速求解arr[L..R]的与值
    int pre[100001][20] = {0};

    int get(int L, int R, int target) {
        int val = 0;
        for (int i = 0, bit = 1; i < 20; i++, bit <<= 1)
            // 如果第 i 个bit 在 [L,R] 中全为 1，那么与值的该bit也必然为 1。
            if (pre[R][i] - pre[L - 1][i] == R - L + 1) { val |= bit; }
        return abs(val - target);
    }

    // 用模拟退火求解关于 L 的局部最优解
    int query(int L, int n, int target) {
        int dir[2] = {-1, 1};  // 两个方向
        int step = 1000;       // 初始步长
        int now = L;           // R 的起始位置
        int best = 100000000;  // 局部最优解

        while (step > 0) {
            int Lpos = now + step * dir[0];
            if (Lpos < L) Lpos = L;
            int Rpos = now + step * dir[1];
            if (Rpos > n) Rpos = n;
            // 向左右两个方向各走一步，求值
            int ldis = get(L, Lpos, target);
            int rdis = get(L, Rpos, target);
            int pbest = best;

            //更新位置及最优解
            if (ldis < best) {
                now = Lpos;
                best = ldis;
            }
            if (rdis < best) {
                now = Rpos;
                best = rdis;
            }

            //如果没有找到更优解，那就缩小步长
            if (pbest == best) { step /= 2; }
        }
        return best;
    }

    int closestToTarget(vector<int>& arr, int target) {
        int anw = 100000000;

        //统计前 i 个数字中，第 j 个bit 为 1 的数量。
        for (int i = 0; i < arr.size(); i++)
            for (int j = 0, bit = 1; j < 20; j++, bit <<= 1)
                pre[i + 1][j] = pre[i][j] + ((bit & arr[i]) ? 1 : 0);

        for (int i = 1; i <= arr.size(); i++)
            anw = min(anw, query(i, arr.size(), target));

        return anw;
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

> [!NOTE] **[LeetCode 2831. 找出最长等值子数组](https://leetcode.cn/problems/find-the-longest-equal-subarray/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典暴力优化
> 
> 按数值对下标分类 随后同向双指针

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 考虑 直接遍历+枚举右端点 or 二分答案都不可行
    //
    // 思考 按照值分类记录下标  双指针维护
    const static int N = 1e5 + 10;
    
    vector<int> xss[N];
    
    int get(int x, int k) {
        auto & xs = xss[x];
        int n = xs.size(), ret = 0;
        for (int i = 0, j = 0, del = 0; j < n; ++ j ) {
            if (j)
                del += xs[j] - xs[j - 1] - 1;
            while (del > k && i < j) {
                i ++ ;
                del -= xs[i] - xs[i - 1] - 1;
            }
            ret = max(ret, j - i + 1);
        }
        return ret;
    }
    
    int longestEqualSubarray(vector<int>& nums, int k) {
        int n = nums.size();
        for (int i = 0; i < n; ++ i )
            xss[nums[i]].push_back(i);
        
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            res = max(res, get(i, k));
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

> [!NOTE] **[LeetCode 2935. 找出强数对的最大异或值 II](https://leetcode.cn/problems/maximum-strong-pair-xor-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典暴力优化
> 
> 双指针(单调性证明) + Trie(删除操作)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 较显然的，需要针对第一题的实现做 "暴力优化"
    //     考虑将强数对的条件发生转化: 排序后，当前位置的元素向前找，差值不能超过当前元素值的所有位置中，异或值最大的
    // => 显然 双指针扫描 + Trie维护 且结合了动态删除
    
    // ATTENTION N 的取值需要是 length * 20
    // 【因为值域原因，每个值都可能产生20个trie中的节点】
    const static int N = 5e4 * 20 + 10, M = 22;
    
    int tr[N][2], cnt[N], idx;
    
    void init() {
        memset(tr, 0, sizeof tr);
        idx = 0;
    }
    void insert(int x) {
        int p = 0;
        for (int i = M - 1; i >= 0; -- i ) {
            int u = x >> i & 1;
            if (!tr[p][u])
                tr[p][u] = ++ idx;
            p = tr[p][u];
            cnt[p] ++ ;     // ATTENTION
        }
    }
    void remove(int x) {
        int p = 0;
        for (int i = M - 1; i >= 0; -- i ) {
            int u = x >> i & 1;
            p = tr[p][u];   // 一定存在
            cnt[p] -- ;     // ATTENTION
        }
    }
    int query(int x) {
        int p = 0, ret = 0;
        for (int i = M - 1; i >= 0; -- i ) {
            int u = x >> i & 1;
            if (!tr[p][!u] || !cnt[tr[p][!u]])  // ATTENTION
                p = tr[p][u];
            else {
                ret |= 1 << i;
                p = tr[p][!u];
            }
        }
        return ret;
    }
    
    int maximumStrongPairXor(vector<int>& nums) {
        init();
        int n = nums.size(), res = 0;
        sort(nums.begin(), nums.end());
        for (int i = 0, j = 0; j < n; ++ j ) {
            while (i < j && nums[j] - nums[i] > nums[i])    // ATTENTION 思考条件
                remove(nums[i ++ ]);
            // cout << " j = " << j << " i = " << i << " query = " << query(nums[j]) << endl;
            res = max(res, query(nums[j]));
            insert(nums[j]);
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

> [!NOTE] **[LeetCode 2953. 统计完全子字符串](https://leetcode.cn/problems/count-complete-substrings/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推导 双指针(滑动窗口)优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 最简单暴力的处理是:
    //  1. 按照前缀和统计每个区间内的特定字符的数量
    //  2. 枚举区间 O(n^2) 再判断区间内出现的字符是否符合要求 O(26) + O(n)
    //  3. 统计求和
    //
    // 复杂度显然无法接受 考虑如何优化
    //  1. 对于某一个确定的右端点i 思考其可能包含多少个字符 假定为 c
    //  2. 则区间固定为 [i-c*k+1, i] 或者 (i-c*k, i]  => 枚举代价仍然无法接受
    // 
    // 尝试发现单调性以优化该过程
    // 容易想到: 对于一个确定的i 向左侧不会无限延伸 因为字符数量有上限 最左侧的边界也不能超过k
    //        且伴随着i右移 这个边界同样会右移【且由于第二个条件的存在 不会跨该条件的边界点】
    // 则 双指针维护区间 并维护区间内的种类数量以及各种类个数 【在有限的范围内进行枚举】
    
    const static int N = 1e5 + 10, M = 26;
    
    int sum[N][M];
    int n, k;
    
    int vars, bigger;
    int cnt[M];
    void init() {
        vars = 0, bigger = 0;
        memset(cnt, 0, sizeof cnt);
    }
    void add(char c) {
        int t = c - 'a';
        cnt[t] ++ ;
        if (cnt[t] == 1)
            vars ++ ;
        if (cnt[t] == k + 1)
            bigger ++ ;
    }
    void sub(char c) {
        int t = c - 'a';
        cnt[t] -- ;
        if (cnt[t] == 0)
            vars -- ;
        if (cnt[t] == k)
            bigger -- ;
    }
    bool invalid() {
        return bigger > 0;
    }
    bool split(char a, char b) {
        return abs(a - b) > 2;
    }
    bool check(int l, int r) {
        for (int i = 0; i < M; ++ i )
            if (sum[r][i] - sum[l - 1][i] != 0 && sum[r][i] - sum[l - 1][i] != k)
                return false;
        return true;
    }
    
    int countCompleteSubstrings(string word, int k) {
        this->n = word.size(), this->k = k;
        {
            memset(sum, 0, sizeof sum);
            for (int i = 1; i <= n; ++ i ) {
                for (int j = 0; j < M; ++ j )
                    sum[i][j] = sum[i - 1][j];
                sum[i][word[i - 1] - 'a'] ++ ;
            }
            
            init();
        }
        
        int res = 0;
        for (int i = 1, j = 1; j <= n; ++ j ) {
            if (j > 1 && split(word[j - 1], word[j - 1 - 1])) {
                init(); // 提前收缩区间
                i = j;
            }
            // 更新当前末尾 并收缩区间
            add(word[j - 1]);
            while (i <= j && invalid()) {
                sub(word[i - 1]);
                i ++ ;
            }
            
            // ok, 现在 [i, j] 是一个 [每一种的数量不超k & 不违背字母序差值过大的区间了]
            // 并且此时 符合条件的字母个数不会超过 vars 个 => 枚举?
            //
            // ATTENTION 条件必不可少 j-x*k+1 >= l
            for (int x = 1; x <= vars && j - x * k + 1 >= i; ++ x ) {
                int l = j - x * k + 1, r = j;
                if (check(l, r))
                    res ++ ;
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

> [!NOTE] **[LeetCode 2968. 执行操作使频率分数最大](https://leetcode.cn/problems/apply-operations-to-maximize-frequency-score/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 暴力优化 显然需要使用排序后的一段连续区间 考虑枚举区间右端点
> 
> - 二分
> 
> - 双指针

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 二分**

```cpp
class Solution {
public:
    // 数的数值范围很大 但数量只有1e5 => 考虑枚举每一个值 如果作为最终的众数 能构造多少个
    //    枚举 O(n) 随后往前往后延伸(动态边界)... 不太好接受
    //    考虑排序后枚举区间右端点(思考:为什么可以这样) 则可以二分众数个数(并快速判断)
    
    using LL = long long;
    const static int N = 1e5 + 10;
    
    vector<int> ns;
    int n;
    
    LL s[N], k;
    
    bool check(int l, int r) {
        int mid = (l + r) / 2;
        // 其他数字都要转化成 mid
        int t = ns[mid - 1];
        int lc = mid - l, rc = r - mid;
        LL tot = ((LL)lc * t - (s[mid - 1] - s[l - 1])) + ((s[r] - s[mid]) - (LL)rc * t);
        return tot <= k;
    }
    
    int maxFrequencyScore(vector<int>& nums, long long k) {
        this->ns = nums;
        this->n = ns.size();
        this->k = k;
        sort(ns.begin(), ns.end());
        {
            s[0] = 0;
            for (int i = 1; i <= n; ++ i )
                s[i] = s[i - 1] + ns[i - 1];
        }
        
        int res = 0;
        for (int i = 1; i <= n; ++ i ) {
            // 找到第一个无法满足的区间长度
            int l = 1, r = i + 1;
            while (l < r) {
                int m = l + (r - l) / 2;
                if (check(i - m + 1, i))
                    l = m + 1;
                else
                    r = m;
            }
            res = max(res, l - 1);
        }
        return res;
    }
};
```

##### **C++ 双指针**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;

    int n;
    vector<int> ns;

    LL s[N], k;

    bool valid(int l, int r) {
        int mid = (l + r) / 2;
        int tar = ns[mid - 1];
        int lc = mid - l, rc = r - mid;
        return ((LL)lc * tar - (s[mid - 1] - s[l - 1])) + ((s[r] - s[mid]) - (LL)rc * tar) <= k;
    }

    int maxFrequencyScore(vector<int>& nums, long long k) {
        this->ns = nums;
        this->n = ns.size();
        this->k = k;
        sort(ns.begin(), ns.end());
        {
            s[0] = 0;
            for (int i = 1; i <= n; ++ i )
                s[i] = s[i - 1] + ns[i - 1];
        }

        int res = 0;
        for (int i = 1, j = 1; j <= n; ++ j ) {
            while (i <= j && !valid(i, j))
                i ++ ;
            res = max(res, j - i + 1);
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

> [!NOTE] **[LeetCode 2972. 统计移除递增子数组的数目 II](https://leetcode.cn/problems/count-the-number-of-incremovable-subarrays-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 + 双指针
> 
> 注意不要漏掉【只保留左侧 l+2】的计数值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 移除一段连续序列 使得左右两侧合并为LIS(严格递增)
    // 求有多少个序列移除办法
    // 
    // 显然要先从左右两侧收缩区间 找到最小的不满足LIS的位置
    // 随后分情况讨论 + 枚举右端点
    
    using LL = long long;
    
    int n, l, r;
    
    long long incremovableSubarrayCount(vector<int>& nums) {
        this->n = nums.size();
        l = 0, r = n - 1;
        {
            while (l + 1 < n && nums[l + 1] > nums[l])
                l ++ ;
            while (r > 0 && nums[r - 1] < nums[r])
                r -- ;
        }
        if (l >= r)
            return (LL)(n + 1) * n / 2;
        
        // ATTENTION: case 1 如果只保留左侧(完全移除右侧)
        LL res = l + 2;
        
        // case 2 只保留右侧位置
        // 枚举右侧的起始位置 则左侧终止位置需要满足特定条件
        for (int i = 0, j = r; j < n; ++ j ) {
            // 找到第一个不符合的位置
            while (i <= l && nums[i] < nums[j])
                i ++ ;
            
            res += (i + 1);
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

> [!NOTE] **[LeetCode 3261. 统计满足 K 约束的子字符串数量 II](https://leetcode.cn/problems/count-substrings-that-satisfy-k-constraint-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分析问题的思路
> 
> 计算拆解 (两部分: 右侧 + 左侧)
> 
> 本题二分分界点可过 => 可以双指针进一步优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 核心: 想到对于每一个位置i 记录其左侧最长延伸距离
    
    using LL = long long;
    const static int N = 1e5 + 10;
    
    int left[N];
    LL sum[N];
    
    vector<long long> countKConstraintSubstrings(string s, int k, vector<vector<int>>& queries) {
        int n = s.size();
        
        memset(left, 0, sizeof left), memset(sum, 0, sizeof sum);
        unordered_map<char, int> h;
        for (int i = 1, j = 1; j <= n; ++ j ) {
            h[s[j - 1]] ++ ;
            while (i < j && (h['0'] > k && h['1'] > k))
                h[s[i - 1]] -- , i ++ ;
            left[j] = i;
            sum[j] = sum[j - 1] + j - i + 1;
        }
        
        vector<LL> res;
        for (auto & q : queries) {
            int l = q[0] + 1, r = q[1] + 1;
            // ATTENTION 思考 left 的单调性质: 找到第一个 left[x] >= l 的 x
            int t = lower_bound(left + l, left + r + 1, l) - left;
            // 右侧的部分 + 左侧必然合法的部分
            res.push_back((sum[r] - sum[t/*ATTENTION*/ - 1]) + (LL)(t - l + 1) * (t - l) / 2);
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

### 边遍历边维护

> [!NOTE] **[LeetCode 2763. 所有子数组中不平衡数字之和](https://leetcode.cn/problems/sum-of-imbalance-numbers-of-all-subarrays/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然是经典的暴力优化 边遍历边维护计数值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 1e3 显然最多 n^2logn
    //  枚举左右端点 枚举右端点过程中【维护计数值】

    int sumImbalanceNumbers(vector<int>& nums) {
        int n = nums.size(), res = 0;
        for (int i = 0; i < n; ++ i ) {
            map<int, int> h;
            int t = 0;
            for (int j = i; j < n; ++ j ) {
                int x = nums[j];
                // 已存在的情况 计数值没有任何变化
                if (h[x]) {
                    res += t;
                    continue;
                }

                // 不存在的情况 会新增一个数值
                //  此时
                //  - [-1, +1] 减少一个
                //  - [-1,  _] 不变
                //  - [ _, +1] 不变
                //  - [ _,  _] 根据左右侧有没有判断是否会增加   // ATTENTION 如果都有要-1

                h[x] ++ ;   // 方便后续查找迭代器

                if (h.count(x - 1) && h.count(x + 1))       // ATTENTION h[x-1] 会直接创建 default 值
                    t -- ;
                else if (!h.count(x - 1) && !h.count(x + 1)) {
                    int l = 0, r = 0;
                    if (h.lower_bound(x) != h.begin())      // x 左侧存在其他值
                        l = 1;
                    if (h.upper_bound(x) != h.end())        // x 右侧存在其他值
                        r = 1;
                    
                    t += l + r - (l && r ? 1 : 0);          // 如果左右都存在，则还要-1
                } // else do nothing

                h[x] ++ ;
                res += t;
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

> [!NOTE] **[LeetCode 3026. 最大好子数组和](https://leetcode.cn/problems/maximum-good-subarray-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典暴力优化，实际上也不需要 deque，直接维护 min 值就好

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;
    
    LL sum[N];
    unordered_map<LL, deque<int>> h;
    
    long long maximumSubarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        LL res = -1e16;
        sum[0] = 0; // ATTENTION 不能初始化h[0].push_back(0)
        for (int i = 1; i <= n; ++ i ) {
            int x = nums[i - 1];
            
            sum[i] = sum[i - 1] + x;
            
            if (h.count(x - k) && !h[x - k].empty()) {
                int y = h[x - k].front();
                res = max(res, sum[i] - (y ? sum[y - 1] : 0));
            }
            if (h.count(x + k) && !h[x + k].empty()) {
                int y = h[x + k].front();
                res = max(res, sum[i] - (y ? sum[y - 1] : 0));
            }
            
            // ATTENTION: TLE 优化
            // 对于相同的 x 要保留sum最小的那个的下标, 这样才能让区间和最大 => 单调队列 => 需要取队底部故deque
            {
                auto & stk = h[x];
                while (!stk.empty() && sum[stk.front() - 1] >= sum[i - 1])
                    stk.pop_back();
                stk.push_back(i);
            }
        }
        return res == -1e16 ? 0 : res;
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

### 公式转化

> [!NOTE] **[LeetCode 891. 子序列宽度之和](https://leetcode.cn/problems/sum-of-subsequence-widths/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 与 [LeetCode 1498. 满足条件的子序列数目](https://leetcode.cn/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/) 类似
> 
> 显然有 $O(n^2)$ 的统计做法，但是数据范围较大，考虑优化
> 
> 一开始想的是通过等式变形去使用 s 值优化，然而并不好转化 ==> 失败
> 
> 实际上，只需要考虑【**当前值作为最小或最大的贡献**】即可
> 
> 非常经典

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;   // 中间变量较大 需要 LL
    const static int N = 1e5 + 10, MOD = 1e9 + 7;

    int p[N];
    void init() {
        p[0] = 1;
        for (int i = 1; i < N; ++ i )
            p[i] = p[i - 1] * 2 % MOD;
    }

    int sumSubseqWidths(vector<int>& nums) {
        init();
        sort(nums.begin(), nums.end());

        int n = nums.size(), res = 0;
        for (int i = 1, s = 0; i <= n; ++ i ) {
            // 使用 s 维护进而优化下列 for-loop ==> 失败
            // int t = 0;
            // for (int j = 1; j < i; ++ j )   // 包含两个或多个元素
            //     t = (t + p[i - j - 1] * (nums[i - 1] - nums[j - 1]) % MOD) % MOD;
            
            // ATTENTION 考虑贡献，而非直接优化上面的计算式
            int t = ((LL)nums[i - 1] * p[i - 1] % MOD - (LL)nums[i - 1] * p[n - i] % MOD + MOD) % MOD;
            res = (res + t) % MOD;
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

> [!NOTE] **[LeetCode 2488. 统计中位数为 K 的子数组](https://leetcode.cn/problems/count-subarrays-with-median-k/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常经典的暴力优化
>
> 直接 枚举长度 & 起始端点 显然会 TLE
> 
> 考虑原题要求可以等价转化为：
> 
> - **某固定位置的左侧的 `bigger - smaller` 的数量与右侧 `smaller - bigger` 的数量【相等或仅大一个】**
> 
> 则 **枚举左侧起始位置并计算得 `bigger - smaller`，并累加右侧相应值的计数数量即可**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // ATTENTION: nums 中的整数 互不相同
    // 子数组是数组中的一个连续部分
    const static int N = 1e5 + 10;
    
    unordered_map<int, int> hash, rh;
    
    int l[N], r[N];
    
    int countSubarrays(vector<int>& nums, int k) {
        int n = nums.size();
        {
            for (int i = 0; i < n; ++ i )
                hash[nums[i]] = i;
        }
        {
            if (!hash.count(k))
                return 0;
        }
        int p = hash[k];
        {
            for (int i = p - 1, t = 0; i >= 0; -- i ) {
                l[i] = l[i + 1] + (nums[i] < k);
            }
            for (int i = p + 1; i < n; ++ i ) {
                r[i] = r[i - 1] + (nums[i] < k);
            }
            // ATTENTION 计算数量差值的计数 (smaller - bigger)
            for (int i = p; i < n; ++ i ) {
                int x = r[i], y = i - p - r[i];
                rh[r[i] - y] ++ ;
            }
        }
        
        
        int res = 0;
        // 枚举左侧起始点 累加右侧差值的计数
        for (int i = 0; i <= p; ++ i ) {
            int x = (p - i) - l[i], y = x - l[i];   // 大于k的数量为x, (bigger - smaller)的数量为y
            
            int last = res;
            res += rh[y] + rh[y - 1];
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

> [!NOTE] **[LeetCode 2551. 将珠子放入背包中](https://leetcode.cn/problems/put-marbles-in-bags/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然问题可以转化为在连续区间内选择 $k-1$ 个分割点，从而将裸的 `二维 DP` 简化为 `一维问题`
> 
> 进而 直接选择 `最大 & 最小` 的分割点即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;
    // 把原数组拆成 k 段(非空)，每段有其代价(两段点和)
    // 求最大、最小代价差值
    // ==> 转换 对于 [1, n] 
    // 在 [1, n-1] 选择 k-1 个分隔点(左闭)，每个点的收益是 当前分隔点的值+右侧点的值
    // 则放堆里挑个最大的就行 O(n)
    
    int x[N];
    
    long long putMarbles(vector<int>& weights, int k) {
        int n = weights.size();
        if (n == k)
            return 0;
        
        memset(x, 0, sizeof x);
        // 分隔点位置
        for (int i = 1; i < n; ++ i )
            x[i] += weights[i - 1] + weights[i];

        LL maxv = 0, minv = 0;  // or weights[0] + weights[n - 1]
        {
            // 大顶堆
            priority_queue<LL> p;
            for (int i = 1; i < n; ++ i )
                p.push(x[i]);
            for (int i = 0; i < k - 1; ++ i ) {
                maxv += p.top(); p.pop();
            }
        }
        {
            priority_queue<LL, vector<LL>, greater<LL>> p;
            for (int i = 1; i < n; ++ i )
                p.push(x[i]);
            for (int i = 0; i < k - 1; ++ i ) {
                minv += p.top(); p.pop();
            }
        }
        return maxv - minv;
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

> [!NOTE] **[LeetCode 2681. 英雄的力量](https://leetcode.cn/problems/power-of-heroes/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常经典的暴力优化 $O(n^2)=>O(n)$
> 
> 思考细节推导

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10, MOD = 1e9 + 7;

    LL f[N];

    int sumOfPower(vector<int>& nums) {
        f[0] = 1;
        for (int i = 1; i < N; ++ i )
            f[i] = f[i - 1] * 2 % MOD;
        sort(nums.begin(), nums.end());

        LL res = 0, s = 0;
        int n = nums.size();
        for (int i = 0; i < n; ++ i ) {
            LL x = nums[i]; // 作为最大值

            // 把 j=i 的情况单独拎出来 方便后面的循环化简
            res = (res + x * x % MOD * x % MOD) % MOD;

            x = x * x % MOD;

            // LL s = 0;
            // for (int j = 0; j < i; ++ j ) {
            //     LL t = f[i - j - 1], y = nums[j];
            //     s = (s + y * t % MOD) % MOD;
            // }
            if (i)
                s = (s * 2 % MOD + nums[i - 1]) % MOD;
            res = (res + x * s % MOD) % MOD;
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

> [!NOTE] **[LeetCode 3077. K 个不相交子数组的最大能量值 ](https://leetcode.cn/problems/maximum-strength-of-k-disjoint-subarrays/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 DP 优化
> 
> 公式转化 + 边遍历边维护
> 
> 注意细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // [不相交子数组] => 显然是 dp 【线性 划分型DP】
    // ATTENTION: 题意是将原始数组【切分】... 选出来的所有子数组【不】需要覆盖整个数组
    
    using LL = long long;
    const static int N = 1e4 + 10;
    const static LL INF = 1e16;

    int n;
    LL s[N], f[N];
    
    long long maximumStrength(vector<int>& nums, int k) {
        this->n = nums.size();
        {
            s[0] = 0;
            for (int i = 1; i <= n; ++ i )
                s[i] = s[i - 1] + nums[i - 1];
        }
        
        memset(f, 0, sizeof f);
        for (int _k = 1; _k <= k; ++ _k ) {
            static LL pre[N];
            memcpy(pre, f, sizeof f);
            
            // 分情况讨论
            // 1. i 不作为最右侧元素 则为前 i-1 个里选择 _k 个
            //          f[_k][i] = f[_k][i - 1]
            // 2. i 作为最右侧元素 则需要枚举左侧下标 L
            //          f[_k][i] = f[_k - 1][L] + (s[i] - s[L]) * w       其中 [w = (-1)^_k+1 * (k - _k + 1)]
            //  【变形】 f[_k][i] = (f[_k - 1][L] - s[L] * w) + s[i] * w;
            //  其中  f[_k - 1][L] - s[L] * w 可以伴随遍历整体维护
            // 3. 干掉第一维
            
            f[_k - 1] = -INF;   // ATTENTION 必须 否则case3 fail, 写在 for-loop 没用不会执行到

            LL max_v = -INF;
            int w = ((_k & 1) ? 1 : -1) * (k - _k + 1);
            
            for (int i = _k; i <= n - (k - _k)/*ATTENTION 后面需要留k-_k 个数*/; ++ i ) {
                max_v = max(max_v, pre[i - 1] - s[i - 1] * w); // refresh
                f[i] = max(f[i - 1], s[i] * w + max_v);
            }
        }
        
        return f[n];
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

### 二维偏序 (BIT)


> [!NOTE] **[LeetCode 2926. 平衡子序列的最大和](https://leetcode.cn/problems/maximum-balanced-subsequence-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 暴力优化 + BIT 维护区间最值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 原题转换: nums[i_j] - i_j >= nums[i_j-1] - i_j-1
    //          nums[y]-y >= nums[x]-x => 这样 y 才能接在 x 后面
    //    则 可以直接计算偏移量
    
    using LL = long long;
    using PII = pair<int, int>;
    
    const static int N = 1e5 + 10;
    
    // -------------------------- begin --------------------------
    vector<int> t;
    int find(int x) {
        return lower_bound(t.begin(), t.end(), x) - t.begin();
    }
    
    LL tr[N], w[N];
    int lowbit(int x) {
        return x & -x;
    }
    void modify(int x, LL y) {
        w[x] = max(w[x], y);
        for (int i = x; i < N; i += lowbit(i)) {
            tr[i] = max(tr[i], y);
        }
    }
    LL query(int l, int r) {
        LL ret = 0;
        for (; l <= r; ) {
            ret = max(ret, w[r]);
            for ( -- r ; r >= l + lowbit(r); r -= lowbit(r))
                ret = max(ret, tr[r]);
        }
        return ret;
    }
    
    // --------------------------- end ---------------------------
    
    LL f[N];
    
    long long maxBalancedSubsequenceSum(vector<int>& nums) {
        int n = nums.size();
        
        {
            // 离散化
            this->t.clear();
            for (int i = 0; i < n; ++ i )
                t.push_back(nums[i] - i);
            
            sort(t.begin(), t.end());
            t.erase(unique(t.begin(), t.end()), t.end());
        }
        
        vector<PII> xs;     // 计算偏移量
        for (int i = 0; i < n; ++ i )
            // xs.push_back({nums[i] - i, nums[i]});
            xs.push_back({find(nums[i] - i) + 1, nums[i]}); // +1 方便 BIT 维护
        
        LL res = -1e15;
        // 状态转移
        /*
        for (int i = 0; i < n; ++ i ) {
            f[i] = 0;
            for (int j = 0; j < i; ++ j ) {
                if (xs[j].first <= xs[i].first) {
                    f[i] = max(f[i], f[j]);
                }
            }
            f[i] += nums[i];
            res = max(res, f[i]);
        }
        */
        // 前面本质上是从已有的一堆里面，找到一个符合条件的最大的
        // 显然可以 bit，但是需要将 xs[i].first = nums[i]-i 进行离散化操作   #L34
        memset(tr, 0xcf, sizeof tr), memset(w, 0xcf, sizeof w);    // ATTENTION: -INF
        for (int i = 0; i < n; ++ i ) {
            auto [x, y] = xs[i];
            LL v = query(1, x);
            res = max(res, v + y);
            modify(x, v + y);
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

### 在线 BFS

> [!NOTE] **[LeetCode 2612. 最少翻转操作数](https://leetcode.cn/problems/minimum-reverse-operations/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 较显然的是 BFS 思路；关键的就是左右边界的公式推导
> 
> - 显然有会 TLE 的暴力做法
> 
> - 【在线 BFS】在此基础上，注意到需要关注的位置都是没有走过的位置，且这些位置位于一个连续区间内，显然可以 set 维护 “未走过的位置” 的列表避免重复遍历从而降低复杂度
> 
> - 更进一步的：涉及到一段连续区间内找到 “下一个未走过的位置” 可以直接并查集维护

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 暴力 TLE**

```cpp
class Solution {
public:
    // 长度为 n 只有 p 处为 1
    //  每次只能翻转长度为 k 的连续子序列 => 从原坐标 u -> v 的 v 有限制 (在某个范围内【不能超过 xx】且不能被 banned)
    // => 这题最关键的就是左右边界的公式推导
    
    const static int N = 1e5 + 10, INF = 0x3f3f3f3f;
    
    int d[N];
    bool b[N];
    
    vector<int> minReverseOperations(int n, int p, vector<int>& banned, int k) {
        memset(d, 0x3f, sizeof d);
        memset(b, 0, sizeof b);
        for (auto x : banned)
            b[x] = true;
        
        queue<int> q;
        q.push(p); d[p] = 0;
        while (q.size()) {      // 约束它一直往右走 -> wrong, 会有 p 在最左侧的情况
            int u = q.front(); q.pop();
            
            if (k & 1) {
                // 枚举中心位置
                //  =< x
                // for (int i = max(u + 1, k / 2); i + k / 2 < n && (i - k / 2 <= u); ++ i ) {
                for (int i = max(u - k / 2, k / 2); i + k / 2 < n && (i - k / 2 <= u); ++ i ) {
                    int v = i + (i - u);
                    if (b[v])
                        continue;
                    if (d[v] > d[u] + 1) {
                        d[v] = d[u] + 1;
                        q.push(v);
                    }
                }
            } else {
                // 枚举中心靠左位置
                for (int i = max(u - k / 2, k / 2 - 1); i + k / 2 < n && (i - k / 2 + 1 <= u); ++ i ) {
                    int v = i + (i - u + 1);
                    if (b[v])
                        continue;
                    if (d[v] > d[u] + 1) {
                        d[v] = d[u] + 1;
                        q.push(v);
                    }
                }
            }
        }
        
        vector<int> res(n, -1);
        for (int i = 0; i < n; ++ i )
            if (d[i] < INF / 2)
                res[i] = d[i];
        return res;
    }
};
```

##### **C++ set 维护**

```cpp
class Solution {
public:
    // 长度为 n 只有 p 处为 1
    //  每次只能翻转长度为 k 的连续子序列 => 从原坐标 u -> v 的 v 有限制 (在某个范围内【不能超过 xx】且不能被 banned)
    // => 这题最关键的就是左右边界的公式推导
    
    // => 伴随着区间的滑动 翻转后所有的位置组成了一个公差为 2 的等差数列
    // 考虑:
    //  1. 区间最多影响到的元素为 [i - k + 1, i + k - 1]
    //  2. 考虑左边界 0: L=0,R=k-1       对应的翻转位置是 0+(k-1)-i=k-i-1        小于这个的位置都没法到
    //  3. 考虑右边界 n-1: L=n-k,R=n-1   对应的翻转位置是 (n-k)+(n-1)-i=2n-k-i-1 大于这个的位置都没法到
    // => [max(i-k+1,k-i-1), min(i+k-1, 2n-k-i-1)]
    
    const static int N = 1e5 + 10, INF = 0x3f3f3f3f;
    
    int d[N];
    
    vector<int> minReverseOperations(int n, int p, vector<int>& banned, int k) {
        set<int> S[2];
        {
            for (int i = 0; i < n; ++ i )
                S[i % 2].insert(i);
            for (auto x : banned)
                S[x % 2].erase(x);
        }
        
        memset(d, 0x3f, sizeof d);
        queue<int> q;
        {
            q.push(p);
            d[p] = 0; S[p % 2].erase(p);
        }
        while (!q.empty()) {
            int i = q.front(); q.pop();
            int L = i < k ? (k - 1) - i : i - (k - 1);
            int R = i + k - 1 < n ? i + (k - 1) : n + n - k - 1 - i;

            auto & s = S[L % 2];

            //  ATTENTION for-loop 写法
            for (auto it = s.lower_bound(L); it != s.end() && *it <= R; it = s.erase(it)) {
                d[*it] = d[i] + 1;
                q.push(*it);
            }
        }
        
        vector<int> res(n, -1);
        for (int i = 0; i < n; ++ i )
            if (d[i] < INF / 2)
                res[i] = d[i];
        return res;
    }
};
```

##### **C++ DSU**

```cpp
class Solution {
public:
    // 长度为 n 只有 p 处为 1
    //  每次只能翻转长度为 k 的连续子序列 => 从原坐标 u -> v 的 v 有限制 (在某个范围内【不能超过 xx】且不能被 banned)
    // => 这题最关键的就是左右边界的公式推导
    
    // => 伴随着区间的滑动 翻转后所有的位置组成了一个公差为 2 的等差数列
    // 考虑:
    //  1. 区间最多影响到的元素为 [i - k + 1, i + k - 1]
    //  2. 考虑左边界 0: L=0,R=k-1       对应的翻转位置是 0+(k-1)-i=k-i-1        小于这个的位置都没法到
    //  3. 考虑右边界 n-1: L=n-k,R=n-1   对应的翻转位置是 (n-k)+(n-1)-i=2n-k-i-1 大于这个的位置都没法到
    // => [max(i-k+1,k-i-1), min(i+k-1, 2n-k-i-1)]
    //
    // => 进阶: 直接使用并查集跳过区间
    
    const static int N = 1e5 + 10, INF = 0x3f3f3f3f;
    
    int pa[N];
    void init() {
        for (int i = 0; i < N; ++ i )
            pa[i] = i;
    }
    int find(int x) {
        if (pa[x] != x)
            pa[x] = find(pa[x]);
        return pa[x];
    }
    
    int d[N];
    
    vector<int> minReverseOperations(int n, int p, vector<int>& banned, int k) {
        init();
        for (auto x : banned)   // ATTENTION 同奇偶 所以是2 => 跳过被 ban 的节点
            pa[x] = x + 2;
        pa[p] = p + 2;          // ATTENTION 同奇偶 所以是2
        
        memset(d, 0x3f, sizeof d);
        queue<int> q;
        {
            q.push(p);
            d[p] = 0;
        }
        while (!q.empty()) {
            int i = q.front(); q.pop();
            int L = i < k ? (k - 1) - i : i - (k - 1);
            int R = i + k - 1 < n ? i + (k - 1) : n + n - k - 1 - i;

            for (int t = find(L); t <= R; t = find(t)) {    // ATTENTION 细节
                d[t] = d[i] + 1;
                pa[t] = t + 2;     // ATTENTION 同奇偶 所以是2
                q.push(t);
            }
        }
        
        vector<int> res(n, -1);
        for (int i = 0; i < n; ++ i )
            if (d[i] < INF / 2)
                res[i] = d[i];
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

> [!NOTE] **[LeetCode 2617. 网格图中最少访问的格子数](https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典在线 BFS

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10, INF = 0x3f3f3f3f;
    
    int n, m;
    vector<vector<int>> g;
    
    vector<set<int>> r, c;  // 行 列
    
    int d[N], q[N * 2];
    
    int minimumVisitedCells(vector<vector<int>>& grid) {
        this->g = grid;
        this->n = g.size(), this->m = g[0].size();
        r.resize(n), c.resize(m);
        
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j ) {
                int t = i * m + j;
                r[i].insert(t), c[j].insert(t);
            }
        
        memset(d, 0x3f, sizeof d);
        d[0] = 0;
        r[0].erase(0), c[0].erase(0);
        
        int hh = 0, tt = -1;
        q[ ++ tt] = 0;
        
        while (hh <= tt) {
            int u = q[hh ++ ];
            int x = u / m, y = u % m, v = g[x][y];
            {
                auto & s = r[x];
                for (auto it = s.lower_bound(x * m + y + 1); it != s.end() && *it <= x * m + y + v; it = s.erase(it)) {
                    int t = *it;
                    d[t] = d[u] + 1;
                    q[ ++ tt] = t;
                    
                    c[t % m].erase(t);
                }
            }
            {
                auto & s = c[y];
                for (auto it = s.lower_bound(x * m + y + 1); it != s.end() && *it <= (x + v) * m + y; it = s.erase(it)) {
                    int t = *it;
                    d[t] = d[u] + 1;
                    q[ ++ tt] = t;
                    
                    r[t / m].erase(t);
                }
            }
        }
        
        // for (int i = 0; i < n; ++ i ) {
        //     for (int j = 0; j < m; ++ j )
        //         cout << d[i * m + j] << ' ';
        //     cout << endl;
        // }
        
        return d[n * m - 1] < INF / 2 ? d[n * m - 1] + 1: -1;
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

### BIT TRICK

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
