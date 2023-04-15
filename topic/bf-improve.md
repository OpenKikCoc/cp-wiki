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

##### **C++ 直接统计**

```cpp
class Solution {
public:
    int subarrayBitwiseORs(vector<int>& arr) {
        unordered_set<int> S;
        int n = arr.size();
        for (int i = 0; i < n; ++ i ) {
            S.insert(arr[i]);
            for (int j = i - 1; j >= 0; -- j ) {
                // ATTENTION
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

> [!NOTE] **[LeetCode 1521. 找到最接近目标值的函数值](https://leetcode-cn.com/problems/find-a-value-of-a-mysterious-function-closest-to-target/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可以部分参考 [898. 子数组按位或操作](https://leetcode-cn.com/problems/bitwise-ors-of-subarrays/)
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