

## 习题

> [!NOTE] **[AcWing 901. 滑雪](https://www.acwing.com/problem/content/903/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 这个题常见的解法应该是 dfs 和bfs，dp的思想是对以及访问过的点算出来最大路径，从而每次不用重复计算，这个优化dfs本身的方法其实是一样的
>
> 1. 状态表示：f[i,j] 所有从(i,j)开始滑的路径。
>
> 2. 状态计算：按照第一步往哪个方向滑的，可以把所有情况分成四类：向上，下，左，右。
>
> 比如向右滑的最大值：(i,j)-->(i,j+1) ，那就是求(i,j+1)开始滑的最大值，那就是f[i,j+1],有一个前提条件：那就是h[i,j]>h[i,j+1]

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 305;
int n, m;
int h[maxn][maxn];
int f[maxn][maxn];

int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
int dp(int x, int y) {
    if (f[x][y] != -1) return f[x][y];
    f[x][y] = 1;
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx < 1 || nx > n || ny < 1 || ny > m || h[nx][ny] >= h[x][y])
            continue;
        f[x][y] = max(f[x][y], dp(nx, ny) + 1);
    }
    return f[x][y];
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j) cin >> h[i][j];
    int res = 0;
    memset(f, -1, sizeof f);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j) res = max(res, dp(i, j));
    cout << res << endl;
}
```

##### **Python**

```python
def dp(x, y):
    if f[x][y] != -1:  # 踩坑：判断是否已经计算过了，计算过了 直接返回，这就是记忆化搜索
        return f[x][y] 
    f[x][y] = 1  # 如果这个点没有被计算过，那初始值是为1的。（自己也算一步）
    dx, dy = [0, 0, -1, 1], [1, -1, -0, 0]
    for i in range(4):
        a, b = x + dx[i], y + dy[i]
        if 1 <= a <= n and 1 <= b <= m and g[x][y] > g[a][b]:
            f[x][y] = max(f[x][y], dp(a, b) + 1)
    return f[x][y]


if __name__ == '__main__':
    N = 310
    g = [[0] * N for _ in range(N)]
    f = [[-1] * N for _ in range(N)]
    dx = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    n, m = map(int, input().split())
    for i in range(1, n + 1):
        nums = map(int, input().split())
        for j, val in enumerate(nums):
            g[i][j + 1] = val
    # 这样处理也可以哦         
    # g1=[[0]  *N]
    # for i in range(n):
    #   g1.append([0] + list(map(int, input().split())))

    # 更简约的处理输入：==> 就让下标从0开始，下面遍历也需要改成i,j从0到n/m，dp函数判断也是[0,n/m]
    # g = []
    # for i in range(n):
    #    g.append(list(map(int, input().split())))
    res = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            res = max(res, dp(i, j))
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 这是动态规划里非常经典的一道题目，几乎是所有学编程的同学都会遇到的一道题目。
>
> 1. 状态表示：f[i, j] 表示走到(i, j)这个格子时的长度。属性：max
> 2. 状态转移：枚举四个方向，如果某个格子比当前格子低，那么可以用该格子更新当前格子的最大长度
>
> 这道题依赖关系比较复杂，所以可以用记忆化搜索来做：如果某个状态还没计算过，则递归计算该状态的值。
>
> 3. 时间复杂度：一共有$O(n^2)$ 状态，状态转移是常量，所以总的时间复杂度是$O(n^2)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, m;
    vector<vector<int>> f, w;
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

    int dp(int x, int y) {
        auto& v = f[x][y];
        if (v != -1) return v;
        v = 1;
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && w[x][y] < w[a][b])
                v = max(v, dp(a, b) + 1);
        }
        return v;
    }

    int longestIncreasingPath(vector<vector<int>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;
        w = matrix;
        n = w.size(), m = w[0].size();
        f = vector<vector<int>>(n, vector<int>(m, -1));

        int res = 0;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                res = max(res, dp(i, j));
        return res;
    }
};
```

##### ****Python-记忆化搜索1****

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        n, m  = len(matrix), len(matrix[0])

        # -1：未计算 0：不合法 1：合法；
        f = [[-1] * m for i in range(n)]
        res = 0

        def dp(x, y):
            # 计算过则直接返回（记忆化搜索）
            if f[x][y] != -1:return f[x][y]
            f[x][y] = 1
            dx, dy = [-1, 0, 1, 0], [0, -1, 0, 1]

            for i in range(4):
                a, b  = x + dx[i], y + dy[i]
                if 0 <= a < n and 0 <= b < m and matrix[a][b] > matrix[x][y]:
                    f[x][y] = max(f[x][y], dp(a, b) + 1)
            return f[x][y]

        for i in range(n):
            for j in range(m):
                res = max(res, dp(i, j))

        return res
```



**Python-记忆化搜索2**

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:return 0
        n, m = len(matrix), len(matrix[0])
        dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
        
        @lru_cache(None)
        def dfs(x: int, y: int) -> int:
            best = 1
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and matrix[nx][ny] > matrix[x][y]:
                    best = max(best, dfs(nx, ny) + 1)
            return best

        res = 0
        for i in range(n):
            for j in range(m):
                res = max(res, dfs(i, j))
        return res
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 403. 青蛙过河](https://leetcode-cn.com/problems/frog-jump/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> **记忆化搜索**
>
> 1. 状态表示： f[i, j] 表示跳到了第i个点，且从第i个点往后跳的长度可以为j的所有方案的集合；属性：是否为空
>
>    答案应该是：从f(n, 0), f(n, 1)...f(n, n) 这些集合是否存在一个非空，存在的话返回True
>
> 2. 状态转移：一般根据最后一个不同点作为划分依据。我们的定义是从i到下一个点的长度是j，那么从上一个点跳到i这个点的长度是[j-1, j, j+1]。所以可以用这三个作为划分（到第i个点的方案）
>
>    1）如果上一次跳长度是j-1，那我们就去找有没有哪个点的坐标是P[i]-(j-1), 如果没有的话，那就上一跳的长度不可能是j-1；如果可以找到的话，假设坐标是P[k]，那递推下去如果想看这类方案是否存在，其实就是f[k, j-1]
>
>    2）其他两种情况同理，只要这三种情况有一个不为空，就代表存在。
>
> 3. 这道题还需要快速判断是否存在一个点的坐标是P[i]-(j-1), 用哈希表进行映射。
>
> 4. 时间复杂度：状态是$O(N^2)$，每个状态有三种情况就是O(1), 整个时间复杂度是$O(N^2)$
>
>    记忆化搜索：可减少一些对答案无关的状态计算！优化一点常数！目前直接使用DP两层循环是超时的！

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    unordered_map<int, int> hash;
    vector<int> stones;
    map<pair<int, int>, bool> f;

    bool dp(int i, int j) {
        if (f.count({i, j})) return f[{i, j}];
        f[{i, j}] = false;
        for (int k = max(1, j - 1); k <= j + 1; ++ k )
            if (hash.count(stones[i] - k)) {
                int p = hash[stones[i] - k];
                if (dp(p, k)) {
                    f[{i, j}] = true;
                    break;
                }
            }
        return f[{i, j}];
    }

    bool canCross(vector<int>& _stones) {
        stones = _stones;
        int n = stones.size();
        for (int i = 0; i < n; ++ i ) hash[stones[i]] = i;
        // 从0向【后】跳1初始化合法
        f[{0, 1}] = true;
        for (int i = 0; i < n; ++ i )
            // 遍历可能结果集
            if (dp(n - 1, i)) return true;
        return false;
    }
};
```

##### **C++**

```cpp
// yxc
const int N = 2010;

int f[N][N];

class Solution {
public:
    unordered_map<int, int> hash;
    vector<int> stones;

    int dp(int x, int y) {
        if (f[x][y] != -1) return f[x][y];
        f[x][y] = 0;
        for (int k = max(1, y - 1); k <= y + 1; k ++ ) {
            int z = stones[x] - k;
            if (hash.count(z)) {
                int p = hash[z];
                if (dp(p, k)) {
                    f[x][y] = 1;
                    break;
                }
            }
        }
        return f[x][y];
    }

    bool canCross(vector<int>& _stones) {
        stones = _stones;
        int n = stones.size();
        for (int i = 0; i < n; i ++ ) hash[stones[i]] = i;
        memset(f, -1, sizeof f);
        f[0][1] = 1;
        for (int i = 0; i < n; i ++ )
            if (dp(n - 1, i))
                return true;
        return false;
    }
};
```

##### **Python 记忆化搜索**

```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        n = len(stones)
        def dp(i, j):
          	# 计算过则直接返回（记忆化搜索）
            if f[i][j] != -1:return f[i][j]
            f[i][j] = 0
            
            # 枚举上一次跳的三种情况：j - 1, j, j + 1
            for k in range(max(1, j - 1),  j + 2):
                if my_dict[stones[i] - k] != -1:  # 确保该位置有石头
                    p = my_dict[stones[i] - k]
                    if dp(p, k): 
                        f[i][j] = 1
                        break
            return f[i][j]
        
        #  -1：未计算 0：不合法 1：合法；
        f = [[-1] * (n + 1) for _ in range(n + 1)]
        my_dict = collections.defaultdict(lambda: -1)
        for i in range(n):
            my_dict[stones[i]] = i
        
        f[0][1] = 1 # 初始化，0号位第一块石头跳一步状态合法（题意）
        
        # 枚举能否最后一块石头能否往后跳i步！有一个合法即可以跳到最后一块石头！
        for i in range(n):
            if dp(n - 1, i):
                return True
        return False
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 464. 我能赢吗](https://leetcode-cn.com/problems/can-i-win/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 优化后的状压搜索实现
>
> 1. 状态表示：由于所有整数最大不会超过20，所以可以开一个n位的二进制的状态，每一位非0即1，1表示当前位的数被用过了，用一个二进制数表示当前状态的情况。f[state]表示两个人报数的状态是state, 我要从这个state状态开始去报数，这种情况下，我是必胜还是必败的。
>
> 2. 状态转移：（博弈论里的经典转移）怎么去判断当前状态是必赢还是必输呢？
>
>    枚举一下所有没有报的数，但凡存在一种情况使得总和>=m，那就是必胜（这是边界）。
>
>    如果不是边界的情况呢，但凡我能走到一个对手必败的情况，那就是必胜。
>
>    如果我转移的所有情况，都是对手必胜，那我就是必败；如果我转移的情况有一个是对手必败，那我必胜。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 优化**

```cpp
class Solution {
public:
// 优化 according
    // f[x] 表示某个状态是必胜还是必败
    vector<int> f;
    int n, m;
    int dp(int x) {
        if (f[x] != -1) return f[x];
        int sum = 0;
        for (int i = 0; i < n; ++ i )
            if (x >> i & 1)
                sum += i + 1;
        for (int i = 0; i < n; ++ i ) {
            if (x >> i & 1) continue;
            if (sum + i + 1 >= m) return f[x] = 1;  // 可以拿 i 必胜
            if (!dp(x + (1 << i))) return f[x] = 1; // 可以导致对方必败 必胜
        }
        return f[x] = 0;
    }
    bool canIWin(int _n, int _m) {
        n = _n, m = _m;
        if (n * (n + 1) / 2 < m) return false;
        f.resize(1 << n, -1);
        return dp(0);
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    // 不能使用重复整数 状态压缩20个位
    // dp[i][j] 表示剩下整数状态为i时 累积和为j的先手结果
    // dp[i][j] = 穷举接下来选择每一个数的下一个状态是否可能为true
    // 如果有任意一个值为true 则当前状态必败 否则必胜
    int mxInt, tot;
    //unordered_map<pair<int,int>, int> m;  // hash不可用
    map<pair<int, int>, int> m;
    bool dfs(int state, int sum) {
        if (sum >= tot) return true;
        if (m.find({state, sum}) != m.end()) return m[{state, sum}];
        bool f = true;
        for (int i = 1; i <= mxInt; ++ i )
            if ((state & (1 << i)) == 0)
                if (dfs(state | (1 << i), sum + i)) {
                    f = false;
                    break;
                }
        return m[{state,sum}] = f;
    }
    bool canIWin(int maxChoosableInteger, int desiredTotal) {
        mxInt = maxChoosableInteger, tot = desiredTotal;
        // 所有数相加仍小于tot false
        if ((mxInt + 1) * mxInt / 2 < tot) return false;
        for (int i = 1; i <= mxInt; ++ i )
            // 选择 i 当前sum为i
            if (dfs(1 << i, i))
                return true;
        return false;
    }
};
```

##### **Python 记忆化搜索1**

```python
class Solution:
    def canIWin(self, n: int, m: int) -> bool:
        if n * (n + 1) // 2 < m:
            return False
        f = [-1] * (1 << n)

        def dp(x):
            # 记忆化：已经被算过了，直接返回
            if f[x] != -1:
                return f[x]
            sumn = 0
            for i in range(n):
                if x >> i & 1:
                    sumn += i + 1
            
            for i in range(n):
              	# 当前位是1，说明已经被人用了
                if x >> i & 1:continue
                # 当前状态的总和 + 选择了当前位数字 > m
                if sumn + i + 1 >= m:
                    f[x] = 1
                    return f[x]
                # 遍历所有对手可能从哪个状态开始选数字
                if not dp(x + (1 << i)):
                    f[x] = 1
                    return f[x]
            f[x] = 0
            return f[x]

        return bool(dp(0))
```



**Python 记忆化搜索2**

```python
class Solution:
    def canIWin(self, n, m):
        if n * (n + 1) / 2 < m:
            return False

        @lru_cache(None)
        def dfs(x, s):
            for i in range(n):
                if (x >> i) & 1:
                    continue
                if s + i + 1 >= m:
                    return True 
                if not dfs(x + (1 << i), s + i + 1):
                    return True
            return False 
            
        return dfs(0, 0)
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1240. 铺瓷砖](https://leetcode-cn.com/problems/tiling-a-rectangle-with-the-fewest-squares/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有 横切、纵切、横纵切中间留方形 的 dp 做法 正确性持疑 ==> TODO
> 
> 有打表

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
vector<vector<int>> v = {{1},
                         {2, 1},
                         {3, 3, 1},
                         {4, 2, 4, 1},
                         {5, 4, 4, 5, 1},
                         {6, 3, 2, 3, 5, 1},
                         {7, 5, 5, 5, 5, 5, 1},
                         {8, 4, 5, 2, 5, 4, 7, 1},
                         {9, 6, 3, 6, 6, 3, 6, 7, 1},
                         {10, 5, 6, 4, 2, 4, 6, 5, 6, 1},
                         {11, 7, 6, 6, 6, 6, 6, 6, 7, 6, 1},
                         {12, 6, 4, 3, 6, 2, 6, 3, 4, 5, 7, 1},
                         {13, 8, 7, 7, 6, 6, 6, 6, 7, 7, 6, 7, 1}};

int tilingRectangle(int n, int m) {
    if (n < m) swap(n, m);
    return v[n - 1][m - 1];
}
```

##### **Python**

```python

```





<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1553. 吃掉 N 个橘子的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-eat-n-oranges/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 记忆化dfs or 记忆化bfs
>
> 原先写的bfs加个 map 就过了
>
> 1）吃一个；2）可以整除2:吃一半； 3）可以整除3，吃2/3
>
> 1. 尽可能选2）和3）方案
> 2. 对于任意n，2）和3）都不行时，就先选择方案1），直到2）和3）成立，然后比较两种情况下哪种天数更少

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ dfs**

```cpp
class Solution {
public:
    // f[n] n个橘子可以吃的最小天数
    // f[0] = 0, f[1] = 1;
    // f[x] = v
    //          f[x+1] = min(f[x+1], f[x]+1);
    //          f[x*2] = min(f[x*2], f[x]+1);
    //          f[x*3] = min(f[x*3], f[x]+1);
    // n范围比较大 说明可以找规律
    // ====> 找规律失败 说明考虑记忆化
    // 记忆化的策略由推理可知 尽可能吃2/3

    // 12ms
    unordered_map<int, int> m;
    int find(int x) {
        if (m.count(x)) return m[x];
        if (x == 0)
            return 0;
        else if (x == 1)
            return 1;
        else if (x == 2)
            return 2;
        return m[x] = min(find(x / 3) + x % 3, find(x / 2) + x % 2) + 1;
    }
    int minDays(int n) { return find(n); }
};
```

##### **C++ bfs**

```cpp
class Solution {
public:
    // 广搜
    // 156ms
    int minDays(int n) {
        unordered_map<int, bool> m;
        queue<int> q;
        q.push(n);
        int d = 0;
        while (!q.empty()) {
            int sz = q.size();
            while (sz--) {
                int nv = q.front();
                q.pop();
                if (!nv) return d;
                if (nv % 3 == 0 && !m[nv / 3]) {
                    q.push(nv / 3);
                    m[nv / 3] = true;
                }
                if (nv % 2 == 0 && !m[nv / 2]) {
                    q.push(nv / 2);
                    m[nv / 2] = true;
                }
                if (!m[nv - 1]) {
                    q.push(nv - 1);
                    m[nv - 1] = true;
                }
            }
            ++d;
        }
        return d;
    }
};
```

##### **Python**

```python
class Solution:
    @lru_cache(None)  # 记忆化搜索
    def minDays(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        return 1 + min(self.minDays(n // 2) + n % 2, self.minDays(n // 3) + n % 3)
```



**Python-BFS**

```python
class Solution:
    def minDays(self, n: int) -> int:
        q = deque([(n, 0)])
        my_set = set()
        while q:
            nums, cnt = q.popleft()
            if not nums:
                return cnt
            if nums in my_set:
                continue
            my_set.add(nums)
            q.append([nums - 1, cnt + 1])
            if nums % 2 == 0:
                q.append([nums // 2, cnt + 1])
            if nums % 3 == 0:
                q.append([nums // 3, cnt + 1])
```



<!-- tabs:end -->
</details>

<br>

* * *