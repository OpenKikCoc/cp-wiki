
> [!TIP] **常见优化**
> 
> 1. 优化搜索顺序 优先选择可选范围小的
> 2. 排除冗余信息
> 3. 可行性剪枝
> 4. 最优性剪枝
> 5. 记忆化（对于爆搜每次状态都不同 记忆化没用）

## 剪枝方法

最常用的剪枝有三种，记忆化搜索、最优性剪枝、可行性剪枝。

### 记忆化搜索

相同的传入值往往会带来相同的解，那我们就可以用数组来记忆，详见 [记忆化搜索](dp/memo.md)。

### 最优性剪枝

在搜索中导致运行慢的原因还有一种，就是在当前解已经比已有解差时仍然在搜索，那么我们只需要判断一下当前解是否已经差于已有解。

### 可行性剪枝

在搜索过程中当前解已经不可用了还继续搜索下去也是运行慢的原因。

## 剪枝思路

剪枝思路有很多种，大多需要对于具体问题来分析，在此简要介绍几种常见的剪枝思路。

- 极端法：考虑极端情况，如果最极端（最理想）的情况都无法满足，那么肯定实际情况搜出来的结果不会更优了。

- 调整法：通过对子树的比较剪掉重复子树和明显不是最有“前途”的子树。

- 数学方法：比如在图论中借助连通分量，数论中借助模方程的分析，借助不等式的放缩来估计下界等等。

## 例题

> [!NOTE] **工作分配问题**
>
> **题目描述**
>
> 有 $n$ 份工作要分配给 $n$ 个人来完成，每个人完成一份。第 $i$ 个人完成第 $k$ 份工作所用的时间为一个正整数 $t_{i,k}$，其中 $1 \leq i, k \leq n$。试确定一个分配方案，使得完成这 $n$ 份工作的时间总和最小。
>
> 输入包含 $n + 1$ 行。
>
> 第 1 行为一个正整数 $n$。
>
> 第 2 行到第 $n + 1$ 行中每行都包含 $n$ 个正整数，形成了一个 $n \times n$ 的矩阵。在该矩阵中，第 $i$ 行第 $k$ 列元素 $t_{i,k}$ 表示第 $i$ 个人完成第 $k$ 件工作所要用的时间。
>
> 输出包含一个正整数，表示所有分配方案中最小的时间总和。
>
> **数据范围**
>
> $1 \leq n \leq  15$
>
> $1 \leq t_{i,k} \leq 10^4$
>
> **输入样例**
>
>     5
>     9 2 9 1 9
>     1 9 8 9 6
>     9 9 9 9 1
>     8 8 1 8 4
>     9 1 7 8 9
>
> **输出样例**
>
>     5

> [!TIP] **思路**
>
>   由于每个人都必须分配到工作，在这里可以建一个二维数组 `time[i][j]`，用以表示 $i$ 个人完成 $j$ 号工作所花费的时间。
>
>   给定一个循环，从第 1 个人开始循环分配工作，直到所有人都分配到。为第 $i$ 个人分配工作时，再循环检查每个工作是否已被分配，没有则分配给 $i$ 个人，否则检查下一个工作。
>
>   可以用一个一维数组 `is_working[j]` 来表示第 $j$ 号工作是否已被分配，未分配则 `is_working[j]=0`，否则 `is_working[j]=1`。利用回溯思想，在工人循环结束后回到上一工人，取消此次分配的工作，而去分配下一工作直到可以分配为止。
>
>   这样，一直回溯到第 1 个工人后，就能得到所有的可行解。
>
>   检查工作分配，其实就是判断取得可行解时的二维数组的第一维下标各不相同并且第二维下标各不相同。
>
>   而我们是要得到完成这 $n$ 份工作的最小时间总和，即可行解中时间总和最小的一个，故需要再定义一个全局变量 `cost_time_total_min` 表示目前找到的解中最小的时间总和，初始 `cost_time_total_min` 为 `time[i][i]` 之和，即对角线工作时间相加之和。
>
>   在所有人分配完工作时，比较 `count` 与 `cost_time_total_min` 的大小，如果 `count` 小于 `cost_time_total_min`，说明找到了一个最优解，此时就把 `count` 赋给 `cost_time_total_min`。
>
>   但考虑到算法的效率，这里还有一个剪枝优化的工作可以做。就是在每次计算局部费用变量 `count` 的值时，如果判断 `count` 已经大于 `cost_time_total_min`，就没必要再往下分配了，因为这时得到的解必然不是最优解。

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

* * *

## 习题

### 简单预处理优化

> [!NOTE] **[LeetCode 131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)**
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
    vector<vector<bool>> f;
    vector<vector<string>> ans;
    vector<string> path;

    vector<vector<string>> partition(string s) {
        int n = s.size();
        f = vector<vector<bool>>(n, vector<bool>(n));
        for (int j = 0; j < n; j ++ )
            for (int i = 0; i <= j; i ++ )
                if (i == j) f[i][j] = true;
                else if (s[i] == s[j]) {
                    if (i + 1 > j - 1 || f[i + 1][j - 1]) f[i][j] = true;
                }

        dfs(s, 0);
        return ans;
    }

    void dfs(string& s, int u) {
        if (u == s.size()) ans.push_back(path);
        else {
            for (int i = u; i < s.size(); i ++ )
                if (f[u][i]) {
                    path.push_back(s.substr(u, i - u + 1));
                    dfs(s, i + 1);
                    path.pop_back();
                }
        }
    }
};
```

##### **Python**

```python
#python3
#法一：区间dp + dfs
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        
        def dfs(u, path):
            if u == n:  # u表示当前搜到第几位，当u ==n: 说明一遍已经搜完，计入答案
                res.append(path[:])
                return 
            for j in range(u, n): # 开始枚举下一段
                if f[u][j]:
                    path.append(s[u:j + 1])  # 加入路径中
                    dfs(j + 1, path)  # 递归到下一层，注意是j + 1, 因为j是上一段最后一个字符
                    path.pop()
        n = len(s)
        f = [[False] * n for _ in range(n)]
				
        for j in range(n):   # 由于递推式f[i][j] = (f[i+1][j-1]),在计算f[i][j]时j-1必须先被算出来，那么应该先枚举j
            for i in range(n):
                if i == j:  # 只有一个字符的情况下
                    f[i][j] = True
                elif s[i] == s[j]:
                    if i + 1 > j - 1 or ((s[i] == s[j] and f[i + 1][j - 1])): # 两个字符
                        f[i][j] = True
        # 区间dp经典写法
        #f[0][0] = True
        #for i in range(1, n):
        #    f[i][i] = True
        #    f[i-1][i] = (s[i-1] == s[i])
        #for length in range(2, n):
        #    for i in range(n - length):
        #        j = i + length
        #        if s[i] == s[j] and f[i + 1][j - 1] == 1:
        #            f[i][j] = True
    
        return res
    
    
# 法2: 回溯
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        
        def dfs(s, tmp):
            if not s:
                res.append(tmp)
                return 
            for i in range(1, len(s) + 1):
                if s[:i] == s[:i][::-1]:
                    dfs(s[i:], tmp + [s[:i]])
        dfs(s, [])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)**
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
    int minCut(string s) {
        int n = s.size();
        s = ' ' + s;
        vector<vector<bool>> g(n + 1, vector<bool>(n + 1));
        vector<int> f(n + 1, 1e8);

        for (int j = 1; j <= n; j ++ )
            for (int i = 1; i <= n; i ++ )
                if (i == j) g[i][j] = true;
                else if (s[i] == s[j]) {
                    if (i + 1 > j - 1 || g[i + 1][j - 1]) g[i][j] = true;
                }

        f[0] = 0;
        for (int i = 1; i <= n; i ++ ) {
            for (int j = 1; j <= i; j ++ )
                if (g[j][i])
                    f[i] = min(f[i], f[j - 1] + 1);
        }

        return f[n] - 1;
    }
};
```

##### **Python**

```python
"""
一共进行两次动态规划。

第一次动规：计算出每个子串是否是回文串。
状态表示：st[i][j] 表示 s[i…j] 是否是回文串;
转移方程：s[i…j] 是回文串当且仅当 s[i] 等于s[j] 并且 s[i+1…j−1] 是回文串；
边界情况：如果s[i…j] 的长度小于等于2，则st[i][j]=(s[i]==s[j]);

在第一次动规的基础上，我们进行第二次动规。
状态表示：f[i] 表示把前 i 个字符划分成回文串，最少划分成几部分；
状态转移：枚举最后一段回文串的起点 jj，然后利用 st[j][i] 可知 s[j…i] 是否是回文串，如果是回文串，则 f[i]可以从 f[j−1]+1 转移；
边界情况：0个字符可以划分成0部分，所以 f[0]=0。

题目让我们求最少切几刀，所以答案是 f[n]−1
时间复杂度分析：两次动规都是两重循环，所以时间复杂度是 O(n2)。

"""
class Solution:
    def minCut(self, s: str) -> int:
        n=len(s)
        s=' '+s
        # step1 先使用g表示g[i][j]是否是回文串
        g = [[False]*(n+1) for _ in range(n+1)]
        for j in range(1,n+1):
            for i in range(j+1):
                if i==j:
                    g[i][j]=True
                else:
                    if s[i]==s[j]:
                        if i+1>j-1 or g[i+1][j-1]:
                            g[i][j]=True

        # step2 使用dp[i]表示到第i个字符结尾，最少的分隔次数
        f=[float('inf')]*(n+1)
        f[0]=0
        for j in range(1,n+1):
            for i in range(1,j+1):
                if g[i][j]:
                    f[j]=min(f[j], f[i-1]+1)
        return f[n]-1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1278. 分割回文串 III](https://leetcode-cn.com/problems/palindrome-partitioning-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 区间dp预处理得到转变某个区间到回文串的 cost
> 
> 随后线性dp即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int palindromePartition(string s, int k) {
        int n = s.size();
        vector<vector<int>> cost(n + 1, vector<int>(n + 1));
        for (int len = 2; len <= n; ++len)
            for (int l = 1; l + len - 1 <= n; ++l) {
                int r = l + len - 1;
                cost[l][r] =
                    cost[l + 1][r - 1] + (s[l - 1] == s[r - 1] ? 0 : 1);
            }
        // 前i个字符拆成j个子串所需要的最小修改次数
        vector<vector<int>> f(n + 1, vector<int>(n + 1, inf));
        f[0][0] = 0;
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= min(k, j); ++j) {
                if (j == 1)
                    f[i][j] = cost[1][i];
                else {
                    // j-1 前面j-1段则长度最少j-1
                    for (int x = j - 1; x < i; ++x)
                        f[i][j] = min(f[i][j], f[x][j - 1] + cost[x + 1][i]);
                }
            }
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

> [!NOTE] **[LeetCode 1284. 转化为全零矩阵的最少反转次数](https://leetcode-cn.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 搜索 注意状态的改变 不止修改四个方向 还有本格
> 
> 记录写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int inf = 0x3f3f3f3f;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    int minFlips(vector<vector<int>>& mat) {
        int m = mat.size(), n = mat[0].size();
        int tot = n * m;
        vector<int> f(1 << tot, inf);
        int st = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (mat[i][j]) st ^= 1 << (i * n + j);
        f[st] = 0;
        queue<int> q;
        q.push(st);
        while (!q.empty()) {
            int x = q.front();
            q.pop();
            if (!x) return f[x];
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    // 改哪一个位置
                    int y = x ^ (1 << (i * n + j));
                    for (int k = 0; k < 4; ++k) {
                        int ni = i + dx[k], nj = j + dy[k];
                        if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
                        y ^= 1 << (ni * n + nj);
                    }
                    if (f[y] != inf) continue;
                    f[y] = f[x] + 1;
                    q.push(y);
                }
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

> [!NOTE] **[LeetCode 1307. 口算难题](https://leetcode-cn.com/problems/verbal-arithmetic-puzzle/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 初步考虑深搜回溯：
>
> >   在遍历 words 中每个字符代表对数字之后，需要再在此基础上遍历 result 串。
>
> 本质考察剪枝【后来加强过代码，很多人的略暴力的解法都过不了】。
>
> **考虑：**
>
> 1.  统计每个字符在数位上的贡献
>
> ```cpp
> unordered_map<char, int> m;
> 
>     for (auto w : words) {
>         int sz = w.size(), v = 1;
>         for(int i = sz - 1; i >= 0; -- i ) {
>             m[w[i]] += v;
>             v *= 10;
>         }
>         front[w[0]] = true; // 标记首位字符非零
>     }
> ```
>
> 2.  记录前导 0 以在搜索中跳过
>
> ```cpp
> unordered_map<char, bool> front;
> 
>     // front[w[0]] = true;
>     front[result[0]] = true;
> ```
>
> 3.  标记某个字符是否已有意义 以及意义为何数值；相应的 记录某个数值是否已被某字符代表
>
> ```cpp
> unordered_map<char, int> vis;
> vector<bool> use;
> 
>     // dfs
> ```
>
> 然后这样会超时。。。
> 
> **有预处理超快代码**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 800ms**

```cpp
class Solution {
public:
    int fac[100005], ss[100005];
    int use[16];

    set <char> all;
    vector <char> v;
    bool dfs(int pos, long long sum) {
        if (pos == v.size()) {
            return sum == 0;
        }
        
        char cur = v[pos];
        int st = 0;
        if (ss[cur] == 1) {
            st = 1;
        }
        for (int i = st; i < 10; i++) {
            if (!use[i]) {
                use[i] = 1;
                if (dfs(pos + 1, sum + (long long)fac[cur] * i)) {
                    return true;
                }
                use[i] = 0;
            }
        }
        return false;
    }
    
    bool isSolvable(vector<string>& words, string r) {
        for (char c = 'A'; c <= 'Z'; c++) {
            fac[c] = 0;
            ss[c] = 0;
        }
        for (int i = 0; i <= 10; i++) {
            use[i] = 0;
        }
        for (string w : words) {
            int cur = 1;
            for (int i = w.size() - 1; i >= 0; i--) {
                fac[w[i]] += cur;
                all.insert(w[i]);
                cur *= 10;
            }
            if (w.size() > 1) {
                ss[w[0]] = 1;
            }
        }
        int cur = 1;
        for (int i = r.size() - 1; i >= 0; i--) {
            fac[r[i]] -= cur;
            all.insert(r[i]);
            cur *= 10;
        }
        if (r.size() > 1) {
            ss[r[0]] = 1;
        }
        v.clear();
        for (char c : all) {
            v.push_back(c);
        }
        
        return dfs(0, 0);
    }
};
```

##### **C++ 4ms**

```cpp
using PCI = pair<char, int>;

class Solution {
private:
    vector<PCI> weight;
    vector<int> suffix_sum_min, suffix_sum_max;
    vector<int> lead_zero;
    bool used[10];

public:
    int pow10(int x) {
        int ret = 1;
        for (int i = 0; i < x; ++i) {
            ret *= 10;
        }
        return ret;
    }

    bool dfs(int pos, int total) {
        if (pos == weight.size()) {
            return total == 0;
        }
        if (!(total + suffix_sum_min[pos] <= 0 && 0 <= total + suffix_sum_max[pos])) {
            return false;
        }
        for (int i = lead_zero[pos]; i < 10; ++i) {
            if (!used[i]) {
                used[i] = true;
                bool check = dfs(pos + 1, total + weight[pos].second * i);
                used[i] = false;
                if (check) {
                    return true;
                }
            }
        }
        return false;
    }

    bool isSolvable(vector<string>& words, string result) {
        unordered_map<char, int> _weight;
        unordered_set<char> _lead_zero;
        for (const string& word: words) {
            for (int i = 0; i < word.size(); ++i) {
                _weight[word[i]] += pow10(word.size() - i - 1);
            }
            if (word.size() > 1) {
                _lead_zero.insert(word[0]);
            }
        }
        for (int i = 0; i < result.size(); ++i) {
            _weight[result[i]] -= pow10(result.size() - i - 1);
        }
        if (result.size() > 1) {
            _lead_zero.insert(result[0]);
        }

        weight = vector<PCI>(_weight.begin(), _weight.end());
        sort(weight.begin(), weight.end(), [](const PCI& u, const PCI& v) {
            return abs(u.second) > abs(v.second);
        });
        int n = weight.size();
        suffix_sum_min.resize(n);
        suffix_sum_max.resize(n);
        for (int i = 0; i < n; ++i) {
            vector<int> suffix_pos, suffix_neg;
            for (int j = i; j < n; ++j) {
                if (weight[j].second > 0) {
                    suffix_pos.push_back(weight[j].second);
                }
                else if (weight[j].second < 0) {
                    suffix_neg.push_back(weight[j].second);
                }
                sort(suffix_pos.begin(), suffix_pos.end());
                sort(suffix_neg.begin(), suffix_neg.end());
            }
            for (int j = 0; j < suffix_pos.size(); ++j) {
                suffix_sum_min[i] += (suffix_pos.size() - 1 - j) * suffix_pos[j];
                suffix_sum_max[i] += (10 - suffix_pos.size() + j) * suffix_pos[j];
            }
            for (int j = 0; j < suffix_neg.size(); ++j) {
                suffix_sum_min[i] += (9 - j) * suffix_neg[j];
                suffix_sum_max[i] += j * suffix_neg[j];
            }
        }

        lead_zero.resize(n);
        for (int i = 0; i < n; ++i) {
            lead_zero[i] = (_lead_zero.count(weight[i].first) ? 1 : 0);
        }
        
        memset(used, false, sizeof(used));
        return dfs(0, 0);
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

### TODO 整理下列其他优化

> [!NOTE] **[AcWing 165. 小猫爬山](https://www.acwing.com/problem/content/167/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dfs分组 + 剪枝

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 20;

int n, m;
int w[N];
int sum[N];
int ans = N;

void dfs(int u, int k) {
    // 最优性剪枝
    if (k >= ans) return;
    if (u == n) {
        ans = k;
        return;
    }

    for (int i = 0; i < k; i++)
        if (sum[i] + w[u] <= m)  // 可行性剪枝
        {
            sum[i] += w[u];
            dfs(u + 1, k);
            sum[i] -= w[u];  // 恢复现场
        }

    // 新开一辆车
    sum[k] = w[u];
    dfs(u + 1, k + 1);
    sum[k] = 0;  // 恢复现场
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; i++) cin >> w[i];

    // 优化搜索顺序
    sort(w, w + n);
    reverse(w, w + n);

    dfs(0, 0);

    cout << ans << endl;

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

> [!NOTE] **[AcWing 166. 数独](https://www.acwing.com/problem/content/168/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **必备题 学习如何快速状态表示**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 常用优化方式：
//  1. 优化搜索顺序 优先选择可选范围小的
//  2. 排除冗余信息
//  3. 可行性剪枝
//  4. 最优性剪枝
//  5. 记忆化（对于爆搜每次状态都不同 记忆化没用）

// 剪枝策略 每次选择分支较小的点进行dfs

const int N = 9, M = 1 << N;

// 这俩数组开优化
// 返回某个数有多少个1   map记录每个数的1在第几位
int ones[M], idx[M];
int row[N], col[N], cell[3][3];
char str[100];

inline int lowbit(int x) { return x & -x; }

inline int get(int x, int y) {
    // 获取当前位置可选数字
    return row[x] & col[y] & cell[x / 3][y / 3];
}

void init() {
    // 一开始每一行每一列的数字可以随便选
    int ALL = (1 << N) - 1;
    for (int i = 0; i < N; ++i)
        // row[i] = col[i] = (1 << N) - 1;
        row[i] = col[i] = ALL;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            // cell[i][j] = (1 << N) - 1;
            cell[i][j] = ALL;
}

void draw(int x, int y, int t, bool is_set) {
    if (is_set)
        str[x * N + y] = '1' + t;
    else
        str[x * N + y] = '.';

    int v = 1 << t;
    if (!is_set) v = -v;
    row[x] -= v;
    col[y] -= v;
    cell[x / 3][y / 3] -= v;
}

bool dfs(int cnt) {
    if (!cnt) return true;

    // 找到可选方案数最少的空格
    int minv = 10;
    int x, y;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (str[i * N + j] == '.') {
                int state = get(i, j);
                if (ones[state] < minv) minv = ones[state], x = i, y = j;
            }

    int state = get(x, y);
    for (int i = state; i; i -= lowbit(i)) {
        // 具体的数值 (t = v - 1)
        int t = idx[lowbit(i)];
        draw(x, y, t, true);
        if (dfs(cnt - 1)) return true;
        draw(x, y, t, false);
    }
    return false;
}

int main() {
    for (int i = 0; i < N; ++i) idx[1 << i] = i;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) ones[i] += i >> j & 1;
    while (cin >> str, str[0] != 'e') {
        init();

        int cnt = 0;
        // 把已经填的数加入限制里去
        for (int i = 0, k = 0; i < N; ++i)
            for (int j = 0; j < N; ++j, ++k)
                if (str[k] != '.')
                    draw(i, j, str[k] - '1', true);
                else
                    ++cnt;
        dfs(cnt);
        puts(str);
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

> [!NOTE] **[AcWing 167. 木棒](https://www.acwing.com/problem/content/169/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 学习剪枝姿势 for循环更新姿势

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 70;
int n, sum, length;
int w[N];
bool vis[N];

// 当前处于第几根棒  正在拼的木棒的长度 正在拼的木棒可用的第一个编号
bool dfs(int u, int cur, int start) {
    // 已有u个木棒 方案合法 返回
    if (u * length == sum) return true;
    // 当前棒已经到达长度 要一个新的棒 ==> 已知进入dfs前确保不会超过length了
    if (cur == length) return dfs(u + 1, 0, 0);

    for (int i = start; i < n; ++i) {
        if (vis[i] || cur + w[i] > length) continue;

        vis[i] = true;
        // 剪枝2：避免重复路径 选了大的不能再选比它小的 否则有重复
        if (dfs(u, cur + w[i], i + 1)) return true;
        vis[i] = false;

        // 剪枝3：当该木棍在开头和结尾都不可以使用的时候
        if (!cur || cur + w[i] == length) return false;

        int j = i;
        while (j < n && w[j] == w[i]) ++j;
        i = j - 1;
    }
    return false;
}

int main() {
    while (cin >> n, n) {
        memset(vis, false, sizeof vis);
        sum = 0;

        for (int i = 0; i < n; ++i) cin >> w[i], sum += w[i];
        // 剪枝1：优先选长的
        sort(w, w + n);
        reverse(w, w + n);

        // length 可以再优化 获取w最大值
        length = 1;
        for (;;) {
            if (sum % length == 0 && dfs(0, 0, 0)) break;
            ++length;
        }
        cout << length << endl;
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

> [!NOTE] **[AcWing 168. 生日蛋糕](https://www.acwing.com/problem/content/170/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 学习剪枝姿势

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 优先选择可选较少的 依据题目条件下面底面积大 故从下往上搜

// r^2h <= n-V
// 先枚举r

const int N = 25;
const int inf = 1e9;

int n, m, res;
int R[N], H[N];        // 每一层的半径和高度
int minv[N], mins[N];  // 前i层体积、侧面积最小值

void dfs(int u, int v, int s) {
    if (v + minv[u] > n) return;     // 剪枝5：不合法
    if (s + mins[u] >= res) return;  // 剪枝6：不会更优
    if (s + 2 * (n - v) / R[u + 1] >= res)
        return;  // 剪枝4：当前表面积+未来最小的表面积>res 不需要再继续

    if (!u) {
        if (v == n) {
            // res = min(res, s);
            res = s;
        }
        return;
    }

    // 剪枝2：先枚举r 再枚举h
    // 剪枝3：r h的起始范围
    for (int r = min(R[u + 1] - 1, (int)sqrt(n - v)); r >= u; --r)
        for (int h = min(H[u + 1] - 1, (n - v) / r / r); h >= u; --h) {
            int t = 0;
            if (u == m) t = r * r;  // 最后一层 计算从上至下累加表面积
            R[u] = r, H[u] = h;
            dfs(u - 1, v + r * r * h, s + 2 * r * h + t);
        }
}

int main() {
    res = inf;
    cin >> n >> m;
    // 从上到下 每一层最小体积和表面积
    for (int i = 1; i <= m; ++i) {
        minv[i] = minv[i - 1] + i * i * i;
        mins[i] = mins[i - 1] + 2 * i * i;
    }
    R[m + 1] = H[m + 1] = inf;
    // 剪枝1：从下至上枚举
    dfs(m, 0, 0);

    if (res == inf) res = 0;
    cout << res << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1418. 栅栏围栏](https://www.acwing.com/problem/content/1420/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **复杂剪枝 贪心 优化**
> 
> 爆搜即可
> 
> 显然如果短的可以满足 更长的一定可以满足 故首先【从小到大排】
> 
> 枚举顺序和剪枝细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 55, M = 1050;

int n, m;
int board[N], rail[M], sum[M], tot;
int mid;

// u 要裁减的编号 start 木板号
bool dfs(int u, int start) {
    if (!u) return true;
    // 总长度都剪不出来 false
    if (tot < sum[u]) return false;
    // 循环 注意条件
    if (u + 1 > mid || rail[u] != rail[u + 1]) start = 1;
    for (int i = start; i <= n; ++ i ) {
        // 去重
        if (i > start && board[i] == board[i - 1]) continue;
        if (board[i] >= rail[u]) {
            // 更改
            tot -= rail[u];
            board[i] -= rail[u];
            if (board[i] < rail[1]) tot -= board[i];
            
            if (dfs(u - 1, i)) {
                // 恢复
                if (board[i] < rail[1]) tot += board[i];
                board[i] += rail[u];
                tot += rail[u];
                return true;
            }
            
            // 恢复
            if (board[i] < rail[1]) tot += board[i];
            board[i] += rail[u];
            tot += rail[u];
        }
    }
    return false;
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i ) cin >> board[i], tot += board[i];
    cin >> m;
    for (int i = 1; i <= m; ++ i ) cin >> rail[i];
    
    sort(board + 1, board + n + 1);
    sort(rail + 1, rail + m + 1);
    for (int i = 1; i <= m; ++ i ) sum[i] = sum[i - 1] + rail[i];
    
    // 有单调性
    int l = 0, r = m;
    while (l < r) {
        mid = l + r + 1 >> 1;
        // 写法
        if (dfs(mid, 1)) l = mid;
        else r = mid - 1;
    }
    cout << l << endl;
    
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

> [!NOTE] **[AcWing 1419. 牛的密码学](https://www.acwing.com/problem/content/1421/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂模拟 爆搜 剪枝

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 爆搜
#include <bits/stdc++.h>
using namespace std;

using ULL = unsigned long long;
const int P = 131;

string target = "Begin the Escape execution at the Break of Dawn";
unordered_set<ULL> S, valid;

ULL get_hash(string state) {
    ULL res = 0;
    for (auto c : state) res = res * P + c;
    return res;
}

// 检查其他字符够不够
bool check1(string state) {
    int c = 0, o = 0, w = 0;
    string a = target, b;
    for (auto x : state)
        if (x == 'C') ++ c ;
        else if (x == 'O') ++ o ;
        else if (x == 'W') ++ w ;
        else b += x;
    sort(a.begin(), a.end()), sort(b.begin(), b.end());
    return a == b;
}

// state是否合法
bool check2(string state) {
    if (state.size() == target.size()) return false;
    // 找到最外的两个子段 检查
    int l = 0, r = state.size() - 1;
    while (state[l] != 'C' && state[l] != 'O' && state[l] != 'W') ++ l ;
    while (state[r] != 'C' && state[r] != 'O' && state[r] != 'W') -- r ;
    if (state[l] != 'C' || state[r] != 'W') return false;
    if (state.substr(0, l) != target.substr(0, l)) return false;
    if (state.substr(r + 1) != target.substr(target.size() - (state.size() - r - 1))) return false;
    
    // 判断每一个子串 是否出现过
    string s;
    for (int i = l + 1; i <= r; ++ i ) {
        auto c = state[i];
        if (c == 'C' || c == 'O' || c == 'W') {
            if (valid.count(get_hash(s)) == 0) return false;
            s.clear();
        } else s += c;
    }
    return true;
}

bool dfs(string state) {
    if (state == target) return true;
    if (!check2(state)) return false;   // 保证了state不会比target短
    auto h = get_hash(state);
    if (S.count(h)) return false;
    S.insert(h);
    
    for (int o = 0; o < state.size(); ++ o ) {
        if (state[o] != 'O') continue;
        for (int c = 0; c < o; ++ c ) {
            if (state[c] != 'C') continue;
            for (int w = o + 1; w < state.size(); ++ w ) {
                if (state[w] != 'W') continue;
                auto s1 = state.substr(0, c), s2 = state.substr(c + 1, o - c - 1);
                auto s3 = state.substr(o + 1, w - o - 1), s4 = state.substr(w + 1);
                if (dfs(s1 + s3 + s2 + s4)) return true;
            }
        }
    }
    return false;
}

int main() {
    // 加入目标的所有子串的hash
    valid.insert(0);
    for (int i = 0; i < target.size(); ++ i )
        for (int j = i; j < target.size(); ++ j )
            valid.insert(get_hash(target.substr(i, j - i + 1)));
    
    string start;
    getline(cin, start);
    if (!check1(start)) cout << "0 0" << endl;
    else if (!dfs(start)) cout << "0 0" << endl;
    else cout << 1 << ' ' << (start.size() - target.size()) / 3 << endl;

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

> [!NOTE] **[AcWing 1122. 质数方阵](https://www.acwing.com/problem/content/1124/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 搜索顺序 贪心 剪枝
> 
> 从搜索顺序来优化
> 
> 先设置对角线
> 
> cx[] 表示第x位共用的情况下都有哪些数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010, M = 10;

int n, m;
int primes[N], cnt;
bool st[N];
vector<int> c1[M], c3[M], c15[M][M], c124[M][M][M];
int g[6][6];
vector<string> ans;

int get(int x, int k) {
    int t = 1;
    for (int i = 0; i < 5 - k; ++ i ) t *= 10;
    return x / t % 10;
}

void init() {
    for (int i = 2; i < N; ++ i ) {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= N / i; ++ j ) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
    
    for (int i = 0; i < cnt; ++ i ) {
        int p = primes[i];
        if (p < 10000 || p > 99999) continue;
        int n1 = get(p, 1), n2 = get(p, 2), n3 = get(p, 3), n4 = get(p, 4), n5 = get(p, 5);
        if (n1 + n2 + n3 + n4 + n5 != n) continue;
        c1[n1].push_back(p);
        c3[n3].push_back(p);
        if (n2 && n3 && n4)
            c15[n1][n5].push_back(p);
        c124[n1][n2][n4].push_back(p);
    }
}

bool check(int x1, int y1, int x2, int y2) {
    int s = 0;
    for (int i = x1; i <= x2; ++ i )
        for (int j = y1; j <= y2; ++ j ) {
            int x = g[i][j];
            if (x < 0 || x > 9) return false;
            s = s * 10 + x;
        }
    return !st[s];
}

// 按特定顺序爆搜
void dfs(int u) {
    if (u > 7) {
        g[3][5] = n - g[3][1] - g[3][2] - g[3][3] - g[3][4];
        g[4][5] = n - g[1][5] - g[2][5] - g[3][5] - g[5][5];
        g[4][3] = n - g[4][1] - g[4][2] - g[4][4] - g[4][5];
        g[5][3] = n - g[5][1] - g[5][2] - g[5][4] - g[5][5];

        if (check(1, 3, 5, 3) && check(1, 5, 5, 5) && check(3, 1, 3, 5) &&
            check(4, 1, 4, 5) && check(5, 1, 5, 5)) {
            string s;
            for (int i = 1; i <= 5; i ++ )
                for (int j = 1; j <= 5; j ++ )
                    s += to_string(g[i][j]);
            ans.push_back(s);
        }
        return;
    }
    if (u == 1) {
        for (auto x : c1[g[1][1]]) {
            g[2][2] = get(x, 2), g[3][3] = get(x, 3), g[4][4] = get(x, 4), g[5][5] = get(x, 5);
            dfs(u + 1);
        }
    } else if (u == 2) {
        for (auto x : c3[g[3][3]]) {
            g[5][1] = get(x, 1), g[4][2] = get(x, 2), g[2][4] = get(x, 4), g[1][5] = get(x, 5);
            dfs(u + 1);
        }
    } else if (u == 3) {
        for (auto x : c15[g[1][1]][g[1][5]]) {
            g[1][2] = get(x, 2), g[1][3] = get(x, 3), g[1][4] = get(x, 4);
            dfs(u + 1);
        }
    } else if (u == 4) {
        for (auto x : c124[g[1][2]][g[2][2]][g[4][2]]) {
            g[3][2] = get(x, 3), g[5][2] = get(x, 5);
            dfs(u + 1);
        }
    } else if (u == 5) {
        for (auto x : c124[g[1][4]][g[2][4]][g[4][4]]) {
            g[3][4] = get(x, 3), g[5][4] = get(x, 5);
            dfs(u + 1);
        }
    } else if (u == 6) {
        for (auto x : c15[g[1][1]][g[5][1]]) {
            g[2][1] = get(x, 2), g[3][1] = get(x, 3), g[4][1] = get(x, 4);
            dfs(u + 1);
        }
    } else {
        for (auto x : c124[g[2][1]][g[2][2]][g[2][4]]) {
            g[2][3] = get(x, 3), g[2][5] = get(x, 5);
            dfs(u + 1);
        }
    }
}

int main() {
    cin >> n >> m;
    init();
    
    g[1][1] = m;
    dfs(1);
    
    if (ans.empty()) cout << "NONE" << endl;
    else {
        sort(ans.begin(), ans.end());
        for (int i = 0; i < ans.size(); ++ i ) {
            for (int j = 0; j < 25; ++ j ) {
                cout << ans[i][j];
                if ((j + 1) % 5 == 0) cout << endl;
            }
            if (i + 1 < ans.size()) cout << endl;
        }
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

> [!NOTE] **[AcWing 1422. 拉丁矩阵](https://www.acwing.com/problem/content/1424/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 搜索
> 
> **分析性质 剪枝 置换群**
> 
> 性质
> 
> 1. 交换两行 / 两列仍是
> 
> 2. 1-n置换 仍是
> 
> 已经要求了第一行 1-N
> 
> 根据置换分析 固定第一列也是1-N
> 
> 置换圈个数相同及对应的置换圈内元素个数相同即视为相同方案

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 10;

int n;
int g[N][N];
bool row[N][N], col[N][N];
map<vector<int>, int> f;

void put(int x, int y, int k) {
    g[x][y] = k;
    row[x][k] = true;
    col[y][k] = true;
}

void unput(int x, int y) {
    int k = g[x][y];
    row[x][k] = false;
    col[y][k] = false;
    g[x][y] = 0;
}

int dfs(int x, int y) {
    // 从下一行再开始
    if (y > n) ++ x, y = 2;
    // 【剪枝 2】第n行不用判断
    if (x == n) return 1;
    
    // 【剪枝 3】 置换群的优化剪枝
    // x = 3 y = 2 时前两行的置换已经找完了
    // [只要其结构相同 就算一种]
    if (x == 3 && y == 2) {
        vector<int> line;
        bool st[N] = {0};
        for (int i = 1; i <= n; ++ i ) {
            if (st[i]) continue;
            int s = 0;
            for (int j = i; !st[j]; j = g[2][j]) {
                ++ s;
                st[j] = true;
            }
            line.push_back(s);
        }
        sort(line.begin(), line.end());
        if (f.count(line)) return f[line];
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            if (!row[x][i] && !col[y][i]) {
                put(x, y, i);
                res += dfs(x, y + 1);
                unput(x, y);
            }
        return f[line] = res;
    }
    
    int res = 0;
    for (int i = 1; i <= n; ++ i )
        if (!row[x][i] && !col[y][i]) {
            put(x, y, i);
            res += dfs(x, y + 1);
            unput(x, y);
        }
    return res;
}

int main() {
    cin >> n;
    // 【剪枝 1】第一列也1-n
    for (int i = 1; i <= n; ++ i ) {
        put(1, i, i);
        put(i, 1, i);
    }
    
    LL res = dfs(2, 2);
    // 计算置换
    for (int i = 1; i <= n - 1; ++ i ) res *= i;
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

> [!NOTE] **[AcWing 1430. 贝特西之旅](https://www.acwing.com/problem/content/1432/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **搜索 若数据范围更大需插头dp**
> 
> 分析 剪枝
> 
> [剪枝1] 每个各自出入各来自一个方向 四选二
> 
> 判断周围空格数量是否小于等于1
> 
> [剪枝1] 上下 or 左右 都走过 必然无解

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 10;

int n;
bool st[N][N];
int dx[] = {-1, 0, 0, 1}, dy[] = {0, -1, 1, 0};

// 周围空格数量小于等于1
bool check(int x, int y) {
    int cnt = 0;
    for (int i = 0; i < 4; ++ i ) {
        int a = x + dx[i], b = y + dy[i];
        if (!st[a][b])
            ++ cnt;
    }
    return cnt <= 1;
}

int dfs(int x, int y, int u) {
    if (x == n && y == 1) {
        if (u == n * n) return 1;
        return 0;
    }
    
    // 剪枝2
    if (st[x - 1][y] && st[x + 1][y] && !st[x][y - 1] && !st[x][y + 1] || 
        st[x][y - 1] && st[x][y + 1] && !st[x - 1][y] && !st[x + 1][y])
        return 0;
    
    // 【该位置的周围空格】的周围空格数量有几个大于等于1
    int cnt = 0, sx, sy;
    for (int i = 0; i < 4; ++ i ) {
        int nx = x + dx[i], ny = y + dy[i];
        if (!(nx == n && ny == 1) && !st[nx][ny] && check(nx, ny)) {
            cnt ++ ;
            sx = nx, sy = ny;
        }
    }
    
    int res = 0;
    if (cnt > 1) return 0;
    else if (cnt == 1) {
        st[sx][sy] = true;
        res += dfs(sx, sy, u + 1);
        st[sx][sy] = false;
    } else {
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (!st[nx][ny]) {
                st[nx][ny] = true;
                res += dfs(nx, ny, u + 1);
                st[nx][ny] = false;
            }
        }
    }
    return res;
}

int main() {
    cin >> n;
    for (int i = 0; i <= n + 1; ++ i )
        st[i][0] = st[i][n + 1] = st[0][i] = st[n + 1][i] = true;
        
    st[1][1] = true;
    cout << dfs(1, 1, 1) << endl;
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



> [!NOTE] **[AcWing 1431. 时钟](https://www.acwing.com/problem/content/1433/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 搜索
> 
> 分析 剪枝

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 每个操作最多执行3次
#include <bits/stdc++.h>
using namespace std;

const int N = 10;

int q[N];
vector<int> ans, path;
string ops[9] = {
    "ABDE", "ABC", "BCEF", "ADG", "BDEFH",
    "CFI", "DEGH", "GHI", "EFHI"
};

bool check() {
    for (int i = 0; i < 9; ++ i )
        if (q[i] != 12)
            return false;
    return true;
}

void rotate(int u) {
    for (auto c : ops[u]) {
        int k = c - 'A';
        q[k] += 3;
        if (q[k] == 15) q[k] = 3;
    }
}

// 爆搜每种操作执行多少次
void dfs(int u) {
    if (u == 9) {
        if (check()) {
            if (ans.empty() || ans.size() > path.size() || 
                ans.size() == path.size() && ans > path)
                ans = path;
        }
        return;
    }
    
    // 最多旋转3次 第4次相当于恢复
    for (int i = 0; i < 4; ++ i ) {
        // u + 1 为编号 u 为下标
        dfs(u + 1);
        path.push_back(u + 1);
        rotate(u);
    }
    for (int i = 0; i < 4; ++ i ) path.pop_back();
}

int main() {
    for (int i = 0; i < 9; ++ i ) cin >> q[i];
    dfs(0);
    
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

> [!NOTE] **[Luogu [NOIP2018 提高组] 旅行](https://www.luogu.com.cn/problem/P5022)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **暴力 + 剪枝**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 题目条件很重要 n == m(有一个环) OR n == m + 1
//
// 对于无环情况爆搜即可
// 对于有环情况枚举环上要删除的边 再暴力枚举 并与现有答案比对剪枝即可
// O(n^2)

using PII = pair<int, int>;
const int N = 5010;

int n, m;
vector<int> e[N];   // 需要排序
PII edge[N];
int del_u, del_v;
vector<int> ans(N, N);
vector<int> path(N);
bool st[N];
int cnt, state;

// 思考使用 bool 返回值的意义
// 本质是如果发现大于就一直返回 直到不大于为止再更新新的答案
bool dfs(int u) {
    if (!state) {
        // TODO
        if (u > ans[cnt])
            return true;
        if (u < ans[cnt])
            state = -1;
    }
    
    st[u] = true;
    path[cnt ++ ] = u;
    
    for (int i = 0; i < e[u].size(); ++ i ) {
        int x = e[u][i];
        if (!(x == del_u && u == del_v) && !(x == del_v && u == del_u) && !st[x])
            if (dfs(x))
                return true;
    }
    return false;
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < m; ++ i ) {
        int a, b;
        cin >> a >> b;
        e[a].push_back(b);
        e[b].push_back(a);
        edge[i] = {a, b};
    }
    
    for (int i = 1; i <= n; ++ i )
        sort(e[i].begin(), e[i].end());
    
    if (n == m) {
        for (int i = 0; i < m; ++ i ) {
            del_u = edge[i].first, del_v = edge[i].second;
            
            memset(st, 0, sizeof st);
            cnt = state = 0;
            dfs(1); // 显然要字典序最小 总是要从1开始
            if (cnt == n)
                ans = path;
        }
    } else {
        dfs(1);
        if (cnt == n)
            ans = path;
    }
    
    for (int i = 0; i < n; ++ i )
        cout << ans[i] << ' ';
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

> [!NOTE] **[LeetCode 473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/)**
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
    vector<int> nums;
    vector<bool> st;
    // 最多15根火柴 压缩
    bool dfs(int start, int cur, int length, int cnt) {
        if (cnt == 3) return true;
        if (cur == length) return dfs(0, 0, length, cnt + 1);
        for (int i = start; i < nums.size(); ++ i ) {
            if (st[i]) continue;
            if (cur + nums[i] <= length) {
                st[i] = true;
                if (dfs(i + 1, cur + nums[i], length, cnt)) return true;
                st[i] = false;
            }
            // !cur 说明还未使用的最长火柴都不可匹配
            // cur + nums[i] == length 后面的和都会小于 length
            if (!cur || cur + nums[i] == length) return false;
            while (i + 1 < nums.size() && nums[i + 1] == nums[i]) ++ i ;
        }
        return false;
    }

    bool makesquare(vector<int>& nums) {
        this->nums = nums;
        if (nums.empty()) return false;
        int sum = 0, n = nums.size();
        for (auto v : nums) sum += v;
        if (sum % 4) return false;
        
        st.resize(nums.size());
        sum /= 4;
        sort(nums.begin(), nums.end(), greater<int>());
        return dfs(0, 0, sum, 0);
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

> [!NOTE] **[LeetCode 491. 递增子序列](https://leetcode-cn.com/problems/increasing-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **极好的搜索题 堪称LeetCode最佳**
> 
> 需去重
> 
> 重点在于递归时枚举的含义
> 
> 【每个位置该放什么】进而用 set 去重
> 
> > 需判重 枚举path中每个位置应该放什么

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// yxc
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void dfs(vector<int> & nums, int pos) {
        if (path.size() >= 2) res.push_back(path);
        if (pos == nums.size()) return;
        unordered_set<int> S;
        for (int i = pos; i < nums.size(); ++ i )
            if (path.empty() || nums[i] >= path.back()) {
                // if (i > pos && nums[i] == nums[i - 1]) continue; // is wrong cause the `nums` is not sorted
                if (S.count(nums[i])) continue;
                S.insert(nums[i]);
                path.push_back(nums[i]);
                dfs(nums, i + 1);
                path.pop_back();
            }
    }
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        dfs(nums, 0);
        return res;
    }
};

```

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> stack;
    bool is_first(vector<int>& nums, int last, int pos) {
        for (int i = last + 1; i < pos; ++ i )
            if (nums[i] == nums[pos]) return false;
        return true;
    }
    void dfs(vector<int>& nums, int last, int pos) {
        if (pos == nums.size()) return;
        if ((stack.empty() || nums[pos] >= stack.back()) && is_first(nums, last, pos)) {
            stack.push_back(nums[pos]);
            if (stack.size() >= 2) res.push_back(stack);
            dfs(nums, pos, pos + 1);
            stack.pop_back();
        }
        dfs(nums, last, pos + 1);
    }
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        dfs(nums, -1, 0);
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