## 习题


> [!NOTE] **[LeetCode 1145. 二叉树着色游戏](https://leetcode.cn/problems/binary-tree-coloring-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 检查 x 点所划分的不同连通块最大的一个的个数即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int s1, s2, x;
    
    int count(TreeNode * root) {
        if (!root)
            return 0;
        
        int l = count(root->left), r = count(root->right);
        
        if (root->val == x)
            s1 = l, s2 = r;
        
        return l + r + 1;
    }
    
    void dfs(TreeNode * root) {
        if (!root)
            return;
        
        if (root->val == x) {
            count(root);
        } else {
            dfs(root->left);
            dfs(root->right);
        }
    }
    
    bool btreeGameWinningMove(TreeNode* root, int n, int x) {
        this->x = x;
        s1 = s2 = 0;
        dfs(root);
        
        int v = max(max(s1, s2), n - 1 - s1 - s2);
        return v > n / 2;
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

> [!NOTE] **[LeetCode 1728. 猫和老鼠 II](https://leetcode.cn/problems/cat-and-mouse-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂博弈 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 可能会有循环 走 x 步之后 [cx, cy]  [mx, my] 又回到原来的位置
// 故加一维 k 记录步数   至于出现 1000 步的情况推测在之前某个位置就循环
// 某个位置设置为 200
int f[8][8][8][8][200];

class Solution {
public:
    int n, m, cj, mj;
    vector<string> g;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, 1, -1, 0};
    
    int dp(int cx, int cy, int mx, int my, int k) {
        if (k >= 200) return 0;
        auto & v = f[cx][cy][mx][my][k];
        if (v != -1) return v;
        
        if (k & 1) {    // 猫
            for (int i = 0; i < 4; ++ i )
                for (int j = 0; j <= cj; ++ j ) {
                    int x = cx + dx[i] * j, y = cy + dy[i] * j;
                    if (x < 0 || x >= n || y < 0 || y >= m || g[x][y] == '#') break;
                    if (x == mx && y == my) return v = 0;
                    if (g[x][y] == 'F') return v = 0;
                    if (!dp(x, y, mx, my, k + 1)) return v = 0;
                }
            return v = 1;
        } else {        // 老鼠
            for (int i = 0; i < 4; ++ i )
                for (int j = 0; j <= mj; ++ j ) {
                    int x = mx + dx[i] * j, y = my + dy[i] * j;
                    if (x < 0 || x >= n || y < 0 || y >= m || g[x][y] == '#') break;
                    if (x == cx && y == cy) continue;
                    if (g[x][y] == 'F') return v = 1;
                    if (dp(cx, cy, x, y, k + 1)) return v = 1;
                }
            return v = 0;
        }
    }
    
    bool canMouseWin(vector<string>& grid, int catJump, int mouseJump) {
        g = grid;
        n = g.size(), m = g[0].size(), cj = catJump, mj = mouseJump;
        int cx, cy, mx, my;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (g[i][j] == 'C') cx = i, cy = j;
                else if (g[i][j] == 'M') mx = i, my = j;
        memset(f, -1, sizeof f);
        return dp(cx, cy, mx, my, 0);
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

> [!NOTE] **[LeetCode 1927. 求和游戏](https://leetcode.cn/problems/sum-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单博弈 推导易知：
> 
> 1. 对于前后都有的可填充位置一一模仿
> 
> 2. 单个方向多余的位值分给两个人
> 
> 3. 如果alice总有办法使得前后不等 返回true
> 
> 4. 其余返回false
> 
> 另一解法: 分情况讨论（比较麻烦）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 博弈**

```cpp
class Solution {
public:
    bool sumGame(string num) {
        int n = num.size();
        int c1 = 0, c2 = 0;
        int s1 = 0, s2 = 0;
        for (int i = 0; i < n / 2; ++ i )
            if (num[i] != '?')
                s1 += num[i] - '0';
            else
                c1 ++ ;
        for (int i = n / 2; i < n; ++ i )
            if (num[i] != '?')
                s2 += num[i] - '0';
            else
                c2 ++ ;
        int sd = abs(s1 - s2), cd = abs(c1 - c2);
        int t1 = (cd + 1) / 2, t2 = cd / 2;
        if (t1 * 9 > sd || t2 * 9 < sd)
            return true;
        return false;
    }
};
```

##### **C++ 分情况讨论**

```cpp
class Solution {
public:
    bool sumGame(string num) {
        int sum = 0, cnt = 0, n = num.size();
        for (int i = 0; i < n / 2; i ++ ) {
            if (num[i] == '?') cnt ++ ;
            else sum += num[i] - '0';
        }
        for (int i = n / 2; i < n; i ++ ) {
            if (num[i] == '?') cnt -- ;
            else sum -= num[i] - '0';
        }

        if (!sum) return cnt;
        if (sum < 0) sum *= -1, cnt *= -1;
        if (cnt >= 0) return true;
        cnt *= -1;
        if (cnt % 2) return true;
        if (cnt / 2 * 9 == sum) return false;
        return true;
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

> [!NOTE] **[LeetCode 2029. 石子游戏 IX](https://leetcode.cn/problems/stone-game-ix/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始在想 区间DP
> 
> 其实是情况较复杂的分情况讨论
> 
> > 这题的题意设计很有意思 1 和 2 先后选取恰好对应 mod 3 的各类情况
> 
> 既是博弈论题，也是分情况讨论题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool stoneGameIX(vector<int>& stones) {
        int s[3] = {0, 0, 0};
        for (int i : stones)
            s[i % 3] ++ ;
        
        // s[0] 仅用作换手
        
        // 当 s[0] 为偶数，显然消除换手，只考虑 s[1] s[2] 即可
        // 如果 s[1] s[2] 任一为 0，则 alice 必败
        // ==> 分情况讨论
        //      s[1] = 0: alice 只能取 2 后面 bob 跟着取 2
        //                      后面 [取光] 或 [alice 三的倍数] 必败
        //      s[2] = 0: alice 只能取 1 后面 bob 跟着取 1
        //                      同理
        // 否则必胜
        if (s[0] % 2 == 0)
            return s[1] != 0 && s[2] != 0;
        
        // s[0] % 2 == 1 必然有一次换手
        // ==> 分情况讨论
        //      s[1] = s[2]: 则相当于 bob 先手选 s[1] s[2]
        //                   alice 为了跟上 bob 必须跟着取 最终取到最后石子(三的倍数) 必败
        //      abs(s[1] - s[2]) <= 2:  不管 alice 先取哪个 bob 都可以换手
        //                              最终石子取完 必败
        //      abs(s[1] - s[2]) > 2:   alice 取较多的 最终 bob 会到达三的倍数的情况 必胜
        return abs(s[1] - s[2]) > 2;
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

> [!NOTE] **[LeetCode 810. 黑板异或游戏](https://leetcode.cn/problems/chalkboard-xor-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本质是分情况讨论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool xorGame(vector<int>& nums) {
        // 如果长度为偶数，不管当前状况如何一定可以赢
        // - 0 直接赢
        // - 非0 则一定可以拿一个数得到新的非0 随后进入循环操作
        if (nums.size() % 2 == 0)
            return true;
        
        // 如果长度为奇数，只能当前局面 0
        int x = 0;
        for (auto y : nums)
            x ^= y;
        return x == 0;
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

> [!NOTE] **[LeetCode 913. 猫和老鼠](https://leetcode.cn/problems/cat-and-mouse/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 博弈论 注意转移设计

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // O(n^3) DP
    using TIII = tuple<int, int, int>;
    const static int N = 51;

    // 老鼠在 i 猫在 j；0 则下一步老鼠动 1 则下一步猫动
    int f[N][N][2], deg[N][N][2];

    int catMouseGame(vector<vector<int>>& graph) {
        int n = graph.size();

        memset(f, 0, sizeof f);
        for (int i = 0; i < n; ++ i )
            for (int j = 1; j < n; ++ j ) {
                // TODO
                deg[i][j][0] = graph[i].size();
                deg[i][j][1] = graph[j].size();
        }
        for (int i = 0; i < n; ++ i )
            for (auto x : graph[0])
                deg[i][x][1] -- ;
        
        // 将已经确定的状态加入队列，按照拓扑顺序进行动态规划的转移。
        // 队列里只存储可以完全确定老鼠获胜或者猫获胜的状态，不考虑平局的未知状态。
        queue<TIII> q;
        // 初始化老鼠必胜
        for (int j = 1; j < n; ++ j ) {
            f[0][j][0] = f[0][j][1] = 1;
            q.push({0, j, 0}), q.push({0, j, 1});
        }
        // 初始化猫必胜
        for (int i = 1; i < n; ++ i ) {
            f[i][i][0] = f[i][i][1] = 2;
            q.push({i, i, 1}), q.push({i, i, 0});
        }

        while (!q.empty()) {
            auto [i, j, k] = q.front(); q.pop();
            if (i == 1 && j == 2 && k == 0)
                break;
            
            if (k == 0) {
                // ATTENTION 实现
                // 如果当前状态是老鼠移动，且是猫获胜，则上一步猫移动时，则必定会走到这个猫必胜的状态
                // 所以所有相连的上一步的猫状态为猫必胜，且进队。
                for (auto x : graph[j]) {
                    if (x == 0)
                        continue;
                    if (f[i][x][1] != 0)
                        continue;
                    
                    if (f[i][j][k] == 2) {
                        f[i][x][1] = 2;
                        q.push({i, x, 1});
                    } else {
                        deg[i][x][1] -- ;
                        if (deg[i][x][1] == 0) {
                            f[i][x][1] = 1;
                            q.push({i, x, 1});
                        }
                    }
                }
            } else {
                for (auto x : graph[i]) {
                    if (f[x][j][0] != 0)
                        continue;
                    
                    if (f[i][j][k] == 1) {
                        f[x][j][0] = 1;
                        q.push({x, j, 0});
                    } else {
                        deg[x][j][0] -- ;
                        if (deg[x][j][0] == 0) {
                            f[x][j][0] = 2;
                            q.push({x, j, 0});
                        }
                    }
                }
            }
        }
        return f[1][2][0];
    }
};
```

##### **C++ 记忆化搜索 TLE**

```cpp
class Solution {
public:
    const static int N = 210;

    int f[N * 2][N][N], n;
    vector<vector<int>> g;

    int dp(int k, int i, int j) {
        int & v = f[k][i][j];
        if (v != -1)
            return v;
        // k > n * 2 认为平局，实际上这样会 WA 66/92
        // if (k > n * 2)
        if (k > n * 2 * 4)  // 用 k > n * 8 则超时 92/92
            return v = 0;
        if (!i)
            return v = 1;
        if (i == j)
            return v = 2;
        
        if (k % 2 == 0) {
            // 老鼠走
            int draws = 0;
            for (auto x : g[i]) {
                int t = dp(k + 1, x, j);
                if (t == 1)
                    return v = 1;
                if (!t)
                    draws ++ ;
            }
            if (draws)  // 如果不能赢 能平则平
                return v = 0;
            return v = 2;
        } else {
            int draws = 0;
            for (auto x : g[j]) {
                if (!x)
                    continue;
                int t = dp(k + 1, i, x);
                if (t == 2)
                    return v = 2;
                if (!t)
                    draws ++ ;
            }
            if (draws)
                return v = 0;
            return v = 1;
        }
    }

    int catMouseGame(vector<vector<int>>& graph) {
        this->g = graph;
        this->n = g.size();
        memset(f, -1, sizeof f);
        return dp(0, 1, 2);
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

> [!NOTE] **[LeetCode 3227. 字符串元音游戏](https://leetcode.cn/problems/vowels-game-in-a-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 较为简单 推导需要加速
> 
> - 分情况讨论 只要存在元音 (vowels) 则必胜

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    unordered_set<char> S = {'a', 'e', 'i', 'o', 'u'};
    
    bool doesAliceWin(string s) {
        int cnt = 0;
        for (auto c : s)
            if (S.count(c))
                cnt ++ ;
        return cnt;
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