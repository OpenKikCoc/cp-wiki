## 习题


> [!NOTE] **[LeetCode 1145. 二叉树着色游戏](https://leetcode-cn.com/problems/binary-tree-coloring-game/)**
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

> [!NOTE] **[LeetCode 1728. 猫和老鼠 II](https://leetcode-cn.com/problems/cat-and-mouse-ii/)** [TAG]
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

> [!NOTE] **[LeetCode 1927. 求和游戏](https://leetcode-cn.com/problems/sum-game/)**
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