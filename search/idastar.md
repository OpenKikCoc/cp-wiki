前置知识：[$A*$](search/astar.md) 算法、[迭代加深搜索](search/iterative.md)。

本页面将简要介绍 $IDA*$算法。

## 简介

$IDA*$为采用了迭代加深算法的 $A*$ 算法。由于 $IDA*$改成了深度优先的方式，相对于 $A*$ 算法，它的优势如下：

1. 不需要判重，不需要排序；
2. 空间需求减少。

## 优点

1. 空间开销小：每个深度下实际上是一个深度优先搜索，不过深度有限制，使用 DFS 可以减小空间消耗；
2. 利于深度剪枝。

## 缺点

重复搜索：即使前后两次搜索相差微小，回溯过程中每次深度变大都要再次从头搜索。

## 例题

> [!NOTE] **[埃及分数](https://loj.ac/p/10022)**
> 
> 在古埃及，人们使用单位分数的和（即 $\frac{1}{a}$，$a\in\mathbb{N}^*$）表示一切有理数。例如，$\frac{2}{3}=\frac{1}{2}+\frac{1}{6}$，但不允许 $\frac{2}{3}=\frac{1}{3}+\frac{1}{3}$，因为在加数中不允许有相同的。
> 
> 对于一个分数 $\frac{a}{b}$，表示方法有很多种，其中加数少的比加数多的好，如果加数个数相同，则最小的分数越大越好。例如，$\frac{19}{45}=\frac{1}{5}+\frac{1}{6}+\frac{1}{18}$ 是最优方案。
> 
> 输入整数 $a,b$（$0<a<b<500$），试编程计算最佳表达式。
> 
> 样例输入：
> 
> ```text
> 495 499
> ```  
> 
> 样例输出：
> 
> ```text
> Case 1: 495/499 = 1/2 + 1/5 + 1/6 + 1/8 + 1/3992 + 1/14970
> ```

> [!TIP] 解题思路
> 
> 这道题目理论上可以用回溯法求解，但是解答树会非常“恐怖”——不仅深度没有明显的上界，而且加数的选择理论上也是无限的。换句话说，如果用宽度优先遍历，连一层都扩展不完，因为每一层都是 **无限大** 的。
> 
> 解决方案是采用迭代加深搜索：从小到大枚举深度上限 $\textit{maxd}$，每次执行只考虑深度不超过 $\textit{maxd}$ 的节点。这样，只要解的深度优先，则一定可以在有限时间内枚举到。
> 
> 深度上限 $\mathit{maxd}$ 还可以用来 **剪枝**。按照分母递增的顺序来进行扩展，如果扩展到 i 层时，前 $i$ 个分数之和为 $\frac{c}{d}$，而第 $i$ 个分数为 $\frac{1}{e}$，则接下来至少还需要 $\frac{\frac{a}{b}-\frac{c}{d}}{\frac{1}{e}}$ 个分数，总和才能达到 $\frac{a}{b}$。例如，当前搜索到 $\frac{19}{45}=\frac{1}{5}+\frac{1}{100}+\cdots$，则后面的分数每个最大为 $\frac{1}{101}$，至少需要 $\frac{\frac{19}{45}-\frac{1}{5}}{\frac{1}{101}}=23$ 项总和才能达到 $\frac{19}{45}$，因此前 $22$ 次迭代是根本不会考虑这棵子树的。这里的关键在于：可以估计至少还要多少步才能出解。
> 
> 注意，这里使用 **至少** 一词表示估计都是乐观的。形式化地，设深度上限为 $\textit{maxd}$，当前结点 $n$ 的深度为 $g(n)$，乐观估价函数为 $h(n)$，则当 $g(n)+h(n)>\textit{maxd}$ 时应该剪枝。这样的算法就是 $IDA*$。当然，在实战中不需要严格地在代码里写出 $g(n)$ 和 $h(n)$，只需要像刚才那样设计出乐观估价函数，想清楚在什么情况下不可能在当前的深度限制下出解即可。
> 
> 如果可以设计出一个乐观估价函数，预测从当前结点至少还需要扩展几层结点才有可能得到解，则迭代加深搜索变成了 $IDA*$算法。

> [!TIP] 示例代码

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

- [UVa1343 旋转游戏](https://www.luogu.com.cn/problem/UVA1343)



> [!NOTE] **[AcWing 180. 排书](https://www.acwing.com/problem/content/182/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 转化为图论 估价函数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 1. 单词移动 对应两种方法(把某段移前等价于把它前面某段移后)
// 2. 每次操作断开三个连接并新建三个连接 ==> 每次操作最多修正三个连接
const int N = 15;

int n;
int q[N];
int w[5][N];

int f() {
    int cnt = 0;
    for (int i = 0; i + 1 < n; i ++ )
        if (q[i + 1] != q[i] + 1)
            cnt ++ ;
    return (cnt + 2) / 3;
}

bool check() {
    for (int i = 0; i + 1 < n; i ++ )
        if (q[i + 1] != q[i] + 1)
            return false;
    return true;
}

bool dfs(int depth, int max_depth) {
    if (depth + f() > max_depth) return false;
    if (check()) return true;

    for (int len = 1; len <= n; len ++ )
        for (int l = 0; l + len - 1 < n; l ++ ) {
            int r = l + len - 1;
            for (int k = r + 1; k < n; k ++ ) {
                memcpy(w[depth], q, sizeof q);
                int x, y;
                for (x = r + 1, y = l; x <= k; x ++, y ++ ) q[y] = w[depth][x];
                for (x = l; x <= r; x ++, y ++ ) q[y] = w[depth][x];
                if (dfs(depth + 1, max_depth)) return true;
                memcpy(q, w[depth], sizeof q);
            }
        }

    return false;
}

int main() {
    int T;
    cin >> T;
    while (T -- ) {
        cin >> n;
        for (int i = 0; i < n; i ++ ) cin >> q[i];

        int depth = 0;
        while (depth < 5 && !dfs(0, depth)) depth ++ ;
        if (depth >= 5) puts("5 or more");
        else cout << depth << endl;
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

> [!NOTE] **[AcWing 181. 回转游戏](https://www.acwing.com/problem/content/183/)**
> 
> 题意: TODO

> [!TIP]
>
> **估价函数**
> 
> 统计中间8个方格中出现次数最多的数出现了多少次，记为 k 次。
> 
> 每次操作会从中间8个方格中移出一个数，再移入一个数，所以最多会减少一个不同的数。
因此估价函数可以设为 8−k。
> 
> **剪枝**
> 
> 记录上一次的操作，本次操作避免枚举上一次的逆操作。
> 
> **如何保证答案的字典序最小？**
> 
> 由于最短操作步数是一定的，因此每一步枚举时先枚举字典序小的操作即可。
> 
> **时间复杂度**
> 
> 假设答案最少需要 k 步，每次需要枚举 7 种不同操作（除了上一步的逆操作），因此最坏情况下需要枚举 7k 种方案。但加入启发函数后，实际枚举到的状态数很少。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 24;

int op[8][7] = {
    {0, 2, 6, 11, 15, 20, 22},
    {1, 3, 8, 12, 17, 21, 23},
    {10, 9, 8, 7, 6, 5, 4},
    {19, 18, 17, 16, 15, 14, 13},
    {23, 21, 17, 12, 8, 3, 1},
    {22, 20, 15, 11, 6, 2, 0},
    {13, 14, 15, 16, 17, 18, 19},
    {4, 5, 6, 7, 8, 9, 10}
};

int oppsite[8] = {5, 4, 7, 6, 1, 0, 3, 2};
int center[8] = {6, 7, 8, 11, 12, 15, 16, 17};

int q[N];
int path[100];

int f() {
    static int sum[4];
    memset(sum, 0, sizeof sum);
    for (int i = 0; i < 8; i ++ ) sum[q[center[i]]] ++ ;

    int maxv = 0;
    for (int i = 1; i <= 3; i ++ ) maxv = max(maxv, sum[i]);

    return 8 - maxv;
}

void operate(int x) {
    int t = q[op[x][0]];
    for (int i = 0; i < 6; i ++ ) q[op[x][i]] = q[op[x][i + 1]];
    q[op[x][6]] = t;
}

bool dfs(int depth, int max_depth, int last) {
    if (depth + f() > max_depth) return false;
    if (f() == 0) return true;

    for (int i = 0; i < 8; i ++ )
        if (last != oppsite[i]) {
            operate(i);
            path[depth] = i;
            if (dfs(depth + 1, max_depth, i)) return true;
            operate(oppsite[i]);
        }

    return false;
}

int main() {
    while (cin >> q[0], q[0]) {
        for (int i = 1; i < 24; i ++ ) cin >> q[i];

        int depth = 0;
        while (!dfs(0, depth, -1)) depth ++ ;

        if (!depth) printf("No moves needed");
        else {
            for (int i = 0; i < depth; i ++ ) printf("%c", 'A' + path[i]);
        }
        printf("\n%d\n", q[6]);
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

> [!NOTE] **[LeetCode 488. 祖玛游戏](https://leetcode-cn.com/problems/zuma-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂模拟 **记录优雅写法**
> 
> 剪枝优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    unordered_map<string, int> f;
    unordered_map<char, int> cnt;
    int res = 6;

    string get() {
        string ret;
        for (auto [x, c] : cnt) ret += to_string(c);
        return ret;
    }

    // 至少还需要多少次操作 剪枝 常数优化
    int h(string board) {
        sort(board.begin(), board.end());
        int ret = 0;
        // not ++ i 
        // or use ++ i and i = j - 1
        for (int i = 0; i < board.size();) {
            int j = i + 1;
            while (j < board.size() && board[j] == board[i]) ++ j ;
            if (j - i + cnt[board[i]] < 3) return 6;
            if (j - i <= 2) ret += 3 - (j - i);
            i = j; 
        }
        return ret;
    }

    void dfs(string board, string hand) {
        if (f[board + ' ' + hand] + h(board) >= res) return ;
        for (auto [x, c] : cnt) if (c) {
            -- cnt[x];
            for (int i = 0; i <= board.size(); ++ i ) {
                auto r = clear_up(board.substr(0, i) + x + board.substr(i));
                auto s = r + ' ' + get();
                if (f.count(s) == 0 || f[s] > f[board + ' ' + hand] + 1) {
                    f[s] = f[board + ' ' + hand] + 1;
                    if (r.empty()) res = min(res, f[s]);
                    dfs(r, get());
                }
            }
            ++ cnt[x];
        }
    }

    int findMinStep(string board, string hand) {
        for (auto c : hand) ++ cnt[c] ;
        f[board] = 0;
        dfs(board, get());
        if (res == 6) res = -1;
        return res;
    }

    string clear_up(string s) {
        bool is_changed = true;
        while (is_changed) {
            is_changed = false;
            for (int i = 0; i < s.size(); ++ i ) {
                int j = i + 1;
                while (j < s.size() && s[j] == s[i]) ++ j ;
                if (j - i >= 3) {
                    s = s.substr(0, i) + s.substr(j);
                    is_changed = true;
                    break;
                }
            }
        }
        return s;
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