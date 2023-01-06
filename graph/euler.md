本页面将简要介绍欧拉图的概念、实现和应用。

## 定义

通过图中所有边恰好一次且行遍所有顶点的通路称为欧拉通路。

通过图中所有边恰好一次且行遍所有顶点的回路称为欧拉回路。

具有欧拉回路的无向图或有向图称为欧拉图。

具有欧拉通路但不具有欧拉回路的无向图或有向图称为半欧拉图。

有向图也可以有类似的定义。

非形式化地讲，欧拉图就是从任意一个点开始都可以一笔画完整个图，半欧拉图必须从某个点开始才能一笔画完整个图。

## 性质

欧拉图中所有顶点的度数都是偶数。

若 $G$ 是欧拉图，则它为若干个环的并，且每条边被包含在奇数个环内。

## 判别法

对于无向图 $G$，$G$ 是欧拉图当且仅当 $G$ 是连通的且没有奇度顶点。

对于无向图 $G$，$G$ 是半欧拉图当且仅当 $G$ 是连通的且 $G$ 中恰有 $0$ 个或 $2$ 个奇度顶点。

对于有向图 $G$，$G$ 是欧拉图当且仅当 $G$ 的所有顶点属于同一个强连通分量且每个顶点的入度和出度相同。

对于有向图 $G$，$G$ 是半欧拉图当且仅当

- 如果将 $G$ 中的所有有向边退化为无向边时，那么 $G$ 的所有顶点属于同一个连通分量。
- 最多只有一个顶点的出度与入度差为 $1$。
- 最多只有一个顶点的入度与出度差为 $1$。
- 所有其他顶点的入度和出度相同。

## 求欧拉回路或欧拉路

### Fleury 算法

也称避桥法，是一个偏暴力的算法。

算法流程为每次选择下一条边的时候优先选择不是桥的边。

一个广泛使用但是错误的实现方式是先 Tarjan 预处理桥边，然后再 DFS 避免走桥。但是由于走图过程中边会被删去，一些非桥边会变为桥边导致错误。最简单的实现方法是每次删除一条边之后暴力跑一遍 Tarjan 找桥，时间复杂度是 $\Theta(m(n+m))=\Theta(m^2)$。复杂的实现方法要用到动态图等，实用价值不高。

### Hierholzer 算法

也称逐步插入回路法。

算法流程为从一条回路开始，每次任取一条目前回路中的点，将其替换为一条简单回路，以此寻找到一条欧拉回路。如果从路开始的话，就可以寻找到一条欧拉路。

Hierholzer 算法的暴力实现如下：

$$
\begin{array}{ll}
1 &  \textbf{Input. } \text{The edges of the graph } e , \text{ where each element in } e \text{ is } (u, v) \\
2 &  \textbf{Output. } \text{The vertex of the Euler Road of the input graph}.\\
3 &  \textbf{Method. } \\
4 &  \textbf{Function } \text{Hierholzer } (v) \\
5 &  \qquad circle \gets \text{Find a Circle in } e \text{ Begin with } v \\
6 &  \qquad \textbf{if } circle=\varnothing \\
7 &  \qquad\qquad \textbf{return } v \\
8 &  \qquad e \gets e-circle \\
9 &  \qquad \textbf{for} \text{ each } v \in circle \\
10&  \qquad\qquad v \gets \text{Hierholzer}(v) \\
11&  \qquad \textbf{return } circle \\
12&  \textbf{Endfunction}\\
13&  \textbf{return } \text{Hierholzer}(\text{any vertex})
\end{array}
$$

这个算法的时间复杂度约为 $O(nm+m^2)$。实际上还有复杂度更低的实现方法，就是将找回路的 DFS 和 Hierholzer 算法的递归合并，边找回路边使用 Hierholzer 算法。

如果需要输出字典序最小的欧拉路或欧拉回路的话，因为需要将边排序，时间复杂度是 $\Theta(n+m\log m)$（计数排序或者基数排序可以优化至 $\Theta(n+m)$）。如果不需要排序，时间复杂度是 $\Theta(n+m)$。

### 应用

有向欧拉图可用于计算机译码。

设有 $m$ 个字母，希望构造一个有 $m^n$ 个扇形的圆盘，每个圆盘上放一个字母，使得圆盘上每连续 $n$ 位对应长为 $n$ 的符号串。转动一周（$m^n$ 次）后得到由 $m$ 个字母产生的长度为 $n$ 的 $m^n$ 个各不相同的符号串。

![](images/euler1.svg)

构造如下有向欧拉图：

设 $S = \{a_1, a_2, \cdots, a_m\}$，构造 $D=<V, E>$，如下：

$V = \{a_{i_1}a_{i_2}\cdots a_{i_{n-1}} |a_i \in S, 1 \leq i \leq n - 1 \}$

$E = \{a_{j_1}a_{j_2}\cdots a_{j_{n-1}}|a_j \in S, 1 \leq j \leq n\}$

规定 $D$ 中顶点与边的关联关系如下：

顶点 $a_{i_1}a_{i_2}\cdots a_{i_{n-1}}$ 引出 $m$ 条边：$a_{i_1}a_{i_2}\cdots a_{i_{n-1}}a_r, r=1, 2, \cdots, m$。

边 $a_{j_1}a_{j_2}\cdots a_{j_{n-1}}$ 引入顶点 $a_{j_2}a_{j_3}\cdots a_{j_{n}}$。

![](images/euler2.svg)

这样的 $D$ 是连通的，且每个顶点入度等于出度（均等于 $m$），所以 $D$ 是有向欧拉图。

任求 $D$ 中一条欧拉回路 $C$，取 $C$ 中各边的最后一个字母，按各边在 $C$ 中的顺序排成圆形放在圆盘上即可。

## 例题

> [!NOTE] **[洛谷 P2731 骑马修栅栏](https://www.luogu.com.cn/problem/P2731)**
> 
> 给定一张有 500 个顶点的无向图，求这张图的一条欧拉路或欧拉回路。如果有多组解，输出最小的那一组。
> 
> 在本题中，欧拉路或欧拉回路不需要经过所有顶点。
> 
> 边的数量 m 满足 $1\leq m \leq 1024$。

> [!TIP] **解题思路**
> 
> 用 Fleury 算法解决本题的时候只需要再贪心就好，不过由于复杂度不对，还是换 Hierholzer 算法吧。
> 
> 保存答案可以使用 `stack<int>`，因为如果找的不是回路的话必须将那一部分放在最后。
> 
> 注意，不能使用邻接矩阵存图，否则时间复杂度会退化为 $\Theta(nm)$。由于需要将边排序，建议使用前向星或者 vector 存图。示例代码使用 vector。



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

- [洛谷 P1341 无序字母对](https://www.luogu.com.cn/problem/P1341)

- [洛谷 P2731 骑马修栅栏](https://www.luogu.com.cn/problem/P2731)

### 一般欧拉

> [!NOTE] **[LeetCode 2097. 合法重新排列数对](https://leetcode-cn.com/problems/valid-arrangement-of-pairs/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分析和建图较为简单直接
> 
> **主要在于欧拉路算法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准 删边**

```cpp
class Solution {
public:
    // 1e9 显然需要离散化 长度不超过2e5 ==> 之前写法TLE 考虑直接 unordered_map<int, vector<int>> g;
    //                                  以及直接一个 deg 数组而非两个 din dout
    // 
    // 求欧拉路径可以使用 Fleury 算法，从起点开始深度优先遍历整个图，最后记录当前点作为路径，然后反向输出路径。
    // 如果所有点的入度等于出度，则任意点都可以当做起点。否则，需要从出度减入度等于 1 的点开始遍历。
    
    unordered_map<int, vector<int>> g;
    unordered_map<int, int> deg;
    vector<vector<int>> res;
    
    void dfs(int u) {
        auto & es = g[u];
        while (!es.empty()) {
            int v = es.back();
            es.pop_back();  // 删边 ==> 比设置 st[i] = true 效果好太多
            dfs(v);
            res.push_back(vector<int>{u, v});
        }
    }
    
    vector<vector<int>> validArrangement(vector<vector<int>>& pairs) {
        for (auto & p : pairs) {
            g[p[0]].push_back(p[1]);
            deg[p[0]] -- , deg[p[1]] ++ ;
        }
        
        int start = -1;
        for (auto & [k, v] : deg)
            if (v == -1) {
                start = k;
                break;
            }
        if (start == -1)    // 有环 任意起点
            start = deg.begin()->first;
        
        dfs(start);
        reverse(res.begin(), res.end());
        return res;
    }
};
```

##### **C++ 最初版 简单修改TLE**

```cpp
// 33 / 39 个通过测试用例 TLE
class Solution {
public:
    // 1e9 显然需要离散化 长度不超过2e5
    // 
    // 求欧拉路径可以使用 Fleury 算法，从起点开始深度优先遍历整个图，最后记录当前点作为路径，然后反向输出路径。
    // 如果所有点的入度等于出度，则任意点都可以当做起点。否则，需要从出度减入度等于 1 的点开始遍历。
    const static int N = 2e5 + 10;
    
    int n, m;
    int h[N], e[N], w[N], ne[N], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b, int c) {
        e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    vector<vector<int>> ps;
    unordered_map<int, int> ids;
    
    int din[N], dout[N];
    bool st[N];
    vector<vector<int>> res;
    
    void dfs(int u, int fa) {
        for (int i = h[u]; ~i; i = ne[i])
            if (!st[i]) {
                st[i] = true;   // ATTENTION 删边
                int v = e[i], idx = w[i];
                dfs(v, u);
                res.push_back(ps[idx]);
            }
    }
    
    vector<vector<int>> validArrangement(vector<vector<int>>& pairs) {
        {
            // 1. 离散化
            ps = pairs;
            vector<int> t;
            for (auto & p : ps)
                t.push_back(p[0]), t.push_back(p[1]);
            sort(t.begin(), t.end());
            t.erase(unique(t.begin(), t.end()), t.end());
            
            m = t.size();
            for (int i = 0; i < m; ++ i )
                ids[t[i]] = i;
            n = ids.size();
        }
        {
            // 2. 建图 ==> ATTENTION 不需要topo
            init();
            memset(din, 0, sizeof din);
            int sz = ps.size();
            for (int i = 0; i < sz; ++ i ) {
                auto & p = ps[i];
                add(ids[p[0]], ids[p[1]], i);
                dout[ids[p[0]]] ++ , din[ids[p[1]]] ++ ;
            }
        }
        
        int start = -1;
        for (int i = 0; i < n; ++ i )
            if (din[i] + 1 == dout[i]) {
                start = i;
                break;
            }
        if (start == -1)    // 有环
            start = 0;
        
        memset(st, 0, sizeof st);
        dfs(start, -1);
        reverse(res.begin(), res.end());
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

> [!NOTE] **[AcWing 1123. 铲雪车](https://www.acwing.com/problem/content/1125/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本题有向图 且所有的出度都等于入度
> 
> 故必然存在欧拉回路
> 
> 则最小时间就是所有（单向）边长 * 2

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

int main() {
    double x1, y1, x2, y2;
    cin >> x1 >> y1;
    
    double sum = 0;
    while (cin >> x1 >> y1 >> x2 >> y2) {
        double dx = x1 - x2;
        double dy = y1 - y2;
        sum += sqrt(dx * dx + dy * dy) * 2;
    }
    int minutes = round(sum / 1000 / 20 * 60);
    int hours = minutes / 60;
    minutes %= 60;
    
    printf("%d:%02d\n", hours, minutes);
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

> [!NOTE] **[AcWing 1184. 欧拉回路](https://www.acwing.com/problem/content/1186/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 对有无向图G和有向图H：
> 
> 1. 图G存在欧拉路径与欧拉回路的充要条件分别是：
> 
>    - 欧拉路径： 图中所有奇度点的数量为0或2。
> 
>    - 欧拉回路： 图中所有点的度数都是偶数。
> 
> 2. 图H存在欧拉路径和欧拉回路的充要条件分别是：
> 
>    - 欧拉路径： 所有点的入度等于出度 或者 存在一点出度比入度大1(起点)，
>              一点入度比出度大1(终点)，其他点的入度均等于出度。
>    - 欧拉回路：所有点的入度等于出度。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 100010, M = 400010;

int type;
int n, m;
int h[N], e[M], ne[M], idx;
bool used[M];
//
int ans[M], cnt;
int din[N], dout[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u) {
    for (int & i = h[u]; ~i;) {
        if (used[i]) {
            i = ne[i];
            continue;
        }
        used[i] = true;
        if (type == 1) used[i ^ 1] = true;  // 无向图标记反向边
        
        // 获取对应边序号
        int t;
        if (type == 1) {
            t = i / 2 + 1;
            if (i & 1) t = -t;
        } else t = i + 1;
        
        int j = e[i];
        i = ne[i];
        dfs(j);
        
        ans[ ++ cnt] = t;
    }
}

int main() {
    memset(h, -1, sizeof h);
    
    cin >> type;
    cin >> n >> m;
    for (int i = 0; i < m; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b);
        if (type == 1) add(b, a);
        din[b] ++ , dout[a] ++ ;
    }
    
    if (type == 1) {
        // 无向图 所有度数都是偶数
        for (int i = 1; i <= n; ++ i )
            if (din[i] + dout[i] & 1) {
                cout << "NO" << endl;
                return 0;
            }
    } else {
        // 有向图 入度等于出度
        for (int i = 1; i <= n; ++ i )
            if (din[i] != dout[i]) {
                cout << "NO" << endl;
                return 0;
            }
    }
    
    for (int i = 1; i <= n; ++ i )
        if (h[i] != -1) {
            // 任一点起始
            dfs(i);
            break;
        }
    if (cnt < m) {
        cout << "NO" << endl;
        return 0;
    }
    cout << "YES" << endl;
    for (int i = cnt; i; -- i ) cout << ans[i] << ' ';
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

> [!NOTE] **[AcWing 1124. 骑马修栅栏](https://www.acwing.com/problem/content/1126/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 无向图的欧拉路径
> 
> 只有0或2个点的度数是奇数
> 
> 输出字典序最小的欧拉路径

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 510, M = 1050;

int n = 500, m;
// 方便不用排序直接用邻接矩阵
int g[N][N];
int ans[M], cnt;
int d[N];

// 按从小到大搜
void dfs(int u) {
    for (int i = 1; i <= n; ++ i )
        if (g[u][i]) {
            g[u][i] -- , g[i][u] -- ;
            dfs(i);
        }
    ans[++ cnt] = u;
}

int main() {
    cin >> m;
    
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        g[a][b] ++ , g[b][a] ++ ;
        d[a] ++ , d[b] ++ ;
    }
    int start = 1;
    // 不一定从一号点开始走 故先找到第一个度数不为0的点
    while (!d[start]) start ++ ;
    // 找到度数为奇数的点
    for (int i = 1; i <= 500; ++ i )
        if (d[i] & 1) {
            start = i;
            break;
        }
    dfs(start);
    
    for (int i = cnt; i; -- i ) cout << ans[i] << endl;
    
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

> [!NOTE] **[LeetCode 332. 重新安排行程](https://leetcode-cn.com/problems/reconstruct-itinerary/)**
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
    unordered_map<string, multiset<string>> g;
    vector<string> ans;

    vector<string> findItinerary(vector<vector<string>>& tickets) {
        for (auto& e: tickets) g[e[0]].insert(e[1]);
        dfs("JFK");
        reverse(ans.begin(), ans.end());
        return ans;
    }

    void dfs(string u) {
        while (g[u].size()) {
            auto ver = *g[u].begin();
            g[u].erase(g[u].begin());
            dfs(ver);
        }
        ans.push_back(u);
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

### 建图

> [!NOTE] **[AcWing 1185. 单词游戏](https://www.acwing.com/problem/content/1187/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 原来自己见到这类题 思考的是把单词作为一个点，相同字母可以连一条边
> 
> 而本题：每个字母是一个点，单词本身是一条边
> 
> 显然所有单词都需要出现一次，转化为 ==> 所有边都恰好被访问一次 ==> 欧拉路径
> 
> 1. 除了起点终点之外 其他点入度=出度
> 2. 连通

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 30, M = 100010;

int n;
int din[N], dout[N], p[N];
bool st[N]; // 每个字母是否被用过

int find(int x) {
    if (p[x] != x) return p[x] = find(p[x]);
    return x;
}

int main() {
    char str[1010];
    
    int t;
    scanf("%d", &t);
    while (t -- ) {
        memset(din, 0, sizeof din);
        memset(dout, 0, sizeof dout);
        memset(st, 0, sizeof st);
        for (int i = 0; i < 26; ++ i ) p[i] = i;
        
        scanf("%d", &n);
        for (int i = 0; i < n; ++ i ) {
            scanf("%s", str);
            int len = strlen(str);
            int a = str[0] - 'a', b = str[len - 1] - 'a';
            st[a] = st[b] = true;
            dout[a] ++ , din[b] ++ ;
            p[find(a)] = find(b);
        }
        // 判断条件1
        int start = 0, end = 0;
        bool success = true;
        for (int i = 0; i < 26; ++ i )
            if (din[i] != dout[i]) {
                if (din[i] == dout[i] + 1) ++ end;
                else if (din[i] + 1 == dout[i]) ++ start;
                else {
                    success = false;
                    break;
                }
            }
        if (success && !(!start && !end || start == 1 && end == 1)) success = false;
        
        int rep = -1;
        for (int i = 0; i < 26; ++ i )
            if (st[i]) {
                if (rep == -1) rep = find(i);
                else if (rep != find(i)) {
                    success = false;
                    break;
                }
            }
        if (success) puts("Ordering is possible.");
        else puts("The door cannot be opened.");
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

### 复杂

> [!NOTE] **[LeetCode 753. 破解保险箱](https://leetcode-cn.com/problems/cracking-the-safe/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> de Bruijn 序列
> 
> 在最短串内枚举所有 n 位 k 进制数
> 
> 转化为欧拉回路问题
> 
> 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 如何在一个最短的串内枚举所有的n位k进制数排列
    // de Bruijn 序列
    // 转化为欧拉回路问题
    unordered_set<string> S;
    string res;
    int k;
    void dfs(string u) {
        for (int i = 0; i < k; ++ i ) {
            auto v = u + to_string(i);
            if (!S.count(v)) {
                S.insert(v);
                dfs(v.substr(1));
                res += to_string(i);
            }
        }
    }

    string crackSafe(int n, int k) {
        this->k = k;
        string start(n - 1, '0');
        dfs(start);
        return res + start;
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

> [!NOTE] **[Codeforces Sereja and the Arrangement of Numbers](http://codeforces.com/problemset/problem/367/C)**
> 
> 题意: 
> 
> 给出 $m$ 个不同的数 $q_i$，并且每个数都有个对应的费用 $w_i$ 。
> 
> 现在要在 $m$ 个数中选择一些数（每个数可使用任意多次，但是代价只会计算一次），用这些数组成一个长度为 $n$ 的数列，并且满足任意两个不同种类的数，至少有一次相邻。
> 
> 问最大费用。

> [!TIP] **思路**
> 
> 在一副图中，如果 `全部点的度数是偶数` / `只有2个点是奇数` ，则能一笔画。
> 
> - 图的点数 $k$ 为奇数的时候，那么每个点的度数都是偶数点，可以直接一笔画，答案为 $1+i*(i-1)/2$;
> 
> - $k$ 为偶数的时候，所有的点是奇数点，我们保留 2 个点是奇数点，将其他的点改为偶数点，就可以一笔画。 $1+i*(i-1)/2+i/2-1$ .

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Sereja and the Arrangement of Numbers
// Contest: Codeforces - Codeforces Round #215 (Div. 1)
// URL: https://codeforces.com/problemset/problem/367/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10;

int n, m, k;
int a[N], b[N];
LL f[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    {
        for (LL i = 1; i < N; ++i)
            if (i & 1)
                f[i] = 1ll + i * (i - 1) / 2;
            else
                //                           (i / 2) - 1 是需要加的最少的边
                f[i] = 1ll + i * (i - 1) / 2 + i / 2 - 1;
    }

    cin >> n >> m;
    for (int i = 1; i <= m; ++i)
        cin >> a[i] >> b[i];
    sort(b + 1, b + 1 + m, greater<int>());

    k = m;
    while (f[k] > n)
        k--;

    LL res = 0;
    for (int i = 1; i <= k; ++i)
        res += b[i];
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
