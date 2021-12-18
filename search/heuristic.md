启发式搜索（英文：heuristic search）是一种改进的搜索算法。它在普通搜索算法的基础上引入了启发式函数，该函数的作用是基于已有的信息对搜索的每一个分支选择都做估价，进而选择分支。简单来说，启发式搜索就是对取和不取都做分析，从中选取更优解或删去无效解。


下面将简要介绍 $A*$ 算法。

## 简介

$A*$ 搜索算法（英文：$A*$ search algorithm，$A*$ 读作 A-star），简称 $A*$ 算法，是一种在图形平面上，对于有多个节点的路径求出最低通过成本的算法。它属于图遍历（英文：Graph traversal）和最佳优先搜索算法（英文：Best-first search），亦是 `BFS` 的改进。

定义起点 $s$，终点 $t$，从起点（初始状态）开始的距离函数 $g(x)$，到终点（最终状态）的距离函数 $h(x)$，$h^{\ast}(x)$ [^note1]，以及每个点的估价函数 $f(x)=g(x)+h(x)$。

  $A*$ 算法每次从优先队列中取出一个 $f$ 最小的元素，然后更新相邻的状态。

如果 $h\leq h*$，则 $A*$ 算法能找到最优解。

上述条件下，如果 $h$ 满足三角形不等式，则 $A*$ 算法不会将重复结点加入队列。

当 $h=0$ 时， $A*$ 算法变为 `DFS`；当 $h=0$ 并且边权为 $1$ 时变为 `BFS`。

## 例题

> [!NOTE] **[「NOIP2005 普及组」采药](https://www.luogu.com.cn/problem/P1048)**
> 
> 题目大意：有 $N$ 种物品和一个容量为 $W$ 的背包，每种物品有重量 $w_i$ 和价值 $v_i$ 两种属性，要求选若干个物品（每种物品只能选一次）放入背包，使背包中物品的总价值最大，且背包中物品的总重量不超过背包的容量。

> [!TIP] **解题思路**
> 
> 我们写一个估价函数 $f$，可以剪掉所有无效的 $0$ 枝条（就是剪去大量无用不选枝条）。
> 
> 估价函数 $f$ 的运行过程如下：
> 
> 我们在取的时候判断一下是不是超过了规定体积（可行性剪枝）；在不取的时候判断一下不取这个时，剩下的药所有的价值 + 现有的价值是否大于目前找到的最优解（最优性剪枝）。

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

> [!NOTE] **[八数码](https://www.luogu.com.cn/problem/P1379)**
> 
> 题目大意：在 $3\times 3$ 的棋盘上，摆有八个棋子，每个棋子上标有 $1$ 至 $8$ 的某一数字。棋盘中留有一个空格，空格用 $0$ 来表示。空格周围的棋子可以移到空格中，这样原来的位置就会变成空格。给出一种初始布局和目标布局（为了使题目简单，设目标状态如下），找到一种从初始布局到目标布局最少步骤的移动方法。
> 
> ```plain
> 123
> 804
> 765
> ```

> [!TIP] **解题思路**
> 
> $h$ 函数可以定义为，不在应该在的位置的数字个数。
> 
> 容易发现 $h$ 满足以上两个性质，此题可以使用 $A*$ 算法求解。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
int f(string state) {
    int res = 0;
    for (int i = 0; i < state.size(); i ++ )
        if (state[i] != 'x') {
            int t = state[i] - '1';
            res += abs(i / 3 - t / 3) + abs(i % 3 - t % 3);
        }
    return res;
}

string bfs(string start) {
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    char op[4] = {'u', 'r', 'd', 'l'};

    string end = "12345678x";
    unordered_map<string, int> dist;
    unordered_map<string, pair<string, char>> prev;
    priority_queue<pair<int, string>, vector<pair<int, string>>, greater<pair<int, string>>> heap;

    heap.push({f(start), start});
    dist[start] = 0;

    while (heap.size()) {
        auto t = heap.top();
        heap.pop();

        string state = t.second;

        if (state == end) break;

        int step = dist[state];
        int x, y;
        for (int i = 0; i < state.size(); i ++ )
            if (state[i] == 'x') {
                x = i / 3, y = i % 3;
                break;
            }
        string source = state;
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < 3 && b >= 0 && b < 3) {
                swap(state[x * 3 + y], state[a * 3 + b]);
                if (!dist.count(state) || dist[state] > step + 1) {
                    dist[state] = step + 1;
                    prev[state] = {source, op[i]};
                    heap.push({dist[state] + f(state), state});
                }
                swap(state[x * 3 + y], state[a * 3 + b]);
            }
        }
    }

    string res;
    while (end != start) {
        res += prev[end].second;
        end = prev[end].first;
    }
    reverse(res.begin(), res.end());
    return res;
}

int main() {
    string g, c, seq;
    while (cin >> c) {
        g += c;
        if (c != "x") seq += c;
    }

    int t = 0;
    for (int i = 0; i < seq.size(); i ++ )
        for (int j = i + 1; j < seq.size(); j ++ )
            if (seq[i] > seq[j])
                t ++ ;

    if (t % 2) puts("unsolvable");
    else cout << bfs(g) << endl;

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

> [!NOTE] **[k 短路](https://www.luogu.com.cn/problem/P2483)**
> 
> 按顺序求一个有向图上从结点 $s$ 到结点 $t$ 的所有路径最小的前任意多（不妨设为 $k$）个。

> [!TIP] **解题思路**
> 
> 很容易发现，这个问题很容易转化成用 $A*$ 算法解决问题的标准程式。
> 
> 初始状态为处于结点 $s$，最终状态为处于结点 $t$，距离函数为从 $s$ 到当前结点已经走过的距离，估价函数为从当前结点到结点 $t$ 至少要走过的距离，也就是当前结点到结点 $t$ 的最短路。
> 
> 就这样，我们在预处理的时候反向建图，计算出结点 $t$ 到所有点的最短路，然后将初始状态塞入优先队列，每次取出 $f(x)=g(x)+h(x)$ 最小的一项，计算出其所连结点的信息并将其也塞入队列。当你第 $k$ 次走到结点 $t$ 时，也就算出了结点 $s$ 到结点 $t$ 的 $k$ 短路。
> 由于设计的距离函数和估价函数，每个状态需要存储两个参数，当前结点 $x$ 和已经走过的距离 $v$。
> 
> 我们可以在此基础上加一点小优化：由于只需要求出第 $k$ 短路，所以当我们第 $k+1$ 次或以上走到该结点时，直接跳过该状态。因为前面的 $k$ 次走到这个点的时候肯定能因此构造出 $k$ 条路径，所以之后再加边更无必要。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
typedef pair<int, int> PII;
typedef pair<int, PII> PIII;

const int N = 1010, M = 200010;

int n, m, S, T, K;
int h[N], rh[N], e[M], w[M], ne[M], idx;
int dist[N], cnt[N];
bool st[N];

void add(int h[], int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

void dijkstra() {
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, T});

    memset(dist, 0x3f, sizeof dist);
    dist[T] = 0;

    while (heap.size()) {
        auto t = heap.top();
        heap.pop();

        int ver = t.y;
        if (st[ver]) continue;
        st[ver] = true;

        for (int i = rh[ver]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[ver] + w[i]) {
                dist[j] = dist[ver] + w[i];
                heap.push({dist[j], j});
            }
        }
    }
}

int astar() {
    priority_queue<PIII, vector<PIII>, greater<PIII>> heap;
    heap.push({dist[S], {0, S}});

    while (heap.size()) {
        auto t = heap.top();
        heap.pop();

        int ver = t.y.y, distance = t.y.x;
        cnt[ver] ++ ;
        if (cnt[T] == K) return distance;

        for (int i = h[ver]; ~i; i = ne[i]) {
            int j = e[i];
            if (cnt[j] < K)
                heap.push({distance + w[i] + dist[j], {distance + w[i], j}});
        }
    }

    return -1;
}

int main() {
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);
    memset(rh, -1, sizeof rh);

    for (int i = 0; i < m; i ++ ) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(h, a, b, c);
        add(rh, b, a, c);
    }
    scanf("%d%d%d", &S, &T, &K);
    if (S == T) K ++ ;

    dijkstra();
    printf("%d\n", astar());

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

> [!NOTE] **[AcWing 179. 八数码](https://www.acwing.com/problem/content/181/)**
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
#include <algorithm>
#include <cstring>
#include <iostream>
#include <queue>
#include <unordered_map>

using namespace std;

int f(string state) {
    int res = 0;
    for (int i = 0; i < state.size(); i++)
        if (state[i] != 'x') {
            int t = state[i] - '1';
            res += abs(i / 3 - t / 3) + abs(i % 3 - t % 3);
        }
    return res;
}

string bfs(string start) {
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    char op[4] = {'u', 'r', 'd', 'l'};

    string end = "12345678x";
    unordered_map<string, int> dist;
    unordered_map<string, pair<string, char>> prev;
    priority_queue<pair<int, string>, vector<pair<int, string>>,
                   greater<pair<int, string>>>
        heap;

    heap.push({f(start), start});
    dist[start] = 0;

    while (heap.size()) {
        auto t = heap.top();
        heap.pop();

        string state = t.second;

        if (state == end) break;

        int step = dist[state];
        int x, y;
        for (int i = 0; i < state.size(); i++)
            if (state[i] == 'x') {
                x = i / 3, y = i % 3;
                break;
            }
        string source = state;
        for (int i = 0; i < 4; i++) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < 3 && b >= 0 && b < 3) {
                swap(state[x * 3 + y], state[a * 3 + b]);
                if (!dist.count(state) || dist[state] > step + 1) {
                    dist[state] = step + 1;
                    prev[state] = {source, op[i]};
                    heap.push({dist[state] + f(state), state});
                }
                swap(state[x * 3 + y], state[a * 3 + b]);
            }
        }
    }

    string res;
    while (end != start) {
        res += prev[end].second;
        end = prev[end].first;
    }
    reverse(res.begin(), res.end());
    return res;
}

int main() {
    string g, c, seq;
    while (cin >> c) {
        g += c;
        if (c != "x") seq += c;
    }

    int t = 0;
    for (int i = 0; i < seq.size(); i++)
        for (int j = i + 1; j < seq.size(); j++)
            if (seq[i] > seq[j]) t++;

    if (t % 2)
        puts("unsolvable");
    else
        cout << bfs(g) << endl;

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

## 习题


## 参考资料与注释

[^note1]: 此处的 h 意为 heuristic。详见 [启发式搜索 - 维基百科](https://zh.wikipedia.org/wiki/%E5%90%AF%E5%8F%91%E5%BC%8F%E6%90%9C%E7%B4%A2) 和 [  $A*$ search algorithm - Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm#Bounded_relaxation) 的 Bounded relaxation 一节。
