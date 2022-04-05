## 基础

### 基环树的一半处理思路

1. 找到环

2. 将环断开，让它成为一棵树，对于断开边的两个端点分别进行树形dp。当然，需要注意一些关于这两个点的限制条件。

3. 将信息整合，更新答案

### 一般建图

每个 `下标 x` 是唯一的，而这个 `下标 x` 会与 `其他某个下标 y` 有一定关系。

由出边唯一的树根等等性质，**往往由 y 向 x 连接一条边**。

## 习题

> [!NOTE] **[AcWing 358. 岛屿](https://www.acwing.com/problem/content/360/)**
> 
> 题意: 
> 
> **找基环树的直径**。
> 
> 本题是一个基环森林，可以从一棵基环树跳到其他基环树上，所以基环树之间不会影响，要求的就是每个基环树的直径总和。

> [!TIP] **思路**
> 
> - 基环树直径有两种可能：
> 
> 1. 直径是环上某个节点子树的直径
> 
> 2. 算出环上节点到子树最大距离 $d[i]$ ，然后结果就是环上节点的组合。也就是 $max(d[i]+d[j]+max(len,len−dis(i,j))$ 。
> 
>    后者这个就是一个环形DP问题了，可以用单调队列解决。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e6 + 10, M = N << 1;

int n;

// ------------------- graph -------------------
int h[N], e[M], w[M], ne[M], idx;
void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

// ------------------- circle -------------------
int fu[N], fw[N], q[N]; // 找环过程中记录路径
int cir[N], ed[N], cnt; // 记录环 ==> TODO
bool st[N], ins[N];     // 找环
LL s[N], d[M], sum[M];  // 前缀和  破环成链需要两倍长度数组

void dfs_c(int u, int from) {
    st[u] = ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i]) {
        if (i == (from ^ 1))
            continue;   // 反向边跳过
        int j = e[i];
        fu[j] = u, fw[j] = w[i];    // 记录上一个位置 以及边长
        if (!st[j])
            dfs_c(j, i);
        else if (ins[j]) {          // 已在栈中 说明有环
            // 环个数+1
            cnt ++ ;
            ed[cnt] = ed[cnt - 1];
            // 逆序找所有的环上的点 统计前缀和
            // ==> why 前缀和？ 因为【前面的分析过程 优化】
            //                         for-loop这里必须先用k 再改k
            for (LL k = fu[j], tot = fw[j]; ; tot += fw[k], k = fu[k]) {
                s[k] = tot;
                cir[ ++ ed[cnt]] = k;	// ATTENTION trick
                if (k == j)
                    break;
            }
        }
    }
    ins[u] = false;
}

// ------------------- dp -------------------
LL ans;
LL dfs_d(int u) {
    st[u] = true;
    LL d1 = 0, d2 = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (st[j])
            continue;
        LL dist = dfs_d(j) + w[i];
        if (dist >= d1)
            d2 = d1, d1 = dist;
        else if (dist > d2)
            d2 = dist;
    }
    ans = max(ans, d1 + d2);
    return d1;
}

void init() {
    // graph
    memset(h, -1, sizeof h);
    idx = 0;
    
    // circle
    memset(st, 0, sizeof st);
    memset(ins, 0, sizeof ins);
    // ...
}
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    
    init();
    cin >> n;
    for (int i = 1; i <= n; ++ i ) {
        int j, c;
        cin >> j >> c;
        add(i, j, c), add(j, i, c);
    }
    for (int i = 1; i <= n; ++ i )
        if (!st[i])
            dfs_c(i, -1);   // 找环
            
    memset(st, 0, sizeof st);   // reset
    for (int i = 1; i <= ed[cnt]; ++ i )
        st[cir[i]] = true;      // 标记所有环上的点 
    
    LL res = 0; // 整个基环树森林的总直径
    for (int i = 1; i <= cnt; ++ i ) {
        ans = 0;    //reset 计算单个基环树内部的直径
        int sz = 0;
        for (int j = ed[i - 1] + 1; j <= ed[i]; ++ j ) {
            int k = cir[j];
            d[sz] = dfs_d(k);   // 该点对应树的最远距离
            sum[sz] = s[k];     // 拷贝前缀和
            sz ++ ;
        }
        // pre-sum
        for (int j = 0; j < sz; ++ j )
            d[sz + j] = d[j], sum[sz + j] = sum[j] + sum[sz - 1];
        // 单调队列  公式转化 ==> d[x] - s[x]
        int hh = 0, tt = -1;
        for (int j = 0; j < sz * 2; ++ j ) {
            if (hh <= tt && j - q[hh] >= sz)
                hh ++ ;
            if (hh <= tt)
                ans = max(ans, d[j] + sum[j] + d[q[hh]] - sum[q[hh]]);
            while (hh <= tt && d[q[tt]] - sum[q[tt]] <= d[j] - sum[j])
                tt -- ;
            q[ ++ tt] = j;
        }
        res += ans;
    }
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

> [!NOTE] **[AcWing 1080. 骑士](https://www.acwing.com/problem/content/1082/)**
> 
> 题意: 
> 
> 类似只有上司的舞会，但是基环树

> [!TIP] **思路**
> 
> 考虑根据某个**基环树的树根**（必在环上）开始断开其出边
> 
> 随后分情况讨论
> 
> - 建边思维
> 
>   把 x 所讨厌的人 y 设置为 x 的父节点（父节点向子节点连边
> 
>   **这样每个人都有且仅有一条出边【重要】**
> 
>   只有跟节点可以成环 而环一定包含根节点

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e6 + 10, INF = 1e9;

int n;
int h[N], e[N], rm[N], w[N], ne[N], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

LL f1[N][2], f2[N][2];
bool st[N], ins[N];
LL res;

// ap为不能选的点
void dfs_f(int u, int ap, LL f[][2]) {
    // get f[u][0]
    for (int i = h[u]; ~i; i = ne[i]) {
        if (rm[i])
            continue;
        int j = e[i];
        dfs_f(j, ap, f);
        f[u][0] += max(f[j][0], f[j][1]);
    }
    // get f[u][1]
    f[u][1] = -INF;
    if (u != ap) {
        f[u][1] = w[u];
        for (int i = h[u]; ~i; i = ne[i]) {
            if (rm[i])
                continue;
            int j = e[i];
            f[u][1] += f[j][0];
        }
    }
}

void dfs_c(int u, int from) {
    st[u] = ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!st[j])
            dfs_c(j, i);
        else if (ins[j]) {  // 找到环
            rm[i] = 1;      // 标记 删除这条边
            // f[j][0] 表示以u为根的子树 不选u的所有方案的最大值
            // f[j][1] 表示..............选u的.............
            //
            // 计算得 f[j][0] 可以直接用
            // 使用 f[j][1] 时需特判下 u 不能选的方案
            // 最后取max即可
            // 
            // ATTENTION 思考: 为什么是用j不是用u   ==> 与建边方向相关
            dfs_f(j, -1, f1);
            dfs_f(j, u, f2);
            res += max(f1[j][0], f2[j][1]);
        }
    }
    ins[u] = false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    
    init();
    cin >> n;
    for (int i = 1; i <= n; ++ i ) {
        int v, j;
        cin >> v >> j;
        add(j, i);  // 建立一条边 TODO方向why
        w[i] = v;
    }
    for (int i = 1; i <= n; ++ i )
        if (!st[i])
            dfs_c(i, -1);
    
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

> [!NOTE] **[AcWing 359. 创世纪](https://www.acwing.com/activity/content/problem/content/3188/)**
> 
> 题意: 
> 
> 要求被选择的元素，必然至少有一个未被选的指向它

> [!TIP] **思路**
> 
> 推导
> 
> 思考建边和建图

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const static int N = 1e6 + 10, INF = 1e8;

int n;
int h[N], e[N], rm[N], ne[N], idx;
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int f1[N][2], f2[N][2];
bool st[N], ins[N];
int res;

// TODO 一堆推导
void dfs_f(int u, int ap, int f[][2]) {
    for (int i = h[u]; ~i; i = ne[i]) {
        if (rm[i])
            continue;
        int j = e[i];
        dfs_f(j, ap, f);
        f[u][0] += max(f[j][0], f[j][1]);
    }
    if (u == ap)  // 必选
        f[u][1] = f[u][0] + 1, f[u][0] = -INF;
    else {        // 可选可不选
        f[u][1] = -INF;
        for (int i = h[u]; ~i; i = ne[i]) {
            if (rm[i])
                continue;
            int j = e[i];
            //                   枚举j，除了j之外其他都可选可不选      + 第j个不选 + 1(u本身) 
            // =              所有的可选可不选 -  第j个点可选可不选。   + 第j个不选 + 1(u本身)
            f[u][1] = max(f[u][1], f[u][0] - max(f[j][0], f[j][1]) + f[j][0] + 1);
        }
    }
}

void dfs_c(int u, int from) {
    st[u] = ins[u] = true;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!st[j])
            dfs_c(j, i);
        else if (ins[j]) {
            rm[i] = 1;
            dfs_f(j, -1, f1); // 可选可不选 u
            dfs_f(j, u, f2);  // 必选 u
            res += max(max(f1[j][0], f1[j][1]), f2[j][0]);
        }
    }
    ins[u] = false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    
    init();
    cin >> n;
    for (int i = 1; i <= n; ++ i ) {
        int j;
        cin >> j;
        add(j, i);
    }
    
    for (int i = 1; i <= n; ++ i )
        if (!st[i])
            dfs_c(i, -1);
            
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