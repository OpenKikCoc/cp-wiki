
## 定义

对于树上的每一个点，计算其所有子树中最大的子树节点数，这个值最小的点就是这棵树的重心。

（这里以及下文中的“子树”都是指无根树的子树，即包括“向上”的那棵子树，并且不包括整棵树自身。）

## 性质

以树的重心为根时，所有子树的大小都不超过整棵树大小的一半。

树中所有点到某个点的距离和中，到重心的距离和是最小的；如果有两个重心，那么到它们的距离和一样。

把两棵树通过一条边相连得到一棵新的树，那么新的树的重心在连接原来两棵树的重心的路径上。

在一棵树上添加或删除一个叶子，那么它的重心最多只移动一条边的距离。

## 求法

在 DFS 中计算每个子树的大小，记录“向下”的子树的最大大小，利用总点数 - 当前子树（这里的子树指有根树的子树）的大小得到“向上”的子树的大小，然后就可以依据定义找到重心了。


```cpp
// 这份代码默认节点编号从 1 开始，即 i ∈ [1,n]
int size[MAXN],  // 这个节点的“大小”（所有子树上节点数 + 该节点）
    weight[MAXN],  // 这个节点的“重量”
    centroid[2];   // 用于记录树的重心（存的是节点编号）
void GetCentroid(int cur, int fa) {  // cur 表示当前节点 (current)
    size[cur] = 1;
    weight[cur] = 0;
    for (int i = head[cur]; i != -1; i = e[i].nxt) {
        if (e[i].to != fa) {  // e[i].to 表示这条有向边所通向的节点。
            GetCentroid(e[i].to, cur);
            size[cur] += size[e[i].to];
            weight[cur] = max(weight[cur], size[e[i].to]);
        }
    }
    weight[cur] = max(weight[cur], n - size[cur]);
    if (weight[cur] <= n / 2) {  // 依照树的重心的定义统计
        centroid[centroid[0] != 0] = cur;
    }
}
```

## 参考

<http://fanhq666.blog.163.com/blog/static/81943426201172472943638/>([博客园转载](https://www.cnblogs.com/qlky/p/5781081.html)，[Internet Archive](https://web.archive.org/web/20181122041458/http://fanhq666.blog.163.com/blog/static/81943426201172472943638))

<https://blog.csdn.net/weixin_43810158/article/details/88391828>

<https://www.cnblogs.com/zinthos/p/3899075.html>

## 习题

- [POJ 1655 Balancing Art](http://poj.org/problem?id=1655)（模板题）
- [洛谷 P1364 医院设置](https://www.luogu.com.cn/problem/P1364)
- [CodeForces 1406C Link Cut Centroids](https://codeforces.com/contest/1406/problem/C)

> [!NOTE] **[AcWing 846. 树的重心](https://www.acwing.com/problem/content/848/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 5;
int n, res;
int node[maxn * 2], nxt[maxn * 2], head[maxn], tot;
bool vis[maxn];

void init() {
    res = maxn;
    tot = 0;
    for (int i = 0; i < maxn; ++i) head[i] = -1;
}

void add(int u, int v) {
    node[++tot] = v, nxt[tot] = head[u];
    head[u] = tot;
}

int dfs(int u) {
    vis[u] = true;
    int size = 0, sum = 1;  // 以u为根的节点数 包括u
    for (int i = head[u]; i != -1; i = nxt[i]) {
        int v = node[i];
        // 【其实对于树这种结构，这里标记u访问过即可
        // 因为只有u一个入口进来，即j!=u】
        if (vis[v]) continue;
        int s = dfs(v);
        size = max(size, s);
        sum += s;
    }
    size = max(size, n - sum);  // 以父亲节点为根 的树 的节点数
    res = min(res, size);
    return sum;
}

int main() {
    init();
    cin >> n;
    //
    int u, v;
    for (int i = 0; i < n; ++i) {
        cin >> u >> v;
        add(u, v);
        add(v, u);
    }
    dfs(1);
    cout << res << endl;
}
```

##### **Python**

```python
"""
​	图论题中，默认n个点，m条边。
- 用数组建立邻接表(基本上所有的树和图的结构都是这么存储的)：也就是 【链式前向星存储图】
  - 需要的变量有：h[N], ev[N], ne[N], idx；
  - h存的是n个链表的链表头（可以访问到的节点的链表的头部的idx）；
  - ev[i]存储某个点的值是多少，ne[i]存储某个点的next节点的下标/索引是多少（用下标关联起来的）；空节点的下标用-1来表示；idx表示下一个可以存储元素的位置索引。

"""


# #插入一条a指向b的边，其实就是在a所对应的的邻接表里插入一个节点b
def add_edge(a, b):
    global idx
    ev[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx += 1


# 返回以u为根的子树中的点的数量
def dfs(u):
    global ans
    st[u] = True
    sm, res = 1, 0
    i = h[u]
    while i != -1:  # 遍历x节点的子节点 
        j = ev[i]
        if not st[j]:
            s = dfs(j)  # 返回以j为根节点的子树的节点数
            res = max(res, s)
            sm += s
        i = ne[i]
    res = max(res, n - sm)
    ans = min(res, ans)
    return sm


if __name__ == '__main__':
    N = 100010

    # 邻接表的初始化（基本上所有的树和图的结构都是这么存储的）
    h = [-1] * N  # h存的是n个链表的链表头；让n个头结点全部初始化为-1，就实现了邻接表的初始化
    ev = [0] * 2 * N  # ev存的是每个节点的值
    ne = [0] * 2 * N  # ne存的是每个节点的next的值
    st = [False] * N  # 存储这个点 是否有被使用过；默认初始化全部都没有使用
    idx = 0

    ans = float('inf')
    n = int(input())

    # 加入n-1条边，加入两次
    # 由于题意是无向图：所以在建图建边的时候，每一条边都要建一条相反方向的边；
    for i in range(1, n):
        a, b = map(int, input().split())
        # 连接一条从a到b的边
        add_edge(a, b)
        # 连接一条从b到a的边
        add_edge(b, a)
    dfs(1)
    print(ans)
```

<!-- tabs:end -->
</details>

<br>

* * *