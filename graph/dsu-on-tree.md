
### 引入

启发式算法是什么呢？

启发式算法是基于人类的经验和直观感觉，对一些算法的优化。

给个例子？

最常见的就是并查集的按秩合并了，有带按秩合并的并查集中，合并的代码是这样的：

```cpp
void merge(int x, int y) {
    int xx = find(x), yy = find(y);
    if (size[xx] < size[yy]) swap(xx, yy);
    fa[yy] = xx;
    size[xx] += size[yy];
}
```

在这里，对于两个大小不一样的集合，我们将小的集合合并到大的集合中，而不是将大的集合合并到小的集合中。

为什么呢？这个集合的大小可以认为是集合的高度（在正常情况下），而我们将集合高度小的并到高度大的显然有助于我们找到父亲

让高度小的树成为高度较大的树的子树，这个优化可以称为启发式合并算法。

### 算法内容

树上启发式合并（dsu on tree）对于某些树上离线问题可以速度大于等于大部分算法且更易于理解和实现的算法。

考虑下面的问题：

> [!NOTE] **例题引入**
> 
> 给出一棵 $n$ 个节点以 $1$ 为根的树，节点 $u$ 的颜色为 $c_u$，现在对于每个结点 $u$ 询问 $u$ 子树里一共出现了多少种不同的颜色。
> 
> $n\le 2\times 10^5$。

![dsu-on-tree-1.png](./images/dsu-on-tree-1.svg)

对于这种问题解决方式大多是运用大量的数据结构（树套树等），如果可以离线，询问的量巨大，是不是有更简单的方法？

树上莫队！

不行，莫队带根号，我要 log

既然支持离线，考虑预处理后 $O(1)$ 输出答案。

直接暴力预处理的时间复杂度为 $O(n^2)$，即对每一个子节点进行一次遍历，每次遍历的复杂度显然与 $n$ 同阶，有 $n$ 个节点，故复杂度为 $O(n^2)$。

可以发现，每个节点的答案由其子树和其本身得到，考虑利用这个性质处理问题。

我们可以先预处理出每个节点子树的大小和它的重儿子，重儿子同树链剖分一样，是拥有节点最多子树的儿子，这个过程显然可以 $O(n)$ 完成

我们用 cnt[i]表示颜色 $i$ 的出现次数，ans[u]表示结点 $u$ 的答案。

遍历一个节点 $u$，我们按以下的步骤进行遍历：

1. 先遍历 $u$ 的轻（非重）儿子，并计算答案，但 **不保留遍历后它对 cnt 数组的影响**；
2. 遍历它的重儿子，**保留它对 cnt 数组的影响**；
3. 再次遍历 $u$ 的轻儿子的子树结点，加入这些结点的贡献，以得到 $u$ 的答案。

![dsu-on-tree-2.png](./images/dsu-on-tree-2.svg)

上图是一个例子。

这样，对于一个节点，我们遍历了一次重子树，两次非重子树，显然是最划算的。

通过执行这个过程，我们获得了这个节点所有子树的答案。

为什么不合并第一步和第三步呢？因为 cnt 数组不能重复使用，否则空间会太大，需要在 $O(n)$ 的空间内完成。

显然若一个节点 $u$ 被遍历了 $x$ 次，则其重儿子会被遍历 $x$ 次，轻儿子（如果有的话）会被遍历 $2x$ 次。

注意除了重儿子，每次遍历完 cnt 要清零。

### 复杂度

（对于不关心复杂度证明的，可以跳过不看）

我们像树链剖分一样定义重边和轻边（连向重儿子的为重边，其余为轻边）关于重儿子和重边的定义，可以见下图，对于一棵有 $n$ 个节点的树：

根节点到树上任意节点的轻边数不超过 $\log n$ 条。我们设根到该节点有 x 条轻边该节点的子树大小为 $y$，显然轻边连接的子节点的子树大小小于父亲的一半（若大于一半就不是轻边了），则 $y<n/2^x$，显然 $n>2^x$，所以 $x<\log n$。

又因为如果一个节点是其父亲的重儿子，则他的子树必定在他的兄弟之中最多，所以任意节点到根的路径上所有重边连接的父节点在计算答案是必定不会遍历到这个节点，所以一个节点的被遍历的次数等于他到根节点路径上的轻边树 $+1$（之所以要 $+1$ 是因为他本身要被遍历到），所以一个节点的被遍历次数 $=\log n+1$, 总时间复杂度则为 $O(n(\log n+1))=O(n\log n)$，输出答案花费 $O(m)$.

![dsu-on-tree-3.png](./images/dsu-on-tree-3.svg)

*图中标粗的即为重边，重边连向的子节点为重儿子*


```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 2e5 + 5;

int n;

// g[u]: 存储与 u 相邻的结点
vector<int> g[N];

// sz: 子树大小
// big: 重儿子
// col: 结点颜色
// L[u]: 结点 u 的 DFS 序
// R[u]: 结点 u 子树中结点的 DFS 序的最大值
// Node[i]: DFS 序为 i 的结点
// ans: 存答案
// cnt[i]: 颜色为 i 的结点个数
// totColor: 目前出现过的颜色个数
int sz[N], big[N], col[N], L[N], R[N], Node[N], totdfn;
int ans[N], cnt[N], totColor;

void add(int u) {
    if (cnt[col[u]] == 0) ++totColor;
    cnt[col[u]]++;
}
void del(int u) {
    cnt[col[u]]--;
    if (cnt[col[u]] == 0) --totColor;
}
int getAns() { return totColor; }

void dfs0(int u, int p) {
    L[u] = ++totdfn;
    Node[totdfn] = u;
    sz[u] = 1;
    for (int v : g[u])
        if (v != p) {
            dfs0(v, u);
            sz[u] += sz[v];
            if (!big[u] || sz[big[u]] < sz[v]) big[u] = v;
        }
    R[u] = totdfn;
}

void dfs1(int u, int p, bool keep) {
    // 计算轻儿子的答案
    for (int v : g[u])
        if (v != p && v != big[u]) { dfs1(v, u, false); }
    // 计算重儿子答案并保留计算过程中的数据（用于继承）
    if (big[u]) { dfs1(big[u], u, true); }
    for (int v : g[u])
        if (v != p && v != big[u]) {
            // 子树结点的 DFS 序构成一段连续区间，可以直接遍历
            for (int i = L[v]; i <= R[v]; i++) { add(Node[i]); }
        }
    add(u);
    ans[u] = getAns();
    if (keep == false) {
        for (int i = L[u]; i <= R[u]; i++) { del(Node[i]); }
    }
}

int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%d", &col[i]);
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d%d", &u, &v);
        g[u].push_back(v);
        g[v].push_back(u);
    }
    dfs0(1, 0);
    dfs1(1, 0, false);
    for (int i = 1; i <= n; i++) printf("%d%c", ans[i], " \n"[i == n]);
    return 0;
}
```

### 运用

1.  某些出题人设置的正解是 dsu on tree 的题

    如 [CF741D](http://codeforces.com/problemset/problem/741/D)。给一棵树，每个节点的权值是'a' 到'v' 的字母，每次询问要求在一个子树找一条路径，使该路径包含的字符排序后成为回文串。

    因为是排列后成为回文串，所以一个字符出现了两次相当于没出现，也就是说，这条路径满足 **最多有一个字符出现奇数次**。

    正常做法是对每一个节点 dfs，每到一个节点就强行枚举所有字母找到和他异或后结果为 1 的个数大于 1 的路径，再取最长值，这样是 $O(n^2\log n)$ 的，可以用 dsu on tree 优化到 $O(n\log^2n)$。关于具体做法，可以参考下面的扩展阅读

2.  可以用 dsu 乱搞的题

    可以水一些树套树的部分分（没有修改操作），还可以把树上莫队的 $O(n\sqrt{m})$ 吊着打

### 练习题

[CF600E Lomsat gelral](http://codeforces.com/problemset/problem/600/E)

题意翻译：树的节点有颜色，一种颜色占领了一个子树，当且仅当没有其他颜色在这个子树中出现得比它多。求占领每个子树的所有颜色之和。

[UOJ284 快乐游戏鸡](https://uoj.ac/problem/284)

### 参考资料/扩展阅读

[CF741D 作者介绍的 dsu on tree](http://codeforces.com/blog/entry/44351)

[这位作者的题解](http://codeforces.com/blog/entry/48871)

## 习题

> [!NOTE] **[LeetCode 2003. 每棵子树内缺失的最小基因值](https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **启发式合并 通用做法 【重复做增强熟练度】**
> 
> 1. 递归遍历，每次通过启发式合并每个节点下所有子树节点，启发式合并也是按 size 合并，即小的 size 暴力合并到大的 size 上。
> 
> 2. 对于每个节点，其答案不会小于所有子树答案的最大值，且不会超过以当前节点子树的大小加 1，此时可以暴力从最大值开始枚举答案，判断是否为第一个缺失的值。最后枚举的次数累加起来最多为 nn 次。
> 
> 3. 启发式合并的算法**无需基因值互不相同**，更加通用。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
private:
    vector<int> ans;
    vector<vector<int>> graph;
    vector<int> f;
    vector<unordered_set<int>> seen;  // set

    void merge(int x, int y) {
        int fx = f[x], fy = f[y];
        if (seen[fx].size() < seen[fy].size()) {
            for (int num : seen[fx])
                seen[fy].insert(num);
            f[x] = fy;
        } else {
            for (int num : seen[fy])
                seen[fx].insert(num);
            f[y] = fx;
        }
    }

    void dfs(int u) {
        int mex = 1;
        for (int v : graph[u]) {
            dfs(v);
            merge(u, v);
            mex = max(mex, ans[v]);
        }

        int fu = f[u];
        while (seen[fu].find(mex) != seen[fu].end())
            mex++;

        ans[u] = mex;
    }

public:
    vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums) {
        const int n = parents.size();
        graph.resize(n);
        for (int i = 1; i < n; i++)
            graph[parents[i]].push_back(i);

        f.resize(n);
        seen.resize(n);
        for (int i = 0; i < n; i++) {
            f[i] = i;
            seen[i].insert(nums[i]);
        }

        ans.resize(n);
        dfs(0);
        return ans;
    }
};
```

##### **C++ 遍历找1 不通用也很丑的做法**

1. 由于基因值互不相同，所以可以首先找到基因值为 1 的节点，确定当前节点到根节点的一条链，记为 ancestors（按顺序由底到根）。非链上的节点的答案显然都为 1。
2. 求出每个节点，其祖先节点第一次到达链的节点，记为 belong(i)。
3. 开始遍历 ancestors，并从 2 开始枚举值。
   - 对于当前值 m，如果没有对应的节点，则直接将尚未遍历的 ancestors 链中的节点的答案都置为 m，然后结束。
   - 如果有对应的节点，找到其 belong 值 target。
     - 如果 target 没有被遍历过，则从之前遍历到的位置开始遍历 ancestor 直到 target，并将遍历过程中的节点答案置为 mm。
     - 如果 target 已经被遍历过，则无需操作。
   - 如果 ancestors 已经被遍历完了，则结束。

```cpp
class Solution {
private:
    vector<int> ans;
    vector<vector<int>> graph;
    vector<int> belong;

    void mark(int u, int banned, int ancestor) {
        belong[u] = ancestor;

        for (int v : graph[u])
            if (v != banned)
                mark(v, banned, ancestor);
    }

public:
    vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums) {
        const int n = parents.size();
        graph.resize(n);
        for (int i = 1; i < n; i++)
            graph[parents[i]].push_back(i);

        vector<int> h(n + 2, -1);
        for (int i = 0; i < n; i++)
            if (nums[i] <= n)
                h[nums[i]] = i;

        ans.resize(n, 1);
        vector<int> ancestors;

        int p = h[1];
        while (p != -1) {
            ancestors.push_back(p);
            ans[p] = 0;
            p = parents[p];
        }

        belong.resize(n);
        for (int i = ancestors.size() - 1; i >= 0; i--)
            mark(ancestors[i], i == 0 ? -1 : ancestors[i - 1], ancestors[i]);

        for (int m = 2, i = 0; i < ancestors.size(); m++) {
            int target = -1;
            if (h[m] != -1)
                target = belong[h[m]];

            if (target != -1 && ans[target] > 0)
                continue;

            while (i < ancestors.size() && ancestors[i] != target) {
                ans[ancestors[i]] = m;
                i++;
            }
        }

        return ans;
    }
};
```

##### **C++ 暴力bitset TLE**

```cpp
// TLE 51 / 67
class Solution {
public:
    const static int N = 1e5 + 10;
    
    int h[N], e[N], ne[N], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    int n;
    vector<int> w, res;
    vector<bitset<N>> f; // ...不会初始化 vector<bitset<N>>
    
    void dfs(int u) {
        f[u].set();
        f[u].set(w[u], 0);
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            dfs(j);
            f[u] &= f[j];
        }
        res[u] = f[u]._Find_next(0);
        // cout << " u = " << u << " f[u] = " << f[u] << " res[u] = " << res[u] <<  endl;
    }
    
    vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums) {
        init();
        this->w = nums;
        this->n = parents.size();
        this->res = vector<int>(n);
        this->f = vector<bitset<N>>(n);
        
        for (int i = 0; i < n; ++ i )
            if (parents[i] != -1)
                add(parents[i], i);
        
        dfs(0);
        
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