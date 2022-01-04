## 主席树

主席树全称是可持久化权值线段树，参见 [知乎讨论](https://www.zhihu.com/question/59195374)。

> [!WARNING] **关于函数式线段树**
> 
> **函数式线段树** 是指使用函数式编程思想的线段树。在函数式编程思想中，将计算机运算视为数学函数，并避免可改变的状态或变量。不难发现，函数式线段树是 [完全可持久化](persistent/#fully-persistent) 的

面对眼前的区间第 $k$ 小问题，你该何从下手？

一种可行的方案是：使用主席树。  
主席树的主要思想就是：保存每次插入操作时的历史版本，以便查询区间第 $k$ 小。

怎么保存呢？简单暴力一点，每次开一棵线段树呗。  
那空间还不爆掉？

那么我们分析一下，发现每次修改操作修改的点的个数是一样的。  
（例如下图，修改了 $[1,8]$ 中对应权值为 1 的结点，红色的点即为更改的点）  
![](./images/persistent-seg.png)

只更改了 $O(\log{n})$ 个结点，形成一条链，也就是说每次更改的结点数 = 树的高度。  
注意主席树不能使用堆式存储法，就是说不能用 $x\times 2$，$x\times 2+1$ 来表示左右儿子，而是应该动态开点，并保存每个节点的左右儿子编号。  
所以我们只要在记录左右儿子的基础上存一下插入每个数的时候的根节点就可以持久化辣。

我们把问题简化一下：每次求 $[1,r]$ 区间内的 $k$ 小值。  
怎么做呢？只需要找到插入 r 时的根节点版本，然后用普通权值线段树（有的叫键值线段树/值域线段树）做就行了。

那么这个相信大家很简单都能理解，把问题扩展到原问题——求 $[l,r]$ 区间 $k$ 小值。  
这里我们再联系另外一个知识理解：**前缀和**。  
这个小东西巧妙运用了区间减法的性质，通过预处理从而达到 $O(1)$ 回答每个询问。

那么我们阔以发现，主席树统计的信息也满足这个性质。  
所以……如果需要得到 $[l,r]$ 的统计信息，只需要用 $[1,r]$ 的信息减去 $[1,l - 1]$ 的信息就行了。

那么至此，该问题解决！（完结撒花）

关于空间问题，我们分析一下：由于我们是动态开点的，所以一棵线段树只会出现 $2n-1$ 个结点。  
然后，有 $n$ 次修改，每次至多增加 $\lceil\log_2{n}\rceil+1$ 个结点。因此，最坏情况下 $n$ 次修改后的结点总数会达到 $2n-1+n(\lceil\log_2{n}\rceil+1)$。
此题的 $n \leq 10^5$，单次修改至多增加 $\lceil\log_2{10^5}\rceil+1 = 18$ 个结点，故 $n$ 次修改后的结点总数为 $2\times 10^5-1+18\times 10^5$，忽略掉 $-1$，大概就是 $20\times 10^5$。

最后给一个忠告：千万不要吝啬空间（大多数题目中空间限制都较为宽松，因此一般不用担心空间超限的问题）！保守一点，直接上个 $2^5\times 10^5$，接近原空间的两倍（即 `n << 5`）。

代码：

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;
const int maxn = 1e5;  // 数据范围
int tot, n, m;
int sum[(maxn << 5) + 10], rt[maxn + 10], ls[(maxn << 5) + 10],
    rs[(maxn << 5) + 10];
int a[maxn + 10], ind[maxn + 10], len;
inline int getid(const int &val) {  // 离散化
    return lower_bound(ind + 1, ind + len + 1, val) - ind;
}
int build(int l, int r) {  // 建树
    int root = ++tot;
    if (l == r) return root;
    int mid = l + r >> 1;
    ls[root] = build(l, mid);
    rs[root] = build(mid + 1, r);
    return root;  // 返回该子树的根节点
}
int update(int k, int l, int r, int root) {  // 插入操作
    int dir = ++tot;
    ls[dir] = ls[root], rs[dir] = rs[root], sum[dir] = sum[root] + 1;
    if (l == r) return dir;
    int mid = l + r >> 1;
    if (k <= mid)
        ls[dir] = update(k, l, mid, ls[dir]);
    else
        rs[dir] = update(k, mid + 1, r, rs[dir]);
    return dir;
}
int query(int u, int v, int l, int r, int k) {  // 查询操作
    int mid = l + r >> 1,
        x = sum[ls[v]] - sum[ls[u]];  // 通过区间减法得到左儿子的信息
    if (l == r) return l;
    if (k <= x)  // 说明在左儿子中
        return query(ls[u], ls[v], l, mid, k);
    else  // 说明在右儿子中
        return query(rs[u], rs[v], mid + 1, r, k - x);
}
inline void init() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; ++i) scanf("%d", a + i);
    memcpy(ind, a, sizeof ind);
    sort(ind + 1, ind + n + 1);
    len = unique(ind + 1, ind + n + 1) - ind - 1;
    rt[0] = build(1, len);
    for (int i = 1; i <= n; ++i) rt[i] = update(getid(a[i]), 1, len, rt[i - 1]);
}
int l, r, k;
inline void work() {
    while (m--) {
        scanf("%d%d%d", &l, &r, &k);
        printf("%d\n", ind[query(rt[l - 1], rt[r], 1, len, k)]);  // 回答询问
    }
}
int main() {
    init();
    work();
    return 0;
}
```

## 参考

<https://en.wikipedia.org/wiki/Persistent_data_structure>

<https://www.cnblogs.com/zinthos/p/3899565.html>


## 习题

> [!NOTE] **[AcWing 255. 第K小数](https://www.acwing.com/problem/content/257/)**
> 
> 题意: TODO

> [!TIP] **插播**
> 
> https://www.acwing.com/solution/content/4224/
> 
> 指路学习笔记：可持久化线段树（主席树）：静态 + 动态
> 
> 建立一颗权值线段树，每个点存储的信息为该值域区间存在的数的个数。

> [!TIP] **思路**
>
> 可持久化线段树 = 主席树
>
> 线段树 每次操作最多修改 4*logn个区间
> 同时不能再类似堆的方式存 而是指针方式
> ===> l r 表示左右子节点的下标 cnt表示当前区间中一共有多少个数
>
> 可持久化线段树难以进行区间修改操作 个别懒标记永久化
>
> 本题：
> 特点1.静态问题 过程中无变化
>     对于静态问题：  1.1 划分树 求区间最小       nlogn
>                     1.2 树套树 线段树套平衡树   nlog^2 n 【支持修改】
>                     1.3 可持久化线段树          nlogn 不支持修改 完整包含划分树
>
> 1. 离散化：数值比较大 10^9
> 2. 在数值上建线段树 维护每个数值区间中一共有多少个数
>
> 【线段树二叉树本身是可以做二分的】
>
> 考虑只有R限制 显然是可持久化线段树
> 加上L限制 上题是存在性 本题是求一个值故做法不同 但对于线段树结构是是一样的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 100010, M = 10010;

int n, m;
int a[N];
vector<int> nums;
struct Node{
    int l, r;
    int cnt;
}tr[N * 4 + N * 17];    // logn = 17

int root[N], idx;

int find(int x) {
    return lower_bound(nums.begin(), nums.end(), x) - nums.begin();
}

// 要返回idx
int build(int l, int r) {
    int p = ++ idx;
    if (l == r) return p;
    else {
        int mid = l + r >> 1;
        tr[p].l = build(l, mid), tr[p].r = build(mid + 1, r);
        return p;
    }
}

int insert(int p, int l, int r, int x) {
    int q = ++ idx;
    tr[q] = tr[p];
    if (l == r) {
        tr[q].cnt ++ ;
        return q;
    } else {
        int mid = l + r >> 1;
        if (x <= mid) tr[q].l = insert(tr[p].l, l, mid, x);
        else tr[q].r = insert(tr[p].r, mid + 1, r, x);
        tr[q].cnt = tr[tr[q].l].cnt + tr[tr[q].r].cnt;
        return q;
    }
}

int query(int q, int p, int l, int r, int k) {
    if (l == r) return r;
    else {
        int cnt = tr[tr[q].l].cnt - tr[tr[p].l].cnt;
        int mid = l + r >> 1;
        if (k <= cnt) return query(tr[q].l, tr[p].l, l, mid, k);
        else return query(tr[q].r, tr[p].r, mid + 1, r, k - cnt);
    }
}

int main() {
    scanf("%d%d", &n, &m);
    
    for (int i = 1; i <= n; ++ i ) {
        scanf("%d", &a[i]);
        nums.push_back(a[i]);
    }
    sort(nums.begin(), nums.end());
    nums.erase(unique(nums.begin(), nums.end()), nums.end());
    
    root[0] = build(0, nums.size() - 1);
    
    for (int i = 1; i <= n; ++ i )
        root[i] = insert(root[i - 1], 0, nums.size() - 1, find(a[i]));
    
    while (m -- ) {
        int l, r, k;
        scanf("%d%d%d", &l, &r, &k);
        printf("%d\n", nums[query(root[r], root[l - 1], 0, nums.size() - 1, k)]);
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

## 习题

> [!NOTE] **[LeetCode 1157. 子数组中占绝大多数的元素](https://leetcode-cn.com/problems/online-majority-element-in-subarray/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 权值线段树

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MajorityChecker {
private:
    unordered_map<int, vector<int>> pos;
    vector<int> tree;
    vector<int> a;
    
public:
    MajorityChecker(vector<int>& arr): a(arr) {
        for (int i = 0; i < arr.size(); ++i) {
            pos[arr[i]].push_back(i);
        }
        tree = vector<int>(arr.size() * 4, -1);
        build_tree(1, 0, arr.size() - 1);
    }
    
    void build_tree(int tree_pos, int l, int r) {
        if (l == r) {
            tree[tree_pos] = a[l];
            return;
        }
        int mid = (l + r) >> 1;
        build_tree(tree_pos * 2, l, mid);
        build_tree(tree_pos * 2 + 1, mid + 1, r);
        if (tree[tree_pos * 2] != -1 && get_occurrence(tree[tree_pos * 2], l, r) * 2 > r - l + 1) {
            tree[tree_pos] = tree[tree_pos * 2];
        }
        else if (tree[tree_pos * 2 + 1] != -1 && get_occurrence(tree[tree_pos * 2 + 1], l, r) * 2 > r - l + 1) {
            tree[tree_pos] = tree[tree_pos * 2 + 1];
        }
    }
    
    pair<int, int> query(int tree_pos, int l, int r, int queryl, int queryr) {
        if (l > queryr || r < queryl) {
            return make_pair(-1, -1);
        }
        if (queryl <= l && r <= queryr) {
            if (tree[tree_pos] == -1) {
                return make_pair(-1, -1);
            }
            int occ = get_occurrence(tree[tree_pos], queryl, queryr);
            if (occ * 2 > queryr - queryl + 1) {
                return make_pair(tree[tree_pos], occ);
            }
            else {
                return make_pair(-1, -1);
            }
        }
        int mid = (l + r) >> 1;
        pair<int, int> res_l = query(tree_pos * 2, l, mid, queryl, queryr);
        if (res_l.first > -1) {
            return res_l;
        }
        pair<int, int> res_r = query(tree_pos * 2 + 1, mid + 1, r, queryl, queryr);
        if (res_r.first > -1) {
            return res_r;
        }
        return make_pair(-1, -1);
    }
    
    int get_occurrence(int num, int l, int r) {
        auto iter = pos.find(num);
        if (iter == pos.end()) {
            return 0;
        }
        const auto& vec = iter->second;
        auto iter_l = lower_bound(vec.begin(), vec.end(), l);
        if (iter_l == vec.end()) {
            return 0;
        }
        auto iter_r = upper_bound(vec.begin(), vec.end(), r);
        return iter_r - iter_l;
    }
    
    int query(int left, int right, int threshold) {
        pair<int, int> ans = query(1, 0, a.size() - 1, left, right);
        if (ans.second >= threshold) {
            return ans.first;
        }
        return -1;
    }
};

/**
 * Your MajorityChecker object will be instantiated and called as such:
 * MajorityChecker* obj = new MajorityChecker(arr);
 * int param_1 = obj->query(left,right,threshold);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *