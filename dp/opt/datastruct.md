# 数据结构优化DP

## 习题

### 线段树优化 / 树状数组优化

> [!NOTE] **[LeetCode 2407. 最长递增子序列 II](https://leetcode.cn/problems/longest-increasing-subsequence-ii/)**
> 
> 题意: 
> 
> 严格最长上升子序列，但要求相邻两个数字之间的值之差不超过 $k$

> [!TIP] **思路**
> 
> $f[i] = max{f[j]} + 1$
> 
> 其中 $max(1, i - k) <= j < i$
> 
> 显然涉及到区间查询，以及后续的单点修改
> 
> 线段树 / 树状数组【树状数组新姿势】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 线段树**

```cpp
const static int N = 1e5 + 10;

int w[N];

struct Node {
    int l, r;
    int maxv;
} tr[N << 2];

void pushup(int u) {
    tr[u].maxv = max(tr[u << 1].maxv, tr[u << 1 | 1].maxv);
}

void eval(Node & t, int v) {
    t.maxv = v;
}

void build(int u, int l, int r) {
    if (l == r)
        tr[u] = {l, r, w[l]};
    else {
        tr[u] = {l, r, 0};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

void modify(int u, int l, int r, int k) {
    if (tr[u].l >= l && tr[u].r <= r)
        eval(tr[u], k);
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)
            modify(u << 1, l, r, k);
        if (r > mid)
            modify(u << 1 | 1, l, r, k);
        pushup(u);
    }
}

int query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)
        return tr[u].maxv;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        
        int x = 0;
        if (l <= mid)
            x = max(x, query(u << 1, l, r));
        if (r > mid)
            x = max(x, query(u << 1 | 1, l, r));
        return x;
    }
}

class Solution {
public:
    int lengthOfLIS(vector<int>& nums, int k) {
        memset(w, 0, sizeof w);
        build(1, 1, N - 1);
        
        for (auto x : nums) {
            int t = query(1, x - k, x - 1);
            modify(1, x, x, t + 1);
        }
        return query(1, 1, N - 1);
    }
};
```

##### **C++ 树状数组**

```cpp
const static int N = 1e5 + 10;

int w[N];   // 必须有两个数组
int tr[N];

int lowbit(int x) {
    return x & -x;
}
void modify(int x, int y) {
    w[x] = y;
    for (int i = x; i < N; i += lowbit(i))
        tr[i] = max(tr[i], y);
}
int query(int l, int r) {
    int res = 0;
    for (; l <= r;) {
        res = max(res, w[r]);   // ATTENTION
        for ( -- r ; r >= l + lowbit(r); r -= lowbit(r))
            res = max(res, tr[r]);
    }
    return res;
}

class Solution {
public:
    int lengthOfLIS(vector<int>& nums, int k) {
        memset(w, 0, sizeof w), memset(tr, 0, sizeof tr);
        for (auto y : nums) {
            int t = query(max(1, y - k), y - 1);
            modify(y, t + 1);
        }
        return query(1, N - 1);
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