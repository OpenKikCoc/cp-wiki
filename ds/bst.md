## 二叉搜索树简介

二叉搜索树是一种二叉树的树形数据结构，其定义如下：

1. 空树是二叉搜索树。

2. 若二叉搜索树的左子树不为空，则其左子树上所有点的附加权值均小于其根节点的值。

3. 若二叉搜索树的右子树不为空，则其右子树上所有点的附加权值均大于其根节点的值。

4. 二叉搜索树的左右子树均为二叉搜索树。

二叉搜索树上的基本操作所花费的时间与这棵树的高度成正比。对于一个有 $n$ 个结点的二叉搜索树中，这些操作的最优时间复杂度为 $O(\log n)$，最坏为 $O(n)$。随机构造这样一棵二叉搜索树的期望高度为 $O(\log n)$。

## 基本操作

在接下来的代码块中，我们约定 $n$ 为结点个数，$h$ 为高度，`val[x]` 为结点 $x$ 处存的数值，`cnt[x]` 为结点 $x$ 存的值所出现的次数，`lc[x]` 和 `rc[x]` 分别为结点 $x$ 的左子结点和右子结点。

### 遍历二叉搜索树

由二叉搜索树的递归定义可得，二叉搜索树的中序遍历权值的序列为非降的序列。时间复杂度为 $O(n)$。

遍历一棵二叉搜索树的代码如下：

```cpp
void print(int o) {
    // 遍历以 o 为根节点的二叉搜索树
    if (!o) return;  // 遇到空树，返回
    print(lc[o]);    // 递归遍历左子树
    for (int i = 1; i <= cnt[o]; i++) printf("%d\n", val[o]);  // 输出根节点信息
    print(rc[o]);  // 递归遍历右子树
}
```

### 查找最小/最大值

由二叉搜索树的性质可得，二叉搜索树上的最小值为二叉搜索树左链的顶点，最大值为二叉搜索树右链的顶点。时间复杂度为 $O(h)$。

findmin 和 findmax 函数分别返回最小值和最大值所对应的结点编号 $o$，用 `val[o]` 可以获得相应的最小/最大值。

```cpp
int findmin(int o) {
    if (!lc[o]) return o;
    return findmin(lc[o]);  // 一直向左儿子跳
}
int findmax(int o) {
    if (!rc[o]) return o;
    return findmax(rc[o]);  // 一直向右儿子跳
}
```

### 插入一个元素

定义 `insert(o,v)` 为在以 $o$ 为根节点的二叉搜索树中插入一个值为 $v$ 的新节点。

分类讨论如下：

若 $o$ 为空，直接返回一个值为 $v$ 的新节点。

若 $o$ 的权值等于 $v$，该节点的附加域该值出现的次数自增 $1$。

若 $o$ 的权值大于 $v$，在 $o$ 的左子树中插入权值为 $v$ 的节点。

若 $o$ 的权值小于 $v$，在 $o$ 的右子树中插入权值为 $v$ 的节点。

时间复杂度为 $O(h)$。

```cpp
void insert(int& o, int v) {
    if (!o) {
        val[o = ++sum] = v;
        cnt[o] = siz[o] = 1;
        lc[o] = rc[o] = 0;
        return;
    }
    siz[o]++;
    if (val[o] == v) {
        cnt[o]++;
        return;
    }
    if (val[o] > v) insert(lc[o], v);
    if (val[o] < v) insert(rc[o], v);
}
```

### 删除一个元素

定义 `del(o,v)` 为在以 $o$ 为根节点的二叉搜索树中删除一个值为 $v$ 的节点。

先在二叉搜索树中找到权值为 $v$ 的节点，分类讨论如下：

若该节点的附加 $\textit{cnt}$ 大于 $1$，只需要减少 $\textit{cnt}$。

若该节点的附加 $\textit{cnt}$ 为 $1$：

若 $o$ 为叶子节点，直接删除该节点即可。

若 $o$ 为链节点，即只有一个儿子的节点，返回这个儿子。

若 $o$ 有两个非空子节点，一般是用它左子树的最大值或右子树的最小值代替它，然后将它删除。

时间复杂度 $O(h)$。

```cpp
int deletemin(int& o) {
    if (!lc[o]) {
        int u = o;
        o = rc[o];
        return u;
    } else {
        int u = deletemin(lc[o]);
        siz[o] -= cnt[u];
        return u;
    }
}
void del(int& o, int v) {
    // 注意 o 有可能会被修改
    siz[o]--;
    if (val[o] == v) {
        if (cnt[o] > 1) {
            cnt[o]--;
            return;
        }
        if (lc[o] && rc[o]) o = deletemin(rc[o]);
        // 这里以找右子树的最小值为例
        else
            o = lc[o] + rc[o];
        return;
    }
    if (val[o] > v) del(lc[o], v);
    if (val[o] < v) del(rc[o], v);
}
```

### 求元素的排名

排名定义为将数组元素排序后第一个相同元素之前的数的个数加一。

维护每个根节点的子树大小 $\textit{siz}$。查找一个元素的排名，首先从根节点跳到这个元素，若向右跳，答案加上左儿子节点个数加当前节点重复的数个数，最后答案加上终点的左儿子子树大小加一。

时间复杂度 $O(h)$。

```cpp
int queryrnk(int o, int v) {
    if (val[o] == v) return siz[lc[o]] + 1;
    if (val[o] > v) return queryrnk(lc[o], v);
    if (val[o] < v) return queryrnk(rc[o], v) + siz[lc[o]] + cnt[o];
}
```

### 查找排名为 k 的元素

在一棵子树中，根节点的排名取决于其左子树的大小。

若其左子树的大小大于等于 $k$，则该元素在左子树中；

若其左子树的大小在区间 $[k-\textit{cnt},k-1]$($\textit{cnt}$ 为当前结点的值的出现次数）中，则该元素为子树的根节点；

若其左子树的大小小于 $k-\textit{cnt}$，则该元素在右子树中。

时间复杂度 $O(h)$。

```cpp
int querykth(int o, int k) {
    if (siz[lc[o]] >= k) return querykth(lc[o], k);
    if (siz[lc[o]] < k - cnt[o])
        return querykth(rc[o], k - siz[lc[o]] - cnt[o]);
    return val[o];
    // 如要找排名为 k 的元素所对应的结点，直接 return o 即可
}
```

## 习题

### 平衡树

> [!NOTE] **[AcWing 253. 普通平衡树](https://www.acwing.com/problem/content/255/)**
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
#include<bits/stdc++.h>
using namespace std;

const int N = 100010, INF = 1e8;

int n;
struct Node {
    int l, r;
    int key, val;
    int cnt, size;  // 数量 树大小
}tr[N];

int root, idx;

void pushup(int p) {
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
}

int get_node(int key) {
    tr[ ++ idx].key = key;
    tr[idx].val = rand();
    tr[idx].cnt = tr[idx].size = 1;
    return idx;
}

// 右旋
void zig(int & p) {
    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
    pushup(tr[p].r), pushup(p);
}

// 左旋
void zag(int & p) {
    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
    pushup(tr[p].l), pushup(p);
}

void build() {
    // 哨兵节点 -inf inf
    get_node(-INF), get_node(INF);
    // 1节点-inf 2节点inf
    root = 1, tr[1].r = 2;
    pushup(root);
    
    if (tr[1].val < tr[2].val) zag(root);
}

// 注意引用 以及修改p的lr
void insert(int & p, int key) {
    if (!p) p = get_node(key);
    else if (tr[p].key == key) tr[p].cnt ++ ;
    else if (tr[p].key > key) {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val) zig(p);
    } else {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val) zag(p);
    }
    pushup(p);
}

void remove(int & p, int key) {
    if (!p) return ;    // 不存在 不用管
    if (tr[p].key == key) {
        if (tr[p].cnt > 1) tr[p].cnt -- ;
        else if (tr[p].l || tr[p].r) {
            // 非叶子节点 要删除当前节点
            // 右空或 左val>右val 右旋
            if (!tr[p].r || tr[tr[p].l].val > tr[tr[p].r].val) {
                // zag之后p引用指向的节点会变为tr[p].r 故递归处理
                zig(p);
                remove(tr[p].r, key);
            } else {
                zag(p);
                remove(tr[p].l, key);
            }
        } else p = 0;
    } else if (tr[p].key > key) remove(tr[p].l, key);
    else remove(tr[p].r, key);
    pushup(p);
}

// 通过数值找排名
int get_rank_by_key(int p, int key) {
    if (!p) return 0;   // 本题中不会发生此情况
    if (tr[p].key == key) return tr[tr[p].l].size + 1;
    if (tr[p].key > key) return get_rank_by_key(tr[p].l, key);
    return tr[tr[p].l].size + tr[p].cnt + get_rank_by_key(tr[p].r, key);
}

// 通过排名找数值
int get_key_by_rank(int p, int rank) {
    if (!p) return INF; // 本题中不会发生此情况
    if (tr[tr[p].l].size >= rank) return get_key_by_rank(tr[p].l, rank);
    if (tr[tr[p].l].size + tr[p].cnt >= rank) return tr[p].key;
    return get_key_by_rank(tr[p].r, rank - tr[tr[p].l].size - tr[p].cnt);
}

// 找到严格小于key的最大数
int get_prev(int p, int key) {
    if (!p) return -INF;
    if (tr[p].key >= key) return get_prev(tr[p].l, key);
    return max(tr[p].key, get_prev(tr[p].r, key));
}

// 找到严格大于key的最小数
int get_next(int p, int key) {
    if (!p) return INF;
    if (tr[p].key <= key) return get_next(tr[p].r, key);
    return min(tr[p].key, get_next(tr[p].l, key));
}

int main() {
    build();
    
    scanf("%d", &n);
    while (n -- ) {
        int opt, x;
        scanf("%d%d", &opt, &x);
        if (opt == 1) insert(root, x);
        else if (opt == 2) remove(root, x);
        else if (opt == 3) printf("%d\n", get_rank_by_key(root, x) - 1);    // 去除前面一个哨兵
        else if (opt == 4) printf("%d\n", get_key_by_rank(root, x + 1));    // 算上前面一个哨兵
        else if (opt == 5) printf("%d\n", get_prev(root, x));
        else printf("%d\n", get_next(root, x));
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

> [!NOTE] **[AcWing 265. 营业额统计](https://www.acwing.com/problem/content/267/)**
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
// 本质上找prev next
// 不需要存相同值
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 33010, INF = 1e7;

int n;
struct Node {
    int l, r;
    int key, val;
}tr[N];

int root, idx;

int get_node(int key) {
    tr[++ idx].key = key;
    tr[idx].val = rand();
    return idx;
}

void build() {
    get_node(-INF), get_node(INF);
    root = 1, tr[1].r = 2;
    // pushup
    // if ... zag
}

void zig(int & p) {
    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
}

void zag(int & p) {
    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
}

void insert(int & p, int key) {
    if (!p) p = get_node(key);
    else if (tr[p].key == key) return;  // 不需要存相同值
    else if (tr[p].key > key) {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val) zig(p);
    } else {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val) zag(p);
    }
}

int get_prev(int p, int key) {
    if (!p) return -INF;
    if (tr[p].key > key) return get_prev(tr[p].l, key);
    return max(tr[p].key, get_prev(tr[p].r, key));
}

int get_next(int p, int key) {
    if (!p) return INF;
    if (tr[p].key < key) return get_next(tr[p].r, key);
    return min(tr[p].key, get_next(tr[p].l, key));
}

int main() {
    build();
    scanf("%d", &n);
    
    LL res = 0;
    for (int i = 1; i <= n; ++ i ) {
        int x;
        scanf("%d", &x);
        if (i == 1) res += x;
        else res += min(x - get_prev(root, x), get_next(root, x) - x);
        insert(root, x);
    }
    printf("%lld\n", res);
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

> [!NOTE] **[LeetCode 729. 我的日程安排表 I](https://leetcode-cn.com/problems/my-calendar-i/)**
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
class MyCalendar {
public:
    using PII = pair<int, int>;
    const int INF = 2e9;

    set<PII> S;

    MyCalendar() {
        S.insert({-INF, -INF});
        S.insert({INF, INF});
    }
    
    // 是否有交集
    bool check(PII a, PII b) {
        if (a.second <= b.first || b.second <= a.first)
            return false;
        return true;
    }

    bool book(int start, int end) {
        auto i = S.lower_bound({start, -INF});
        auto j = i;
        j -- ;
        PII t(start, end);
        if (check(*i, t) || check(*j, t))
            return false;
        S.insert(t);
        return true;
    }
};

/**
 * Your MyCalendar object will be instantiated and called as such:
 * MyCalendar* obj = new MyCalendar();
 * bool param_1 = obj->book(start,end);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1825. 求出 MK 平均值]()** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然平衡树

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MKAverage {
public:
    using LL = long long;

    struct Range {
        multiset<int> s;
        LL sum = 0;
        void insert(int x) {
            s.insert(x);
            sum += x;
        }
        void remove(int x) {
            s.erase(s.find(x));
            sum -= x;
        }
    } L, M, R;

    int m, k;
    vector<int> q;

    MKAverage(int m, int k) {
        this->m = m, this->k = k;
    }
    
    void addElement(int num) {
        q.push_back(num);
        if (q.size() < m)
            return;
        if (q.size() == m) {
            auto w = q;
            sort(w.begin(), w.end());
            for (int i = 0; i < k; ++ i )
                L.insert(w[i]);
            for (int i = k; i < m - k; ++ i )
                M.insert(w[i]);
            for (int i = m - k; i < m; ++ i )
                R.insert(w[i]);
        } else {
            M.insert(num);
            if (*M.s.begin() < *L.s.rbegin()) {
                int x = *M.s.begin(), y = *L.s.rbegin();
                M.remove(x), L.insert(x);
                L.remove(y), M.insert(y);
            }
            if (*M.s.rbegin() > *R.s.begin()) {
                int x = *M.s.rbegin(), y = *R.s.begin();
                M.remove(x), R.insert(x);
                R.remove(y), M.insert(y);
            }

            num = q[q.size() - 1 - m];
            if (M.s.count(num))
                M.remove(num);
            else if (L.s.count(num)) {
                L.remove(num);
                int x = *M.s.begin();
                M.remove(x), L.insert(x);
            } else {
                R.remove(num);
                int x = *M.s.rbegin();
                M.remove(x), R.insert(x);
            }
        }
    }
    
    int calculateMKAverage() {
        if (q.size() < m)
            return -1;
        return M.sum / M.s.size();
    }
};

/**
 * Your MKAverage object will be instantiated and called as such:
 * MKAverage* obj = new MKAverage(m, k);
 * obj->addElement(num);
 * int param_2 = obj->calculateMKAverage();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### BST

> [!NOTE] **[LeetCode 98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)**
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
    bool helper(TreeNode* r, long& pre) {
        if (!r) return true;
        if (!helper(r->left, pre)) return false;
        if (r->val <= pre) return false;
        pre = r->val;
        return helper(r->right, pre); 
    }
    bool isValidBST(TreeNode* root) {
        long pre = LONG_MIN;
        return helper(root, pre);
    }
};
```

##### **Python**

```python
# 法一：按照定义来递归求解：
# 1） 左子树的所有节点的值都小于当前节点的值；
# 2） 右子树的所有节点的值都大于当前节点的值；
# 3） 左子树和右子树都必须是合法的二叉搜索树；
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        res = []

        def dfs(u):
            if not u:return 
            dfs(u.left)
            res.append(u.val)
            dfs(u.right)
        
        dfs(root)
        p1 = 0 
        while p1 + 1 < len(res):
            if res[p1] >= res[p1 + 1]:
                return False 
            p1 += 1
        return True


# 法二：中序遍历时，判断当前节点是否大于中序遍历的前一个节点，如果大于，说明满足 BST，继续遍历；否则直接返回 false。

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        self.pre = None
        
        def isBST(u):
            if not u:
                return True
            if not isBST(u.left): # 访问左子树 
                return False
            if self.pre and self.pre.val >= u.val: # 访问当前节点：如果当前节点小于等于中序遍历的前一个节点，说明不满足BST，返回 false；否则继续遍历。
                return False
            self.pre = u  # 将当前节点置为“前一个节点”
            return  isBST(u.right) # 访问右子树

        return isBST(root)
        

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/)**
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
class BSTIterator {
private:
    stack<TreeNode*> s;
public:
    BSTIterator(TreeNode* root) {
        while (root) {
            s.push(root);
            root = root->left;
        }
    }
    
    /** @return the next smallest number */
    int next() {
        TreeNode* cur = s.top();
        s.pop();
        int res = cur->val;
        if (cur->right) {
            cur = cur->right;
            while (cur) {
                s.push(cur);
                cur = cur->left;
            }
        }
        return res;
    }
    
    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !s.empty();
    }
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
```

##### **Python**

```python
# 1. init: 将根节点的左子树遍历，并且压入栈
# 2. next: 1) 因为是中序遍历，所以当前栈顶元素就是最小元素；2）弹出栈顶元素后，再将该节点向右移动，并且将左子树都遍历，压入栈中
# 3. hasNext: 判断栈中是否还有元素

class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        cur = self.stack.pop()
        node = cur.right
        while node:
            self.stack.append(node)
            node = node.left 
        return cur.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> > morris遍历利用的是树的叶节点左右孩子为空（树的大量空闲指针）
> >
> > 实现空间开销的极限缩减。
>
> [!TIP] **morris遍历的实现原则**
>
> - 记作当前节点为 `cur`
>
>   - 如果 `cur` 无左孩子，`cur` 向右移动（`cur=cur.right`）
>
>   - 如果 `cur` 有左孩子，找到 `cur` 左子树上最右的节点，记为 `mostright`
>
>     - 如果 `mostright` 的 `right` 指针指向空，让其指向 `cur` ，`cur` 向左移动（`cur=cur.left`）
>
>     - 如果 `mostright` 的 `right` 指针指向 `cur` ，让其指向空，`cur` 向右移动（`cur=cur.right`）
>
> 实现以上的原则，即实现了 `morris` 遍历。
>
> **morris遍历的实质:**
>
> 建立一种机制，对于没有左子树的节点只到达一次，对于有左子树的节点会到达两次

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
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void recoverTree(TreeNode* root) {
        TreeNode * p1 = nullptr, * p2 = nullptr, * cur = root, * pre = nullptr;
        while (cur) {
            if (cur->left) {
                auto p = cur->left;
                while (p->right && p->right != cur)
                    p = p->right;
                if (p->right) {
                    // cause p->right = cur;
                    p->right = nullptr;
                } else {
                    p->right = cur;
                    cur = cur->left;
                    continue;
                }
            }
            if (pre && cur->val < pre->val) {
                if (p1 == nullptr)
                    p1 = pre;
                p2 = cur;
            }
            pre = cur;
            cur = cur->right;
        }
        swap(p1->val, p2->val);
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

> [!NOTE] **[LeetCode 450. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)**
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
    TreeNode* deleteNode(TreeNode* root, int key) {
        del(root, key);
        return root;
    }

    void del(TreeNode* &root, int key) {
        if (!root) return;
        if (key == root->val) {
            if (!root->left && !root->right) root = NULL;  // 子节点
            else if (!root->left) root = root->right;  // 只有右儿子
            else if (!root->right) root = root->left;  // 只有左儿子
            else {  // 左右儿子都有
                auto p = root->right;
                while (p->left) p = p->left;  // 找后继
                root->val = p->val;
                del(root->right, p->val);
            }
        }
        else if (key < root->val) del(root->left, key);
        else del(root->right, key);
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

> [!NOTE] **[LeetCode 501. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)**
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
// 1. 符合惯性 但实现并不友好
class Solution {
public:
    vector<int> res;
    int maxc = 0, cnt = 0, last = INT_MIN;
    void dfs(TreeNode * root) {
        if (!root) return;
        dfs(root->left);
        if (root->val == last) ++ cnt;
        else {
            if (cnt > maxc) res = {last}, maxc = cnt;
            else if (cnt == maxc) res.push_back(last);
            cnt = 1;
        }
        last = root->val;
        dfs(root->right);
    }
    vector<int> findMode(TreeNode* root) {
        dfs(root);
        if (cnt > maxc) res = {last}, maxc = cnt;
        else if (cnt == maxc && cnt) res.push_back(last);
        return res;
    }
};


// 2. 新写法
// yxc
class Solution {
public:
    vector<int> res;
    int maxc = 0, cnt = 0, last = INT_MIN;
    void dfs(TreeNode * root) {
        if (!root) return;
        dfs(root->left);
        if (!cnt || root->val == last) ++ cnt;
        else cnt = 1;
        last = root->val;
        if (cnt > maxc) res = {last}, maxc = cnt;
        else if (cnt == maxc) res.push_back(last);
        dfs(root->right);
    }
    vector<int> findMode(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        if not root:
            return
        res = []
        max_count =  count = 0 
        base = float("-inf") 
        #对当前值进行处理
        def update(x):
            nonlocal max_count,count,base,res
            #相等则计数+1
            if x == base:
                count += 1
            else:
                #不相等，说明该数的节点已经全部遍历完成，更新base，count为1
                base = x
                count = 1
            #计数如果等于max，则加入result
            if count == max_count:
                res.append(base)
            #计数如果大于max，则要重置result，并把该值加入      
            if count > max_count:
                max_count = count
                res = []
                res.append(base)
        #二叉查找树，中序遍历，数据从小到大顺序处理
        def dfs(root):
            if root:
                nonlocal max_count,count,base,res
                dfs(root.left)
                update(root.val)
                dfs(root.right)
        dfs(root)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 701. 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)**
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
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (!root)
            return new TreeNode(val);
        if (root->val > val)
            root->left = insertIntoBST(root->left, val);
        else
            root->right = insertIntoBST(root->right, val);
        return root;
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

## TODO(再整理): 拓展 morris各种遍历方式

- 先序

  ```
  // [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)
  class Solution {
  public:
      vector<int> preorderTraversal(TreeNode* root) {
          vector<int> res;
          TreeNode * cur = root;
          while (cur) {
              if (cur->left) {
                  auto p = cur->left;
                  while (p->right && p->right != cur)
                      p = p->right;
                  if (p->right) {
                      p->right = nullptr;
                  } else {
                      // -------------------- 先序输出 --------------------
                      res.push_back(cur->val);
                      // -------------------------------------------------
                      p->right = cur;
                      cur = cur->left;
                      continue;
                  }
              } else {
                  // 相对于普通流程新增的 else 
                  // -------------------- 先序输出 --------------------
                  res.push_back(cur->val);
                  // -------------------------------------------------
              }
              cur = cur->right;
          }
          return res;
      }
  };
  ```

- 中序

  ```
          while (cur) {
              if (cur->left) {
                  auto p = cur->left;
                  while (p->right && p->right != cur)
                      p = p->right;
                  if (p->right) {
                      p->right = nullptr;
                  } else {
                      p->right = cur;
                      cur = cur->left;
                      continue;
                  }
              }
              // -------------------- 中序输出 --------------------
              cout << cur->val << endl;
              // -------------------------------------------------
              cur = cur->right;
          }
  ```

- 后序

  ```
  class Solution {
  public:
      vector<int> res;
      
      void printNode(TreeNode * node) {
          vector<int> t;
          auto cur = node;
          while (cur) {
              t.push_back(cur->val);
              cur = cur->right;
          }
          for (int i = t.size() - 1; i >= 0; -- i )
              res.push_back(t[i]);
      }
  
      vector<int> postorderTraversal(TreeNode* root) {
          TreeNode * cur = root;
          while (cur) {
              if (cur->left) {
                  auto p = cur->left;
                  while (p->right && p->right != cur)
                      p = p->right;
                  if (p->right) {
                      p->right = nullptr;
                      // -------------------- 后序输出 --------------------
                      // 逆序打印左子树的右边界
                      printNode(cur->left);
                      // -------------------------------------------------
                  } else {
                      p->right = cur;
                      cur = cur->left;
                      continue;
                  }
              }
              cur = cur->right;
          }
          // -------------------- 后序输出 --------------------
          printNode(root);
          // -------------------------------------------------
          return res;
      }
  };
  ```

  `printNode` 可以同样规避对数组的使用：

  ```
      TreeNode * reverseEdge(TreeNode * node) {
          TreeNode * pre = nullptr, * cur = node;
          while (cur) {
              auto next = cur->right;
              cur->right = pre;
              pre = cur, cur = next;
          }
          return pre;
      }
  
      void printNode(TreeNode * node) {
          auto tail = reverseEdge(node);
          auto cur = tail;
          while (cur) {
              res.push_back(cur->val);
              cur = cur->right;
          }
          reverseEdge(tail);
      }
  ```

  ### 构造 BST

  > [!NOTE] **[LeetCode 108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)**
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
    TreeNode* helper(vector<int>& nums, int l, int r) {
        if (l > r) return nullptr;
        int mid = l + (r - l) / 2;
        TreeNode* node = new TreeNode(nums[mid]);
        node->left = helper(nums, l, mid - 1);
        node->right = helper(nums, mid + 1, r);
        return node;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        int n = nums.size();
        return helper(nums, 0, n - 1);
    }
};
```

##### **Python**

```python
# 递归建立整棵二叉树: 每次以中点为根，以左半部分为左子树，右半部分为右子树。先分别递归建立左子树和右子树，然后令根节点的指针分别指向两棵子树。

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def dfs(l, r):
            if l > r:return None
            m = l + (r - l) // 2
            root = TreeNode(nums[m])
            root.left = dfs(l, m - 1)
            root.right = dfs(m + 1, r)
            return root 
        return dfs(0, len(nums) - 1)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
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
    TreeNode* sortedListToBST(ListNode* head) {
        if (!head) return NULL;
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;
        if (n == 1) return new TreeNode(head->val);
        auto cur = head;
        for (int i = 0; i < n / 2 - 1; i ++ ) cur = cur->next;
        auto root = new TreeNode(cur->next->val);
        root->right = sortedListToBST(cur->next->next);
        cur->next = NULL;
        root->left = sortedListToBST(head);
        return root;
    }
};
```

##### **C++ 2**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
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
    TreeNode* build(ListNode* head, ListNode* tail) {
        if (head == tail) return nullptr;
        ListNode *slow = head, *fast = head;
        while (fast != tail && fast->next != tail) {
            slow = slow->next;
            fast = fast->next->next;
        }
        TreeNode * n = new TreeNode(slow->val);
        n->left = build(head, slow);
        n->right = build(slow->next, tail);
        return n;
    }

    TreeNode* sortedListToBST(ListNode* head) {
        return build(head, nullptr);
    }
};
```

##### **Python**

```python
# 写法1
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head or not head.next:return head 

        def dfs(head):
            if not head:return None
            p, n = head, 0 
            while p:
                p = p.next 
                n += 1
            if n == 1:return TreeNode(head.val)   # 踩坑： 这里要判断当前链表长度只有一个的时候 就直接返回
            cur = head
            for i in range(n//2 - 1):
                cur = cur.next
            root = TreeNode(cur.next.val)
            root.right = dfs(cur.next.next)  # 踩坑！ 先递归root.right, 然后再把cur.next 置为None
            cur.next = None
            root.left = dfs(head)
            return root
        
        return dfs(head)
      
# 写法2       
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        def dfs(head, tail):
            if head == tail:return None
            slow, fast = head, head 
            while fast!= tail and fast.next != tail:  # 踩坑！注意 这里要fast != tail 
                slow = slow.next 
                fast = fast.next.next
            root = TreeNode(slow.val)
            root.left = dfs(head, slow)
            root.right = dfs(slow.next, tail)
            return root
            
        return dfs(head, None)
```

<!-- tabs:end -->
</details>

<br>

* * *