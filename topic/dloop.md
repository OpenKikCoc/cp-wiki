> [!TIP] **定义**
>
> **给一个序列，序列两两元素可以任意交换，求最少的交换次数使得序列满足指定顺序**
>
> 这是一个经典问题，一般有两种做法（详见 https://www.geeksforgeeks.org/minimum-number-swaps-required-sort-array/ ）：
>
> 1.  从 $1$ 到 $n$ 枚举下标 $i$ 。设当前序列第 $i$ 个位置的数为 $a_i$，目标序列第 $i$ 个位置的数为 $b_i$ 。
> 
>     若 $a_i \ne b_i$ 则 不断 将 $ai$ 交换到目标位置，直到 $a_i = b_i$ 。交换次数就是答案。
> 
> 2.  求整个序列中置换环的数量，答案就是序列长度减去置换环的数量。【可以直接使用**并查集**实现】

## 习题


> [!NOTE] **[LeetCode 765. 情侣牵手](https://leetcode.cn/problems/couples-holding-hands/)**
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
    // 经典图论模型 环状图 每次交换环中断
    // 目标 把环状图变为若干个自环

    vector<int> p;

    int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }

    int minSwapsCouples(vector<int>& row) {
        int n = row.size() / 2;
        for (int i = 0; i < n; ++ i )
            p.push_back(i);
        
        int cnt = n;
        for (int i = 0; i < n * 2; i += 2 ) {
            int a = row[i] / 2, b = row[i + 1] / 2;
            // 环等价于联通块 找环即找联通块
            if (find(a) != find(b)) {
                p[find(a)] = find(b);
                cnt -- ;
            }
        }
        return n - cnt;
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

> [!NOTE] **[LeetCode 2471. 逐层排序二叉树所需的最少操作数目](https://leetcode.cn/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/)**
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
    vector<int> pa;
    int find(int x) {
        if (pa[x] != x)
            pa[x] = find(pa[x]);
        return pa[x];
    }
    
    int get(vector<int> & t) {
        int n = t.size();
        unordered_map<int, int> hash;   // 离散化
        pa.clear();
        for (int i = 0; i < n; ++ i ) {
            pa.push_back(i);
            hash[t[i]] = i;
        }
        sort(t.begin(), t.end());   // ATTENTION
        // 并查集求环大小
        int cnt = 0;
        for (int i = 0; i < n; ++ i ) {
            int a = find(i), b = find(hash[t[i]]);
            if (a != b) {
                pa[a] = b;
                cnt ++ ;
            }
        }
        return cnt;
    }
    
    int minimumOperations(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        int res = 0;
        while (q.size()) {
            int sz = q.size();
            vector<int> t;
            while (sz -- ) {
                auto u = q.front(); q.pop();
                t.push_back(u->val);
                if (u->left)
                    q.push(u->left);
                if (u->right)
                    q.push(u->right);
            }
            res += get(t);
        }
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