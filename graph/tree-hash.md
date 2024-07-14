我们有时需要判断一些树是否同构。这时，选择恰当的哈希方式来将树映射成一个便于储存的哈希值（一般是 32 位或 64 位整数）是一个优秀的方案。

树哈希有很多种哈希方式，下面将选出几种较为常用的方式来加以介绍。

## 方法一

### 公式

$$
f_{now}=size_{now} \times \sum f_{son(now,i)}\times seed^{i-1}
$$

#### 注

其中 $f_x$ 为以节点 $x$ 为根的子树对应的哈希值。特殊地，我们令叶子节点的哈希值为 $1$。

$size_{x}$ 表示以节点 $x$ 为根的子树大小。

$son_{x,i}$ 表示 $x$ 所有子节点以 $f$ 作为关键字排序后排名第 $i$ 的儿子。

$seed$ 为选定的一个合适的种子（最好是质数，对字符串 hash 有了解的人一定不陌生）

上述哈希过程中，可以适当取模避免溢出或加快运行速度。

#### Hack

![treehash1](./images/tree-hash1-hack.svg)

上图中，可以计算出两棵树的哈希值均为 $60(1+seed)$。

## 方法二

### 公式

$$
f_{now}=\bigoplus f_{son(now,i)}\times seed+size_{son(now,i)}
$$

#### 注

其中 $f_x$ 为以节点 $x$ 为根的子树对应的哈希值。特殊地，我们令叶子节点的哈希值为 $1$。

$size_{x}$ 表示以节点 $x$ 为根的子树大小。

$son_{x,i}$ 表示 $x$ 所有子节点之一（不用排序）。

$seed$ 为选定的一个合适的质数。

$\bigoplus$ 表示异或和。

#### Hack

由于异或的性质，如果一个节点下有多棵本质相同的子树，这种哈希值将无法分辨该种子树出现 $1,3,5,\dots$ 次的情况。

## 方法三

### 公式

$$
f_{now}=1+\sum f_{son(now,i)} \times prime(size_{son(now,i)})
$$

### 注

其中 $f_x$ 为以节点 $x$ 为根的子树对应的哈希值。

$size_{x}$ 表示以节点 $x$ 为根的子树大小。

$son_{x,i}$ 表示 $x$ 所有子节点之一（不用排序）。

$prime(i)$ 表示第 $i$ 个质数。

> [!WARNING]
> 
> 对于两棵大小不同的树 $T_1,T_2$，$f_{T_1}=f_{T_2}$ 是可能的，因此在判断树同构前要先判断大小是否相等。

## 例题

### 例题一 [「BJOI2015」树的同构](https://www.luogu.com.cn/problem/P5043)

我们用上述方式任选其一进行哈希，注意到我们求得的是子树的 hash 值，也就是说只有当根一样时同构的两棵子树 hash 值才相同。由于数据范围较小，我们可以暴力求出以每个点为根时的哈希值，也可以通过 up and down 树形 dp 的方式，遍历树两遍求出以每个点为根时的哈希值，排序后比较。

如果数据范围较大，我们可以通过找重心的方式来优化复杂度。（一棵树的重心最多只有两个，分别比较即可）

#### 做法一

> [!TIP] **例题参考代码**

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

#### 做法二

> [!TIP] **例题参考代码**

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

### 例题二 [HDU 6647](http://acm.hdu.edu.cn/showproblem.php?pid=6647)

题目要求的是遍历一棵无根树产生的本质不同括号序列方案数。

首先，注意到一个结论，对于两棵有根树，如果他们不同构，一定不会生成相同的括号序列。我们先考虑遍历有根树能够产生的本质不同括号序列方案数，假设我们当前考虑的子树根节点为 $u$，记 $f(u)$ 表示这棵子树的方案数，$son(u)$ 表示 $u$ 的儿子节点集合，从 $u$ 开始往下遍历，顺序可以随意选择，产生 $|son(u)|!$ 种排列，遍历每个儿子节点 $v$，$v$ 的子树内有 $f(v)$ 种方案，因此有 $f(u)=|son(u)|! \cdot \prod_{v\in son(u)} f(v)$。但是，同构的子树之间会产生重复，$f(u)$ 需要除掉每种本质不同子树出现次数阶乘的乘积，类似于多重集合的排列。

通过上述树形 dp，可以求出根节点的方案数，再通过 up and down 树形 dp，将父亲节点的哈希值和方案信息转移给儿子，可以求出以每个节点为根时的哈希值和方案数，每种不同的子树只需要计数一次即可。

注意，本题数据较强，树哈希很容易发生冲突。这里提供一个比较简单的解决方法，求出一个节点子树的哈希值后，可以将其前后分别插入一个值再计算一遍哈希值。

#### 做法

> [!TIP] **例题参考代码**

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

## 写在最后

事实上，树哈希是可以很灵活的，可以有各种各样奇怪的姿势来进行 hash，只需保证充分性与必要性，选手完全可以设计出与上述方式不同的 hash 方式。

## 参考资料

方法三参考自博客 [树 hash](https://www.cnblogs.com/huyufeifei/p/10817673.html)。

## 习题

> [!NOTE] **[LeetCode 572. 另一个树的子树](https://leetcode.cn/problems/subtree-of-another-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ hash**

```cpp
// yxc 树hash做法
class Solution {
public:
    const int P = 131, Q = 159, MOD = 1e7 + 7;
    int T = -1;
    bool ans = false;

    int dfs(TreeNode* root) {
        if (!root) return 12345;
        int left = dfs(root->left), right = dfs(root->right);
        int x = (root->val % MOD + MOD) % MOD;
        if (left == T || right == T) ans = true;
        return (x + left * P % MOD + right * Q) % MOD;
    }

    bool isSubtree(TreeNode* s, TreeNode* t) {
        T = dfs(t);
        if (T == dfs(s)) ans = true;
        return ans;
    }
};
```

##### **C++ 传统**

```cpp
class Solution {
public:
    bool helper(TreeNode* s, TreeNode* t) {
        // 注意 需完全一致
        if (s == nullptr && t == nullptr) return true;
        else if (s == nullptr || t == nullptr) return false;
        return s->val == t->val && helper(s->left, t->left) && helper(s->right, t->right);
    }
    bool isSubtree(TreeNode* s, TreeNode* t) {
        if (s == nullptr && t == nullptr) return true;
        else if (s == nullptr || t == nullptr) return false;
        if(s->val == t->val) return helper(s, t) || isSubtree(s->left, t) || isSubtree(s->right, t);
        return isSubtree(s->left, t) || isSubtree(s->right, t);
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

> [!NOTE] **[LeetCode 652. 寻找重复的子树](https://leetcode.cn/problems/find-duplicate-subtrees/)**
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
    // 将一颗树唯一【映射】到一个整数
    vector<TreeNode*> res;

    // 唯一id
    int cnt = 0;
    unordered_map<string, int> ids;
    unordered_map<int, int> hash;

    int dfs(TreeNode * root) {
        if (!root) return 0;
        int left = dfs(root->left);
        int right = dfs(root->right);
        string key = to_string(root->val) + ' ' + to_string(left) + ' ' + to_string(right);
        if (ids.count(key) == 0) ids[key] = ++ cnt ;
        int id = ids[key];
        if (++ hash[id] == 2) res.push_back(root);
        return id;
    }


    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        dfs(root);
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