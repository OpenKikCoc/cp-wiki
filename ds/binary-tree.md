## 习题

### 重建二叉树

> [!NOTE] **[LeetCode 105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)**
> 
> **[SwordOffer ...]()**
> 
> 题意: 根据前序和中序

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
    int n;
    unordered_map<int, int> mp;

    TreeNode* helper(vector<int> & pre, int l, int r, int p) {
        if (l > r)
            return nullptr;
        
        int pa = pre[p], id = mp[pa], llen = id - l;
        TreeNode * ret = new TreeNode(pa);
        ret->left = helper(pre, l, id - 1, p + 1);
        ret->right = helper(pre, id + 1, r, p + 1 + llen);
        return ret;
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        n = inorder.size();
        for (int i = 0; i < n; ++ i )
            mp[inorder[i]] = i;
        return helper(preorder, 0, n - 1, 0);
    }
};
```

##### **Python**

```python
#算法流程：
#1. 在前序遍历中找到根节点：前序遍历的第一个数 就是根节点的值
#2. 在中序遍历中找到根节点对应的位置k，则 k的左边就是左子树的中序遍历，k的右边就是右子树的中序遍历 （这一步需要用一个字典来存储对应的位置）
#3. 假设左子树的长度为l,那么在前序遍历里，根节点后l个数 是左子树的前序遍历，剩下的树就是右子树的前序遍历
#4. 有了左右子树的前序遍历和中序遍历，我们可以先递归创建出左右子树，然后再创建根节点；
class Solution:
    def buildTree(self, pre: List[int], ino: List[int]) -> TreeNode:
        my_dict = dict()
        for i in range(len(ino)):
            my_dict[ino[i]] = i

        def dfs(pre_L, pre_R, ino_L, ino_R):
            # 踩坑： 只能 大于 的时候 才能return! 进入 dfs 后，每次都记得先想一下终止条件！
            if pre_L > pre_R:return   
            # 进入 dfs ，把每一层的 root 节点构造出来！
            root = TreeNode(pre[pre_L])   
            idx = my_dict[pre[pre_L]]
            root.left = dfs(pre_L + 1, idx - ino_L + pre_L, ino_L, idx - 1)
            root.right = dfs(idx - ino_L + pre_L + 1, pre_R, idx + 1, ino_R)
            # 递归返回这一层对应的root , 也就是重建后的二叉树
            return root  
        
        return dfs(0, len(pre) - 1, 0, len(ino) - 1)      
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)**
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
    vector<int> po;
    int n;
    unordered_map<int, int> mp;

    TreeNode * helper(int l, int r, int px) {
        if (l > r)
            return nullptr;
        
        int v = po[px], idx = mp[v], len = r - idx;
        TreeNode * t = new TreeNode(v);
        t->left = helper(l, idx - 1, px - len - 1);
        t->right = helper(idx + 1, r, px - 1);
        return t;
    }

    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        this->po = postorder;
        this->n = po.size();
        for (int i = 0; i < n; ++ i )
            mp[inorder[i]] = i;
        return helper(0, n - 1, n - 1);
    }
};
```

##### **Python**

```python
#算法流程：
#1. 在后续遍历中找到根节点：后续遍历的第一个数 就是根节点的值
#2. 在中序遍历中找到根节点对应的位置idx，则idx的左边就是左子树的中序遍历，右边的就是右子树的中序遍历(这一步需要用一个字典来存储对应的位置）
#3. 假设左子树的长度为l,那么在后序遍历里，从第一个数算开始的 1 个数是左子树的前序遍历，剩下的树就是右子树的后续遍历
#4. 有了左右子树的中序遍历和后序遍历，我们可以创造出根节点，然后递归它的左右子树；

class Solution:
    def buildTree(self, ino: List[int], po: List[int]) -> TreeNode:
        my_dict = {}
        for i in range(len(ino)):
            my_dict[ino[i]] = i 
        
        def dfs(ino_L, ino_R, po_L, po_R):
            if ino_L > ino_R:return 
            root = TreeNode(po[po_R])
            idx = my_dict[po[po_R]]
            root.left = dfs(ino_L, idx - 1, po_L, idx - ino_L + po_L - 1)
            root.right = dfs(idx + 1, ino_R, idx - ino_L + po_L, po_R - 1)
            return root 
        
        return dfs(0, len(ino) - 1, 0 , len(po) - 1)
```

<!-- tabs:end -->
</details>

<br>

* * *

### 转换

> [!NOTE] **[LeetCode 114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)**
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
    // ATTENTION
    void flatten(TreeNode* root) {
        while (root) {
            if (root->left) {
                TreeNode *pre = root->left;
                while (pre->right)
                    pre = pre->right;
                pre->right = root->right;
                root->right = root->left;
                root->left = nullptr;
            }
            root = root->right;
        }
        return;
    }
    // another
    TreeNode* pre = nullptr;
    void flatten(TreeNode* root) {
        if (!root) return;
        flatten(root->right);
        flatten(root->left);
        root->right = pre;
        root->left = nullptr;
        pre = root;
    }
};
```

##### **Python**

```python
# 找规律 + 模拟
# 1. 当前节点存在左子树，将左子树的右链插入到当前点的右边
# 2. 否则 不存在的时候，遍历当前节点的右儿子
# 原地：不能用递归
class Solution:
    def flatten(self, root: TreeNode) -> None:
        while root:
            p = root.left
            if p:
                while p.right:
                    p = p.right 
                p.right = root.right
                root.right = root.left
                root.left = None # 一定要记得清空，因为左儿子已经移走了
            root = root.right  # 上述步骤完成后，当前节点一定没有左儿子了，所以只能往右边走
```

<!-- tabs:end -->
</details>

<br>

* * *

### 序列化

> [!NOTE] **[LeetCode 297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc**

```cpp
// yxc
class Codec {
public:
    string path;
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        dfs_s(root);
        return path;
    }

    void dfs_s(TreeNode* root) {
        if (!root) path += "#,";
        else {
            path += to_string(root->val) + ',';
            dfs_s(root->left);
            dfs_s(root->right);
        }
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        int u = 0;
        return dfs_d(data, u);
    }

    TreeNode* dfs_d(string& data, int& u) {
        if (data[u] == '#') {
            u += 2;
            return NULL;
        } else {
            int k = u;
            while (data[u] != ',') u ++ ;
            auto root = new TreeNode(stoi(data.substr(k, u - k)));
            u ++ ;
            root->left = dfs_d(data, u);
            root->right = dfs_d(data, u);
            return root;
        }
    }
};
```

##### **C++**

```cpp
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        //if(!root) return "";
        stringstream ss;
        dfs(root, ss);
        return ss.str();
    }
    void dfs(TreeNode* r, stringstream& ss) {
        if (!r) {
            ss << "# ";
            return;
        }
        ss << to_string(r->val) << " ";
        dfs(r->left, ss);
        dfs(r->right, ss);
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        //if(data.empty()) return nullptr;
        stringstream ss(data);
        TreeNode* r = nullptr;
        rebuild(r, ss);
        return r;
    }
    void rebuild(TreeNode* & r, stringstream& ss) {
        string t;
        ss >> t;
        if (t[0] == '#') {
            r = nullptr;
            return;
        }
        r = new TreeNode(stoi(t));
        rebuild(r->left, ss);
        rebuild(r->right, ss);
    }
};
```

##### **Python**

```python
# python3
# 【二叉树被序列化为一个【字符串】！ 并且讲这个【字符串】反序列化为原始的树结构】
# 题目要求的 序列化 和 反序列化 是 可逆操作。因此，序列化的字符串应携带 完整的二叉树信息。【通常使用的前序、中序、后序、层序遍历记录的二叉树的信息不完整，即唯一的输出序列可能对应着多种二叉树可能性。】
# 序列化：通过 层序遍历 实现
# 反序列化：根据序列化拿到的层序遍历的结果，按照 层 重构二叉树。借助一个指针 i 指向当前节点 root 的左、右结点，每构建一个 node 的左右节点，指针就向右移动 1 位（i += 1)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    def serialize(self, root):
        if not root:return 
        q = collections.deque()
        q.append(root)
        res = []
        while q:
            node = q.popleft()
            if node:
                res.append(str(node.val))
                # 不管 node.left 是否存在 都要放到队列中，这样如果不存在，该位置就可以被置为'null'
                q.append(node.left)  
                q.append(node.right)
            else:  
                # 当前位置没有结点时，需要进行标识为'null'
                res.append("null")  
        return '[' + ','.join(res) + ']'

    def deserialize(self, data):
        if not data:return
        
        # 前后的 [ ] 这两个字符串 不需要进入重构二叉树
        nums = data[1:-1].split(',')  
        
        # 层序遍历的第一个点 就是 root 的值
        root = TreeNode(int(nums[0]))        
        q = collections.deque()
        q.append(root)
        i = 1
        while q:
            node = q.popleft()
            if nums[i] != "null":
                node.left = TreeNode(int(nums[i]))
                q.append(node.left)
            i += 1
            if nums[i] != "null":
                node.right = TreeNode(int(nums[i]))
                q.append(node.right)
            i += 1
        return root
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 449. 序列化和反序列化二叉搜索树](https://leetcode-cn.com/problems/serialize-and-deserialize-bst/)**
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
class Codec {
public:
// yxc 极致紧凑
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string res;
        dfs_s(root, res);
        return res;
    }

    void dfs_s(TreeNode* root, string& res) {
        if (!root) return;
        res += to_string(root->val) + ' ';
        dfs_s(root->left, res), dfs_s(root->right, res);
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string str) {
        vector<int> data;
        stringstream ssin(str);
        int x, u = 0;
        while (ssin >> x) data.push_back(x);
        return dfs_d(data, u, INT_MIN, INT_MAX);
    }

    TreeNode* dfs_d(vector<int>& data, int& u, int minv, int maxv) {
        if (u == data.size() || data[u] < minv || data[u] > maxv) return NULL;
        auto root = new TreeNode(data[u ++ ]);
        root->left = dfs_d(data, u, minv, root->val);
        root->right = dfs_d(data, u, root->val + 1, maxv);
        return root;
    }


    // 原本自己写法 不算最紧凑
    string serialize(TreeNode* root) {
        if(!root) return string();
        stringstream ss;
        dfs(root,ss);
        return ss.str();
    }
    void dfs(TreeNode* rt,stringstream& ss){
        if(!rt){
            ss << "# ";
            return ;
        }
        ss << to_string(rt->val) <<" ";
        dfs(rt->left,ss);
        dfs(rt->right,ss);
    }
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if(data.empty()) return NULL;
        TreeNode* rt = NULL;
        stringstream ss(data);
        rebuild(rt,ss);
        return rt;
    }
    void rebuild(TreeNode* & rt,stringstream& ss){
        string t;
        ss >> t;
        if(t[0] == '#'){
            rt = NULL;
            return;
        }
        int v = stoi(t);
        rt = new TreeNode(v);
        rebuild(rt->left, ss);
        rebuild(rt->right, ss);
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

### 连接节点

> [!NOTE] **[LeetCode 116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)**
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
    Node* connect(Node* root) {
        if (!root) return root;
        auto source = root; // 存一下根，返回用
        // 当有左儿子时说明有下一层
        while (root->left) {
            // 使用next遍历这一层每个结点p，以处理下一层的连接关系
            for (auto p = root; p; p = p->next) {
                // p的左儿子的next就是p的右儿子
                p->left->next = p->right;
                // p的右儿子的next就是p的next的左儿子，要保证p->next存在
                if (p->next)
                    p->right->next = p->next->left;
            }
            root = root->left; // 每次向左儿子走就走到了下一层的第一个结点
        }
        return source;
    }
    Node* connect2(Node* root) {
        if (!root) return root;
        Node *left = root->left, *right = root->right;
        while (left) {
            left->next = right;
            left = left->right;
            right = right->left;
        }
        connect(root->left);
        connect(root->right);
        return root;
    }
};
```

##### **Python**

```python
#  完美二叉树！通过题意给出的构造函数，可以看出每个点的next指针都是默认初始化为空，所以每一行的最后一个节点不需要处理
#  BFS，需要一个队列，但是有next指针，所以队列可以省掉，那就是符合题意：常数空间
#  算法： 从根节点开始宽度优先遍历，每次遍历一层，遍历时按从左到右的顺序，对于每个节点，先让左儿子指向右儿子，然后让右儿子指向下一个节点的左儿子。最后让这一层最右侧的节点指向NULL。 遍历到叶节点所在的层为止。

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:return 
        source = root  #  存一下根，返回用
        while root.left:   # 当前节点的左儿子存在，表示还有下一层需要处理. 直到遍历到叶节点
            p = root 
            while p:
                p.left.next = p.right 
                if p.next:
                    p.right.next = p.next.left 
                p = p.next  #  使用next遍历这一层每个结点p，以处理下一层的连接关系
            root = root.left   # 每次向左儿子走就走到了下一层的第一个结点
        return source
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 117. 填充每个节点的下一个右侧节点指针 II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)**
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
    Node* connect(Node* root) {
        if (!root) return nullptr;
        queue<Node*> q;
        q.push(root);
        while (!q.empty()) {
            int sz = q.size();
            while (sz--) {
                Node* t = q.front(); q.pop();
                if (sz) t->next = q.front();
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
        }
        return root;
    }
};


// 常数空间
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        auto cur = root;
        while (cur) {
            auto dummy = new Node(-1);
            auto pre = dummy;
            for (auto p = cur; p; p = p->next) {
                if (p->left) pre = pre->next = p->left;
                if (p->right) pre = pre->next = p->right;
            }
            cur = dummy->next;
        }
        return root;
    }
};
```

##### **Python**

```python
# 非完美二叉树
# 算法：1. 从根节点开始BFS，每次遍历一层，从左到右依次遍历每个节点；
# 2. 遍历时维护下一层节点的链表。对于每个节点，依次判断他的左儿子和右儿子是否存在，如果存在，则插入到下一层链表的末尾
# 3. 并且每次链表的tail 都要往后走一位。

# 用root记层该层访问节点
# 用dummy记录下一层合成链表的开头.
# 用tail记录下一层合成链表的结尾.

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:return 
        head = root 
        while root:
            dummy = ListNode(None)
            tail = dummy 
            while root:
                if root.left:
                    tail.next = root.left 
                    tail = tail.next 
                if root.right:
                    tail.next = root.right
                    tail = tail.next 
                root = root.next 
            root = dummy.next 
        return head
```

<!-- tabs:end -->
</details>

<br>

* * *

### trick

> [!NOTE] **[LeetCode 331. 验证二叉树的前序序列化](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二叉树性质 思维 trick	

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool f = true;
    void dfs(string & preorder, int & u) {
        if (u == preorder.size()) {
            f = false;
            return;
        }
        if (preorder[u] == '#') {
            u += 2;
            return;
        }
        while (preorder[u] != ',') ++ u;
        ++ u;
        dfs(preorder, u);
        dfs(preorder, u);
    }
    bool isValidSerialization(string preorder) {
        preorder += ',';
        int u = 0;
        dfs(preorder, u);
        return f && u == preorder.size();
    }
};
```

##### **C++ very trick**

```cpp
/*
https://www.youtube.com/watch?v=_mbnPPHJmTQ

1.For any tree, number of nodes == number of edges + 1. (so we add 1 to number of edges first)
2.The hashtag(#) should only appear when there's edge available.
Then we have the algorithm or statement:

1.each node consumes 1 edge
2.each non-leaf node creates two edges
3.whenever edges are smaller than 0, return false, which means number of hashtag(#) is too much
4. Finally, edges should be zero to meet the 1st constraint which is number of nodes == number of edges + 1

*/
class Solution {
public:
    bool isValidSerialization(string preorder) {
        int degree = 1;
        string tmp;
        stringstream ss;
        ss << preorder;
        while (getline(ss, tmp, ',')){
            degree -- ;         // consume one edge
            if (degree < 0) {
                return false;
            }
            if (tmp[0]!='#') {
                degree += 2;    // generate 2 edges
            }
        }
        return degree == 0;
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

> [!NOTE] **[LeetCode 2415. 反转二叉树的奇数层](https://leetcode.cn/problems/reverse-odd-levels-of-binary-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 直接用 queue 铁铁 heap MLE
> 
> 手写 queue 可过
> 
> 标准做法 DFS （类似 101 对称二叉树）

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
    const static int N = 1e5 + 10;
    
    TreeNode * q[N], * g[N];
    
    TreeNode* reverseOddLevels(TreeNode* root) {
        int hh = 0, tt = -1;
        q[ ++ tt] = root;
        int dep = 0;
        while (hh <= tt) {
            int cnt = tt - hh + 1;
            int tot = 0;
            for (int i = 0; i < cnt; ++ i ) {
                auto & u = q[hh ++ ];
                if (u->left)
                    g[tot ++ ] = u->left;
                if (u->right)
                    g[tot ++ ] = u->right;
            }
            dep ++ ;
            if (dep & 1) {
                for (int i = 0, j = tot - 1; i < j; ++ i , -- j )
                    swap(g[i]->val, g[j]->val);
            }
            hh = 0, tt = tot - 1;
            memcpy(q, g, sizeof g);
        }
        return root;
    }
};
```

##### **C++ dfs**

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
    // 要想到  101. 对称二叉树
    void dfs(TreeNode * l, TreeNode * r, int dep) {
        if (!l || !r)
            return;
        if (dep & 1)
            swap(l->val, r->val);
        dfs(l->left, r->right, dep + 1), dfs(l->right, r->left, dep + 1);
    }

    TreeNode* reverseOddLevels(TreeNode* root) {
        dfs(root->left, root->right, 1);
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

> [!NOTE] **[LeetCode 366. 寻找二叉树的叶子节点](https://leetcode.cn/problems/find-leaves-of-binary-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 实现 trick

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
    vector<vector<int>> res;

    int dfs(TreeNode * u) {
        if (!u)
            return -1;          // ATTENTION
        int l = dfs(u->left), r = dfs(u->right);
        int d = max(l, r) + 1;
        if (d >= res.size())    // ATTENTION
            res.push_back({});
        res[d].push_back(u->val);
        return d;
    }

    vector<vector<int>> findLeaves(TreeNode* root) {
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

## TODO 其他序列


