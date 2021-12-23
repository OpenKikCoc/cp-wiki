## 习题

### 重建二叉树

> [!NOTE] **[SwordOffer ]()**
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

## TODO 其他序列


