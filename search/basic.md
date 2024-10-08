## 习题

> [!WARNING] **一般指 dfs**

> [!NOTE] **【回溯】特别注意**
> 
> 回溯时需特别注意 **恢复现场** 的相关实现
> 
> 如下面 [LeetCode LCP 58. 积木拼接](https://leetcode.cn/problems/De4qBB/) 一题中使用全局变量执行修改与回溯
> 
> 但全局变量本身会在下一层递归中被再次修改，这样在上层递归中执行【恢复现场】时必然有问题。

### 构造

> [!NOTE] **[LeetCode 22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 首先思考递归的出口：
>
>    当 *左边括号的数量 == 右边括号数量 == n*：这种情况下的有效路径可以加入到 res 中。
>
> 2. 那么，如何保证加入到的路径是有效的呢？也就是递归的顺序。
>
>    1） 每次可以加入左括号的条件是：当前左括号的数目不超过  n 
>
>    2） 每次可以加入右括号的条件是：当前右括号的数目不超过  n，并且右括号的数目不能超过左括号的数目。（这样才能保证生成的括号是有效的）
>
> **时间复杂度**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:

    vector<string> ans;

    vector<string> generateParenthesis(int n) {
        dfs(n, 0, 0, "");
        return ans;
    }

    void dfs(int n, int lc, int rc, string seq) {
        if (lc == n && rc == n) ans.push_back(seq);
        else {
            if (lc < n) dfs(n, lc + 1, rc, seq + '(');
            if (rc < n && lc > rc) dfs(n, lc, rc + 1, seq + ')');
        }
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    vector<string> res;
    void dfs(int n, int l, int r, string now) {
        if (l == n && r == n) {
            res.push_back(now);
            return;
        }
        if (l > r) {
            string next = now;
            next.push_back(')');
            dfs(n, l, r + 1, next);
        }
        if(l < n) {
            string next = now;
            next.push_back('(');
            dfs(n, l + 1, r, next);
        }
    }
    vector<string> generateParenthesis(int n) {
        dfs(n, 0, 0, "");
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        # lc:当前左括号数，rc:当前右括号数，path:当前的序列
        def dfs(lc, rc, path):
            if lc == rc == n:
                res.append(path)
                return 
            if lc < n:
                dfs(lc + 1, rc, path + '(')
            if rc < n and lc > rc:
                dfs(lc, rc + 1, path + ')')
                
        dfs(0, 0, "")
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 95. 不同的二叉搜索树 II](https://leetcode.cn/problems/unique-binary-search-trees-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 暴搜：递归所有方案
>
> 1. 对于每段连续的序列 l, l + 1, ..., l + k, ... , r, 枚举二叉搜索树根节点 root 的位置 i 
>
> 2. 分别递归求出左右子树的所有方案：
>
>    1） 对于左子树而言，节点的数值范围是：[l, i - 1]
>
>    2） 对于右子树而言，节点的数值范围是：[i + 1, r]
>
> 3. 左子树的任意一种方案和右子树的任意一种方案拼在一起，就可以得到当前节点的一种方案（因为根据步骤2求出的左右子树，始终都满足 **左子树 < 根节点 < 右子树**），所以将左右子树的所有方案两两组和，并记录到答案中
>
> **时间复杂度**

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
    vector<TreeNode*> helper(int s, int t) {
        vector<TreeNode*> ret;
        if (s > t) {
            ret.push_back(nullptr);
            return ret;
        }
        for (int i = s; i <= t; ++ i ) {
            auto ls = helper(s, i - 1);
            auto rs = helper(i + 1, t);
            for (auto l : ls) for (auto r : rs) {
                TreeNode * n = new TreeNode(i);
                n->left = l, n->right = r;
                ret.push_back(n);
            }
        }
        return ret;
    }
    vector<TreeNode*> generateTrees(int n) {
        vector<TreeNode*> res;
        if (!n) return res;
        res = helper(1, n);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if not n:return []

        def dfs(l, r):
            res = []   # 踩坑1: res 一定要放在递归函数里，每次进来都是空列表
            if l > r:return [None]
            for i in range(l, r + 1): # 遍历根节点的位置，每个数值都可能成为根节点
                left = dfs(l, i - 1)
                right = dfs(i + 1, r)
                for j in left:
                    for k in right:
                        root = TreeNode(i)  # 踩坑2: 每次都要生成一个新的 root 节点树
                        root.left, root.right = j, k
                        res.append(root)
            return res

        return dfs(1, n)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> - 区间 dp
>
> 1. 状态表示：f[n] 表示 n 个节点的二叉搜索树共有多少种。
> 2. 左子树可以有 0,1,…, n−1 个节点，对应的右子树有 n−1, n−2, …, 0 个节点，f[n] 是所有这些情况的加和；f[n]=∑n−1k=0 f[k]∗f[n−1−k]
>
> **时间复杂度**：状态总共有 n 个，状态转移的复杂度是 *O(n)*  所以总时间复杂度是 *O(n^2)*

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int numTrees(int n) {
        long c = 1;
        // 卡特兰数
        for(int i = 0; i < n; i++)
            c = c * 2 * (2 * i  + 1) /(i + 2);
        return c;
    }
  
    int numTrees(int n) {
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;
        
        for (int i = 2; i <= n; i ++ )         // i 为长度
            for (int j = 1; j <= i; j ++ )     // 以 j 为根
                dp[i] += dp[j - 1] * dp[i - j];
        return dp[n];
    }
  
    // 0s 100%, 6.4MB 100%
    int numTrees(int n) {
        vector<vector<int>> dp(n, vector<int>(n));
        for(int i = 0; i < n; ++ i ) dp[i][i] = 1;
        // 区间dp 
        for (int i = n - 1; i >= 0; -- i ) {
            for (int j = i + 1; j < n; ++ j ) {
                // i ~ j 可以组成多少个二叉搜索树
                // 以 k 为根
                for (int k = i; k <= j; ++ k ) {
                    dp[i][j] += (k > i ? dp[i][k - 1] : 1) * (k < j ? dp[k + 1][j] : 1);
                }

            }
        }
        return dp[0][n - 1];
    }
};
```

##### **Python**

```python
class Solution:
    def numTrees(self, n: int) -> int:
        f = [0] * (n + 1)
        f[0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                f[i] += f[j - 1] * f[i - j]
        return f[-1]
```

<!-- tabs:end -->
</details>

<br>

* * *

### 子集

> [!NOTE] **[LeetCode 78. 子集](https://leetcode.cn/problems/subsets/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. DFS：子集问题，是属于 **不区分顺序**
> 2. 对于这类不区分顺序的题，不用额外用一个数组标记是否被用过，可以直接从前到后遍历一遍就可以保证不会被重复使用。
> 3. 全排列这类题都是区分顺序的题，是需要一个额外的数组进行标记状态的。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> ns;
    int n;
    vector<vector<int>> res;
    vector<int> t;

    void dfs(int u) {
        if (u == ns.size()) {
            res.push_back(t);
            return;
        }
        dfs(u + 1);
        t.push_back(ns[u]);
        dfs(u + 1);
        t.pop_back();
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        this->ns = nums;
        this->n = ns.size();
        dfs(0);
        return res;
    }
};
```

##### **C++ 状压枚举**

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> res;
        for (int i = 0; i < 1 << n; ++ i ) {
            vector<int> t;
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    t.push_back(nums[j]);
            res.push_back(t);
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if not nums:return []
        n = len(nums)
        res = []
        def dfs(path, idx):
            res.append(path[:])
            for k in range(idx, n):
                path.append(nums[k])
                dfs(path, k + 1)
                path.pop()

        dfs([], 0)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 90. 子集 II](https://leetcode.cn/problems/subsets-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 这道题和上道题的区别在于，原数组可能会存在重复数字，需要进行判重：
>
> 1. 排序
> 2. 每次在进入递归的时候，把当前数字和上一个数字进行对比。（详细见代码）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> t;
    void dfs(vector<int>& nums, int step) {
        res.push_back(t);
        for (int i = step; i < nums.size(); ++ i ) {
            if (i > step && nums[i] == nums[i - 1]) continue;
            t.push_back(nums[i]);
            dfs(nums, i + 1);
            t.pop_back();
        }
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        dfs(nums, 0);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        num.sort()  # 踩坑1: 忘记排序！！
        n = len(nums)
        res = []

        def dfs(path, i):
            res.append(path[:])
            for k in range(i, n):
                if k > i and nums[k] == nums[k-1]:
                  	continue # 踩坑2: k > i（missed）
                path.append(nums[k])
                dfs(path, k + 1)
                path.pop()
        
        dfs([], 0)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 搜索 剪枝 重复
>
> 1. 先求出数组的总和，如果总和不是 k 的倍数，则直接返回 false。
> 2. 求出子集和的目标值后，如果数组中有数字大于这个子集和目标值，则返回 false。
> 3. 先去搜索第一组，再搜索第二组，以此类推。那么，组内如何搜索呢？很明显，组内的顺序是不影响总和的，所以组内是按照顺序来搜的，相对顺序不变来进行搜索的。
> 3. 第一个优化点：【将数组 从大到小 排序】。贪心地来看，先安排大的数字会更容易先找到答案，这是因为小的数字再后期更加灵活。
> 3. 第二个剪枝优化点：如果当前搜索的数 nums[i] == nums[i-1], 并且用 nums[i] 失败了，那么用 nums[i-1] 也一定失败。
> 3. 第三个剪枝：如果当前组第一个数失败了，那么后面一定无解。（trick 认真思考）
> 3. 第四个剪枝：如果当前组最后一个数失败了，那么也一定失败。（trick 认真思考）
> 4. 我们按照子集去递归填充，即维护一个当前总和 cur，如果当前值等于了目标值，则进行下一个子集的填充。最终，如果只剩下一个子集尚未填充（无需 0 个），则可以直接返回 true。
> 5. 还需要同时维护一个 start，即表示当前这个子集，需要从数组的哪个位置开始尝试。
> 6. 这里还有一个非常重要的策略，就是如果当前总和 cur 为 0 时，我们直接指定第一个未被使用的数字作为当前这个子集的第一个数字（最大的数字）。这是防止重复枚举，因为第一个未被使用的数字一定要出现在某个子集中。否则，如果我们最终没有使用这个数字，在尝试下一个集合时，还会重复尝试使用这个数字，造成大量重复计算。（相当于是第三 第四个剪枝）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int len;
    vector<int> nums;
    vector<bool> st;

    bool dfs(int start, int cur, int k) {
        if (!k) return true;
        if (cur == len) return dfs(0, 0, k - 1);
        for (int i = start; i < nums.size(); ++ i ) {
            if (st[i]) continue;
            if (cur + nums[i] <= len) {
                st[i] = true;
                if (dfs(i + 1, cur + nums[i], k)) return true;
                st[i] = false;
            }
            while (i + 1 < nums.size() && nums[i + 1] == nums[i]) i ++ ;
            // 重要剪枝
            if (!cur || cur + nums[i] == len) return false;
        }
        return false;
    }

    bool canPartitionKSubsets(vector<int>& _nums, int k) {
        nums = _nums;
        st.resize(nums.size());
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % k) return false;
        len = sum / k;
        sort(nums.begin(), nums.end(), greater<int>());
        return dfs(0, 0, k);
    }
};
```

##### **Python**

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        def dfs(start, cur, k):
            if k == 0:
                return True
            if cur == avg:
                return dfs(0, 0, k - 1)
            for i in range(start, len(nums)):
                if st[i]:continue
                if cur + nums[i] <= avg:
                    st[i] = True
                    if dfs(i + 1, cur + nums[i], k):
                        return True 
                    st[i] = False
                # 对应第二个剪枝
                while i + 1 < len(nums) and nums[i + 1] == nums[i]:
                    i += 1
                # 对应第三个 第四个剪枝
                if not cur or (cur + nums[i] == avg):
                    return False

        nums.sort(reverse = True)
        st = [False] * 20
        sumn = sum(nums)
        if sumn % k != 0:
            return False
        avg = sumn // k
        return dfs(0, 0, k)
```

<!-- tabs:end -->
</details>

<br>

* * *


### 排列

> [!NOTE] **[LeetCode 46. 全排列](https://leetcode.cn/problems/permutations/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. DFS + 回溯
>
> 2. 用一个数组标记当前数字是否被用过
> 3. 递归出口：len(path) == n

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> ns;
    int n;

    vector<vector<int>> res;
    vector<int> t;
    vector<bool> st;

    void dfs(int u) {
        if (u == n) {
            res.push_back(t);
            return;
        }

        for (int i = 0; i < n; ++ i ) {
            if (st[i])
                continue;
            st[i] = true;
            t.push_back(ns[i]);
            dfs(u + 1);
            t.pop_back();
            st[i] = false;
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        this->ns = nums;
        this->n = ns.size();
        st = vector<bool>(n);
        
        dfs(0);

        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        used = [False] * n 

        def dfs(path):
            if len(path) == n:
                res.append(path[:])
                return 
            for i in range(n):
                if not used[i]:
                    used[i] = True
                    path.append(nums[i])
                    dfs(path)
                    path.pop()
                    used[i] = False  # 回溯：还原现场
        
        dfs([])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 可能存在重复数字，需要判重。
>
> 1. 排序
> 2. 每次在进入递归的时候，把当前数字和上一个数字进行对比。
> 3. 详细见代码，需要注意的地方：`!vis[i - 1]`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> nums;
    vector<bool> vis;
    vector<int> t;
    void dfs(int pos) {
        if (pos == nums.size()) {
            res.push_back(t);
            return;
        }
        for (int i = 0; i < nums.size(); ++ i ) {
            if (vis[i]) continue;
            if (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1]) continue; // ATTENTION
            vis[i] = true;
            t.push_back(nums[i]);
            dfs(pos + 1);
            t.pop_back();
            vis[i] = false;
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        this->nums = nums;
        this->vis = vector<bool>(nums.size());
        dfs(0);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        if not nums:return []
        n = len(nums)
        res = []
        nums.sort()   # 踩坑1: 一定要先排序！
        used = [False] * n

        def dfs(path):
            if len(path) == n:
                res.append(path[:])
                return 
            for i in range(n):
                if not used[i]:
                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:continue # 踩坑2: not used[i-1] 和 used[i-1] 都可以，但是 not used[i-1] 在常数级别下会更快!!! 
                    used[i] = True
                    path.append(nums[i])
                    dfs(path)
                    used[i] = False 
                    path.pop()
        dfs([])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> DFS：从左到右枚举；
>
> 注意写法 无需 `for` 以及 `^32的trick`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    string s;
    vector<string> res;

    void dfs(int p) {
        if (p == n) {
            res.push_back(s);
            return;
        }
        dfs(p + 1);
        if (!isdigit(s[p])) {
            s[p] ^= 32;
            dfs(p + 1);
            s[p] ^= 32;
        }
    }

    vector<string> letterCasePermutation(string s) {
        n = s.size();
        this->s = s;
        dfs(0);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        res = []
        def dfs(path, s):
            if not s:
                res.append(path)
                return #一定要有 return, 不然会继续执行下一行，导致list out of range
            c = s[0]
            if c.isalpha():
                dfs(path + c.lower(), s[1:])
                dfs(path + c.upper(), s[1:])
            else:
                dfs(path + c, s[1:])
        
        # 可以直接用字符串这个数据结构；也可以用 list，会比字符串效率要高
        dfs('', s)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

### 组合

> [!NOTE] **[LeetCode 77. 组合](https://leetcode.cn/problems/combinations/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 这道题其实是子集类型的题。实际上是求由 1 ~ n 的数字组成集合的子集，并且子集的个数刚好是 k 的所有子集方案。
>
> 从 1 到 n 依次枚举加入（递增加入，不需要判重），当路径里的个数等于 k 时，就可以把当前路径加入到 res 中（记得回溯）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, k;
    vector<vector<int>> res;
    vector<int> t;
    void dfs(int step) {
        if (t.size() == k) {
            res.push_back(t);
            return;
        }
        for (int i = step; i <= n; ++ i ) {
            t.push_back(i);
            dfs(i + 1);
            t.pop_back();
        }
    }
    vector<vector<int>> combine(int n, int k) {
        this->n = n;
        this->k = k;
        dfs(1);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        if not n:return []
        res =[]

        def dfs(path, i):
            if len(path) == k:
                res.append(path[:])
                return 
            for u in range(i, n + 1):
                # 踩坑：需要在循环里把当前数加入到 path 中，否则组合就是必须从第一个数字开始组合
                path.append(u)   
                dfs(path, u + 1)
                path.pop()  

        dfs([], 1)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 39. 组合总和](https://leetcode.cn/problems/combination-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 先对原数组排序，方便后续剪枝（当  nums[k] > target的时，不符合条件，可以直接 return）
> 2. 从前往后枚举数组，由于可以重复添加，所以每次递归都可以从当前数添加，同时将当前 target 值减 nums[i]
> 3. 递归的出口是：target == 0 时候，递归结束；还有一个是当 nums[i] > target，也直接 return
> 4. 【注意】避免重复：对于这一类可以以 **任意顺序**  返回结果（类似于集合不计顺序）的问题，在搜索的时候就需要 **按某种顺序搜索**，才可以避免重复。具体的做法是：只要限制下一次选择的起点，是基于本次的选择，这样下一次就不会选到本次选择同层左边的数。也就是：通过控制 for 遍历的起点，去掉会产生重复组合的选项。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> t;
    void dfs(vector<int>& c, int p, int tar) {
        if (!tar) {
            res.push_back(t);
            return;
        }
        for (int i = p; i < c.size(); ++ i ) {
            if (c[i] > tar) return;
            t.push_back(c[i]);
            dfs(c, i, tar - c[i]);
            t.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        dfs(candidates, 0, target);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        if not nums:return []
        n = len(nums)
        # !! 要先排序
        nums.sort()
        res = []

        def dfs(path, target, i):
            if target == 0:
                res.append(path[:])
                return 
            # 这里的顺序 k 开始搜索，可以保证每个数在其他结果中不会被重复用 
            for k in range(i, n):  
                if nums[k] > target:return  # 剪枝的过程
                path.append(nums[k])
                dfs(path, target - nums[k], k)
                path.pop()

        dfs([], target, 0)
        return res
```

**Python 答案重复**

```python
# 这样的写法 res: [[2,2,3],[2,3,2],[3,2,2],[7]]
# 正确答案 res：[[2,2,3],[7]]
class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []
        n = len(nums)
        nums.sort()

        def dfs(path, target):
            if target == 0:
                res.append(path[:])
                return 
            for i in range(n):
                if nums[i] > target:return
                path.append(nums[i])
                dfs(path, target - nums[i])
                path.pop()
        
        dfs([], target)
        return res
```

**Python 错误答案**

```python
"""
以下 dfs 对于这道题来说是错误的，这个代码的意思是，不管怎么样，一定要把第一个元素加入到路径中
这种代码是类似求 【二叉树的 根 到 叶子结点的路径】的题
"""      

  def dfs(idx, path, target):
    	if nums[idx] > target:
        	return 
      path.append(nums[idx])
      target -= nums[idx]
			if target == 0:
        	res.append(path[:])
          return 
      for k in range(idx, n):
          dfs(k, path, target)
          path.pop()
      dfs(0, [], target)
      return res
```

 

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 和  **[LeetCode 39. 组合总和](https://leetcode.cn/problems/combination-sum/)** 思路类似，不同点在于：
>
> 1. 递归出口规则稍微有点不同：这道题要求每种组合的个数 == k
> 2. 这道题不允许有重复数字，所以在进行本层递归的下一次添加数字，要从 u + 1 开始

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, k;
    vector<vector<int>> res;
    vector<int> t;
    void dfs(int pos, int tar) {
        if (!tar) {
            if (t.size() == k) res.push_back(t);
            return;
        }
        for (int i = pos; i <= 9; ++ i ) {
            t.push_back(i);
            dfs(i + 1, tar - i);
            t.pop_back();
        }
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        this->n = n; this->k = k;
        dfs(1, n);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        if n == 0:return []
        res = []

        def dfs(path, n, i):
            if n == 0 and len(path) == k:  # 踩坑1: 别忘了【个数为 k 】这个条件！！
                res.append(path[:])
                return 
            for u in range(i, 10):  # 踩坑2：这里遍历的尾部 不能写 n, 因为 n 在每次递归的过程中都发生了变化
                if u > n:return  # 剪枝
                path.append(u)
                dfs(path, n - u, u + 1) 
                path.pop()
        dfs([], n, 1)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 本题用 DFS 会超时。(背包问题一般都是不考虑顺序的) 但这道题不考虑顺序，所以不能用背包问题的思路来解决。
>
> 1. 状态表示：f[i] 表示总和为 i 的所有划分方案的集合(有多少个数字无所谓)；属性：数量
>
> 2. 状态转移：dp 一般是把一个集合划分为若干个子集。
>
>    - 划分方式：以最后一个数的数值为划分转移，最后一个数可以是[nums[1],num[2],nums[3],..., nums[k],...,nums[n]]（这样划分方式，是不重不漏）
>
>    - 当最后一个数都是nums[k], 那不同的划分方案数量取决于除去最后一个数 nums[k] 后， 前面那些数的划分方案数量，相当于是求解 i - nums[k] 的划分数量。所以表达式为：
>
>       f[i] = f[i-nums[k]]
>
> 3. 初始化：当 target == 0，只有一种划分方案，就是空集。所以f[0] = 1
>
> 4. 由于nums[k] > 0, 所以从小到大循环遍历就可以算出来了。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        int n = nums.size();
        vector<long long> f(target + 1);
        f[0] = 1;
        for (int i = 0; i <= target; ++ i )
            for (auto v : nums)
                if (i >= v) f[i] = (f[i] + f[i - v]) % INT_MAX;
        return f[target];
    }
};
```

##### **C++ 错误定义**

```cpp
// 以下状态定义是错误的
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        int n = nums.size();
        vector<long long> f(target + 1);
        f[0] = 1;
        for (auto v : nums)
            for (int i = target; i >= v; -- i )
                f[i] = (f[i] + f[i - v]) % INT_MAX;
        return f[target];
    }
};
```


##### **Python**

```python
#法一：用dfs，会超时
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        self.res = 0
        res = []
        n = len(nums)
        nums.sort()

        def dfs(path, target):
            if target == 0:
                res.append(path[:])
                self.res += 1
                return
            # 由于不同顺序也算不同的答案，所以这里不需要用 k 来遍历保证无重复
            for i in range(n):
                if nums[i] > target:
                    return 
                path.append(nums[i])
                dfs(path, target - nums[i])
        
        dfs([], target)
        print(res)
        return self.res
```

**Python dp**

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0 or target <= 0:return 0
        f = [0 for _ in range(target + 1)]
        f[0] = 1  # 初始化
        for i in range(1, target + 1):
            for x in nums:
                if i >= x:
                    f[i] += f[i - x]
        return f[target]
```



<!-- tabs:end -->
</details>

<br>

* * *

### dfs 递归

> [!NOTE] **[LeetCode 112. 路径总和](https://leetcode.cn/problems/path-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 递归，自顶向下从根节点往叶节点走，每走过一个节点，就让 sum 减去该节点的值，则如果走到某个叶节点时，sum 恰好为0，则说明从根节点到这个叶节点的路径上的数的和等于 sum。
> 只要找到一条满足要求的路径，递归即可返回。
>
> **时间复杂度**
>
> 每个节点仅被遍历一次，且递归过程中维护 sum 的时间复杂度是 O(1)，所以总时间复杂度是 O(n)。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if (!root) return false;
        if (!root->left && !root->right) return root->val == sum;
        return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
    }
};
```

##### **Python**

```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:return False  
        if not root.left and not root.right:return root.val == sum 
        return self.hasPathSum(root.left, sum - root.val) or  self.hasPathSum(root.right, sum - root.val)   
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 这道题目和  **[LeetCode 112. 路径总和](https://leetcode.cn/problems/path-sum/)** 基本一样，不同点在于需要把**所有路径**记录下来。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> t;
    void dfs(TreeNode* root, int sum) {
        if (!root) return;
        t.push_back(root->val);
        if (!root->left && !root->right)
            if (root->val == sum)
                res.push_back(t);
        dfs(root->left, sum - root->val);
        dfs(root->right, sum - root->val);
        t.pop_back();
    }
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        dfs(root, sum);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        res = []

        def dfs(p, path, sum):
            if not p:return 
            path.append(p.val)
            sum -= p.val
            if not p.left and not p.right and sum == 0:
                res.append(path[:])
                # return   踩坑：这里不能写return。它和dfs在同一层级，不能return 
            dfs(p.left, path, sum)
            dfs(p.right, path, sum)
            path.pop()

        dfs(root, [], sum)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 递归遍历整棵树，递归时维护从每个节点开始往下延伸的最大路径和
>
> 2. 对于每个点，递归计算其左右子树后，我们将左右子树维护的两条最大路径，和当前结点的值加起来，就可以得到当前点的最大路径和
> 3. 计算出所有点的最大路径和，求个最大的返回即可。
>
> **时间复杂度**：每个节点仅会遍历一次，所以时间复杂度是 O(n)。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int res;
    int dfs(TreeNode* root) {
        if (!root) return 0;
        int l = max(0, dfs(root->left)), r = max(0, dfs(root->right));
        res = max(res, l + r + root->val);
        return max(l, r) + root->val;
    }
    int maxPathSum(TreeNode* root) {
        res = INT_MIN;
        dfs(root);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.res = float('-inf')

        def dfs(root):
            if not root:return 0
            left = max(0, dfs(root.left))
            right = max(0, dfs(root.right))
            self.res = max(self.res, left + right + root.val)
            return max(left, right) + root.val

        dfs(root)
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 386. 字典序排数](https://leetcode.cn/problems/lexicographical-numbers/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> （暴力做法）最坏的做法就是把所有的数全部转化为 string，再对 string 排序。
>
> Trie 树：一般字符串用 Trie 排序，时间复杂度可以少 log(n)
>
> 1. 用一个 Trie 树表示所有数字，如果当前数是在 1～n 范围内，就插入答案中（由于表示了所有数，所以没必要真的模拟一个 Trie 数并插入）
> 2. 在 Trie 中搜索，遍历的顺序按照：根，0子树，1子树，2子树...，就可以保证当前字典序是递增



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> res;

    vector<int> lexicalOrder(int n) {
        for (int i = 1; i <= 9; i ++ ) dfs(i, n);
        return res;
    }

    void dfs(int cur, int n) {
        if (cur <= n) res.push_back(cur);
        else return;
        for (int i = 0; i <= 9; i ++ ) dfs(cur * 10 + i, n);
    }
};


class Solution {
public:
    void dfs(int k, int n, vector<int>& res) {
        if (k > n) return;
        if (k != 0) res.push_back(k);
        for (int i = 0; i <= 9; ++ i ) {
            if (10 * k + i > 0) {
                dfs(10 * k + i, n, res);
            }
        }
    }
    vector<int> lexicalOrder(int n) {
        vector<int> res;
        dfs(0, n, res);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        res = []

        def dfs(u):
            if u > n:
                return
            res.append(u)
            for k in range(u * 10, u * 10 + 10):
                dfs(k)

        for i in range(1, 10):
            dfs(i)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1215. 步进数](https://leetcode.cn/problems/stepping-numbers/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> - 从1~9开始向下深度优先搜索所有小于等于high的步进数
>
> 搜索生成所有步进数即可，注意排除 `v + 1 = 10` 这类以及负数
>
> 也即: `if (v + 1 <= 9)` 和 `if (v - 1 >= 0)`
>
> 也有 bfs 构造，略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    typedef long long LL;
    vector<LL> t;
    void dfs(int v, LL s, int p) {
        if (p > 9) {
            return;
        }
        t.push_back(s * 10 + v);
        if (v + 1 <= 9) dfs(v + 1, s * 10 + v, p + 1);
        if (v - 1 >= 0) dfs(v - 1, s * 10 + v, p + 1);
    }
    void get() {
        for (int i = 0; i < 10; ++ i )
            dfs(i, 0, 0);
        sort(t.begin(), t.end());
        t.erase(unique(t.begin(), t.end()), t.end());
        //for (auto v : t) cout << v << endl;
    }
    vector<int> countSteppingNumbers(int low, int high) {
        get();
        vector<int> res;
        for (auto v : t)
            if (v >= low && v <= high)
                res.push_back(v);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def countSteppingNumbers(self, low: int, high: int) -> List[int]:
        res, ans = [], []

        # last 表示上一个数的个位数字的数值，num 是本轮的数值
        def dfs(last, num):
            if num > high:return
            res.append(num)
            if last != 9:
                dfs(last + 1, num * 10 + last + 1)
            if last != 0:
                dfs(last - 1, num * 10 + last - 1)

        if low <= 0:res.append(low)
        for i in range(1, 10):
            dfs(i, i)
        for c in res:
            if c >= low:
                ans.append(c)
        ans.sort()
        return ans
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1273. 删除树节点](https://leetcode.cn/problems/delete-tree-nodes/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> DFS 先递归搜索它的所有子节点，然后再加上当前节点的值，去判断是否为0
>
> （这是类似于后续遍历）
>
> 有种更优雅的写法
>
> > 之前有一种：该写法在子树被切掉时影响父节点的忧虑
> > 
> > 其实不必的，子树被切掉其和必然为0，直接传回 {0, 0} 即可，对父的求和无影响

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> sons;
    vector<int> vals;

    pair<int, int> dfs(int x) {
        int size = 1, sum = vals[x];
        for (auto y : sons[x]) {
            auto [sz, val] = dfs(y);
            size += sz, sum += val;
        }
        if (sum == 0) return {0, 0};
        return {size, sum};
    }

    int deleteTreeNodes(int nodes, vector<int>& parent, vector<int>& value) {
        sons.clear();
        sons.resize(nodes);
        vals = value;
        for (int i = 0; i < nodes; ++i) {
            if (parent[i] == -1) continue;
            sons[parent[i]].push_back(i);
        }
        int ans = dfs(0).first;
        return ans;
    }
};
```

##### **Python**

```python
class Solution:
    def deleteTreeNodes(self, nodes: int, parent: List[int], value: List[int]) -> int:
        def dfs(i):
            val = value[i]
            cnts = 1
            for v in edges[i]:
                vj, cj = dfs(v)
                val += vj
                cnts += cj
            if val == 0:
                return 0, 0 
            else:
                return val, cnts

        edges = collections.defaultdict(list)
        for k, v in enumerate(parent):
            if v != -1:
                edges[v].append(k)
        return dfs(0)[1]
      
      
class Solution:
    def deleteTreeNodes(self, nodes: int, parent: List[int], value: List[int]) -> int:
        res = [1] * nodes
        def dfs(u):
            for v in edges[u]:
                dfs(v)
                value[u] += value[v]
                res[u] += res[v]
            if value[u] == 0:
                res[u] = 0

        edges = collections.defaultdict(list)
        for k, v in enumerate(parent):
            if v != -1:
                edges[v].append(k)
        dfs(0)
        return res[0]
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1315. 祖父节点值为偶数的节点和](https://leetcode.cn/problems/sum-of-nodes-with-even-valued-grandparent/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 直接记录父节点和祖父节点的值 对于root其父和祖父均为**任意奇数**（这里使用-1）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int res;
    void dfs(TreeNode* r, int f, int ff) {
        // if(!r) return;
        if (ff % 2 == 0) res += r->val;
        if (r->left) dfs(r->left, r->val, f);
        if (r->right) dfs(r->right, r->val, f);
    }
    int sumEvenGrandparent(TreeNode* root) {
        res = 0;
        dfs(root, -1, -1);
        return res;
    }
};
```

##### **Python - DFS**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        self.res = 0

        def dfs(r, f, ff):
            if ff % 2 == 0:
                self.res += r.val
            if r.left:
                dfs(r.left, r.val, f)
            if r.right:
                dfs(r.right, r.val, f)
        
        dfs(root, -1, -1)
        return self.res
```

##### **Python - BFS**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        q = collections.deque()
        q.append(root)
        res = 0
        while q:
            node = q.popleft()
            if node.val % 2 == 0:
                if node.left:
                    if node.left.left:
                        res += node.left.left.val 
                    if node.left.right:
                        res += node.left.right.val 
                if node.right:
                    if node.right.left:
                        res += node.right.left.val 
                    if node.right.right:
                        res += node.right.right.val
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        return res
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1415. 长度为 n 的开心字符串中字典序第 k 小的字符串](https://leetcode.cn/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 只有abc组成的字符串 排列 递归
> 
> TODO: 更优雅的类似数位 dp 解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int tot;
    string path, res;
    void dfs(int c, int left, int k) {
        path.push_back('a' + c);
        if (left == 0) {
            ++tot;
            if (tot == k) res = path;
        }

        for (int i = 0; i < 3; ++i) {
            if (i == c) continue;
            if (left - 1 >= 0) dfs(i, left - 1, k);
        }
        path.pop_back();
    }

    string getHappyString(int n, int k) {
        tot = 0;
        dfs(0, n - 1, k);
        if (tot >= k) return res;

        dfs(1, n - 1, k);
        if (tot >= k) return res;

        dfs(2, n - 1, k);
        if (tot >= k) return res;

        if (tot < k) return "";
        return res;
    }
```

##### **Python**

```python
class Solution:
    def getHappyString(self, n: int, k: int) -> str:
        self.res = ""
        self.cnt = 0

        def dfs(n, k, cur):
            for x in ['a', 'b', 'c']:
                if cur and x == cur[-1]:
                    continue
                if len(cur + x) == n:
                    self.cnt += 1
                    if self.cnt == k:
                        self.res = cur + x
                        break
                elif self.res == "":
                    dfs(n, k, cur + x)

        dfs(n, k, "")
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1530. 好叶子节点对的数量](https://leetcode.cn/problems/number-of-good-leaf-nodes-pairs/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 记一下这种写法
>
>  
>
> 1. 对于树这类的题目，绝大多数是可以使用递归来做的: dfs 或者 bfs
> 2. 根据题意，只有深度有用。需要记录的就是每一个节点的叶子节点到该点的距离。(不需要管 root.val 的值)
> 3. 对于每一个子节点，都需要更新self.res的值
> 4. 对于 dfs, 返回的是：某个节点的所有叶子结点到该节点的距离（以 list 形式返回）
> 5. 需要注意的是，这看上去是个后序遍历，但其实和后续遍历没有关系，因为先dfs递归右子树也完全可以AC





<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countPairs(TreeNode *root, int distance) {
        int res = 0;
        function<vector<int>(TreeNode *)> dfs = [&](TreeNode *p) {
            if (!p->left && !p->right) return vector<int>(1, 0);
            vector<int> ls, rs;
            if (p->left) ls = dfs(p->left);    // 左子树叶子节点距离
            if (p->right) rs = dfs(p->right);  // 右子树叶子节点距离
            for (int &l : ls) ++l;
            for (int &r : rs) ++r;
            for (const int &l : ls)
                for (const int &r : rs) {
                    if (l + r <= distance) ++res;
                }
            ls.insert(ls.end(), rs.begin(), rs.end());
            // for(auto v : rs) ls.push_back(v);
            return ls;
        };
        dfs(root);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        self.res = 0

        def dis(root):
            if not root:
                return []
            if not root.left and not root.right:
                return [1]
            l, r = dis(root.left), dis(root.right)
            for i in l:
                for j in r:
                    if i + j <= distance:
                        self.res += 1
            depth = [i + 1 for i in l + r]  # 记录该点到所有叶子结点的距离
            return depth
        dis(root)
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2014. 重复 K 次的最长子序列](https://leetcode.cn/problems/longest-subsequence-repeated-k-times/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 较显然的：最好预先处理原串 生成所有字符都有可能出现的新串
>
> > 容易想到由预处理后的串生成子串（预处理后的串的每个字符都可以出现无限多次）
> >
> > 显然是一个递归的流程 递归参数即 【还需要增加的字符长度, 当前构造的串】
> >
> > 初始化 depth = n / k
> >
> > 且统计答案显然应放在每层 dfs 中而非出口（因为显然有结果子串的长度 <= n / k）
> >
> > 这样一来 中间会出现大量的重复计算流程 **显然需要记忆化**
>
> 题意清晰 重复练习即可
>
> 主要学习【trick check】【nxt构造】【分析构造思维】
>
> **重复重复反复做 TOOD**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 习惯写法: 字符串暴力匹配(可以更trick) + 记忆化搜索**

TLE 几次 其实加个记忆化就过

学习 trick 的 check 函数

```cpp
class Solution {
public:
    int n, k;
    string s;
    unordered_map<char, int> hash;
    
    string update(string s) {
        for (auto c : s)
            hash[c] ++ ;
        string ret;
        for (auto c : s)
            if (hash[c] >= k)
                ret.push_back(c);
        return ret;
    }
    
    vector<char> chs;
    string res;

    // 值得学习
    // Trick 的 check 写法
    bool check(string t) {
        if (t.empty())
            return true;
        int d = t.size(), p = 0;
        for (auto c : s)
            if (c == t[p % d])
                p ++ ;
        return p >= k * d;
    }
    // 习惯的写法
    bool check(string t) {
        string tt;
        for (int i = 0; i < k; ++ i )
            tt += t;
        int m = tt.size(), p = 0;
        for (auto c : s)
            if (p < m && c == tt[p])
                p ++ ;
            else if (p == m)
                return true;
        return p == m;
    }

    // The most important thing is `how to design the DFS func`
    set<pair<int, string>> S;
    void dfs(int depth, string now) {
        // Do this firstly
        if (S.count({depth, now}))
            return;
        S.insert({depth, now});
        
        if (!check(now))
            return;
        if (now.size() > res.size() || now.size() == res.size() && now > res)
            res = now;
        
        if (!depth)
            return;
        
        for (auto c : s)
            if (hash[c] >= k) {
                hash[c] -= k;
                dfs(depth - 1, now + c);
                hash[c] += k;
            }
    }
    
    string longestSubsequenceRepeatedK(string s, int k) {
        this->n = s.size(), this->k = k;
        this->s = update(s);
        
        chs.clear();
        for (auto [c, v] : hash)
            chs.push_back(c);
        sort(chs.begin(), chs.end());
        reverse(chs.begin(), chs.end());    // for ordered result
        
        // it's [n/k] not [chs.size()]
        dfs(n / k, "");
        
        return res;
    }
};
```

##### **C++ 无需预处理原串 构造 next 数组加速字符串匹配(非常常见) + 记忆化**

学习 nxt 构造的思想

```cpp
class Solution {
public:
    int n, k;
    string s;
    unordered_map<char, int> hash;
    
    string update(string s) {
        // ...
    }
    
    // Data structure
    // TLE
    // vector<vector<char, int>> nxt;
    int nxt[16010][26];
    
    bool check(string t) {
        // Change the meaning of `p`
        int p = 0;
        for (int _ = 0; _ < k; ++ _ ) {
            for (auto c : t) {
                if (nxt[p][c - 'a'] >= n)
                    return false;
                p = nxt[p][c - 'a'] + 1;
            }
        }
        return true;
    }
    
    vector<char> chs;
    string res, tmp;

    // The most important thing is `how to design the DFS func`
    set<pair<int, string>> S;
    void dfs(int depth, string now) {
        if (S.count({depth, now}))
            return;
        S.insert({depth, now});
        
        if (!check(now))
            return;
        if (now.size() > res.size() || now.size() == res.size() && now > res)
            res = now;
        
        if (!depth)
            return;
        
        for (auto c : s)
            if (hash[c] >= k) {
                hash[c] -= k;
                dfs(depth - 1, now + c);
                hash[c] += k;
            }
    }
    
    string longestSubsequenceRepeatedK(string s, int k) {
        this->s = update(s);
        this->n = s.size(), this->k = k;
        
        chs.clear();
        for (auto [c, v] : hash)
            chs.push_back(c);
        // sort(chs.begin(), chs.end());
        // reverse(chs.begin(), chs.end());    // for ordered result
        
        // ATTENTION: 经典构造nxt数组来加速字符串匹配
        // nxt = vector<unordered_map<char, int>>(n + 1);
        for (auto c : chs)
            nxt[n][c - 'a'] = n;
        for (int i = n - 1; i >= 0; -- i ) {
            for (auto c : chs)
                nxt[i][c - 'a'] = nxt[i + 1][c - 'a'];
            nxt[i][s[i] - 'a'] = i;
        }
        
        // it's [n/k] not [chs.size()]
        dfs(n / k, "");
        
        return res;
    }
};
```

##### **C++ 由短串构造长串 思维**

根据题意 更长的合法串必然是由短一个长度的合法串新增字符而来

学习推导和构造的思维

```cpp
class Solution {
public:
    int n, k;
    string s;
    unordered_map<char, int> hash;
    
    string update(string s) {
        // ...
    }
    
    bool check(string t) {
        // ...
    }

    vector<char> chs;
    
    string longestSubsequenceRepeatedK(string s, int k) {
        this->n = s.size(), this->k = k;
        this->s = update(s);
        
        chs.clear();
        for (auto [c, v] : hash)
            chs.push_back(c);
        sort(chs.begin(), chs.end());
        reverse(chs.begin(), chs.end());    // for ordered result
        
        // Reason 1: we get the longer valid string step by step
        //           so iterate from 1 to chs.size()
        vector<vector<string>> mem(n / k + 1);
        mem[0].push_back("");   // empty string
        for (int len = 1; len <= n / k; ++ len )
            for (auto & pre : mem[len - 1])
                for (auto c : chs) {
                    string now = pre + c;
                    if (check(now))
                        mem[len].push_back(now);
                }
        
        // Reason 2: we get the result from chs.size() to 1
        for (int len = n / k; len >= 1; -- len )
            if (mem[len].size())
                return mem[len][0];
        
        return "";
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

> [!NOTE] **[LeetCode 2056. 棋盘上有效移动组合的数目](https://leetcode.cn/problems/number-of-valid-move-combinations-on-chessboard/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 重点在于理解题意
> 
> 爆搜即可
> 
> **值得学习的是简洁优雅的代码实现方式**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    vector<string> pc;
    vector<vector<int>> pt;
    
    int dx[8] = {-1, -1, 0, 1, 1, 1, 0, -1};    // trick
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    
    vector<vector<int>> path;
    int res = 0;
    int p[5][2];
    
    bool check() {
        // 静态数组单独存 加速访问
        for (int i = 0; i < n; ++ i )
            p[i][0] = pt[i][0], p[i][1] = pt[i][1];
        
        // 所有可能的时间
        for (int i = 1; ; ++ i ) {
            // ATTENTION: flag 是否有棋子没有走到目的地
            // 题意：【每一秒，每个棋子都沿着它们选择的方向往前移动 `一步` ，直到它们到达目标位置。】
            bool flag = false;
            for (int j = 0; j < n; ++ j ) {
                int d = path[j][0], t = path[j][1];
                // ATTENTION: <= 当前棋子还可以走
                if (i <= t) {
                    flag = true;
                    p[j][0] += dx[d], p[j][1] += dy[d];
                }
            }
            if (!flag)
                break;
            
            for (int j = 0; j < n; ++ j )
                for (int k = j + 1; k < n; ++ k )
                    if (p[j][0] == p[k][0] && p[j][1] == p[k][1])
                        return false;
        }
        return true;
    }
    
    void dfs(int u) {
        if (u == n) {
            if (check())
                res ++ ;
            return;
        }
        
        // 不动 方向0 距离0
        path.push_back({0, 0});
        dfs(u + 1);
        path.pop_back();
        
        // 方向
        for (int i = 0; i < 8; ++ i ) {
            string & s = pc[u];
            // trick
            if (s == "rook" && i % 2)
                continue;
            if (s == "bishop" && i % 2 == 0)
                continue;
            
            // 起始位置
            int x = pt[u][0], y = pt[u][1];
            // 有效步数
            for (int j = 1; ; ++ j ) {
                x += dx[i], y += dy[i];
                if (x < 1 || x > 8 || y < 1 || y > 8)
                    break;
                path.push_back({i, j});
                dfs(u + 1);
                path.pop_back();
            }
        }
    }
    
    int countCombinations(vector<string>& pieces, vector<vector<int>>& positions) {
        pc = pieces, pt = positions;
        n = pc.size();
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

> [!NOTE] **[LeetCode 254. 因子的组合](https://leetcode.cn/problems/factor-combinations/)** [TAG]
> 
> 题意: 
> 
> 接收一个整数 n 并返回该整数所有的因子组合

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
    // l 代表因子的起始数，保证因子的有序性可以做到天然的去重
    vector<vector<int> > dfs(int n, int l) {
        vector<vector<int> > res;
        for (int i = l; i * i <= n; ++i) {
            if (n % i == 0) {
                res.push_back({n / i, i});
                for (auto v : dfs(n / i, i)) {
                    v.push_back(i);
                    res.push_back(v);
                }
            }
        }
        return res;
    }
    vector<vector<int>> getFactors(int n) {
        return dfs(n, 2);
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

> [!NOTE] **[LeetCode 3260. 找出最大的 N 位 K 回文数](https://leetcode.cn/problems/find-the-largest-palindrome-divisible-by-k/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 找规律 比较麻烦不适用
> 
> - 字典序搜索 结合pow预处理和贪心剪枝

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // n digits... => 很长 显然没有办法枚举回文串
    // 考虑从 k 入手, 其范围数据 [1, 9]
    // - 1: 999...999
    // - 2: 899...998
    // - 3: 999...999
    // - 4: 899.8.998 ?
    //
    // => 转为搜索
    
    const static int N = 1e5 + 10;
    
    int n, m, k;
    
    int p[N];   // 10^x % k 的余数
    void init() {
        p[0] = 1;   // ATTENTION 特殊值
        for (int i = 1; i < N; ++ i )
            p[i] = p[i - 1] * 10 % k;
    }
    
    bool st[N][10];
    string res;
    
    // 从前往后填充到第i个位置 当前的模数为j (没填充的都按0...)
    bool dfs(int i, int j) {
        if (i == m)
            return j == 0;  // 需要整除
        
        st[i][j] = true;
        // 贪心: 倒序填充
        for (int d = 9; d >= 0; -- d ) {
            int j2;
            if (n % 2 && i == m - 1)
                // 奇数长度的中间位置
                j2 = (j + d * p[i]) % k;
            else
                // 其他位置 前后下标对称
                j2 = (j + d * (p[i] + p[n - 1 - i])) % k;
            
            if (!st[i + 1][j2] && dfs(i + 1, j2)) {
                // ATTENTION 细节 不访问之前访问过的点 (因为之前已经贪心有序)
                res[i] = res[n - 1 - i] = '0' + d;
                return true;
            }
        }
        return false;
    }
    
    string largestPalindrome(int n, int k) {
        this->n = n, this->m = (n + 1) / 2, this->k = k;
        this->res = string(n, '0');
        init();
        
        memset(st, 0, sizeof st);
        dfs(0, 0);
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

### dfs 回溯

> [!NOTE] **[LeetCode 51. N 皇后](https://leetcode.cn/problems/n-queens/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 全排列搜索的拓展：每次不仅要判断【列】有没有1，还要判断两条【对角线】有没有1
>
> 由于每行只能有一个皇后，所以可以依次枚举每一行的皇后放到哪个位置（这样时间复杂度会从下降）
>
> 对角线的表达式，可以根据画图得出：y = x + k & y = -x + k
>
> ==> k = y - x + n;(下标不能为负数，做一个映射)；k = y + x

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:

    int n;
    vector<bool> col, dg, udg;
    vector<vector<string>> ans;
    vector<string> path;

    vector<vector<string>> solveNQueens(int _n) {
        n = _n;
        col = vector<bool>(n);
        dg = udg = vector<bool>(n * 2);
        path = vector<string>(n, string(n, '.'));

        dfs(0);
        return ans;
    }

    void dfs(int u) {
        if (u == n) {
            ans.push_back(path);
            return;
        }

        for (int i = 0; i < n; i ++ ) {
            if (!col[i] && !dg[u - i + n] && !udg[u + i]) {
                col[i] = dg[u - i + n] = udg[u + i] = true;
                path[u][i] = 'Q';
                dfs(u + 1);
                path[u][i] = '.';
                col[i] = dg[u - i + n] = udg[u + i] = false;
            }
        }
    }
};
```

##### **Python**

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        path = [["."] * n for _ in range(n)]
        col = [False] * n 
        dg = [False] * 2 * n 
        udg = [False] * 2 * n 

        def dfs(u):
           #搜到最后一行的下一个位置（行）
            if u == n: 
                # 把每一行都转换拼接为一个字符串
                res.append(["".join(path[i]) for i in range(n)])
                return
             
            # dg[u-i+n] 写成：dg[i-u+n] 也可以 ac
            for i in range(n):
                if not col[i] and not dg[u-i+n] and not udg[u+i]:
                    path[u][i] = "Q"
                    col[i] = dg[u-i+n] = udg[u+i] = True
                    dfs(u + 1)
                    path[u][i] = "."
                    col[i] = dg[u-i+n] = udg[u+i] = False
        dfs(0)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 52. N皇后 II](https://leetcode.cn/problems/n-queens-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思路同上

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    vector<bool> col, dg, udg;

    int totalNQueens(int _n) {
        n = _n;
        col = vector<bool>(n);
        dg = udg = vector<bool>(2 * n);
        return dfs(0);
    }

    int dfs(int u) {
        if (u == n) return 1;
        int res = 0;
        for (int i = 0; i < n; i ++ ) {
            if (!col[i] && !dg[u - i + n] && !udg[u + i]) {
                col[i] = dg[u - i + n] = udg[u + i] = true;
                res += dfs(u + 1);
                col[i] = dg[u - i + n] = udg[u + i] = false;
            }
        }

        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        self.res = 0
        path = [['.'] * n for _ in range(n)]
        col = [False] * n 
        dg = [False] * 2 * n 
        udg = [False] * 2 * n 

        def dfs(u):
            if u == n:
                self.res += 1
            for i in range(n):
                if not col[i] and not dg[u-i+n] and not udg[u+i]:
                    path[u][i] = 'Q'
                    col[i] = dg[u-i+n] = udg[u+i] = True 
                    dfs(u + 1)
                    path[u][i] = '.'
                    col[i] = dg[u-i+n] = udg[u+i] = False

        dfs(0)
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 直接暴力搜索出所有合法方案。合法的 IP 地址由四个 [0,255] 的整数组成。
> 我们直接枚举四个整数的位数，然后判断每个数的范围是否在 [0,255] 之间。
>
> 搜索题：最重要的是 搜索顺序！ 先搜第一个数 再是第二个数...
> 1. 要求每个数都在 0~255 之间；
> 2. 必须是一个合法的数：就是【不能有前导0】
>
> 搜索顺序：先搜第一个数，然后 第二个， 再三个，最后一个。



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    vector<string> ans;
    vector<string> restoreIpAddresses(string s) {
        dfs(s, 0, 0, "");
        return ans;
    }

    void dfs(string& s, int u, int k, string path) {
        if (u == s.size()) {
            if (k == 4) {
                path.pop_back();
                ans.push_back(path);
            }
            return;
        }
        if (k == 4) return;
        for (int i = u, t = 0; i < s.size(); i ++ ) {
            if (i > u && s[u] == '0') break;  // 有前导0
            t = t * 10 + s[i] - '0';
            if (t <= 255) dfs(s, i + 1, k + 1, path + to_string(t) + '.');
            else break;
        }
    }
};
```

##### **C++ 旧代码**

```cpp
class Solution {
public:
    vector<string> res;
    vector<string> path;
    bool isValid(string& ip) {
        int val = stoi(ip);
        if (val > 255 || (ip.size() >= 2 && ip[0] == '0')) return false;
        return true;
    }
    void dfs(string& s, int step) {
        int maxl = (4 - path.size()) * 3;   // 剩下最多多少位
        if (s.size() - step > maxl) return;
        if (path.size() == 4 && step == s.size()) {
            string ans;
            for (int i = 0; i < 4; ++ i ) {
                if (i) ans.push_back('.');
                ans += path[i];
            }
            res.push_back(ans);
            return;
        }
        for (int i = step; i < s.size() && i <= step + 2; ++ i ) {
            string ip = s.substr(step, i - step + 1);
            if (!isValid(ip)) continue;
            path.push_back(ip);
            dfs(s, i + 1);
            path.pop_back();
        }
    }
    vector<string> restoreIpAddresses(string s) {
        int n = s.size();
        if (n < 4) return res;
        dfs(s, 0);
        return res;
    }
};
```
##### **Python**

```python
# 直接用字符串
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []

        # u表示枚举到的字符串下标，k表示当前截断的IP个数，s表示原字符串
        def dfs(s, u, k, path):
            # 遍历完了所有的字符，且分割为合法的四份
            if u == len(s):
                if k == 4:
                    res.append(path[:-1])
                # return  # 可写 可不写
            # 没遍历完所有字符，就已经有了 4 个合法值，那么不能再分割了
            if k == 4:
                return 
            
            t = 0
            for i in range(u, len(s)):
                if i > u and int(s[u]) == 0:break # 去除前导0
                t = t * 10 + int(s[i])
                # 数值合法，则继续深搜
                if t <= 255:
                    dfs(s, i+1, k+1, path + str(t) + '.')
                else:
                    break
                # 代码中并没有改变 path 的值，不需要恢复现场。

        dfs(s, 0, 0, "")
        return res
      
      
# 用数组保存
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        path = []

        # u表示枚举到的字符串下标，k表示当前截断的IP个数，s表示原字符串
        def dfs(u, k, s):
          	# 当搜完了所有位，如果当前个数有4个的话，就加入答案
            if u == len(s):
                if k == 4:
                    ans = str(path[0])
                    for i in range(1, 4):
                        ans += '.' + str(path[i])
                    res.append(ans)
						
            # 剪枝：当前已经有四个数了，说明后面还有其他的数 5个甚至更多，所以就直接return 【超过12位的情况】
            if k > 4:
                return

            t = 0
            for i in range(u, len(s)):
                t = t * 10 + int(s[i])
                if t >= 0 and t < 256:
                    path.append(t)
                    dfs(i + 1, k + 1, s)
                    path.pop()
                # 当前数太大，直接 break
                if not t:
                    break
                    
				# 从 0 位开始搜，搜第 0 个数
        dfs(0, 0, s)
        return res      
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 301. 删除无效的括号](https://leetcode.cn/problems/remove-invalid-parentheses/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂回溯
> 
> 非常好的题 重复做
> 
> 括号匹配问题两个原则: 
> 
> - 左右括号数量相同
> - 对于任意前缀，左括号数量都大于等于右括号数量

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 风格一致**

```cpp
class Solution {
public:
    vector<string> ans;

    vector<string> removeInvalidParentheses(string s) {
        // l 代表左括号比右括号多的个数(不计要删的右括号)
        // r 代表需要删除的右括号个数
        int l = 0, r = 0;
        for (auto x: s)
            if (x == '(') l ++ ;
            else if (x == ')') {
                if (l == 0) r ++ ;
                else l -- ;
            }

        dfs(s, 0, "", 0, l, r);
        return ans;
    }

    void dfs(string& s, int u, string path, int cnt, int l, int r) {
        if (u == s.size()) {
            if (!cnt) ans.push_back(path);
            return;
        }

        if (s[u] != '(' && s[u] != ')') dfs(s, u + 1, path + s[u], cnt, l, r);
        else if (s[u] == '(') {
            int k = u;
            while (k < s.size() && s[k] == '(') k ++ ;
            l -= k - u;
            // 剪枝: 枚举删除多少个括号字符 而非在哪个位置删除
            for (int i = k - u; i >= 0; i -- ) {
                if (l >= 0) dfs(s, k, path, cnt, l, r);
                path += '(';
                cnt ++, l ++ ;
            }
        } else if (s[u] == ')') {
            int k = u;
            while (k < s.size() && s[k] == ')') k ++ ;
            r -= k - u;
            for (int i = k - u; i >= 0; i -- ) {
                // cnt >= 0 : 必须时刻满足当前已有的左括号数量大于等于右括号数量
                if (cnt >= 0 && r >= 0) dfs(s, k, path, cnt, l, r);
                path += ')';
                cnt --, r ++ ;
            }
        }
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    bool isvalid(string s) {
        int cnt = 0;
        for (auto c : s) {
            if (c == '(') cnt ++ ;
            else if (c == ')') {
                cnt -- ;
                if (cnt < 0) return false;
            }
        }
        return cnt == 0;
    }

    void dfs(string s, int st, int l, int r, vector<string>& ans) {
        if (l == 0 && r == 0) {
            if (isvalid(s)) ans.push_back(s);
            return;
        }
        for (int i = st; i < s.size(); ++ i ) {
            if (i != st && s[i] == s[i - 1]) continue;
            if (s[i] == '(' && l > 0)
                dfs(s.substr(0, i) + s.substr(i + 1, s.size() - 1 - i), i, l - 1, r, ans);
            if (s[i] == ')' && r > 0)
                dfs(s.substr(0, i) + s.substr(i + 1, s.size() - 1 - i), i, l, r - 1, ans);
        }
    }

    vector<string> removeInvalidParentheses(string s) {
        int left = 0, right = 0;
        vector<string> ans;

        for (auto c : s) {
            if (c == '(')
                left ++ ;
            else if (c == ')') {
                if (left > 0)
                    left -- ;
                else
                    right++;
            }
        }
        // left和right表示左右括号要删除的个数
        dfs(s, 0, left, right, ans);
        return ans;
    }
};
```

##### **Python**

```python
"""
(DFS+括号序列) O( pow(2,n) * n)
这是一道有关括号序列的问题。那么如何判断一个括号序列是否合法呢？
判断方法：从前往后扫描字符串，维护一个计数器，遇到(就加一，遇到)就减一，如果过程中计数器的值都是非负数，且最终计数器的值是零，则括号序列是合法的。

左括号和右括号可以分开考虑，我们先来考虑删除哪些多余的右括号。
扫描字符串时，如果计数器的值小于0，说明前面的)太多，我们可以dfs暴力枚举删除前面哪些)，使得计数器的值大于等于0。
当处理完)后，我们只需将整个字符串逆序，同时把左右括号互换，就可以用相同的处理流程处理(了。

暴力dfs搜索空间比较高，我们要想办法进行剪枝。
剪枝方法：对于连续的)，不论删除哪一个，得到的方案都是相同的，所以我们对于所有连续的)，只枚举删除多少个。

剪枝后的算法在LeetCode上击败了 100% 的代码。

时间复杂度分析：我们先来考虑搜索空间有多大，最坏情况下，对于每个括号都有删或不删两种选择，所以共有 pow(2,n) 种不同方案。对于每种方案，最后还需要 O(n) 的计算量来记录答案，所以总时间复杂度是O( pow(2,n) )

"""
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        l, r = 0, 0
        for c in s:
            if c == '(':
                l += 1
            elif c == ')':
                if l == 0:
                    r += 1
                else:
                    l -= 1

        ans = []

        def dfs(u, path, cnt, l, r):
            # cnt 表示当前左括号-右括号数量
            if u == len(s):
                if cnt == 0:
                    ans.append(path)
                return 

            if s[u] != '(' and s[u] != ')':
                dfs(u+1, path+s[u], cnt, l, r)
            elif s[u] == '(':
                k = u
                while k < len(s) and s[k] == '(':
                    k += 1
                l -= k - u
                for i in range(k-u, -1, -1):
                    if cnt >= 0 and l >= 0:
                        dfs(k, path, cnt, l, r)
                    path += '('
                    cnt += 1
                    l += 1
            else:
                k = u 
                while k < len(s) and s[k] == ')':
                    k += 1
                r -= k - u
                for i in range(k-u, -1, -1):
                    if cnt >= 0 and r >= 0:
                        dfs(k, path, cnt, l, r)
                    path += ')'
                    cnt -= 1
                    r += 1

        dfs(0, "", 0, l, r)

        return ans
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 529. 扫雷游戏](https://leetcode.cn/problems/minesweeper/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> DFS + 模拟，理解题意，一共有三种可能的情况：
>
> 1. 如果当前选中的点恰好是地雷，那修改其为X，然后直接返回即可
> 2. 如果不是地雷的话，那就看周边是否有其他地雷，如果没有的话，就把这些点改为N，然后继续dfs去判断周围的其他点
> 3. 如果不是地雷，并且周边有地雷，那就修改为对应的数字

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int dx[8] = {-1, 0, 0, 1, -1, -1, 1, 1}, dy[8] = {0, -1, 1, 0, -1, 1, -1, 1};
    void dfs(vector<vector<char>>& board, int x, int y) {
        int cnt = 0;
        for (int i = 0; i < 8; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= board.size() || ny < 0 || ny >= board[0].size()) continue;
            cnt += board[nx][ny] == 'M';
        }
        if (cnt > 0) board[x][y] = cnt + '0';
        else {
            board[x][y] = 'B';
            for (int i = 0; i < 8; ++ i ) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx < 0 || nx >= board.size() || ny < 0 || ny >= board[0].size() || board[nx][ny] != 'E') continue;
                dfs(board, nx, ny);
            }
        }
    }
    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        int x = click[0], y = click[1];
        if (board[x][y] == 'M') board[x][y] = 'X';
        else dfs(board, x, y);
        return board;
    }
};


// yxc
class Solution {
public:
    int n, m;

    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        n = board.size(), m = board[0].size();
        int x = click[0], y = click[1];
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
            return board;
        }
        dfs(board, x, y);
        return board;
    }

    void dfs(vector<vector<char>>& board, int x, int y) {
        if (board[x][y] != 'E') return;
        int s = 0;
        for (int i = max(x - 1, 0); i <= min(n - 1, x + 1); i ++ )
            for (int j = max(y - 1, 0); j <= min(m - 1, y + 1); j ++ )
                if (i != x || j != y)
                    if (board[i][j] == 'M' || board[i][j] == 'X')
                        s ++ ;
        if (s) {
            board[x][y] = '0' + s;
            return;
        }
        board[x][y] = 'B';
        for (int i = max(x - 1, 0); i <= min(n - 1, x + 1); i ++ )
            for (int j = max(y - 1, 0); j <= min(m - 1, y + 1); j ++ )
                if (i != x || j != y)
                    dfs(board, i, j);
    }
};
```

##### **Python**

```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        global n, m
        n, m = len(board), len(board[0])
        x, y = click
        if board[x][y] == 'M':
            board[x][y] = 'X'
            return board
        self.dfs(board, x, y)
        return board

    def dfs(self, board, x, y):
        if board[x][y] != 'E':
            return
        s = 0
        for i in range(max(x - 1, 0), min(n - 1, x + 1) + 1):
            for j in range(max(y - 1, 0), min(m - 1, y + 1) + 1):
                if i != x or j != y:
                    if board[i][j] == 'M' or board[i][j] == 'X':
                        s += 1
        if s:
            board[x][y] = str(s)
            return 
        board[x][y] = 'B'
        for i in range(max(x - 1, 0), min(n - 1, x + 1) + 1):
            for j in range(max(y - 1, 0), min(m - 1, y + 1) + 1):
                if i != x or j != y:
                    self.dfs(board, i, j)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 679. 24 点游戏](https://leetcode.cn/problems/24-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 经典 递归 trick
>
> 由于可以从数组任意取数，所以括号可以忽略，暴搜所有方案：
>
> 1. 选两个数，进行运算操作，就变成了三个数的组合: 4 * 3 * 4（4种运算符）
>
> 2. 再从中选两个数，进行运算操作，就变成了两个数的组合: （4 * 3 * 4） * 3 * 2 *4
>
> 3. 最后只剩下两个数，进行运算操作，就变成了一个数的组合: 
>
>    （(4 * 3 * 4） * (3 * 2 * 4)) * 2 * 1 * 4 = 9216 种方案
>
> 4. 最后就看这剩下的一个数是不是24（由于这道题的除法不是整数除法，所以要算一个差值）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<double> get(vector<double>& nums, int i, int j, double x) {
        vector<double> ret;
        for (int k = 0; k < nums.size(); ++ k )
            if (k != i && k != j)
                ret.push_back(nums[k]);
        ret.push_back(x);
        return ret;
    }

    bool dfs(vector<double> nums) {
        if (nums.size() == 1)
            return fabs(nums[0] - 24) < 1e-8;
        for (int i = 0; i < nums.size(); ++ i )
            for (int j = 0; j < nums.size(); ++ j )
                if (i != j) {
                    double a = nums[i], b = nums[j];
                    if (dfs(get(nums, i, j, a + b)))
                        return true;
                    if (dfs(get(nums, i, j, a - b)))
                        return true;
                    if (dfs(get(nums, i, j, a * b)))
                        return true;
                    if (b && dfs(get(nums, i, j, a / b)))
                        return true;
                }
        return false;
    }

    bool judgePoint24(vector<int>& nums) {
        vector<double> a(nums.begin(), nums.end());
        return dfs(a);
    }
};
```

##### **Python**

```python
class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        def get(nums, i, j, x):
            res = []
            for k in range(len(nums)):
                if k != i and k != j:
                    res.append(nums[k])
            res.append(x)
            return res

        def dfs(nums: List[float]):
            if len(nums) == 1:
                return abs(nums[0] - 24) < 1e-8
            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i != j:
                        a, b = nums[i], nums[j]
                        if dfs(get(nums, i, j, a + b)):
                            return True
                        if dfs(get(nums, i, j, a - b)):
                            return True
                        if dfs(get(nums, i, j, a * b)):
                            return True
                        if b and dfs(get(nums, i, j, a / b)):
                            return True
            return False
        return dfs(cards)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1219. 黄金矿工](https://leetcode.cn/problems/path-with-maximum-gold/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 回溯思想的实现 trick
> 
> 一开始想复杂了，直接 tmp 记录 g 随后恢复即可（记得修改当前值为0，避免重复计算/走回头路）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, m;
    vector<vector<int>> gs, g;
    
    vector<int> dx = {-1, 0, 0, 1}, dy = {0, -1, 1, 0};
    int dfs(int x, int y) {
        int ret = g[x][y], t = 0;
        int tmp = g[x][y];
        g[x][y] = 0;
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m || !g[nx][ny]) continue;
            t = max(t, dfs(nx, ny));
        }
        g[x][y] = tmp;
        return ret + t;
    }
    
    int getMaximumGold(vector<vector<int>>& grid) {
        gs = grid;
        n = gs.size(), m = gs[0].size();
        
        int res = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j ) if (gs[i][j]) {
                g = gs;
                res = max(res, dfs(i, j));
            }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        self.res = 0

        def dfs(x, y, gold):
            gold += grid[x][y]
            self.res = max(self.res, gold)
            tmp = grid[x][y]
            grid[x][y] = 0
            dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] != 0:
                    dfs(nx, ny, gold)
            grid[x][y] = tmp

        for i in range(n):
            for j in range(m):
                if grid[i][j] != 0:
                    dfs(i, j, 0)
        
        return self.res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode LCP 58. 积木拼接](https://leetcode.cn/problems/De4qBB/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常好的回溯
> 
> 比赛的时候一个细节出错的地方：
> 
> **问题在于：把 $shape$ 作为全局变量来执行 $draw$ 操作。在回溯时 $shape$ 已经被下一层递归修改，【而非本层构造的内容】，所以恢复现场必错**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 底部和顶部   侧面
    // 1 * 5 * 8 * (4 * 3 * 2 * 8 * 8 * 8) = 40 * 24 * 512 = 500000 50w
    const static int N = 11;
    void mirror(vector<string> & state) {
        for (int i = 0; i < m; ++ i )
            for (int j = 0, k = m - 1; j < k; ++ j , -- k )
                swap(state[i][j], state[i][k]);
    }
    void rotate(vector<string> & state) {
        for (int i = 0; i < m; ++ i )
            for (int j = 0; j < i; ++ j )
                swap(state[i][j], state[j][i]);
        mirror(state);
    }

    vector<vector<string>> g;
    int n, m;
    bool st[6];
    bool has[N][N][N];
    vector<string> hash[N][N];
    vector<string> generateShape(int i, int change) {
        auto shape = g[i];
        // flip
        if (change / 4)
            mirror(shape);
        for (int i = 0; i < change % 4; ++ i )
            rotate(shape);
        return hash[i][change] = shape;
    }
    bool candraw(int u, vector<string> & shape) {
        if (u == 0) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][j][0])
                        return false;
        } else if (u == 1) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][0][j])
                        return false;
        } else if (u == 2) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[0][i][j])
                        return false;
        } else if (u == 3) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][m - 1][j])
                        return false;
        } else if (u == 4) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[m - 1][i][j])
                        return false;
        } else if (u == 5) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][j][m - 1])
                        return false;
        }
        return true;
    }
    void draw(int u, vector<string> & shape, bool flag) {
        if (u == 0) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[i][j][0] = flag;
        } else if (u == 1) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                         has[i][0][j] = flag;
        } else if (u == 2) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[0][i][j] = flag;
        } else if (u == 3) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[i][m - 1][j] = flag;
        } else if (u == 4) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[m - 1][i][j] = flag;
        } else if (u == 5) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[i][j][m - 1] = flag;
        }
    }

    // [d, l, b, r, f, u]
    bool dfs(int u) {
        if (u == 6)
            return true;
        for (int i = 0; i < n; ++ i )
            if (!st[i]) {
                st[i] = true;
                for (int j = 0; j < 8; ++ j ) {
                    auto & shape = hash[i][j];
                    if (candraw(u, shape)) {
                        draw(u, shape, true);
                        if (dfs(u + 1))
                            return true;
                        draw(u, shape, false);
                    }
                }
                st[i] = false;
            }
        return false;
    }
    
    bool composeCube(vector<vector<string>>& shapes) {
        this->g = shapes;
        n = g.size();
        m = g[0].size();
        
        // precheck
        {
            int tot = 0;
            for (auto & shape : g)
                for (auto s : shape)
                    for (auto c : s)
                        if (c == '1')
                            tot ++ ;
            if (tot != m * m * m - (m - 2) * (m - 2) * (m - 2))
                return false;
        }
        
        memset(has, 0, sizeof has);
        memset(st, 0, sizeof st);

        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < 8; ++ j )
                hash[i][j] = generateShape(i, j);

        return dfs(0);
    }
};
```

##### **C++ WA**

```cpp
// WA 29 代码
class Solution {
public:
    // 底部和顶部   侧面
    // 1 * 5 * 8 * (4 * 3 * 2 * 8 * 8 * 8) = 40 * 24 * 512 = 500000 50w
    const static int N = 11;
    
    vector<vector<string>> g;
    bool has[N][N][N];
    
    int n, m;
    bool st[6];
    
    void mirror(vector<string> & state) {
        for (int i = 0; i < m; ++ i )
            for (int j = 0, k = m - 1; j < k; ++ j , -- k )
                swap(state[i][j], state[i][k]);
    }
    void rotate(vector<string> & state) {
        for (int i = 0; i < m; ++ i )
            for (int j = 0; j < i; ++ j )
                swap(state[i][j], state[j][i]);
        mirror(state);
    }
    
    bool xs[N][N];
    vector<string> shape;
    void fixshape(int i, int change) {
        this->shape = g[i];
        // flip
        if (change / 4)
            mirror(shape);
        for (int i = 0; i < change % 4; ++ i )
            rotate(shape);
    }
    
    bool candraw(int u) {
        if (u == 0) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][j][0])
                        return false;
        } else if (u == 1) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][0][j])
                        return false;
        } else if (u == 2) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[0][i][j])
                        return false;
        } else if (u == 3) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][m - 1][j])
                        return false;
        } else if (u == 4) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[m - 1][i][j])
                        return false;
        } else if (u == 5) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1' && has[i][j][m - 1])
                        return false;
        }
        return true;
    }
    void draw(int u, bool flag) {
        if (u == 0) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[i][j][0] = flag;
        } else if (u == 1) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                         has[i][0][j] = flag;
        } else if (u == 2) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[0][i][j] = flag;
        } else if (u == 3) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[i][m - 1][j] = flag;
        } else if (u == 4) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[m - 1][i][j] = flag;
        } else if (u == 5) {
            for (int i = 0; i < m; ++ i )
                for (int j = 0; j < m; ++ j )
                    if (shape[i][j] == '1')
                        has[i][j][m - 1] = flag;
        }
    }
    
    // [d, l, b, r, f, u]
    bool dfs(int u) {
        if (u == 6)
            return true;
        for (int i = 0; i < n; ++ i )
            if (!st[i]) {
                st[i] = true;
                for (int j = 0; j < 8; ++ j ) {
                    fixshape(i, j);
                    if (candraw(u)) {
                        draw(u, true);
                        if (dfs(u + 1))
                            return true;
                        draw(u, false); // WA: 此时的 shape 已经不是本层递归中构造的，故恢复现场必错
                    }
                }
                st[i] = false;
            }
        return false;
    }
    
    void printShape() {
        for (auto s : shape)
            cout << s << endl;
        cout << endl;
    }
    
    bool composeCube(vector<vector<string>>& shapes) {
        this->g = shapes;
        m = g[0].size();
        
        // precheck
        {
            int tot = 0;
            for (auto & shape : g)
                for (auto & s : shape)
                    for (auto c : s)
                        if (c == '1')
                            tot ++ ;
            if (tot != m * m * m - (m - 2) * (m - 2) * (m - 2))
                return false;
        }
        n = g.size();
        
        memset(st, 0, sizeof st);
        return dfs(0);
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

> [!NOTE] **[Codeforces Inna and Dima](http://codeforces.com/problemset/problem/374/C)**
> 
> 题意: 
> 
> 可以按照 $DIMA$ 的顺序循环走，问最多能走几次，或者无限次

> [!TIP] **思路**
> 
> DFS 推理不需要关注 4 的长度
> 
> 有点 trick 的 dfs

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Inna and Dima
// Contest: Codeforces - Codeforces Round #220 (Div. 2)
// URL: https://codeforces.com/problemset/problem/374/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e3 + 10, M = 1e6 + 10;

int h[M], e[M << 1], ne[M << 1], idx;  // 这里直接写成了 N M 导致TLE 排错很久
void init() {
    memset(h, -1, sizeof h);
    idx = 0;
}
void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int n, m;
char g[N][N];
int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};

bool st[M], flag;
int dis[M];
void dfs(int x) {
    if (dis[x])
        return;
    st[x] = true;  // 回溯
    dis[x] = 1;
    for (int i = h[x]; ~i; i = ne[i]) {
        int j = e[i];
        if (st[j]) {
            // 走到了本轮次可以走到的点，则可以无限循环
            flag = true;
            break;  // st[x] = false; break;
        }
        dfs(j);
        if (flag)
            break;  // ...
        dis[x] = max(dis[x], dis[j] + 1);
    }
    st[x] = false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    init();

    cin >> n >> m;
    for (int i = 0; i < n; ++i)
        cin >> g[i];

    unordered_map<char, int> hash;
    hash['D'] = 0, hash['I'] = 1, hash['M'] = 2, hash['A'] = 3;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < 4; ++k) {
                int ni = i + dx[k], nj = j + dy[k];
                if (ni < 0 || ni >= n || nj < 0 || nj >= m)
                    continue;
                if ((hash[g[i][j]] + 1) % 4 == hash[g[ni][nj]]) {
                    int a = i * m + j, b = ni * m + nj;
                    add(a, b);
                }
            }

    int res = 0;
    memset(st, 0, sizeof st);
    flag = false;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (g[i][j] == 'D') {
                dfs(i * m + j);
                if (flag) {  // 可以走无限多次
                    cout << "Poor Inna!" << endl;
                    return 0;
                }
                if (dis[i * m + j] > res)
                    res = max(res, dis[i * m + j]);
            }

    res /= 4;
    if (res)
        cout << res << endl;
    else
        cout << "Poor Dima!" << endl;

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


### dfs 分组

> [!NOTE] **[AcWing 1118. 分成互质组](https://www.acwing.com/problem/content/1120/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **标准dfs分组**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, res, len;
vector<int> c;
vector<vector<int>> g;

int gcd(int a, int b) {
    if (b) return gcd(b, a % b);
    return a;
}

bool check(int group, int x) {
    // for(int i = 0; i < g[group].size(); ++i)
    for (auto& v : g[group])
        if (gcd(v, x) > 1) return false;
    return true;
}

void dfs(int u) {
    if (u == n) {
        res = min(res, len);
        return;
    }
    // 使用已有的组
    for (int i = 0; i < len; ++i)
        if (check(i, c[u])) {
            g[i].push_back(c[u]);
            dfs(u + 1);
            g[i].pop_back();
        }
    // 单独放一组　开辟新的组
    g[len++].push_back(c[u]);
    dfs(u + 1);
    g[--len].pop_back();
}

int main() {
    cin >> n;
    c = vector<int>(n);
    g = vector<vector<int>>(n);  // 最多用ｎ组
    for (int i = 0; i < n; ++i) cin >> c[i];
    // sort(c.begin(), c.end());

    // res = inf, len = 0;
    res = 10;
    dfs(0);

    cout << res << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 表达式类求解

> [!NOTE] **[LeetCode 241. 为运算表达式设计优先级](https://leetcode.cn/problems/different-ways-to-add-parentheses/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 本题有一个限制：只能加括号，不能调整数的顺序。
> 那么对于任意一个表达式，转化为二叉树之后，虽然树的形态各异，但是它们的中序遍历都是一样的。那这道题和 95 题就很像（知识背景）
>
> 直接搜索就可以。
> 1. 先把字符串转化为列表，方便操作，并且效率会更高（list的效率高于 string）
> 2. 枚举整个字符串的list，当遇到操作符时，就把当前操作符作为根节点，那么左子树为：[0, i-1]，右子树为：[i+1, n-1]，并将两个子树递归下去，搜索出所有形态各异的树的种类
> 3. 左子树和右子树集合中的任意二叉树都可以两两相结合与当前根节点 i 形成新的二叉树（也就是新的组合结果）
> 4. 当递归过程中，枚举的字符串只是一个数字时，直接返回该数。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<string> expr;
    vector<int> dfs(int l, int r) {
        if (l == r) return {stoi(expr[l])};
        vector<int> res;
        // 枚举当前区间最后一个算的运算符
        for (int i = l + 1; i < r; i += 2) {
            auto left = dfs(l, i - 1), right = dfs(i + 1, r);
            for (auto x : left) for (auto y : right) {
                int z;
                if (expr[i] == "+") z = x + y;
                else if (expr[i] == "-") z = x - y;
                else z = x * y;
                res.push_back(z);
            }
        }
        return res;
    }
    vector<int> diffWaysToCompute(string s) {
        for (int i = 0; i < s.size(); ++ i )
            if (isdigit(s[i])) {
                int j = i, x = 0;
                while (j < s.size() && isdigit(s[j]))
                    x = x * 10 + (s[j ++ ] - '0');
                i = j - 1;
                expr.push_back(to_string(x));
            } else expr.push_back(s.substr(i, 1));
        return dfs(0, expr.size() - 1);
    }
};
```

##### **Python**

```python
class Solution:
    def diffWaysToCompute(self, s: str) -> List[int]:
        exp = list()
        i = 0
        while i < len(s):
            j = i 
            if s[j].isdigit():
                x = 0
                while j < len(s) and s[j].isdigit():
                    x = x * 10 + int(s[j])
                    j += 1
                exp.append(x)
            else:
                exp.append(s[i]) 
                j = i + 1
            i = j
        
        def dfs(l, r):
            # 表示当前只有一个数字
            if l == r:
                return [int(exp[l])]
            res = []
            # 当前表达式，第一个和最后一个一定都是数字，然后隔一个是一个操作符
            # 遍历字符串，枚举每一个根节点的操作符
            for i in range(l + 1, r, 2):
                left = dfs(l, i - 1)
                right = dfs(i + 1, r)
                for x in left:
                    for y in right:
                        if exp[i] == '+':
                            z = x + y
                        elif exp[i] == '-':
                            z = x - y
                        else:
                            z = x * y
                        res.append(z)
            return res
        return dfs(0, len(exp) - 1)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 282. 给表达式添加运算符](https://leetcode.cn/problems/expression-add-operators/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 维护抽象代数结构 `a + b * _`
> 
> 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 代数结构：维护一个 【a + b * c _】 的结构
    typedef long long LL;
    vector<string> res;
    string path;
    void dfs(string & num, int u, int len, LL a, LL b, LL target) {
        if (u == num.size()) {
            // 0 - len-1 去除最后一个加号
            if (a == target) res.push_back(path.substr(0, len - 1));
            return;
        }
        LL c = 0;
        for (int i = u; i < num.size(); ++ i ) {
            c = c * 10 + num[i] - '0';
            path[len ++ ] = num[i];
            
            // +
            path[len] = '+';
            dfs(num, i + 1, len + 1, a + b * c, 1, target);

            if (i + 1 < num.size()) {
                // -
                path[len] = '-';
                dfs(num, i + 1, len + 1, a + b * c, -1, target);

                // *
                path[len] = '*';
                dfs(num, i + 1, len + 1, a, b * c, target);
            }
            // 去除前导 0 
            if (num[u] == '0') break;
        }
    }
    vector<string> addOperators(string num, int target) {
        path.resize(100);
        dfs(num, 0, 0, 0, 1, target);
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

### 进阶

> [!NOTE] **[LeetCode 126. 单词接龙 II](https://leetcode.cn/problems/word-ladder-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> bfs找最短路 + dfs获取方案

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 标准
class Solution {
public:
    vector<vector<string>> ans;
    vector<string> path;
    unordered_set<string> S;
    unordered_map<string, int> dist;

    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        for (auto& word: wordList) S.insert(word);
        queue<string> q;
        q.push(beginWord);
        dist[beginWord] = 0;
        while (q.size()) {
            auto t = q.front();
            q.pop();
            string r = t;
            for (int i = 0; i < t.size(); i ++ ) {
                t = r;
                for (char j = 'a'; j <= 'z'; j ++ )
                    if (j != r[i]) {
                        t[i] = j;
                        if (S.count(t) && dist.count(t) == 0) {
                            dist[t] = dist[r] + 1;
                            if (t == endWord) break;
                            q.push(t);
                        }
                    }
            }
        }

        if (dist.count(endWord)) {
            path.push_back(beginWord);
            dfs(beginWord, endWord);
        }

        return ans;
    }

    void dfs(string st, string ed) {
        if (st == ed) {
            ans.push_back(path);
            return;
        }

        string r = st;
        for (int i = 0; i < st.size(); i ++ ) {
            st = r;
            for (char j = 'a'; j <= 'z'; j ++ )
                if (j != r[i]) {
                    st[i] = j;
                    if (S.count(st) && dist[r] + 1 == dist[st]) {
                        path.push_back(st);
                        dfs(st, ed);
                        path.pop_back();
                    }
                }
        }
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

> [!NOTE] **[LeetCode 127. 单词接龙](https://leetcode.cn/problems/word-ladder/)**
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
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> wordSet(wordList.begin(), wordList.end());
        if (!wordSet.count(endWord)) return 0;
        queue<string> q{{beginWord}};
        int res = 0;
        while (!q.empty()) {
            for (int k = q.size(); k > 0; --k) {
                string word = q.front(); q.pop();
                if (word == endWord) return res + 1;
                for (int i = 0; i < word.size(); ++i) {
                    string newWord = word;
                    for (char ch = 'a'; ch <= 'z'; ++ch) {
                        newWord[i] = ch;
                        if (wordSet.count(newWord) && newWord != word) {
                            q.push(newWord);
                            wordSet.erase(newWord);
                        }   
                    }
                }
            }
            ++res;
        }
        return 0;
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

> [!NOTE] **[LeetCode 139. 单词拆分](https://leetcode.cn/problems/word-break/)**
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
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        typedef unsigned long long ULL;
        const int P = 131;
        unordered_set<ULL> hash;
        for (auto& word: wordDict) {
            ULL h = 0;
            for (auto c: word) h = h * P + c;
            hash.insert(h);
        }

        int n = s.size();
        vector<bool> f(n + 1);
        f[0] = true;
        s = ' ' + s;
        for (int i = 0; i < n; i ++ )
            if (f[i]) {
                ULL h = 0;
                for (int j = i + 1; j <= n; j ++ ) {
                    h = h * P + s[j];
                    if (hash.count(h)) f[j] = true;
                }
            }

        return f[n];
    }
};
```

##### **C++ substring**

```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<bool> f(n+1);
        f[0] = true;
        for (int i = 1; i <= n; ++ i ) {
            for (auto w : wordDict) {
                int len = w.size();
                if (i >= len && s.substr(i - len, len) == w) f[i] = f[i] || f[i - len];
            }
        }
        return f[n];
    }
};
```

##### **Python**

```python
# f[i] 表示s[0:i]能否被break； f[1] 表示s[0]单独一个字母是可以被break的

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        if not wordDict: return not s
        f = [False] * (n + 1)  # n个字符，就有n + 1 个半 
        f[0] = True # f[0]是s[0]前面的隔板
        for i in range(1, n + 1):
            for j in range(i - 1, -1, -1):
                if f[j] and s[j:i] in wordDict:
                    f[i] = True
                    break
        return f[-1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 140. 单词拆分 II](https://leetcode.cn/problems/word-break-ii/)**
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
class Solution {
public:
    // 1. pass
    unordered_map<int, vector<string>> ans;
    unordered_set<string> wordSet;
    
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        wordSet = unordered_set(wordDict.begin(), wordDict.end());
        backtrack(s, 0);
        return ans[0];
    }

    void backtrack(const string& s, int index) {
        if (!ans.count(index)) {
            if (index == s.size()) {
                ans[index] = {""};
                return;
            }
            ans[index] = {};
            for (int i = index + 1; i <= s.size(); ++i) {
                string word = s.substr(index, i - index);
                if (wordSet.count(word)) {
                    backtrack(s, i);
                    for (const string& succ: ans[i]) {
                        ans[index].push_back(succ.empty() ? word : word + " " + succ);
                    }
                }
            }
        }
    }
}
```

##### **C++ 2**

```cpp
class Solution {
public:
    // 2. yxc
    vector<bool> f;
    vector<string> ans;
    unordered_set<string> hash;
    int n;

    vector<string> wordBreak(string s, vector<string>& wordDict) {
        n = s.size();
        f.resize(n + 1);
        for (auto word: wordDict) hash.insert(word);
        f[n] = true;
        for (int i = n - 1; ~i; i -- )
            for (int j = i; j < n; j ++ )
                if (hash.count(s.substr(i, j - i + 1)) && f[j + 1])
                    f[i] = true;

        dfs(s, 0, "");
        return ans;
    }

    void dfs(string& s, int u, string path) {
        if (u == n) {
            path.pop_back();
            ans.push_back(path);
        } else {
            for (int i = u; i < n; i ++ )
                if (hash.count(s.substr(u, i - u + 1)) && f[i + 1])
                    dfs(s, i + 1, path + s.substr(u, i - u + 1) + ' ');
        }
    }
}
```

##### **C++ 3 TLE**

```cpp
class Solution {
public:
    // 3. 超时代码 重复计算部分记忆化搜索即过 参考1
/*
在用例 31 / 36 : 
"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
*/
    int n;
    vector<bool> f;             // f[i] 表示 [0, i] 是否可以被完整分割
    unordered_set<string> mp;
    vector<string> res;
    void dfs(string & s, int u, string path) {
        if(u == n) {
            path.pop_back();    // remove ' '
            res.push_back(path);
            return;
        }
        for(int i = u; i < n; ++i) {
            string ncur = s.substr(u, i-u+1);
            if(mp.count(ncur) && f[i+1])
                dfs(s, i+1, path + ncur + ' ');
        }
    }

    vector<string> wordBreak(string s, vector<string>& wordDict) {
        n = s.size();
        mp = unordered_set<string>(wordDict.begin(), wordDict.end());
        f.resize(n+1);
        f[0] = true;
        for(int i = 1; i <= n; ++i)
            for(int j = 0; j <i; ++j) {
                string cur = s.substr(j, i-j);
                if(mp.count(cur)) f[i] = f[i] | f[j];
            }
        
        dfs(s, 0, "");
        return res;
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

> [!NOTE] **[LeetCode 1900. 最佳运动员的比拼回合](https://leetcode.cn/problems/the-earliest-and-latest-rounds-where-players-compete/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然二进制枚举搜索即可
> 
> 比赛时忘了加记忆化 TLE 数次... 加行记忆化就过...
> 
> 加强搜索敏感度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, p1, p2;
    int minr, maxr;
    
    unordered_set<int> S;
    
    void dfs(int st, int d) {
        if (S.count(st))
            return;
        S.insert(st);
        
        int sz = __builtin_popcount(st);
        if (sz < 2)
            return;
        
        int cp = sz / 2;
        
        vector<int> ve;
        for (int i = 0; i < n; ++ i )
            if (st >> i & 1)
                ve.emplace_back(i);
        
        for (int i = 0; i < cp; ++ i )
            if (ve[i] + 1 == p1 && ve[sz - i - 1] + 1 == p2 || ve[i] + 1 == p2 && ve[sz - i - 1] + 1 == p1) {
                minr = min(minr, d), maxr = max(maxr, d);
                return;
            }
        
        // 某位为1则对应前半部分被淘汰
        for (int i = 0; i < 1 << cp; ++ i ) {
            int t = st;
            for (int j = 0; j < cp; ++ j )
                if (i >> j & 1)
                    t ^= 1 << ve[j];
                else
                    t ^= 1 << ve[sz - j - 1];
            if ((t >> (p1 - 1) & 1) == 0 || (t >> (p2 - 1) & 1) == 0)
                continue;
            
            dfs(t, d + 1);
        }
    }
    
    vector<int> earliestAndLatest(int n, int firstPlayer, int secondPlayer) {
        this->n = n, this->p1 = firstPlayer, this->p2 = secondPlayer;
        minr = 2e9, maxr = -2e9;
        
        dfs((1 << n) - 1, 1);
        
        return {minr, maxr};
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