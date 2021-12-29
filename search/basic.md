## 习题

> [!WARNING] **一般指 dfs**

### 构造

> [!NOTE] **[LeetCode 22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)**
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
# 递归，当左边括号的数量 == 右边括号数量 == n：就可以加入到答案中
# 如何保证加入到的路径是有效的呢？
# 1. 每次可以放置左括号的条件是：当前左括号的数目不超过 n。
# 2. 每次可以放置右括号的条件是：当前右括号的数目不超过左括号的数目。

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

> [!NOTE] **[LeetCode 95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)**
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
# 暴搜：递归所有方案
# 1. 对于每段连续的序列 l, l+1,...,r, 枚举二叉搜索树根节点的位置
# 2. 分别递归求出左右子树的所有方案
# 3. 左子树的任意一种方案和右子树的任意一种方案拼在一起，可以得到当前节点的一种方案，所以将左右子树的所有方案两两组和，并记录到答案中

class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if not n:return []

        def dfs(l, r):
            res = []   # 踩坑1: res一定要放在递归函数里，每次进来都是[]！
            if l > r:return [None]
            for i in range(l, r + 1):# 遍历根节点的位置，每个数值都可能成为根节点
                left = dfs(l, i - 1)
                right = dfs(i + 1, r)
                for j in left:
                    for k in right:
                        root = TreeNode(i)  # 踩坑2:每次都要生成一个新的root节点树
                        root.left, root.right = j, k
                        res.append(root)
            return res

        return dfs(1, n)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)**
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
"""
(动态规划) O(n2)
状态表示：f[n] 表示 n个节点的二叉搜索树共有多少种。
状态转移：左子树可以有 0,1,…n−1 个节点，对应的右子树有 n−1,n−2,…,0 个节点，f[n] 是所有这些情况的加和，所以 f[n]=∑n−1k=0 f[k]∗f[n−1−k]

时间复杂度分析：状态总共有 nn 个，状态转移的复杂度是 O(n)，所以总时间复杂度是 O(n2)。
"""

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

> [!NOTE] **[LeetCode 78. 子集](https://leetcode-cn.com/problems/subsets/)**
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
# DFS：子集问题，是属于 【不区分顺序】，而全排列这类题都是区分顺序的题

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

> [!NOTE] **[LeetCode 90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)**
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
                if k > i and nums[k] == nums[k-1]:continue  # 踩坑2 ：if k > i !这个条件！
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

> [!NOTE] **[LeetCode 698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 搜索 剪枝 重复
>
> 1. 先求出数组的总和，如果总和不是 k 的倍数，则直接返回 false。
> 2. 求出子集和的目标值，如果数组中有数字大于这个子集和目标值，则返回 false。
> 3. 将数组 从大到小 排序。贪心地来看，先安排大的数字会更容易先找到答案，这是因为小的数字再后期更加灵活。
> 4. 我们按照子集去递归填充，即维护一个当前总和 cur，如果当前值等于了目标值，则进行下一个子集的填充。最终，如果只剩下一个子集尚未填充（无需 0 个），则可以直接返回 true。
> 5. 还需要同时维护一个 start，即表示当前这个子集，需要从数组的哪个位置开始尝试。
> 6. 这里还有一个非常重要的策略，就是如果当前总和 cur 为 0 时，我们直接指定第一个未被使用的数字作为当前这个子集的第一个数字（最大的数字）。这是防止重复枚举，因为第一个未被使用的数字一定要出现在某个子集中。否则，如果我们最终没有使用这个数字，在尝试下一个集合时，还会重复尝试使用这个数字，造成大量重复计算。

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

```

<!-- tabs:end -->
</details>

<br>

* * *


### 排列

> [!NOTE] **[LeetCode 46. 全排列](https://leetcode-cn.com/problems/permutations/)**
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
                    used[i] = False
        
        dfs([])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> `!vis[i - 1]`

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
        nums.sort()   # 踩坑1:一定要先排序！
        used = [False] * n

        def dfs(path):
            if len(path) == n:
                res.append(path[:])
                return 
            for i in range(n):
                if not used[i]:
                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:continue # 踩坑2: not used[i-1] !!! 
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

> [!NOTE] **[LeetCode 784. 字母大小写全排列](https://leetcode-cn.com/problems/letter-case-permutation/)**
> 
> 题意: TODO

> [!TIP] **思路**
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
# 特别简单的暴搜DFS...从左到右枚举遍历
# 直接用字符串这个数据结构

class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        res = []
        def dfs(path, s):
            if not s:
                res.append(path)
                return  #一定要有return, 不然会继续执行下一行，导致list out of range
            c = s[0]
            if c.isalpha():
                dfs(path + c.lower(), s[1:])
                dfs(path + c.upper(), s[1:])
            else:
                dfs(path + c, s[1:])
        
        dfs('', S)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

### 组合

> [!NOTE] **[LeetCode 77. 组合](https://leetcode-cn.com/problems/combinations/)**
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
"""
深度优先搜索，每层枚举第 u 个数选哪个，一共枚举 k 层。由于这道题要求组合数，不考虑数的顺序，所以我们需要再记录一个值，表示当前数需要从几开始选，来保证所选的数递增。

时间复杂度分析：一共有 Ckn 个方案，另外记录每个方案时还需要 O(k)的时间，所以时间复杂度是 O(Ckn×k)。
"""

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        if not n:return []
        res =[]

        def dfs(path, i):
            if len(path) == k:
                res.append(path[:])
                return 
            for u in range(i, n + 1):
                # 踩坑：需要在循环里把当前数加入到path中，否则就是必须从第一个数字开始组合
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

> [!NOTE] **[LeetCode 39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)**
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
# 无重复元素，并且 每个数字可以用无数次

class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        if not nums:return []
        n = len(nums)
        # 要先排序，排序是为了剪枝；不然后面 nums[k] > target的时候 就没办法直接 return
        nums.sort()
        res = []

        def dfs(path, target, i):
            if target == 0:
                res.append(path[:])
                return 
            # 这里的顺序k开始搜索，可以保证每个数在其他结果中不会被重复用，不需要用另外一个变量st
            for k in range(i, n):  
                if nums[k] > target:return 
                path.append(nums[k])
                dfs(path, target - nums[k], k)
                path.pop()

        dfs([], target, 0)
        return res
      
      
      
      
      # !!!这种 dfs 是错误的，这个代码的意思是，不管怎么样，一定要把第一个元素加入到路径中
      # 这种代码是类似 求 二叉树的 根 到 叶子结点的路径的题
        def dfs(idx, path, target):
            if nums[idx] > target:return 
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

> [!NOTE] **[LeetCode 216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)**
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
            if n == 0 and len(path) == k:  # 踩坑1: 别忘了 个数 要为 k 这个条件！！
                res.append(path[:])
                return 
            for j in range(i, 10):  # 踩坑2：这里遍历的尾部 不能写 n, 因为n在每次递归的过程中 都发生了变化
                if j > n:return  # 剪枝
                path.append(j)
                dfs(path, n - j, j + 1) 
                path.pop()
        dfs([], n, 1)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)**
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
        self.ans=0
        nums.sort()

        def dfs(index,target):
            if target==0:
                self.ans+=1             
            for i in range(len(nums)):
                if nums[i]>target:
                    return
                dfs(i,target-nums[i])
                
        dfs(0,target)
        return self.ans
        
#法二，用dp（其实就是经典的完全背包问题）：每个数字可以用无数次，不区分先后顺序
#dp[i]：表示总和为i的所有方案(有多少个数字无所谓)
#状态转移计算：以最后一个数的数值为转移，最后一个数可以是[nums[0],num[1],nums[2],..nums[k],...,nums[n-1]]
#当最后一个数是nums[k], dp[i]=dp[i-nums[k]]
#由于nums[k]>0, 所以从小到达循环就可以算出来了。

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        n=len(nums)
        if n==0 or target<=0:return 0
        dp=[0 for _ in range(target+1)]
        dp[0]=1
        for i in range(1,target+1):
            for j in range(n):
                if i>=nums[j]:
                    dp[i]+=dp[i-nums[j]]
        return dp[target]


```

<!-- tabs:end -->
</details>

<br>

* * *

### dfs 递归

> [!NOTE] **[LeetCode 112. 路径总和](https://leetcode-cn.com/problems/path-sum/)**
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
        if not root:return False   # 即使root.left不存在，也会返回false；所以不需要写if root.left 这个判断
        if not root.left and not root.right:return root.val == sum 
        return self.hasPathSum(root.left, sum - root.val) or  self.hasPathSum(root.right, sum - root.val)   
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)**
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

> [!NOTE] **[LeetCode 124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)**
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

> [!NOTE] **[LeetCode 386. 字典序排数](https://leetcode-cn.com/problems/lexicographical-numbers/)**
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

```

<!-- tabs:end -->
</details>

<br>

* * *

### dfs 回溯

> [!NOTE] **[LeetCode 51. N 皇后](https://leetcode-cn.com/problems/n-queens/)**
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
#全排列搜索的拓展：每次不仅要判断 列 有没有1，还要判断 对角线 有没有1
#由于每行只能有一个皇后，所以可以依次枚举每一行的皇后放到哪个位置。这样时间复杂度会从下降

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        path = [["."] * n for _ in range(n)]
        col = [False] * n 
        dg = [False] * 2 * n 
        udg = [False] * 2 * n 

        def dfs(u):
           #搜到最后一行的 下一个位置（行）
            if u == n: 
                # 把每一行都转换拼接为一个字符串
                res.append(["".join(path[i]) for i in range(n)])
                return
            
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

> [!NOTE] **[LeetCode 52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/)**
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

> [!NOTE] **[LeetCode 93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses)**
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
"""
直接暴力搜索出所有合法方案。合法的IP地址由四个0到255的整数组成。
我们直接枚举四个整数的位数，然后判断每个数的范围是否在0到255。

搜索题：最重要的是 搜索顺序！ 先搜第一个数 再是第二个数...
1. 要求每个数都在0 - 255之间；2. 必须是一个合法的数：就是【不能有前导0】
搜索顺序：先搜第一个数，然后 第二个， 再三个，最后一个。
时间复杂度分析：一共 n 个数字，n−1 个数字间隔，相当于从 n−1个数字间隔中挑3个断点，所以计算量是 O(Cn-1 -- 3)。【一个组合】
"""

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

> [!NOTE] **[LeetCode 301. 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)**
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

> [!NOTE] **[LeetCode 529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/)**
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

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 679. 24 点游戏](https://leetcode-cn.com/problems/24-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 递归 trick

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

> [!NOTE] **[LeetCode 241. 为运算表达式设计优先级](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/)**
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

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode ]()**
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

> [!NOTE] **[LeetCode 126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)**
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

> [!NOTE] **[LeetCode 127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)**
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

> [!NOTE] **[LeetCode 139. 单词拆分](https://leetcode-cn.com/problems/word-break/)**
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
        f = [False] * (n + 1)  # n个字符，就有n + 1个个半 
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

> [!NOTE] **[LeetCode 140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/)**
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