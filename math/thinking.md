## 习题

> [!NOTE] **[Luogu 对角线](https://www.luogu.com.cn/problem/P2181)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 数学

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using ULL = unsigned long long;

ULL n;

int main() {
    cin >> n;
    cout << n * (n - 1) / 2 * (n - 2) / 3 * (n - 3) / 4 << endl;
    
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

> [!NOTE] **[LeetCode 222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)**
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
    int countNodes(TreeNode* root) {
        if (!root) return 0;
        auto l = root->left, r = root->right;
        int x = 1, y = 1;
        while (l) l = l->left, x ++ ;
        while (r) r = r->right, y ++ ;
        // 本层完全
        if (x == y) return (1 << x) - 1;
        return countNodes(root->left) + 1 + countNodes(root->right);
    }
};
```

##### **Python**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


#暴力解法：
# class Solution:
#     def countNodes(self, root: TreeNode) -> int:
#         if not root:return 0
#         return 1+self.countNodes(root.left)+self.countNodes(root.right)


        

class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:return 0
        l,r=root.left,root.right
        x,y=1,1
        while l:
            l=l.left
            x+=1
        while r:
            r=r.right
            y+=1
        if x==y:
            return pow(2,x)-1
        return self.countNodes(root.left)+1+self.countNodes(root.right)

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 223. 矩形面积](https://leetcode-cn.com/problems/rectangle-area/)**
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
    // 相交面积 = 在x轴投影的相交长度 * 在y轴投影的相交长度
    // 线段相交长度 = 右端点的最小值 - 左端点的最大值  , 小于0则不相交
    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        long long X = min(C, G) + 0ll - max(A, E);
        long long Y = min(D, H) + 0ll - max(B, F);
        return (C - A) * (D - B) - max(0ll, X) * max(0ll, Y) + (G - E) * (H - F);
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

> [!NOTE] **[LeetCode 441. 排列硬币](https://leetcode-cn.com/problems/arranging-coins/)**
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
    // k * (k + 1) / 2 <= n
    // k <= 下界[-1 + sqrt(1 + 8.0 * n)] / 2
    int arrangeCoins(int n) {
        return (-1 + sqrt(1 + 8.0 * n)) / 2;
    }
    int arrangeCoins_2(int n) {
        int line = 0;
        while (n > line)  ++ line, n -= line;
        return line;
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

> [!NOTE] **[LeetCode 483. 最小好进制](https://leetcode-cn.com/problems/smallest-good-base/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1.  假设进制为 k ，所有数位都是 1 且共有 t 位时，表示的数在十进制下为 `k^0 + k^1 + k^2 + … + k^t−1`。
> 2.  显然 t 越大，可能的 k 就会越小。由于 n 最大为 10^18，所以我们可以从 1+⌈logn⌉ 开始向下枚举 t。最坏情况下，t==2 时必定存在k=n−1 满足条件，故枚举的下界为 3。
> 3.  定 t 后，计算一个待验证的 k=⌊t-1√n⌋。
> 4.  验证这一对 k 和 t 是否合法。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    typedef long long LL;
    string smallestGoodBase(string ns) {
        LL n = stoll(ns);
        for (int t = log2(n) + 1; t >= 3; -- t ) {
            LL k = pow(n, 1.0 / (t - 1));
            LL r = 0;
            for (int i = 0; i < t; ++ i ) r = r * k + 1;
            if (r == n) return to_string(k);
        }
        return to_string(n - 1);
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

> [!NOTE] **[LeetCode 628. 三个数的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/)**
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
    // - - +
    // + + +
    // - - -
    int maximumProduct(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        return max(nums[n - 2] * nums[n - 3] * nums[n - 1], nums[0] * nums[1] * nums[n - 1]);
    }
};
```

##### **Python**

```python
# 原数组排序后共有两种情况：
# 1） 最小的两个数<0,乘上最大的数是最大值; 2） 最大的三个数乘积是最大值 ==> 所以两者取max即可。
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()    
        return max(nums[n-1] * nums[0] * nums[1], nums[n-1] * nums[n-2] * nums[n-3])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 754. 到达终点数字](https://leetcode-cn.com/problems/reach-a-number/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数学题 两种思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int reachNumber(int target) {
        target = abs(target);
        int s = 0, p = 0;
        while (s < target || (s - target) % 2)
            p ++ , s += p;
        return p;
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

> [!NOTE] **[LeetCode 775. 全局倒置与局部倒置](https://leetcode-cn.com/problems/global-and-local-inversions/)**
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
    bool isIdealPermutation(vector<int>& A) {
        for (int i = 0; i < A.size(); ++ i )
            if (abs(A[i] - i) > 1)
                return false;
        return true;
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

> [!NOTE] **[LeetCode 781. 森林中的兔子](https://leetcode-cn.com/problems/rabbits-in-forest/)**
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
    // 有v个兔子声称其同色的还有k个
    int get(int k, int v) {
        return v % (k + 1) ? (v / (k + 1) + 1) * (k + 1) : v;
    }

    int numRabbits(vector<int>& answers) {
        unordered_map<int, int> hash;
        for (auto v : answers)
            hash[v] ++ ;
        int res = 0;
        for (auto & [k, v] : hash)
            res += get(k, v);
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int numRabbits(vector<int>& answers) {
        unordered_map<int, int> cnt;
        for (auto x: answers) cnt[x] ++ ;
        int res = 0;
        for (auto [k, v]: cnt)
            // 【v / (k + 1) 向上取整】个组
            res += (v + k) / (k + 1) * (k + 1);
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