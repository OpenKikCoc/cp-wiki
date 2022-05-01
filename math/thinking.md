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

> [!NOTE] **[LeetCode 1197. 进击的骑士](https://leetcode-cn.com/problems/minimum-knight-moves/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 时限较紧，使用 map pair 记录距离超时，故使用静态数组
> 
> 另外 这类题显然可以慢慢推数学结论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 400, M = 800;
    int d[M][M];
    vector<int> dx = {-1, -2, -2, -1, 1, 2, 2, 1}, dy = {-2, -1, 1, 2, 2, 1, -1, -2};
    int minKnightMoves(int x, int y) {
        memset(d, -1, sizeof d);
        queue<PII> q;
        q.push({x + N, y + N});
        d[x + N][y + N] = 0;
        while (q.size()) {
            auto [x, y] = q.front(); q.pop();
            if (x == N && y == N) return d[x][y];
            for (int i = 0; i < 8; ++ i ) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx < 0 || nx >= M || ny < 0 || ny >= M || d[nx][ny] != -1) continue;
                q.push({nx, ny});
                d[nx][ny] = d[x][y] + 1;
            }
        }
        return -1;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int minKnightMoves(int x, int y) {
        x = abs(x), y = abs(y);
        // 特判
        if (x + y == 1) return 3;
        // (x + 1) / 2,(y + 1) / 2      走到x，y所需要的最小步数
        // (x + y + 2) / 3              每次最多走三步，所以走到 x + y 的最小步数为 (x + y + 2) / 3
        int res = max(max((x + 1) / 2, (y + 1) / 2), (x + y + 2) / 3);
        // 结论：如果x,y 同奇同偶 只有偶数步才能走到x,y点
        //      如果    一奇一偶 奇数步才能走到x,y点
        res += (res ^ x ^ y) & 1;
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

> [!NOTE] **[LeetCode 1330. 翻转子数组得到最大的数组值](https://leetcode-cn.com/problems/reverse-subarray-to-maximize-array-value/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 比赛的时候n^2 TLE一发 尝试优化表达式 思路可以参考 [这里](https://leetcode-cn.com/problems/reverse-subarray-to-maximize-array-value/solution/onzuo-fa-jie-jue-ci-wen-ti-by-hu-tu-tu-7/)
>
> 更进一步：找到一个最大一个最小
>
> 贪心：
>
> >   一个数组的[数组值] 即为 na ,可以表示为:
> >
> >   $$na = \sum_{i=1}^{n} max(nums[i], nums[i−1]) − \sum_{i=1}^{n} min(nums[i], nums[i−1])$$
> >
> >   $$na′ = na + 2 * maxMinu − 2 * minAdd$$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ TLE**

```cpp
class Solution {
public:
    // 翻转子数组不会影响子数组内部的数组值 只会影响子数组边界
    // nums[l-1] nums[l] ... nums[r] nums[r+1] => nums[l-1] nums[r] ... nums[l]
    // nums[r+1] 所以枚举 l r 即可 区间dp dp[i][j] 为翻转 i~j 带来的增益 增益 =
    // abs(nums[l-1] - nums[r]) + abs(nums[l] - nums[r+1]) - abs(nums[l-1] -
    // nums[l]) - abs(nums[r] - nums[r+1]) 直接这样算 n^2 超时
    int maxValueAfterReverse(vector<int>& nums) {
        int osum = 0;
        int n = nums.size();
        for (int i = 0; i < n - 1; ++i) osum += abs(nums[i] - nums[i + 1]);
        vector<vector<int>> dp(n, vector<int>(n));
        int maxv = 0;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int lc = (i - 1 >= 0) ? abs(nums[i - 1] - nums[j]) -
                                            abs(nums[i - 1] - nums[i])
                                      : 0;
                int rc = (j + 1 < n) ? abs(nums[i] - nums[j + 1]) -
                                           abs(nums[j] - nums[j + 1])
                                     : 0;
                dp[i][j] = lc + rc;
                maxv = max(maxv, dp[i][j]);
            }
        }
        return osum + maxv;
    }
};
```

##### **C++ 优化**

```cpp
class Solution {
public:
    //
    int maxValueAfterReverse(vector<int> &nums) {
        int osum = 0;
        int n = nums.size(), maxv = 0;
        for (int i = 0; i < n - 1; ++i) osum += abs(nums[i] - nums[i + 1]);
        // 边界情况
        for (int i = 0; i < n; ++i) {
            if (i != n - 1)
                maxv = max(maxv, abs(nums[0] - nums[i + 1]) -
                                     abs(nums[i] -
                                         nums[i + 1]));  // 左端点为0右端点为i
            if (i != 0)
                maxv = max(
                    maxv,
                    abs(nums[n - 1] - nums[i - 1]) -
                        abs(nums[i] - nums[i - 1]));  // 右端点为n-1,左端点为i
        }

        int mx[4] = {1, 1, -1, -1};
        int my[4] = {1, -1, 1, -1};
        // 枚举四种情况
        for (int i = 0; i < 4; ++i) {
            vector<int> v1, v2;
            for (int j = 0; j < n - 1; ++j) {
                int a = mx[i] * nums[j];
                int b = my[i] * nums[j + 1];
                int cur = abs(nums[j] - nums[j + 1]);
                v1.push_back(a + b - cur);
                v2.push_back(a + b + cur);
            }
            int a = get_max(v1);
            int b = get_min(v2);
            maxv = max(maxv, a - b);
        }
        return osum + maxv;
    }

    int get_max(vector<int> &v) {
        int res = INT_MIN;
        for (auto x : v) res = max(res, x);
        return res;
    }
    int get_min(vector<int> &v) {
        int res = INT_MAX;
        for (auto x : v) res = min(res, x);
        return res;
    }
};
```

##### **C++ 更进一步的贪心**

```cpp
class Solution {
public:
    int maxValueAfterReverse(vector<int>& nums) {
        int ans = 0, n = nums.size();
        for (int i = 0; i < n - 1; i++) ans += abs(nums[i + 1] - nums[i]);
        int tmp = 0;
        for (int i = 1; i < n - 1; i++) {
            tmp = max(tmp,
                      abs(nums[i + 1] - nums[0]) - abs(nums[i + 1] - nums[i]));
            tmp = max(tmp, abs(nums[n - 1] - nums[i - 1]) -
                               abs(nums[i] - nums[i - 1]));
        }
        int a = INT_MIN, b = INT_MAX;
        for (int j = 0; j < n - 1; j++) {
            a = max(a, min(nums[j], nums[j + 1]));
            b = min(b, max(nums[j], nums[j + 1]));
        }
        return ans + max(tmp, 2 * (a - b));
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

> [!NOTE] **[LeetCode 1359. 有效的快递序列数目](https://leetcode-cn.com/problems/count-all-valid-pickup-and-delivery-options/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 位置插入
> 
> 递推

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int MOD = 1e9+7;
    int countOrders(int N) {
        long long P = 1;
        for (int n = 2; n <= N; n++) {
            long long a = 2 * (n - 1) + 1;
            long long b = a * (a - 1) / 2 + a;
            P = (b * P) % MOD;
        }
        return P;
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

> [!NOTE] **[LeetCode 1404. 将二进制表示减到 1 的步骤数](https://leetcode-cn.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/)**
> 
> 题意: 
> 
> 二进制形式表示的数字 s 返回按下述规则将其减少到 1 所需要的步骤数：
> 
> 如果当前数字为偶数，则将其除以 2 。 如果当前数字为奇数，则将其加上 1 。

> [!TIP] **思路**
> 
> - 模拟
> 
>   如果当前数字为偶数，则将其除以 2 。 如果当前数字为奇数，则将其加上 1 。
> 
> - 计数
> 
>   只有在一开始的时候，我们才需要考虑字符串最低位为0的情况，我们通过若干次操作删去低位的所有0；
> 
>   在任意时刻，字符串的最低位均为1。如果有k个1那么我们需要k+1 次操作（1次加一操作和k次除二操作）将所有的1变为0并删除。并且在这 k+1 次操作后，原本最靠近右侧的那个0变为了1。这也解释了为什么我们不需要考虑最低位为0的情况了。
> 
> - trick TODO
> 


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 模拟**

```cpp
class Solution {
public:
    string add(string s) {
        int len = s.size();
        int flag = 1;
        for (int i = len - 1; i >= 0; --i) {
            s[i] = s[i] + flag;
            if (s[i] == '2') {
                flag = 1;
                s[i] = '0';
            } else
                flag = 0;
        }
        if (flag) return "1" + s;
        return s;
    }
    int numSteps(string s) {
        if (s.size() == 1 && s[0] == '1') return 0;
        int res = 0;
        string ns;
        for (;;) {
            if (s[s.size() - 1] == '0') {
                ns = s.substr(0, s.size() - 1);
            } else
                ns = add(s);
            ++res;
            // cout <<"res="<<res<<" ns="<<ns<<endl;
            if (ns.size() == 1 && ns[0] == '1') break;
            s = ns;
        }
        return res;
    }
};
```

##### **C++ 计数**

```cpp
class Solution {
public:
    int numSteps(string s) {
        int n = s.size();
        int ans = 0;
        // meet1 记录我们有没有遇见过字符 1
        bool meet1 = false;
        // 从后向前遍历字符
        for (int i = n - 1; i >= 0; --i) {
            if (s[i] == '0') {
                // 如果当前字符为 0，分为两种情况
                // (1) 还没有遇见过字符 1，那么这个 0 是字符串低位的 0，需要一次除二操作
                // (2) 遇见过字符 1，那么这个 0 会因为它右侧的某次加一操作变为 1，因此它需要一次加一和一次除二操作
                ans += (meet1 ? 2 : 1);
            } else {
                // 如果当前字符为 1，分为两种情况
                // (1) 还没有遇见过字符 1，那么这个 1 需要一次加一和一次除二操作
                //     这里需要考虑一种特殊情况，就是这个 1 是字符串最左侧的 1，它并不需要任何操作
                // (2) 遇见过字符 1，那么这个 1 会因为它右侧的某次加一操作变为 0，因此它只需要一次除二操作
                if (!meet1) {
                    if (i != 0) {
                        ans += 2;
                    }
                    meet1 = true;
                }
                else {
                    ++ans;
                }
            }
        }
        return ans;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int numSteps(string s) {
        int index = s.length() - 1;
        int ans = 0;
        while (index >= 0) {
            if (s[index] == '0') {
                ++ans;
                --index;
            } else {
                if (index == 0) break;
                // 计算连续进位
                while ((index >= 0) && (s[index] == '1')) {
                    ++ans;
                    --index;
                }
                ++ans;
                if (index >= 0) { s[index] = '1'; }
                continue;
            }
        }
        return ans;
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

> [!NOTE] **[LeetCode 1518. 换酒问题](https://leetcode-cn.com/problems/water-bottles/)**
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
    int numWaterBottles(int numBottles, int numExchange) {
        int res = numBottles;
        while (numBottles >= numExchange) {
            res += numBottles / numExchange;
            numBottles = numBottles / numExchange + numBottles % numExchange;
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

> [!NOTE] **[LeetCode 1523. 在区间范围内统计奇数数目](https://leetcode-cn.com/problems/count-odd-numbers-in-an-interval-range/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然：对于某个数 x ，小于等于它的所有奇数个数为 `(x + 1) / 2`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countOdds(int low, int high) { return (high + 1) / 2 - low / 2; }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1526. 形成目标数组的子数组最少增加次数](https://leetcode-cn.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典结论题
> 
> TODO: AcWing or Luogu 有个类似题目 没找到 整理到一起 ==> 就是下面那个 【铺设道路】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minNumberOperations(vector<int>& target) {
        int res = target[0], n = target.size();
        for (int i = 1; i < n; ++i)
            res += max(target[i] - target[i - 1], 0);
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

> [!NOTE] **[Luogu [NOIP2018 提高组] 铺设道路](https://www.luogu.com.cn/problem/P5019)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典贪心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 1e6 + 10;

int n;
LL d[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> d[i];
    
    LL res = 0;
    stack<int> st;
    // i = 0 -----> d[i] = 0
    for (int i = 1; i <= n; ++ i )
        if (d[i] > d[i - 1])
            res += d[i] - d[i - 1];
    cout << res << endl;
    
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

> [!NOTE] **[LeetCode 1675. 数组的最小偏移量](https://leetcode-cn.com/problems/minimize-deviation-in-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 堆
> 
> 显然奇数只能乘 2 后再除 2 两种可能，处理后在堆中只除 2

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    /*
    较显然的做法：
        第一步:先把所有奇数乘 2 , 这样就等价于只有操作 1
        第二步:操作 1 的只会减少某个数的值,而只有减少最大值结果才可能更优.
        第三步:使用平衡树或堆维护最大值,直到最大值为奇数不能再操作为止.
    考虑优化:
        https://leetcode-cn.com/problems/minimize-deviation-in-array/solution/yi-chong-fu-za-du-geng-di-de-zuo-fa-by-heltion-2/
    */
    int minimumDeviation(vector<int>& nums) {
        int n = nums.size(), mi = INT_MAX, res = INT_MAX;
        priority_queue<int> q;
        for (auto v : nums) {
            if (v & 1) v *= 2;
            q.push(v);
            mi = min(mi, v);
        }
        while (true) {
            int t = q.top(); q.pop();
            res = min(res, t - mi);
            if (t & 1) break;   // 奇数不能再除2 最大值不可能再变 结果也就不会更小
            t >>= 1;
            mi = min(mi, t);
            q.push(t);
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

> [!NOTE] **[LeetCode 1716. 计算力扣银行的钱](https://leetcode-cn.com/problems/calculate-money-in-leetcode-bank/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推公式 花时间就比暴力长点

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    // 28 35 
    // (28 + (w - 1) * 7 + 28) / 2 * w
    // 
    // (w + 1 + d + w) / 2 * d
    int totalMoney(int n) {
        int w = n / 7, d = n % 7;
        int v1 = (49 + 7 * w) * w / 2;
        int v2 = (w + d + 1 + w) * d / 2;
        // cout << v1 << ' ' << v2 << endl;
        return v1 + v2;
    }
};
```

##### **C++ 2 暴力**

```cpp
class Solution {
public:
    int totalMoney(int n) {
        int res = 0;
        for (int i = 1; i <= n; i ++ ) {
            int r = (i - 1) / 7, c = (i - 1) % 7;
            res += r + c + 1;
        }
        return res;
    }
};
```

##### **C++ 3 暴力**

```cpp
class Solution {
public:
    int totalMoney(int n) {
        int ans = 0;
        int cur = 0;
        for (int i = 0; i < n; ++i) {
            if (i % 7 == 0) {
                cur = i / 7 + 1;
            }
            else {
                ++cur;
            }
            ans += cur;
        }
        return ans;
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

> [!NOTE] **[LeetCode 1726. 同积元组](https://leetcode-cn.com/problems/tuple-with-same-product/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一个四元组改变顺序可以拓展为 8 个可行解
> 
> > `v * (v - 1) / 2 * 8`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int tupleSameProduct(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> hash;
        for (int i = 0; i < n; ++ i )
            for (int j = i + 1; j < n; ++ j )
                ++ hash[nums[i] * nums[j]];
    
        int res = 0;
        for (auto & [k, v] : hash) res += v * (v - 1) * 4;
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

> [!NOTE] **[LeetCode 1780. 判断一个数字是否可以表示成三的幂的和](https://leetcode-cn.com/problems/check-if-number-is-a-sum-of-powers-of-three/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思考转化三进制即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool checkPowersOfThree(int n) {
        for (; n; n /= 3)
            if (n % 3 == 2)
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

> [!NOTE] **[LeetCode 1806. 还原排列的最少操作步数](https://leetcode-cn.com/problems/minimum-number-of-operations-to-reinitialize-a-permutation/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟即可 前面在想数学解法花了一些时间
> 
> 数学：
> 
> 本质上，每一次 get 都是一次矩阵映射，显然具有数学性质
> 
> 推导易知 0 与 n-1 位置的数值永远不变，其余元素有 $f[i] = 2i mod n-1$
> 
> 要还原序列，则需 $f[i]^k = 2^k*i = i mod n-1$
> 
> 以 1 为例， $f[1]^k = 2^k = 1 mod n-1$ ，故找到最小 k 使得 $2^k = 1 mod n-1$ 即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 模拟**

```cpp
class Solution {
public:
    int n;
    vector<int> get(vector<int> & t) {
        vector<int> nt(n);
        for (int i = 0; i < n; ++ i )
            if (i % 2)
                nt[i] = t[n / 2 + (i - 1) / 2];
            else
                nt[i] = t[i / 2];
        return nt;
    }
    
    int reinitializePermutation(int n) {
        this->n = n;
        vector<int> t;
        for (int i = 0; i < n; ++ i )
            t.push_back(i);
        auto tar = t;
        int res = 0;
        for (;;) {
            res ++ ;
            auto nt = get(t);
            if (nt == tar)
                break;
            else
                t = nt;
        }
        return res;
    }
};
```

##### **C++ 数学**

```cpp
class Solution {
public:
    int reinitializePermutation(int n) {
        if (n == 2)
            return 1;

        int k = 1;
        int pow2 = 2;
        while (pow2 != 1) {
            k ++ ;
            pow2 = pow2 * 2 % (n - 1);
        }
        return k;
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

> [!NOTE] **[LeetCode 1862. 向下取整数对和](https://leetcode-cn.com/problems/sum-of-floored-pairs/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 正解是计数 BIT会超时

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ BIT TLE**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 200010;
    int tr[N];
    
    int lowbit(int x) {
        return x & -x;
    }
    
    void add(int x, int c) {
        for (int i = x; i < N; i += lowbit(i)) tr[i] += c;
    }
    
    LL sum(int x) {
        LL res = 0;
        for (int i = x; i; i -= lowbit(i)) res += tr[i];
        return res;
    }
    
    const int MOD = 1e9 + 7;
    vector<int> ns;
    int n;
    
    LL get(int x) {
        int top = ns.back() / x;
        LL ret = 0;
        for (int i = 1; i <= top; ++ i ) {
            LL cnt = sum(x * (i + 1) - 1) - sum(x * i - 1);
            ret = (ret + i * cnt) % MOD;
        }
        return ret;
    }
    
    // 显然只计算某个值和后面比它大的
    int sumOfFlooredPairs(vector<int>& nums) {
        this->ns = nums;
        n = ns.size();
        memset(tr, 0, sizeof tr);
        for (auto v : ns)
            add(v, 1);
        sort(ns.begin(), ns.end());
        LL res = 0;
        for (auto v : ns)
            res = (res + get(v)) % MOD;
        return res;
    }
};
```

##### **C++ 计数**

```cpp
typedef long long LL;

const int N = 100010, MOD = 1e9 + 7;

int s[N];

class Solution {
public:
    int sumOfFlooredPairs(vector<int>& nums) {
        memset(s, 0, sizeof s);
        for (auto x: nums) s[x] ++ ;
        for (int i = 1; i < N; i ++ ) s[i] += s[i - 1];
        int res = 0;
        
        for (int i = 1; i < N; i ++ ) {
            for (int j = 1; j * i < N; j ++ ) {
                int l = j * i, r = min(N - 1, (j + 1) * i - 1);
                int sum = (LL)(s[r] - s[l - 1]) * j % MOD;
                res = (res + (LL)sum * (s[i] - s[i - 1])) % MOD;
            }
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

> [!NOTE] **[LeetCode 1904. 你完成的完整对局数](https://leetcode-cn.com/problems/the-number-of-full-rounds-you-have-played/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 统计其中有多少个 15 整倍数即可，注意加 15 的偏移量保证其为完整的回合
> 
> - 也可以数学解法，其实就是去掉个 for 循环

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int get(string s) {
        int h = (s[0] - '0') * 10 + s[1] - '0';
        int m = (s[3] - '0') * 10 + s[4] - '0';
        return h * 60 + m;
    }
    
    int numberOfRounds(string startTime, string finishTime) {
        int st = get(startTime), ed = get(finishTime);
        
        if (st > ed)
            ed += get("24:00");
        
        int res = 0;
        for (int i = st + 15; i <= ed; ++ i )
            if (i % 15 == 0)
                res ++ ;
        
        return res;
    }
};
```

##### **C++ 数学**

```cpp
// yxc
class Solution {
public:
    int get(string s) {
        int a, b;
        sscanf(s.c_str(), "%d:%d", &a, &b);
        return a * 60 + b;
    }
    
    int numberOfRounds(string a, string b) {
        int x = get(a), y = get(b);
        if (x > y) y += 1440;
        x = (x + 14) / 15, y /= 15;
        return max(0, y - x);
    }
};
```

##### **Python**

```python
# python
class Solution:
    def numberOfRounds(self, startTime: str, finishTime: str) -> int:
        def get(s):
            h = int(s[0]) * 10 + int(s[1])
            m = int(s[3]) * 10 + int(s[4])
            return h * 60 + m 
        
        st, ed = get(startTime), get(finishTime)
        
        if st > ed:
            ed += get("24:00")
        
        res = 0 
        for i in range(st + 15, ed + 1):
            if i % 15 == 0:
                res += 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1954. 收集足够苹果的最小花园周长](https://leetcode-cn.com/problems/minimum-garden-perimeter-to-collect-enough-apples/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始题意理解错了。。
> 
> - 显然可以直接递增
> 
>   1. 以原点为中心
>   2. 为正方形
> 
>   显然每次 l ++ 会增加 3 * x * x
> 
>   对于全局来说，每次 l ++ 会增加 12 * x * x
> 
>   这样累加即可
> 
> - 也可以直接二分在某一象限内的小正方形边长，也可通过公式累加
> 
>   平方和公式如下二分



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 比赛代码
class Solution {
public:
    using LL = long long;
    
    LL get(LL x) {
        return (x - 1) * x * 3 + x * 3;
    }
    
    long long minimumPerimeter(long long neededApples) {
        LL tot = 0, l = 0;
        while (tot * 4 < neededApples) {
            l ++ ;
            tot += get(l);
        }
        return l * 8;
    }
};
```

##### **C++ 二分**

```cpp
class Solution {
public:
    using LL = long long;

    LL get(LL x) {
        // x 为在某一象限内某一维度的长
        // 每增加一圈 增加数量为 12*i*i
        // 平方和公式 得整个正方形内苹果个数
        // 1^2 + 2^2 + 3^2 ... + n^2 = n * (n + 1) * (2 * n + 1) / 6
        // 注意需乘12
        return 2 * x * (1 + x) * (2 * x + 1);
    }

    long long minimumPerimeter(long long neededApples) {
        LL l = 0, r = 1e6;
        while (l < r) {
            LL m = l + r >> 1;
            if (get(m) < neededApples)
                l = m + 1;
            else
                r = m;
        }
        return l * 8;
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

> [!NOTE] **[LeetCode 1963. 使字符串平衡的最小交换次数](https://leetcode-cn.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有点思维 其实分析知 `(stk + 1) / 2` （向上取整）即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    int minSwaps(string s) {
        this->n = s.size();
        int stk = 0, cnt = 0;
        for (auto c : s) {
            if (c == '[')
                stk ++ ;
            else {
                if (stk > 0)
                    stk -- ;
                else {
                    cnt ++ ;
                }
            }
        }
        
        return (stk + 1) / 2;
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

> [!NOTE] **[LeetCode 1969. 数组元素的最小非零乘积](https://leetcode-cn.com/problems/minimum-non-zero-product-of-the-array-elements/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 分析知：
>
> - 所有元素的和总是一样的（交换 `0` `1` 本质是加减某个 `2 次幂`）
>
> - 乘积最小则各元素的差别应尽可能大（向两侧离散）
>
>   以 n = 3 为例
>
>   ```
>   0 1 2 3 | 4 5 6 7
>   -->
>   0 1 6 1 | 6 1 6 7
>   ```
>
>   增加一个数 `0` 与 `7` 对应，则余下位置一定是 `1` 和 `(1 << p) - 1 - 1` 一一对应
>
> 显然最终答案即 `res = ((1 << p) - 1) * pow((1 << p) - 1 - 1, (1 << p - 1) - 1)`
>
> > 以 3 为例
> >
> > 最终答案即 7 * pow(6, 3)
>
> **注意写 long long `1ll << p` wa 2发**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const LL MOD = 1e9 + 7;
    
    LL quick_pow(LL a, LL b) {
        a %= MOD;
        LL ret = 1;
        while (b) {
            if (b & 1)
                ret = ret * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return ret;
    }
    
    int minNonZeroProduct(int p) {
        LL t = (1ll << p) - 1ll;
        LL g = (t - 1);
        t %= MOD, g %= MOD;
        LL res = t;
        res = res * quick_pow(g, (1ll << p - 1) - 1) % MOD;
        return res;
    }
};
```

##### **C++**

```cpp
// Heltion
using LL = long long;
constexpr LL mod = 1'000'000'007;
LL power(LL a, LL r) {
    LL res = 1;
    for (a %= mod; r; r >>= 1, a = a * a % mod)
        if (r & 1) res = res * a % mod;
    return res;
}
class Solution {
public:
    int minNonZeroProduct(int p) {
        return ((1LL << p) - 1) % mod * power((1LL << p) - 2, (1LL << (p - 1)) - 1) % mod;
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

> [!NOTE] **[LeetCode 1980. 找出不同的二进制字符串](https://leetcode-cn.com/problems/find-unique-binary-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 考虑显然每个二进制串唯一对应一个数字，转化数字标识即可
> 
> - 赛榜第一 sfiction 巨巨的优雅代码：评论区称之为 【康托对角线】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 65536; // 2^16 = 1024*64
    
    bool st[N];
    
    string findDifferentBinaryString(vector<string>& nums) {
        int n = nums.size();
        for (auto & s : nums) {
            int v = 0;
            for (int i = 0; i < n; ++ i )
                if (s[i] == '1')
                    v += 1 << i;
            st[v] = true;
        }
        for (int i = 0; i < N; ++ i )
            if (!st[i]) {
                string res(n, '0');
                for (int j = 0; j < n; ++ j )
                    if (i >> j & 1)
                        res[j] = '1';
                return res;
            }
        return "";
    }
};
```


##### **C++ sfiction**

```cpp
// 评论区称之为 【康托对角线】
class Solution {
public:
    string findDifferentBinaryString(vector<string>& a) {
        int n= a.size();
        string s;
        for (int i = 0; i < n; ++i)
            s.push_back(a[i][i] ^ 1);
        return s;
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

> [!NOTE] **[LeetCode 2183. 统计可以被 K 整除的下标对数目](https://leetcode-cn.com/problems/count-array-pairs-divisible-by-k/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单数学思维 **注意如何一步一步优化**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;
    
    LL cnt[N], hash[N];
    
    long long countPairs(vector<int>& nums, int k) {
        for (auto x : nums)
            cnt[x] ++ ;
        
        // 记忆: 每个数的倍数出现了多少次
        // O(nlogn)
        // unordered_map<int, int> hash; ==> TLE
        for (int i = 1; i < N; ++ i )
            if (cnt[i])
                hash[i] = cnt[i];
        for (int i = 1; i < N; ++ i )
            for (int j = i + i; j < N; j += i )
                hash[i] += hash[j];
        
        LL res = 0;
        for (int i = 1; i < N; ++ i )
            if (cnt[i]) {
                LL t = k / __gcd(k, i);
                res += cnt[i] * hash[t];
                if ((LL)i * i % k == 0)
                    res -= cnt[i];
            }
        
        return res / 2;
    }
};
```

##### **C++ 直观 TLE**

```cpp
// TLE 114 / 115
class Solution {
public:
    using LL = long long;
    const static int N = 1e5 + 10;
    
    LL cnt[N], s[N];
    
    long long countPairs(vector<int>& nums, int k) {
        for (auto x : nums)
            cnt[x] ++ ;
        for (int i = 1; i < N; ++ i )
            s[i] = s[i - 1] + cnt[i];
        
        LL res = 0, n = nums.size();
        for (int i = 1; i < N; ++ i )
            if (cnt[i]) {
                if ((LL)i * i % k == 0)
                    res += cnt[i] * (cnt[i] - 1) / 2;
                
                if ((LL)i % k == 0) {
                    res += cnt[i] * s[i - 1];
                } else {
                    LL t = k / __gcd(k, i);
                    // 需要优化此处
                    for (int j = t; j < i; j += t )
                        if ((LL)i * j % k == 0)
                            res += cnt[i] * cnt[j];
                }
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

> [!NOTE] **[LeetCode 理财产品](https://leetcode-cn.com/contest/cnunionpay-2022spring/problems/I4mOGz/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数学计算 细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int MOD = 1e9 + 7;
    
    int maxInvestment(vector<int>& product, int limit) {
        sort(product.begin(), product.end());
        reverse(product.begin(), product.end());
        
        int n = product.size();
        LL s = 0, tot = 0, res = 0;
        for (int i = 0; i < n; ++ i ) {
            s += product[i], tot += (1LL + product[i]) * product[i] / 2;
            LL t = s - limit;
            if (t > 0) {
                // 平均还剩a 有b个剩余a+1
                LL a = t / (i + 1), b = t % (i + 1);
                // 为了后面ret计算准确 如果当前不合法直接break
                if (a > product[i])
                    break;
                
                {
                    LL ret = 0;
                    if (b)
                        ret += (0LL + a + 1) * b;
                    ret += (1LL + a) * a / 2 * (i + 1);
                    res = max(res, tot - ret);
                // cout << " i = " << i << " p[i] = " << product[i] << " t = " << t << " a = " << a << " b = " << b << " tot = " << tot << " ret = " << ret << " tot-ret = " << tot - ret << endl;
                }
            } else {
                // cout << " i = " << i << " p[i] = " << product[i] << " t = " << t << " tot = " << tot << endl;
                res = max(res, tot);
            }
        }
        return res % MOD;
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

> [!NOTE] **[Codeforces B. Routine Problem](https://codeforces.com/problemset/problem/337/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 使用公式转化 gcd 而非浮点数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Routine Problem
// Contest: Codeforces - Codeforces Round #196 (Div. 2)
// URL: https://codeforces.com/problemset/problem/337/B
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 数学题 浮点数的思路是错误的
// WA https://codeforces.com/contest/337/submission/109684884
// 正解 gcd
// 只有两种情况，长等于长，宽等于宽
// 1. 长等于长，即有c = a，d = a*d/c;  已使用的宽占比为 a*d/(b*c)
// 2. 宽等于宽     d = b，c = b*c/d;  已使用的长占比为 b*c/(a*d)

int main() {
    int a, b, c, d;
    cin >> a >> b >> c >> d;

    a *= d, b *= c;
    if (a > b)
        swap(a, b);

    a = b - a;
    c = __gcd(a, b);

    cout << a / c << '/' << b / c << endl;

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

> [!NOTE] **[Codeforces B. Friends and Presents](https://codeforces.com/problemset/problem/483/B)**
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
// Problem: B. Friends and Presents
// Contest: Codeforces - Codeforces Round #275 (Div. 2)
// URL: http://codeforces.com/problemset/problem/483/B
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 二分错误
// http://codeforces.com/contest/483/submission/110988631

int main() {
    int a, b, x, y;
    cin >> a >> b >> x >> y;

    // https://www.luogu.com.cn/blog/over-knee-socks/solution-cf483b-o1
    int v = 0;
    v = max(v, (int)floor((double)(a + b - 1) * x * y / (x * y - 1)));
    v = max(v, (int)floor((double)(a - 1) * x / (x - 1)));
    v = max(v, (int)floor((double)(b - 1) * y / (y - 1)));

    cout << v + 1 << endl;

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

> [!NOTE] **[Codeforces Pipeline](http://codeforces.com/problemset/problem/287/B)**
> 
> 题意: 
> 
> 不超过 k 的自然数 x 每个用一个，产生 `x-1` 的效果，最终要从 1 到 n
> 
> 求能否达成，能的话最少多少个数

> [!TIP] **思路**
> 
> 显然贪心每次取最大，最后取个补足的数。连续贪心的个数二分计算即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Pipeline
// Contest: Codeforces - Codeforces Round #176 (Div. 2)
// URL: https://codeforces.com/problemset/problem/287/B
// Memory Limit: 256 MB
// Time Limit: 400 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;

LL n, k;

int main() {
    cin >> n >> k;
    n--, k--;

    if (n == 0)
        cout << 0 << endl;
    else if (n > (k + 1) * k / 2)
        cout << -1 << endl;
    else {
        LL l = 0, r = k;
        while (l < r) {
            LL m = l + r >> 1;

            LL s = (k + (k - m + 1)) * m / 2;
            if (s >= n)
                r = m;
            else
                l = m + 1;
        }
        cout << l << endl;
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

> [!NOTE] **[Codeforces Count Good Substrings](http://codeforces.com/problemset/problem/451/D)**
> 
> 题意: 
> 
> 一个字符串是“好的”，当且仅当合并其中的连续区间后，它是一个回文串。比如 `aabba` 是好的，因为在合并后它变成了 `aba`
> 
> 给你一个字符串，现在要你分别求出长度为奇数和偶数的“好的”子串数量。（提示：不是本质不同的子串，不允许空串）

> [!TIP] **思路**
> 
> 因为这些字符串中的字符只有 a, b 所以首位相同的字串都可以满足
> 
> 这样就分别统计奇数和偶数位置的字符的个数，然后相互组合就可以

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Count Good Substrings
// Contest: Codeforces - Codeforces Round #258 (Div. 2)
// URL: https://codeforces.com/problemset/problem/451/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;

// 因为这些字符串中的字符只有 a, b 所以首位相同的字串都可以满足
// 这样就分别统计奇数和偶数位置的字符的个数，然后相互组合就可以

int main() {
    string s;
    cin >> s;

    int n = s.size();
    LL x = 0, y = 0;
    for (LL i = 0, oa = 0, ob = 0, ea = 0, eb = 0; i < n; ++i) {
        x++;  // single char
        if (i & 1) {
            if (s[i] == 'a')
                x += oa, y += ea, oa++;
            else
                x += ob, y += eb, ob++;
        } else {
            if (s[i] == 'a')
                x += ea, y += oa, ea++;
            else
                x += eb, y += ob, eb++;
        }
    }

    cout << y << ' ' << x << endl;

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

> [!NOTE] **[Codeforces Andrey and Problem](http://codeforces.com/problemset/problem/442/B)**
> 
> 题意: 
> 
> Andrey 请教朋友 OI 题，他知道第 $i$ 个朋友解答出问题题的概率 $p_{i}$
> 
> 他只想让朋友解答出且仅解答出 1 道题。
> 
> 求概率最大。

> [!TIP] **思路**
> 
> 贪心 + 数学 1A

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Andrey and Problem
// Contest: Codeforces - Codeforces Round #253 (Div. 1)
// URL: https://codeforces.com/problemset/problem/442/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using PDD = pair<double, double>;
const static int N = 110;

int n;
PDD p[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> p[i].first;
        p[i].second = 1.0 - p[i].first;
    }
    sort(p + 1, p + n + 1);

    double p1 = 0, p2 = 1.0, res = 0;
    for (int i = n; i >= 1; --i) {
        auto [px, py] = p[i];
        double new_p1 = p1 * py + px * p2;
        double new_p2 = p2 * py;
        p1 = new_p1, p2 = new_p2;
        // cout << " i = " << i << " p1 = " << p1 << " p2 = " << p2 << endl;
        res = max(res, p1);
    }

    printf("%.12lf\n", res);

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
