本页面将简要介绍双指针。

双指针是一种简单而又灵活的技巧和思想，单独使用可以轻松解决一些特定问题，和其他算法结合也能发挥多样的用处。

双指针顾名思义，就是同时使用两个指针，在序列、链表结构上指向的是位置，在树、图结构中指向的是节点，通过或同向移动，或相向移动来维护、统计信息。

接下来我们来看双指针的几个具体使用方法。

## 维护区间信息

如果不和其他数据结构结合使用，双指针维护区间信息的最简单模式就是维护具有一定单调性，新增和删去一个元素都很方便处理的信息，就比如正数的和、正整数的积等等。

> [!NOTE] **例题 1 [leetcode 713. 乘积小于 K 的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/)**
> 
> 给定一个长度为 $n$ 的正整数数组 $\mathit{nums}$ 和整数 $k$, 找出该数组内乘积小于 $k$ 的连续子数组的个数。$1 \leq n \leq 3 \times 10^4, 1 \leq nums[i] \leq 1000, 0 \leq k \leq 10^6$

设两个指针分别为 $l,r$, 另外设置一个变量 $\mathit{tmp}$ 记录 $[l,r]$ 内所有数的乘积。最开始 $l,r$ 都在最左面，先向右移动 $r$，直到第一次发现 $\mathit{tmp}\geq k$,  这时就固定 $r$，右移 $l$，直到 $\mathit{tmp}\lt k$。那么对于每个 $r$，$l$ 是它能延展到的左边界，由于正整数乘积的单调性，此时以 $r$ 为右端点的满足题目条件的区间个数为 $r-l+1$ 个。

```cpp
int numSubarrayProductLessThanK(vector<int>& nums, int k) {
    long long ji = 1ll, ans = 0;
    int l = 0;
    for (int i = 0; i < nums.size(); ++i) {
        ji *= nums[i];
        while (l <= i && ji >= k) ji /= nums[l++];
        ans += i - l + 1;
    }
    return ans;
}
```

使用双指针维护区间信息也可以与其他数据结构比如差分、单调队列、线段树、主席树等等结合使用。另外将双指针技巧融入算法的还有莫队，莫队中将询问离线排序后，一般也都是用两个指针记录当前要处理的区间，随着指针一步步移动逐渐更新区间信息。

接下来看一道在树上使用双指针并结合树上差分的例题：

> [!NOTE] **例题 2 [luogu P3066 Running Away From the Barn G](https://www.luogu.com.cn/problem/P3066)**
> 
> 给定一颗 $n$ 个点的有根树，边有边权，节点从 1 至 $n$ 编号，1 号节点是这棵树的根。再给出一个参数 $t$，对于树上的每个节点 $u$，请求出 $u$ 的子树中有多少节点满足该节点到 $u$ 的距离不大于 $t$。数据范围：$1\leq n \leq 2\times 10^5,1 \leq t \leq 10^{18},1 \leq p_i \lt i,1 \leq w_i \leq 10^{12}$

从根开始用 dfs 遍历整棵树，使用一个栈来记录当前的这条树链，设一个指针 $u$ 指向当前节点，另一个指针 $p$ 指在此树链中的某个位置，使得 $p$ 为与 $u$ 距离不大于 $t$ 的节点中深度最小的。这里的维护方式和在序列上基本一致。此时 $u$ 对 $p$ 到 $u$ 路径上的所有节点都有一个贡献，可以用树上差分来记录。

## 子序列匹配

> [!NOTE] **例题 3 [leetcode 524. 通过删除字母匹配到字典里最长单词](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)**
> 
> 给定一个字符串 $s$ 和一个字符串数组 $\mathit{dictionary}$ 作为字典，找出并返回字典中最长的字符串，该字符串可以通过删除 $s$ 中的某些字符得到。

此类问题需要将字符串 $s$ 与 $t$ 进行匹配，判断 $t$ 是否为 $s$ 的子序列。解决这种问题只需先将两个指针一个 $i$ 放在 $s$ 开始位置，一个 $j$ 放在 $t$ 开始位置，如果 $s[i]=t[j]$ 说明 $t$ 的第 $j$ 位已经在 $s$ 中找到了第一个对应，可以进而检测后面的部分了，那么 $i$ 和 $j$ 同时加一。如果上述等式不成立，则 $t$ 的第 $j$ 位仍然没有被匹配上，所以只给 $i$ 加一，在 $s$ 的后面部分再继续寻找。最后，如果 $j$ 已经移到了超尾位置，说明整个字符串都可以被匹配上，也就是 $t$ 是 $s$ 的一个子序列，否则不是。

```cpp
string findLongestWord(string s, vector<string>& dictionary) {
    sort(dictionary.begin(), dictionary.end());
    int mx = 0, r = 0;
    string ans = "";
    for (int i = dictionary.size() - 1; i >= 0; i--) {
        r = 0;
        for (int j = 0; j < s.length(); ++j) {
            if (s[j] == dictionary[i][r]) r++;
        }
        if (r == dictionary[i].length()) {
            if (r >= mx) {
                mx = r;
                ans = dictionary[i];
            }
        }
    }
    return ans;
}
```

这种两个指针指向不同对象然后逐步进行比对的方法还可以用在一些 dp 中。

## 利用序列有序性

很多时候在序列上使用双指针之所以能够正确地达到目的，是因为序列的某些性质，最常见的就是利用序列的有序性。

> [!NOTE] **例题 4 [leetcode 167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)**
> 
> 给定一个已按照 **升序排列** 的整数数组 `numbers`，请你从数组中找出两个数满足相加之和等于目标数 `target`。

这种问题也是双指针的经典应用了，虽然二分也很方便，但时间复杂度上多一个 $\log{n}$，而且代码不够简洁。

接下来介绍双指针做法：既然要找到两个数，且这两个数不能在同一位置，那其位置一定是一左一右。由于两数之和固定，那么两数之中的小数越大，大数越小。考虑到这些性质，那我们不妨从两边接近它们。

首先假定答案就是 1 和 n，如果发现 $num[1]+num[n]\gt \mathit{target}$，说明我们需要将其中的一个元素变小，而 $\mathit{num}[1]$ 已经不能再变小了，所以我们把指向 $n$ 的指针减一，让大数变小。

同理如果发现 $num[1]+num[n]\lt \mathit{target}$，说明我们要将其中的一个元素变大，但 $\mathit{num}[n]$ 已经不能再变大了，所以将指向 1 的指针加一，让小数变大。

推广到一般情形，如果此时我们两个指针分别指在 $l,r$ 上，且 $l\lt r$, 如果 $num[l]+num[r]\gt \mathit{target}$，就将 $r$ 减一，如果 $num[l]+num[r]\lt \mathit{target}$，就将 $l$ 加一。这样 $l$ 不断右移，$r$ 不断左移，最后两者各逼近到一个答案。

```cpp
vector<int> twoSum(vector<int>& numbers, int target) {
    int r = numbers.size() - 1, l = 0;
    vector<int> ans;
    ans.clear();
    while (l < r) {
        if (numbers[l] + numbers[r] > target)
            r--;
        else if (numbers[l] + numbers[r] == target) {
            ans.push_back(l + 1), ans.push_back(r + 1);
            return ans;
        } else
            l++;
    }
    return ans;
}
```

在归并排序中，在 $O(n+m)$ 时间内合并两个有序数组，也是保证数组的有序性条件下使用的双指针法。

## 在单向链表中找环

在单向链表中找环也是有多种办法，不过快慢双指针方法是其中最为简洁的方法之一，接下来介绍这种方法。

首先两个指针都指向链表的头部，令一个指针一次走一步，另一个指针一次走两步，如果它们相遇了，证明有环，否则无环，时间复杂度 $O(n)$。

如果有环的话，怎么找到环的起点呢？

我们列出式子来观察一下，设相遇时，慢指针一共走了 $k$ 步，在环上走了 $l$ 步（快慢指针在环上相遇时，慢指针一定没走完一圈）。快指针走了 $2k$ 步，设环长为 $C$，则有

$$
 \ 2 k=n \times C+l+(k-l) \\
 \ k=n \times C \\
$$

第一次相遇时 $n$ 取最小正整数 1。也就是说 $k=C$。那么利用这个等式，可以在两个指针相遇后，将其中一个指针移到表头，让两者都一步一步走，再度相遇的位置即为环的起点。

## 习题

### 直观双指针

> [!NOTE] **[AcWing 799. 最长连续不重复子序列](https://www.acwing.com/activity/content/11/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>

using namespace std;

const int N = 100010;

int n;
int q[N], s[N];

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &q[i]);

    int res = 0;
    for (int i = 0, j = 0; i < n; i ++ ) {
        s[q[i]] ++ ;
        while (j < i && s[q[i]] > 1) s[q[j ++ ]] -- ;
        res = max(res, i - j + 1);
    }

    cout << res << endl;

    return 0;
}
```

##### **Python**

```python
if __name__=='__main__':
    n = int(input())
    arr = list(map(int,input().split()))
    l = 0
    import collections
    res, my_dict = 0, collections.defaultdict(int)
    for r in range(len(arr)):
        my_dict[arr[r]] += 1
        while my_dict[arr[r]] > 1:
            my_dict[arr[l]] -= 1
            l += 1 
        res = max(res, r - l + 1)
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 800. 数组元素的目标和](https://www.acwing.com/problem/content/802/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>

using namespace std;

const int N = 1e5 + 10;

int n, m, x;
int a[N], b[N];

int main() {
    scanf("%d%d%d", &n, &m, &x);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    for (int i = 0; i < m; i ++ ) scanf("%d", &b[i]);

    for (int i = 0, j = m - 1; i < n; i ++ ) {
        while (j >= 0 && a[i] + b[j] > x) j -- ;
        if (j >= 0 && a[i] + b[j] == x) cout << i << ' ' << j << endl;
    }

    return 0;
}
```

##### **Python**

```python
if __name__=='__main__':
    n, m, x = map(int,input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    p1, p2 = 0, m - 1
    while p1 < n and p2 >= 0:
        v = A[p1] + B[p2]
        if v == x:
            break
        elif v < x:
            p1 += 1
        else:
            p2 -= 1
    print(p1, p2)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 2816. 判断子序列](https://www.acwing.com/problem/content/2818/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>
#include <cstring>

using namespace std;

const int N = 100010;

int n, m;
int a[N], b[N];

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    for (int i = 0; i < m; i ++ ) scanf("%d", &b[i]);

    int i = 0, j = 0;
    while (i < n && j < m) {
        if (a[i] == b[j]) i ++ ;
        j ++ ;
    }

    if (i == n) puts("Yes");
    else puts("No");

    return 0;
}
```

##### **Python**

```python
#扫描B，这样的逻辑就很轻松了
if __name__ == '__main__':
    n, m = map(int,input().split())
    a = list(map(int,input().split()))
    b = list(map(int,input().split()))
    
    p1, p2 = 0, 0
    while p2 < m and p1 < n:
        if a[p1] == b[p2]:
            p1 += 1
        p2 += 1
    if p1 == n:
        print('Yes')
    else:
        print('No')
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)**
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
    vector<int> twoSum(vector<int>& numbers, int target) {
        for (int i = 0, j = numbers.size() - 1; i < j; i ++ ) {
            while (i < j && numbers[i] + numbers[j] > target) j -- ;
            if (i < j && numbers[i] + numbers[j] == target) return {i + 1, j + 1};
        }
        return {};
    }
    
    vector<int> twoSum_2(vector<int>& numbers, int target) {
        int n = numbers.size();
        vector<int> res;
        if (n < 2) return res;
        int l = 0, r = n - 1, v;
        while (l < r) {
            v = numbers[l] + numbers[r];
            if (v == target) {
                res.push_back(l + 1);
                res.push_back(r + 1);
                break;
            } else if (v < target) ++ l ;
            else -- r ;
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def twoSum(self, arr: List[int], target: int) -> List[int]:
        n = len(arr)
        sumn = 0
        l, r = 0, n - 1
        while l < r:
            sumn = arr[l] + arr[r]
            if sumn > target:
                r -= 1
            elif sumn < target:
                l += 1
            else:return [l + 1, r + 1]
        return [-1, -1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 611. 有效三角形的个数](https://leetcode-cn.com/problems/valid-triangle-number/)**
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
    int triangleNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int res = 0;
        // 枚举最大数 次大数 最小数   双指针
        for (int i = 0; i < nums.size(); ++ i )
            for (int j = i - 1, k = 0; j > 0 && k < j; -- j ) {
                while (k < j && nums[k] <= nums[i] - nums[j]) ++ k ;
                res += j - k;
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

> [!NOTE] **[LeetCode 633. 平方数之和](https://leetcode-cn.com/problems/sum-of-square-numbers/)**
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
    bool judgeSquareSum(int c) {
        long l = 0, r = sqrt(c);
        long res;
        while (l <= r) {
            res = l * l + r * r;
            if (res == c) return true;
            else if (res > c) -- r ;
            else ++ l ;
        }
        return false;
    }

    bool judgeSquareSum_2(int c) {
        // 直角三角形 一个边从1开始小于等于sqrt(c/2)
        int top = sqrt(c / 2);  // top < 4*10^4
        int t, j;
        for (int i = 0; i <= top; ++ i ) {
            t = c - i * i;
            j = sqrt(t);
            if (j * j == t) return true;
        }
            
        return false;
    }
};
```

##### **Python**

```python
#法一：直接枚举（两种写法）
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        # i = 0
        # while i * i <= c:
        #     j = c - i * i 
        #     r = int(math.sqrt(j))
        #     if r * r == j:return True 
        #     i += 1
        # return False
        for i in range(int(c ** 0.5) + 1):
            j = c - i ** 2
            r = int(j ** 0.5)
            if r ** 2 == j:return True 
        return False
      
#法二：双指针算法
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        # j = int(math.sqrt(c))
        j = int(c ** 0.5) + 1
        i = 0
        while i <=j:
            if c == i * i + j * j:
                return True
            elif i * i + j * j > c:
                j -= 1
            else:
                i += 1
        return False
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 719. 找出第 k 小的距离对](https://leetcode-cn.com/problems/find-k-th-smallest-pair-distance/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 二分 + 双指针

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int get(vector<int>& nums, int mid) {
        int res = 0;
        for (int l = 0, r = 0; r < nums.size(); ++ r ) {
            while (nums[r] - nums[l] > mid)
                ++ l ;
            res += r - l;
        }
        return res;
    }

    int smallestDistancePair(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int l = 0, r = 1e6;
        while (l < r) {
            int mid = l + r >> 1;
            if (get(nums, mid) >= k) r = mid;
            else l = mid + 1;
        }
        return r;
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

> [!NOTE] **[LeetCode 1213. 三个有序数组的交集](https://leetcode-cn.com/problems/intersection-of-three-sorted-arrays/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 多种方法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 分解两次操作**

```cpp
class Solution {
public:
    vector<int> get(vector<int> & a1, vector<int> & a2) {
        vector<int> res;
        for (int i = 0, j = 0; i < a1.size() && j < a2.size(); ) {
            if (a1[i] == a2[j]) {
                res.push_back(a1[i]);
                ++ i, ++ j;
            } else if (a1[i] < a2[j]) ++ i;
            else ++ j;
        }
        return res;
    }
    vector<int> arraysIntersection(vector<int>& arr1, vector<int>& arr2, vector<int>& arr3) {
        auto t = get(arr1, arr2);
        return get(t, arr3);
    }
};
```

##### **C++ hashTable**

```cpp
class Solution {
public:
    vector<int> arraysIntersection(vector<int>& arr1, vector<int>& arr2, vector<int>& arr3) {
        unordered_map<int, int> cnt;
        vector<int> res;
        for (int v : arr1) ++ cnt[v];
        for (int v : arr2) ++ cnt[v];
        for (int v : arr3) ++ cnt[v];
        for (auto & [k, v] : cnt)
            if (v == 3) res.push_back(k);
        return res;
    }
};
```

##### **C++ 三指针**

```cpp
class Solution {
public:
    vector<int> arraysIntersection(vector<int>& arr1, vector<int>& arr2,
                                   vector<int>& arr3) {
        int n1 = arr1.size(), n2 = arr2.size(), n3 = arr3.size();
        vector<int> res;
        for (int i = 0, j = 0, k = 0; i < n1; i++) {
            for (; j < n2 && arr2[j] < arr1[i]; j++)
                ;
            if (j == n2) break;
            for (; k < n3 && arr3[k] < arr1[i]; k++)
                ;
            if (k == n3) break;
            if (arr2[j] == arr1[i] && arr3[k] == arr1[i])
                res.push_back(arr1[i]);
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

> [!NOTE] **[LeetCode 1638. 统计只差一个字符的子串数目](https://leetcode-cn.com/problems/count-substrings-that-differ-by-one-character/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 向两侧扩展的双指针
> 
> 枚举优化：【枚举不同点 向两端扩展】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
    int countSubstrings(string s, string t) {
        int n = s.size(), m = t.size();
        int mlen = min(n, m);
        int ans = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                if (s[i] == t[j]) continue;
                int l = 0;
                while (i - (l + 1) >= 0 && j - (l + 1) >= 0 &&
                       s[i - (l + 1)] == t[j - (l + 1)])
                    l++;
                int r = 0;
                while (i + (r + 1) < n && j + (r + 1) < m &&
                       s[i + (r + 1)] == t[j + (r + 1)])
                    r++;
                ans += (l + 1) * (r + 1);
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

> [!NOTE] **[LeetCode 1498. 满足条件的子序列数目](https://leetcode.cn/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 想到 **排序双指针** 的 trick 思维
> 
> PS: 是 [LeetCode 891. 子序列宽度之和](https://leetcode.cn/problems/sum-of-subsequence-widths/) 的修改版，都要想到先排序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e6 + 10, MOD = 1e9 + 7;

    int p[N];
    void init() {
        p[0] = 1;
        for (int i = 1; i < N; ++ i )
            p[i] = p[i - 1] * 2 % MOD;
    }

    int numSubseq(vector<int>& nums, int target) {
        init();
        sort(nums.begin(), nums.end()); // trick 直接排序双指针

        int n = nums.size(), res = 0;
        for (int i = 0, j = n - 1; i <= j; ++ i ) { // i <= j
            while (i <= j && nums[i] + nums[j] > target)
                j -- ;
            // 实际上，对于重复值来说确实需要着重关心一下
            // 但我们可以人为的定义其先后关系，保证计算没有多
            if (i <= j && nums[i] + nums[j] <= target)
                res = (res + p[j - i]) % MOD;
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

### 快慢指针思想

> 链表判环、找中点的快慢指针应用参见链表部分

> [!NOTE] **[LeetCode 202. 快乐数](https://leetcode-cn.com/problems/happy-number/)**
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
    int get(int x) {
        int res = 0;
        while (x) {
            res += (x % 10) * (x % 10);
            x /= 10;
        }
        return res;
    }

    bool isHappy(int n) {
        int fast = get(n), slow = n;
        while (fast != slow) {
            fast = get(get(fast));
            slow = get(slow);
        }
        return fast == 1;
    }
};
```

##### **C++ 模拟**

```cpp
class Solution {
public:
    int trans(int n) {
        int res = 0;
        while (n) {
            res += (n % 10) * (n % 10);
            n /= 10;
        }
        return res;
    }

    bool isHappy(int n) {
        unordered_map<int, bool> m;
        m[n] = true;
        while (n != 1) {
            n = trans(n);
            if (m[n]) return false;
            m[n] = true;
        }
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

> [!NOTE] **[LeetCode 457. 环形数组循环](https://leetcode-cn.com/problems/circular-array-loop/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 使用大数标记的思维 优雅实现
>
> 另有 **快慢指针做法**
>
> 将问题抽象，**每个点最多只有一个出边**，故其一定是一个类似链表环的形式（因为如果还能在环内出去的话一定是在环上某点有两个出边，显然不可能）。
>
> 这样显然有一个快慢指针的实现方式。
>
> 考虑枚举的思路：
>
> >   如果遍历到一个【本次】【在此前遍历到】的点
> >
> >   为什么 `last % n` 可以判断是否自环？

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size(), Base = 10000;
        for (int i = 0; i < n; ++ i ) {
            // 已经遍历过
            if (nums[i] >= Base) continue;
            int k = i, S = Base + i, t = nums[k] > 0;
            int last = -1;  // 最后一个位置 用于判断是否自环
            do {
                // k + nums[k];
                int p = ((k + nums[k]) % n + n) % n;
                last = nums[k], nums[k] = S;
                k = p;
            } while (k != i && nums[k] < Base && (t ^ (nums[k] > 0)) == 0);
            // while 后面访问到 且初次访问 且符号相同

            // last % n 非 0 则没有自环;  nums[k] == S 说明是本次遍历此前遍历到的，长度大于1
            if (last % n && nums[k] == S) return true;
        }
        return false;
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

### 优化双指针

> [!NOTE] **[LeetCode 15. 三数之和](https://leetcode-cn.com/problems/3sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); i ++ ) {
            if (i && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1, k = nums.size() - 1; j < k; j ++ ) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                // 可以适应三数之和为任意值
                while (j < k - 1 && nums[i] + nums[j] + nums[k - 1] >= 0) k -- ; // ATTENTION
                if (nums[i] + nums[j] + nums[k] == 0) {
                    res.push_back({nums[i], nums[j], nums[k]});
                }
            }
        }

        return res;
    }
};
```

##### **Python**

```python
# 双指针算法 一定要基于有序才能做。一般都是先想暴力怎么求解，然后用双指针进行优化，可以将时间复杂度降低一个维度
# （本题也可以用哈希表，但是空间复杂度就高一些）
# 1. 先将nums排序，然后 固定指针i， 遍历数组，对于每一个i， 移动指针L和R， 找到nums[i] + nums[L] + nums[R] == 0
# 2. 由于nums是有序的，所以当L++， 那么对应的R就会减小（初始设置 L = i + 1; R = n - 1 ) 
# 3. 至于去重，只需要判断每个指针位置的下一个位置的值 和 该指针当前值是否相等，如果相等 直接跳过即可。

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []; n = len(nums)
        if n < 3:return []
        nums.sort()
        sumn = 0
        for i in range(n - 2):
            if nums[i] > 0:break
            if i > 0 and nums[i] == nums[i - 1]:continue  # 去重
            l = i + 1; r = n - 1
            while l < r:
                sumn = nums[i] + nums[l] + nums[r]
                if sumn == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l + 1] == nums[l]:l += 1
                    while l < r and nums[r - 1] == nums[r]:r -= 1
                    l += 1
                    r -= 1
                elif sumn < 0:
                    l += 1
                else:
                    r -= 1
        return res
      
      
# 偷懒做法，用set保存结果，最后再变成list类型
# 可以省掉两个指针的去重判断
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = set()
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            low, high = i + 1, len(nums) - 1
            while low < high:
                s = nums[i] + nums[low] + nums[high]
                if s > 0:
                    high -= 1
                elif s < 0:
                    low += 1
                else:
                    res.add((nums[i], nums[low], nums[high]))
                    low += 1
                    high -= 1
        return list(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        pair<int, int> res(INT_MAX, INT_MAX);
        for (int i = 0; i < nums.size(); i ++ )
            for (int j = i + 1, k = nums.size() - 1; j < k; j ++ ) {
                while (k - 1 > j && nums[i] + nums[j] + nums[k - 1] >= target) k -- ;
                int s = nums[i] + nums[j] + nums[k];
                res = min(res, make_pair(abs(s - target), s));
                if (k - 1 > j) {
                    s = nums[i] + nums[j] + nums[k - 1];
                    res = min(res, make_pair(target - s, s));
                }
            }
        return res.second;
    }
};
```

##### **Python**

```python
# 考虑在target右边/左边；本题只有一个答案，就不需要判重

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # res 初始化一个很大的值
        res = float('inf')
        n = len(nums)
        nums.sort()
        sumn = 0
        for i in range(n - 2):
            l = i + 1; r = n - 1
            while l < r:
                sumn = nums[i] + nums[r] + nums[l] 
                v = target - sumn
                if abs(v) < abs(target - res):res = sumn
                if v < 0:r -= 1
                elif v > 0:l += 1
                else:return target
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 18. 四数之和](https://leetcode-cn.com/problems/4sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        for (int i = 0; i < nums.size(); i ++ ) {
            if (i && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1; j < nums.size(); j ++ ) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                for (int k = j + 1, u = nums.size() - 1; k < u; k ++ ) {
                    if (k > j + 1 && nums[k] == nums[k - 1]) continue;
                    while (u - 1 > k && nums[i] + nums[j] + nums[k] + nums[u - 1] >= target) u -- ;
                    if (nums[i] + nums[j] + nums[k] + nums[u] == target) {
                        res.push_back({nums[i], nums[j], nums[k], nums[u]});
                    }
                }
            }
        }

        return res;
    }
};
```

##### **Python**

```python
# 先枚举前两个遍历，后两个遍历用双指针算法进行优化；
# 去重方法和之前的一样：当前数和下一个数一致，下一个数就直接跳过

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        n = len(nums); res = []
        if n < 4:return []
        nums.sort()
        sumn = 0
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:continue
            for j in range(i + 1, n - 2):  # 踩坑： j 是从 i+1 开始遍历的
                if j > i + 1 and nums[j] == nums[j - 1]:continue
                l = j + 1; r = n - 1
                while l < r:
                    sumn = nums[i] + nums[j] + nums[l] + nums[r]
                    if sumn == target:
                        res.append([nums[i], nums[j], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l + 1]:l += 1
                        while l < r and nums[r] == nums[r - 1]:r -= 1
                        l += 1; r -= 1
                    elif sumn > target:
                        r -= 1
                    else:l += 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 786. 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分 + **双指针单调优化**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const double eps = 1e-8;
    int A, B, n;
    vector<int> a;

    // 仍然可优化 显然随着i增加j只会更靠右 具有单调性
    int get(double m) {
        int ret = 0;
        for (int i = 0, j = 0; i < n; ++ i ) {
            while ((double)a[j + 1] / a[i] <= m)
                j ++ ;
            if ((double)a[j] / a[i] <= m)
                ret += j + 1;
            if (fabs((double)a[j] / a[i] - m) < eps)
                A = a[j], B = a[i];
        }
        return ret;
    }

    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        this->a = arr; n = a.size();
        double l = 0, r = 1;
        while (r - l > eps) {
            double m = (l + r) / 2;
            if (get(m) < k)
                l = m;
            else
                r = m;
        }
        get(l);
        return {A, B};
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

> [!NOTE] **[LeetCode 1610. 可见点的最大数目](https://leetcode-cn.com/problems/maximum-number-of-visible-points/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 思路：计算所有点到location的角度，排序再在后面补，方便循环查找。
>
> 需要注意：
>
> 1.  自己写的处理大 case 超时，原因在于 for 内部再二分仍然很大。实际上，可以用双指针替代二分搜索，进一步降低复杂度。
> 2.  关于 pi 的处理，直接写 3.1415926 还会有精度问题，使用 acos(-1.0)
>
> c++ 三角函数:
>
> cos 余弦 sin 正弦 tan 正切 acos 反余弦 asin 反正弦 atan 反正切 atan2 取值范围为 (-pi, pi]

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int visiblePoints(vector<vector<int>>& points, int angle,
                      vector<int>& location) {
        double pi = acos(-1.0);
        int x = location[0], y = location[1];
        vector<double> ps;
        int same = 0;
        for (auto& p : points) {
            if (p[0] == x && p[1] == y) {
                ++same;
                continue;
            }
            ps.push_back(atan2(p[1] - y, p[0] - x));
        }
        sort(ps.begin(), ps.end());
        int n = ps.size();
        for (int i = 0; i < n; ++i) ps.push_back(ps[i] + acos(-1.) * 2);
        double t = 1.0 * angle * pi / 180;
        int res = 0;
        for (int i = 0, j = 0; i < n; ++i) {
            while (j < 2 * n && ps[j] - ps[i] <= t) ++j;
            res = max(res, j - i);
        }
        return same + res;
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

> [!NOTE] **[LeetCode 1855. 下标对中的最大距离](https://leetcode-cn.com/problems/maximum-distance-between-a-pair-of-values/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 双指针就可以 很好的双指针题目

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 更优的写法
class Solution {
public:
    int maxDistance(vector<int>& nums1, vector<int>& nums2) {
        int res = 0;
        for (int i = 0, j = 0; i < nums1.size() && j < nums2.size(); ++ i ) {
            while (j < nums2.size() && nums2[j] >= nums1[i])
                j ++ ;
            if (j - 1 >= i && nums2[j - 1] >= nums1[i])
                res = max(res, j - 1 - i);                
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

### 推导进阶

> [!NOTE] **[Luogu [USACO15OPEN]Trapped in the Haybales S](https://www.luogu.com.cn/problem/P3124)**
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
#include <bits/stdc++.h>
using namespace std;

// 引用某题解的话
// > 先给大家表演一波错误的思路：
// > 从初始位置出发向左右两边分别破障，取最小值。
// > 错误的思路加上理解题意错误，题意只能增加一捆稻草。
// > 导致浪费时间+自闭自闭自闭
// > WA WA WA
// > 这就是思路不清晰的后果，
// > 一定要想好了再动手！！！！想好了再动手！！！
//
// 题意为【只增加一个草堆的数量检查能否困住奶牛】
//
// 需要加固该草堆的size就等于跑动的距离减去该草堆的大小。
// 那么某一个草堆能够被撞击的最大加速值是在不打破该草堆的前提下
// 让Bessie随意倒腾能够开辟出的最大空间。
// ===> 反映在代码实现中就是只要另一侧可以继续开扩则continue开阔
//
// 枚举，分别向左向右i枚举每一个草堆
// 假设加高当前草堆 则需找到另一侧最近的不被击破的草堆位值
// 1. 另一侧草堆被击破 继续向边界搜索直到越界
// 2. 不被击破 根据距离 可以求出本个草堆需要增加的高度
//    如果本个草堆就算不加稻草也不会被击破 显然输出0结束
//    如果本个草堆会被击破 计算稻草【并在同一侧开阔区间】
//
// 那么问题来了，为什么可以在本个草堆被击破时开阔本侧？
// ===> 因为考虑枚举同侧更靠外的一个点，还要重复之前开阔另一侧的过程
//      已知Bessie已经可以自由走到另一侧的某个位置，无需再走一遍
//
// ===> 归根到底是：双指针的Trick优化

const int N = 1e5 + 10, INF = 0x3f3f3f3f;

int n, b;
struct K {
    int s, p;
} k[N];

int main() {
    cin >> n >> b;
    for (int i = 1; i <= n; ++ i )
        cin >> k[i].s >> k[i].p;
    sort(k + 1, k + n + 1, [](const K & a, const K & b) {
        return a.p < b.p;
    });
    
    int id;
    for (int i = 1; i <= n; ++ i )
        if (k[i].p > b) {
            id = i;
            break;
        }
    
    int res = INF;
    {
        // 初始化
        int l = id - 1, r = id, d = k[r].p - k[l].p;
        while (l >= 1 && r <= n) {
            if (k[l].s >= d && k[r].s >= d) {
                cout << 0 << endl;
                return 0;
            }
            if (k[r].s < d) {
                d += k[r + 1].p - k[r].p;
                r ++ ;
                // think why
                continue;
            }
            if (k[l].s < d) {
                res = min(res, d - k[l].s);
                d += k[l].p - k[l - 1].p;
                l -- ;
            }
        }
    }
    {
        int l = id - 1, r = id, d = k[r].p - k[l].p;
        while (l >= 1 && r <= n) {
            if (k[l].s >= d && k[r].s >= d) {
                cout << 0 << endl;
                return 0;
            }
            if (k[l].s < d) {
                d += k[l].p - k[l - 1].p;
                l -- ;
                continue;
            }
            if (k[r].s < d) {
                res = min(res, d - k[r].s);
                d += k[r + 1].p - k[r].p;
                r ++ ;
            }
        }
    }
    
    cout << (res == INF ? -1 : res) << endl;
    
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

> [!NOTE] **[LeetCode 11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)**
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
    int maxArea(vector<int>& height) {
        int n = height.size();
        int l = 0, r = n - 1, res = 0;
        while (l < r) {
            res = max(res, min(height[l], height[r]) * (r - l));
            if (height[l] < height[r]) ++l;
            else --r;
        }
        return res;
    }
};
```

##### **Python**

```python
# 这道题不适合用单调栈来做(比较复杂），和接雨水的题做对比

# 思维具有跳跃性，脑筋急转弯类型的题目。（需要记住思路）
# 做法：用两个指针 l, r 分别指向首尾，如果 al > ar，则 r−−，因为更长的柱子 以后更可能会被用到；否则 l++，直到 l == r为止，每次迭代更新最大值。

class Solution:
    def maxArea(self, h: List[int]) -> int:
        n = len(h)
        l, r = 0, n - 1
        res = float('-inf')
        while l < r:
            if h[l] < h[r]:
                res = max(res, h[l] * (r - l))
                l += 1 
            else:
                res = max(res, h[r] * (r - l))
                r -= 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 532. 数组中的 k-diff 数对](https://leetcode-cn.com/problems/k-diff-pairs-in-an-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然有单调性质
> 
> 双指针进阶 思想 重复
> 
> 思考变种题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findPairs(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int res = 0, n = nums.size();
        for (int l = 0, r = 0; r < n; ++ r ) {
            while (r + 1 < n && nums[r + 1] == nums[r])
                r ++ ;
            while (l < r && nums[r] - nums[l] > k)
                l ++ ;
            if (l < r && nums[r] - nums[l] == k)
                res ++ ;
        }
        return res;
    }
};
```

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    int findPairs(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int i = 0, j = 0; i < nums.size(); ++ i ) {
            // 枚举后面的 可以保证解决 k = 0 的情况
            while (i + 1 < nums.size() && nums[i + 1] == nums[i]) ++ i ;
            while (j < i && nums[i] - nums[j] > k) ++ j ;
            if (j < i && nums[i] - nums[j] == k) ++ res;
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

> [!NOTE] **[LeetCode 1537. 最大得分](https://leetcode-cn.com/problems/get-the-maximum-score/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 两个递增有序数组 从任意一个头部起遇相同数字可换位置 求最终总和：
>
> [![img](https://camo.githubusercontent.com/e2552b955a88c82b9b08ba7b330444ad9e274fd4d925b26ea127b5cbfffabd24/68747470733a2f2f6173736574732e6c656574636f64652d636e2e636f6d2f616c6979756e2d6c632d75706c6f61642f75706c6f6164732f323032302f30382f30322f73616d706c655f315f313839332e706e67)](https://camo.githubusercontent.com/e2552b955a88c82b9b08ba7b330444ad9e274fd4d925b26ea127b5cbfffabd24/68747470733a2f2f6173736574732e6c656574636f64652d636e2e636f6d2f616c6979756e2d6c632d75706c6f61642f75706c6f6164732f323032302f30382f30322f73616d706c655f315f313839332e706e67)
>
> 如图 2+4+6+8+10 = 30
>
> **考虑：** 每一个相同数字相当于分界点 相同数字间取最大即可
>
> 双指针解决

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxSum(vector<int>& nums1, vector<int>& nums2) {
        int n1 = nums1.size(), n2 = nums2.size();
        long long s1 = 0, s2 = 0, mod = 1e9 + 7;
        int i = n1 - 1, j = n2 - 1;
        while (~i && ~j) {
            if (nums1[i] > nums2[j]) {
                s1 += nums1[i];
                --i;
            } else if (nums1[i] < nums2[j]) {
                s2 += nums2[j];
                --j;
            } else {
                s1 = s2 = max(s1, s2) + nums1[i];
                --i, --j;
            }
        }
        while (~i) s1 += nums1[i--];
        while (~j) s2 += nums2[j--];
        return max(s1, s2) % mod;
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

> [!NOTE] **[LeetCode 1712. 将数组分成三个子数组的方案数](https://leetcode-cn.com/problems/ways-to-split-array-into-three-subarrays/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 注意枚举方法 **单调性证明**【j k 只会随 i 向右】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int mod = 1e9 + 7;
    using LL = long long;
    int waysToSplit(vector<int>& nums) {
        int n = nums.size();
        vector<LL> s(n + 1);
        for (int i = 1; i <= n; ++ i )
            s[i] = s[i - 1] + nums[i - 1];
        
        int res = 0;
        // [0, x - 1], [x, i - 1], [i, n]
        // j 为最左，k为最右
        for (int i = 3, j = 2, k = 2; i <= n; ++ i ) {
            while (s[n] - s[i - 1] < s[i - 1] - s[j - 1]) j ++ ;
            while (k + 1 < i && s[i - 1] - s[k] >= s[k]) k ++ ;
            if (j <= k && s[n] - s[i - 1] >= s[i - 1] - s[j - 1] && s[i - 1] - s[k - 1] >= s[k - 1])
                res = (res + k - j + 1) % mod;
        }
        return res;
    }
};
```

##### **C++ 二分**

最初二分写法修正：

二分搜索时应从 `s.begin() + 1` 开始

```cpp
class Solution {
public:
    const int mod = 1e9 + 7;
    using LL = long long;
    int waysToSplit(vector<int>& nums) {
        int n = nums.size();
        vector<LL> s(n + 1);
        for (int i = 1; i <= n; ++ i )
            s[i] = s[i - 1] + nums[i - 1];
        
        LL res = 0;
        for (int i = 2; i < n; ++ i ) {
            LL rv = s[n] - s[i];
            if (rv * 2 < s[i]) break;
            int ridx = upper_bound(s.begin() + 1, s.begin() + i, s[i] / 2) - s.begin();     // 保证 left <= mid
            int lidx = lower_bound(s.begin() + 1, s.begin() + i, s[i] - rv) - s.begin();    // 保证 mid <= right
            res = (res + ridx - lidx) % mod;
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

> [!NOTE] **[LeetCode 1793. 好子数组的最大分数](https://leetcode-cn.com/problems/maximum-score-of-a-good-subarray/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 直观单调栈 重复TODO
> 
> **有线性做法 推导**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 双指针**

```cpp
class Solution {
public:
    int maximumScore(vector<int>& nums, int k) {
        int n = nums.size(), res = 0;
        for (int i = nums[k], l = k, r = k; i >= 1; -- i ) {
            while (l - 1 >= 0 && nums[l - 1] >= i) -- l;
            while (r + 1 < n && nums[r + 1] >= i) ++ r;
            res = max(res, (r - l + 1) * i);
        }
        return res;
    }
};
```

##### **C++ 单调栈1**

```cpp
class Solution {
public:
    int maximumScore(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> left(n, -1), right(n, n);
        stack<int> st;
        for (int i = 0; i < n; ++i) {
            while (!st.empty() && nums[st.top()] > nums[i]) {
                right[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }
        st = stack<int>();
        for (int i = n - 1; i >= 0; --i) {
            while (!st.empty() && nums[st.top()] > nums[i]) {
                left[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            int l = left[i] + 1, r = right[i] - 1;
            if (l <= k && r >= k)
                ans = max(ans, (r - l + 1) * nums[i]);
        }
        return ans;
    }
};
```

##### **C++ 单调栈2**

```cpp
class Solution {
public:
    int maximumScore(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> h(n + 2, -1), l(n + 2), r(n + 2), stk(n + 2);
        for (int i = 1; i <= n; i ++ ) h[i] = nums[i - 1];
        int tt = 0;
        stk[0] = 0;
        for (int i = 1; i <= n; i ++ ) {
            while (h[stk[tt]] >= h[i]) tt -- ;
            l[i] = stk[tt];
            stk[ ++ tt] = i;
        }
        tt = 0, stk[0] = n + 1;
        for (int i = n; i; i -- ) {
            while (h[stk[tt]] >= h[i]) tt -- ;
            r[i] = stk[tt];
            stk[ ++ tt] = i;
        }
        k ++ ;
        int res = 0;
        for (int i = 1; i <= n; i ++ )
            if (l[i] < k && r[i] > k)
                res = max(res, (r[i] - l[i] - 1) * h[i]);
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

### 双指针区间维护

> [!NOTE] **[Codeforces R2D2 and Droid Army](http://codeforces.com/problemset/problem/514/D)**
> 
> 题意: 
> 
> 一共有N个人，每个人有M个属性值，当一个人的所有属性值都小于等于0的时候，这个人就算被销毁了。
> 
> 我们每次操作可以选一种属性值进行攻击，使得所有人的这个属性的值都-1.
> 
> 我们最多可以进行K次操作，
> 
> 问我们最多可以干掉多少个连续的人。
> 
> 问这种时候的具体操作（每一种属性用了多少次操作）。

> [!TIP] **思路**
> 
> 多维双指针 维护区间最值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. R2D2 and Droid Army
// Contest: Codeforces - Codeforces Round #291 (Div. 2)
// URL: https://codeforces.com/problemset/problem/514/D
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10, M = 6;

int n, m;
LL k;

LL c[N][M];

deque<LL> cost[M];  // 本质单调队列
void add(int x) {
    for (int i = 1; i <= m; ++i) {
        int t = c[x][i];
        while (cost[i].size() && cost[i].back() < t)
            cost[i].pop_back();
        cost[i].push_back(t);
    }
}
void sub(int x) {
    for (int i = 1; i <= m; ++i) {
        int t = c[x][i];
        if (cost[i].front() == t)
            cost[i].pop_front();
    }
}
LL sum() {
    LL ret = 0;
    for (int i = 1; i <= m; ++i)
        if (cost[i].size())
            ret += cost[i].front();
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m >> k;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            cin >> c[i][j];

    int res = 0;
    int t[M];
    memset(t, 0, sizeof t);
    for (int i = 0; i < M; ++i)
        cost[i].clear();
    for (int l = 1, r = 1; r <= n; ++r) {	// ATTENTION <= n WA1
        add(r);
        while (l <= r && sum() > k)
            sub(l++);
        if (r - l + 1 > res) {
            res = r - l + 1;
            for (int i = 1; i <= m; ++i)
                t[i] = cost[i].front();
        }
    }

    for (int i = 1; i <= m; ++i)
        cout << t[i] << ' ';
    cout << endl;

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

> [!NOTE] **[LeetCode 992. K 个不同整数的子数组](https://leetcode.cn/problems/subarrays-with-k-different-integers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 枚举右端点，维护左侧为 `k`, `k-1` 的区间范围

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int subarraysWithKDistinct(vector<int>& nums, int k) {
        unordered_map<int, int> S1, S2;
        int res = 0;
        for (int i = 0, j1 = 0, j2 = 0, c1 = 0, c2 = 0; i < nums.size(); ++ i ) {
            if (!S1[nums[i]])
                c1 ++ ;
            S1[nums[i]] ++ ;
            while (j1 <= i && c1 > k) {
                S1[nums[j1]] -- ;
                if (!S1[nums[j1]])
                    c1 -- ;
                j1 ++ ;
            }

            if (!S2[nums[i]])
                c2 ++ ;
            S2[nums[i]] ++ ;
            while (j2 <= i && c2 >= k) {
                S2[nums[j2]] -- ;
                if (!S2[nums[j2]])
                    c2 -- ;
                j2 ++ ;
            }

            res += j2 - j1;
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

### 优雅的双指针实现

> [!NOTE] **[LeetCode 2332. 坐上公交的最晚时间](https://leetcode.cn/problems/the-latest-time-to-catch-a-bus/) [TAG]**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常有意思的模拟
> 
> **优雅的双指针实现**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int latestTimeCatchTheBus(vector<int>& b, vector<int>& p, int cap) {
        sort(b.begin(), b.end()); sort(p.begin(), p.end());
        int n = b.size(), m = p.size();
        
        int j = 0, c = 0;   // ATTENTION: 置于外部的细节
        for (int i = 0; i < n; ++ i ) {
            c = cap;        // ATTENTION: 使用外部变量
            while (j < m && p[j] <= b[i] && c)
                j ++ , c -- ;
        }
        j -- ;
        
        int res;
        if (c)  // 车装不满所有人 则可以直接从最后一个车的发车时间开始往前
            res = b.back();
        else    // 否则在最后一个可以上车的人的时间往前
            res = p[j];     // 注意这里直接赋值 p[j] 而不是 p[j]-1 是为了后面 while 好写
        
        while (j >= 0 && p[j] == res)
            res -- , j -- ;
        
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