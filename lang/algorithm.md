STL 提供了大约 100 个实现算法的模版函数，基本都包含在 `<algorithm>` 之中，还有一部分包含在 `<numeric>` 和 `<functional>`。完备的函数列表请 [参见参考手册](https://zh.cppreference.com/w/cpp/algorithm)，排序相关的可以参考 [排序内容的对应页面](basic/stl-sort.md)。

### 基本内容

- `find`：顺序查找。`find(v.begin(), v.end(), value)`，其中 `value` 为需要查找的值。
- `find_end`：逆序查找。`find_end(v.begin(), v.end(), value)`。
- `reverse`：翻转数组、字符串。`reverse(v.begin(), v.end())` 或 `reverse(a + begin, a + end)`。
- `unique`：去除容器中相邻的重复元素。`unique(ForwardIterator first, ForwardIterator last)`，返回值为指向 **去重后** 容器结尾的迭代器，原容器大小不变。与 `sort` 结合使用可以实现完整容器去重。
- `random_shuffle`：随机地打乱数组。`random_shuffle(v.begin(), v.end())` 或 `random_shuffle(v + begin, v + end)`。

> [!WARNING] **`random_shuffle` 函数在最新 C++ 标准中已被移除**
> 
> 在 C++11 以及更新的标准中，可以使用 `shuffle` 函数代替原来的 `random_shuffle`。使用方法为 `shuffle(v.begin(),v.end(),rand)`（最后一个参数传入的是使用的随机数生成器，一般情况下传入 `rand` 即可）。

- `sort`：排序。`sort(v.begin(), v.end(), cmp)` 或 `sort(a + begin, a + end, cmp)`，其中 `end` 是排序的数组最后一个元素的后一位，`cmp` 为自定义的比较函数。
- `stable_sort`：稳定排序，用法同 `sort()`。
- `nth_element`：按指定范围进行分类，即找出序列中第 $n$ 大的元素，使其左边均为小于它的数，右边均为大于它的数。`nth_element(v.begin(), v.begin() + mid, v.end(), cmp)` 或 `nth_element(a + begin, a + begin + mid, a + end, cmp)`。
- `binary_search`：二分查找。`binary_search(v.begin(), v.end(), value)`，其中 `value` 为需要查找的值。
- `merge`：将两个（已排序的）序列 **有序合并** 到第三个序列的 **插入迭代器** 上。`merge(v1.begin(), v1.end(), v2.begin(), v2.end() ,back_inserter(v3))`。
- `inplace_merge`：将两个（已按小于运算符排序的）：`[first,middle), [middle,last)` 范围 **原地合并为一个有序序列**。`inplace_merge(v.begin(), v.begin() + middle, v.end())`。
- `lower_bound`：在一个有序序列中进行二分查找，返回指向第一个 **大于等于**  $x$ 的元素的位置的迭代器。如果不存在这样的元素，则返回尾迭代器。`lower_bound(v.begin(),v.end(),x)`。
- `upper_bound`：在一个有序序列中进行二分查找，返回指向第一个 **大于**  $x$ 的元素的位置的迭代器。如果不存在这样的元素，则返回尾迭代器。`upper_bound(v.begin(),v.end(),x)`。

> [!WARNING] **`lower_bound` 和 `upper_bound` 的时间复杂度**
> 
> 在一般的数组里，这两个函数的时间复杂度均为 $O(\log n)$，但在 `set` 等关联式容器中，直接调用 `lower_bound(s.begin(),s.end(),val)` 的时间复杂度是 $O(n)$ 的。
> 
> `set` 等关联式容器中已经封装了 `lower_bound` 等函数（像 `s.lower_bound(val)` 这样），这样调用的时间复杂度是 $O(\log n)$ 的。

- `next_permutation`：将当前排列更改为 **全排列中的下一个排列**。如果当前排列已经是 **全排列中的最后一个排列**（元素完全从大到小排列），函数返回 `false` 并将排列更改为 **全排列中的第一个排列**（元素完全从小到大排列）；否则，函数返回 `true`。`next_permutation(v.begin(), v.end())` 或 `next_permutation(v + begin, v + end)`。
- `partial_sum`：求前缀和。设源容器为 $x$，目标容器为 $y$，则令 $y[i]=x[0]+x[1]+...+x[i]$。`partial_sum(src.begin(), src.end(), back_inserter(dst))`。

### 使用样例

- 使用 `next_permutation` 生成 $1$ 到 $9$ 的全排列。例题：[Luogu P1706 全排列问题](https://www.luogu.com.cn/problem/P1706)

```cpp
int N = 9, a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
do {
    for (int i = 0; i < N; i++) cout << a[i] << ' ';
    cout << endl;
} while (next_permutation(a, a + N));
```

- 使用 `lower_bound` 与 `upper_bound` 查找有序数组 $a$ 中小于 $x$，等于 $x$，大于 $x$ 元素的分界线。

```cpp
int N = 10, a[] = {1, 1, 2, 4, 5, 5, 7, 7, 9, 9}, x = 5;
int i = lower_bound(a, a + N, x) - a, j = upper_bound(a, a + N, x) - a;
// a[0] ~ a[i - 1] 为小于x的元素， a[i] ~ a[j - 1] 为等于x的元素， a[j] ~ a[N -
// 1] 为大于x的元素
cout << i << ' ' << j << endl;
```

- 使用 `partial_sum` 求解 $src$ 中元素的前缀和，并存储于 $dst$ 中。

```cpp
vector<int> src = {1, 2, 3, 4, 5}, dst;
// 求解src中元素的前缀和，dst[i] = src[0] + ... + src[i]
// back_inserter 函数作用在 dst 容器上，提供一个迭代器
partial_sum(src.begin(), src.end(), back_inserter(dst));
for (unsigned int i = 0; i < dst.size(); i++) cout << dst[i] << ' ';
```

- 使用 `lower_bound` 查找有序数组 $a$ 中最接近 $x$ 的元素。例题：[UVA10487 Closest Sums](https://www.luogu.com.cn/problem/UVA10487)

```cpp
int N = 10, a[] = {1, 1, 2, 4, 5, 5, 8, 8, 9, 9}, x = 6;
// lower_bound将返回a中第一个大于等于x的元素的地址，计算出的i为其下标
int i = lower_bound(a, a + N, x) - a;
// 在以下两种情况下，a[i] (a中第一个大于等于x的元素) 即为答案：
// 1. a中最小的元素都大于等于x；
// 2. a中存在大于等于x的元素，且第一个大于等于x的元素 (a[i])
// 相比于第一个小于x的元素 (a[i - 1]) 更接近x；
// 否则，a[i - 1] (a中第一个小于x的元素) 即为答案
if (i == 0 || (i < N && a[i] - x < x - a[i - 1]))
    cout << a[i];
else
    cout << a[i - 1];
```

- 使用 `sort` 与 `unique` 查找数组 $a$ 中 **第 $k$ 小的值**（注意：重复出现的值仅算一次，因此本题不是求解第 $k$ 小的元素）。例题：[Luogu P1138 第 k 小整数](https://www.luogu.com.cn/problem/P1138)

```cpp
int N = 10, a[] = {1, 3, 3, 7, 2, 5, 1, 2, 4, 6}, k = 3;
sort(a, a + N);
// unique将返回去重之后数组最后一个元素之后的地址，计算出的cnt为去重后数组的长度
int cnt = unique(a, a + N) - a;
cout << a[k - 1];
```

## 习题

### next_permutation (一系列 包括周赛逆操作)

> [!NOTE] **[Luogu NOIP2004 普及组 火星人](https://www.luogu.com.cn/problem/P1088)**
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

int n, m;

int main() {
    cin >> n >> m;
    
    vector<int> ve;
    for (int i = 0; i < n; ++ i ) {
        int v;
        cin >> v;
        ve.push_back(v);
    }
    
    while (m -- ) {
        next_permutation(ve.begin(), ve.end());
    }
    for (auto v : ve)
        cout << v << ' ';
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

> [!NOTE] **[LeetCode 31. 下一个排列](https://leetcode.cn/problems/next-permutation/)**
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
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        int p = n - 1;
        while (p && nums[p] <= nums[p - 1])
            p -- ;
        if (p == 0)
            reverse(nums.begin(), nums.end());
        else {
            int t = p + 1;
            while (t < n && nums[t] > nums[p - 1])
                t ++ ;
            swap(nums[p - 1], nums[t - 1]);
            reverse(nums.begin() + p, nums.end());
        }
    }
};
```

##### **Python**

```python
# 答题思路：从后往前寻找第一个升序对(i,j)即nums[i]<nums[j] 再从后往前找第一个大于nums[i]的数即为大数，交换着两个元素即将大数换到前面，然后将大数后面的部分倒序
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        def reversed(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1

        n = len(nums)
        i = n - 1
        
        # 踩坑！ 一定要记得 当nums[i] > nums[i-1]的时候 要跳出循环 有一种写法很容易进入死循话
        # while i > 0:
        #      if nums[i] <= nums[i-1]:
        #              i -= 1
        
        while i > 0 and nums[i] <= nums[i-1]:    
            i -= 1
        if i == 0:return nums.reverse()
        j = i - 1
        p = n - 1
        while p > j and nums[p] <= nums[j]:
            p -= 1
        nums[j], nums[p] = nums[p], nums[j]
        reversed(j+1, n-1)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1830. 使字符串有序的最少操作次数](https://leetcode.cn/problems/minimum-number-of-operations-to-make-string-sorted/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 逆序观察 操作的过程（与求下一个排列恰好是逆操作）恰好是全排列
> 
> 故原题可以转化为求原排列是初始（升序）排列的第几个排列
> 
> 显然有 **康托展开** 或 **数位DP** ==> TODO 整理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 康托展开**

```cpp
// 康托展开
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    LL quick_pow(LL a, LL b, LL m) {
        LL res = 1;
        a %= m;
        while (b) {
            if (b & 1)
                res = res * a % m;
            a = a * a % m;
            b >>= 1;
        }
        return res;
    }
    
    int makeStringSorted(string s) {
        int n = s.size();
        LL fact = 1, dup = 1;
        LL res = 0;
        vector<int> seen(26, 0);
        for (int i = n - 1; i >= 0; -- i ) {
            seen[s[i] - 'a'] ++ ;
            dup = dup * seen[s[i] - 'a'] % MOD;
            
            LL rk = 0;
            for (int j = 0; j < s[i] - 'a'; ++ j )
                rk += seen[j];
            
            res = (res + rk * fact % MOD * quick_pow(dup, MOD - 2, MOD) % MOD) % MOD;
            fact = fact * (n - i) % MOD;
        }
        return res;
    }
};
```

##### **C++ 数位DP**

```cpp
// 数位DP
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    const static int N = 3010;
    
    LL f[N], g[N];
    
    LL qmi(LL a, int b) {
        LL res = 1;
        while (b) {
            if (b & 1)
                res = res * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return res;
    }
    
    // 重复排列问题
    int get(vector<int> & cnt) {
        int sum = 0;
        for (int i = 0; i < 26; ++ i )
            sum += cnt[i];
        int res = f[sum];
        for (int i = 0; i < 26; ++ i )
            res = (LL)res * g[cnt[i]] % MOD;
        return res;
    }
    
    int makeStringSorted(string s) {
        f[0] = g[0] = 1;
        for (int i = 1; i <= s.size(); ++ i ) {
            f[i] = f[i - 1] * i % MOD;
            g[i] = qmi(f[i], MOD - 2);
        }
        
        int res = 0;
        vector<int> cnt(26, 0);
        for (auto c : s)
            cnt[c - 'a'] ++ ;
        for (auto c : s) {
            int x = c - 'a';
            for (int i = 0; i < x; ++ i ) {
                if (!cnt[i])
                    continue;
                cnt[i] -- ;
                res = (res + get(cnt)) % MOD;
                cnt[i] ++ ;
            }
            cnt[x] -- ;
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

> [!NOTE] **[LeetCode 2375. 根据模式串构造最小数字](https://leetcode.cn/problems/construct-smallest-number-from-di-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始在想贪心填，实际上 `9! = 362880` 可以直接生成
> 
> `next_permutation`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    
    bool check(string & a, string & b) {
        for (int i = 1; i <= n; ++ i )
            if ((a[i] > a[i - 1]) && (b[i - 1] != 'I') || (a[i] < a[i - 1]) && (b[i - 1] != 'D'))
                return false;
        return true;
    }
    
    string smallestNumber(string pattern) {
        n = pattern.size();
        string t, res;
        for (int i = 1; i <= n + 1; ++ i )
            t.push_back('0' + i);
        
        do {
            if (check(t, pattern)) {
                res = t;
                break;
            }
        } while (next_permutation(t.begin(), t.end()));
        
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

### min

> [!NOTE] **[LeetCode 1189. “气球” 的最大数量](https://leetcode.cn/problems/maximum-number-of-balloons/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> STL min 用法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxNumberOfBalloons(string text) {
        unordered_map<char, int> cnt;
        for (auto c : text) ++ cnt[c];
        int res = INT_MAX;
        res = min(res, cnt['b']);
        res = min(res, cnt['a']);
        res = min(res, cnt['l'] / 2);
        res = min(res, cnt['o'] / 2);
        res = min(res, cnt['n']);
        return res;
    }
};
```

##### **C++ STL**

```cpp
class Solution {
public:
    int maxNumberOfBalloons(string text) {
        map<char, int> F;
        for (auto c : text) F[c] ++;
        return min({F['b'], F['a'], F['l']/2, F['o']/2, F['n']});
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

### 链表

> [!NOTE] **[LeetCode 2296. 设计一个文本编辑器](https://leetcode.cn/problems/design-a-text-editor/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 可以链表模拟，重点在于 STL 的一些返回值及操作
> 
>   TODO
> 
> - 标准做法 **对顶栈**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ STL**

```cpp
class TextEditor {
public:
    list<char> L;
    list<char>::iterator p;
    
    TextEditor() {
        p = L.begin();
    }
    
    string print() {
        string t;
        for (auto c : L)
            t.push_back(c);
        return t;
    }
    
    void addText(string text) {
        for (auto c : text)
            L.insert(p, c); // ATTENTION STL insert 后的变化
    }
    
    int deleteText(int k) {
        int c = 0;
        for (; k && p != L.begin(); -- k )
            p = L.erase(prev(p)), c ++ ;    // ATTENTION erase 的返回值
        return c;
    }
    
    string getText() {
        string t;
        auto it = p;
        for (int k = 10; k && it != L.begin(); -- k ) {
            it = prev(it);
            t.push_back(*it);
        }
        reverse(t.begin(), t.end());
        return t;
    }
    
    string cursorLeft(int k) {
        for (; k && p != L.begin(); -- k )
            p = prev(p);
        return getText();
    }
    
    string cursorRight(int k) {
        for (; k && p != L.end(); -- k )
            p = next(p);
        return getText();
    }
};

/**
 * Your TextEditor object will be instantiated and called as such:
 * TextEditor* obj = new TextEditor();
 * obj->addText(text);
 * int param_2 = obj->deleteText(k);
 * string param_3 = obj->cursorLeft(k);
 * string param_4 = obj->cursorRight(k);
 */
```

##### **C++ 对顶栈**

```cpp
class TextEditor {
public:
    vector<char> l, r;
    
    TextEditor() {
        l.clear(), r.clear();
    }
    
    void addText(string text) {
        for (auto c : text)
            l.push_back(c);
    }
    
    int deleteText(int k) {
        int c = 0;
        while (l.size() && k)
            l.pop_back(), k -- , c ++ ;
        return c;
    }
    
    string getText() {
        string t;
        int n = l.size();
        for (int i = max(0, n - 10); i < n; ++ i )
            t.push_back(l[i]);
        return t;
    }
    
    string cursorLeft(int k) {
        while (l.size() && k) {
            char c = l.back();
            r.push_back(c), l.pop_back();
            k -- ;
        }
        return getText();
    }
    
    string cursorRight(int k) {
        while (r.size() && k) {
            char c = r.back();
            l.push_back(c), r.pop_back();
            k -- ;
        }
        return getText();
    }
};

/**
 * Your TextEditor object will be instantiated and called as such:
 * TextEditor* obj = new TextEditor();
 * obj->addText(text);
 * int param_2 = obj->deleteText(k);
 * string param_3 = obj->cursorLeft(k);
 * string param_4 = obj->cursorRight(k);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *