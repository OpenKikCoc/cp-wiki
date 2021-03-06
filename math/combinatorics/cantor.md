康托展开可以用来求一个 $1\sim n$ 的任意排列的排名。

## 什么是排列的排名？

把 $1\sim n$ 的所有排列按字典序排序，这个排列的位次就是它的排名。

## 时间复杂度？

康托展开可以在 $O(n^2)$ 的复杂度内求出一个排列的排名，在用到树状数组优化时可以做到 $O(n\log n)$。

## 怎么实现？

因为排列是按字典序排名的，因此越靠前的数字优先级越高。也就是说如果两个排列的某一位之前的数字都相同，那么如果这一位如果不相同，就按这一位排序。

比如 $4$ 的排列，$[2,3,1,4]<[2,3,4,1]$，因为在第 $3$ 位出现不同，则 $[2,3,1,4]$ 的排名在 $[2,3,4,1]$ 前面。

## 举个栗子

我们知道长为 $5$ 的排列 $[2,5,3,4,1]$ 大于以 $1$ 为第一位的任何排列，以 $1$ 为第一位的 $5$ 的排列有 $4!$ 种。这是非常好理解的。但是我们对第二位的 $5$ 而言，它大于 **第一位与这个排列相同的，而这一位比 $5$ 小的** 所有排列。不过我们要注意的是，这一位不仅要比 $5$ 小，还要满足没有在当前排列的前面出现过，不然统计就重复了。因此这一位为 $1,3$ 或 $4$，第一位为 $2$ 的所有排列都比它要小，数量为 $3\times 3!$。

按照这样统计下去，答案就是 $1+4!+3\times 3!+2!+1=46$。注意我们统计的是排名，因此最前面要 $+1$。

注意到我们每次要用到 **当前有多少个小于它的数还没有出现**，这里用树状数组统计比它小的数出现过的次数就可以了。

## 逆康托展开

因为排列的排名和排列是一一对应的，所以康托展开满足双射关系，是可逆的。可以通过类似上面的过程倒推回来。

如果我们知道一个排列的排名，就可以推出这个排列。因为 $4!$ 是严格大于 $3\times 3!+2\times 2!+1\times 1!$ 的，所以可以认为对于长度为 $5$ 的排列，排名 $x$ 除以 $4!$ 向下取整就是有多少个数小于这个排列的第一位。

## 引用上面展开的例子

首先让 $46-1=45$，$45$ 代表着有多少个排列比这个排列小。$\lfloor\frac {45}{4!}\rfloor=1$，有一个数小于它，所以第一位是 $2$。

此时让排名减去 $1\times 4!$ 得到 $21$，$\lfloor\frac {21}{3!}\rfloor=3$，有 $3$ 个数小于它，去掉已经存在的 $2$，这一位是 $5$。

$21-3\times 3!=3$，$\lfloor\frac {3}{2!}\rfloor=1$，有一个数小于它，那么这一位就是 $3$。

让 $3-1\times 2!=1$，有一个数小于它，这一位是剩下来的第二位，$4$，剩下一位就是 $1$。即 $[2,5,3,4,1]$。

实际上我们得到了形如 **有两个数小于它** 这一结论，就知道它是当前第 $3$ 个没有被选上的数，这里也可以用线段树维护，时间复杂度为 $O(n\log n)$。

> [!NOTE] **[LeetCode 60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)**
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
    string getPermutation(int n, int k) {
        string res;
        vector<bool> st(n);

        for (int i = 0; i < n; ++ i ) {
            // 剩下的个数
            int f = 1;
            for (int j = 1; j < n - i; ++ j ) f *= j;

            for (int j = 0; j < n; ++ j )
                if (!st[j]) {
                    if (k <= f) {
                        res += to_string(j + 1);
                        st[j] = true;
                        break;
                    }
                    k -= f;
                }
        }
        return res;
    }
};

class Solution {
public:
    string getPermutation(int n, int k) {
        string res;
        for (int i = 1; i <= n; i ++ ) res += to_string(i);
        for (int i = 0; i < k - 1; i ++ ) {
            next_permutation(res.begin(), res.end());
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    // 康托展开和逆康托展开  https://blog.csdn.net/ajaxlt/article/details/86544074
    string getPermutation(int n, int k) {
        int fac[10] = {1};
        for (int i = 1; i < 10; ++ i ) fac[i] = fac[i - 1] * i;
        k = k - 1;
        vector<char> chs = {'1','2','3','4','5','6','7','8','9'};
        string res;
        while (n -- ) {
            int min = k / fac[n]; // 得到小的个数
            res += chs[min];
            chs.erase(chs.begin() + min);
            k %= fac[n];
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