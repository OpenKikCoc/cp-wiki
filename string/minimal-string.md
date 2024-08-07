最小表示法是用于解决字符串最小表示问题的方法。

## 字符串的最小表示

### 循环同构

当字符串 $S$ 中可以选定一个位置 $i$ 满足

$$
S[i\cdots n]+S[1\cdots i-1]=T
$$

则称 $S$ 与 $T$ 循环同构

### 最小表示

字符串 $S$ 的最小表示为与 $S$ 循环同构的所有字符串中字典序最小的字符串

## 最小表示法

### 算法核心

考虑对于一对字符串 $A,B$, 它们在原字符串 $S$ 中的起始位置分别为 $i,j$, 且它们的前 $k$ 个字符均相同，即

$$
A[i \cdots i+k-1]=B[j \cdots j+k-1]
$$

不妨先考虑 $A[i+k]>B[j+k]$ 的情况，我们发现起始位置下标 $l$ 满足 $i\le l\le i+k$ 的字符串均不能成为答案。因为对于任意一个字符串 $S_{i+p}$（表示以 $i+p$ 为起始位置的字符串）一定存在字符串 $S_{j+p}$ 比它更优。

所以我们比较时可以跳过下标 $l\in [i,i+k]$, 直接比较 $S_{i+k+1}$

这样，我们就完成了对于上文暴力的优化。

### 时间复杂度

$O(n)$

### 算法流程

1. 初始化指针 $i$ 为 $0$，$j$ 为 $1$；初始化匹配长度 $k$ 为 $0$
2. 比较第 $k$ 位的大小，根据比较结果跳转相应指针。若跳转后两个指针相同，则随意选一个加一以保证比较的两个字符串不同
3. 重复上述过程，直到比较结束
4. 答案为 $i,j$ 中较小的一个

### 代码

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

###### **C++**

```cpp
// C++ Version
int k = 0, i = 0, j = 1;
while (k < n && i < n && j < n) {
    if (sec[(i + k) % n] == sec[(j + k) % n]) {
        k++;
    } else {
        sec[(i + k) % n] > sec[(j + k) % n] ? i = i + k + 1 : j = j + k + 1;
        if (i == j) i++;
        k = 0;
    }
}
i = min(i, j);
```

###### **Python**

```python
# Python Version
k, i, j = 0, 0, 1
while k < n and i < n and j < n:
    if sec[(i + k) % n] == sec[(j + k) % n]:
        k += 1
    else:
        if sec[(i + k) % n] > sec[(j + k) % n]:
            i = i + k + 1
        else:
            j = j + k + 1
        if i == j:
            i += 1
        k = 0
i = min(i, j)
```

<!-- tabs:end -->
</details>

## 习题

### 最小表示法

> [!NOTE] **[AcWing 158. 项链](https://www.acwing.com/problem/content/160/)**
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

const int N = 2000010;

int n;
char a[N], b[N];

int get_min(char s[]) {
    int i = 0, j = 1;
    while (i < n && j < n) {
        int k = 0;
        while (k < n && s[i + k] == s[j + k])
            k ++ ;
        if (k == n)
            break;
        if (s[i + k] > s[j + k])
            i += k + 1;
        else
            j += k + 1;
        if (i == j)
            j ++ ;
    }
    int k = min(i, j);
    s[k + n] = 0;   // 标记
    return k;
}

int main() {
    // scanf("%s%s", a, b);
    cin >> a >> b;
    n = strlen(a);
    memcpy(a + n, a, n);
    memcpy(b + n, b, n);
    
    int x = get_min(a), y = get_min(b);
    if (strcmp(a + x, b + y))
        cout << "No" << endl;
    else {
        cout << "Yes" << endl;
        cout << a + x << endl;
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

> [!NOTE] **[AcWing 1410. 隐藏密码](https://www.acwing.com/problem/content/1412/)**
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
// 求字符串的最小表示
// 经典算法 背过
#include <bits/stdc++.h>
using namespace std;

const int N = 200010;

int n;
string s;

int get_min() {
    // 用两个指针来找
    // 枚举从 i / j 开始的两个连续区间
    int i = 0, j = 1;
    while (i < n && j < n) {
        int k = 0;
        while (k < n && s[i + k] == s[j + k]) ++ k ;
        if (k == n) break;
        if (s[i + k] > s[j + k]) i += k + 1;
        else j += k + 1;
        if (i == j) ++ j ;
    }
    return min(i, j);
}

int main() {
    cin >> n;
    string line;
    while (cin >> line) s += line;
    s += s; // 后面再接一份
    cout << get_min() << endl;
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

> [!NOTE] **[LeetCode 899. 有序队列](https://leetcode.cn/problems/orderly-queue/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单推导，显然在 `k = 1` 时是最小表示法
> 
> 其他情况都可以直接搞成全局最小 `sorting`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string orderlyQueue(string s, int k) {
        if (k == 1) {
            int i = 0, j = 1, n = s.size();
            s = s + s;
            while (i < n && j < n) {
                int k = 0;
                while (k < n && s[i + k] == s[j + k])
                    k ++ ;
                if (k == n)
                    break;
                if (s[i + k] > s[j + k])
                    i += k + 1;
                else
                    j += k + 1;
                if (i == j)
                    j ++ ;
            }
            int k = min(i, j);
            return s.substr(k, n);
        }
        sort(s.begin(), s.end());
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

### 最大表示法

> [!NOTE] **[LeetCode 1163. 按字典序排在最后的子串](https://leetcode.cn/problems/last-substring-in-lexicographical-order/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 字符串最大表示法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int get_max(string s) {
        int i = 0, j = 1, n = s.size();
        while (i < n && j < n) {
            int k = 0;
            while (k < n && s[i + k] == s[j + k]) ++ k ;
            if (k == n) break;
            // `>` ---> `<`
            if (s[i + k] < s[j + k]) i += k + 1;
            else j += k + 1;
            if (i == j) ++ j ;
        }
        return min(i, j);
    }

    string lastSubstring(string s) {
        return s.substr(get_max(s));
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