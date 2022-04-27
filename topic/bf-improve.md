## 习题

> [!NOTE] **[Codeforces The Brand New Function](http://codeforces.com/problemset/problem/243/A)**
> 
> 题意: TODO
> 
> 定义函数 $f(l,r)$ $(1 \le l,r \le n)$ ，表示序列的子串 $[l,r]$ 各项的 `或` 和: 
> 
> $f(l,r)=a_l|a_{l+1}|⋯|a_r$
> 
> 求整个数组有多少个不同的 `或` 和

> [!TIP] **思路**
> 
> **非常经典的暴力优化**
> 
> 需要严格数学推导

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. The Brand New Function
// Contest: Codeforces - Codeforces Round #150 (Div. 1)
// URL: https://codeforces.com/problemset/problem/243/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 1e5 + 10;

int n;
int a[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n;

    unordered_set<int> S;
    for (int i = 1, x; i <= n; ++i) {
        cin >> a[i];
        S.insert(a[i]);
        // TRICK: 经过严谨数学证明的剪枝与实现方式
        for (int j = i - 1; j; --j) {
            // ATTENTION: trick
            // if-condition 满足时必然此前已计算过同样值的了，直接break
            if ((a[j] | a[i]) == a[j])
                break;
            // 为什么可以直接或 ？ 更改后是否影响正确性 ？
            // 1. 区间具有包含性质
            // 2. 由 1 后续使用的必然包含上一次使用的，正确性不变
            a[j] |= a[i];
            S.insert(a[j]);
        }
    }

    cout << S.size() << endl;

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
