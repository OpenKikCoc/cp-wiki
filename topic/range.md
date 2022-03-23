## 习题

> [!NOTE] **[Codeforces A. Lucky Sum](https://codeforces.com/problemset/problem/121/A)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始从左侧开始扫，实现很复杂，还wa
> 
> https://codeforces.com/contest/121/submission/109555573
> 
> 直接右区间 - 左区间即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Lucky Sum
// Contest: Codeforces - Codeforces Beta Round #91 (Div. 1 Only)
// URL: https://codeforces.com/problemset/problem/121/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// 2^11 最多2048个 lucky number
using LL = long long;
const int N = 2100;

LL ln[N], cnt;

void dfs(int u, LL v) {
    ln[cnt++] = v;
    if (u == 10)
        return;

    dfs(u + 1, v * 10 + 4);
    dfs(u + 1, v * 10 + 7);
}

void init() {
    dfs(0, 0);
    sort(ln, ln + cnt);
}

LL f(int n) {
    if (!n)
        return 0;

    LL ret = 0;
    for (int i = 1; i < cnt; ++i)
        if (ln[i] < n)
            ret += (ln[i] - ln[i - 1]) * ln[i];
        else {
            ret += (n - ln[i - 1]) * ln[i];
            break;
        }
    return ret;
}

int main() {
    init();

    int l, r;
    cin >> l >> r;

    cout << f(r) - f(l - 1) << endl;

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