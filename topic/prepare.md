## 习题

> [!NOTE] **[LeetCode 336. 回文对](https://leetcode-cn.com/problems/palindrome-pairs/)**
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
    vector<vector<int>> palindromePairs(vector<string>& words) {
        unordered_map<string, int> S;
        for (int i = 0; i < words.size(); i ++ ) {
            string key = words[i];
            reverse(key.begin(), key.end());
            S[key] = i;
        }
        vector<vector<int>> res;
        if (S.count("")) {
            for (int i = 0; i < words.size(); i ++ )
                if (words[i] != "" && is_palindrome(words[i]))
                    res.push_back({S[""], i});
        }
        for (int i = 0; i < words.size(); i ++ )
            for (int j = 0; j < words[i].size(); j ++ ) {
                string left = words[i].substr(0, j);
                string right = words[i].substr(j);
                if (S.count(left) && is_palindrome(right) && S[left] != i) res.push_back({i, S[left]});
                if (S.count(right) && is_palindrome(left) && S[right] != i) res.push_back({S[right], i});
            }
        return res;
    }

    bool is_palindrome(string &word) {
        for (int i = 0, j = word.size() - 1; i < j; i ++ , j -- )
            if (word[i] != word[j])
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

> [!NOTE] **[Codeforces Modified GCD](http://codeforces.com/problemset/problem/75/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 预处理 `gcd` 的所有因子，随后二分查找

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Modified GCD
// Contest: Codeforces - Codeforces Beta Round #67 (Div. 2)
// URL: https://codeforces.com/problemset/problem/75/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, a, b;

    cin >> a >> b >> n;
    int g = __gcd(a, b);
    vector<int> f;
    for (int i = 1; i <= g / i; ++i)
        if (g % i == 0) {
            f.push_back(i);
            if (g / i != i)
                f.push_back(g / i);
        }

    sort(f.begin(), f.end());
    int m = f.size();

    for (int i = 0; i < n; ++i) {
        int L, R;
        cin >> L >> R;
        int l = 0, r = m;
        while (l < r) {
            int mid = l + r >> 1;
            if (f[mid] <= R)
                l = mid + 1;
            else
                r = mid;
        }
        if (l && f[l - 1] >= L)
            cout << f[l - 1] << endl;
        else
            cout << -1 << endl;
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
