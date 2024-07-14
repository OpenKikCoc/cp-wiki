## 习题

> [!NOTE] **[LeetCode 336. 回文对](https://leetcode.cn/problems/palindrome-pairs/)**
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

> [!NOTE] **[LeetCode 2911. 得到 K 个半回文串的最少修改次数](https://leetcode.cn/problems/minimum-changes-to-make-k-semi-palindromes/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性 DP + 区间 DP 预处理优化
> 
> 注意预处理计算细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 210;

    int c[N][N], f[N][N];

    int minimumChanges(string s, int k) {
        int n = s.size();
        
        // 预处理
        // 计算每一个合法区间被修改为半回文串的代价
        // ATTENTION 是模d下不同分组的总开销
        memset(c, 0x3f, sizeof c);

        // 区间 dp 预处理
        for (int d = 1; d < n; ++ d ) {
            // 临时数组 记录当前d形态下的单组的回文开销
            static int t[N][N];
            for (int i = 0; i < N; ++ i )
                t[i][i] = 0;
            // 思考 枚举顺序与数据维护的顺序
            for (int r = d; r < n; ++ r ) {
                int l = r - d;
                t[l][r] = s[l] != s[r];
                for (int k = l - d; k >= 0; k -= d )
                    t[k][r] = t[k + d][r - d] + (s[k] != s[r]);
            }

            /*
            for (int i = 0; i < n - d; ++ i )
                // ATTENTION j = i + d - 1 就会 WA
                for (int j = i + d; j < n; ++ j ) {
                    int len = j - i + 1;
                    if (len % d)
                        continue;
                    
                    int sum = 0;
                    for (int x = 0; x < d; ++ x )
                        sum += t[i + x][j + 1 - d + x];
                    c[i][j] = min(c[i][j], sum);
                }
            */

            // 在d形态下 一段区间内按模d划分后的总开销
            for (int r = d - 1; r < n; ++ r )
                // int w = r - l + 1;
                // ATTENTION w % d == 0 是必要条件
                //
                // ATTENTION 必须要减2个d ==> 否则是单个元素 无法算价值 ==> 思考
                for (int l = r - d + 1 - d; l >= 0; l -= d ) {
                    int sum = 0;
                    for (int x = 0; x < d; ++ x )
                        sum += t[l + x][r - d + 1 + x];
                    c[l][r] = min(c[l][r], sum);
                }
        }

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= k; ++ j )
                // 上一个结束的位置在 p (已经偏移 1-idx)
                // 新带来的开销是 c[p][i - 1]
                for (int p = 0; p < i; ++ p )
                    f[i][j] = min(f[i][j], f[p][j - 1] + c[p][i - 1]);
        return f[n][k];
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
