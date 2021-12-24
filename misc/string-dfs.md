## 习题

> [!NOTE] **[Luogu 外星密码](https://www.luogu.com.cn/problem/P1928)**
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

// https://leetcode-cn.com/problems/number-of-atoms/

string s;
int n;

string dfs(int & u) {
    string ret;
    while (u < n) {
        if (s[u] == '[') {
            u ++ ;  // '['
            
            int k = u, cnt = 0;
            while (k < n && isdigit(s[k]))
                k ++ ;
            if (k > u) {
                // must be
                cnt = stoi(s.substr(u, k - u));
                u = k;
            }
            
            auto t = dfs(u);
            for (int i = 0; i < cnt; ++ i )
                ret += t;
            
            u ++ ;  // ']'
        } else if (s[u] == ']') {
            break;
        } else {
            int k = u + 1;
            while (k < n && isalpha(s[k]))
                k ++ ;
            
            auto ctx = s.substr(u, k - u);
            u = k;
            
            ret += ctx;    
        }
    }
    return ret;
}

int main() {
    cin >> s;
    n = s.size();
    
    int p = 0;
    cout << dfs(p) << endl;
    
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