## 习题

> [!NOTE] **[Luogu 覆盖墙壁](https://www.luogu.com.cn/problem/P1990)**
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

// https://www.acwing.com/solution/content/16126/

const int N = 1e6 + 10, MOD = 1e4;

int n;
int f[N][4];
// f[i][j] :
//      i-1 has been filled, the i-th state is j

int main() {
    cin >> n;
    
    f[0][3] = f[0][0] = 1;
    
    for (int i = 1; i <= n; ++ i ) {
        f[i][0] = f[i - 1][3];
        f[i][1] = (f[i - 1][0] + f[i - 1][2]) % MOD;
        f[i][2] = (f[i - 1][0] + f[i - 1][1]) % MOD;
        f[i][3] = (f[i - 1][3] + f[i - 1][0] + f[i - 1][1] + f[i - 1][2]) % MOD;
    }
    cout << f[n][0] << endl;
    
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