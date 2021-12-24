## 习题

> [!NOTE] **[Luogu 对角线](https://www.luogu.com.cn/problem/P2181)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 数学

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using ULL = unsigned long long;

ULL n;

int main() {
    cin >> n;
    cout << n * (n - 1) / 2 * (n - 2) / 3 * (n - 3) / 4 << endl;
    
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