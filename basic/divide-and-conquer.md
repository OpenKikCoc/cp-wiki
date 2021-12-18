## 习题

> [!NOTE] **[AcWing 95. 费解的开关](https://www.acwing.com/problem/content/97/)**
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
#include<bits/stdc++.h>

using namespace std;

const int N = 6;

char g[N][N], bg[N][N];

int dx[5] = {-1, 0, 1, 0, 0}, dy[5] = {0, 1, 0, -1, 0};

void turn(int x, int y) {
    for (int i = 0; i < 5; ++ i ) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx < 0 || nx >= 5 || ny < 0 || ny >= 5) continue;
        g[nx][ny] ^= 1;
    }
}

int main() {
    int T;
    scanf("%d", &T);
    while (T -- ) {
        for (int i = 0; i < 5; ++ i ) scanf("%s", bg[i]);
        int res = 10;
        // 枚举第一行的所有操作方案 2^5
        for (int op = 0; op < 32; ++ op ) {
            int cnt = 0;
            memcpy(g, bg, sizeof g);
            for (int i = 0; i < 5; ++ i )
                if (op >> i & 1) {
                    turn(0, i);
                    ++ cnt;
                }
            // 递推1~4行开关的状态
            for (int i = 0; i < 4; ++ i )
                for (int j = 0; j < 5; ++ j )
                    if (g[i][j] == '0') {
                        turn(i + 1, j);
                        ++ cnt;
                    }
            bool success = true;
            for (int i = 0; i < 5; ++ i )
                if (g[4][i] == '0')
                    success = false;
            if (success && res > cnt) res = cnt;
        }
        if (res > 6) res = -1;
        printf("%d\n", res);
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

> [!NOTE] **[AcWing 97. 约数之和](https://www.acwing.com/problem/content/99/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> ![png](https://github.com/OpenKikCoc/AcWing/raw/master/02_senior/97/题解.png)
> 
> 约数个数 & 约数之和 公式
> 
> 本题本身也有公式

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

const int mod = 9901;

int qmi(int a, int k) {
    int res = 1;
    a %= mod;
    while (k) {
        if (k & 1) res = res * a % mod;
        a = a * a % mod;
        k >>= 1;
    }
    return res;
}

int sum(int p, int k) {
    if (k == 1) return 1;
    if (k % 2 == 0) return (1 + qmi(p, k / 2)) * sum(p, k / 2) % mod;
    return (sum(p, k - 1) + qmi(p, k - 1)) % mod;
}

int main() {
    int a, b;
    cin >> a >> b;
    int res = 1;
    
    // 对a分解质因数
    for (int i = 2; i * i <= a; ++ i )
        if (a % i == 0) {
            int s = 0;
            while (a % i == 0) {
                a /= i, ++ s;
            }
            res = res * sum(i, b * s + 1) % mod;
        }
    if (a > 1) res = res * sum(a, b + 1) % mod;
    if (a == 0) res = 0;
    cout << res << endl;
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

> [!NOTE] **[AcWing 98. 分形之城](https://www.acwing.com/problem/content/100/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 不断地重复旋转复制，也就是N级城市，可以由4个N−1级城市构造，

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

struct Point {
    LL x, y;
};

Point get(LL n, LL a) {
    if (n == 0) return {0, 0};
    LL block = 1ll << n * 2 - 2, len = 1ll << n - 1;
    auto p = get(n - 1, a % block);
    LL x = p.x, y = p.y;
    int z = a / block;
    
    if (z == 0) return {y, x};
    else if(z == 1) return {x, y + len};
    else if (z == 2) return {x + len, y + len};
    return {len * 2 - 1 - y, len - 1 - x};
}

int main() {
    int T;
    cin >> T;
    while (T -- ) {
        LL n, a, b;
        cin >> n >> a >> b;
        auto pa = get(n, a - 1);
        auto pb = get(n, b - 1);
        double dx = pa.x - pb.x, dy = pa.y - pb.y;
        printf("%.0lf\n", sqrt(dx * dx + dy * dy) * 10);
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