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

> [!NOTE] **[Luogu 地毯填补问题](https://www.luogu.com.cn/problem/P1228)**
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

int n, x, y;

void dfs(
    int t, int sx, int sy, int x,
    int y) {  // sx，sy代表此正方形左上角位置，xy表示公主所在位置（或被占掉的位置）
    if (t == 0)
        return;
    int t1 = (1 << t - 1);           //小正方形边长
    if (x < sx + t1 && y < sy + t1)  //左上角
    {
        printf("%d %d %d\n", sx + t1, sy + t1, 1);
        dfs(t - 1, sx, sy, x, y), dfs(t - 1, sx, sy + t1, sx + t1 - 1, sy + t1);
        dfs(t - 1, sx + t1, sy, sx + t1, sy + t1 - 1),
            dfs(t - 1, sx + t1, sy + t1, sx + t1, sy + t1);
    } else if (x < sx + t1)  //右上角
    {
        printf("%d %d %d\n", sx + t1, sy + t1 - 1, 2);
        dfs(t - 1, sx, sy, sx + t1 - 1, sy + t1 - 1),
            dfs(t - 1, sx, sy + t1, x, y);
        dfs(t - 1, sx + t1, sy, sx + t1, sy + t1 - 1),
            dfs(t - 1, sx + t1, sy + t1, sx + t1, sy + t1);
    } else if (y < sy + t1)  //左下角
    {
        printf("%d %d %d\n", sx + t1 - 1, sy + t1, 3);
        dfs(t - 1, sx, sy, sx + t1 - 1, sy + t1 - 1),
            dfs(t - 1, sx, sy + t1, sx + t1 - 1, sy + t1);
        dfs(t - 1, sx + t1, sy, x, y),
            dfs(t - 1, sx + t1, sy + t1, sx + t1, sy + t1);
    } else  //右下角
    {
        printf("%d %d %d\n", sx + t1 - 1, sy + t1 - 1, 4);
        dfs(t - 1, sx, sy, sx + t1 - 1, sy + t1 - 1),
            dfs(t - 1, sx, sy + t1, sx + t1 - 1, sy + t1);
        dfs(t - 1, sx + t1, sy, sx + t1, sy + t1 - 1),
            dfs(t - 1, sx + t1, sy + t1, x, y);
    }
}

int main() {
    scanf("%d%d%d", &n, &x, &y);
    dfs(n, 1, 1, x, y);
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

> [!NOTE] **[Luogu 平面上的最接近点对](https://www.luogu.com.cn/problem/P1257)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 最经典分治之一
> 
> 推理排序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// TODO
// 分治

const int N = 1e5 + 10, INF = 1 << 20;

int n, t[N];
struct Point {
    double x, y;
} S[N];

double dist(int i, int j) {
    double dx = S[i].x - S[j].x;
    double dy = S[i].y - S[j].y;
    return sqrt(dx * dx + dy * dy);
}

double merge(int l, int r) {
    if (l >= r)
        return INF;
    // if (l + 1 == r)
    // return dist(l, r);

    int m = l + r >> 1;
    double d1 = merge(l, m), d2 = merge(m + 1, r);
    double d = min(d1, d2);

    int k = 0;
    for (int i = l; i <= r; i++)
        if (fabs(S[m].x - S[i].x) <= d)
            t[k++] = i;

    sort(t, t + k, [](const int &a, const int &b) { return S[a].y < S[b].y; });

    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k && S[t[j]].y - S[t[i]].y < d; j++)
            d = min(d, dist(t[i], t[j]));
    return d;
}

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        scanf("%lf%lf", &S[i].x, &S[i].y);

    sort(S, S + n, [](const Point &a, const Point &b) {
        if (a.x == b.x)
            return a.y < b.y;
        else
            return a.x < b.x;
    });

    printf("%.4lf\n", merge(0, n - 1));

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