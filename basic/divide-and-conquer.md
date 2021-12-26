## 习题

### 递归

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

### 分治

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