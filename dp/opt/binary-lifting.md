## 习题

> [!NOTE] **[Luogu [NOIP2012 提高组] 开车旅行](https://www.luogu.com.cn/problem/P1081)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 倍增优化DP
> 
> 经典 倍增预处理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// https://www.acwing.com/solution/content/6562/
// 1. 倍增预处理
// 2. 遍历计算

using LL = long long;
using PLI = pair<LL, int>;

const int N = 1e5 + 10, M = 17;
const LL INF = 1e12;

int n;
int h[N];
int ga[N], gb[N];
int f[M][N][2];
LL da[M][N][2], db[M][N][2];

// 计算从某点出发下一个目的地
void init_g() {
    set<PLI> S;
    S.insert({INF, 0}), S.insert({INF + 1, 0});
    S.insert({-INF, 0}), S.insert({-INF - 1, 0});
    
    // 邻值问题
    // 找右侧 故逆序
    for (int i = n; i; -- i ) {
        PLI t(h[i], i);
        auto j = S.lower_bound(t);
        // 找上下四个
        j ++ ;
        vector<PLI> cand;
        for (int k = 0; k < 4; ++ k ) {
            cand.push_back(*j);
            j -- ;
        }
        LL d1 = INF, d2 = INF;
        int p1 = 0, p2 = 0;
        for (int k = 3; k >= 0; k -- ) {
            LL d = abs(h[i] - cand[k].first);
            if (d < d1) {
                d2 = d1, d1 = d;
                p2 = p1, p1 = cand[k].second;
            } else if (d < d2) {
                d2 = d;
                p2 = cand[k].second;
            }
        }
        ga[i] = p2, gb[i] = p1;
        S.insert(t);
    }
}

// 计算从某点出发 2^i 天会到达哪个目的地
void init_f() {
    for (int i = 0; i < M; ++ i )
        for (int j = 1; j <= n; ++ j )
            if (!i)
                f[0][j][0] = ga[j], f[0][j][1] = gb[j];
            else {
                for (int k = 0; k < 2; ++ k )
                    // ℹ == 1 时未经过一个完整周期 必然换人
                    if (i == 1)
                        // f[i][j][k] = f[i - 1][f[i - 1][j][k]][1 - k];
                        f[1][j][k] = f[0][f[0][j][k]][1 - k];
                    // 完整周期 还是原来的人(k)
                    else
                        f[i][j][k] = f[i - 1][f[i - 1][j][k]][k];
            }
}

int get_dist(int a, int b) {
    return abs(h[a] - h[b]);
}

// 计算从某点出发 2^i 天各自走了多少距离
void init_d() {
    for (int i = 0; i < M; ++ i )
        for (int j = 1; j <= n; ++ j )
            if (!i) {
                da[0][j][0] = get_dist(j, ga[j]), da[0][j][1] = 0;
                db[0][j][1] = get_dist(j, gb[j]), db[0][j][0] = 0;
            } else {
                for (int k = 0; k < 2; ++ k )
                    if (i == 1) {
                        da[1][j][k] = da[0][j][k] + da[0][f[0][j][k]][1 - k];
                        db[1][j][k] = db[0][j][k] + db[0][f[0][j][k]][1 - k];
                    } else {
                        da[i][j][k] = da[i - 1][j][k] + da[i - 1][f[i - 1][j][k]][k];
                        db[i][j][k] = db[i - 1][j][k] + db[i - 1][f[i - 1][j][k]][k];
                    }
            }
}

// 计算从某点出发 ab各走多远
void calc(int p, int x, int & la, int & lb) {
    la = lb = 0;
    for (int i = M - 1; i >= 0; -- i )
        if (f[i][p][0] && la + lb + da[i][p][0] + db[i][p][0] <= x) {
            la += da[i][p][0], lb += db[i][p][0];
            p = f[i][p][0];
        }
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> h[i];
    
    init_g();
    init_f();
    init_d();
    
    int p, x;
    cin >> x;
    int res = 0, max_h = 0;
    double min_ratio = INF;
    for (int i = 1; i <= n; ++ i ) {
        int la, lb;
        calc(i, x, la, lb);
        double ratio = lb ? (double)la / lb : INF;
        if (ratio < min_ratio || ratio == min_ratio && h[i] > max_h) {
            min_ratio = ratio;
            max_h = h[i];
            res = i;
        }
    }
    cout << res << endl;
    
    int m;
    cin >> m;
    while (m -- ) {
        cin >> p >> x;
        int la, lb;
        calc(p, x, la, lb);
        cout << la << ' ' << lb << endl;
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