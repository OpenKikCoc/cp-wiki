## 习题

> [!NOTE] **[AcWing 1416. 包装矩形](https://www.acwing.com/problem/content/1418/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂分类讨论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
#define x first
#define y second

const int N = 4;

PII rt[N];
int p[N] = {0, 1, 2, 3};
vector<PII> ans;

// 宽 高
void update(int a, int b) {
    if (a > b) swap(a, b);
    if (ans.empty() || a * b < ans[0].x * ans[0].y) ans = {{a, b}};
    else if (a * b == ans[0].x * ans[0].y) ans.push_back({a, b});
}

void work() {
    auto a = rt[p[0]], b = rt[p[1]], c = rt[p[2]], d = rt[p[3]];
    update(a.x + b.x + c.x + d.x, max(a.y, max(b.y, max(c.y, d.y))));
    update(max(a.x + b.x + c.x, d.x), d.y + max(a.y, max(b.y, c.y)));
    update(max(a.x + b.x, d.x) + c.x, max(max(a.y, b.y) + d.y, c.y));
    update(a.x + d.x + max(b.x, c.x), max(a.y, max(d.y, b.y + c.y)));
    update(max(a.x, d.x) + b.x + c.x, max(b.y, max(c.y, a.y + d.y)));
    if (b.x >= a.x && c.y >= b.y) {
        // 最后一种情况 最右侧底部的方块u较矮
        if (c.y < a.y + b.y) {
            if (d.x + a.x <= b.x + c.x)
                update(b.x + c.x, max(a.y + b.y, c.y + d.y));
        } else update(max(d.x, b.x + c.x), c.y + d.y);
    }
}

int main() {
    for (int i = 0; i < 4; ++ i ) cin >> rt[i].x >> rt[i].y;
    
    // 4 * 3 * 2 * 1 = 24 全排列
    for (int i = 0; i < 24; ++ i ) {
        // 每个矩形是否翻转
        for (int j = 0; j < 16; ++ j ) {
            for (int k = 0; k < 4; ++ k )
                if (j >> k & 1)
                    swap(rt[p[k]].x, rt[p[k]].y);
            work();
            for (int k = 0; k < 4; ++ k )
                if (j >> k & 1)
                    swap(rt[p[k]].x, rt[p[k]].y);
        }
        next_permutation(p, p + 4);
    }
    
    sort(ans.begin(), ans.end());
    ans.erase(unique(ans.begin(), ans.end()), ans.end());
    
    cout << ans[0].x * ans[0].y << endl;
    for (auto & a : ans) cout << a.x << ' ' << a.y << endl;
    
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