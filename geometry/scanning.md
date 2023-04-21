## 简介

扫描线一般运用在图形上面，它和它的字面意思十分相似，就是一条线在整个图上扫来扫去，它一般被用来解决图形面积，周长等问题。

## Atlantis 问题

### 题意

在二维坐标系上，给出多个矩形的左下以及右上坐标，求出所有矩形构成的图形的面积。

### 解法

根据图片可知总面积可以直接暴力即可求出面积，如果数据大了怎么办？这时就需要讲到 **扫描线** 算法。

### 流程

现在假设我们有一根线，从下往上开始扫描：

![](./images/scanning.svg)

- 如图所示，我们可以把整个矩形分成如图各个颜色不同的小矩形，那么这个小矩形的高就是我们扫过的距离，那么剩下了一个变量，那就是矩形的长一直在变化。
- 我们的线段树就是为了维护矩形的长，我们给每一个矩形的上下边进行标记，下面的边标记为 1，上面的边标记为 -1，每遇到一个矩形时，我们知道了标记为 1 的边，我们就加进来这一条矩形的长，等到扫描到 -1 时，证明这一条边需要删除，就删去，利用 1 和 -1 可以轻松的到这种状态。
- 还要注意这里的线段树指的并不是线段的一个端点，而指的是一个区间，所以我们要计算的是 $r+1$ 和 $r-1$。
- 需要 [离散化](misc/discrete.md)。


```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
#define maxn 300
using namespace std;

int lazy[maxn << 3];  // 标记了这条线段出现的次数
double s[maxn << 3];

struct node1 {
    double l, r;
    double sum;
} cl[maxn << 3];  // 线段树

struct node2 {
    double x, y1, y2;
    int flag;
} p[maxn << 3];  // 坐标

//定义sort比较
bool cmp(node2 a, node2 b) { return a.x < b.x; }

//上传
void pushup(int rt) {
    if (lazy[rt] > 0)
        cl[rt].sum = cl[rt].r - cl[rt].l;
    else
        cl[rt].sum = cl[rt * 2].sum + cl[rt * 2 + 1].sum;
}

//建树
void build(int rt, int l, int r) {
    if (r - l > 1) {
        cl[rt].l = s[l];
        cl[rt].r = s[r];
        build(rt * 2, l, (l + r) / 2);
        build(rt * 2 + 1, (l + r) / 2, r);
        pushup(rt);
    } else {
        cl[rt].l = s[l];
        cl[rt].r = s[r];
        cl[rt].sum = 0;
    }
    return;
}

//更新
void update(int rt, double y1, double y2, int flag) {
    if (cl[rt].l == y1 && cl[rt].r == y2) {
        lazy[rt] += flag;
        pushup(rt);
        return;
    } else {
        if (cl[rt * 2].r > y1) update(rt * 2, y1, min(cl[rt * 2].r, y2), flag);
        if (cl[rt * 2 + 1].l < y2)
            update(rt * 2 + 1, max(cl[rt * 2 + 1].l, y1), y2, flag);
        pushup(rt);
    }
}

int main() {
    int temp = 1, n;
    double x1, y1, x2, y2, ans;
    while (scanf("%d", &n) && n) {
        ans = 0;
        for (int i = 0; i < n; i++) {
            scanf("%lf %lf %lf %lf", &x1, &y1, &x2, &y2);
            p[i].x = x1;
            p[i].y1 = y1;
            p[i].y2 = y2;
            p[i].flag = 1;
            p[i + n].x = x2;
            p[i + n].y1 = y1;
            p[i + n].y2 = y2;
            p[i + n].flag = -1;
            s[i + 1] = y1;
            s[i + n + 1] = y2;
        }
        sort(s + 1, s + (2 * n + 1));  // 离散化
        sort(p, p + 2 * n, cmp);  // 把矩形的边的横坐标从小到大排序
        build(1, 1, 2 * n);       // 建树
        memset(lazy, 0, sizeof(lazy));
        update(1, p[0].y1, p[0].y2, p[0].flag);
        for (int i = 1; i < 2 * n; i++) {
            ans += (p[i].x - p[i - 1].x) * cl[1].sum;
            update(1, p[i].y1, p[i].y2, p[i].flag);
        }
        printf("Test case #%d\nTotal explored area: %.2lf\n\n", temp++, ans);
    }
    return 0;
}
```

## 练习

- [「HDU1542」Atlantis](http://acm.hdu.edu.cn/showproblem.php?pid=1542)

- [「HDU1828」Picture](http://acm.hdu.edu.cn/showproblem.php?pid=1828)

- [「HDU3265」Posters](http://acm.hdu.edu.cn/showproblem.php?pid=3265)

## 参考资料

- <https://www.cnblogs.com/yangsongyi/p/8378629.html>

- <https://blog.csdn.net/riba2534/article/details/76851233>

- <https://blog.csdn.net/winddreams/article/details/38495093>

## 习题

> [!NOTE] **[AcWing 1406. 窗口面积](https://www.acwing.com/problem/content/1408/)**
> 
> 题意: 扫描线求矩形面积的并

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

using PII = pair<int, int>;
#define x first
#define y second

struct Rect {
    char c;
    PII a, b;
};
list<Rect> rect;

// 求合并覆盖后的长度
int get_intersection(int a, int b, int c, int d) {
    if (b <= c || d <= a) return 0;
    return min(b, d) - max(a, c);
}

double get_area(char c) {
    // 保存当前矩形上方的所有矩形有哪些
    vector<Rect> cur;
    for (auto r : rect)
        if (r.c == c || cur.size())
            cur.push_back(r);
    // xs保存所有竖线
    vector<int> xs;
    for (int i = 0; i < cur.size(); ++ i ) {
        auto & r = cur[i];
        // 左右边界 是否在矩形内
        if (r.a.x >= cur[0].a.x && r.a.x <= cur[0].b.x)
            xs.push_back(r.a.x);
        if (r.b.x >= cur[0].a.x && r.b.x <= cur[0].b.x)
            xs.push_back(r.b.x);
    }
    sort(xs.begin(), xs.end());
    // 算面积
    int res = 0;
    for (int i = 0; i + 1 < xs.size(); ++ i )
        // 没有重合，其实可以unique一下
        if (xs[i] != xs[i + 1]) {
            // 区域左右x边界
            int a = xs[i], b = xs[i + 1];
            // 保存矩形和本次计算的区域是否有交集
            vector<PII> q;
            for (int j = 1; j < cur.size(); ++ j ) {
                auto & r = cur[j];
                if (r.a.x <= a && r.b.x >= b)
                    q.push_back({r.a.y, r.b.y});
            }
            if (q.size()) {
                // 此时q保存的y 即所有本区域内的横线
                // 合并计数其长度 同时计算面积
                sort(q.begin(), q.end());
                int st = q[0].x, ed = q[0].y;
                for (int j = 1; j < q.size(); ++ j )
                    if (q[j].x <= ed) ed = max(ed, q[j].y);
                    else {
                        // st ed 要和当前区域求交集
                        res += get_intersection(st, ed, cur[0].a.y, cur[0].b.y) * (b - a);
                        st = q[j].x, ed = q[j].y;
                    }
                res += get_intersection(st, ed, cur[0].a.y, cur[0].b.y) * (b - a);
            }
        }
    return (1 - (double)res / (cur[0].b.x - cur[0].a.x) / (cur[0].b.y - cur[0].a.y)) * 100;
}

int main() {
    char op;
    while (cin >> op) {
        if (op == 'w') {
            char c;
            int x1, y1, x2, y2;
            scanf("(%c,%d,%d,%d,%d)", &c, &x1, &y1, &x2, &y2);
            rect.push_back({c, {min(x1, x2), min(y1, y2)}, {max(x1, x2), max(y1, y2)}});
        } else {
            char c;
            scanf("(%c)", &c);
            // 找到当前窗口
            list<Rect>::iterator it;
            for (auto i = rect.begin(); i != rect.end(); ++ i )
                if (i->c == c) {
                    it = i;
                    break;
                }
            if (op == 't') rect.push_back(*it), rect.erase(it);
            else if (op == 'b') rect.push_front(*it), rect.erase(it);
            else if (op == 'd') rect.erase(it);
            else printf("%.3lf\n", get_area(c));
        }
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

> [!NOTE] **[AcWing 262. 海报](https://www.acwing.com/problem/content/264/)**
> 
> 题意: 扫描线求矩形周长

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

using PII = pair<int, int>;
#define x first
#define y second

const int N = 5010;

int n;
struct Rect {
    PII a, b;
}rect[N];

int get_range_len(int a, int b) {
    vector<PII> q;
    for (int i = 0; i < n; ++ i ) {
        auto & r = rect[i];
        if (r.a.x <= a && r.b.x >= b)
            q.push_back({r.a.y, r.b.y});
    }
    if (q.empty()) return 0;
    
    sort(q.begin(), q.end());
    int res = 0, st = q[0].x, ed = q[0].y;
    for (int i = 1; i < q.size(); ++ i )
        if (q[i].x <= ed) ed = max(ed, q[i].y);
        else {
            res += (b - a) * 2;
            st = q[i].x, ed = q[i].y;
        }
    res += (b - a) * 2;
    return res;
}

int get_len() {
    vector<int> xs;
    for (int i = 0; i < n; ++ i ) {
        xs.push_back(rect[i].a.x);
        xs.push_back(rect[i].b.x);
    }
    sort(xs.begin(), xs.end());
    int res = 0;
    for (int i = 0; i + 1 < xs.size(); ++ i )
        if (xs[i] != xs[i + 1])
            res += get_range_len(xs[i], xs[i + 1]);
    return res;
}

int main() {
    cin >> n;
    for (int i = 0; i < n; ++ i ) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        rect[i] = {{x1, y1}, {x2, y2}};
    }
    // 分别算水平和垂直方向的周长
    int res = get_len();
    for (int i = 0; i < n; ++ i ) {
        swap(rect[i].a.x, rect[i].a.y);
        swap(rect[i].b.x, rect[i].b.y);
    }
    cout << res + get_len() << endl;
    
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

> [!NOTE] **[Luogu [USACO12FEB]Overplanting S]()**
> 
> 题意: 扫描线求矩形交

> [!TIP] **思路**
> 
> 扫描线十分特殊，推导可知无需pushdown

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 1010;

int n;
struct Seg {
    int x, y1, y2;
    int k;
    bool operator< (const Seg & t) const {
        return x < t.x;
    }
} seg[N << 1];
struct Node {
    int l, r;
    int cnt, len;
} tr[N << 3];

vector<int> ys;

int find(int y) {
    return lower_bound(ys.begin(), ys.end(), y) - ys.begin();
}

void pushup(int u) {
    if (tr[u].cnt)
        tr[u].len = ys[tr[u].r + 1] - ys[tr[u].l];
    else if (tr[u].l == tr[u].r)
        tr[u].len = 0;
    else
        tr[u].len = tr[u << 1].len + tr[u << 1 | 1].len;
}

void build(int u, int l, int r) {
    if (l == r)
        tr[u] = {l, r, 0, 0};
    else {
        tr[u] = {l, r, 0, 0};
        int m = l + (r - l) / 2;
        build(u << 1, l, m), build(u << 1 | 1, m + 1, r);
    }
}

void modify(int u, int l, int r, int k) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].cnt += k;
        pushup(u);
    } else {
        int m = tr[u].l + (tr[u].r - tr[u].l) / 2;
        if (l <= m)
            modify(u << 1, l, r, k);
        if (r > m)
            modify(u << 1 | 1, l, r, k);
        pushup(u);
    }
}

int main() {
    cin >> n;
    for (int i = 0, j = 0; i < n; ++ i ) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        // seg[j ++ ] = {x1, y1, y2, 1};
        // seg[j ++ ] = {x2, y1, y2, -1};
        seg[j ++ ] = {x1, y2, y1, 1};
        seg[j ++ ] = {x2, y2, y1, -1};
        ys.push_back(y1), ys.push_back(y2);
    }
    
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());
    
    // 保存的区间比size-1还要小1
    build(1, 0, ys.size() - 2);
    sort(seg, seg + 2 * n);
    
    LL res = 0;
    for (int i = 0; i < 2 * n; ++ i ) {
        if (i)
            res += (LL)tr[1].len * (seg[i].x - seg[i - 1].x);
        modify(1, find(seg[i].y1), find(seg[i].y2) - 1, seg[i].k);
    }
    
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

> [!NOTE] **[LeetCode 850. 矩形面积 II](https://leetcode.cn/problems/rectangle-area-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 原来想着二维离散化 + 差分，实际上只离散化一维就够了
> 
> **标准扫描线思想**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    using PII = pair<int, int>;
    const static int MOD = 1e9 + 7;
    
    vector<vector<int>> recs;

    LL get(int l, int r) {
        vector<PII> t;
        for (auto & rec : recs)
            if (rec[0] <= l && rec[2] >= r)
                t.push_back({rec[1], rec[3]});  // 能够占据 [l,r] 完整的一段
        
        sort(t.begin(), t.end());
        LL ret = 0, st = -1, ed = -1;
        for (auto [d, u] : t)
            if (d > ed) {
                ret += ed - st;
                st = d, ed = u;
            } else
                ed = max(ed, (LL)u);
        ret += ed - st;
        return ret * (r - l);
    }

    int rectangleArea(vector<vector<int>>& rectangles) {
        this->recs = rectangles;

        vector<int> xs; // 原来想着二维离散化 + 差分，实际上只离散化一维就够了
        for (auto & rec : recs)
            xs.push_back(rec[0]), xs.push_back(rec[2]);
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());

        LL res = 0;
        for (int i = 1; i < xs.size(); ++ i )
            res += get(xs[i - 1], xs[i]);
        return res % MOD;
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

> [!NOTE] **[LeetCode LCCUP-2023-Spring 3. 最强祝福力场](https://leetcode.cn/contest/season/2023-spring/problems/xepqZ5/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准做法显然是扫描线 => TODO
> 
> 数据范围比较小（矩形数量很少）显然可以离散化之后直接二维前缀和求解

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 扫描线 TODO**

```cpp

```

##### **C++ 离散化二维前缀和**

```cpp
class Solution {
public:
    // 矩形数量不超过 100 => 都可以暴力做了
    // 考虑：如果 side 长度为奇数怎么半？直接全部 * 2
    using LL = long long;
    const static int N = 1010;
    
    int find(vector<LL> & s, LL x) {
        return lower_bound(s.begin(), s.end(), x) - s.begin() + 1;
    }
    
    int g[N][N];
    
    int fieldOfGreatestBlessing(vector<vector<int>>& forceField) {
        vector<LL> xs, ys;
        for (auto & f : forceField) {
            LL x = 2ll * f[0], y = 2ll * f[1], w = 2ll * f[2];
            xs.push_back(x - w / 2), xs.push_back(x + w / 2);
            ys.push_back(y - w / 2), ys.push_back(y + w / 2);
        }
        sort(xs.begin(), xs.end()); xs.erase(unique(xs.begin(), xs.end()), xs.end());
        sort(ys.begin(), ys.end()); ys.erase(unique(ys.begin(), ys.end()), ys.end());
        
        memset(g, 0, sizeof g);
        for (auto & f : forceField) {
            LL x = 2ll * f[0], y = 2ll * f[1], w = 2ll * f[2];
            LL u = find(xs, x - w / 2), d = find(xs, x + w / 2);
            LL l = find(ys, y - w / 2), r = find(ys, y + w / 2);
            g[u][l] ++ , g[u][r + 1] -- , g[d + 1][l] -- , g[d + 1][r + 1] ++ ;
        }
        
        int n = xs.size(), m = ys.size();
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j )
                g[i][j] += g[i][j - 1] + g[i - 1][j] - g[i - 1][j - 1];
        
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j )
                res = max(res, g[i][j]);
        return res;
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