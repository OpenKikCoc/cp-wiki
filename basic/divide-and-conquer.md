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

> [!NOTE] **[LeetCode 454. 四数相加 II](https://leetcode.cn/problems/4sum-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 典型分治

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> cnt;
        for (auto c : C)
            for (auto d : D)
                ++ cnt[c + d];
        int res = 0;
        for (auto a : A)
            for (auto b : B)
                res += cnt[- (a + b)];
        return res;
    }
};
```

##### **Python**

```python
# 枚举A，再枚举B，然后 找到c和d满足条件的数值；（根据数据范围，这道题只能遍历两次，所以用空间换时间 哈希表来处理）
# 用哈希表存储每种和 有多少种组合；（前两个枚举， 后两个就可以直接在哈希表里查找）

class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        import collections
        my_dict = collections.defaultdict(int)
        for c in nums3:
            for d in nums4:
                my_dict[c + d] += 1
        res = 0
        for a in nums1:
            for b in nums2:
                res += my_dict[-(a + b)]
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 779. 第K个语法符号](https://leetcode.cn/problems/k-th-symbol-in-grammar/)**
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
    // 新的左侧和原一致 右侧和原取反
    int kthGrammar(int N, int K) {
        K -- ;
        int res = 0;
        while (K)
            res ^= K & 1, K >>= 1;
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

> [!NOTE] **[LeetCode 1274. 矩形内船只的数目](https://leetcode.cn/problems/number-of-ships-in-a-rectangle/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 二维 分治+四分查找
>
> > 对于当前查找的矩形区域，如果 API 返回 False ，我们得到区域内没有船只，可以舍弃该区域。
> >
> > 但如果 API 返回 True，我们得到区域内有船只，为了确定船只的位置，我们需要将区域划分成若干个互不相交的小区域，分别调用 API 继续进行查找。
> >
> > 直到某一次查找的区域为一个点，即满足 topRight == bottomLeft 时，如果 API 返回 True，我们就确定了一艘船只的位置，可以将计数器增加 1。
> >
> > 在查找完成后，计数器中的值即为船只的数目。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countShips(Sea sea, vector<int> topRight, vector<int> bottomLeft) {
        int x1 = topRight[0], y1 = topRight[1], x2 = bottomLeft[0],
            y2 = bottomLeft[1];
        if (x1 < x2 || y1 < y2 || !sea.hasShips(topRight, bottomLeft)) return 0;
        if (x1 == x2 && y1 == y2) return 1;
        int mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        return countShips(sea, {mx, my}, {x2, y2}) +
               countShips(sea, {mx, y1}, {x2, my + 1}) +
               countShips(sea, {x1, my}, {mx + 1, y2}) +
               countShips(sea, {x1, y1}, {mx + 1, my + 1});
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

> [!NOTE] **[Codeforces C. Painting Fence](https://codeforces.com/problemset/problem/448/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常好的贪心分治题目 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Painting Fence
// Contest: Codeforces - Codeforces Round #256 (Div. 2)
// URL: https://codeforces.com/problemset/problem/448/C
// Memory Limit: 512 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 思维 非常非常好的分治题目 重复做

using LL = long long;
const int N = 100010;

int n;
int h[N];

int paint(int s, int t) {
    if (s > t)
        return 0;

    // 先找最低的可以横向涂的
    int minv = 1e9, c = 0;
    for (int i = s; i <= t; ++i)
        minv = min(h[i], minv);
    // 累加操作数 更新高度
    c += minv;
    for (int i = s; i <= t; ++i)
        h[i] -= minv;

    // 分治
    int ns = s;
    for (int i = s; i <= t; ++i)
        if (i == t && h[i])
            c += paint(ns, i), ns = i + 1;
        else if (h[i] == 0)
            c += paint(ns, i - 1), ns = i + 1;

    // 与竖着涂对比 取最小值
    return min(t - s + 1, c);
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; i++)
        cin >> h[i];

    cout << paint(1, n) << endl;

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
