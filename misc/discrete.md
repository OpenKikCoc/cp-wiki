> [!NOTE] **离散化后坐标映射时应使用 lower_bound**
> 
> 经验来看，直接用 lower_bound 比建立 unordered_map 再查找更快
## 简介

离散化本质上可以看成是一种 [哈希](string/hash.md)，其保证数据在哈希以后仍然保持原来的全/偏序关系。

通俗地讲就是当有些数据因为本身很大或者类型不支持，自身无法作为数组的下标来方便地处理，而影响最终结果的只有元素之间的相对大小关系时，我们可以将原来的数据按照从大到小编号来处理问题，即离散化。

用来离散化的可以是大整数、浮点数、字符串等等。

## 实现

C++ 离散化有现成的 STL 算法：

### 离散化数组

将一个数组离散化，并进行查询是比较常用的应用场景：

```cpp
// a[i] 为初始数组,下标范围为 [1, n]
// len 为离散化后数组的有效长度
std::sort(a + 1, a + 1 + n);

len = std::unique(a + 1, a + n + 1) - a - 1;
// 离散化整个数组的同时求出离散化后本质不同数的个数。
```

在完成上述离散化之后可以使用 `std::lower_bound` 函数查找离散化之后的排名（即新编号）：

```cpp
std::lower_bound(a + 1, a + len + 1, x) - a;  // 查询 x 离散化后对应的编号
```

同样地，我们也可以对 `vector` 进行离散化：

```cpp
// std::vector<int> a, b; // b 是 a 的一个副本
std::sort(a.begin(), a.end());
a.erase(std::unique(a.begin(), a.end()), a.end());
for (int i = 0; i < n; ++i)
    b[i] = std::lower_bound(a.begin(), a.end(), b[i]) - a.begin();
```

## 习题

> [!NOTE] **[AcWing 802. 区间和](https://www.acwing.com/problem/content/804/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc**

分析一下y总的代码。

主要分为5大步： 

1. 读输入。将每次读入的 x c push_back()到 add 中，将每次读入的位置 x push_back() 到 alls 中，将每次读入的 l r push_back() 到 query 中。 

2. 排序、去重。 
   
3. 通过遍历 add ，完成在离散化的数组映射到的 a 数组中进行加上 c 的操作（用到 find 函数）。 
   
4. 初始化 s 数组。 5.通过遍历 query ，完成求区间 [l,r] 的和。

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef pair<int, int> PII;

const int N = 300010;

int n, m;
int a[N], s[N];

vector<int> alls;
vector<PII> add, query;

int find(int x) {
    int l = 0, r = alls.size() - 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1;
}

vector<int>::iterator unique(vector<int> &a) {
    int j = 0;
    for (int i = 0; i < a.size(); i ++ )
        if (!i || a[i] != a[i - 1])
            a[j ++ ] = a[i];
    // a[0] ~ a[j - 1] 所有a中不重复的数

    return a.begin() + j;
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; i ++ ) {
        int x, c;
        cin >> x >> c;
        add.push_back({x, c});

        alls.push_back(x);
    }

    for (int i = 0; i < m; i ++ ) {
        int l, r;
        cin >> l >> r;
        query.push_back({l, r});

        alls.push_back(l);
        alls.push_back(r);
    }

    // 去重
    sort(alls.begin(), alls.end());
    alls.erase(unique(alls), alls.end());

    // 处理插入
    for (auto item : add) {
        int x = find(item.first);
        a[x] += item.second;
    }

    // 预处理前缀和
    for (int i = 1; i <= alls.size(); i ++ ) s[i] = s[i - 1] + a[i];

    // 处理询问
    for (auto item : query) {
        int l = find(item.first), r = find(item.second);
        cout << s[r] - s[l - 1] << endl;
    }

    return 0;
}
```

##### **C++**

官方题解中：

> find 函数的功能，输入一个离散数组的位置（映射前的位置）x ：返回连续数组的位置 +1 （映射后的位置 +1 ）。 
> 
> +1 的目的是为了求区间和时少一步下标为 0 的判断

这个说法站不住脚，其实求区间和时可以直接让 find 返 回 l ，求区间和时使用 sum[i-1] + arr[i-1] 。

原题返回 +1 本质是使得 find 找到的下标范围0 ~ n-1 与前缀和的范围 1 ~ n 保持一致。其实在输出的时候求 sum[r+1]-sum[l] 同样能够实现这样的效果，而且更易理解。

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, m, x, c, l, r;
int arr[300005], sum[300005];
vector<int> alls;  // 所有坐标 用于离散化
vector<pair<int, int>> add, query;

int find(int x) {
    int l = 0, r = alls.size() - 1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (alls[m] < x)
            l = m + 1;
        else
            r = m;
    }
    return l;
}

/*
vector<int>::iterator unique(vector<int> &a) {
    int j = 0;
    for (int i = 0; i < a.size(); ++ i )
        if (!i || a[i] != a[i - 1]) a[j ++ ] = a[i];
    return a.begin() + j;
}
*/

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 0; i < n; ++i) {
        scanf("%d%d", &x, &c);
        add.push_back({x, c});
        alls.push_back(x);
    }
    for (int i = 0; i < m; ++i) {
        scanf("%d%d", &l, &r);
        query.push_back({l, r});
        alls.push_back(l);
        alls.push_back(r);
    }
    // 排序去重
    sort(alls.begin(), alls.end());
    // c++内置函数
    alls.erase(unique(alls.begin(), alls.end()), alls.end());
    // alls.erase(unique(alls), alls.end());
    // 处理插入
    for (auto p : add) {
        int x = find(p.first);
        arr[x] += p.second;
    }
    // 预处理前缀和
    for (int i = 1; i <= alls.size(); ++i) sum[i] = sum[i - 1] + arr[i - 1];
    // 处理询问
    for (auto p : query) {
        int l = find(p.first), r = find(p.second);
        printf("%d\n", sum[r + 1] - sum[l]);
    }
}

/*
0 1 2 3 4 5 6
x 1 3 4 6 7 8
  2 6 0 0 5 0
0 2 8 8 8 13 13
查 1,3 下标 1,2 但是使用同一个find函数返回的是 0 1
*/
```

##### **Python**

```python
# 整数的离散化
# 比如：一个a有序序列的值域很大，但是数的个数很少；这个时候不能开一个非常大的数组（会超时），这个时候就需要把值域映射到从0开始自然数
# 离散化有两个问题：1）a数组中可能有重复元素，所以需要去重 ===> 在c++中推荐用库函数来完成：								 
#                     	 sort(alls.begin(), alls.end());
#                     	 alls.erase(unique(alls), alls.end());
#               2）需要可以快速映射，如何算出a里的每个值映射后的值是多少；找一个x在a中的下标是多少 ===> 由于是有序的，用二分来找。
#												 find(x) //找到第一个大于等于x的位置
def find(x):
    """二分查找模板，从索引数组alls中找到大于等于x的最小的索引"""
    l = 0
    r = len(alls) - 1
    while l < r:
        mid = l + r >> 1
        if alls[mid] >= x:
            r = mid  # ！！！if条件忘记了=号
        else:
            l = mid + 1
    return l + 1  # 因为要计算前缀和，所以加1保证索引从1开始


if __name__ == "__main__":
    n, m = map(int, input().split())
    N = 300010
    a = [0] * N  # 用于存储离散化后的索引和对应值，其中索引对应离散化后的索引，值对应离散化前索引的取值
    s = [0] * N  # 存a数组的前缀和数组

    add = []  # 存储插入操作的二元组
    query = []  # 存储查询操作的二元组

    alls = []  # 存储离散化前输入的所有索引，n+2*m

    for i in range(n):
        x, c = map(int, input().split())
        add.append((x, c))
        alls.append(x)

    for i in range(m):
        l, r = map(int, input().split())
        query.append((l, r))
        alls.append(l)
        alls.append(r)

    alls = list(set(sorted(alls)))  # 将alls数组排序并去重

    # 1. 处理插入
    for x, c in add:
        x2 = find(x)
        a[x2] += c

    # 2. 处理前缀和
    for i in range(1, len(alls) + 1):
        s[i] = s[i - 1] + a[i]

    # 3. 处理查询
    for l, r in query:
        l2 = find(l)
        r2 = find(r)
        res = s[r2] - s[l2 - 1]
        print(res)

# Acwing 759---橙子染色 也是一个离散化的问题。

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2013. 检测正方形](https://leetcode.cn/problems/detect-squares/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数值范围较大 所以对其中一维离散化 在询问时查询离散化结果即可
> 
> > 时间复杂度要求不高 甚至一些暴力遍历之前所有点的代码也能AC

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class DetectSquares {
public:
    const static int N = 1010;
    
    int g[N][N];
    set<int> xs[N];
    vector<int> dir = {-1, 1};
    
    int get(int a, int b, int c) {
        return a * b * c;
    }
    
    bool check(int x, int y) {
        // can be zero
        return x >= 0 && x < N && y >= 0 && y < N;
    }
    
    DetectSquares() {
        memset(g, 0, sizeof g);
    }
    
    void add(vector<int> point) {
        int x = point[0], y = point[1];
        g[x][y] ++ ;
        xs[x].insert(y);    // we donot care whether it's duplicated
    }
    
    int count(vector<int> point) {
        int res = 0;
        int x1 = point[0], y1 = point[1];
        for (auto y2 : xs[x1]) 
            if (y2 != y1) {
                int d = y2 - y1, x2 = x1;
                for (auto f : dir) {
                    int x3 = x1 + f * d, y3 = y1;
                    int x4 = x1 + f * d, y4 = y2;
                    if (check(x2, y2) && check(x3, y3) && check(x4, y4))
                        res += get(g[x2][y2], g[x3][y3], g[x4][y4]);
                }
            }
        return res;
    }
};

/**
 * Your DetectSquares object will be instantiated and called as such:
 * DetectSquares* obj = new DetectSquares();
 * obj->add(point);
 * int param_2 = obj->count(point);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces Almost Arithmetical Progression](http://codeforces.com/problemset/problem/255/C)**
> 
> 题意: 
> 
> 隔位的两个数相等 问最长子序列长度

> [!TIP] **思路**
> 
> 离散化 否则MLE

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Almost Arithmetical Progression
// Contest: Codeforces - Codeforces Round #156 (Div. 2)
// URL: https://codeforces.com/problemset/problem/255/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 4e3 + 10;

int n, a[N];
// unordered_map<int, int> f[N]; // MLE
int f[N][N + N];

int idx = 0;
unordered_map<int, int> h;
int get(int x) {
    if (h.count(x))
        return h[x];
    return h[x] = ++idx;
}

int main() {
    cin >> n;
    int res = 0;
    for (int i = 1; i <= n; ++i) {
        cin >> a[i];
        a[i] = get(a[i]);
        for (int j = 1; j < i; ++j) {
            int d = a[i] - a[j];
            f[i][d + N] = max(f[i][d + N], f[j][-d + N] + 1);
            res = max(res, f[i][d + N] + 1);
        }
    }
    res = max(res, min(2, n));
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

> [!NOTE] **[Codeforces King's Path](http://codeforces.com/problemset/problem/242/C)**
> 
> 题意: 
> 
> 大坐标系下的 bfs 最短路

> [!TIP] **思路**
> 
> 数值大而数据范围不大，离散化即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. King's Path
// Contest: Codeforces - Codeforces Round #149 (Div. 2)
// URL: https://codeforces.com/problemset/problem/242/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PII = pair<int, int>;

int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0}, dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

int sx, sy, tx, ty;
int m;
unordered_map<int, int> r2idx;
vector<vector<PII>> segs;
unordered_map<LL, int> dis;

LL encode(LL x, LL y) { return (x << 32) | y; }

bool check(int x, int y) {
    if (r2idx.find(x) == r2idx.end())
        return false;
    auto& seg = segs[r2idx[x]];
    for (auto& [l, r] : seg)
        if (y >= l && y <= r)
            return true;
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> sx >> sy >> tx >> ty;

    cin >> m;
    while (m--) {
        int r, a, b;
        cin >> r >> a >> b;
        if (r2idx.find(r) != r2idx.end())
            segs[r2idx[r]].push_back({a, b});
        else {
            r2idx[r] = segs.size();
            segs.push_back({{a, b}});
        }
    }
    for (auto& seg : segs)
        sort(seg.begin(), seg.end());

    if (check(sx, sy)) {
        queue<PII> q;
        q.push({sx, sy});
        dis[encode(sx, sy)] = 0;
        while (!q.empty()) {
            auto& [x, y] = q.front();
            q.pop();
            int d = dis[encode(x, y)];
            if (x == tx && y == ty) {
                cout << d << endl;
                return 0;
            }
            for (int i = 0; i < 8; ++i) {
                int nx = x + dx[i], ny = y + dy[i];
                if (check(nx, ny) && dis.find(encode(nx, ny)) == dis.end()) {
                    dis[encode(nx, ny)] = d + 1;
                    q.push({nx, ny});
                }
            }
        }
    }

    cout << -1 << endl;
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
