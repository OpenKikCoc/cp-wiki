算法（英语：greedy algorithm），是用计算机来模拟一个“”的人做出决策的过程。这个人十分贪婪，每一步行动总是按某种指标选取最优的操作。而且他目光短浅，总是只看眼前，并不考虑以后可能造成的影响。

## 证明方法

算法有两种证明方法：反证法和归纳法。一般情况下，一道题只会用到其中的一种方法来证明。

1. 反证法：如果交换方案中任意两个元素/相邻的两个元素后，答案不会变得更好，那么可以推定目前的解已经是最优解了。
2. 归纳法：先算得出边界情况（例如 $n = 1$）的最优解 $F_1$，然后再证明：对于每个 $n$，$F_{n+1}$ 都可以由 $F_{n}$ 推导出结果。

## 要点

### 常见题型

在提高组难度以下的题目中，最常见的有两种。

- 「我们将 XXX 按照某某顺序排序，然后按某种顺序（例如从小到大）选择。」。
- 「我们每次都取 XXX 中最大/小的东西，并更新 XXX。」（有时「XXX 中最大/小的东西」可以优化，比如用优先队列维护）

二者的区别在于一种是离线的，先处理后选择；一种是在线的，边处理边选择。

### 排序解法

用排序法常见的情况是输入一个包含几个（一般一到两个）权值的数组，通过排序然后遍历模拟计算的方法求出最优值。

### 后悔解法

思路是无论当前的选项是否最优都接受，然后进行比较，如果选择之后不是最优了，则反悔，舍弃掉这个选项；否则，正式接受。如此往复。

## 例题

### 邻项交换法的例题

> [!NOTE] **[NOIP 2012 国王游戏](https://vijos.org/p/1779)**
> 
> 恰逢 H 国国庆，国王邀请 n 位大臣来玩一个有奖游戏。首先，他让每个大臣在左、右手上面分别写下一个整数，国王自己也在左、右手上各写一个整数。然后，让这 n 位大臣排成一排，国王站在队伍的最前面。排好队后，所有的大臣都会获得国王奖赏的若干金币，每位大臣获得的金币数分别是：排在该大臣前面的所有人的左手上的数的乘积除以他自己右手上的数，然后向下取整得到的结果。
>
> 国王不希望某一个大臣获得特别多的奖赏，所以他想请你帮他重新安排一下队伍的顺序，使得获得奖赏最多的大臣，所获奖赏尽可能的少。注意，国王的位置始终在队伍的最前面。

> [!TIP] **解题思路**
> 
> 设排序后第 $i$ 个大臣左右手上的数分别为 $a_i, b_i$。考虑通过邻项交换法推导策略。
> 
> 用 $s$ 表示第 $i$ 个大臣前面所有人的 $a_i$ 的乘积，那么第 $i$ 个大臣得到的奖赏就是 $\dfrac{s} {b_i}$，第 $i + 1$ 个大臣得到的奖赏就是 $\dfrac{s \cdot a_i} {b_{i+1}}$。
>
> 如果我们交换第 $i$ 个大臣与第 $i + 1$ 个大臣，那么此时的第 $i$ 个大臣得到的奖赏就是 $\dfrac{s} {b_{i+1}}$，第 $i + 1$ 个大臣得到的奖励就是 $\dfrac{s \cdot a_{i+1}} {b_i}$。
>
> 如果交换前更优当且仅当
>
>   $$
>   \max \left(\dfrac{s} {b_i}, \dfrac{s \cdot a_i} {b_{i+1}}\right)  < \max \left(\dfrac{s} {b_{i+1}}, \dfrac{s \cdot a_{i+1}} {b_i}\right)
>   $$
> 
>   提取出相同的 $s$ 并约分得到
>
>   $$
>   \max \left(\dfrac{1} {b_i}, \dfrac{a_i} {b_{i+1}}\right)  < \max \left(\dfrac{1} {b_{i+1}}, \dfrac{a_{i+1}} {b_i}\right)
>   $$
> 
>   然后分式化成整式得到
> 
>   $$
>   \max (b_{i+1}, a_i\cdot b_i)  < \max (b_i, a_{i+1}\cdot b_{i+1})
>   $$
> 
>   实现的时候我们将输入的两个数用一个结构体来保存并重载运算符：
    

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>


### 后悔法的例题

> [!NOTE] **[「USACO09OPEN」工作调度 Work Scheduling](https://www.luogu.com.cn/problem/P2949)**
> 
> 约翰的工作日从 $0$ 时刻开始，有 $10^9$ 个单位时间。在任一单位时间，他都可以选择编号 $1$ 到 $N$ 的 $N(1 \leq N \leq 10^5)$ 项工作中的任意一项工作来完成。工作 $i$ 的截止时间是 $D_i(1 \leq D_i \leq 10^9)$，完成后获利是 $P_i( 1\leq P_i\leq 10^9 )$。在给定的工作利润和截止时间下，求约翰能够获得的利润最大为多少。

> [!TIP] **解题思路**
> 
> 1. 先假设每一项工作都做，将各项工作按截止时间排序后入队；
> 
> 2.  在判断第 i 项工作做与不做时，若其截至时间符合条件，则将其与队中报酬最小的元素比较，若第 i 项工作报酬较高（后悔），则 `ans += a[i].p - q.top()`。  
> 
> 用优先队列（小根堆）来维护队首元素最小。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

## 习题

- [P1209\[USACO1.3\]修理牛棚 Barn Repair - 洛谷](https://www.luogu.com.cn/problem/P1209)
- [P2123 皇后游戏 - 洛谷](https://www.luogu.com.cn/problem/P2123)
- [LeetCode 上标签为算法的题目](https://leetcode-cn.com/tag/greedy/)

### 基本贪心

> [!NOTE] **[AcWing 1348. 搭配牛奶](https://www.acwing.com/problem/content/1350/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单学习下 struct 比较级函数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5010;

int n, m;
struct Milk {
    int p, a;
    bool operator< (const Milk& t) const {
        return p < t.p;
    }
}milk[N];

int main() {
    cin >> n >> m;
    for (int i = 0; i < m; ++ i ) cin >> milk[i].p >> milk[i].a;
    sort(milk, milk + m);
    
    int res = 0;
    for (int i = 0; i < m && n; ++ i ) {
        int add = min(n, milk[i].a);
        n -= add;
        res += add * milk[i].p;
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

> [!NOTE] **[AcWing 1349. 修理牛棚](https://www.acwing.com/problem/content/1351/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 补集的思想 先把0～c-1所有的用一个木板覆盖 随后断开

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 210;

int m, s, c;
int a[N], b[N]; // V 表示间隙

int main() {
    cin >> m >> s >> c;
    for (int i = 0; i < c; ++ i ) cin >> a[i];
    sort(a, a + c);
    
    int res = a[c - 1] - a[0] + 1;
    
    for (int i = 1; i < c; ++ i ) b[i] = a[i] - a[i - 1] - 1;
    sort(b + 1, b + c, greater<int>());
    
    // 至多断开m - 1次
    for (int i = 1; i <= m - 1 && i < c; ++ i )
        res -= b[i];
    
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

> [!NOTE] **[Luogu [NOIP2018 提高组] 铺设道路](https://www.luogu.com.cn/problem/P5019)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典贪心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const int N = 1e6 + 10;

int n;
LL d[N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> d[i];
    
    LL res = 0;
    stack<int> st;
    // i = 0 -----> d[i] = 0
    for (int i = 1; i <= n; ++ i )
        if (d[i] > d[i - 1])
            res += d[i] - d[i - 1];
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

> [!NOTE] **[Luogu [USACO07MAR]Face The Right Way G](https://www.luogu.com.cn/problem/P2882)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 反转问题/开关问题
> 
> 贪心 动态维护

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5010;

int n;
int d[N], f[N];

int get(int k) {
    memset(f, 0, sizeof f);
    
    // s 维护会影响到i的点击数
    int s = 0, res = 0;
    for (int i = 0; i + k <= n; ++ i ) {
        // 第i头牛 + 之前能影响到它的点击数
        // &1 说明当前朝后 需要点击
        if ((d[i] + s) & 1)
            f[i] = 1, res ++ ;
        
        s += f[i];
        if (i - k + 1 >= 0)
            s -= f[i - k + 1];
    }
    
    // 检查最后一段
    for (int i = n - k + 1; i < n; ++ i ) {
        if ((d[i] + s) & 1)
            return -1;
        if (i - k + 1 >= 0)
            s -= f[i - k + 1];
    }
    return res;
}

int main() {
    cin >> n;
    
    for (int i = 0; i < n; ++ i ) {
        char c;
        cin >> c;
        if (c == 'F')
            d[i] = 0;
        else
            d[i] = 1;
    }
    
    int K = 1, M = n;
    for (int k = 1; k <= n; ++ k ) {
        int m = get(k);
        if (m >= 0 && m < M)
            M = m, K = k;
    }
    cout << K << ' ' << M << endl;
    
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


### 排序不等式

> [!NOTE] **[AcWing 1395. 产品处理](https://www.acwing.com/problem/content/1397/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **思维题**
> 
> Q1 
> 
> Q2 需要排序不等式

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1010;

int n, m1, m2;
int a[N], b[N];
struct Node {
    int finish_time, cost;
    bool operator< (const Node & t) const {
        return finish_time > t.finish_time;
    }
};

void work(int m, int f[]) {
    priority_queue<Node> heap;
    for (int i = 0; i < m; ++ i ) {
        int cost;
        cin >> cost;
        // 加入的是 【洗完下一件衣服最早结束的时间,单个时间消耗】
        heap.push({cost, cost});
    }
    
    for (int i = 1; i <= n; ++ i ) {
        auto t = heap.top(); heap.pop();
        f[i] = t.finish_time;
        t.finish_time += t.cost;
        heap.push(t);
    }
}

int main() {
    cin >> n >> m1 >> m2;
    work(m1, a);
    work(m2, b);
    
    // attention 互补这样
    // ------  ---
    // ----  -----
    // -- --------
    // - ---------
    int res = 0;
    for (int i = 1, j = n; i <= n; ++ i , -- j )
        res = max(res, a[i] + b[j]);
    cout << a[n] << ' ' << res << endl;
    
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

> [!NOTE] **[AcWing 913. 排队打水](https://www.acwing.com/problem/content/description/915/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 排序不等式
> 
> 显然耗时最短的越先打水

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, x;

int main() {
    cin >> n;
    vector<int> ve;
    for (int i = 0; i < n; ++i) {
        cin >> x;
        ve.push_back(x);
    }
    sort(ve.begin(), ve.end());
    long long res = 0;
    for (auto x : ve) { res += x * (--n); }
    cout << res << endl;
}
```

##### **Python**

```python
# 直觉就是：让权重小的人排在前面；最优解是：从小到大排序，那么总时间最小。
# 证明：(用调整法/反证法证明)
# 假设最优解不是从小到大排序，那会存在相邻两个点是逆序的；交换两个位置，交换前的值 > 交换后的值。说明交换后总时间会降低。那就和假设矛盾，所以最优解一定是从小到大的排序。

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))

    arr.sort()
    res = 0
    for i in range(n):
        res += arr[i] * (n - i - 1)
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *


### 绝对值不等式

> [!NOTE] **[AcWing 104. 货仓选址](https://www.acwing.com/problem/content/106/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 绝对值不等式
> 
> 经典 选择中间的两个点之间的任意位置

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, x;

int main() {
    cin >> n;
    vector<int> ve;
    for (int i = 0; i < n; ++i) {
        cin >> x;
        ve.push_back(x);
    }
    sort(ve.begin(), ve.end());
    // n为奇数 中间为 3/2=1 (0,1,2)  n为偶数 中间为 4/2=2 (0,1,2,3)
    int res = 0;
    for (auto v : ve) res += abs(v - ve[n / 2]);
    cout << res << endl;
}
```

##### **Python**

```python
if __name__ == '__main__':
    n = int(input())
    nums = list(map(int, input().split()))
    # 踩坑：不是nums=nums.sort()
    nums.sort()
    res = 0
    m = (n - 1) // 2
    for i in range(n):
        res += abs(nums[i] - nums[m])
    print(res)

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 105. 七夕祭](https://www.acwing.com/problem/content/107/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 货仓选址进阶
> 
> 操作行和列互不影响
> 
> 则总体最小 为 操作行最+操作列最小
> 
> 模型：环形纸牌 邮递员问题 视频关于n-1个等式的转化和分解 ==> 取中位数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 100010;

int row[N], col[N], s[N], c[N];

LL work(int n, int a[]) {
    for (int i = 1; i <= n; ++ i ) s[i] = s[i - 1] + a[i];
    if (s[n] % n) return -1;
    
    int avg = s[n] / n;
    // 求Ci
    c[1] = 0;
    for (int i = 2; i <= n; ++ i ) c[i] = s[i - 1] - (i - 1) * avg;
    
    sort(c + 1, c + n + 1);
    LL res = 0;
    for (int i = 1; i <= n; ++ i ) res += abs(c[i] - c[(n + 1) / 2]);
    return res;
}

int main() {
    int n, m, cnt;
    cin >> n>> m >> cnt;
    while (cnt -- ) {
        int x, y;
        cin >> x >> y;
        ++ row[x], ++ col[y];
    }
    LL r = work(n, row);
    LL c = work(m, col);
    
    if (r != -1 && c != -1) cout << "both "<< r + c << endl;
    else if (r != -1) cout << "row " << r << endl;
    else if (c != -1) cout << "column " << c << endl;
    else cout << "impossible" << endl;

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

### 区间问题

> [!NOTE] **[AcWing 803. 区间合并](https://www.acwing.com/problem/content/805/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

typedef pair<int, int> PII;

void merge(vector<PII> &segs) {
    vector<PII> res;

    sort(segs.begin(), segs.end());

    int st = -2e9, ed = -2e9;
    for (auto seg : segs)
        if (ed < seg.first) {
            if (st != -2e9) res.push_back({st, ed});
            st = seg.first, ed = seg.second;
        } else
            ed = max(ed, seg.second);

    if (st != -2e9) res.push_back({st, ed});

    segs = res;
}

int main() {
    int n;
    scanf("%d", &n);

    vector<PII> segs;
    for (int i = 0; i < n; i++) {
        int l, r;
        scanf("%d%d", &l, &r);
        segs.push_back({l, r});
    }

    merge(segs);

    cout << segs.size() << endl;

    return 0;
}
```

##### **Python**

```python
# 按照区间左端点大小排序
if __name__ == '__main__':
    n = int(input())
    intervals = [list(map(int, input().split())) for _ in range(n)]

    res = []
    intervals.sort()
    l, r = intervals[0][0], intervals[0][1]
    for i in range(1, n):
        if intervals[i][0] <= r:
            r = max(intervals[i][1], r)
        else:
            res.append([l, r])
            l, r = intervals[i][0], intervals[i][1]
    res.append([l, r])
    print(len(res))

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 905. 区间选点](https://www.acwing.com/problem/content/907/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 按右端点排序 如果当前 l 比上次标记大则新增

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;
int n, a, b;
int main() {
    cin >> n;
    vector<pair<int, int>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        ve.push_back({b, a});
    }
    sort(ve.begin(), ve.end());
    int res = 0, mxr = -inf;
    for (auto [r, l] : ve) {
        if (l <= mxr) continue;
        ++res;
        mxr = r;
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 算法过程：
# 1. 将所有的区间按照右端点从小到大排序
# 2. 从前往后依次枚举每个区间，如果当前区间中已经包含，则pass；否则，选择当前区间的右端点
if __name__ == '__main__':
    n = int(input())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort()
    res = 1;
    r = arr[0][1]
    for i in range(n):
        if arr[i][0] > r:
            res += 1
            r = arr[i][1]
    print(res)

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 908. 最大不相交区间数量](https://www.acwing.com/problem/content/910/)**
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

const int inf = 0x3f3f3f3f;
int n, a, b;
int main() {
    cin >> n;
    vector<pair<int, int>> ve;
    // r, l
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        ve.push_back({b, a});
    }
    sort(ve.begin(), ve.end());
    int res = 0, mxr = -inf;
    for (auto [r, l] : ve) {
        if (l <= mxr) continue;
        ++res;
        mxr = r;
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 算法过程：
# 1. 将所有的区间按照右端点从小到大排序
# 2. 从前往后依次枚举每个区间，

if __name__ == '__main__':
    n = int(input())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort(key=lambda x: x[1])
    res = n;
    r = arr[0][1]
    for i in range(1, n):
        if arr[i][0] <= r:
            res -= 1
        else:
            r = arr[i][1]
    # 等价变换：      
    # for i in range(1, n):
    #    if arr[i][0] > r:
    #        r = arr[i][1]
    #    else:
    #        res -= 1
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 906. 区间分组](https://www.acwing.com/problem/content/908/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 按左端点排序 + 小根堆

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;
int n, a, b;

int main() {
    cin >> n;
    vector<pair<int, int>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        ve.push_back({a, b});
    }
    sort(ve.begin(), ve.end());
    priority_queue<int, vector<int>, greater<int>>
        heap;  //小根堆始终保证所有组中的最小的右端点为根节点
    for (auto [l, r] : ve) {
        if (heap.empty() || heap.top() >= l)
            heap.push(r);
        else {
            heap.pop();
            heap.push(r);
        }
    }
    cout << heap.size() << endl;
}
```

##### **Python**

```python
# 算法过程：
# 1. 将所有区间按照左端点从小到大排序
# 2. 从前向后处理每个区间：判断能否将其放到某个现有的组中（L[i]>Max_R)
#    1) 如果不存在这样的组，则开新组，然后再将其放进去；
#    2）如果存在这样的组，将其放进去，并且更新当前组的Max_R
if __name__ == '__main__':
    n = int(input())
    arr = []
    for i in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort(key=lambda x: x[0])

    import heapq

    h = []
    for start, end in arr:
        if len(h) == 0 or h[0] >= start:
            heapq.heappush(h, end)
        else:
            heapq.heapreplace(h, end)
    print(len(h))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 907. 区间覆盖](https://www.acwing.com/problem/content/909/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典贪心 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;
int n, a, b, s, t;

int main() {
    cin >> s >> t;
    cin >> n;
    vector<pair<int, int>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        // 略去无关区间
        if (a > t || b < s) continue;
        ve.push_back({a, b});
    }
    sort(ve.begin(), ve.end());
    int m = ve.size();
    // 剩下的区间全部都有经过[s,t]
    int res = 0;
    bool f = false;
    for (int i = 0; i < m; ++i) {
        int j = i, mx = -inf;
        while (j < m && ve[j].first <= s) mx = max(mx, ve[j].second), ++j;
        // 最靠右的位置都不能满足起始位置（因为是按左端点排序
        if (mx < s) break;
        ++res;
        if (mx >= t) {
            f = true;
            break;
        }
        s = mx;
        i = j - 1;
    }
    if (f)
        cout << res << endl;
    else
        cout << -1 << endl;
}
```

##### **Python**

```python
#算法过程：
#1. 将所有区间按照左端点从小到大排序
#2. 从前往后依次枚举每个区间，在所有能覆盖start的区间中，选择右端点最大的区间；然后将start更新为右端点的最大值。
```

<!-- tabs:end -->
</details>

<br>

* * *

### huffman

> [!NOTE] **[AcWing 148. 合并果子](https://www.acwing.com/problem/content/150/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> Huffman 树

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, w;
int main() {
    cin >> n;
    priority_queue<int, vector<int>, greater<int>> heap;
    for (int i = 0; i < n; ++i) {
        cin >> w;
        heap.push(w);
    }
    int res = 0;
    while (heap.size() > 1) {
        int v1 = heap.top();
        heap.pop();
        int v2 = heap.top();
        heap.pop();
        res += v1 + v2;
        heap.push(v1 + v2);
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 每次找最小的两个字 可以用堆（优先队列），小根堆。
# 维护一个最小的heap，每次都取最小的两个，并将“和”加入到heap

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))

    import heapq

    heapq.heapify(arr)
    res = 0

    while len(arr) >= 2:
        a = heapq.heappop(arr)
        b = heapq.heappop(arr)
        res += a + b
        heapq.heappush(arr, a + b)
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

### 相邻交换

> [!NOTE] **[AcWing 125. 耍杂技的牛](https://www.acwing.com/problem/content/description/127/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推公式
> 
> 经典相邻交换法贪心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
long long w, s;
int main() {
    cin >> n;
    vector<pair<long long, long long>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> w >> s;
        ve.push_back({w + s, s});
    }
    sort(ve.begin(), ve.end());
    long long res = -1e18, sum = 0;
    for (auto [t, s] : ve) {
        sum -= s;
        res = max(res, sum);
        sum += t;
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 算法：按照wi+si从小到大的顺序排，最大的危险系数一定是最小的。
# 证明：1）贪心的答案>=最优解；2）贪心得到的答案<=最优解；

# 证明1）很好证，在这里证明2）：反证法证明。假设最优解不是按照wi+si从小到大排序，那一定存在wi+si>w(i+1)+s(i+1)，此时交换一下i和i+1头牛的位置。
# 交换前：第i头牛：w1+w2+...+w(i-1)-si ; i+1头牛：w1+w2+...+wi-s(i+1)
# 交换后：w1+w2+..+w(i-1)-s(i+1); w1+w2+...+w(i-1)+w(i+1)-si

# 最后会变成：
# s(i+1) --- wi+si 
# si ---- w(i+1)+s(i+1)

# 交换后两个数的最大值 会严格变小。即可证明2）

if __name__ == '__main__':
    n = int(input())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort(key=lambda x: x[0] + x[1])

    res = float('-inf')
    prefix_weight = 0
    for w, s in arr:
        res = max(res, prefix_weight - s)
        prefix_weight += w
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 734. 能量石](https://www.acwing.com/problem/content/description/736/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心的假设和推导原则【贪心的微扰（邻项交换）证法】 
> 
> 先贪心排序 国王游戏 耍杂技的牛
> 
> **[如果表示“恰好”，那么需要把所有非法状态初始化成负无穷。]**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 110, M = 10010;

int n;
struct Stone {
    int s, e, l;
} stones[N];

// 优先选择 Si/Li 最小的值
bool cmp(Stone a, Stone b) { return a.s * b.l < b.s * a.l; }

int f[N][M];

int main() {
    int T;
    cin >> T;
    for (int C = 1; C <= T; C++) {
        cin >> n;
        int m = 0;
        for (int i = 1; i <= n; i++) {
            int s, e, l;
            cin >> s >> e >> l;
            stones[i] = {s, e, l};
            m += s;
        }

        sort(stones + 1, stones + 1 + n, cmp);

        for (int i = 1; i <= n; i++)
            for (int j = 0; j <= m; j++) {
                f[i][j] = f[i - 1][j];
                if (j >= stones[i].s) {
                    int s = stones[i].s, e = stones[i].e, l = stones[i].l;
                    f[i][j] =
                        max(f[i][j], f[i - 1][j - s] + max(0, e - l * (j - s)));
                }
            }

        int res = 0;
        for (int i = 0; i <= m; i++) res = max(res, f[n][i]);

        printf("Case #%d: %d\n", C, res);
    }

    return 0;
}
```

##### **Python**

```python
#贪心+dp：所有不同吃法的最优解
#1）决定可以选择吃哪些（不吃能量变成0的能量石） 2）按照什么顺序吃能量石 （两维变化，不好直接做）
# 有一个非常巧妙的地方：先做一个贪心，找到最优解的一个小子集里。先证明这个子集的存在，然后考虑这个子集里所有的最大值
# 复习一下贪心题：耍杂技的牛 & 国王游戏

# 贪心思路：发现有可能存在最优解的某些宝石的贡献为00，我们剔除了这些宝石。
# 假设最优解的能量石排列长度为k(1<=k<=n) 因为去掉了那些没有贡献的宝石，位置为：a1,a2,a3…aka1,a2,a3…ak。
# 那么对于任意两个位置i=al,j=al+1(1<=l<k)，在最优解中，交换后两个宝石的贡献总和不会变得更大,(假设之前的总时间为t）：整理后：
# Si∗Lj<=Sj∗Li，调整一下: Si/Li<=Sj/Lj
# 这样，我们只要以如上条件作为宝石间排序的条件，进行一次sortsort。
# 那么最优解的坐标（新的坐标）一定满足：ai<a2<a3<...<ak

# dp: 0/1背包问题，Si 作为费用，max(0,Ei−(t−Si)∗Li) 作为价值 (t为当前花费时长)。
# f[t] 表示当“恰好”花时间t 得到的最大能量


if __name__ == '__main__':
    T = int(input())
    for t in range(T):
        n = int(input())
        nums = []
        m = 0
        for i in range(n):
            nums.append(list(map(int,input().split())))
            m += nums[i][0]
        nums.sort(key = lambda x : x[0] / max(x[2], 1e-10))  # 预处理 排序
        f = [0] * (m + 1)
        for i in range(n):
            s, e, l = nums[i]
            for j in range(m, s - 1, -1):
                f[j] = max(f[j], f[j - s] + e - (j - s) * l)  # 这里 f[j + 1] >= f[j] 不一定成立
        res = f[0]
        for i in range(1, m + 1):   # 这里 f[M] 不一定是最大值，这是因为 j 更大的情况下，(j - s) * l 也就是损耗更大
            res = max(res, f[i])
        print(f'Case #{t + 1}: {res}')
        # print("Case #{}: {}".format(i+1,r[i]))
```

<!-- tabs:end -->
</details>

<br>

* * *

### 堆 + 后悔

> [!NOTE] **[Luogu [JSOI2007]建筑抢修](https://www.luogu.com.cn/problem/P4053)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **贪心思路推导 + 堆**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 核心：贪心策略
// 先按 t 贪心，中途再更改
// 1. 按 t 从小到大排序之后，开始轮流遍历每个建筑
// 2. 如果中途某个建筑 i 无法在 t_i 的时间内修复，
//    那么在先前选择修复的建筑中拿出 w_j 最大的 j 号建筑
//    若 w_i < w_j ，则放弃 j 转而修 i。

const int N = 150010;

int n, T;  // T指遍历时经过了多久时间
struct node {
    int w, t;
} a[N];
priority_queue<int> Q;  //优先队列

bool cmp(node x, node y) {
    return x.t < y.t;  //按t从小到大排序
}
int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
        scanf("%d%d", &a[i].w, &a[i].t);
    sort(a + 1, a + n + 1, cmp);

    int res = 0;
    for (int i = 1; i <= n; i++)
        //如果无法修复此楼
        if (T + a[i].w > a[i].t) {
            // ai < aj
            if (a[i].w < Q.top()) {
                //注意这里要减掉
                T -= Q.top();
                Q.pop();
                Q.push(a[i].w);
                T += a[i].w;
            }
        } else {
            Q.push(a[i].w);
            res++;
            T += a[i].w;
        }

    printf("%d\n", res);

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

### trick

> [!NOTE] **[Luogu 最大乘积](https://www.luogu.com.cn/problem/P1249)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心 + 大数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 1. 贪心从 2 开始递增累加(s) 直到大于 n ，把 s-n 删掉
// 2. 高精度累乘

vector<int> mul(vector<int> & a, int b) {
    vector<int> c;
    for (int i = 0, t = 0; i < a.size() || t; ++ i ) {
        if (i < a.size())
            t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }
    while (c.size() > 1 && c.back() == 0)
        c.pop_back();
    return c;
}

int main() {
    int n;
    cin >> n;

    int v = 1, s = 0;
    vector<int> ve;
    while (s < n) {
        v ++ ;
        ve.push_back(v);
        
        s += v;
    }
    
    int nouse = 0;
    if (s > n)
        nouse = s - n;
    
    vector<int> res(1, 1);
    for (auto v : ve) {
        if (v == nouse)
            continue;
        cout << v << ' ';
        res = mul(res, v);
    }
    cout << endl;

    for (int i = res.size() - 1; i >= 0; -- i )
        cout << res[i];
    cout << endl;
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

> [!NOTE] **[Luogu [AHOI2018初中组]分组](https://www.luogu.com.cn/problem/P4447)**
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

using PII = pair<int, int>;
#define x first
#define y second
const int N = 1e5 + 10;

int n;
unordered_map<int, int> cnt;
vector<PII> ve;

int main() {
    cin >> n;
    for (int i = 0; i < n; ++ i ) {
        int v;
        cin >> v;
        cnt[v] ++ ;
    }
    
    for (auto [k, v] : cnt)
        ve.push_back({k, v});
    sort(ve.begin(), ve.end());
    
    int res = 2e9;
    int m = ve.size();
    for (int i = 0; i < m; ++ i ) {
        // 先消耗一个连续上升区间
        ve[i].y -- ;
        int j = i + 1;
        while (j < m && ve[j].x == ve[j - 1].x + 1 && ve[j].y > ve[j - 1].y)
            ve[j].y -- , j ++ ;
        
        // [l, j - 1]
        int len = j - i;
        if (len < res)
            res = len;

        int k = i;
        while (k < m && ve[k].y == 0)
            k ++ ;
        i = k - 1;
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
