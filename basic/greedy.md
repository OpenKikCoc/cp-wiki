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