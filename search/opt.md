
> [!TIP] **常见优化**
> 
> 1. 优化搜索顺序 优先选择可选范围小的
> 2. 排除冗余信息
> 3. 可行性剪枝
> 4. 最优性剪枝
> 5. 记忆化（对于爆搜每次状态都不同 记忆化没用）

## 剪枝方法

最常用的剪枝有三种，记忆化搜索、最优性剪枝、可行性剪枝。

### 记忆化搜索

相同的传入值往往会带来相同的解，那我们就可以用数组来记忆，详见 [记忆化搜索](dp/memo.md)。

### 最优性剪枝

在搜索中导致运行慢的原因还有一种，就是在当前解已经比已有解差时仍然在搜索，那么我们只需要判断一下当前解是否已经差于已有解。

### 可行性剪枝

在搜索过程中当前解已经不可用了还继续搜索下去也是运行慢的原因。

## 剪枝思路

剪枝思路有很多种，大多需要对于具体问题来分析，在此简要介绍几种常见的剪枝思路。

- 极端法：考虑极端情况，如果最极端（最理想）的情况都无法满足，那么肯定实际情况搜出来的结果不会更优了。

- 调整法：通过对子树的比较剪掉重复子树和明显不是最有“前途”的子树。

- 数学方法：比如在图论中借助连通分量，数论中借助模方程的分析，借助不等式的放缩来估计下界等等。

## 例题

> [!NOTE] **工作分配问题**
>
> **题目描述**
>
> 有 $n$ 份工作要分配给 $n$ 个人来完成，每个人完成一份。第 $i$ 个人完成第 $k$ 份工作所用的时间为一个正整数 $t_{i,k}$，其中 $1 \leq i, k \leq n$。试确定一个分配方案，使得完成这 $n$ 份工作的时间总和最小。
>
> 输入包含 $n + 1$ 行。
>
> 第 1 行为一个正整数 $n$。
>
> 第 2 行到第 $n + 1$ 行中每行都包含 $n$ 个正整数，形成了一个 $n \times n$ 的矩阵。在该矩阵中，第 $i$ 行第 $k$ 列元素 $t_{i,k}$ 表示第 $i$ 个人完成第 $k$ 件工作所要用的时间。
>
> 输出包含一个正整数，表示所有分配方案中最小的时间总和。
>
> **数据范围**
>
> $1 \leq n \leq  15$
>
> $1 \leq t_{i,k} \leq 10^4$
>
> **输入样例**
>
>     5
>     9 2 9 1 9
>     1 9 8 9 6
>     9 9 9 9 1
>     8 8 1 8 4
>     9 1 7 8 9
>
> **输出样例**
>
>     5

> [!TIP] **思路**
>
>   由于每个人都必须分配到工作，在这里可以建一个二维数组 `time[i][j]`，用以表示 $i$ 个人完成 $j$ 号工作所花费的时间。
>
>   给定一个循环，从第 1 个人开始循环分配工作，直到所有人都分配到。为第 $i$ 个人分配工作时，再循环检查每个工作是否已被分配，没有则分配给 $i$ 个人，否则检查下一个工作。
>
>   可以用一个一维数组 `is_working[j]` 来表示第 $j$ 号工作是否已被分配，未分配则 `is_working[j]=0`，否则 `is_working[j]=1`。利用回溯思想，在工人循环结束后回到上一工人，取消此次分配的工作，而去分配下一工作直到可以分配为止。
>
>   这样，一直回溯到第 1 个工人后，就能得到所有的可行解。
>
>   检查工作分配，其实就是判断取得可行解时的二维数组的第一维下标各不相同并且第二维下标各不相同。
>
>   而我们是要得到完成这 $n$ 份工作的最小时间总和，即可行解中时间总和最小的一个，故需要再定义一个全局变量 `cost_time_total_min` 表示目前找到的解中最小的时间总和，初始 `cost_time_total_min` 为 `time[i][i]` 之和，即对角线工作时间相加之和。
>
>   在所有人分配完工作时，比较 `count` 与 `cost_time_total_min` 的大小，如果 `count` 小于 `cost_time_total_min`，说明找到了一个最优解，此时就把 `count` 赋给 `cost_time_total_min`。
>
>   但考虑到算法的效率，这里还有一个剪枝优化的工作可以做。就是在每次计算局部费用变量 `count` 的值时，如果判断 `count` 已经大于 `cost_time_total_min`，就没必要再往下分配了，因为这时得到的解必然不是最优解。

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

* * *

## 习题

> [!NOTE] **[AcWing 165. 小猫爬山](https://www.acwing.com/problem/content/167/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dfs分组 + 剪枝

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 20;

int n, m;
int w[N];
int sum[N];
int ans = N;

void dfs(int u, int k) {
    // 最优性剪枝
    if (k >= ans) return;
    if (u == n) {
        ans = k;
        return;
    }

    for (int i = 0; i < k; i++)
        if (sum[i] + w[u] <= m)  // 可行性剪枝
        {
            sum[i] += w[u];
            dfs(u + 1, k);
            sum[i] -= w[u];  // 恢复现场
        }

    // 新开一辆车
    sum[k] = w[u];
    dfs(u + 1, k + 1);
    sum[k] = 0;  // 恢复现场
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; i++) cin >> w[i];

    // 优化搜索顺序
    sort(w, w + n);
    reverse(w, w + n);

    dfs(0, 0);

    cout << ans << endl;

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

> [!NOTE] **[AcWing 166. 数独](https://www.acwing.com/problem/content/168/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **必备题 学习如何快速状态表示**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 常用优化方式：
//  1. 优化搜索顺序 优先选择可选范围小的
//  2. 排除冗余信息
//  3. 可行性剪枝
//  4. 最优性剪枝
//  5. 记忆化（对于爆搜每次状态都不同 记忆化没用）

// 剪枝策略 每次选择分支较小的点进行dfs

const int N = 9, M = 1 << N;

// 这俩数组开优化
// 返回某个数有多少个1   map记录每个数的1在第几位
int ones[M], idx[M];
int row[N], col[N], cell[3][3];
char str[100];

inline int lowbit(int x) { return x & -x; }

inline int get(int x, int y) {
    // 获取当前位置可选数字
    return row[x] & col[y] & cell[x / 3][y / 3];
}

void init() {
    // 一开始每一行每一列的数字可以随便选
    int ALL = (1 << N) - 1;
    for (int i = 0; i < N; ++i)
        // row[i] = col[i] = (1 << N) - 1;
        row[i] = col[i] = ALL;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            // cell[i][j] = (1 << N) - 1;
            cell[i][j] = ALL;
}

void draw(int x, int y, int t, bool is_set) {
    if (is_set)
        str[x * N + y] = '1' + t;
    else
        str[x * N + y] = '.';

    int v = 1 << t;
    if (!is_set) v = -v;
    row[x] -= v;
    col[y] -= v;
    cell[x / 3][y / 3] -= v;
}

bool dfs(int cnt) {
    if (!cnt) return true;

    // 找到可选方案数最少的空格
    int minv = 10;
    int x, y;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (str[i * N + j] == '.') {
                int state = get(i, j);
                if (ones[state] < minv) minv = ones[state], x = i, y = j;
            }

    int state = get(x, y);
    for (int i = state; i; i -= lowbit(i)) {
        // 具体的数值 (t = v - 1)
        int t = idx[lowbit(i)];
        draw(x, y, t, true);
        if (dfs(cnt - 1)) return true;
        draw(x, y, t, false);
    }
    return false;
}

int main() {
    for (int i = 0; i < N; ++i) idx[1 << i] = i;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) ones[i] += i >> j & 1;
    while (cin >> str, str[0] != 'e') {
        init();

        int cnt = 0;
        // 把已经填的数加入限制里去
        for (int i = 0, k = 0; i < N; ++i)
            for (int j = 0; j < N; ++j, ++k)
                if (str[k] != '.')
                    draw(i, j, str[k] - '1', true);
                else
                    ++cnt;
        dfs(cnt);
        puts(str);
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 167. 木棒](https://www.acwing.com/problem/content/169/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 学习剪枝姿势 for循环更新姿势

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 70;
int n, sum, length;
int w[N];
bool vis[N];

// 当前处于第几根棒  正在拼的木棒的长度 正在拼的木棒可用的第一个编号
bool dfs(int u, int cur, int start) {
    // 已有u个木棒 方案合法 返回
    if (u * length == sum) return true;
    // 当前棒已经到达长度 要一个新的棒 ==> 已知进入dfs前确保不会超过length了
    if (cur == length) return dfs(u + 1, 0, 0);

    for (int i = start; i < n; ++i) {
        if (vis[i] || cur + w[i] > length) continue;

        vis[i] = true;
        // 剪枝2：避免重复路径 选了大的不能再选比它小的 否则有重复
        if (dfs(u, cur + w[i], i + 1)) return true;
        vis[i] = false;

        // 剪枝3：当该木棍在开头和结尾都不可以使用的时候
        if (!cur || cur + w[i] == length) return false;

        int j = i;
        while (j < n && w[j] == w[i]) ++j;
        i = j - 1;
    }
    return false;
}

int main() {
    while (cin >> n, n) {
        memset(vis, false, sizeof vis);
        sum = 0;

        for (int i = 0; i < n; ++i) cin >> w[i], sum += w[i];
        // 剪枝1：优先选长的
        sort(w, w + n);
        reverse(w, w + n);

        // length 可以再优化 获取w最大值
        length = 1;
        for (;;) {
            if (sum % length == 0 && dfs(0, 0, 0)) break;
            ++length;
        }
        cout << length << endl;
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 168. 生日蛋糕](https://www.acwing.com/problem/content/170/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 学习剪枝姿势

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 优先选择可选较少的 依据题目条件下面底面积大 故从下往上搜

// r^2h <= n-V
// 先枚举r

const int N = 25;
const int inf = 1e9;

int n, m, res;
int R[N], H[N];        // 每一层的半径和高度
int minv[N], mins[N];  // 前i层体积、侧面积最小值

void dfs(int u, int v, int s) {
    if (v + minv[u] > n) return;     // 剪枝5：不合法
    if (s + mins[u] >= res) return;  // 剪枝6：不会更优
    if (s + 2 * (n - v) / R[u + 1] >= res)
        return;  // 剪枝4：当前表面积+未来最小的表面积>res 不需要再继续

    if (!u) {
        if (v == n) {
            // res = min(res, s);
            res = s;
        }
        return;
    }

    // 剪枝2：先枚举r 再枚举h
    // 剪枝3：r h的起始范围
    for (int r = min(R[u + 1] - 1, (int)sqrt(n - v)); r >= u; --r)
        for (int h = min(H[u + 1] - 1, (n - v) / r / r); h >= u; --h) {
            int t = 0;
            if (u == m) t = r * r;  // 最后一层 计算从上至下累加表面积
            R[u] = r, H[u] = h;
            dfs(u - 1, v + r * r * h, s + 2 * r * h + t);
        }
}

int main() {
    res = inf;
    cin >> n >> m;
    // 从上到下 每一层最小体积和表面积
    for (int i = 1; i <= m; ++i) {
        minv[i] = minv[i - 1] + i * i * i;
        mins[i] = mins[i - 1] + 2 * i * i;
    }
    R[m + 1] = H[m + 1] = inf;
    // 剪枝1：从下至上枚举
    dfs(m, 0, 0);

    if (res == inf) res = 0;
    cout << res << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *