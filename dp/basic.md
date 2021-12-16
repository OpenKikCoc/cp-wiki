
> [!NOTE] **动态规划的两种实现方法**
> 
> - 带备忘的自顶向下法（记忆化搜索）；
> 
> - 自底向上法（将子问题按规模排序，类似于递推）。

算导用子问题图上按照逆拓扑序求解问题，引出记忆化搜索。

重构解（输出方案）：转移的时候记录最优子结构的位置。

## 钢条切割

> [!NOTE] **题意**
> 
> 给定一段钢条，和不同长度的价格，问如何切割使得总价格最大。

> [!TIP] **思路**
> 
> 为了求解规模为 $n$ 的原问题，我们先求解形式完全一样，但规模更小的子问题。
> 
> 即当完成首次切割后，我们将两段钢条看成两个独立的钢条切割问题实例。
> 
> 我们通过组合相关子问题的最优解，并在所有可能的两段切割方案中选取组合收益最大者，构成原问题的最优解。

> 最优子结构：问题的最优解由相关子问题的最优解组合而成，而这些子问题可以独立求解。

## 矩阵链乘法

> [!NOTE] **题意**
> 
> 给出 $n$ 个矩阵的序列，希望计算他们的乘积，问最少需要多少次乘法运算？

> [!TIP] **思路**
> 
> （认为 $p \times q$ 的矩阵与 $q\times r$ 的矩阵相乘代价是 $p\times q\times r$。）
> 
> 调整计算顺序使得连乘最小即可

TODO@binacs 细节

## 最长公共子序列

> [!NOTE] **题意**
> 
> 子序列允许不连续。

> [!TIP] **思路**
> 
> 假设两个字符串分别为 $S_1$ 与 $S_2$，同时创建一个二维数组 $c$ 储存不同状态下的最优解，即最长公共子序列长度。
> 
> 比如说，$c[i][j]$ 储存了 $S_1$ 前 $i$ 位字符与 $S_2$ 前 $j$ 位字符状态下的最优解。
> 
> 每个 $c[i][j]$ 只依赖于 $c[i - 1][j]$、$c[i][j - 1]$ 和 $c[i - 1][j - 1]$。
> 
> 记录最优方案的时候可以不需要额外建表（优化空间），因为重新选择一遍（转移过程）也是 $O(1)$ 的。
> 
> 可参考此 [交互网页](http://lcs-demo.sourceforge.net/) 来更好地理解 LCS 的实现过程。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
string a, b;

int main() {
    cin >> n >> m;
    cin >> a >> b;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            if (a[i - 1] == b[j - 1])
                f[i][j] = f[i - 1][j - 1] + 1;
            else
                f[i][j] = max(f[i - 1][j], f[i][j - 1]);
    cout << f[n][m] << endl;
}
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

* * *

## 最优二叉搜索树

> [!NOTE] **题意**
> 
> 给二叉搜索树的每个节点定义一个权值，问如何安排使得权值和深度的乘积最小。

> [!TIP] **思路**
> 
> 考虑当一棵子树成为了一个节点的子树时，答案（期望搜索代价）有何变化？
> 
> 由于每个节点的深度都增加了 1，这棵子树的期望搜索代价的增加值应为所有概率之和。

> tD/eD 动态规划：
> 
> 状态空间是 $O(n^t)$ 的，每一项依赖其他 $O(n^e)$ 项。

TODO@binacs 细节

## 最长连续不下降子序列

> [!NOTE] **题意**
> 
> 求出给定序列的一个最长的连续子序列，满足这个序列中的后一个元素 **不小于** 前一个元素。

> [!TIP] **思路**
> 
> 因为是连续的，所以只要与上一个元素进行比较即可。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->
###### **C++**

```cpp
// C++ Version
int a[MAXN];
int dp() {
    int now = 1, ans = 1;
    for (int i = 2; i <= n; i++) {
        if (a[i] >= a[i - 1])
            now++;
        else
            now = 1;
        ans = max(now, ans);
    }
    return ans;
}
```

###### **Python**
```python
# Python Version
a = [0] * MAXN
def dp():
    now, ans = 1, 1
    for i in range(2, n + 1):
        if a[i] >= a[i + 1]:
            now += 1
        else:
            now = 1
        ans = max(now, ans)
    return ans
```
<!-- tabs:end -->
</details>

<br>

* * *

## 最长不下降子序列

> [!NOTE] **题意**
> 
> 求出给定序列的一个最长的子序列，满足这个序列中的后一个元素 **不小于** 前一个元素。
> 
> 与最长连续不下降子序列不同的是，不需要这个子序列是连续的了。

> [!TIP] **思路**
> 
> LIS 优化版本

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
int main() {
    int n, v;
    cin >> n;
    vector<int> f;
    for (int i = 0; i < n; ++i) {
        cin >> v;
        if (f.empty() || f.back() <= v) // <=
            f.push_back(v);
        else
            *lower_bound(f.begin(), f.end(), v) = v;
    }
    cout << f.size() << endl;
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

## 经典问题（来自习题）

### DAG 中的最长简单路径

> [!TIP] **思路**
> 
> $dp[i] = \max(dp[j] + 1), ((j, i) \in E)$

### 最长回文子序列


> [!TIP] **思路**
> 
> $$
> dp[i][i + len] =
> \begin{cases}
> dp[i + 1][i + len - 1] + 2,  & \text{if}\ s[i] = s[i + len] \\[2ex]
> \max(dp[i + 1][i + len], dp[i][i + len - 1]), & \text{else}
> \end{cases}
> $$
> 
> 边界：$dp[i][i] = 1$。
>
> **注意：$dp[i][j]$ 表示的是闭区间。**

也可以转化为 LCS 问题，只需要把 $a$ 串反转当做 $b$，对 $a$ 和 $b$ 求 LCS 即可。

证明在 [这里](https://www.zhihu.com/question/34580085/answer/59539708)。

**注意区分子串（要求连续）的问题。**

### 最长回文子串

> [!TIP] **思路**
> 
> $O(n^2)$：$dp[i] = \max(dp[j] + 1), s(j + 1 \cdots i)$ 是回文
> 
> $O(n)$：[Manacher](string/manacher.md)

### 双调欧几里得旅行商问题

> [!NOTE] **题意**
> 
> [程设期末推荐练习](https://ir1d.cf/2018/06/23/cssx/程设期末推荐练习/) 里面的。

> [!TIP] **思路**
> 
> 书上的提示是：从左到右扫描，对巡游路线的两个部分分别维护可能的最优解。
> 
> 说的就是把回路给拆开吧。

TODO@binacs

#### 思路一

$dp[i][j]$ 表示 $1 \cdots i$ 和 $1 \cdots j$ 两条路径。

我们可以人为要求 $1 \cdots i$ 是更快的那一条路径。

这样考虑第 $i$ 个点分给谁。

如果是分给快的那条：

$dp[i][j] = \min(dp[i - 1][j] + dis[i - 1][i]),\ j = 1 \cdots i$

如果是慢的，原来是慢的那条就变成了快的，所以另一条是到 $i - 1$ 那个点：

$dp[i][j] = \min(dp[i - 1][j] + dis[j][i]),\ j = 1 \cdots i$

答案是 $\min(dp[n][i] + dis[n][i])$。
（从一开始编号，终点是 $n$）

#### 思路二

把 $dp[i][j]$ 定义反过来，不是 $1 \cdots i$ 和 $1 \cdots j$。

改成是 $i..n$ 和 $j \cdots n$，不要求哪个更快。

这样的转移更好写：

我们记 $k = \max(i, j) + 1$

$k$ 这个点肯定在两条路中的一个上，$dp[i][j]$ 取两种情况的最小值即可。

$dp[i][j] = \min(dp[i][k] + dis[k][j], dp[k][j] + dis[i][k])$

边界是：$dp[i][n] = dp[n][i] = dis[n][i]$。

答案是 $dp[1][1]$。

### 整齐打印

> [!NOTE] **题意**
> 
> 希望最小化所有行的额外空格数的立方之和。

> [!TIP] **思路**
> 
> 注意到实际问题要求单词不能打乱顺序，所以就好做了起来。**不要把题目看复杂。**
> 
> $dp[i] = \min(dp[j] + cost[j][i])$

### 编辑距离

> [!TIP] **思路**
> 
> 变换操作有 $6$ 种，复制、替换、删除、插入、旋转、终止（结束转换过程）。

TODO@binacs

简单版：插入 / 删除 / 替换

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
int n, m;
char a[N], b[N];
int f[N][N];

int main() {
    scanf("%d%s", &n, a + 1);
    scanf("%d%s", &m, b + 1);

    for (int i = 0; i <= m; i ++ ) f[0][i] = i;
    for (int i = 0; i <= n; i ++ ) f[i][0] = i;

    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ ) {
            f[i][j] = min(f[i - 1][j] + 1, f[i][j - 1] + 1);
            if (a[i] == b[j]) f[i][j] = min(f[i][j], f[i - 1][j - 1]);
            else f[i][j] = min(f[i][j], f[i - 1][j - 1] + 1);
        }

    printf("%d\n", f[n][m]);

    return 0;
}
```

##### **Python**

```python
#状态表示：f[i,j] : 所有将a[1-i]变成b[1 - j]的操作方式的集合；属性：min
#状态转移：以对a中的第i个字母操作不同划分

#1）在该字母之后添加一个字母之后变得相同，说明没有添加前a的前i个已经和b的前j-1个已经相同 
#即：dp[i][j] = dp[i][j - 1] + 1
#2）删除该字母之后变得相同，说明没有删除前a中前i-1已经和b的前j个已经相同
#即 ： dp[i][j] = dp[i - 1][j] + 1
#替换该字母；替换说明对应结尾字母不同，则看倒数第二个
#即： dp[i][j] = dp[i - 1][j - 1] + 1
#啥也不做：对应结尾字母相同，直接比较倒数第二个
#即： dp[i][j] = dp[i - 1][j - 1]


if __name__=='__main__':
    N = 1010
    f = [[0] * N for _ in range(N)]
    
    n = int(input())
    a = ' ' + input()
    m = int(input())
    b = ' ' + input()
    
    #初始化很重要，不然会报错
    for i in range(1, n + 1):
        f[i][0] = i
    for j in range(1, m + 1):
        f[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i] == b[j]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
    print(f[n][m])
```

<!-- tabs:end -->
</details>

<br>

* * *

### 最优对齐问题

> [!NOTE] **题意**
> 
> 把空格符插入到字符串里，使得相似度最大。

> [!TIP] **思路**
> 
> 定义了按字符比较的相似度。
> 
> 然后发现最优对齐问题可以转换为编辑距离问题。
> 
> 相当于仅有三个操作的带权编辑距离。
> 
> ```text
> copy    :  1
> replace : -1
> insert  : -2
> ```

### 公司聚会计划

> [!NOTE] **题意**
> 
> 没有上司的舞会。

> [!TIP] **思路**
> 
> $dp[x][0]$ 是没去，$dp[x][1]$ 是去了。
> 
> $dp[u][0] = \max(dp[v][0], dp[v][1]), v \in son(u)$
> 
> $dp[u][1] = w[u] + dp[v][0], v \in son(u)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 6010;

int n;
int h[N], e[N], ne[N], idx;
int happy[N];
int f[N][2];
bool has_fa[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u) {
    f[u][1] = happy[u];

    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        dfs(j);

        f[u][1] += f[j][0];
        f[u][0] += max(f[j][0], f[j][1]);
    }
}

int main() {
    scanf("%d", &n);

    for (int i = 1; i <= n; i ++ ) scanf("%d", &happy[i]);

    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i ++ ) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(b, a);
        has_fa[a] = true;
    }

    int root = 1;
    while (has_fa[root]) root ++ ;

    dfs(root);

    printf("%d\n", max(f[root][0], f[root][1]));

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

### 译码算法

[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) 之前写词性标注的时候有用到，好像用在输入法里面也是类似的。

本题中用来实现语音识别，其实就是找一条对应的概率最大的路径。

ref：<https://segmentfault.com/a/1190000008720143>

TODO@binacs

### 基于接缝裁剪的图像压缩

> [!NOTE] **题意**
> 
> 要求每一行删除一个像，每个像素都有代价，要求总代价最小。
> 
> 限制：要求相邻两行中删除的像素必须位于同一列或相邻列。

> [!TIP] **思路**
> 
> $dp[i][j] = \min(dp[i - 1][j], dp[i - 1][j - 1], dp[i - 1][j + 1]) + cost[i][j]$
> 
> 边界：$dp[1][i] = cost[1][i]$。

### 字符串拆分

> [!NOTE] **题意**
> 
> 相当于问怎么按顺序拼起来使得总代价最小。
> 
> 等价于之前那个最优二叉搜索树。

> [!TIP] **思路**
> 
> $dp[i][j] = \min(dp[i][k] + dp[k][j]) + l[j] - l[i] + 1,\ k = i + 1 \cdots j - 1$
> 
> 注意 $l[i]$ 表示的是第 i 个切分点的位置。
> 
> 边界：$dp[i][i] = 0$。
> 
> 就按照区间 dp 的姿势来写就好了。

### 投资策略规划

> 引理：存在最优投资策略，每年都将所有钱投入到单一投资中。

这是个很有趣的结论，dp 问题中很常见。

<https://fogsail.github.io/2017/05/08/20170508/>

剩下的就是个二维 dp，想成从 $(1, i)$ 走到 $(n, m)$ 的路径的问题，然后收益和代价就是边权，网格图只能往右下方走。

TODO@binacs

### 库存规划

> [!NOTE] **题意**
> 
> 生产多了少了都有额外的成本，问怎么安排生产策略使得额外的成本尽可能地少。

> [!TIP] **思路**
> 
> $cost[i][j]$ 表示剩下 $i$ 个月，开始的时候有 $j$ 台库存的最小成本。

<https://walkccc.github.io/CLRS/Chap15/Problems/15-11/>

TODO@binacs

### 签约棒球自由球员

> [!NOTE] **题意**
> 
> TODO

> [!TIP] **思路**
> 
> $v[i][j]$ 是考虑 $i$ 之后的位置，总费用为 $x$ 的最大收益。

<https://walkccc.github.io/CLRS/Chap15/Problems/15-12/>

类似于背包问题。

* * *

## DP 转化思想

> [!NOTE]
> 
> **当选取的状态难以进行递推时（分解出的子问题和原问题形式不一样），考虑将问题状态分类细化，增加维度。**

TODO@binacs 比如拆点


## 习题

### 数字三角形模型

> [!NOTE] **[AcWing 1015. 摘花生](https://www.acwing.com/problem/content/1017/)**
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
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 110;

int n, m;
int w[N][N];
int f[N][N];

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        scanf("%d%d", &n, &m);
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++) scanf("%d", &w[i][j]);

        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                f[i][j] = max(f[i - 1][j], f[i][j - 1]) + w[i][j];

        printf("%d\n", f[n][m]);
    }

    return 0;
}
```

##### **Python**

```python
# 本题求最大值。本题题意 所有位置都是非负数，所以 f的状态全部初始化为0 也不会影响后面的转移（因为正数都比0大）
# 如果此题的花生有负数，那就需要把f初始化为float('-inf')

N = 110
a = [[0] * N for _ in range(N)]
f = [[0] * N for _ in range(N)]

if __name__ == '__main__':
    T = int(input())
    while T:
        R, C = map(int, input().split())
        for i in range(1, R + 1):
            nums = list(map(int, input().split()))
            for j, val in enumerate(nums):
                a[i][j + 1] = val
        # f[1][0], f[0][1] = 0, 0 # 初始化（如果存在负数的数，就需要初始化负无穷）
        for i in range(1, R + 1):
            for j in range(1, C + 1):
                f[i][j] = max(f[i][j - 1], f[i - 1][j]) + a[i][j]
        print(f[R][C])
        T -= 1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1018. 最低通行费](https://www.acwing.com/problem/content/1020/)**
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
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 110, INF = 1e9;

int n;
int w[N][N];
int f[N][N];

int main() {
    scanf("%d", &n);

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++) scanf("%d", &w[i][j]);

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            if (i == 1 && j == 1)
                f[i][j] = w[i][j];  // 特判左上角
            else {
                f[i][j] = INF;
                if (i > 1)
                    f[i][j] = min(
                        f[i][j],
                        f[i - 1][j] +
                            w[i][j]);  // 只有不在第一行的时候，才可以从上面过来
                if (j > 1)
                    f[i][j] = min(
                        f[i][j],
                        f[i][j - 1] +
                            w[i][j]);  // 只有不在第一列的时候，才可以从左边过来
            }

    printf("%d\n", f[n][n]);

    return 0;
}
```

##### **Python**

```python
# 时间限制 不能走回头路。只能向下 或者 向左 走。
# ！注意：和 种花生的区别：求最小！！
# 如果不想写边界值的判断，就需要如下定义f数组 以及 初始化
N = 110
a = [[0] * N for _ in range(N)]
f = [[float('inf')] * N for _ in range(N)]  # 这道题 求最小值，初始化为最大值
# 求最大，全部初始化为最小值（特别的初始值设为0）；求最小，全部初始化为最大值（特别的初始值设为0）
# 求方案数，全部初始化为0（特别的初始值设为1）

if __name__ == '__main__':
    n = int(input())
    for i in range(1, n + 1):
        nums = list(map(int, input().split()))
        for j, val in enumerate(nums):
            a[i][j + 1] = val 
    f[1][0], f[0][1] = 0, 0 # 初始化很重要！后面需要从哪里转移过来，就初始化哪里。
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            f[i][j] = min(f[i - 1][j], f[i][j - 1]) + a[i][j]
    print(f[n][n])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1027. 方格取数](https://www.acwing.com/problem/content/1029/)**
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
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 15;

int n;
int w[N][N];
int f[N * 2][N][N];

int main() {
    scanf("%d", &n);

    int a, b, c;
    while (cin >> a >> b >> c, a || b || c) w[a][b] = c;

    for (int k = 2; k <= n + n; k++)
        for (int i1 = 1; i1 <= n; i1++)
            for (int i2 = 1; i2 <= n; i2++) {
                int j1 = k - i1, j2 = k - i2;
                if (j1 >= 1 && j1 <= n && j2 >= 1 && j2 <= n) {
                    int t = w[i1][j1];
                    if (i1 != i2) t += w[i2][j2];
                    int &x = f[k][i1][i2];
                    x = max(x, f[k - 1][i1 - 1][i2 - 1] + t);
                    x = max(x, f[k - 1][i1 - 1][i2] + t);
                    x = max(x, f[k - 1][i1][i2 - 1] + t);
                    x = max(x, f[k - 1][i1][i2] + t);
                }
            }

    printf("%d\n", f[n + n][n][n]);
    return 0;
}
```

##### **Python**

```python
# 这里是走两次，但是每个方格的数字只能被取一次！但是两个可以走到同一个方格上。
# 1）假设两次同时走的思路：类推到 走两次：f[i1, j1, i2, j2]表示所有从(1,1)，(1,1)分别走到(i1, j1)和(i2, j2)的路径的最大值。（两条路同时走，那么每个时刻，他们的横纵坐标之和 一定相等）
# 2）如何处理 同一个格子不能被重复选择的问题？
#    ==> 什么时候两个格子可能会有交集？ 充分条件：i1 + j1 == i2 + j2（由于存在这个等价关系，所以可把状态优化成三个状态）
# 状态表示： f[k, i1, i2]： 表示所有从(1,1)，(1,1)分别走到(i1, k - i1)和(i2, k - i2)的路径的最大值。（k表示当前格子的横纵坐标之和）
#             k = i1 + j1 = i2 + j2
# 状态转移：根据两条路线最后一步从哪里走过来：1）第一条 向下 + 第二条 向下；2）第一条 向下 + 第二条 向右； 1）第一条 向右 + 第二条 向下； 4） 第一条 向右 + 第二条 向右
# 对于第一种情况：(1, 1) -> (i1 - 1, j1) -> （向下走到）(i1, j1) ； (1, 1) -> (i2 - 1, j2) -> (向下走到) (i2, j2)
# 第一种情况即为：f[k - 1, i1 - 1, i2 - 1]; 第二部分需要判断 (i1, j1)和(i2, j2)是否重合；如果重合了，只需要加一次，如果不重复 两个都需要加。
# 其他的三种情况同理可得出。最后的最大值 是每类的最大值 取一个max 

N = 11
f = [[[0] * N for _ in range(N)] for _ in range(2 * N)]
w = [[0]*N for _ in range(N)]

if __name__ == '__main__':
    n = int(input())
    while True:
        a, b, c = map(int,input().split(' '))
        if a == 0 and b == 0 and c == 0:
            break 
        else:
            w[a][b] = c
    for k in range(2, 2 * n + 1):
        for i1 in range(1, n + 1):
            for i2 in range(1, n + 1):
                j1, j2 = k - i1, k - i2
                if 1 <= j1 <= n and 1 <= j2 <= n:
                    t = w[i1][j1]
                    if i1 != i2:
                        t += w[i2][j2]
                    f[k][i1][i2] = max(f[k - 1][i1 - 1][i2 - 1],f[k - 1][i1 - 1][i2],f[k - 1][i1][i2 - 1],f[k - 1][i1][i2]) + t
    print(f[n + n][n][n])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 275. 传纸条](https://www.acwing.com/problem/content/description/277/)**
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
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 55;

int n, m;
int g[N][N];
int f[N * 2][N][N];

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) scanf("%d", &g[i][j]);

    for (int k = 2; k <= n + m; k++)
        for (int i = max(1, k - m); i <= n && i < k; i++)
            for (int j = max(1, k - m); j <= n && j < k; j++)
                for (int a = 0; a <= 1; a++)
                    for (int b = 0; b <= 1; b++) {
                        int t = g[i][k - i];
                        if (i != j || k == 2 ||
                            k ==
                                n + m)  // 除了起点和终点之外，其余每个格子只能走一次
                        {
                            t += g[j][k - j];
                            f[k][i][j] =
                                max(f[k][i][j], f[k - 1][i - a][j - b] + t);
                        }
                    }

    printf("%d\n", f[n + m][n][n]);

    return 0;
}
```

##### **Python**

```python
# 目标：集合：所有从左上走到右下，走两遍的方案；属性Max
# 状态表示：走两遍 可以看成是同时走。但是不能走到一个格子上。f(x1, y1, x2, y2)：表示所有从左上（1，1）第一条走走到（x1, y1), 第二条走到（x2, y2)的这样的两条路线总和的集合。
# 这两条路线什么情况下 会相互影响到？==> 重叠时，能分析清楚就可以。
# 当x1 + y1 == x2 + y2时，可能会重合。（两条路线 如果有重合的部分，一定是坐标之和相等） k：表示横纵坐标之和。
# 状态表示：f(k,x1,x2): 所有第一条路径从（1，1）走到（x1,k-x1)，第二条路线从（1，1）走到（x2, k-x2）的路线组合的集合； 属性：max
# 状态计算：把集合划分为若干个子集，求子集的max：一共有四种方案


# 只有当权值都为非负数时，才可以复用方格取数的代码（题解区有证明）
# 两次路线经过同一个点，但格子里的数只能算一次，这样一来，其中一条路线从旁边格子绕路，可以多加绕路格子的数值，因为所有权值非负，这样绕路得到的总和不会比路径相交时更小（可以把路径相交理解为多走了一个权值为0的格子，而绕路也是多走一个格子，并且权值>=0）
N = 110
f = [[[0] * N for _ in range(N)] for _ in range(N)]
w = [[0] * N for _ in range(N)]

if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):
        nums = list(map(int, input().split()))
        for j, val in enumerate(nums):
            w[i][j + 1] = val
    for k in range(2, n + m + 1):
        for i1 in range(1, n + 1):
            for i2 in range(1, n + 1):
                j1, j2 = k - i1, k - i2
                if 1 <= j1 <= m and 1 <= j2 <= m:
                    t = w[i1][j1]
                    if i1 != i2:
                        t += w[i2][j2]
                    f[k][i1][i2] = max(f[k - 1][i1 - 1][i2 - 1], f[k - 1][i1 - 1][i2], f[k - 1][i1][i2],
                                       f[k - 1][i1][i2 - 1]) + t
    print(f[n + m][n][n])

# 当权值存在负数时， 如果用以上的代码，那么 走同一个格子得到的结果 可能要优于 “每个格子只能走一次”的结果，所以在进行状态转移的时候，需要做判断。
N = 110
f = [[[0] * N for _ in range(N)] for _ in range(N)]
w = [[0] * N for _ in range(N)]

if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):
        nums = list(map(int, input().split()))
        for j, val in enumerate(nums):
            w[i][j + 1] = val
    for k in range(2, n + m + 1):
        for i1 in range(1, n + 1):
            for i2 in range(1, n + 1):
                j1, j2 = k - i1, k - i2
                if 1 <= j1 <= m and 1 <= j2 <= m:
                    t = w[i1][j1]
                    if i1 != i2 or k == 2 or k == n + m:  # 这里需要判断 才能进入转移
                        t += w[i2][j2]
                        f[k][i1][i2] = max(f[k - 1][i1 - 1][i2 - 1], f[k - 1][i1 - 1][i2], f[k - 1][i1][i2],
                                           f[k - 1][i1][i2 - 1]) + t
    print(f[n + m][n][n])


```

<!-- tabs:end -->
</details>

<br>

* * *

### LIS 模型

> [!NOTE] **[AcWing 482. 合唱队形](https://www.acwing.com/problem/content/484/)**
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
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 1010;

int n;
int h[N];
int f[N], g[N];

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; i++) scanf("%d", &h[i]);

    for (int i = 0; i < n; i++) {
        f[i] = 1;
        for (int j = 0; j < i; j++)
            if (h[i] > h[j]) f[i] = max(f[i], f[j] + 1);
    }

    for (int i = n - 1; i >= 0; i--) {
        g[i] = 1;
        for (int j = n - 1; j > i; j--)
            if (h[i] > h[j]) g[i] = max(g[i], g[j] + 1);
    }

    int res = 0;
    for (int i = 0; i < n; i++) res = max(res, f[i] + g[i] - 1);

    printf("%d\n", n - res);

    return 0;
}
```

##### **Python**

```python
# 最优化的问题 从“集合”的角度进行考虑
# 在原序列中，满足“先递增再递减的子序列”里找到长度最长的子序列

N = 1010
a = [0]*N
f1 = [1]*N
f2 = [1]*N

if __name__ == '__main__':
    n = int(input())
    a[1:] = list(map(int,input().split()))
    for i in range(1, n + 1):
        for j in range(1, i):
            if a[j] < a[i]:
                f1[i] = max(f1[i], f1[j] + 1)
    for i in range(n, 0, -1):
        for j in range(n, i - 1, -1):
            if a[j] < a[i]:
                f2[i] = max(f2[i], f2[j] + 1)
    res = []
    for i in range(1, n + 1):
        res.append(f1[i] + f2[i] - 1)
    print(n - max(res))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1012. 友好城市](https://www.acwing.com/problem/content/1014/)**
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
#include <algorithm>
#include <iostream>

using namespace std;

typedef pair<int, int> PII;

const int N = 5010;

int n;
PII city[N];
int f[N];

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; i++) scanf("%d%d", &city[i].first, &city[i].second);
    sort(city, city + n);

    int res = 0;
    for (int i = 0; i < n; i++) {
        f[i] = 1;
        for (int j = 0; j < i; j++)
            if (city[i].second > city[j].second) f[i] = max(f[i], f[j] + 1);
        res = max(res, f[i]);
    }

    printf("%d\n", res);

    return 0;
}
```

##### **Python**

```python
# 所有合法的建桥方式； 上升子序列 ； 航道交叉也就是a1 >= a2 && b1 <= b2(其中a, b表示两岸)
# 方法：1. 通过排序，固定自变量的顺序； 2.找到因变量的最大上升子序列

N = 5010
f = [1] * N

if __name__=='__main__':
    n = int(input())
    nums = [[0] * 2 for _ in range(n + 1)]
    for i in range(1, n + 1):
        nums[i][0], nums[i][1] = map(int,input().split())
    nums.sort(key = lambda x : x[0])
    for i in range(1, n + 1):
        for j in range(1, i):
            if nums[i][1] > nums[j][1]:
                f[i] = max(f[i], f[j] + 1)
    print(max(f))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1010. 拦截导弹](https://www.acwing.com/problem/content/description/1012/)**
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
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace std;

const int N = 1010;

int n;
int h[N], f[N], q[N];

int main() {
    string line;
    getline(cin, line);
    stringstream ssin(line);
    while (ssin >> h[n]) n++;

    int res = 0, cnt = 0;
    for (int i = 0; i < n; i++) {
        f[i] = 1;
        for (int j = 0; j < i; j++)
            if (h[i] <= h[j]) f[i] = max(f[i], f[j] + 1);
        res = max(res, f[i]);

        int k = 0;
        while (k < cnt && q[k] < h[i]) k++;
        if (k == cnt)
            q[cnt++] = h[i];
        else
            q[k] = h[i];
    }

    printf("%d\n", res);
    printf("%d\n", cnt);
    return 0;
}
```

##### **Python**

```python
# LS + 贪心问题
N = 1010
f = [1] * N
a = [0] * N

if __name__ == '__main__':
    a[1:] = list(map(int, input().split()))
    for i in range(1, len(a)):
        for j in range(1, i):
            if a[i] <= a[j]:
                f[i] = max(f[i], f[j] + 1)
    print(max(f))

    # 逆向思维：（最长上升子序列。可以用贪心解法 来做）
    # 一个序列最少用多少个非下降子序列覆盖完的方案数 == 最长上升子序列的方案数
    f = [1] * N
    for i in range(1, len(a)):
        for j in range(1, i):
            if a[i] > a[j]:
                f[i] = max(f[i], f[j] + 1)
    print(max(f))

# 第二问 也可以用正常的贪心思想来做：
# 第一个导弹加入到系统1，第二个导弹 有两种选择：1. 接在现有的某个子序列之后；2. 创建一个新系统
# 贪心流程：
# 从前往后扫描每个数，对于每个数：
# 情况1:如果现有的子序列的结尾都小于当前数，则创建子序列；情况2:将当前数放在结尾大于等于它的最小的子序列后面；
# 需要证明 这个贪心算法的正确性：常用方式：
# 如果证明两个数相等? 只需要证明：A>=B,A<=B 【A:表示贪心算法得到的序列个数；B表示最优解】
# B<=A:很容易得到；；；A <=B , 用调整法证明：
# 假设最优解对应的方案和当前贪心方案不同，找到第一个不同的数。交换后，还是合法解，并且不会增加新的方案数。
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 187. 导弹防御系统]()**
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

int n, res;
vector<int> c, up, down;

void dfs(int u, int d, int t) {
    if (u + d >= res) return;
    if (t == n) {
        res = min(res, u + d);
        return;
    }
    int i;
    for (i = 1; i <= u; ++i)
        if (up[i] < c[t]) break;
    int tmp = up[i];
    up[i] = c[t];
    dfs(max(u, i), d, t + 1);
    up[i] = tmp;

    for (i = 1; i <= d; ++i)
        if (down[i] > c[t]) break;
    tmp = down[i];
    down[i] = c[t];
    dfs(u, max(d, i), t + 1);
    down[i] = tmp;
};

int main() {
    while (cin >> n, n) {
        // up 存储所有单调上升子序列的末尾元素 down下降末尾元素
        // 加速判断当前数能否放到某一个子序列后面
        // 放的时候贪心：比当前元素小的最大的元素
        c.clear(), up.clear(), down.clear();
        c.resize(n + 1), up.resize(n + 1), down.resize(n + 1);
        for (int i = 1; i <= n; ++i) cin >> c[i];
        res = INT_MAX / 2;

        dfs(0, 0, 1);
        cout << res << endl;
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

> [!NOTE] **[LeetCode 1713. 得到子序列的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/)**
> 
> [Weekly-222](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-01-03_Weekly-222)
> 
> 题意: 大数据范围的 $LCS$

> [!TIP] **思路**
> 
> 一般 $LCS$ 的复杂度为 $O(n^2)$ , 对于数据规模较大的情况无法处理。
> 
> 此时，**在无重复值的情况下可以转化为 $LIS$ 问题**，进而以 $O(nlogn)$ 的复杂度解决。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minOperations(vector<int>& target, vector<int>& arr) {
        unordered_map<int, int> pos;
        for (int i = 0; i < target.size(); ++ i ) pos[target[i]] = i;   // target无重复数组
        vector<int> ve;
        for (auto & x : arr)
            if (pos.count(x)) ve.push_back(pos[x]);
        
        vector<int> f;
        for (auto & x : ve)
            if (f.empty() || f.back() < x) f.push_back(x);
            else *lower_bound(f.begin(), f.end(), x) = x;
        return target.size() - f.size();
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

> [!NOTE] **[AcWing 272. 最长公共上升子序列](https://www.acwing.com/problem/content/description/274/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 注意状态定义：
> 
> $f[i][j]$ 表示以j为结尾 $a[1~i]$ , $b[1~j]$ 都出现过的公共上升子序列集合

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 正解TLE**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> a(n + 1), b(n + 1);
    for (int i = 1; i <= n; ++i) cin >> a[i];
    for (int i = 1; i <= n; ++i) cin >> b[i];
    vector<vector<int>> f(n + 1, vector<int>(n + 1));
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            // not a[i]
            f[i][j] = f[i - 1][j];
            // has a[i]
            if (a[i] == b[j]) {
                int maxv = 1;
                for (int k = 1; k < j; ++k) {
                    // cause a[i] = b[j];
                    if (a[i] > b[k]) maxv = max(maxv, f[i - 1][k] + 1);
                }
                f[i][j] = max(f[i][j], maxv);
            }
        }
    }
    int res = 0;
    for (int i = 1; i <= n; ++i) res = max(res, f[n][i]);
    cout << res << endl;
}
```

##### **C++ 正解优化n^2**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> a(n + 1), b(n + 1);
    for (int i = 1; i <= n; ++i) cin >> a[i];
    for (int i = 1; i <= n; ++i) cin >> b[i];
    vector<vector<int>> f(n + 1, vector<int>(n + 1));
    for (int i = 1; i <= n; ++i) {
        int maxv = 1;
        // j 从小到大 所以每次更新一次即可
        for (int j = 1; j <= n; ++j) {
            // not a[i]
            f[i][j] = f[i - 1][j];
            // has a[i]
            if (a[i] == b[j]) f[i][j] = max(f[i][j], maxv);
            // update maxv
            if (a[i] > b[j])
                maxv = max(maxv, f[i - 1][j] + 1);  // 【注意 这里是 f[i-1][j]】
        }
    }
    int res = 0;
    for (int i = 1; i <= n; ++i) res = max(res, f[n][i]);
    cout << res << endl;
}
```

##### **Python TLE**

```python
# 状态表示：f[i,j]表示的是所有在A[1..i]和B[1...j]中都出现过，并且以B[j]为结尾的所有的公共上升子序列的集合（需要把上一个数 即B[j] 存下来）; 属性：Max
# 状态计算：1）a[i]不包含在公共子序列中  ===> 根据定义：f[i][j] = f[i-1,j]
#         2）a[i]出现在公共子序列中：那肯定是基于a[i] == b[j]前提下, 以倒数第二个数来继续划分以下下子集：   ==> 1. 不存在倒数第二个数：长度就是1； 2. 倒数第二个数是a[1]: f[i - 1, 1] + 1； 3. 倒数第二个数是a[2]: f[i - 1, 2] + 1 ... 倒数第二个数是a[i - k]: f[i - 1, k] + 1...

# 朴素写法，TLE
N = 3010
a = [0] * N
b = [0] * N
f = [[0] * N for _ in range(N)]

if __name__ == '__main__':
    n = int(input())
    a[1:] = list(map(int, input().split()))
    b[1:] = list(map(int, input().split()))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            f[i][j] = f[i - 1][j]
            if a[i] == b[j]:
                maxv = 1
                for k in range(1, j):
                    if a[i] > b[k]:
                        maxv = max(maxv, f[i - 1][k] + 1)
                f[i][j] = max(f[i][j], maxv)
    res = 0
    for i in range(len(f)):
        for j in range(len(f[0])):
            res = max(res, f[i][j])
    print(res)
```

##### **Python 优化**

```python
# 优化
N = 3010
a = [0] * N
b = [0] * N
f = [[0] * N for _ in range(N)]

if __name__=='__main__':
    n = int(input())
    a[1:] = list(map(int,input().split()))
    b[1:] = list(map(int,input().split()))
    for i in range(1, n + 1):
        maxv = 1
        for j in range(1, n + 1):
            f[i][j] = f[i-1][j]
            if a[i] == b[j]:
                f[i][j] = max(f[i][j], maxv)
            if a[i] > b[j]:
                maxv = max(maxv, f[i - 1][j] + 1)
                
    res=0
    for i in range(len(f)):
        for j in range(len(f[0])):
            res = max(res, f[i][j])
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *