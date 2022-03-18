
> t[!NOTE] **动态规划的两种实现方法**
> 
> - 带备忘的自顶向下法（记忆化搜索）；
> 
> - 自底向上法（将子问题按规模排序，类似于递推）。

算导用子问题图上按照逆拓扑序求解问题，引出记忆化搜索。

重构解（输出方案）：转移的时候记录最优子结构的位置。

## 动规思想例题

### 钢条切割

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

### 矩阵链乘法

> [!NOTE] **题意**
> 
> 给出 $n$ 个矩阵的序列，希望计算他们的乘积，问最少需要多少次乘法运算？

> [!TIP] **思路**
> 
> （认为 $p \times q$ 的矩阵与 $q\times r$ 的矩阵相乘代价是 $p\times q\times r$。）
> 
> 调整计算顺序使得连乘最小即可

TODO@binacs 细节

### 最长公共子序列

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

### 最优二叉搜索树

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

### 最长连续不下降子序列

> [!NOTE] **题意**
> 
> 求出给定序列的一个最长的连续子序列，满足这个序列中的后一个元素 **不小于** 前一个元素。

> [!TIP] **思路**
> 
> 因为是连续的，所以只要与上一个元素进行比较即可。

### 最长不下降子序列

> [!NOTE] **题意**
> 
> 求出给定序列的一个最长的子序列，满足这个序列中的后一个元素 **不小于** 前一个元素。
> 
> 与最长连续不下降子序列不同的是，不需要这个子序列是连续的了。

> [!TIP] **思路**
> 
> LIS 优化版本

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
> 本题求最大值。本题题意，所有位置都是非负数，所以 f 的状态全部初始化为0，也不会影响后面的转移（因为正数都比0大）
>
> 如果此题的花生有负数，那就需要把 f 初始化为float('-inf')

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
> 时间限制 不能走回头路。只能向下 或者 向左 走。
>
> ！注意：和 种花生的区别：求最小！！
>
> 如果不想写边界值的判断，就需要如下定义f数组 以及 初始化。
>
> **总结初始化技巧**：
>
> 1. 求最大，全部初始化为最小值（特别的初始值设为0）；
> 2. 求最小，全部初始化为最大值（特别的初始值设为0）
> 3. 求方案数，全部初始化为0（特别的初始值设为1）

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
N = 110
a = [[0] * N for _ in range(N)]
f = [[float('inf')] * N for _ in range(N)]  # 这道题 求最小值，初始化为最大值


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
> 这里是走两次，但是每个方格的数字只能被取一次！但是两个可以走到同一个方格上。
>
> 1）假设两次同时走的思路：
>
> ​	类推到 走两次：f[i1, j1, i2, j2]表示所有从(1,1)，(1,1)分别走到(i1, j1)和(i2, j2)的路径的最大值。（两条路同时走，那么每个时刻，他们的横纵坐标之和 一定相等）
>
> 2）如何处理 同一个格子不能被重复选择的问题？
>
> ==> 什么时候两个格子可能会有交集？ 充分条件：i1 + j1 == i2 + j2（由于存在这个等价关系，所以可把状态优化成三个状态）
>
> 1. 状态表示： f[k, i1, i2]： 表示所有从(1,1)，(1,1)分别走到(i1, k - i1)和(i2, k - i2)的路径的最大值。（k表示当前格子的横纵坐标之和）
>
>    k = i1 + j1 = i2 + j2
>
> 2. 状态转移：根据两条路线最后一步从哪里走过来：
>
>    1）第一条 向下 + 第二条 向下；
>
>    2）第一条 向下 + 第二条 向右； 
>
>    3）第一条 向右 + 第二条 向下； 
>
>    4） 第一条 向右 + 第二条 向右
>
> 对于第一种情况：(1, 1) -> (i1 - 1, j1) -> （向下走到）(i1, j1) ； (1, 1) -> (i2 - 1, j2) -> (向下走到) (i2, j2)
>
> 第一种情况即为：f[k - 1, i1 - 1, i2 - 1]; 第二部分需要判断 (i1, j1)和(i2, j2)是否重合；如果重合了，只需要加一次，如果不重复 两个都需要加。
>
> 其他的三种情况同理可得出。最后的最大值 是每类的最大值 取一个max 

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
> 目标：集合：所有从左上走到右下，走两遍的方案；属性Max
>
> 状态表示：走两遍 可以看成是同时走。但是不能走到一个格子上。f(x1, y1, x2, y2)：表示所有从左上（1，1）第一条走走到（x1, y1), 第二条走到（x2, y2)的这样的两条路线总和的集合。
>
> 这两条路线什么情况下 会相互影响到？==> 重叠时，能分析清楚就可以。
>
> 当x1 + y1 == x2 + y2时，可能会重合。（两条路线 如果有重合的部分，一定是坐标之和相等） k：表示横纵坐标之和。
>
> 1. 状态表示：f(k,x1,x2): 所有第一条路径从（1，1）走到（x1,k-x1)，第二条路线从（1，1）走到（x2, k-x2）的路线组合的集合； 属性：max
>
> 2. 状态转移：把集合划分为若干个子集，求子集的max：一共有四种方案
>
> 只有当权值都为非负数时，才可以复用方格取数的代码（题解区有证明）
>
> 两次路线经过同一个点，但格子里的数只能算一次，这样一来，其中一条路线从旁边格子绕路，可以多加绕路格子的数值，因为所有权值非负，这样绕路得到的总和不会比路径相交时更小（可以把路径相交理解为多走了一个权值为0的格子，而绕路也是多走一个格子，并且权值>=0）

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

> [!NOTE] **[LeetCode 174. 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 特殊构造转移方式
>
> 也可以二分做
>
> 
>
> 这道题不能直接从正向动态规划的原因是：不确定起始点的值，但可以发现，到终点之后健康值为 1 一定是最优解。 
>
> ​	考虑从终点到起点进行 dp
>
> 1. 状态表示：f[i,j] 表示从 [i,j] 成功到达终点时，在 [i,j] 处需要具备的最少能量值
>
>    初始状态，f[n - 1, m - 1]即在终点的最小健康值，max(1, 1 - w[n-1, m-1])  上一步的能量值也至少为 1
>
> 2. 状态转移，f[i, j] = min(f[i + 1, j], f[i, j + 1] - w[i, j]) 但是f[i,j]又必须要大于 1， 所以最后还要和 1 比较取 min
>
>    最终返回的答案: **f[0, 0]** 
>
> 转移详解：
>
> 从[i,j]往下一步走，加上当前权值，一定要大于等于 到下一步最少具备的能量，也就是：f[i,j] + w[i,j] >= f[i + 1, j]
>
> f[i, j] >= f[i + 1, j] - w[i, j], 由于两个方向要取最小的，所以直接： 
>
> f[i, j] = f[i + 1, j] - w[i, j ]





<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    /*
    1.这题不能直接求路径和最大，因为，如果走某一个点和小于等于0，就挂了。
    2.可以倒着走，遇到公主后，血量最少为1.

    f[i][j] = max(1, min(f[i + 1][j], f[i][j + 1]) - dungeon[i][j]);
    f[i][j]表示到(i, j)点需要最少的血量。

    方便计算多添加一行一列。使f[n][m - 1] = 1, f[m][m - 1] = 1;
    */
    const int inf = 0x3f3f3f3f;
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int n = dungeon.size(), m = dungeon[0].size();
        if (!n) return 0;
        vector<vector<int>> f(n + 1, vector<int>(m + 1, inf));
        f[n][m - 1] = f[n - 1][m] = 1;
        for (int i = n - 1; i >= 0; -- i )
            for (int j = m - 1; j >= 0; -- j )
                f[i][j] = max(1, min(f[i + 1][j], f[i][j + 1]) - dungeon[i][j]);
        return f[0][0];
    }
};
```

##### **C++ 2**

```cpp
// yxc
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& w) {
        int n = w.size(), m = w[0].size();
        vector<vector<int>> f(n, vector<int>(m, 1e8));

        for (int i = n - 1; i >= 0; i -- )
            for (int j = m - 1; j >= 0; j -- )
                if (i == n - 1 && j == m - 1) f[i][j] = max(1, 1 - w[i][j]);
                else {
                    if (i + 1 < n) f[i][j] = f[i + 1][j] - w[i][j];
                    if (j + 1 < m) f[i][j] = min(f[i][j], f[i][j + 1] - w[i][j]);
                    f[i][j] = max(1, f[i][j]);
                }

        return f[0][0];
    }
};
```

##### **Python**

```python
class Solution:
    def calculateMinimumHP(self, w: List[List[int]]) -> int:
        n, m = len(w), len(w[0])
        f = [[float('inf')] * (m) for _ in range(n)]
        f[n - 1][m - 1] = max(1, 1 - w[n - 1][m - 1])
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if i + 1 < n:
                    f[i][j] = f[i + 1][j] - w[i][j]  
                if j + 1 < m:
                    f[i][j] = min(f[i][j], f[i][j + 1] - w[i][j])
                f[i][j] = max(1, f[i][j])
        return f[0][0]
      
# f 下标从 [1, 1] 开始      
class Solution:
    def calculateMinimumHP(self, w: List[List[int]]) -> int:
        n, m = len(w), len(w[0])
        f = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        f[n][m] = max(1, 1 - w[n - 1][m - 1])

        for i in range(n, 0, -1):
            for j in range(m, 0, -1):
                if i < n:
                    f[i][j] = f[i + 1][j] - w[i - 1][j - 1]
                if j < m:
                    f[i][j] = min(f[i][j], f[i][j + 1] - w[i - 1][j - 1])
                f[i][j] = max(f[i][j], 1)
        return f[1][1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 状态表示：f[i, j]表示 所有以(i,j)为右下角的且只包含 1 的正方形的边长最大值（属性）
>
> 2. 状态转移（计算）：
>
>    如果该位置的值是 0，则 f[i, j] = 0，因为当前位置不可能在由 1 组成的正方形中
>
>    如果该位置的值是 1，则 f[i, j]的值由其上方、左方和左上方的三个相邻位置的状态值决定。具体而言，当前位置的元素值等于三个相邻位置的元素中的最小值加 1，状态转移方程如下：f[i,j]= min(f[i − 1, j − 1], f[i − 1, j], f[i, j − 1]) + 1
>
>    为什么要三者取最小 +1 ？==> 木桶效应（取最短的边）

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;
        int n = matrix.size(), m = matrix[0].size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1));

        int res = 0;
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ )
                if (matrix[i - 1][j - 1] == '1') {
                    f[i][j] = min(f[i - 1][j], min(f[i][j - 1], f[i - 1][j - 1])) + 1;
                    res = max(res, f[i][j]);
                }

        return res * res;
    }
};
```

##### **Python**

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        n, m = len(matrix), len(matrix[0])
        f = [[0] * (m + 1) for _ in range(n + 1)]

        res = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if matrix[i - 1][j - 1] == '1':
                    f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
                    res = max(res, f[i][j])
        return res * res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2088. 统计农场中肥沃金字塔的数目](https://leetcode-cn.com/problems/count-fertile-pyramids-in-a-land/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> [biweekly-66](https://leetcode-cn.com/contest/biweekly-contest-66/)
> 
> 简单但经典
> 
> 类似最大正方形的递推思想 相对于前缀和极大降低复杂度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ dp递推 类似最大正方形**

```cpp
class Solution {
public:
    const static int N = 1010;

    int f[N][N];
    int n, m;
    vector<vector<int>> g;

    int work() {
        memset(f, -1, sizeof f);
        int ret = 0;
        for (int i = n - 1; i >= 0; -- i )
            for (int j = 0; j < m; ++ j )
                if (g[i][j]) {
                    if (i == n - 1|| j == 0 || j == m - 1)
                        f[i][j] = 0;
                    else {
                        f[i][j] = min({f[i + 1][j - 1], f[i + 1][j], f[i + 1][j + 1]}) + 1;
                        ret += f[i][j];
                    }
                }
        return ret;
    }

    int countPyramids(vector<vector<int>>& grid) {
        this->g = grid;
        n = g.size(), m = g[0].size();

        int res = work();
        reverse(g.begin(), g.end());
        res += work();
        return res;
    }
};
```

##### **C++ 前缀和+模拟**

```cpp
class Solution {
public:
    int n, m;
    vector<vector<int>> g;
    
    int check(int x, int y) {
        if (!g[x][y])
            return 0;
        int ret = 0;
        {
            int w = 1, c = 0;
            int nx = x, ny = y;
            while (nx <= n && ny <= m && g[nx][ny] >= w)
                nx ++ , ny ++ , w += 2, c ++ ;
            ret += c - 1;
        }
        {
            int w = 1, c = 0;
            int nx = x, ny = y;
            while (nx >= 1 && ny <= m && g[nx][ny] >= w)
                nx -- , ny ++ , w += 2, c ++ ;
            ret += c - 1;
        }
        return ret;
    }
    
    int countPyramids(vector<vector<int>>& grid) {
        n = grid.size(), m = grid[0].size();
        g = vector<vector<int>>(n + 1, vector<int>(m + 1));
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j )
                if (grid[i - 1][j - 1])
                    g[i][j] = g[i][j - 1] + 1;
        
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            for (int j = 1; j <= m; ++ j )
                res += check(i, j);
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

> [!NOTE] **[LeetCode 1277. 统计全为 1 的正方形子矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp 统计以每个点为右下角的正方形矩阵个数，该个数即为以该点为右下角的最大正方形边长(同LC221题)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countSquares(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        int res = 0;
        for (int i = 1; i <= m; ++i)
            for (int j = 1; j <= n; ++j)
                if (matrix[i - 1][j - 1] == 1) {
                    f[i][j] =
                        min(min(f[i - 1][j], f[i][j - 1]), f[i - 1][j - 1]) + 1;
                    res += f[i][j];
                }
        return res;
    }
;
```

##### **Python**

```python
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        n, m = len(matrix), len(matrix[0])
        f = [[0] * (m + 1) for _ in range(n + 1)]
        res = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if matrix[i - 1][j - 1] == 1:
                    f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
                    res += f[i][j]
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 514. 自由之路](https://leetcode-cn.com/problems/freedom-trail/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 优雅实现
>
> `if` 实现合法性判断
>
> 1. 状态表示
>
>    （比较难想，多做题）一般题意给出两个字符串就是二维
>
>    f[i, j] 表示**完成输出单词的前 i 个字母，且输出 key[i] 时，指针位于 ring[j]的所有方案**；属性：最小值
>
> 2. 状态转移
>
>    对于f[i, j] ，且当 key[i] == ring[j] 时，(最后一个位置是位于 j ），那以倒数第二个位置不同进行转移, 可以位于[1,2,..,k,..,..n]
>
>    考虑位于位置 k 的最小值。也就是考虑前 i - 1 个字母并且指针位于 ring[k] 的方案最小值，也就是f[i - 1, k] （后面两步是相通的：1）从 k 转到 j； 2）按按钮
>
> 3. 初始值：根据题意最初 ring 的第一个字符就是在正方向, **f[0, 1] = 0** 
> 4. 时间复杂度：状态数量是 O($n^2$), 转移数量是 m, 总的复杂度：O($mn^2$)



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findRotateSteps(string ring, string key) {
        int n = ring.size(), m = key.size();
        ring = ' ' + ring, key = ' ' + key;
        vector<vector<int>> f(m + 1, vector<int>(n + 1, 1e8));
        f[0][1] = 0;
        for (int i = 1; i <= m; i ++ )
            for (int j = 1; j <= n; j ++ )
                if (key[i] == ring[j]) {
                    for (int k = 1; k <= n; k ++ ) {
                        int t = abs(k - j);
                        f[i][j] = min(f[i][j], f[i - 1][k] + min(t, n - t) + 1);
                    }
                }

        int res = 1e8;
        for (int i = 1; i <= n; i ++ ) res = min(res, f[m][i]);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        n, m = len(ring), len(key)
        ring, key = ' ' + ring, ' ' + key
        f = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        f[0][1] = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if key[i] == ring[j]:
                    for k in range(1, n + 1):
                        # 可以顺时针转，也可以逆时针转
                        t = abs(k - j)
                        f[i][j] = min(f[i][j], f[i - 1][k] + min(t, n - t) + 1)
        
        res = float('inf')
        # 枚举指针位置，一定是要输出前 m 个字符
        for i in range(1, n + 1):
            res = min(res, f[m][i])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 741. 摘樱桃](https://leetcode-cn.com/problems/cherry-pickup/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 比摘花生稍复杂一点点 1A
>
> 假设两个人从左上角同时向右下角走，因为每个格子的樱桃只能被取一次，所以当两个人的路线重合时，我们只保证其中有一个人可以去到樱桃。
>
> 1. 状态表示: f[i, j, k]（一共走了 k 步，两个人的横坐标分别是 i, j,那么他们所在的坐标分别是(i, k - i), (j, k - j)），也就是**表示两个人走到(i, k - i), (j, k - j)所有路线的集合**；属性：最大值
>
> 2. 状态转移：这个集合可以分为几类呢？每个格子都有两个选择，两两一组合就有四个不同的组合方式：1）一起往右 2）一起往下 3）右 + 下 4）下 + 右。只要分别求每一类的最大值，然后再取这四类里的最大值即可。
>
> 3. 注意点：
>
>    1）如果两个人走到了同一个樱桃格子上，那樱桃数量只能加一次 
>
>    2）某些格子是不能走的
>
>    3）如果走 k 次取最大价值，不能用 dp 做，应该用最短路思想来做。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    const int INF = 2e9;
    int cherryPickup(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        
        vector<vector<vector<int>>> f(n + m + 1, vector<vector<int>>(n + 1, vector<int>(m + 1, -INF)));
        
        f[1][0][0] = 0;

        for (int k = 2; k <= n + m; ++ k )
            for (int x1 = 1; x1 <= k ; ++ x1 )
                for (int x2 = 1; x2 <= k; ++ x2 ) {
                    int y1 = k - x1, y2 = k - x2;
                    if (x1 < 1 || x1 > n || y1 < 1 || y1 > m)
                        continue;
                    if (x2 < 1 || x2 > n || y2 < 1 || y2 > m)
                        continue;
                    
                    if (grid[x1 - 1][y1 - 1] == -1 || grid[x2 - 1][y2 - 1] == -1)
                        continue;
                    
                    auto & t = f[k][x1][x2];
                    t = max(t, f[k - 1][x1][x2]);
                    t = max(t, f[k - 1][x1 - 1][x2]);
                    t = max(t, f[k - 1][x1][x2 - 1]);
                    t = max(t, f[k - 1][x1 - 1][x2 - 1]);
                    int v = 0;
                    if (grid[x1 - 1][y1 - 1] == 1)
                        ++ v ;
                    if (grid[x2 - 1][y2 - 1] == 1)
                        ++ v ;
                    if (x1 == x2 && v)
                        -- v ;
                    t += v;
                }
        return max(f[n + m][n][m], 0);
    }
};
```

##### **C++ 2**

```cpp
const int N = 55;
int f[N][N][N * 2];

class Solution {
public:
    int cherryPickup(vector<vector<int>>& grid) {
        int n = grid.size();
        memset(f, -0x3f, sizeof f);
        if (grid[0][0] != -1) f[1][1][2] = grid[0][0];
        for (int k = 3; k <= n * 2; k ++ )
            for (int i = max(1, k - n); i <= min(n, k - 1); i ++ )
                for (int j = max(1, k - n); j <= min(n, k - 1); j ++ ) {
                    if (grid[i - 1][k - i - 1] == -1 || grid[j - 1][k - j - 1] == -1) continue;
                    int t = grid[i - 1][k - i - 1];
                    if (i != j) t += grid[j - 1][k - j - 1];
                    for (int a = i - 1; a <= i; a ++ )
                        for (int b = j - 1; b <= j; b ++ )
                            f[i][j][k] = max(f[i][j][k], f[a][b][k - 1] + t);
                }
        return max(0, f[n][n][n * 2]);
    }
};
```

##### **Python**

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        N = 110
        f = [[[float('-inf')] * N for _ in range(N)] for _ in range(N)]
        n = len(grid)
        if grid[0][0] != -1:
            f[1][1][2] = grid[0][0]
        for k in range(1, 2 * n + 1):
            for i in range(max(1, k - n), min(n + 1, k - 1 + 1)):
                for j in range(max(1, k - n), min(n + 1, k - 1 + 1)):
                    if grid[i - 1][k - i - 1] == -1 or grid[j - 1][k - j - 1] == -1:continue
                    t = grid[i - 1][k - i - 1]
                    if i != j:
                        t += grid[j - 1][k - j - 1]
                    for a in range(i - 1, i + 1):
                        for b in range(j - 1, j + 1):
                            f[i][j][k] = max(f[i][j][k], f[a][b][k - 1] + t)
        return max(0, f[n][n][2 * n])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1463. 摘樱桃 II](https://leetcode-cn.com/problems/cherry-pickup-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> dp 注意非法的状态转移
>
> 1. 状态表示：f[l, i, j]表示第一个机器人从 (0, 0) 走到 (l, i), 第二个机器人从 (0, 0) 走到 (l, j)时，收集的樱桃数；属性是：最大值
>
> 2. 状态转移：以走到第 l - 1行时，机器人所在的位置不同进行划分。机器人1和2都有三种可能，所以两两组合，一共有9种组合。
>
>    f[l, i, j] = max(f[l - 1, i - 1, j - 1], f[l - 1, i - 1, j], f[l - 1, i - 1, j + 1],...) + grid[l, i] + grid[l, j]
>
> 3. 初始状态:一开始在第 0 行，第一，二个机器人分别在第 0，m - 1 列:
>
>    f[0, 0, m-1] = grid[0, m - 1] + grid[0, 0]

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int dx[3] = {-1, 0, 1};
    int cherryPickup(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<vector<int>>> dp(
            m + 1, vector<vector<int>>(n + 1, vector<int>(n + 1, -1)));
        dp[0][0][n - 1] = grid[0][0] + grid[0][n - 1];
        for (int i = 1; i < m; ++i)
            for (int l = 0; l < n; ++l)
                for (int r = l + 1; r < n; ++r)
                    for (int d1 = 0; d1 < 3; ++d1)
                        for (int d2 = 0; d2 < 3; ++d2) {
                            int p1 = l + dx[d1], p2 = r + dx[d2];
                            if (p1 < 0 || p1 >= n || p2 < 0 || p2 >= n)
                                continue;
                            if (dp[i - 1][p1][p2] < 0) continue;  // 重要
                            dp[i][l][r] =
                                max(dp[i][l][r], dp[i - 1][p1][p2] +
                                                     grid[i][l] + grid[i][r]);
                        }

        int res = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) res = max(res, dp[m - 1][i][j]);
        return res;
    }
};
```

##### **Python I**

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        # 踩坑：这里 f 初始化为 -1， 表示这个状态不合法（因为机器人始终每一步都只能在下一行往左或者往右一步，下一行往左两步及以上都是不可达的）
        f = [[[-1 for _ in range(m + 1)] for _ in range(m + 1)] for _ in range(n + 1)]
        f[0][0][m - 1] = grid[0][0] + grid[0][m - 1]
        dx = [-1, 0, 1]

        for l in range(1, n):
            for i in range(m):
                # j 是从 i + 1 开始遍历的，所以两个机器人不会走到同一个格子上
                for j in range(i + 1, m):
                    for d1 in range(3):
                        for d2 in range(3):
                            p1, p2 = i + dx[d1], j + dx[d2]
                            if p1 < 0 or p1 >= m or p2 < 0 or p2 >= m:
                                continue
                            if f[l - 1][p1][p2] == -1:continue # 不合法
                            f[l][i][j] = max(f[l][i][j], f[l - 1][p1][p2] + grid[l][i] + grid[l][j])

        res = 0
        for i in range(m):
            for j in range(i + 1, m):
                res = max(res, f[n - 1][i][j])
        return res
```



**Python II** 

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        f = [[[0 for i in range(m + 1)] for i in range(m + 1)] for i in range(n + 1)]
        f[0][0][m - 1] = grid[0][-1] + grid[0][0]
        maxn = 0
        for l in range(1, n):
            for i in range(m):
                for j in range(m):
                    # 不能超过对角线（trick的排除不合法的情况）
                    if i > l: continue
                    if m - 1 - j > l: continue
                    maxnum = 0
                    for ii in range(-1, 2):
                        for jj in range(-1, 2):
                            if i - ii >= 0 and j - jj >= 0:
                                maxnum = max(maxnum, f[l - 1][i - ii][j - jj])
                    if i != j:
                        f[l][i][j] = maxnum + grid[l][i] + grid[l][j]
                    else:
                        f[l][i][j] = maxnum + grid[l][i]
                    maxn = max(maxn, f[l][i][j])
        return maxn

```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1301. 最大得分的路径数目](https://leetcode-cn.com/problems/number-of-paths-with-max-score/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> trick **使用 PII 来同时记录数值和方案**
>
> 1. 状态表示：f[i, j, 2] 表示从右下角到 (i, j) 的最大得分和方案数；属性：最大得分，以及最大得分对应的方案数
>
> 2. 状态转移： 位置 (i, j) 可以由 (i + 1, j), (i + 1, j + 1) 和 (i, j + 1) 三个位置转移过来。
>
> 3. 如果右下角无法到达位置(i, j)，那么f[i, j] = [-1, 0]
>
>    1）位置 (i, j) 有障碍
>
>    2）由于有障碍，无法到达位置 (i, j)
>
> 4. 初始化：f[n - 1, n - 1] = [0, 1] 开始是在右下角，得分为0的方案合法，并且为1



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using PII = pair<int, int>;

class Solution {
private:
    static constexpr int mod = (int)1e9 + 7;

public:
    void update(vector<vector<PII>>& dp, int n, int x, int y, int u, int v) {
        if (u >= n || v >= n || dp[u][v].first == -1) {
            return;
        }
        if (dp[u][v].first > dp[x][y].first) {
            dp[x][y] = dp[u][v];
        }
        else if (dp[u][v].first == dp[x][y].first) {
            dp[x][y].second += dp[u][v].second;
            if (dp[x][y].second >= mod) {
                dp[x][y].second -= mod;
            }
        }
    }

    vector<int> pathsWithMaxScore(vector<string>& board) {
        int n = board.size();
        vector<vector<PII>> dp(n, vector<PII>(n, {-1, 0}));
        dp[n - 1][n - 1] = {0, 1};
        for (int i = n - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                if (!(i == n - 1 && j == n - 1) && board[i][j] != 'X') {
                    update(dp, n, i, j, i + 1, j);
                    update(dp, n, i, j, i, j + 1);
                    update(dp, n, i, j, i + 1, j + 1);
                    if (dp[i][j].first != -1) {
                        dp[i][j].first += (board[i][j] == 'E' ? 0 : board[i][j] - '0');
                    }
                }
            }
        }
        return dp[0][0].first == -1 ? vector<int>{0, 0} : vector<int>{dp[0][0].first, dp[0][0].second};
    }
};
```

##### **Python-I**

```python
"""
把状态转移的过程抽象出一个 update 函数
"""      
class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        n = len(board)
        mod = int(1e9 + 7)
        f = [[[-1, 0] for _ in range(n + 1)] for _ in range(n + 1)]
        f[n - 1][n - 1] = [0, 1]

        def update(i, j, x, y):
            if 0 <= x < n and 0 <= y < n and f[x][y][0] != -1:
                if f[x][y][0] > f[i][j][0]:
                    f[i][j] = f[x][y][:]
                elif f[x][y][0] == f[i][j][0]:
                    f[i][j][1] += f[x][y][1]
                    f[i][j][1] %= mod

        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if board[i][j] != 'X':
                    update(i, j, i + 1, j)
                    update(i, j, i, j + 1)
                    update(i, j, i + 1, j + 1)
                    if f[i][j][0] != -1:
                        if board[i][j].isdigit() == True:
                            f[i][j][0] += int(board[i][j])
        if f[0][0][0] == -1:
            return [0, 0]
        else:
            return f[0][0]
```



**Python-II**

```python

'''
简单记忆化递归进行动态规划
'''


from typing import List
from functools import lru_cache

class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        m, n = len(board), len(board[0])

        # 从ii, jj位置到起点的路径最大和以及方法数
        @lru_cache(typed=False, maxsize=128000000)
        def dp(ii, jj):
            if ii < 0 or ii >= m or jj < 0 or jj >= n:
                return (0, 0)

            if ii == 0 and jj == 0:
                return (0, 1)

            if board[ii][jj] == 'X':
                return (0, 0)

            ans1 = dp(ii, jj-1)
            ans2 = dp(ii-1, jj)
            ans3 = dp(ii-1, jj-1)

            max_score = max(ans1[0], ans2[0], ans3[0])
            ans = [max_score + ord(board[ii][jj]) - ord('0'), 0]
            if ii == m-1 and jj == n-1:
                ans[0] = max_score

            for length, cnt in [ans1, ans2, ans3]:
                if length == max_score:
                    ans[1] += cnt

            if ans[1] == 0:
                ans[0] = 0
            ans[1] %= 1000000007

            return ans

        return dp(m-1, n-1)
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1320. 二指输入的的最小距离](https://leetcode-cn.com/problems/minimum-distance-to-type-a-word-using-two-fingers/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 状态表示 f[i, l, r] 表示走到字符串第 i 个字母的时候，左手的位置在 l, 右手的位置在 r 时的距离；属性：min
>
> 2. 状态转移：对于 f[i, l, r] 来说，有一个重要的性质，在走到第 i 个字母的时候，这个时候要么左手，要么右手是在 word[i] 位置上的。
>
>    1）如果第 i 字母是左手在 word[i] 位置上：那就去看第 i - 1 个字母，去遍历左手和右手的所有情况，最后再加上第 i 个字母，左手从上一个位置到 i 字母的距离
>
>    2）如果第 i 字母是右手在 word[i] 位置上：同上，只是最后加上的就是右手从上一个位置到 i 字母的距离
>
>    最后的答案是：f[n, l, r] 中的最小值，因为 l 和 r 的位置并不是固定在哪就是最小
>
> 3. 初始化：一开始 l 和 r 的起始位置是领代价，所以不管 l 和 r 一开始在哪，
>
>    ​	f[0, l, r] = 0
>
> 4. 我们在转移过程中，还需要求从某个字母到另外一个字母的距离，这里就把二维的表转化为了一维，方便求距离
>
>    
>
> **经典 压掉一维状态**
>
> 减少一维：
>
> >   重新设计一种新的状态：**字母下标**（可以代表**第一个**指头键位），**另外一个指头的键位**。
> >
> >   每次按下一个字母时，要么是字母下标所在的指头（**第一个指头**）移动，要么是**另外一个指头**移动。
> > 
> >   因此我们可以直接使用 dp[i][rest] 进行状态转移，其表示一只手在 word[i] 的位置，另一只手在 rest 的位置的最小移动距离。
> >
> >   我们并不需要关心具体哪只手在 word[i] 的位置，因为两只手是完全对称的。
> >
> >   这样一来，我们将三维的动态规划优化至了二维，大大减少了空间的使用。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 300ms**

```cpp
class Solution {
public:
    int minimumDistance(string word) {
        int n = word.size();
        vector<vector<int>> c(26, vector<int>(26));
        for (int i = 0; i < 26; ++i) {
            int x1 = i / 6, y1 = i % 6;
            for (int j = 0; j < 26; ++j) {
                int x2 = j / 6, y2 = j % 6;
                c[i][j] = c[j][i] = abs(x1 - x2) + abs(y1 - y2);
            }
        }
        vector<vector<vector<int>>> f(
            n + 1, vector<vector<int>>(26, vector<int>(26, INT_MAX / 2)));
        for (int i = 0; i < 26; ++i)
            for (int j = 0; j < 26; ++j) f[0][i][j] = 0;
        for (int i = 1; i <= n; ++i) {
            int x = word[i - 1] - 'A';
            for (int j = 0; j < 26; ++j)
                for (int k = 0; k < 26; ++k) {
                    f[i][j][x] = min(f[i][j][x], f[i - 1][j][k] + c[k][x]);
                    f[i][x][k] = min(f[i][x][k], f[i - 1][j][k] + c[j][x]);
                }
        }
        int res = INT_MAX;
        for (int i = 0; i < 26; ++i)
            for (int j = 0; j < 26; ++j) res = min(res, f[n][i][j]);
        return res;
    }
};
```

##### **C++ 压缩 4ms**

```cpp
class Solution {
public:
    int getDistance(int p, int q) {
        int x1 = p / 6, y1 = p % 6;
        int x2 = q / 6, y2 = q % 6;
        return abs(x1 - x2) + abs(y1 - y2);
    }

    int minimumDistance(string word) {
        int n = word.size();
        int dp[n][26];
        fill(&dp[0][0], &dp[0][0] + n * 26, INT_MAX >> 1);
        fill(&dp[0][0], &dp[0][0] + 26, 0);

        for (int i = 1; i < n; ++i) {
            int cur = word[i] - 'A';
            int prev = word[i - 1] - 'A';
            int d = getDistance(prev, cur);
            for (int j = 0; j < 26; ++j) {
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + d);
                if (prev == j) {
                    for (int k = 0; k < 26; ++k) {
                        int d0 = getDistance(k, cur);
                        dp[i][j] = min(dp[i][j], dp[i - 1][k] + d0);
                    }
                }
            }
        }

        int ans = *min_element(dp[n - 1], dp[n - 1] + 26);
        return ans;
    }
};
```

##### **Python**

```python
class Solution:
    def minimumDistance(self, word: str) -> int:
        n = len(word)
        f = [[[float('inf')] * 26 for _ in range(26)] for _ in range(n + 1)]

        # 二维化为一维 求距离
        c = [[0] * 26 for _ in range(26)]
        for i in range(26):
            x1, y1 = i // 6, i % 6
            for j in range(26):
                x2, y2 = j // 6, j % 6
                c[i][j] = c[j][i] = abs(x1 - x2) + abs(y1 - y2)
        
        # 初始值
        for l in range(26):
            for r in range(26):
                f[0][l][r] = 0
        
        for i in range(1, n + 1):
            x = ord(word[i - 1]) - ord('A')
            for l in range(26):
                for r in range(26):
                    f[i][l][x] = min(f[i][l][x], f[i - 1][l][r] + c[r][x])
                    f[i][x][r] = min(f[i][x][r], f[i - 1][l][r] + c[l][x])

        res = float('inf')
        for l in range(26):
            for r in range(26):
                res = min(res, f[n][l][r])
        return res
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
> 最优化的问题 从“集合”的角度进行考虑
>
> 在原序列中，满足“先递增再递减的子序列”里找到长度最长的子序列

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
> 所有合法的建桥方式； 上升子序列 ； 航道交叉也就是a1 >= a2 && b1 <= b2(其中a, b表示两岸)
>
> 方法：1. 通过排序，固定自变量的顺序； 2. 找到因变量的最大上升子序列

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

> [!NOTE] **[AcWing 314. 低买](https://www.acwing.com/problem/content/316/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **LIS变形 细节清奇**
> 
> **严格下降子序列 求数值不同的所有方案数**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5010;

int n;
int a[N], f[N], g[N];
// f 所有以a[i]结尾的下降子序列的集合的最大长度
// g 数量

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i ) cin >> a[i];
    g[0] = 1;
    
    int res = 0;
    for (int i = 1; i <= n; ++ i ) {
        for (int j = 0; j < i; ++ j )
            if (!j || a[j] > a[i])
                f[i] = max(f[i], f[j] + 1);
        // 考虑 对有相同数字、不同位置方案的情况
        // 只考虑最后一个位置上的方案
        for (int j = 1; j < i; ++ j )
            if (a[j] == a[i])
                f[j] = -1;      // 删除前面的状态
        for (int j = 0; j < i; ++ j )
            if ((!j || a[j] > a[i]) && f[i] == f[j] + 1)
                g[i] += g[j];
        res = max(res, f[i]);
    }
    
    int cnt = 0;
    for (int i = 1; i <= n; ++ i )
        if (f[i] == res)
            cnt += g[i];
    cout << res << ' ' << cnt << endl;
    
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

> [!NOTE] **[LeetCode 300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> **普通 dp**
>
> 1. 状态表示：f[i] 表示以第 i 个字符为结尾的上升子序列长度；属性： max
>
> 2. 状态转移：以上一个字符是否能加入到上升序列切分。
>
>    只有当 nums[i - 1] < nums[i], 那 f[i] = f[i - 1] + 1
>
> 3. 初始化：所有的数字本身都可以构成长度为1的上升子序列，所以f[i] = 1
>
> 4. 时间复杂度：$O(n^2)$
>
> **二分优化**
>
> 1. 需要一个列表 p 来存储所有不同长度的上升子序列的结尾的最小值。p[i] 表示上升子序列长度为 i 的所有序列里末尾最小数的值。
>
> 2. 对于当前数 a[i], 通过二分先在列表 p 中找到小于当前数 a[i] 的最大数，假设是 p[r]，然后把 a[i] 接到 p[r] 后面，就可以构成以 a[i] 为结尾的最长上升子序列
>
>    接着，更新列表 p: p[r + 1] = a[i] 因为p[r + 1] 表示的长度和 a[i]此时形成的最长上升子序列长度一致，但是 a[i] < p[r + 1]，而 p 存储的是长度为 i 的所有序列里末尾最小的数。所以更新为 a[i].
>
>    这样的好处是，对于后续未知的数来说，可以接在 a[i] (比a[i]大，就可以接在后面，使得当前所有最长上升子序列的长度+1) 的数字是比接在 p[r + 1] 后面的范围更大的。
>
> 3. 总的来说，就是直接在列表 p 中找到第一个比 a[i] 大的数 p[l]，然后用 a[i] 替换掉这个数即可。最后返回列表 p 的长度。
>
> 4. 时间复杂度: $O(nlogn)$
>
>    
>
>    我们现在的贪心思想就是，数组p里要保留末尾最小的元素，这样可以获得子序列最长的答案，也就是最优解。
>
>    反证法：假设在最优解里，数组p里保留的不是末尾最小的元素。
>
>    比如数组[2,5,3,4];
>
>    1. 当前（2，5） （2，3）这两个子序列长度都是2，如果p数组记录的不是最小值，那么 p[2] = 5；
>    2. 那么对于数字4，由于比数字5小，所以没有办法接到5后面形成一个上升子序列，那此时就会认为整个数组的最长上升子序列长度就是 2。
>    3. 但这个很明显不是最优解（最长上升子序列长度是3），所以假设不成立



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> f;
        for (int i = 0; i < n; ++ i )
            if (f.empty() || f.back() < nums[i])
                f.push_back(nums[i]);
            else
                *lower_bound(f.begin(), f.end(), nums[i]) = nums[i];
        return f.size();
    }
};
```

##### **C++ 手写二分**

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> f;
        for (auto x : nums) {
            int l = 0, r = f.size();
            while (l < r) {
                int m = l + r >> 1;
                if (f[m] < x)
                    l = m + 1;
                else
                    r = m;
            }
            if (l == f.size())
                f.push_back(x);
            else
                f[l] = x;
        }
        return f.size();
    }
};
```

##### **Python**

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:return 0
        n = len(nums)
        f = [1] * (n + 1)
        res = 1
        for i in range(1, n + 1):
            for j in range(1, i):
                if nums[i - 1] > nums[j - 1]:
                    f[i] = max(f[i], f[j] + 1)
            res = max(res, f[i])
        return res
```



**Python-二分优化（bisect）**

```python
import bisect
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        p = []
        for v in nums:
            idx = bisect.bisect_left(p, v)
            if idx == len(p):
                p.append(v)
            else:
              p[idx] = v
        return len(p)
        
```



**Python - 二分优化**

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:return 0
        n = len(nums)
        p = []

        for v in nums:
          # 如果 p == null or 列表 p 的最后一个数比当前数小，直接把当前数插入
            if not p or v > p[-1]:
                p.append(v)
            else:
                l, r = 0, len(p)
                while l < r:
                    m = l + (r - l) // 2
                    if p[m] < v:
                        l = m + 1
                    else:r = m
                # 由于当前数比所有数都大，会直接append，所以没有数组越界问题
                p[l] = v
        return len(p) 
```





<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> **二分优化**
>
> 1. 思路同一维的最长上升子序列的二分优化方法，这是二维的。
>
> 2. 进行排序，对原序列的e[0]升序，e[1]降序（
>
>    e[1]降序的原因：当有 [2,2], [2,3]这样的序列，如果e[1]升序的话，那[2,3]就会被接在[2,2]后面，但这样实际上是不合法的。所以给e[1]降序可以避免这个问题
>
> 3. 列表 p 存储的是不同长度的上升子序列的结尾的最小值。p[i] 表示上升子序列长度为 i 的所有序列里末尾最小数的值的第二维度的数值。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        int n = envelopes.size(), res = 0;
        sort(envelopes.begin(), envelopes.end(), [](const auto& a, const auto& b){
            return a[0] == b[0] ? a[1] > b[1] : a[0] < b[0];
        });
        vector<int> f;
        for (auto & e : envelopes)
            if (f.empty() || f.back() < e[1])
                f.push_back(e[1]);
            else
                *lower_bound(f.begin(), f.end(), e[1]) = e[1];
        return f.size();
    }
};
```

##### **C++ 暴力**

```cpp
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& w) {
        int n = w.size();
        sort(w.begin(), w.end());
        vector<int> f(n);

        int res = 0;
        for (int i = 0; i < n; i ++ ) {
            f[i] = 1;
            for (int j = 0; j < i; j ++ )
                if (w[j][0] < w[i][0] && w[j][1] < w[i][1])
                    f[i] = max(f[i], f[j] + 1);
            res = max(res, f[i]);
        }

        return res;
    }
};
```



**Python bisect**

```python
import bisect
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        envelopes.sort(key=lambda x:(x[0], -x[1]))

        p = []
        for e in envelopes:
            idx = bisect.bisect_left(p, e[1])
            if idx == len(p):
                p.append(e[1])
            else:
                p[idx] = e[1]
        return len(p)
```



##### **Python 二分优化**

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        envelopes.sort(key=lambda x:(x[0], -x[1]))

        p = []
        for e in envelopes:
            # 如果 p == null or 列表p的最后一个数比当前数小，直接把当前数插入
            if not p or e[1] > p[-1]:
                p.append(e[1])
            else:
                # 二分找到第一个比当前数大的数
                l, r = 0, len(p)
                while l < r:
                    m = l + (r - l) // 2
                    if p[m] < e[1]:
                        l = m + 1
                    else:
                        r = m
                # 由于当前数比所有数都大，会直接append，所以没有数组越界问题
                p[l] = e[1]
        return len(p)
```



**Python dp**

```python
# 暴力会超时
class Solution:
    def maxEnvelopes(self, e: List[List[int]]) -> int:
        n = len(e)
        e.sort()
        f = [1] * n

        res = 1
        for i in range(n):
            for j in range(i):
                if e[j][0] < e[i][0] and e[j][1] < e[i][1]:
                    f[i] = max(f[i], f[j] + 1)
            res = max(res, f[i])
        return res
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1626. 无矛盾的最佳球队](https://leetcode-cn.com/problems/best-team-with-no-conflicts/)**
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
    // 最长上升子序列模型
    int bestTeamScore(vector<int>& scores, vector<int>& ages) {
        vector<pair<int, int>> c;
        int n = scores.size();
        for (int i = 0; i < n; ++i) c.push_back({ages[i], scores[i]});
        sort(c.begin(), c.end());
        int res = 0;
        vector<int> f(n + 1);
        for (int i = 1; i <= n; ++i) {
            auto [aa, as] = c[i - 1];
            f[i] = as;
            for (int j = 1; j < i; ++j) {
                auto [ba, bs] = c[j - 1];
                if (ba <= aa && bs <= as) f[i] = max(f[i], f[j] + as);
            }
            res = max(res, f[i]);
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        n = len(scores)
        nums = list(zip(ages, scores))
        nums.sort(key=lambda x: (x[0], x[1]))
        f = [nums[i][1] for i in range(n)]
        res = 0
        for i in range(n):
            for j in range(i):
                if nums[i][1] >= nums[j][1]:
                    f[i] = max(f[i], f[j] + nums[i][1])
            res = max(res, f[i])
        return res
```



**Python-二分优化**

```python
```





<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1671. 得到山形数组的最少删除次数](https://leetcode-cn.com/problems/minimum-number-of-removals-to-make-mountain-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 简单 LIS
>
> 假设最优解的中心是第k座山，那么T1，T2,..,Ti一定是以Ti结尾的最长上升子序列，同理Tk, Tk-1,..Ti也是以Ti结尾的最长上升子序列。
>
> 对于每座山，先预处理出：
>
> 1. 从前往后以每个点结尾的最长上升子序列长度f[i]
> 2. 从后往前以每个点结尾的最长上升子序列长度g[i]
>
> 那么以k为中心的最大长度就是：f[k]+g[k]-1，遍历k=[1,n]，取最大值即可
>
> 必须有上升有下降 所以不统计 1 和 n 的位置， WA 1
>
> > 关于不使用两头的元素 其实更好的判断是
> > 
> > ```cpp
> > if (fu[i] >= 2 && fd[i] >= 2)
> >     res = max(res, fu[i] + fd[i] - 1);
> > ```

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
ass Solution {
public:
    int minimumMountainRemovals(vector<int>& nums) {
        int n = nums.size();
        vector<int> fu(n + 1), fd(n + 1);
        for (int i = 1; i <= n; ++ i ) {
            fu[i] = 1;
            for (int j = 1; j < i; ++ j )
                if (nums[j - 1] < nums[i - 1]) fu[i] = max(fu[i], fu[j] + 1);
        }
        for (int i = n; i >= 1; -- i ) {
            fd[i] = 1;
            for (int j = n; j > i; -- j )
                if (nums[j - 1] < nums[i - 1]) fd[i] = max(fd[i], fd[j] + 1);
        }
        int res = 0;
        for (int i = 2; i <= n - 1; ++ i )
            if (fu[i] > 1 && fd[i] > 1)
                res = max(res, fu[i] + fd[i] - 1);
        return n - res;
    }
};
```

##### **Python**

```python
class Solution:
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        f, g = [1] * (n + 1), [1] * (n + 1)
        for i in range(1, n + 1):
            for j in range(1, i):
                if nums[j - 1] < nums[i - 1]:
                    f[i] = max(f[i], f[j] + 1)
        for i in range(n, 0, -1):
            for j in range(n, i, -1):
                if nums[j - 1] < nums[i - 1]:
                    g[i] = max(g[i], g[j] + 1)
        res = 0
        for i in range(2, n):
            if f[i] > 1 and g[i] > 1:
                res = max(res, f[i] + g[i] - 1)
        return n - res
```

**Python-二分优化**

```python
```



<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1840. 最高建筑高度](https://leetcode-cn.com/problems/maximum-building-height/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> LIS + 离散化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    // 先上升再下降 LIS问题
    // 直接求俩LIS超时 注意数据范围是1e9
    //
    // restrictions 长度 考虑可以离散化
    using LL = long long;
    unordered_map<int, LL> limit;
    vector<LL> h1, h2, h;
    vector<int> ids;
    
    int get(int i1, int i2) {
        int id1 = ids[i1], id2 = ids[i2];
        int len = id2 - id1 + 1;            // 向上向下总计多少个
        int dis = abs(h[i1] - h[i2]) + 2;   // 向下多少个
        return (len - dis + 1) / 2 + max(h[i1], h[i2]);
    }
    
    int maxBuilding(int n, vector<vector<int>>& restrictions) {
        // 无限制要求下 位置x的最大高度为x-1
        ids.push_back(1);       // case
        ids.push_back(n + 1);   // case
        for (auto & r : restrictions) {
            int id = r[0], mxh = r[1];
            limit[id] = min(id - 1, mxh);
            ids.push_back(id);
        }
        sort(ids.begin(), ids.end());
        int l = ids.size() - 2;     // rest_id = [1, l]
        h1 = h2 = h = vector<LL>(l + 2);
        
        h1[0] = 0;
        for (int i = 1; i <= l; ++ i ) {
            int id = ids[i];
            h1[i] = min(h1[i - 1] + id - ids[i - 1], limit[id]);
        }
        
        h2[l + 1] = 2e9;
        for (int i = l; i >= 1; -- i ) {
            int id = ids[i];
            h2[i] = min(h2[i + 1] + ids[i + 1] - id, limit[id]);
        }
        
        for (int i = 1; i <= l; ++ i )
            h[i] = min(h1[i], h2[i]);
        
        int res = 0;
        if (!l)
            res = n - 1;
        else
            res = n - ids[l] + h[l];
        for (int i = 1; i <= l; ++ i )
            res = max(res, get(i -  1, i));
        
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int maxBuilding(int n, vector<vector<int>>& h) {
        typedef long long LL;
        h.push_back({1, 0});
        sort(h.begin(), h.end());
        if (h.back()[0] != n) h.push_back({n, n - 1});
        int m = h.size();
        vector<LL> f(m + 1, INT_MAX), g(m + 1, INT_MAX);
        f[0] = -1;
        for (int i = 1; i < m; i ++ ) {
            int x = h[i][0], y = h[i][1];
            f[i] = min(f[i - 1], (LL)y - x);
        }
        for (int i = m - 1; i >= 0; i -- ) {
            int x = h[i][0], y = h[i][1];
            g[i] = min(g[i + 1], (LL)y + x);
        }
        LL res = 0;
        for (int i = 0; i < m; i ++ ) {
            int x = h[i][0];
            if (i) {
                LL Y = (f[i - 1] + g[i]) / 2;
                LL X = Y - f[i - 1];
                if (X >= h[i - 1][0] && X <= h[i][0])
                    res = max(res, Y);
            }
            res = max(res, min(x + f[i], -x + g[i]));
        }
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

> [!NOTE] **[LeetCode 1964. 找出到每个位置为止最长的有效障碍赛跑路线](https://leetcode-cn.com/problems/find-the-longest-valid-obstacle-course-at-each-position/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> LIS 优化的应用 略

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obs) {
        int n = obs.size();
        vector<int> res(n), f;
        for (int i = 0; i < n; ++ i ) {
            int v = obs[i];
            if (f.empty() || f.back() <= v) {
                res[i] = f.size() + 1;
                f.push_back(v);
            } else {
                // 第一个大于该值
                auto it = upper_bound(f.begin(), f.end(), v);
                res[i] = it - f.begin() + 1;
                *it = v;
            }
        }
        return res;
    }
};
```

##### **C++ 比赛代码**

```cpp
// 比赛时
class Solution {
public:
    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obs) {
        int n = obs.size();
        vector<int> res(n), f;
        for (int i = 0; i < n; ++ i ) {
            int v = obs[i];
            if (f.empty() || f.back() <= v) {
                res[i] = f.size() + 1;
                f.push_back(v);
            } else {
                auto it = lower_bound(f.begin(), f.end(), v + 1);
                res[i] = it - f.begin() + 1;
                *it = v;
            }
        }
        return res;
    }
};
```

##### **C++ 赛榜**

```cpp
class Solution {
public:
    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {
        int n = obstacles.size();
        vector<int> ans, LIS;
        for (int obstacle : obstacles) {
            auto it = upper_bound(LIS.begin(), LIS.end(), obstacle);
            if (it == LIS.end()) {
                LIS.emplace_back(obstacle);
                ans.emplace_back(LIS.size());
            } else {
                ans.push_back(it - LIS.begin() + 1);
                *it = obstacle;
            }
        }
        return ans;
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

> [!NOTE] **[LeetCode 2111 使数组 K 递增的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-the-array-k-increasing/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> LIS 应用

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int count(vector<int> & vs) {
        vector<int> f;
        for (auto x : vs)
            if (f.empty() || f.back() <= x)
                f.push_back(x);
            else
                *lower_bound(f.begin(), f.end(), x + 1) = x;
        return vs.size() - f.size();
    }
    
    int kIncreasing(vector<int>& arr, int k) {
        vector<vector<int>> g(k);
        int n = arr.size();
        for (int i = 0; i < n; ++ i )
            g[i % k].push_back(arr[i]);
        
        int res = 0;
        for (int i = 0; i < k; ++ i )
            res += count(g[i]);
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

### 类 LIS 思想

> [!NOTE] **[LeetCode 368. 最大整除子集](https://leetcode-cn.com/problems/largest-divisible-subset/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    vector<int> largestDivisibleSubset(vector<int>& w) {
        if (w.empty()) return {};
        sort(w.begin(), w.end());
        int n = w.size();
        vector<int> f(n);

        int k = 0;
        for (int i = 0; i < n; i ++ ) {
            f[i] = 1;
            for (int j = 0; j < i; j ++ )
                if (w[i] % w[j] == 0)
                    f[i] = max(f[i], f[j] + 1);
            if (f[k] < f[i]) k = i;
        }

        vector<int> res(1, w[k]);
        while (f[k] > 1) {
            for (int i = 0; i < k; i ++ )
                if (w[k] % w[i] == 0 && f[k] == f[i] + 1) {
                    res.push_back(w[i]);
                    k = i;
                    break;
                }
        }
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    vector<int> largestDivisibleSubset(vector<int>& nums) {
        int n = nums.size(), maxl = 0, end = -1;
        sort(nums.begin(), nums.end());
        vector<int> f(n, 1), last(n, -1), res;
        for (int i = 0; i < n; ++ i ) {
            for (int j = 0; j < i; ++ j ) {
                if (nums[i] % nums[j] == 0 && f[i] <= f[j]) {
                    f[i] = f[j] + 1;
                    last[i] = j;
                }
            }
            if (f[i] > maxl) {
                maxl = f[i];
                end = i;
            }
        }
        for (int i = end; i != -1; i = last[i])
            res.push_back(nums[i]);
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

> [!NOTE] **[LeetCode 1218. 最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/)**
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
    int longestSubsequence(vector<int>& arr, int difference) {
        unordered_map<int, int> hash;
        int res = 1, n = arr.size();
        vector<int> f(n);
        for (int i = 0; i < n; ++ i ) {
            int last = arr[i] - difference;
            f[i] = 1;
            if (hash.count(last))
                f[i] = max(f[i], f[hash[last]] + 1);
            hash[arr[i]] = i;
        }
        for (int i = 0; i < n; ++ i ) res = max(res, f[i]);
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

### LIS 与 LCS

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

> [!NOTE] **[LeetCode 583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1));
        for (int i = 1; i <= n; i ++ ) f[i][0] = i;
        for (int i = 1; i <= m; i ++ ) f[0][i] = i;
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ ) {
                f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
                if (word1[i - 1] == word2[j - 1])
                    f[i][j] = min(f[i][j], f[i - 1][j - 1]);
            }
        return f[n][m];
    }
};
```

##### **C++**

```cpp
// LCS
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n1 = word1.size(), n2 = word2.size();
        vector<vector<int>> f(n1 + 1, vector<int>(n2 + 1));
        for (int i = 1; i <= n1; ++ i )
            for (int j =1; j <= n2; ++ j )
                if (word1[i - 1] == word2[j - 1])
                    f[i][j] = f[i - 1][j - 1] + 1;
                else
                    f[i][j] = max(f[i - 1][j], f[i][j - 1]);
        return n1 + n2 - 2 * f[n1][n2];
    }
};
```

##### **Python**

```python
#也可以看成最长公共子序列问题===>删除字符的最小值 等价于 剩下字符串的最大值
#等价于1143（最长公共子序列问题）
#闫式分析法：
#状态表示：dp[i][j]：集合：使得s1(1~i)和s2(1-j)变成相同字符串的所有方案；属性：min
#状态计算：分类：根据最后一个字符的情况来氛围若干类===> 根据i和j是否有被剩下来进行分类：一共有四类
#1）i和j存在：那只需要关注dp[i-1][j-1]
#2）i存在，j不存在:相当于要将j删掉，dp[i][j-1]+1
#3）i不存在，j存在:相当于要将i删掉，dp[i-1][j]+1
#4）i和j都不存在:dp[i-1][j-1]+2===> 这种情况 会在第2类和第3类里被删掉。

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m,n=len(word1),len(word2)
        dp=[[0]*(n+1) for _ in range(m+1)]
        dp[0][0]=0
        for i in range(1,m+1):
            dp[i][0]=i
        for j in range(1,n+1):
            dp[0][j]=j
        for i in range(1,m+1):
            for j in range(1,n+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=min(dp[i-1][j],dp[i][j-1])+1
        return dp[-1][-1]
#代码和编辑距离72题一样...      
```

<!-- tabs:end -->
</details>

<br>

* * *

### 状态机模型

> [!NOTE] **[AcWing 1057. 股票买卖 IV](https://www.acwing.com/problem/content/description/1059/)**
> 
> 题意: k 次买卖

> [!TIP] **思路**
> 
> 注意 j 循环从 1 到 k，而非从 1 到 i

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100010, M = 110, INF = 0x3f3f3f3f;

int n, m;
int w[N];
int f[N][M][2];

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);

    memset(f, -0x3f, sizeof f);
    for (int i = 0; i <= n; i++) f[i][0][0] = 0;

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            f[i][j][0] = max(f[i - 1][j][0], f[i - 1][j][1] + w[i]);
            f[i][j][1] = max(f[i - 1][j][1], f[i - 1][j - 1][0] - w[i]);
        }

    int res = 0;
    for (int i = 0; i <= m; i++) res = max(res, f[n][i][0]);

    printf("%d\n", res);

    return 0;
}
```

##### **Python**

```python
# 如何把交易状态描述清楚
# 第一个状态：手中有货； ==> 1）可以持有；2）卖出
# 第二个状态：手中没有货； ==> 1) 不买，就继续是没货；2）第二天买入，就是持有状态
# 状态转移的时候 是有权重的，+ w[i], - w[1]

# 状态表示：f[i, j, 0] : 前i天，已经做完j次交易，并且手中无货的购买方式的集合
#           f[i, j, 1] : 前i天，已经做完前j-1次交易，并且正在进行第j次交易，并且手中有货的购买方式的集合 ！ 注意：这里是正在进行第j次交易
# 状态机的状态表示，实质上是把i的状态进行了，方便后续状态计算； 属性：最大值
# 状态计算：就是状态机的转移

# 注意：
# 1. 初始化的问题：f[i,0,0]表示进行0次交易 手中无货的情况，那就是0，表示这个状态合法，可以从这个状态转移过来；状态不合法的时候，就要初始化无穷大
#    求最大，就初始化为负无穷；求最小，就初始化为最大，表示为：状态不合法，没办法从这个状态转移过来
# 2. 最后的结果输出问题：最后一定是进行了若干次完整的交易，手中无货才是完整交易（买了不卖，不是最优解，买要花钱）

# N = 100010
# f = [[[float('-inf') * 2] for _ in range(N)] for _ in range(N)]
# w = [0] * N

# if __name__ =='__main__':
#     n, m = map(int, input().split())
#     w[1:] = list(map(int, input().split()))
#     for i in range(1, n+1):  # 初始化很重要
#         f[i][0][0] = 0 

#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             f[i][j][0] = max(f[i - 1][j][0], f[i - 1][j][1] + w[i])  #上一次有货，需要卖出，并不需要开启新的交易；所以上一次无货就处于第j次交易中
#             f[i][j][1] = max(f[i - 1][j][1], f[i - 1][j - 1][0] - w[i]) # 上一次无货，需要买入，那么就会开启一个新的交易；所以上一次无货处于第j-1次交易

#     res = 0 
#     for i in range(m + 1):
#         res = max(res, f[n][i][0])
#     print(res)


# 空间压缩，由于第i项完全依赖于i-1项，所以j从大到小 or 从小到大 遍历都是可以的
# 当第i项 依赖于i-1和i项时，j的遍历方向才有影响
N = 100010
f = [[float('-inf')] * 2 for _ in range(N)]
w = [0] * N

if __name__ == '__main__':
    n, m = map(int, input().split())
    w[1:] = list(map(int, input().split()))
    f[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            f[j][0] = max(f[j][0], f[j][1] + w[i])
            f[j][1] = max(f[j][1], f[j - 1][0] - w[i])
    res = 0
    for i in range(1, m + 1):
        res = max(res, f[i][0])
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1058. 股票买卖 V](https://www.acwing.com/problem/content/description/1060/)**
> 
> 题意: 无限买卖含冷冻期

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100010, INF = 0x3f3f3f3f;

int n;
int w[N];
int f[N][3];

int main() {
    scanf("%d", &n);

    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);

    f[0][0] = f[0][1] = -INF, f[0][2] = 0;
    for (int i = 1; i <= n; i++) {
        f[i][0] = max(f[i - 1][0], f[i - 1][2] - w[i]);
        f[i][1] = f[i - 1][0] + w[i];
        f[i][2] = max(f[i - 1][2], f[i - 1][1]);
    }

    printf("%d\n", max(f[n][1], f[n][2]));

    return 0;
}
```

##### **Python**

```python
N = 100010
f = [[float('-inf')] * 3 for _ in range(N)]
w = [0] * N

if __name__ == '__main__':
    n = int(input())
    w[1:] = map(int, input().split())
    f[0][2] = 0  # 初始化，入口很重要

    for i in range(1, n + 1):
        f[i][0] = max(f[i - 1][0], f[i - 1][2] - w[i])
        f[i][1] = f[i - 1][0] + w[i]
        f[i][2] = max(f[i - 1][1], f[i - 1][2])
    print(max(f[n][1], f[n][2]))
    # 2的状态可以由1转移过来，不会增加w值；但存在极端情况，如数列递减
    # 这时不交易才是最大收益，就是f[n][2]，所以出口需要加上f[n][2]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1955. 统计特殊子序列的数目](https://leetcode-cn.com/problems/count-number-of-special-subsequences/)**
> 
> [weekly-252](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-08-01_Weekly-252)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性 DP 求方案数，首先明确状态定义和状态转移
> 
> 经典求方案，定义及转移、滚动数字压缩空间
> 
> 核心在于 状态定义 和 转移
> 
> 前 i 个位置分别构成 0 / 01 / 012 形式序列的方案数


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    int countSpecialSubsequences(vector<int>& nums) {
        LL a = 0, b = 0, c = 0;
        for (auto x : nums) {
            if (x == 0)
                // 不选本个 a
                // 选本个 则可以与前面连也可以不连 共a+1
                // 合计 a*2+1
                a = (a * 2 + 1) % MOD;
            if (x == 1)
                b = (b * 2 + a) % MOD;
            if (x == 2)
                c = (c * 2 + b) % MOD;
        }
        return c;
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

> [!NOTE] **[AcWing 1052. 设计密码](https://www.acwing.com/problem/content/description/1054/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> KMP【两种KMP写法】+复杂状态机
> 
> 在状态机中走 走n步不能始终不会走到状态m的数量
> 
> $ f[i][j] 表示 i 长度 状态为 j 的合法数量，i 最多 n， j 最多 m+1 $
> 
> **重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 55, mod = 1e9 + 7;

int n, m;
char str[N];
int nxt[N];
int f[N][N];

int main() {
    cin >> n >> str + 1;

    m = strlen(str + 1);

    for (int i = 2, j = 0; i <= m; i++) {
        while (j && str[i] != str[j + 1]) j = nxt[j];
        if (str[i] == str[j + 1]) j++;
        nxt[i] = j;
    }

    f[0][0] = 1;
    // 最好用下面这种定义写
    // i 状态定义表示已有的长度 不包括当前枚举的字母
    for (int i = 0; i < n; i++)
        // 枚举子串中的位置
        for (int j = 0; j < m; j++)
            // 枚举i处的字符
            for (char k = 'a'; k <= 'z'; k++) {
                int u = j;
                while (u && k != str[u + 1]) u = nxt[u];
                if (k == str[u + 1]) u++;
                // 没有出现完整子串
                if (u < m) f[i + 1][u] = (f[i + 1][u] + f[i][j]) % mod;
            }

    int res = 0;
    for (int i = 0; i < m; i++) res = (res + f[n][i]) % mod;

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

> [!NOTE] **[AcWing 1053. 修复DNA](https://www.acwing.com/problem/content/description/1055/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> AC自动机 
> 
> 状态定义类似上一道题

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

const int N = 1010;

int n, m;
int tr[N][4], dar[N], idx;
int q[N], ne[N];
char str[N];

int f[N][N];

int get(char c) {
    if (c == 'A') return 0;
    if (c == 'T') return 1;
    if (c == 'G') return 2;
    return 3;
}

void insert() {
    int p = 0;
    for (int i = 0; str[i]; i++) {
        int t = get(str[i]);
        if (tr[p][t] == 0) tr[p][t] = ++idx;
        p = tr[p][t];
    }
    dar[p] = 1;
}

void build() {
    int hh = 0, tt = -1;
    for (int i = 0; i < 4; i++)
        if (tr[0][i]) q[++tt] = tr[0][i];

    while (hh <= tt) {
        int t = q[hh++];
        for (int i = 0; i < 4; i++) {
            int p = tr[t][i];
            if (!p)
                tr[t][i] = tr[ne[t]][i];
            else {
                ne[p] = tr[ne[t]][i];
                q[++tt] = p;
                dar[p] |= dar[ne[p]];
            }
        }
    }
}

int main() {
    int T = 1;
    while (scanf("%d", &n), n) {
        memset(tr, 0, sizeof tr);
        memset(dar, 0, sizeof dar);
        memset(ne, 0, sizeof ne);
        idx = 0;

        for (int i = 0; i < n; i++) {
            scanf("%s", str);
            insert();
        }

        build();

        scanf("%s", str + 1);
        m = strlen(str + 1);

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= idx; j++)
                for (int k = 0; k < 4; k++) {
                    int t = get(str[i + 1]) != k;
                    int p = tr[j][k];
                    if (!dar[p]) f[i + 1][p] = min(f[i + 1][p], f[i][j] + t);
                }

        int res = 0x3f3f3f3f;
        for (int i = 0; i <= idx; i++) res = min(res, f[m][i]);

        if (res == 0x3f3f3f3f) res = -1;
        printf("Case %d: %d\n", T++, res);
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

> [!NOTE] **[LeetCode 551. 学生出勤记录 I](https://leetcode-cn.com/problems/student-attendance-record-i/)**
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
    bool checkRecord(string s) {
        int a = 0, l = 0;
        for (auto c : s) {
            if (c == 'A') ++ a, l = 0;
            else if (c == 'L') ++ l ;
            else l = 0;
            if (a > 1 || l > 2) return false;
        }
            
        return true;
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

> [!NOTE] **[LeetCode 552. 学生出勤记录 II](https://leetcode-cn.com/problems/student-attendance-record-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 状态机dp
> 
> 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int mod = 1e9 + 7, N = 100010;
int f[N][2][3];

// 技巧：不好根据前面的来计算当前作为后面的
// 那么 计算当前作为前面的 后面可以转移为什么

class Solution {
public:
    int checkRecord(int n) {
        memset(f, 0, sizeof f);
        f[0][0][0] = 1;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < 2; ++ j )
                for (int k = 0; k < 3; ++ k ) {
                    // 出现 A
                    if (!j) f[i + 1][j + 1][0] = (f[i + 1][j + 1][0] + f[i][j][k]) % mod;
                    // 出现 L
                    if (k + 1 <= 2) f[i + 1][j][k + 1] = (f[i + 1][j][k + 1] + f[i][j][k]) % mod;
                    // 出现 P
                    f[i + 1][j][0] = (f[i + 1][j][0] + f[i][j][k]) % mod;
                }
        int res = 0;
        for (int j = 0; j < 2; ++ j )
            for (int k = 0; k < 3; ++ k )
                res = (res + f[n][j][k]) % mod;
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

### trick

> [!NOTE] **[LeetCode 576. 出界的路径数](https://leetcode-cn.com/problems/out-of-boundary-paths/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp 细节 可以四个方向移动

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int m, n, N, mod = 1e9 + 7;
    int dx[4] = {0, -1, 1, 0}, dy[4] = {-1, 0, 0, 1};

    int findPaths(int m, int n, int N, int i, int j) {
        if (!N) return 0;
        vector<vector<int>> pre(m + 2, vector<int>(n + 2)), cur(m + 2, vector<int>(n + 2));
        for (int i = 1; i <= m; ++ i ) ++ pre[i][1], ++ pre[i][n];    // 向左右
        for (int i = 1; i <= n; ++ i ) ++ pre[1][i], ++ pre[m][i];    // 向上下
        int res = pre[i + 1][j + 1];
        for (int k = 2; k <= N; ++ k ) {
            for (int x = 1; x <= m; ++ x )
                for (int y = 1; y <= n; ++ y ) {
                    cur[x][y] = 0;
                    cur[x][y] += pre[x - 1][y]; cur[x][y] %= mod;
                    cur[x][y] += pre[x + 1][y]; cur[x][y] %= mod;
                    cur[x][y] += pre[x][y - 1]; cur[x][y] %= mod;
                    cur[x][y] += pre[x][y + 1]; cur[x][y] %= mod;
                }
            res = res + cur[i + 1][j + 1];
            res %= mod;
            pre = cur;
        }
        return res;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int findPaths(int m, int n, int N, int x, int y) {
        if (!N) return 0;
        const int MOD = 1e9 + 7;
        vector<vector<vector<int>>> f(m, vector<vector<int>>(n, vector<int>(N + 1)));
        for (int i = 0; i < n; i ++ ) {
            f[0][i][1] ++ ;
            f[m - 1][i][1] ++ ;
        }
        for (int i = 0; i < m; i ++ ) {
            f[i][0][1] ++ ;
            f[i][n - 1][1] ++ ;
        }

        int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
        for (int k = 1; k <= N; k ++ )
            for (int i = 0; i < m; i ++ )
                for (int j = 0; j < n; j ++ )
                    for (int u = 0; u < 4; u ++ ) {
                        int a = i + dx[u], b = j + dy[u];
                        if (a >= 0 && a < m && b >= 0 && b < n)
                            (f[i][j][k] += f[a][b][k - 1]) %= MOD;
                    }

        int res = 0;
        for (int k = 1; k <= N; k ++ )
            (res += f[x][y][k]) %= MOD;

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

#### 类似编辑距离的进阶

> [!NOTE] **[LeetCode 2060. 同源字符串检测](https://leetcode-cn.com/problems/check-if-an-original-string-exists-given-two-encoded-strings/)** TAG
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 状态转移

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 41, M = 2010, D = 1000;

    bool f[N][N][M];    // 第二个串比第一个串场多少

    bool possiblyEquals(string s1, string s2) {
        int n = s1.size(), m = s2.size();
        s1 = ' ' + s1, s2 = ' ' + s2;

        memset(f, 0, sizeof f);
        f[0][0][D] = true;
        for (int i = 0; i <= n; ++ i )
            for (int j = 0; j <= m; ++ j )
                for (int k = 1; k <= 2000; ++ k )
                    if (f[i][j][k]) {
                        // k == D 为什么要加 k 和 D 的限制条件 ?
                        // "98u8v8v8v89u888u998v88u98v88u9v99u989v8u"
                        // "9v898u98v888v89v998u98v9v888u9v899v998u9"
                        if (k == D && i + 1 <= n && j + 1 <= m && s1[i + 1] == s2[j + 1])
                            f[i + 1][j + 1][k] = true;
                        if (k < D && j + 1 <= m && isalpha(s2[j + 1]))
                            f[i][j + 1][k + 1] = true;
                        if (k > D && i + 1 <= n && isalpha(s1[i + 1]))
                            f[i + 1][j][k - 1] = true;
                        
                        {
                            int u = i + 1, v = 0;
                            while (u <= n && isdigit(s1[u])) {
                                v = v * 10 + s1[u] - '0';
                                if (k - v >= 1)
                                    f[u][j][k - v] = true;
                                u ++ ;
                            }
                        }
                        {
                            int u = j + 1, v = 0;
                            while (u <= m && isdigit(s2[u])) {
                                v = v * 10 + s2[u] - '0';
                                if (k + v <= 2000)
                                    f[i][u][k + v] = true;
                                u ++ ;
                            }
                        }
                    }
        return f[n][m][D];
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