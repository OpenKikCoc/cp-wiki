
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