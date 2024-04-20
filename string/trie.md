> [!TIP]  **关于 LeetCode 使用 Trie 静态数组**
> 
> 最好把静态数组放全局，否则在 class 内非常容易 heap use MLE


## 代码实现

TODO: 一个结构体封装的模板：


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// C++ Version
// TODO@binacs
```

##### **Python**

```python
# Python Version
class trie:
    nex = [[0 for i in range(26)] for j in range(100000)]
    cnt = 0
    exist = [False] * 100000 # 该结点结尾的字符串是否存在

    def insert(s, l): # 插入字符串
p = 0
for i in range(0, l):
    c = ord(s[i]) - ord('a')
    if nex[p][c] == 0:
nex[p][c] = cnt # 如果没有，就添加结点
cnt += 1
    p = nex[p][c]
exist[p] = True
    
    def find(s, l): # 查找字符串
p = 0
for i in range(0, l):
    c = ord(s[i]) - ord('a')
    if nex[p][c] == 0:
return False
    p = nex[p][c]
return exist[p]
```

<!-- tabs:end -->
</details>

<br>

* * *

## 应用

### 检索字符串

字典树最基础的应用——查找一个字符串是否在“字典”中出现过。

> [!NOTE] **[于是他错误的点名开始了](https://www.luogu.com.cn/problem/P2580)**
>   给你 $n$ 个名字串，然后进行 $m$ 次点名，每次你需要回答“名字不存在”、“第一次点到这个名字”、“已经点过这个名字”之一。
>    
>   $1\le n\le 10^4$，$1\le m\le 10^5$，所有字符串长度不超过 $50$。
    
> [!TIP]  **题解**
>    对所有名字建 trie，再在 trie 中查询字符串是否存在、是否已经点过名，第一次点名时标记为点过名。
    
> [!TIP]  **参考代码**

### AC 自动机

trie 是 [AC 自动机](string/ac-automaton.md) 的一部分。

### 维护异或极值

将数的二进制表示看做一个字符串，就可以建出字符集为 $\{0,1\}$ 的 trie 树。

> [!NOTE] **[BZOJ1954 最长异或路径](https://www.luogu.com.cn/problem/P4551)**
>
>   给你一棵带边权的树，求 $(u, v)$ 使得 $u$ 到 $v$ 的路径上的边权异或和最大，输出这个最大值。
>    
>   点数不超过 $10^5$，边权在 $[0,2^{31})$ 内。
    
> [!TIP]  **题解**
> 
> 随便指定一个根 $root$，用 $T(u, v)$ 表示 $u$ 和 $v$ 之间的路径的边权异或和，那么 $T(u,v)=T(root, u)\oplus T(root,v)$，因为 [LCA](graph/lca.md) 以上的部分异或两次抵消了。
> 
> 那么，如果将所有 $T(root, u)$ 插入到一棵 trie 中，就可以对每个 $T(root, u)$ 快速求出和它异或和最大的 $T(root, v)$：
> 
> 从 trie 的根开始，如果能向和 $T(root, u)$ 的当前位不同的子树走，就向那边走，否则没有选择。
> 
> 贪心的正确性：如果这么走，这一位为 $1$；如果不这么走，这一位就会为 $0$。而高位是需要优先尽量大的。
    
> [!TIP]  **参考代码**

### 维护异或和

01-trie 是指字符集为 $\{0,1\}$ 的 trie。01-trie 可以用来维护一些数字的异或和，支持修改（删除 + 重新插入），和全局加一（即：让其所维护所有数值递增 `1`，本质上是一种特殊的修改操作）。

如果要维护异或和，需要按值从低位到高位建立 trie。

**一个约定**：文中说当前节点 **往上** 指当前节点到根这条路径，当前节点 **往下** 指当前结点的子树。

#### 插入 & 删除

如果要维护异或和，我们 **只需要** 知道某一位上 `0` 和 `1` 个数的 **奇偶性** 即可，也就是对于数字 `1` 来说，当且仅当这一位上数字 `1` 的个数为奇数时，这一位上的数字才是 `1`，请时刻记住这段文字：如果只是维护异或和，我们只需要知道某一位上 `1` 的数量即可，而不需要知道 trie 到底维护了哪些数字。

对于每一个节点，我们需要记录以下三个量：

- `ch[o][0/1]` 指节点 `o` 的两个儿子，`ch[o][0]` 指下一位是 `0`，同理 `ch[o][1]` 指下一位是 `1`。
- `w[o]` 指节点 `o` 到其父亲节点这条边上数值的数量（权值）。每插入一个数字 `x`，`x` 二进制拆分后在 trie 上 路径的权值都会 `+1`。
- `xorv[o]` 指以 `o` 为根的子树维护的异或和。

具体维护结点的代码如下所示。

```cpp
void maintain(int o) {
    w[o] = xorv[o] = 0;
    if (ch[o][0]) {
        w[o] += w[ch[o][0]];
        xorv[o] ^= xorv[ch[o][0]] << 1;
    }
    if (ch[o][1]) {
        w[o] += w[ch[o][1]];
        xorv[o] ^= (xorv[ch[o][1]] << 1) | (w[ch[o][1]] & 1);
    }
    // w[o] = w[o] & 1;
    // 只需知道奇偶性即可，不需要具体的值。当然这句话删掉也可以，因为上文就只利用了他的奇偶性。
}
```

插入和删除的代码非常相似。

需要注意的地方就是：

- 这里的 `MAXH` 指 trie 的深度，也就是强制让每一个叶子节点到根的距离为 `MAXH`。对于一些比较小的值，可能有时候不需要建立这么深（例如：如果插入数字 `4`，分解成二进制后为 `100`，从根开始插入 `001` 这三位即可），但是我们强制插入 `MAXH` 位。这样做的目的是为了便于全局 `+1` 时处理进位。例如：如果原数字是 `3`（`11`），递增之后变成 `4`（`100`），如果当初插入 `3` 时只插入了 `2` 位，那这里的进位就没了。

- 插入和删除，只需要修改叶子节点的 `w[]` 即可，在回溯的过程中一路维护即可。

```cpp
namespace trie {
const int MAXH = 21;
int ch[_ * (MAXH + 1)][2], w[_ * (MAXH + 1)], xorv[_ * (MAXH + 1)];
int tot = 0;
int mknode() {
    ++tot;
    ch[tot][1] = ch[tot][0] = w[tot] = xorv[tot] = 0;
    return tot;
}
void maintain(int o) {
    w[o] = xorv[o] = 0;
    if (ch[o][0]) {
        w[o] += w[ch[o][0]];
        xorv[o] ^= xorv[ch[o][0]] << 1;
    }
    if (ch[o][1]) {
        w[o] += w[ch[o][1]];
        xorv[o] ^= (xorv[ch[o][1]] << 1) | (w[ch[o][1]] & 1);
    }
    w[o] = w[o] & 1;
}
void insert(int &o, int x, int dp) {
    if (!o) o = mknode();
    if (dp > MAXH) return (void)(w[o]++);
    insert(ch[o][x & 1], x >> 1, dp + 1);
    maintain(o);
}
void erase(int o, int x, int dp) {
    if (dp > 20) return (void)(w[o]--);
    erase(ch[o][x & 1], x >> 1, dp + 1);
    maintain(o);
}
}  // namespace trie
```

#### 全局加一

所谓全局加一就是指，让这棵 trie 中所有的数值 `+1`。

形式化的讲，设 trie 中维护的数值有 $V_1, V_2, V_3 \dots V_n$, 全局加一后 其中维护的值应该变成 $V_1+1, V_2+1, V_3+1 \dots V_n+1$

```cpp
void addall(int o) {
    swap(ch[o][0], ch[o][1]);
    if (ch[o][0]) addall(ch[o][0]);
    maintain(o);
}
```

我们思考一下二进制意义下 `+1` 是如何操作的。

我们只需要从低位到高位开始找第一个出现的 `0`，把它变成 `1`，然后这个位置后面的 `1` 都变成 `0` 即可。

下面给出几个例子感受一下：（括号内的数字表示其对应的十进制数字）

    1000(10)  + 1 = 1001(11)  ;
    10011(19) + 1 = 10100(20) ;
    11111(31) + 1 = 100000(32);
    10101(21) + 1 = 10110(22) ;
    100000000111111(16447) + 1 = 100000001000000(16448);

对应 trie 的操作，其实就是交换其左右儿子，顺着 **交换后** 的 `0` 边往下递归操作即可。

回顾一下 `w[o]` 的定义：`w[o]` 指节点 `o` 到其父亲节点这条边上数值的数量（权值）。

有没有感觉这个定义有点怪呢？如果在父亲结点存储到两个儿子的这条边的边权也许会更接近于习惯。但是在这里，在交换左右儿子的时候，在儿子结点存储到父亲这条边的距离，显然更加方便。

### 01-trie 合并

指的是将上述的两个 01-trie 进行合并，同时合并维护的信息。

可能关于合并 trie 的文章比较少，其实合并 trie 和合并线段树的思路非常相似，可以搜索“合并线段树”来学习如何合并 trie。

其实合并 trie 非常简单，就是考虑一下我们有一个 `int merge(int a, int b)` 函数，这个函数传入两个 trie 树位于同一相对位置的结点编号，然后合并完成后返回合并完成的结点编号。

考虑怎么实现？
分三种情况：

- 如果 `a` 没有这个位置上的结点，新合并的结点就是 `b`
- 如果 `b` 没有这个位置上的结点，新合并的结点就是 `a`
-   如果 `a`,`b` 都存在，那就把 `b` 的信息合并到 `a` 上，新合并的结点就是 `a`，然后递归操作处理 a 的左右儿子。

    **提示**：如果需要的合并是将 a，b 合并到一棵新树上，这里可以新建结点，然后合并到这个新结点上，这里的代码实现仅仅是将 b 的信息合并到 a 上。

```cpp
int merge(int a, int b) {
    if (!a) return b;  // 如果 a 没有这个位置上的结点，返回 b
    if (!b) return a;  // 如果 b 没有这个位置上的结点，返回 a
    /*
      如果 `a`, `b` 都存在，
      那就把 `b` 的信息合并到 `a` 上。
    */
    w[a] = w[a] + w[b];
    xorv[a] ^= xorv[b];
    /* 不要使用 maintain()，
      maintain() 是合并a的两个儿子的信息
      而这里需要 a b 两个节点进行信息合并
     */
    ch[a][0] = merge(ch[a][0], ch[b][0]);
    ch[a][1] = merge(ch[a][1], ch[b][1]);
    return a;
}
```

其实 trie 都可以合并，换句话说，trie 合并不仅仅限于 01-trie。

> [!NOTE] **[【luogu-P6018】【Ynoi2010】Fusion tree](https://www.luogu.com.cn/problem/P6018)**
> 
> 给你一棵 $n$ 个结点的树，每个结点有权值。$m$ 次操作。
> 
> 需要支持以下操作。
> 
> - 将树上与一个节点 $x$ 距离为 $1$ 的节点上的权值 $+1$。这里树上两点间的距离定义为从一点出发到另外一点的最短路径上边的条数。
> 
> - 在一个节点 $x$ 上的权值 $-v$。
> 
> -   询问树上与一个节点 $x$ 距离为 $1$ 的所有节点上的权值的异或和。

对于 $100\%$ 的数据，满足 $1\le n \le 5\times 10^5$，$1\le m \le 5\times 10^5$，$0\le a_i \le 10^5$，$1 \le x \le n$，$opt\in\{1,2,3\}$。
保证任意时刻每个节点的权值非负。
    
> [!TIP]  **题解**
每个结点建立一棵 trie 维护其儿子的权值，trie 应该支持全局加一。
可以使用在每一个结点上设置懒标记来标记儿子的权值的增加量。
    
> [!TIP]  **参考代码**

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

> [!NOTE] **[【luogu-P6623】 【省选联考 2020 A 卷】 树](https://www.luogu.com.cn/problem/P6623)**
> 
> 给定一棵 $n$ 个结点的有根树 $T$，结点从 $1$ 开始编号，根结点为 $1$ 号结点，每个结点有一个正整数权值 $v_i$。
> 
> 设 $x$ 号结点的子树内（包含 $x$ 自身）的所有结点编号为 $c_1,c_2,\dots,c_k$，定义 $x$ 的价值为：  
> 
> $val(x)=(v_{c_1}+d(c_1,x)) \oplus (v_{c_2}+d(c_2,x)) \oplus \cdots \oplus (v_{c_k}+d(c_k, x))$ 其中 $d(x,y)$。  
> 
> 表示树上 $x$ 号结点与 $y$ 号结点间唯一简单路径所包含的边数，$d(x,x) = 0$。$\oplus$ 表示异或运算。
> 
> 请你求出 $\sum\limits_{i=1}^n val(i)$ 的结果。
    
> [!TIP]  **题解**
> 
> 考虑每个结点对其所有祖先的贡献。
> 
> 每个结点建立 trie，初始先只存这个结点的权值，然后从底向上合并每个儿子结点上的 trie，然后再全局加一，完成后统计答案。


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int _ = 526010;
int n;
int V[_];
int debug = 0;
namespace trie {
const int MAXH = 21;
int ch[_ * (MAXH + 1)][2], w[_ * (MAXH + 1)], xorv[_ * (MAXH + 1)];
int tot = 0;
int mknode() {
    ++tot;
    ch[tot][1] = ch[tot][0] = w[tot] = xorv[tot] = 0;
    return tot;
}
void maintain(int o) {
    w[o] = xorv[o] = 0;
    if (ch[o][0]) {
        w[o] += w[ch[o][0]];
        xorv[o] ^= xorv[ch[o][0]] << 1;
    }
    if (ch[o][1]) {
        w[o] += w[ch[o][1]];
        xorv[o] ^= (xorv[ch[o][1]] << 1) | (w[ch[o][1]] & 1);
    }
    w[o] = w[o] & 1;
}
void insert(int &o, int x, int dp) {
    if (!o) o = mknode();
    if (dp > MAXH) return (void)(w[o]++);
    insert(ch[o][x & 1], x >> 1, dp + 1);
    maintain(o);
}
int merge(int a, int b) {
    if (!a) return b;
    if (!b) return a;
    w[a] = w[a] + w[b];
    xorv[a] ^= xorv[b];
    ch[a][0] = merge(ch[a][0], ch[b][0]);
    ch[a][1] = merge(ch[a][1], ch[b][1]);
    return a;
}
void addall(int o) {
    swap(ch[o][0], ch[o][1]);
    if (ch[o][0]) addall(ch[o][0]);
    maintain(o);
}
}  // namespace trie
int rt[_];
long long Ans = 0;
vector<int> E[_];
void dfs0(int o) {
    for (int i = 0; i < E[o].size(); i++) {
        int node = E[o][i];
        dfs0(node);
        rt[o] = trie::merge(rt[o], rt[node]);
    }
    trie::addall(rt[o]);
    trie::insert(rt[o], V[o], 0);
    Ans += trie::xorv[rt[o]];
}
int main() {
    n = read();
    for (int i = 1; i <= n; i++) V[i] = read();
    for (int i = 2; i <= n; i++) E[read()].push_back(i);
    dfs0(1);
    printf("%lld", Ans);
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

### 可持久化字典树

参见 [可持久化字典树](ds/persistent-trie.md)。


## 习题

### 一般 trie

> [!NOTE] **[AcWing 835. Trie字符串统计](https://www.acwing.com/problem/content/837/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int son[N][26]; // 其中存放的是：子节点对应的idx。其中son数组的第一维是：父节点对应的idx，第第二维计数是：其直接子节点('a' - '0')的值为二维下标。
int cnt [N];    // 以“abc”字符串为例，最后一个字符---‘c’对应的idx作为cnt数组的下标。数组的值是该idx对应的个数。
int idx;        // 将该字符串分配的一个树结构中，以下标来记录每一个字符的位置。方便之后的插入和查找。
char str[N];

void insert(char *str) {
    int p = 0;
    for (int i = 0; str[i]; i++) {
        int u = str[i] - '0';
        if (!son[p][u]) son[p][u] = ++idx;
        p = son[p][u];
    }
    // 此时的p就是str中最后一个字符对应的trie树的位置idx。
    cnt[p]++;
}

int query(char *str) {
    int p = 0;
    for (int i = 0; str[i]; i++) {
        int u = str[i] - '0';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}

int main()
{
    int n;
    scanf("%d", &n);
    char op[2];
    while (n--)  {
        scanf("%s%s", op, str);
        if (op[0] == 'I') insert(str);
        else printf("%d\n", query(str));
    }

    return 0;
}
```

##### **Python**

```python
"""
I. 概念
1. Trie树，又名字典树or前缀树，是一种有序树。用于保存关联数组，其中的键通常是字符串
2. 一个节点的所有子孙都有相同的前缀，也就是这个节点对应的字符串，而根节点对应空字符串
3. 优点：可以快速高效实现插入和查找字符串集合的操作：可以最大限度地减少无谓的字符串的比较（经常被搜索引擎系统用于文本词频统计）
4. 缺点：trie是用空间换时间
5. 一般算法题这类题的字符类型都比较单一，比如全部都是小写/大写字母 或者全部都是数字等		

II. 理解模版代码解答
1. 实现各种数据结构的时候比较容易用到idx变量，这个变量到底有什么用？
    1）在链表/树/堆中，节点一般包含两个基本属性：本身的值和指向下一个结点的指针；但一般算法题用数组实现，因为快：所以用到了idx。
    2）trie树有一个二位数组**son[N ][26]** , 表示当前结点的儿子；如果没有的话，就idx++。trie树本质上是一颗多叉树，对于一个字母而言最多有26个子结点；所以这个二位数组包含了两条信息。比如：son[1][0]=2，表示1结点的一个值为a的子结点为结点2；如果son[1][0]=0, 则意味着没有值为a子结点。
"""


def insert(s):
    global idx
    p = 0
    for i in range(len(s)):
        t = ord(s[i]) - ord('a')  # t=0-25，对应a-z
        if not son[p][t]:
            idx += 1
            son[p][t] = idx
        p = son[p][t]
    cnt[p] += 1


def query(s):
    p = 0
    for i in range(len(s)):
        t = ord(s[i]) - ord('a')
        if not son[p][t]:
            return 0
        p = son[p][t]
    return cnt[p]


if __name__ == '__main__':
    N = int(1e5 + 10)
    son = [[0] * 26 for i in range(N)]  # 26表示每个字母都可能有的26个子结点，son数组表示的是每个节点的儿子
    cnt = [0] * N  # 存以当前点为结尾的单词有多少个
    idx = 0  # 和单链表的idx一致，表示当前节点用到的哪个下标；下标为0的节点 既是根节点 又是空节点

    n = int(input())
    for i in range(n):
        op = input().split()
        if op[0] == 'I':
            insert(op[1])
        else:
            print(query(op[1]))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 143. 最大异或对](https://www.acwing.com/problem/content/145/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int son[N * 32][2], idx;     // 1. N * 32 instead of N

void insert(int x) {
    int p = 0;
    for (int i = 30; i >= 0; -- i ) {
        int u = x >> i & 1;
        if (!son[p][u]) son[p][u] = ++ idx ;
        p = son[p][u];
    }
}

int query(int x) {
    int p = 0, ret = 0;
    for (int i = 30; i >= 0; -- i ) {
        int u = x >> i & 1;
        if (!son[p][!u]) p = son[p][u];
        else {
            ret |= 1 << i;
            p = son[p][!u];
        }
    }
    return ret;
}

int main() {
    int n, res = 0;
    cin >> n;
    for (int i = 0; i < n; ++ i ) {
        int x;
        cin >> x;
        insert(x);  // 2. 先插入 可以避免在 query 时节点不存在的判断
        res = max(res, query(x));
    }
    cout << res << endl;
    return 0;
}
```

##### **Python**

```python
def insert(v):
    global idx
    p = 0
    for i in range(30, -1, -1):  # 第一位31位 是符号位，不做处理。所以从第30位开始处理
        u = v >> i & 1  # 取第i位数的值
        if not son[p][u]:
            idx += 1
            son[p][u] = idx
        p = son[p][u]


def query(v):
    p, res = 0, 0
    for i in range(30, -1, -1):
        u = v >> i & 1
        w = u ^ 1  # 取第i位的 异或值
        if son[p][w]:
            res += 1 << i
            p = son[p][w]
        else:
            p = son[p][u]
    return res


if __name__ == '__main__':
    n = int(input())
    nums = list(map(int, input().split()))

    N = 3000000
    son = [[0] * 2 for _ in range(N)]  # 子结点 要么为0，要么为1
    idx = 0
    res = 0
    for v in nums:
        insert(v)
        res = max(query(v), res)
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 1414. 牛异或](https://www.acwing.com/problem/content/1416/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **细节**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010, M = N * 21;

int n;
int s[N];
int son[M][2], id[M], idx;

void insert(int x, int k) {
    int p = 0;
    for (int i = 20; i >= 0; -- i ) {
        int u = x >> i & 1;
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    id[p] = k;  // 如果有重复值 后面的覆盖前面的
}

int query(int x) {
    int p = 0;
    for (int i = 20; i >= 0; -- i ) {
        int u = x >> i & 1;
        if (son[p][!u]) p = son[p][!u];
        else p = son[p][u];
    }
    return id[p];
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i ) {
        cin >> s[i];
        s[i] ^= s[i - 1];
    }
    
    int res = -1, a, b;
    insert(s[0], 0);
    for (int i = 1; i <= n; ++ i ) {
        int k = query(s[i]);
        int t = s[i] ^ s[k];
        if (t > res) res = t, a = k + 1, b = i;
        insert(s[i], i);
    }
    cout << res << ' ' << a << ' ' << b << endl;
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

> [!NOTE] **[LeetCode 208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)**
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
class Trie {
public:
    struct Node {
        bool is_end;
        Node *son[26];
        Node() {
            is_end = false;
            for (int i = 0; i < 26; i ++ )
                son[i] = NULL;
        }
    }*root;

    /** Initialize your data structure here. */
    Trie() {
        root = new Node();
    }

    /** Inserts a word into the trie. */
    void insert(string word) {
        auto p = root;
        for (auto c: word) {
            int u = c - 'a';
            if (!p->son[u]) p->son[u] = new Node();
            p = p->son[u];
        }
        p->is_end = true;
    }

    /** Returns if the word is in the trie. */
    bool search(string word) {
        auto p = root;
        for (auto c: word) {
            int u = c - 'a';
            if (!p->son[u]) return false;
            p = p->son[u];
        }
        return p->is_end;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string word) {
        auto p = root;
        for (auto c: word) {
            int u = c - 'a';
            if (!p->son[u]) return false;
            p = p->son[u];
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
```

##### **Python**

```python
"""
	1. Trie树，又名字典树or前缀树，是一种有序树。用于保存关联数组，其中的键通常是字符串
	2. 一个节点的所有子孙都有相同的前缀，也就是这个节点对应的字符串，而根节点对应空字符串
	3. 优点：可以快速高效实现插入和查找字符串集合的操作：可以最大限度地减少无谓的字符串的比较（经常被搜索引擎系统用于文本词频统计）
	4. 缺点：trie是用空间换时间
	5. 一般算法题这类题的字符类型都比较单一，比如全部都是小写/大写字母 或者全部都是数字等		
	
	trie树有一个二位数组 son[N ][26] , 表示当前结点的儿子；如果没有的话，就idx++。trie树本质上是一颗多叉树，对于一个字母而言最多有26个子结点；所以这个二位数组包含了两条信息。比如：son[1][0]=2，表示1结点的一个值为a的子结点为结点2；如果son[1][0]=0, 则意味着没有值为a子结点。
"""

class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for s in word:
            if s in node.keys():
                node = node[s]
            else:
                node[s] = {}
                node = node[s]
        node['is_word'] = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for s in word:
            if s in node.keys():
                node = node[s]
            else:
                return False

        if 'is_word' in node.keys():
            return True
        else:
            return False
    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for s in prefix:
            if s in node.keys():
                node = node[s]
            else:
                return False

        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


class Node:
    def __init__(self):
        self.is_end = False
        self.ne = [None for _ in range(26)]

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self.root
        for o in word:
            if not cur.ne[ord(o) - ord("a")]:
                cur.ne[ord(o) - ord("a")] = Node()
            cur = cur.ne[ord(o) - ord("a")]
        cur.is_end = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self.root
        for o in word:
            if not cur.ne[ord(o) - ord("a")]:
                return False
            cur = cur.ne[ord(o) - ord("a")]
        return cur.is_end

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        cur = self.root
        for o in prefix:
            if not cur.ne[ord(o) - ord("a")]:
                return False
            cur = cur.ne[ord(o) - ord("a")]
        return True

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 211. 添加与搜索单词 - 数据结构设计](https://leetcode-cn.com/problems/design-add-and-search-words-data-structure/)**
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
class WordDictionary {
public:
    struct Node {
        bool is_end;
        Node *son[26];
        Node() {
            is_end = false;
            for (int i = 0; i < 26; i ++ ) son[i] = nullptr;
        }
    }*root;

    /** Initialize your data structure here. */
    WordDictionary() {
        root = new Node();
    }
    
    /** Adds a word into the data structure. */
    void addWord(string word) {
        auto p = root;
        for (auto c : word) {
            int u = c - 'a';
            if (!p->son[u]) p->son[u] = new Node();
            p = p->son[u];
        }
        p->is_end = true;
    }
    
    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    bool search(string word) {
        return dfs(root, word, 0);
    }

    bool dfs(Node * p, string & word, int i) {
        if (i == word.size()) return p->is_end;
        if (word[i] != '.') {
            int u = word[i] - 'a';
            if (!p->son[u]) return false;
            return dfs(p->son[u], word, i + 1);
        }
        for (int j = 0; j < 26; j ++ )
            if (p->son[j] && dfs(p->son[j], word, i + 1)) return true;
        return false;
    }
};

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary* obj = new WordDictionary();
 * obj->addWord(word);
 * bool param_2 = obj->search(word);
 */
```

##### **Python**

```python
#用字典
class WordDictionary:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.lookup={}


    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        tree=self.lookup
        for a in word:
            if a not in tree:
                tree[a]={}
            tree=tree[a]
        tree['#']='#'
        #tree['#']=1
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        tree=self.lookup
        
        def dfs(tree,u):
            if u==len(word):
                return "#" in tree
            if word[u]!='.':
                if word[u] not in tree:
                    return False
                tree=tree[word[u]]
                return dfs(tree,u+1)
            else:
                for c in range(ord('a'),ord('z')+1):
                    c=chr(c)
                    if c in tree:
                        n_tree=tree[c]
                        if dfs(n_tree,u+1):
                            return True
                return False
            return Fasle

        return dfs(tree,0)
      
#用结构体/类
class Node:
    def __init__(self):
        self.is_end=False
        self.son=[None for _ in range(26)]

class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root=Node()
        

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        p=self.root
        for c in word:
            if not p.son[ord(c)-ord('a')]:
                p.son[ord(c)-ord('a')]=Node()
            p=p.son[ord(c)-ord('a')]
        p.is_end=True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        def dfs(i,p):
            if i==len(word):
                return p.is_end
            if word[i]=='.':
                for j in range(26):
                    if p.son[j]:
                        if dfs(i+1,p.son[j]):
                            return True
            else:
                j=ord(word[i])-ord('a')
                if p.son[j]:
                    if dfs(i+1,p.son[j]):
                        return True
            return False
        return dfs(0,self.root)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 212. 单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)**
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
    struct Node {
        int id;
        Node * son[26];
        Node() {
            id = -1;
            for (int i = 0; i < 26; ++ i )
                son[i] = nullptr;
        }
    } * root;

    void insert(string & word, int id) {
        auto p = root;
        for (auto c : word) {
            int u = c - 'a';
            if (!p->son[u])
                p->son[u] = new Node();
            p = p->son[u];
        }
        p->id = id;
    }

    vector<vector<char>> g;
    int n, m;
    unordered_set<int> ids;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};

    void dfs(int x, int y, Node * p) {
        if (p->id != -1)
            ids.insert(p->id);
        char t = g[x][y];
        g[x][y] = '.';
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m || g[nx][ny] == '.')
                continue;
            int u = g[nx][ny] - 'a';
            if (p->son[u])
                dfs(nx, ny, p->son[u]);
        }
        g[x][y] = t;
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        g = board; n = g.size(), m = g[0].size();
        root = new Node();
        for (int i = 0; i < words.size(); ++ i )
            insert(words[i], i);
        
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j ) {
                int u = g[i][j] - 'a';
                if (root->son[u])
                    dfs(i, j, root->son[u]);
            }
        
        vector<string> res;
        for (auto id : ids)
            res.push_back(words[id]);
        return res;
    }
};
```

##### **Python**

```python
class Solution {
public:
    struct Node {
        int id;
        Node *son[26];
        Node() {
            id = -1;
            for (int i = 0; i < 26; i ++ ) son[i] = nullptr;
        }
    }*root;

    void insert(string & word, int id) {
        auto p = root;
        for (auto c : word) {
            int u = c - 'a';
            if (!p->son[u]) p->son[u] = new Node();
            p = p->son[u];
        }
        p->id = id;
    }

    vector<vector<char>> g;
    int n, m;
    unordered_set<int> ids;
    int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    void dfs(int x, int y, Node * p) {
        if (p->id != -1) ids.insert(p->id);
        char t = g[x][y];
        g[x][y] = '.';
        for (int i = 0; i < 4; ++ i ) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m || g[nx][ny] == '.') continue;
            int u = g[nx][ny] - 'a';
            if (p->son[u]) dfs(nx, ny, p->son[u]);
        }
        g[x][y] = t;
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        g = board; n = g.size(), m = g[0].size();
        root = new Node();
        for (int i = 0; i < words.size(); ++ i ) insert(words[i], i);

        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++) {
                int u = g[i][j] - 'a';
                if (root->son[u])
                    dfs(i, j, root->son[u]);
            }
        
        vector<string> res;
        for (auto id : ids) res.push_back(words[id]);
        return res;
    }
};
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 421. 数组中两个数的最大异或值](https://leetcode-cn.com/problems/maximum-xor-of-two-numbers-in-an-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> Trie 简单变形

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> s;

    void insert(int x) {
        int p = 0;
        for (int i = 30; i >= 0; -- i ) {
            int u = x >> i & 1;
            if (!s[p][u]) s[p][u] = s.size(), s.push_back({0, 0});
            p = s[p][u];
        }
    }

    int query(int x) {
        int p = 0, res = 0;
        for (int i = 30; i >= 0; -- i ) {
            int u = x >> i & 1;
            // if (s[p][!u]) p = s[p][!u], res = res * 2 + !u;
            // else p = s[p][u], res = res * 2 + u;
            if (s[p][!u]) p = s[p][!u], res |= 1 << i;
            else p = s[p][u];
        }
        // return res ^ x;
        return res;
    }

    int findMaximumXOR(vector<int>& nums) {
        s.push_back({0, 0});
        int res = 0;
        for (auto x : nums) {
            res = max(res, query(x));
            insert(x);
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

> [!NOTE] **[LeetCode 676. 实现一个魔法字典](https://leetcode-cn.com/problems/implement-magic-dictionary/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trie上搜索 细节 实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 10010;

int son[N][26], idx;
bool is_end[N];

class MagicDictionary {
public:
    void insert(string& s) {
        int p = 0;
        for (auto c: s) {
            int u = c - 'a';
            if (!son[p][u]) son[p][u] = ++ idx;
            p = son[p][u];
        }
        is_end[p] = true;
    }

    /** Initialize your data structure here. */
    MagicDictionary() {
        memset(son, 0, sizeof son);
        idx = 0;
        memset(is_end, 0, sizeof is_end);
    }

    void buildDict(vector<string> dictionary) {
        for (auto& s: dictionary) insert(s);
    }

    bool dfs(string& s, int p, int u, int c) {
        if (is_end[p] && u == s.size() && c == 1) return true;
        if (c > 1 || u == s.size()) return false;

        for (int i = 0; i < 26; i ++ ) {
            if (!son[p][i]) continue;
            if (dfs(s, son[p][i], u + 1, c + (s[u] - 'a' != i)))
                return true;
        }
        return false;
    }

    bool search(string searchWord) {
        return dfs(searchWord, 0, 0, 0);
    }
};

/**
 * Your MagicDictionary object will be instantiated and called as such:
 * MagicDictionary* obj = new MagicDictionary();
 * obj->buildDict(dictionary);
 * bool param_2 = obj->search(searchWord);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 677. 键值映射](https://leetcode-cn.com/problems/map-sum-pairs/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trie前缀和

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 2510;

int son[N][26], V[N], S[N], idx;

class MapSum {
public:
    // last 用于修改旧值
    void add(string& s, int value, int last) {
        int p = 0;
        for (auto c: s) {
            int u = c - 'a';
            if (!son[p][u]) son[p][u] = ++ idx;
            p = son[p][u];
            S[p] += value - last;
        }
        V[p] = value;
    }

    int query(string& s) {
        int p = 0;
        for (auto c: s) {
            int u = c - 'a';
            if (!son[p][u]) return 0;
            p = son[p][u];
        }
        return p;
    }

    /** Initialize your data structure here. */
    MapSum() {
        memset(son, 0, sizeof son);
        idx = 0;
        memset(V, 0, sizeof V);
        memset(S, 0, sizeof S);
    }

    void insert(string key, int val) {
        add(key, val, V[query(key)]);
    }

    int sum(string prefix) {
        return S[query(prefix)];
    }
};

/**
 * Your MapSum object will be instantiated and called as such:
 * MapSum* obj = new MapSum();
 * obj->insert(key,val);
 * int param_2 = obj->sum(prefix);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 720. 词典中最长的单词](https://leetcode-cn.com/problems/longest-word-in-dictionary/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> Trie 树 + 搜索

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int N = 30010;

int son[N][26], idx;
int id[N];

class Solution {
public:
    void insert(string& str, int k) {
        int p = 0;
        for (auto c: str) {
            int u = c - 'a';
            if (!son[p][u]) son[p][u] = ++ idx;
            p = son[p][u];
        }
        id[p] = k;
    }

    vector<int> dfs(int p, int len) {
        vector<int> res{len, id[p]};
        for (int i = 0; i < 26; i ++ ) {
            int j = son[p][i];
            if (j && id[j] != -1) {
                auto t = dfs(j, len + 1);
                if (res[0] < t[0]) res = t;
            }
        }
        return res;
    }

    string longestWord(vector<string>& words) {
        memset(id, -1, sizeof id);
        memset(son, 0, sizeof son);
        idx = 0;
        for (int i = 0; i < words.size(); i ++ ) insert(words[i], i);
        auto t = dfs(0, 0);
        if (t[1] != -1) return words[t[1]];
        return "";
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

> [!NOTE] **[LeetCode 2227. 加密解密字符串](https://leetcode-cn.com/problems/encrypt-and-decrypt-strings/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - Trie + 搜索即可
> 
> - **脑筋急转弯做法**
> 
>   一开始预处理 dictionary 里所有数加密后的结果，decrypt 函数直接查表输出即可。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ trie + dfs**

```cpp
const static int N = 18; // 100*100*26 = 2^18
static int tr[1 << N][26], f[1 << N];
int idx;
    

class Encrypter {
public:
    unordered_map<char, string> hash1;
    unordered_map<string, unordered_set<char>> hash2;

    void insert(string s) {
        int p = 0;
        for (auto c : s) {
            int t = c - 'a';
            if (!tr[p][t])
                tr[p][t] = ++ idx ;
            p = tr[p][t];
        }
        f[p] = 1;
    }
    
    Encrypter(vector<char>& keys, vector<string>& values, vector<string>& dictionary) {
        int n = keys.size();
        for (int i = 0; i < n; ++ i )
            hash1[keys[i]] = values[i], hash2[values[i]].insert(keys[i]);
        idx = 0;
        memset(tr, 0, sizeof tr);
        memset(f, 0, sizeof f);
        for (auto & s : dictionary)
            insert(s);
    }
    
    string encrypt(string word1) {
        string res;
        for (auto c : word1)
            res += hash1[c];
        return res;
    }
    
    int cnt, n;
    string s;
    
    void dfs(int u, int p) {
        if (u == n) {
            if (f[p])
                cnt ++ ;
            return;
        }
        string t = s.substr(u, 2);
        for (auto c : hash2[t]) {
            int t = c - 'a', x = tr[p][t];
            if (x)
                dfs(u + 2, x);
        }
    }
    
    int decrypt(string word2) {
        this->cnt = 0;
        this->s = word2;
        this->n = s.size();
        dfs(0, 0);
        return cnt;
    }
};

/**
 * Your Encrypter object will be instantiated and called as such:
 * Encrypter* obj = new Encrypter(keys, values, dictionary);
 * string param_1 = obj->encrypt(word1);
 * int param_2 = obj->decrypt(word2);
 */
```

##### **C++ trick**

```cpp
class Encrypter {
    std::map<char, std::string> en;
    std::map<std::string, int> cnt;

public:
    Encrypter(vector<char>& keys, vector<string>& values,
              vector<string>& dictionary) {
        for (int i = 0; i < int(keys.size()); i++)
        	en[keys[i]] = values[i];
        // 用哈希表记录每个加密后的字符串的出现次数
        for (auto s : dictionary)
        	cnt[encrypt(s)]++;
    }

    string encrypt(string word1) {
        std::string s;
        for (auto c : word1)
        	s += en[c];
        return s;
    }

    int decrypt(string word2) {
    	return cnt[word2];
    }
};

/**
 * Your Encrypter object will be instantiated and called as such:
 * Encrypter* obj = new Encrypter(keys, values, dictionary);
 * string param_1 = obj->encrypt(word1);
 * int param_2 = obj->decrypt(word2);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2935. 找出强数对的最大异或值 II](https://leetcode.cn/problems/maximum-strong-pair-xor-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典暴力优化
> 
> 双指针(单调性证明) + Trie(删除操作)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 较显然的，需要针对第一题的实现做 "暴力优化"
    //     考虑将强数对的条件发生转化: 排序后，当前位置的元素向前找，差值不能超过当前元素值的所有位置中，异或值最大的
    // => 显然 双指针扫描 + Trie维护 且结合了动态删除
    
    // ATTENTION N 的取值需要是 length * 20
    // 【因为值域原因，每个值都可能产生20个trie中的节点】
    const static int N = 5e4 * 20 + 10, M = 22;
    
    int tr[N][2], cnt[N], idx;
    
    void init() {
        memset(tr, 0, sizeof tr);
        idx = 0;
    }
    void insert(int x) {
        int p = 0;
        for (int i = M - 1; i >= 0; -- i ) {
            int u = x >> i & 1;
            if (!tr[p][u])
                tr[p][u] = ++ idx;
            p = tr[p][u];
            cnt[p] ++ ;     // ATTENTION
        }
    }
    void remove(int x) {
        int p = 0;
        for (int i = M - 1; i >= 0; -- i ) {
            int u = x >> i & 1;
            p = tr[p][u];   // 一定存在
            cnt[p] -- ;     // ATTENTION
        }
    }
    int query(int x) {
        int p = 0, ret = 0;
        for (int i = M - 1; i >= 0; -- i ) {
            int u = x >> i & 1;
            if (!tr[p][!u] || !cnt[tr[p][!u]])  // ATTENTION
                p = tr[p][u];
            else {
                ret |= 1 << i;
                p = tr[p][!u];
            }
        }
        return ret;
    }
    
    int maximumStrongPairXor(vector<int>& nums) {
        init();
        int n = nums.size(), res = 0;
        sort(nums.begin(), nums.end());
        for (int i = 0, j = 0; j < n; ++ j ) {
            while (i < j && nums[j] - nums[i] > nums[i])    // ATTENTION 思考条件
                remove(nums[i ++ ]);
            // cout << " j = " << j << " i = " << i << " query = " << query(nums[j]) << endl;
            res = max(res, query(nums[j]));
            insert(nums[j]);
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

### 进阶构造和使用

> [!NOTE] **[LeetCode 745. 前缀和后缀搜索](https://leetcode-cn.com/problems/prefix-and-suffix-search/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trie + trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class WordFilter {
public:
    const static int N = 2000000;
    int son[N][27], w[N], idx;

    void insert(string & s, int id) {
        int p = 0;
        for (auto c : s) {
            int t = c == '#' ? 26 : c - 'a';
            if (!son[p][t])
                son[p][t] = ++ idx ;
            p = son[p][t];
            w[p] = id;
        }
    }

    int query(string s) {
        int p = 0;
        for (auto c : s) {
            int t = c == '#' ? 26 : c - 'a';
            if (!son[p][t])
                return -1;
            p = son[p][t];
        }
        return w[p];
    }

    WordFilter(vector<string>& words) {
        memset(son, 0, sizeof son);
        idx = 0;
        for (int i = 0; i < words.size(); ++ i ) {
            string s = '#' + words[i];
            insert(s, i);
            // trick
            // 生成 [后缀 + '#' + 原串] 的串
            for (int j = words[i].size() - 1; j >= 0; -- j ) {
                s = words[i][j] + s;
                insert(s, i);
            }
        }
    }
    
    int f(string prefix, string suffix) {
        return query(suffix + '#' + prefix);
    }
};

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter* obj = new WordFilter(words);
 * int param_1 = obj->f(prefix,suffix);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1268. 搜索推荐系统](https://leetcode-cn.com/problems/search-suggestions-system/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟题， Trie 标准解法，暴力略加优化也可以过。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
struct Trie {
    unordered_map<char, Trie*> child;
    priority_queue<string> words;
};

class Solution {
private:
    void addWord(Trie* root, const string& word) {
        Trie* cur = root;
        for (const char& ch: word) {
            if (!cur->child.count(ch)) {
                cur->child[ch] = new Trie();
            }
            cur = cur->child[ch];
            cur->words.push(word);
            if (cur->words.size() > 3) {
                cur->words.pop();
            }
        }
    }
    
public:
    vector<vector<string>> suggestedProducts(vector<string>& products, string searchWord) {
        Trie* root = new Trie();
        for (const string& word: products) {
            addWord(root, word);
        }
        
        vector<vector<string>> ans;
        Trie* cur = root;
        bool flag = false;
        for (const char& ch: searchWord) {
            if (flag || !cur->child.count(ch)) {
                ans.emplace_back();
                flag = true;
            }
            else {
                cur = cur->child[ch];
                vector<string> selects;
                while (!cur->words.empty()) {
                    selects.push_back(cur->words.top());
                    cur->words.pop();
                }
                reverse(selects.begin(), selects.end());
                ans.push_back(move(selects));
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

> [!NOTE] **[LeetCode 1803. 统计异或值在范围内的数对有多少](https://leetcode-cn.com/problems/count-pairs-with-xor-in-a-range/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然 Trie 注意求区间变为区间减 以及在 query 中的实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 16;
    int son[1 << N][2], v[1 << N], cnt;
    
    void insert(int x) {
        int p = 0;
        for (int i = 15; i >= 0; -- i ) {
            int t = x >> i & 1;
            if (!son[p][t])
                son[p][t] = ++ cnt ;
            p = son[p][t];
            v[p] ++ ;
        }
    }
    
    // 求小于等于 hi 的节点个数
    int query(int x, int hi) {
        int p = 0, res = 0;
        for (int i = 15; i >= 0; -- i ) {
            int t = x >> i & 1, k = hi >> i & 1;
            if (k) {
                // k = 1 则当前位可以和以前的数值异或 0/1 均合法
                // 相同值产生异或结果为0 直接累加即可
                if (son[p][t])
                    res += v[son[p][t]];
                // 不同值不存在 直接返回
                if (!son[p][!t])
                    return res;
                p = son[p][!t];
            } else {
                // k = 0 只能选择异或结果为0的分支
                if (!son[p][t])
                    return res;
                p = son[p][t];
            }
        }
        res += v[p];
        return res;
    }
    
    int countPairs(vector<int>& nums, int low, int high) {
        memset(son, 0, sizeof son);
        memset(v, 0, sizeof v);
        cnt = 0;
        
        int res = 0;
        for (auto v : nums) {
            insert(v);
            res += query(v, high) - query(v, low - 1);
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

> [!NOTE] **[LeetCode 2977. 转换字符串的最小成本 II](https://leetcode.cn/problems/minimum-cost-to-convert-string-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> Trie + DP

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 2e5 + 10, M = 26, K = 2e3 + 10;  // trie树节点上限: 1000 * 100 * 2(两个串) = 2e5
    const static LL INF = 1e16;

    int son[N][M], idx, id;
    unordered_map<string, int> hash;
    unordered_map<int, int> mem;

    void init() {
        memset(son, 0, sizeof son);
        idx = 0, id = 0;
        hash.clear(), mem.clear();
    }
    void insert(string & s) {
        if (hash.count(s))
            return;
        int p = 0;
        for (auto c : s) {
            int u = c - 'a';
            if (!son[p][u])
                son[p][u] = ++ idx ;
            p = son[p][u];
        }
        hash[s] = ++ id ;   // 从 1 开始
        mem[p] = id;        // 顺便记录树节点与对应值
    }

    LL w[K][K], f[K / 2];

    long long minimumCost(string source, string target, vector<string>& original, vector<string>& changed, vector<int>& cost) {
        init();

        {
            // 初始化 w
            memset(w, 0x3f, sizeof w);
            for (int i = 0; i < K; ++ i )
                w[i][i] = 0;
            for (int i = 0; i < original.size(); ++ i ) {
                string a = original[i], b = changed[i];
                // 插入 trie 同时记录离散化
                insert(a), insert(b);
                int x = hash[a], y = hash[b];
                w[x][y] = min(w[x][y], (LL)cost[i]);
            }

            // floyd
            for (int k = 1; k <= id; ++ k )
                for (int i = 1; i <= id; ++ i )
                    for (int j = 1; j <= id; ++ j )
                        w[i][j] = min(w[i][j], w[i][k] + w[k][j]);
        }

        int n = source.size();
        f[n] = 0;   // 边界值
        for (int i = n - 1; i >= 0; -- i ) {
            f[i] = INF;
            
            if (source[i] == target[i])
                f[i] = f[i + 1];

            // 从当前位置往后 找到可行串与对应消耗
            for (int j = i, p1 = 0, p2 = 0; j < n; ++ j ) {
                int u1 = source[j] - 'a', u2 = target[j] - 'a';
                p1 = son[p1][u1], p2 = son[p2][u2];
                if (p1 == 0 || p2 == 0)
                    break;
                // 如果存在两个对应的串
                if (mem[p1] && mem[p2])
                    // ATTENTION
                    f[i] = min(f[i], f[j + 1] + w[mem[p1]][mem[p2]]);
            }
        }

        if (f[0] >= INF / 2)
            return -1;
        return f[0];
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

### 离线 trie

> [!NOTE] **[LeetCode 1707. 与数组中元素的最大异或值](https://leetcode-cn.com/problems/maximum-xor-with-an-element-from-array/)**
> 
> [Weekly-210](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2020-12-27_Weekly-221)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 在线 Trie 做法贪心选择前面为 1 的时候结果会错，形如：
> 
> [536870912,0,534710168,330218644,142254206] [[558240772,1000000000],[307628050,1000000000],[3319300,1000000000],[2751604,683297522],[214004,404207941]] WRONG: [1050219420,844498962,540190212,539622516,-1] CORRECT: [1050219420,844498962,540190212,539622516,330170208]
> 
> 正确做法，离线 Trie

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
struct Node {
    int x, m, k;
    bool operator< (const Node& t) const {
        return m < t.m;
    }
}q[100010];

int son[3100000][2];

class Solution {
public:
    int idx = 0;
    void insert(int v) {
        int p = 0;
        for (int i = 30; i >= 0; -- i ) {
            int u = (v >> i) & 1;
            if (!son[p][u]) son[p][u] = ++ idx;
            p = son[p][u];
        }
    }
    int query(int v) {
        if (!idx) return -1;
        int p = 0, ret = 0;
        for (int i = 30; i >= 0; -- i ) {
            int u = (v >> i) & 1;
            if(!son[p][!u]) {
                // 相反的不存在 走相同的 当前这一位不产生异或增益
                p = son[p][u];
            } else {
                // 相反的存在 该位异或值为1
                p = son[p][!u];
                ret |= 1 << i;
            }
        }
        return ret;
    }
    vector<int> maximizeXor(vector<int>& nums, vector<vector<int>>& queries) {
        idx = 0;
        memset(son, 0, sizeof son);
        
        int n = nums.size(), m = queries.size();
        for (int i = 0; i < m; ++ i )
            q[i] = {queries[i][0], queries[i][1], i};
        sort(q, q + m);
        sort(nums.begin(), nums.end());
        vector<int> res(m);
        for (int i = 0, j = 0; j < m; ++ j ) {
            while (i < n && nums[i] <= q[j].m) insert(nums[i ++ ]);
            res[q[j].k] = query(q[j].x);
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

> [!NOTE] **[AcWing 1938. 查询最大基因差](https://leetcode-cn.com/problems/maximum-genetic-difference-query/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **Trie + 删除操作 + 离线处理**：
> 
> 另开一个 cnt 数组计数累计出现过的数量，为 0 时即便 son 数组非 0 也无效。
> 
> 结合 dfs 回溯离线处理答案

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 1e5 + 10, M = 2e5 * 18 + 10;
    
    int n, m, root;
    
    // ------------------ es ------------------
    int h[N], e[N], ne[N], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
        tot = 0;    // trie
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    // ------------------ trie ------------------
    int son[M][2], cnt[M], tot; // cnt for count
    void insert(int x) {
        int p = 0;
        for (int i = 30; i >= 0; -- i ) {
            int t = x >> i & 1;
            if (!son[p][t])
                son[p][t] = ++ tot ;
            p = son[p][t];
            cnt[p] ++ ; // ATTENTION
        }
    }
    void remove(int x) {
        int p = 0;
        for (int i = 30; i >= 0; -- i ) {
            int t = x >> i & 1;
            p = son[p][t];
            cnt[p] -- ;
        }
    }
    int query(int x) {
        int p = 0, res = 0;
        for (int i = 30; i >= 0; -- i ) {
            int t = x >> i & 1;
            // ATTENTION
            if (son[p][!t] && cnt[son[p][!t]]) {
                res |= 1 << i;
                p = son[p][!t];
            } else
                p = son[p][t];
        }
        return res;
    }
    
    // ------------------ dfs ------------------
    unordered_map<int, vector<PII>> qs;
    vector<int> res;
    void dfs(int u, int pa) {
        insert(u);
        
        for (auto [val, id] : qs[u])
            res[id] = query(val);
        
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            dfs(j, u);
        }
        remove(u);
    }
    
    vector<int> maxGeneticDifference(vector<int>& parents, vector<vector<int>>& queries) {
        init();
        
        this->n = parents.size();
        for (int i = 0; i < n; ++ i ) {
            if (parents[i] == -1)
                root = i;
            else
                add(parents[i], i);
        }
        
        this->m = queries.size();
        for (int i = 0; i < m; ++ i ) {
            int node = queries[i][0], val = queries[i][1];
            qs[node].push_back({val, i});
        }
        
        res = vector<int>(m);
        
        dfs(root, -1);
        
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

> [!NOTE] **[LeetCode 1707. 与数组中元素的最大异或值](https://leetcode-cn.com/problems/maximum-xor-with-an-element-from-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 在线 Trie 做法贪心选择前面为 1 的时候结果会错，形如：
> 
> [536870912,0,534710168,330218644,142254206] [[558240772,1000000000],[307628050,1000000000],[3319300,1000000000],[2751604,683297522],[214004,404207941]] WRONG: [1050219420,844498962,540190212,539622516,-1] CORRECT: [1050219420,844498962,540190212,539622516,330170208]
> 
> 正确做法，离线 Trie

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
struct Node {
    int x, m, k;
    bool operator< (const Node& t) const {
        return m < t.m;
    }
}q[100010];

int son[3100000][2];

class Solution {
public:
    int idx = 0;
    void insert(int v) {
        int p = 0;
        for (int i = 30; i >= 0; -- i ) {
            int u = (v >> i) & 1;
            if (!son[p][u]) son[p][u] = ++ idx;
            p = son[p][u];
        }
    }
    int query(int v) {
        if (!idx) return -1;
        int p = 0, ret = 0;
        for (int i = 30; i >= 0; -- i ) {
            int u = (v >> i) & 1;
            if(!son[p][!u]) {
                // 相反的不存在 走相同的 当前这一位不产生异或增益
                p = son[p][u];
            } else {
                // 相反的存在 该位异或值为1
                p = son[p][!u];
                ret |= 1 << i;
            }
        }
        return ret;
    }
    vector<int> maximizeXor(vector<int>& nums, vector<vector<int>>& queries) {
        idx = 0;
        memset(son, 0, sizeof son);
        
        int n = nums.size(), m = queries.size();
        for (int i = 0; i < m; ++ i )
            q[i] = {queries[i][0], queries[i][1], i};
        sort(q, q + m);
        sort(nums.begin(), nums.end());
        vector<int> res(m);
        for (int i = 0, j = 0; j < m; ++ j ) {
            while (i < n && nums[i] <= q[j].m) insert(nums[i ++ ]);
            res[q[j].k] = query(q[j].x);
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

> [!NOTE] **[LeetCode 1938. 查询最大基因差](https://leetcode-cn.com/problems/maximum-genetic-difference-query/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 离线处理即可
> 
> **涉及到 Trie 中删除元素 思路**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 1e5 + 10, M = 2e5 * 18 + 10;
    
    int n, m, root;
    
    // ------------------ es ------------------
    int h[N], e[N], ne[N], idx;
    void init() {
        memset(h, -1, sizeof h);
        idx = 0;
        tot = 0;    // trie
    }
    void add(int a, int b) {
        e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
    }
    
    // ------------------ trie ------------------
    int son[M][2], cnt[M], tot; // cnt for count
    void insert(int x) {
        int p = 0;
        for (int i = 30; i >= 0; -- i ) {
            int t = x >> i & 1;
            if (!son[p][t])
                son[p][t] = ++ tot ;
            p = son[p][t];
            cnt[p] ++ ; // ATTENTION
        }
    }
    void remove(int x) {
        int p = 0;
        for (int i = 30; i >= 0; -- i ) {
            int t = x >> i & 1;
            p = son[p][t];
            cnt[p] -- ;
        }
    }
    int query(int x) {
        int p = 0, res = 0;
        for (int i = 30; i >= 0; -- i ) {
            int t = x >> i & 1;
            // ATTENTION
            if (son[p][!t] && cnt[son[p][!t]]) {
                res |= 1 << i;
                p = son[p][!t];
            } else
                p = son[p][t];
        }
        return res;
    }
    
    // ------------------ dfs ------------------
    unordered_map<int, vector<PII>> qs;
    vector<int> res;
    void dfs(int u, int pa) {
        insert(u);
        
        for (auto [val, id] : qs[u])
            res[id] = query(val);
        
        for (int i = h[u]; ~i; i = ne[i]) {
            int j = e[i];
            if (j == pa)
                continue;
            dfs(j, u);
        }
        remove(u);
    }
    
    vector<int> maxGeneticDifference(vector<int>& parents, vector<vector<int>>& queries) {
        init();
        
        this->n = parents.size();
        for (int i = 0; i < n; ++ i ) {
            if (parents[i] == -1)
                root = i;
            else
                add(parents[i], i);
        }
        
        this->m = queries.size();
        for (int i = 0; i < m; ++ i ) {
            int node = queries[i][0], val = queries[i][1];
            qs[node].push_back({val, i});
        }
        
        res = vector<int>(m);
        
        dfs(root, -1);
        
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

> [!NOTE] **[LeetCode 3045. 统计前后缀下标对 II](https://leetcode.cn/problems/count-prefix-and-suffix-pairs-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 题意分析显然需要结合离线 trie，结合扩展 kmp 进行快速校验
> 
> 1. 较直观的思路是：逆序遍历，并将当前串的所有前后缀插入 trie 以供后续遍历到的串做全匹配；本质是枚举当前串来计算能匹配多少个之前出现过的前后缀【由于生成前后缀的插入复杂度 TLE】
> 
> 2. 其实可以换个思路，枚举当前串看前面有多少串可以作为当前串的前后缀【无需插入前后缀，降低复杂度】
> 
> 3. 更进一步，可以在插入同时统计，进一步简化实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 逆序 TLE**

```cpp
// 总长度不超过 5e5
//  与 leetcode 前面有道题类似，本质是把后缀也插入到 trie
//  => 还不够，需要对于当前串求出所有的公共前后缀，然后插入公共前后缀...

using LL = long long;
const static int N = 5e5 + 10;

int son[N][26], idx;
int cnt[N];

void init() {
    memset(son, 0, sizeof son);
    idx = 0;
    memset(cnt, 0, sizeof cnt);
}

void insert(string & s) {
    int p = 0;
    for (auto c : s) {
        int u = c - 'a';
        if (!son[p][u])
            son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;     // ATTENTION cnt 操作时机
}

LL query(string & s) {
    int p = 0;
    for (auto c : s) {
        int u = c - 'a';
        if (!son[p][u])
            return 0;
        p = son[p][u];
    }
    return cnt[p];
}

class Solution {
public:
    vector<int> z_func(string & s) {
        int n = s.size();
        vector<int> z(n);
        for (int i = 1, l = 0, r = 0; i < n; ++ i ) {
            if (i <= r && z[i - l] < r - i + 1)
                z[i] = z[i - l];
            else {
                z[i] = max(0, r - i + 1);
                while (i + z[i] < n && s[z[i]] == s[i + z[i]])
                    z[i] ++ ;
            }
            if (i + z[i] - 1 > r)
                l = i, r = i + z[i] - 1;
        }
        z[0] = n;
        return z;
    }
    
    long long countPrefixSuffixPairs(vector<string>& words) {
        init();
        
        int n = words.size();
        LL res = 0;
        for (int i = n - 1; i >= 0; -- i ) {
            auto & w = words[i];
            // cout << " i = " << i << " w = " << w << " q = " << query(w) << endl;
            res += query(w);
            
            auto z = z_func(w);
            for (int i = 0; i < w.size(); ++ i )
                if (i + z[i] == w.size()) {
                    string t = w.substr(i);
                    // cout << "  at i = " << i << " will insert t = " << t << endl;
                    insert(t);
                }
        }
        return res;
    }
};
```

##### **C++ 正序**

```cpp
// 总长度不超过 5e5
//  与 leetcode 前面有道题类似，本质是把后缀也插入到 trie
//  => 还不够，需要对于当前串求出所有的公共前后缀，然后插入公共前后缀...

using LL = long long;
const static int N = 5e5 + 10;

vector<int> z_func(string & s) {
    int n = s.size();
    vector<int> z(n);
    for (int i = 1, l = 0, r = 0; i < n; ++ i ) {
        if (i <= r && z[i - l] < r - i + 1)
            z[i] = z[i - l];
        else {
            z[i] = max(0, r - i + 1);
            while (i + z[i] < n && s[z[i]] == s[i + z[i]])
                z[i] ++ ;
        }
        if (i + z[i] - 1 > r)
            l = i, r = i + z[i] - 1;
    }
    z[0] = n;
    return z;
}

int son[N][26], idx;
int cnt[N];

void init() {
    memset(son, 0, sizeof son);
    idx = 0;
    memset(cnt, 0, sizeof cnt);
}

void insert(string & s) {
    int p = 0;
    for (auto c : s) {
        int u = c - 'a';
        if (!son[p][u])
            son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;     // ATTENTION cnt 操作时机
}

LL query(string & s) {
    auto z = z_func(s);
    int p = 0, n = z.size();
    LL ret = 0;
    for (int i = 0; i < n; ++ i ) {
        auto c = s[i];
        int u = c - 'a';
        if (!son[p][u])
            break;  // ATTENTION
        p = son[p][u];
        
        // ATTENTION 在这里追加统计逻辑
        if (z[n - 1 - i] == i + 1)  // 是后缀
            ret += cnt[p];
    }
    return ret;
}

class Solution {
public:
    
    long long countPrefixSuffixPairs(vector<string>& words) {
        init();
        
        int n = words.size();
        LL res = 0;
        // ATTENTION: 正序，在插入的同时计算【前置计算过程】
        for (int i = 0; i < n; ++ i ) {
            auto & w = words[i];
            res += query(w);
            insert(w);   // 本质也可以和 query 结合
        }
        return res;
    }
};
```

##### **C++ 正序 简化**

```cpp
// 总长度不超过 5e5
//  与 leetcode 前面有道题类似，本质是把后缀也插入到 trie
//  => 还不够，需要对于当前串求出所有的公共前后缀，然后插入公共前后缀...

using LL = long long;
const static int N = 5e5 + 10;

vector<int> z_func(string & s) {
    int n = s.size();
    vector<int> z(n);
    for (int i = 1, l = 0, r = 0; i < n; ++ i ) {
        if (i <= r && z[i - l] < r - i + 1)
            z[i] = z[i - l];
        else {
            z[i] = max(0, r - i + 1);
            while (i + z[i] < n && s[z[i]] == s[i + z[i]])
                z[i] ++ ;
        }
        if (i + z[i] - 1 > r)
            l = i, r = i + z[i] - 1;
    }
    z[0] = n;
    return z;
}

int son[N][26], idx;
int cnt[N];

void init() {
    memset(son, 0, sizeof son);
    idx = 0;
    memset(cnt, 0, sizeof cnt);
}

LL insert(string & s) {
    auto z = z_func(s);
    int p = 0, n = z.size();
    LL ret = 0;
    for (int i = 0; i < n; ++ i ) {
        auto c = s[i];
        int u = c - 'a';
        if (!son[p][u])
            son[p][u] = ++ idx;
        p = son[p][u];
        
        // ATTENTION 在这里追加统计逻辑
        if (z[n - 1 - i] == i + 1)  // 是后缀
            ret += cnt[p];
    }
    cnt[p] ++ ;     // ATTENTION cnt 操作时机
    return ret;
}

class Solution {
public:
    
    long long countPrefixSuffixPairs(vector<string>& words) {
        init();
        LL res = 0;
        // ATTENTION: 正序，在插入的同时计算【前置计算过程】
        for (auto & w : words)
            res += insert(w);   // inert 和 query 结合
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

### trie 动态拓展 + 树 hash

> [!NOTE] **[LeetCode 1948. 删除系统中的重复文件夹](https://leetcode-cn.com/problems/delete-duplicate-folders-in-system/)**
> 
> [weekly-251](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-07-25_Weekly-251)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然树Hash 之前题意理解有问题
> 
> 需要注意题意并没有要求返回的结果必须是输入的子集，所以可以直接回溯构造

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using ULL = unsigned long long;
    const static ULL P = 131;
    
    // hash
    ULL shash(string & s) {
        ULL ret = 0;
        for (auto c : s)
            ret = ret * P + c;
        return ret;
    }
    
    // Node
    struct Node {
        string name;
        unordered_map<string, Node*> sons;
        ULL hash = 0;
        bool del = false;
        Node(string name) : name(name) {}
    };
    Node * root;
    unordered_map<ULL, vector<Node*>> mp;
    void insert(vector<string> & pth) {
        auto p = root;
        for (auto & x : pth) {
            if (!p->sons.count(x))
                p->sons[x] = new Node(x);
            p = p->sons[x];
        }
        // Nothing
    }
    
    // dfs
    void dfs(Node * p) {
        string s;
        for (auto [name, node] : p->sons) {
            dfs(node);
            // 不加 node->name 就会多删   即加了hash才有区分度
            s += "[" + node->name + to_string(node->hash) + "]";
        }
        p->hash = shash(s);
        // 根据题意解释 必然是非空集合
        if (p->sons.size())
            mp[p->hash].push_back(p);
    }
    vector<string> t;
    vector<vector<string>> res;
    // 直接回溯建立答案列表
    // ATTENTION 题意并没有说输入有漏掉某个可能路径OR输出必须包含于输入 so直接构造所有的可能路径
    void get_res(Node * p) {
        if (p->del)
            return
        if (p->name != "/") {
            t.push_back(p->name);
            res.push_back(t);
        }
        for (auto [name, node] : p->sons)
            get_res(node);
        if (p->name != "/")
            t.pop_back();
    }
    
    vector<vector<string>> deleteDuplicateFolder(vector<vector<string>>& paths) {
        root = new Node("/");
        for (auto & pth : paths)
            insert(pth);
        
        dfs(root);
        for (auto [hash, nodes] : mp)
            if (nodes.size() > 1)
                for (auto node : nodes)
                    node->del = true;
        
        get_res(root);
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

### TODO 思想 结合其他知识

> [!NOTE] **[Luogu P3294 [SCOI2016]背单词](https://www.luogu.com.cn/problem/P3294)**
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

const int N = 100010, M = 600010;
int n, tr[M][26], flag[M], idx_t;  // trie
string str[N];
int h[M], ptr[N], val[N], sz[M], idx_g;  // 链式前向星
void insert(string str) {
    int p = 0, c = str[0] - 'a';
    for (int i = 0; i < (int)str.size(); i++, c = str[i] - 'a')
        p = ((tr[p][c]) ? tr[p][c] : (tr[p][c] = ++idx_t));
    flag[p] = true;  // 标记结尾
}
// 加边函数
void add(int a, int b) { val[idx_g] = b, ptr[idx_g] = h[a], h[a] = idx_g++; }
void update(int p, int top) {
    // 如果 v 是结尾，加边
    for (int i = 0; i < 26; i++)
        if (flag[tr[p][i]])
            add(top, tr[p][i]);
    for (int i = 0; i < 26; i++)
        if (flag[tr[p][i]])
            update(tr[p][i], tr[p][i]);
        else if (tr[p][i])
            update(tr[p][i], top);
}
long long res = 0;
void DFS(int u) {
    // 计算子树大小
    sz[u] = 1;
    for (int i = h[u]; i != -1; i = ptr[i])
        DFS(val[i]), sz[u] += sz[val[i]];
    // 贪心计算答案
    vector<int> temp;
    for (int i = h[u]; i != -1; i = ptr[i])
        temp.push_back(sz[val[i]]);
    sort(temp.begin(), temp.end());
    int delta = 1;  // 偏移量
    for (int i = 0; i < (int)temp.size(); i++)
        res += delta, delta += temp[i];
}
int main() {
    cin >> n;
    for (int i = 1; i <= n; i++)
        cin >> str[i];
    for (int i = 1; i <= n; i++)
        reverse(str[i].begin(), str[i].end());  // 翻转
    for (int i = 1; i <= n; i++)
        insert(str[i]);  // 插入
    memset(h, -1, sizeof h);
    update(0, 0), DFS(0);
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

