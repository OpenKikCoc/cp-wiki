
![](images/disjoint-set.svg)

并查集是一种树形的数据结构，顾名思义，它用于处理一些不交集的 **合并** 及 **查询** 问题。
它支持两种操作：

- 查找（Find）：确定某个元素处于哪个子集；
- 合并（Union）：将两个子集合并成一个集合。

> [!WARNING]
> 
> 并查集不支持集合的分离，但是并查集在经过修改后可以支持集合中单个元素的删除操作（详见 UVA11987 Almost Union-Find）。使用动态开点线段树还可以实现可持久化并查集。

## 初始化


## 查找

通俗地讲一个故事：几个家族进行宴会，但是家族普遍长寿，所以人数众多。由于长时间的分离以及年龄的增长，这些人逐渐忘掉了自己的亲人，只记得自己的爸爸是谁了，而最长者（称为「祖先」）的父亲已经去世，他只知道自己是祖先。为了确定自己是哪个家族，他们想出了一个办法，只要问自己的爸爸是不是祖先，一层一层的向上问，直到问到祖先。如果要判断两人是否在同一家族，只要看两人的祖先是不是同一人就可以了。

在这样的思想下，并查集的查找算法诞生了。

![](images/disjoint-set-find.svg)

显然这样最终会返回 $x$ 的祖先。

### 路径压缩

这样的确可以达成目的，但是显然效率实在太低。为什么呢？因为我们使用了太多没用的信息，我的祖先是谁与我父亲是谁没什么关系，这样一层一层找太浪费时间，不如我直接当祖先的儿子，问一次就可以出结果了。甚至祖先是谁都无所谓，只要这个人可以代表我们家族就能得到想要的效果。**把在路径上的每个节点都直接连接到根上**，这就是路径压缩。

![](images/disjoint-set-compress.svg)

## 合并

宴会上，一个家族的祖先突然对另一个家族说：我们两个家族交情这么好，不如合成一家好了。另一个家族也欣然接受了。  
我们之前说过，并不在意祖先究竟是谁，所以只要其中一个祖先变成另一个祖先的儿子就可以了。

![](images/disjoint-set-merge.svg)

### 启发式合并（按秩合并）

一个祖先突然抖了个机灵：「你们家族人比较少，搬家到我们家族里比较方便，我们要是搬过去的话太费事了。」

由于需要我们支持的只有集合的合并、查询操作，当我们需要将两个集合合二为一时，无论将哪一个集合连接到另一个集合的下面，都能得到正确的结果。但不同的连接方法存在时间复杂度的差异。具体来说，如果我们将一棵点数与深度都较小的集合树连接到一棵更大的集合树下，显然相比于另一种连接方案，接下来执行查找操作的用时更小（也会带来更优的最坏时间复杂度）。

当然，我们不总能遇到恰好如上所述的集合————点数与深度都更小。鉴于点数与深度这两个特征都很容易维护，我们常常从中择一，作为估价函数。而无论选择哪一个，时间复杂度都为 $O (m\alpha(m,n))$，具体的证明可参见 References 中引用的论文。

在算法竞赛的实际代码中，即便不使用启发式合并，代码也往往能够在规定时间内完成任务。在 Tarjan 的论文[1]中，证明了不使用启发式合并、只使用路径压缩的最坏时间复杂度是 $O (m \log n)$。在姚期智的论文[2]中，证明了不使用启发式合并、只使用路径压缩，在平均情况下，时间复杂度依然是 $O (m\alpha(m,n))$。

如果只使用启发式合并，而不使用路径压缩，时间复杂度为 $O(m\log n)$。由于路径压缩单次合并可能造成大量修改，有时路径压缩并不适合使用。例如，在可持久化并查集、线段树分治 + 并查集中，一般使用只启发式合并的并查集。

C++ 的参考实现，其选择点数作为估价函数：

```cpp
// C++ Version
std::vector<int> size(N, 1);  // 记录并初始化子树的大小为 1
void unionSet(int x, int y) {
    int xx = find(x), yy = find(y);
    if (xx == yy) return;
    if (size[xx] > size[yy])  // 保证小的合到大的里
        swap(xx, yy);
    fa[xx] = yy;
    size[yy] += size[xx];
}
```

Python 的参考实现：

```python
# Python Version
size = [1] * N # 记录并初始化子树的大小为 1
def unionSet(x, y):
    xx = find(x); yy = find(y)
    if xx == yy:
        return
    if size[xx] > size[yy]: # 保证小的合到大的里
        xx, yy = yy, xx
    fa[xx] = yy
    size[yy] = size[yy] + size[xx]
```

## 时间复杂度及空间复杂度

### 时间复杂度

同时使用路径压缩和启发式合并之后，并查集的每个操作平均时间仅为 $O(\alpha(n))$，其中 $\alpha$ 为阿克曼函数的反函数，其增长极其缓慢，也就是说其单次操作的平均运行时间可以认为是一个很小的常数。

[Ackermann 函数](https://en.wikipedia.org/wiki/Ackermann_function)  $A(m, n)$ 的定义是这样的：

$A(m, n) = \begin{cases}n+1&\text{if }m=0\\A(m-1,1)&\text{if }m>0\text{ and }n=0\\A(m-1,A(m,n-1))&\text{otherwise}\end{cases}$

而反 Ackermann 函数 $\alpha(n)$ 的定义是阿克曼函数的反函数，即为最大的整数 $m$ 使得 $A(m, m) \leqslant n$。

时间复杂度的证明 [在这个页面中](./dsu-complexity.md)。

### 空间复杂度

显然为 $O(n)$。

## 带权并查集

我们还可以在并查集的边上定义某种权值、以及这种权值在路径压缩时产生的运算，从而解决更多的问题。比如对于经典的「NOI2001」食物链，我们可以在边权上维护模 3 意义下的加法群。

## 经典题目

[「NOI2015」程序自动分析](https://uoj.ac/problem/127)

[「JSOI2008」星球大战](https://www.luogu.com.cn/problem/P1197)

[「NOI2001」食物链](https://www.luogu.com.cn/problem/P2024)

[「NOI2002」银河英雄传说](https://www.luogu.com.cn/problem/P1196)

[UVA11987 Almost Union-Find](https://www.luogu.com.cn/problem/UVA11987)

## 其他应用

[最小生成树算法](graph/mst.md) 中的 Kruskal 和 [最近公共祖先](graph/lca.md) 中的 Tarjan 算法是基于并查集的算法。

相关专题见 [并查集应用](topic/dsu-app.md)。

## 参考资料与拓展阅读

- [1]Tarjan, R. E., & Van Leeuwen, J. (1984). Worst-case analysis of set union algorithms. Journal of the ACM (JACM), 31(2), 245-281.[ResearchGate PDF](https://www.researchgate.net/profile/Jan_Van_Leeuwen2/publication/220430653_Worst-case_Analysis_of_Set_Union_Algorithms/links/0a85e53cd28bfdf5eb000000/Worst-case-Analysis-of-Set-Union-Algorithms.pdf)
- [2]Yao, A. C. (1985). On the expected performance of path compression algorithms.[SIAM Journal on Computing, 14(1), 129-133.](https://epubs.siam.org/doi/abs/10.1137/0214010?journalCode=smjcat)
- [3][知乎回答：是否在并查集中真的有二分路径压缩优化？](<https://www.zhihu.com/question/28410263/answer/40966441>)
- [4]Gabow, H. N., & Tarjan, R. E. (1985). A Linear-Time Algorithm for a Special Case of Disjoint Set Union. JOURNAL OF COMPUTER AND SYSTEM SCIENCES, 30, 209-221.[CORE PDF](https://core.ac.uk/download/pdf/82125836.pdf)


## 习题

> [!NOTE] **[AcWing 836. 合并集合](https://www.acwing.com/problem/content/838/)**
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

int p[N], n, m;

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) p[i] = i;
    
    char op[2];
    int a, b;
    while (m -- ) {
        cin >> op >> a >> b;
        a = find(a), b = find(b);
        if (op[0] == 'M') {
            p[a] = b;
        } else {
            if (a == b) cout << "Yes" << endl;
            else cout << "No" << endl;
        }
    }
    return 0;
}
```

##### **Python**

```python
"""
I. 概念
1. 并查集：主要用于解决一些 **元素分组** 的问题，它管理一系列不相交的集合，并支持两种操作：
   - 合并：把两个不相交的集合合并成一个集合
   - 查询：查询两个元素是否在同一集合中
2. 基本原理：每个集合用一棵树来表示；树根的编号就是整个集合的编号 父节点，p[x]表示x的父节点

II. 理解模版代码
1. 如何判断树根？ if p[x]==x
2. 如何求x的集合编号： while p(x)!=x:x=p[x]
3. 如何合并两个集合：px是x的集合编号，py是y的集合编号：p[x]=y

"""

N = 100010
p = [0] * N


# 查询：用递归的写法实现对元素的查询：一层一层访问父节点，直到祖宗节点（根节点的标志就是父节点是本身
# 要判断两个元素是否属于同一集合，只需要看它们的根节点是否相同即可

def find(x):  # 递归回溯的条件是 p[x] == x, 即找到祖宗节点
    if p[x] != x:  # x的父节点不是祖宗节点
        p[x] = find(p[x])  # 找寻父节点的祖宗节点
    return p[x]


if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):  # 初始化每个元素的祖宗节点为本身
        p[i] = i
    for _ in range(m):
        oper = input().split()
        if oper[0] == 'M':
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b): continue

            # 合并：先找到两个集合的代表元素，然后将前者的父节点设为后者即可
            p[find(a)] = find(b)  # 使a的祖宗节点变成b的祖宗节点
            # 踩坑：p[find(a)] 是把b的祖宗节点 肤质给 找到的a的祖宗节点！错误写法： p[a] = find(b) !xxx 非常错！ 踩坑好几次了！！！
        else:
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b):
                print("Yes")
            else:
                print("No")
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 837. 连通块中点的数量](https://www.acwing.com/problem/content/839/)**
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

int n, m;
int p[N], sz[N];

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i ) p[i] = i, sz[i] = 1;
    
    char op[3];
    int a, b;
    while (m -- ) {
        cin >> op;
        if (op[0] == 'C') {
            cin >> a >> b;
            a = find(a), b = find(b);
            if (a != b) {
                // sz 相加要放在前面，不然先修改父节点就没法算了
                sz[b] += sz[a];
                p[a] = b;
            }
        } else if (op[0] == 'Q' && op[1] == '1') {
            cin >> a >> b;
            a = find(a), b = find(b);
            cout << (a == b ? "Yes" : "No") << endl;
        } else {
            cin >> a;
            cout << sz[find(a)] << endl;
        }
    }
    return 0;
}
```

##### **Python**

```python
N = 100010
p = [0] * N
size = [1] * N  # 初始化 根节点的值为该树的节点数1


def find(x):
    if p[x] != x:
        p[x] = find(p[x])
    return p[x]


if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(1, n + 1):  # 初始化每个元素的树，让根节点等于自己。
        p[i] = i

    for _ in range(m):
        oper = input().split()
        if oper[0] == 'C':
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b): continue  # 特判：两个点在一个集合中就跳过合并
            size[find(b)] += size[find(a)]  # 更新新树的size，注意！！需要先将size合并才能p[a]=b; p[a]=b操作之后a的根就是b了，会出错
            p[find(a)] = find(b)  # a插入b中
        elif oper[0] == 'Q1':
            a, b = int(oper[1]), int(oper[2])
            if find(a) == find(b):
                print("Yes")
            else:
                print("No")
        else:
            a = int(oper[1])
            print(size[find(a)])
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 240. 食物链](https://www.acwing.com/problem/content/242/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

使用 0/1/2 表示同类、吃、被吃的关系

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 50010;

int n, m;
int p[N], d[N];

int find(int x) {
    if (p[x] != x) {
        int root = find(p[x]);
        d[x] += d[p[x]];
        p[x] = root;
    }
    return p[x];
}

int main() {
    cin >> n >> m;
    
    for (int i = 1; i <= n; ++ i ) p[i] = i;
    
    int res = 0;
    int t, x, y;
    while (m -- ) {
        cin >> t >> x >> y;
        if (x > n || y > n) ++ res ;
        else {
            int px = find(x), py = find(y);
            if (t == 1) {
                if (px == py && (d[x] - d[y]) % 3) ++ res ;
                else if (px != py) {
                    p[px] = py;
                    d[px] = d[y] - d[x];
                }
            } else {
                if (px == py && (d[x] - d[y] - 1) % 3) ++ res ;
                else if (px != py) {
                    p[px] = py;
                    d[px] = d[y] + 1 - d[x];
                }
            }
        }
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