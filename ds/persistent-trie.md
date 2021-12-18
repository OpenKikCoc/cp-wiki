可持久化 Trie 的方式和可持久化线段树的方式是相似的，即每次只修改被添加或值被修改的节点，而保留没有被改动的节点，在上一个版本的基础上连边，使最后每个版本的 Trie 树的根遍历所能分离出的 Trie 树都是完整且包含全部信息的。

大部分的可持久化 Trie 题中，Trie 都是以 [01-Trie](https://oi-wiki.org/string/trie/#_5) 的形式出现的。

> [!NOTE] **例题[最大异或和](https://www.luogu.com.cn/problem/P4735)**
> 
> 对一个长度为 $n$ 的数组 $a$ 维护以下操作：
> 
> 1. 在数组的末尾添加一个数 $x$，数组的长度 $n$ 自增 $1$。
> 
> 2. 给出查询区间 $[l,r]$ 和一个值 $k$，求当 $l\le p\le r$ 时，$k \oplus \bigoplus^{n}_{i=p} a_i$。

这个求的值可能有些麻烦，利用常用的处理连续异或的方法，记 $s_x=\bigoplus_{i=1}^x a_i$，则原式等价于 $s_{p-1}\oplus s_n\oplus k$，观察到 $s_n \oplus k$ 在查询的过程中是固定的，题目的查询变化为查询在区间 $[l-1,r-1]$ 中异或定值（$s_n\oplus k$）的最大值。

继续按类似于可持久化线段树的思路，考虑每次的查询都查询整个区间。我们只需把这个区间建一棵 Trie 树，将这个区间中的每个树都加入这棵 Trie 中，查询的时候，尽量往与当前位不相同的地方跳。

查询区间，只需要利用前缀和和差分的思想，用两棵前缀 Trie 树（也就是按顺序添加数的两个历史版本）相减即得到该区间的 Trie 树。再利用动态开点的思想，不添加没有计算过的点，以减少空间占用。


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


> [!NOTE] **[AcWing 256. 最大异或和](https://www.acwing.com/problem/content/258/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 可持久化的前提：本身的拓扑结构，在操作过程中不变
> 
> 可持久化作用：记录数据结构的所有历史版本
> 
> 核心思想：只记录每一个版本与前一个版本不一样的地方
> 
>     对于Trie 每次操作最多只有logn位置不同
> 
> Trie的持久化
> 
> 本题：
> 
> 考虑前缀和(异或)数组 Si = A1 ^ A2 ^ ... ^ Ai
> 
> 则 Ap ^ Ap+1 ^ ... ^ An ^ x  ==   Sp-1 ^ Sn ^ x
> 
> 则每次相当于Sn ^ x固定 左侧[L, R-1]找个值
> 
> 相当于比之前单纯的Trie加了一个范围限制 如果只有R限制 则前R版本即可
> 
> 同时有L和R 则要求子树中至少存在一个数它的下标>=L
> 
> 故在每个节点记录当前子树中下标的最大值 max-id
> 
> ==> 可持久化处理R要求 maxid处理L要求

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

// N 原始数据个数 + 操作个数
// 每次操作最多建立24个节点 加上根节点是25
const int N = 600010, M = N * 25;   // 数据范围 10^7 最多24位

int n, m;
int s[N];
int tr[M][2], max_id[M];
int root[N], idx;

// i 下标
// k 当前处理到第几位
// p 上一个版本
// q 当前版本
void insert(int i, int k, int p, int q) {
    if (k < 0) {
        // 已经处理完了
        max_id[q] = i;
        return;
    }
    // 当前位
    int v = s[i] >> k & 1;
    // 另外一位 使用旧的
    // 对于01trie 直接if判断 但对于更多字符的情况 加一个for就好了 每一个节点都继承
    if (p) tr[q][v ^ 1] = tr[p][v ^ 1];
    tr[q][v] = ++ idx;
    // 插入下一位
    insert(i, k - 1, tr[p][v], tr[q][v]);
    // 同样 对于更多字符的情况 加个for循环
    max_id[q] = max(max_id[tr[q][0]], max_id[tr[q][1]]);
}

int query(int root, int C, int L) {
    int p = root;
    for (int i = 23; i >= 0; -- i ) {
        int v = C >> i & 1;
        // id 大于L
        if (max_id[tr[p][v ^ 1]] >= L) p = tr[p][v ^ 1];
        else p = tr[p][v];
    }
    // 最后只有一个点 max_id 就是这个点自己的下标
    return C ^ s[max_id[p]];
}

int main() {
    scanf("%d%d", &n, &m);
    
    // 比任何节点都小 所以记录为-1
    max_id[0] = -1;
    root[0] = ++ idx;           // 第0个版本
    insert(0, 23, 0, root[0]);  // 地柜形式比较好
    
    for (int i = 1; i <= n; ++ i ) {
        int x;
        scanf("%d", &x);
        s[i] = s[i - 1] ^ x;
        root[i] = ++ idx;
        insert(i, 23, root[i - 1], root[i]);
    }
    
    char op[2];
    int l, r, x;
    while (m -- ) {
        scanf("%s", op);
        if (*op == 'A') {
            scanf("%d", &x);
            ++ n ;
            s[n] = s[n - 1] ^ x;
            root[n] = ++ idx;
            insert(n, 23, root[n - 1], root[n]);
        } else {
            scanf("%d%d%d", &l, &r, &x);
            printf("%d\n", query(root[r - 1], s[n] ^ x, l - 1));
        }
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