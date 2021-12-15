## 例题

> [!NOTE] **[「SCOI2005」互不侵犯](https://loj.ac/problem/2153)**
> 
> 在 $N\times N$ 的棋盘里面放 $K$ 个国王，使他们互不攻击，共有多少种摆放方案。国王能攻击到它上下左右，以及左上左下右上右下八个方向上附近的各一个格子，共 $8$ 个格子。

> [!TIP] 思路
> 
> 我们用 $f(i,j,l)$ 表示只考虑前 $i$ 行，第 $i$ 行按照编号为 $j$ 的状态放置国王，且已经放置 $l$ 个国王时的方案数。
> 
> 对于编号为 $j$ 的状态，我们用二进制整数 $sit(j)$ 表示国王的放置情况，$sit(j)$ 的某个二进制位为 $0$ 表示对应位置不放国王，为 $1$ 表示在对应位置上放置国王；
> 
> 用 $sta(j)$ 表示该状态的国王个数，即二进制数 $sit(j)$ 中 $1$ 的个数。例如，如下图所示的状态可用二进制数 $100101$ 来表示（棋盘左边对应二进制低位），则有 $sit(j)=100101_{(2)}=37, sta(j)=3$。
> 
> ![](./images/SCOI2005-互不侵犯.png)
> 
> 我们需要在刚开始的时候枚举出所有的合法状态（即排除同一行内两个国王相邻的不合法情况），并计算这些状态的 $sit(j)$ 和 $sta(j)$。
> 
> 设上一行的状态编号为 $x$，在保证当前行和上一行不冲突的前提下，枚举所有可能的 $x$ 进行转移，转移方程：
> 
> $$
> f(i,j,l) = \sum f(i-1,x,l-sta(j))
> $$

TODO@binacs

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

[NOI2001 炮兵阵地](https://loj.ac/problem/10173)

[「USACO06NOV」玉米田 Corn Fields](https://www.luogu.com.cn/problem/P1879)

[九省联考 2018 一双木棋](https://loj.ac/problem/2471)

[UVA10817 校长的烦恼 Headmaster's Headache](https://www.luogu.com.cn/problem/UVA10817)

[UVA1252 20 个问题 Twenty Questions](https://www.luogu.com.cn/problem/UVA1252)

> [!NOTE] **[AcWing 291. 蒙德里安的梦想](https://www.acwing.com/problem/content/293/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 先求每个状态是否合法 再状态转移累加合法数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 12, M = 1 << N;
int st[M];
long long f[N][M];

int main() {
    int n, m;
    while (cin >> n >> m && (n || m)) {
        for (int i = 0; i < 1 << n; ++i) {
            int cnt = 0;  // 连续空格的数量
            st[i] = true;
            for (int j = 0; j < n; ++j)
                if (i >> j & 1) {
                    if (cnt & 1)
                        break;  // 为啥不break?==> st[i]=false 可以换成break
                    cnt = 0;
                } else
                    ++cnt;
            if (cnt & 1) st[i] = false;
        }
        memset(f, 0, sizeof f);
        f[0][0] = 1;
        for (int i = 1; i <= m; ++i)
            for (int j = 0; j < 1 << n; ++j)
                for (int k = 0; k < 1 << n; ++k)
                    if ((j & k) == 0 && st[j | k])
                        // j & k == 0 表示 i 列和 i - 1列同一行不同时捅出来
                        // st[j | k] == 1 表示 在 i 列状态 j， i - 1 列状态 k
                        // 的情况下是合法的.
                        f[i][j] += f[i - 1][k];
        cout << f[m][0] << endl;
    }
}
```

##### **Python**

```python
# 1. 核心：先放横着的，再放竖着的。
# 2. 总的方案数：如果只放横着的小方块的话，所有的合法的方案数有多少种。
# 3.如何判断当前方式是不是合法的？==> 当摆放完横的小方块后，所有剩余的位置能够填充满竖的小方块。（可以的话，就是合法的）
# 如何判断 是否能填满竖的小方块呢？ ==> 可以按列看，每一列内部所有连续的空着的小方块的数量需要是偶数个。

# 状态表示（化零为整）：f[i,j]表示已经将前i-1列摆好，且从第i-1列伸出到第i列的状态为j的所有方案数；
# 状态转移（化整为零）：把f[i,j]做分割。

# 化零为整(状态转移): 分割的时候，一般是在找最后一个不同点，以此作为分割。
# f[i,j]表示第i-1列伸到第i列的方案已经固定，那最后没有固定的就是从第i-2列伸到第i-1列的状态，以此来分割为若干种。可以分割成pow(2,n)种，每一行 都有两种选择-伸/不伸。===> f[i,j]：最多会被划分成为pow(2,n)的子集

# 每个子集都表示一个具体的状态，比如k(也是一个二进制数表示的)，比如00100（只有第三行是伸出来的）
# f[i,j]所有的集合的倒数第二步一定是属于这个子集之一，这种划分方案是一定是不重不漏；f[i,j]总共方案数，就是每一个子集的方案数之和。

# 那每个子集的方案数怎么求？
# 假设第i-2列伸到第i-1列的状态是k，第i-1列伸到第i列的状态是j，那方案 数是f[i-1,k]，但能不能转移过来是一个问题？什么情况下j和k可以拼在一起构成合法方案呢？
# 1）j和k不能在同一行都有1 ：需要满足j&k==0
# 2）对于i-1列来说，空的小方块的位置是固定的：空着的小方块可以被竖着的1*2小方块塞满，那就是所有空着的连续的位置的长度必须是偶数。

# 最后返回的方案数怎么表示呢？f[m][0]：m其实是m+1列（下标从0开始），指的是前m列已经伸好了，且从m列伸到m+1列的状态是0的所有方案。恰好就是摆满n*m的所有方案。


if __name__ == "__main__":
    N = 12
    M = 1 << N  # pow(2,n)
    f = [[0] * M for _ in range(N)]
    st = [False] * M  # 判断某个状态是否合法，也就是判断当前这一列所有空着的连续位置是否是偶数。

    n, m = map(int, input().split())
    # 首先预处理st数组，
    while n or m:
        # 清空还原初始值数组
        f = [[0] * M for _ in range(N)]
        st = [False] * M
        # 预处理:1)cnt记录连续0的个数
        # 2）对于i 循环遍历每次右移1位，判断当前数 是否为1:a.如果为1，那么就判断之前连续的0是否是偶数个，如果是偶数个，就直接将cnt置为0，如果是奇数个 说明当前状态i不满足要求，直接将st[i]只为Fasle；b.如果当前数为0，那么cnt++即可。f
        for i in range(1 << n):
            cnt = 0
            st[i] = True
            for j in range(n):
                if i >> j & 1:
                    if cnt & 1:
                        st[i] = False
                    cnt = 0
                else:
                    cnt += 1
            # 最后还要判断一下 最后一段如果也是奇数，那也是不合法的:这里容易忘    
            if cnt & 1: st[i] = False

        # base case很重要
        f[0][0] = 1
        for i in range(1, m + 1):
            for j in range(1 << n):
                for k in range(1 << n):
                    if j & k == 0 and st[j | k]:
                        f[i][j] += f[i - 1][k]

        print(f[m][0])
        n, m = map(int, input().split())

# 写代码的时候，加一下优化，不容易受时间限制卡住：预处理对于每个状态k而言，有哪些状态可以更新到j
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 91. 最短Hamilton路径](https://www.acwing.com/problem/content/93/)**
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

const int N = 21, M = 1 << N;
int d[N][N];
int st[M];
int f[M][N];

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) cin >> d[i][j];
    memset(f, 0x3f, sizeof f);
    f[1][0] = 0;
    for (int i = 0; i < 1 << n; ++i) {  // 走过的状态
        for (int j = 0; j < n; ++j)     // 枚举该状态的最后一个点
            if ((i >> j & 1) == 1)      // 状态包含该点
                for (int k = 0; k < n; ++k)  // 又哪个点转移至j点
                    if (i ^ 1 << j >> k & 1)
                        // 如果从当前状态经过点集 state 中，去掉点 j 后，state
                        // 仍然包含点 k，那么才能从点 k 转移到点 j。
                        if (f[i ^ 1 << j][k] + d[k][j] < f[i][j])
                            f[i][j] = f[i ^ 1 << j][k] + d[k][j];
    }
    cout << f[(1 << n) - 1][n - 1] << endl;
}
```

##### **Python**

```python
#如何走的过程不关心，只关心哪些点被用过，怎么走是最短的：1.哪些点被用过；2. 目前最后停在哪个点上
#用二进制用来表示要走的所有情况的路径，这里用state比作，比如 0,1,4 ==>  state=10011
#状态表示：f[i][j]: 所有从0走点j，走过的所有点的路径集合是i；属性：min（i：就是哪些点被用过了）
#状态转移： 枚举从哪个点转移到点j上来===> 找一个中间点k，将已经走过点的集合i中去掉j(表示j不在经过的点的集合中)，然后再加上从k到j的权值。
#f[state][j]=f[state_k][k]+weight[k][j], state_k= state除掉j之外的集合，state_k要包含k===> f[i][j]=min(f[i][j],f[i-(1<<j)][k]+w[k][j])

N = 22
M = 1 << N
f = [[float('inf')] * N for _ in range(M)]  #要求的是最小距离，所以初始化的时候 要初始化为max
w = []   # w表示的是无权图 

if __name__=='__main__':
    n = int(input())
    for i in range(n):
        w.append(list(map(int,input().split())))
    f[1][0] = 0  #因为零是起点，所以f[1][0]=0,第一个点是不需要任何费用的
    for i in range(1 << n):   #i表示所有的情况,一个方案集合
        for j in range(n):    #j表示走到哪一个点
            #判断状态是否是合法的：状态i里的第j位是否为1
            if i >> j & 1: 
                for k in range(n):  #k表示走到j这个点之前，以k为终点的最短距离
                    if i >> k & 1:  #判断状态是否合法：第j位是否为1
                    #也可以写成：if (i - (1 << j) >> k & 1）:  
                        f[i][j] = min(f[i][j], f[i-(1 << j)][k] + w[k][j])  #更新最短距离
    print(f[(1 << n)-1][n-1])  #表示把所有点都遍历过了，并且停在n-1点上。
```

<!-- tabs:end -->
</details>

<br>

* * *