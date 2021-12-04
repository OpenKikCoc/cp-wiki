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
