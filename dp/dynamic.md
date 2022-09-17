前置芝士： [矩阵](math/linear-algebra/matrix.md) [树链剖分](graph/hld.md)。

动态 DP 问题是猫锟在 WC2018 讲的黑科技，一般用来解决树上的 DP 问题，同时支持点权（边权）修改操作。

在 NOIP2018D2T3 考察后风靡 OI 圈。

## 例题

> [!NOTE] **例题 [洛谷 P4719 【模板】动态 DP](https://www.luogu.com.cn/problem/P4719)**
> 
> 给定一棵 $n$ 个点的树，点带点权。有 $m$ 次操作，每次操作给定 $x,y$ 表示修改点 $x$ 的权值为 $y$。你需要在每次操作之后求出这棵树的最大权独立集的权值大小。

### 广义矩阵乘法

定义广义矩阵乘法 $A\times B=C$ 为：

$$
C_{i,j}=\max_{k=1}^{n}(A_{i,k}+B_{k,j})
$$

相当于将普通的矩阵乘法中的乘变为加，加变为 $\max$ 操作。

同时广义矩阵乘法满足结合律，所以可以使用矩阵快速幂。

### 不带修改操作

令 $f_{i,0}$ 表示不选择 $i$ 的最大答案，$f_{i,1}$ 表示选择 $i$ 的最大答案。

则有 DP 方程：

$$
\begin{cases}f_{i,0}=\sum_{son}\max(f_{son,0},f_{son,1})\\f_{i,1}=w_i+\sum_{son}f_{son,0}\end{cases}
$$

答案就是 $\max(f_{root,0},f_{root,1})$.

### 带修改操作

首先将这棵树进行树链剖分，假设有这样一条重链：

![](./images/dynamic.png)

设 $g_{i,0}$ 表示不选择 $i$ 且只允许选择 $i$ 的轻儿子所在子树的最大答案，$g_{i,1}$ 表示选择 $i$ 的最大答案，$son_i$ 表示 $i$ 的重儿子。

假设我们已知 $g_{i,0/1}$ 那么有 DP 方程：

$$
\begin{cases}f_{i,0}=g_{i,0}+\max(f_{son_i,0},f_{son_i,1})\\f_{i,1}=g_{i,1}+f_{son_i,0}\end{cases}
$$

答案是 $\max(f_{root,0},f_{root,1})$.

可以构造出矩阵：

$$
\begin{bmatrix}
g_{i,0} & g_{i,0}\\
g_{i,1} & -\infty
\end{bmatrix}\times 
\begin{bmatrix}
f_{son_i,0}\\f_{son_i,1}
\end{bmatrix}=
\begin{bmatrix}
f_{i,0}\\f_{i,1}
\end{bmatrix}
$$

注意，我们这里使用的是广义乘法规则。

可以发现，修改操作时只需要修改 $g_{i,1}$ 和每条往上的重链即可。

### 具体思路

1. DFS 预处理求出 $f_{i,0/1}$ 和 $g_{i,0/1}$.

2. 对这棵树进行树剖（注意，因为我们对一个点进行询问需要计算从该点到该点所在的重链末尾的区间矩阵乘，所以对于每一个点记录 $End_i$ 表示 $i$ 所在的重链末尾节点编号），每一条重链建立线段树，线段树维护 $g$ 矩阵和 $g$ 矩阵区间乘积。

3. 修改时首先修改 $g_{i,1}$ 和线段树中 $i$ 节点的矩阵，计算 $top_i$ 矩阵的变化量，修改到 $fa_{top_i}$ 矩阵。

4. 查询时就是 1 到其所在的重链末尾的区间乘，最后取一个 $\max$ 即可。


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

[SPOJ GSS3 - Can you answer these queries III](https://www.spoj.com/problems/GSS3/)

[「NOIP2018」保卫王国](https://loj.ac/p/2955)

[「SDOI2017」切树游戏](https://loj.ac/p/2269)


> [!NOTE] **[Luogu P4719 【模板】"动态 DP"&动态树分治](https://www.luogu.com.cn/problem/P4719)**
> 
> 题意: 
> 
> 给定一棵 $n$ 个点的树，点带点权。
> 
> 有 $m$ 次操作，每次操作给定 $x,y$，表示修改点 $x$ 的权值为 $y$。
> 
> 你需要在每次操作之后求出这棵树的最大权独立集的权值大小。

> [!TIP] **思路**
> 
> TODO

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

> [!NOTE] **[Luogu P4751 【模板】"动态DP"&动态树分治（加强版）](https://www.luogu.com.cn/problem/P4751)**
> 
> 题意: 
> 
> 同上题，但强制在线

> [!TIP] **思路**
> 
> TODO

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

> [!NOTE] **[Luogu P3287 [SCOI2014]方伯伯的玉米田](https://www.luogu.com.cn/problem/P3287)**
> 
> 题意: 
> 
> 一排玉米一共有 $N$ 株，它们的高度参差不齐。方伯伯认为单调不下降序列很美，所以他决定先把一些玉米拔高，再把破坏美感的玉米拔除掉，使得剩下的玉米的高度构成一个单调不下降序列。
> 
> 方伯伯可以选择一个区间，把这个区间的玉米全部拔高 $1$ 单位高度，他可以进行最多 $K$ 次这样的操作。拔玉米则可以随意选择一个集合的玉米拔掉。
> 
> 问能最多剩多少株玉米，来构成一排美丽的玉米。

> [!TIP] **思路**
> 
> TODO

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

> [!NOTE] **[Luogu P2605 [ZJOI2010]基站选址](https://www.luogu.com.cn/problem/P2605)**
> 
> 题意: 
> 
> 有 $N$ 个村庄坐落在一条直线上，第 $i(i>1)$ 个村庄距离第 $1$ 个村庄的距离为 $D_i$。需要在这些村庄中建立不超过 $K$ 个通讯基站，在第 $i$ 个村庄建立基站的费用为 $C_i$。如果在距离第 $i$ 个村庄不超过 $S_i$ 的范围内建立了一个通讯基站，那么就村庄被基站覆盖了。如果第 $i$ 个村庄没有被覆盖，则需要向他们补偿，费用为 $W_i$。现在的问题是，选择基站的位置，使得总费用最小。

> [!TIP] **思路**
> 
> TODO

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

> [!NOTE] **[Luogu P5024 [NOIP2018 提高组] 保卫王国](https://www.luogu.com.cn/problem/P5024)**
> 
> 题意: 
> 
> 有棵树，可以花 $pi$ 代价把 $i$ 点染色，要求任 $2$ 个相邻点至少有 $1$ 个被染色。
> 
> 给出 $m$ 组询问，每次强制两个点的状态(染/不染)，求出每次的最小花费。

> [!TIP] **思路**
> 
> TODO

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