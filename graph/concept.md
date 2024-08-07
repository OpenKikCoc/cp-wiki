本页面概述了图论中的一些概念，这些概念并不全是在 OI 中常见的，对于 OIer 来说，只需掌握本页面中的基础部分即可，如果在学习中碰到了不懂的概念，可以再来查阅。

> [!WARNING]
> 
> 图论相关定义在不同教材中往往会有所不同，遇到的时候需根据上下文加以判断。

## 图

**图 (Graph)** 是一个二元组 $G=(V(G), E(G))$。其中 $V(G)$ 是非空集，称为 **点集 (Vertex set)** ，对于 $V$ 中的每个元素，我们称其为 **顶点 (Vertex)** 或 **节点 (Node)** ，简称 **点** ；$E(G)$ 为 $V(G)$ 各结点之间边的集合，称为 **边集 (Edge set)** 。

常用 $G=(V,E)$ 表示图。

当 $V,E$ 都是有限集合时，称 $G$ 为 **有限图** 。

当 $V$ 或 $E$ 是无限集合时，称 $G$ 为 **无限图** 。

图有多种，包括 **无向图 (Undirected graph)** ，**有向图 (Directed graph)** ，**混合图 (Mixed graph)** 等

若 $G$ 为无向图，则 $E$ 中的每个元素为一个无序二元组 $(u, v)$，称作 **无向边 (Undirected edge)** ，简称 **边 (Edge)** ，其中 $u, v \in V$。设 $e = (u, v)$，则 $u$ 和 $v$ 称为 $e$ 的 **端点 (Endpoint)** 。

若 $G$ 为有向图，则 $E$ 中的每一个元素为一个有序二元组 $(u, v)$，有时也写作 $u \to v$，称作 **有向边 (Directed edge)** 或 **弧 (Arc)** ，在不引起混淆的情况下也可以称作 **边 (Edge)** 。设 $e = u \to v$，则此时 $u$ 称为 $e$ 的 **起点 (Tail)** ，$v$ 称为 $e$ 的 **终点 (Head)** ，起点和终点也称为 $e$ 的 **端点 (Endpoint)** 。并称 $u$ 是 $v$ 的直接前驱，$v$ 是 $u$ 的直接后继。

> [!NOTE] **为什么起点是 Tail，终点是 Head？**
> 
> 边通常用箭头表示，而箭头是从“尾”指向“头”的。

若 $G$ 为混合图，则 $E$ 中既有向边，又有无向边。

若 $G$ 的每条边 $e_k=(u_k,v_k)$ 都被赋予一个数作为该边的 **权** ，则称 $G$ 为 **赋权图** 。如果这些权都是正实数，就称 $G$ 为 **正权图** 。

图 $G$ 的点数 $\left| V(G) \right|$ 也被称作图 $G$ 的 **阶 (Order)** 。

形象地说，图是由若干点以及连接点与点的边构成的。

## 相邻

在无向图 $G = (V, E)$ 中，若点 $v$ 是边 $e$ 的一个端点，则称 $v$ 和 $e$ 是 **关联的 (Incident)** 或 **相邻的 (Adjacent)** 。对于两顶点 $u$ 和 $v$，若存在边 $(u, v)$，则称 $u$ 和 $v$ 是 **相邻的 (Adjacent)** 。

一个顶点 $v \in V$ 的 **邻域 (Neighborhood)** 是所有与之相邻的顶点所构成的集合，记作 $N(v)$。

一个点集 $S$ 的邻域是所有与 $S$ 中至少一个点相邻的点所构成的集合，记作 $N(S)$，即：

$$
N(S) = \bigcup_{v \in S} N(v)
$$

## 度数

与一个顶点 $v$ 关联的边的条数称作该顶点的 **度 (Degree)** ，记作 $d(v)$。特别地，对于边 $(v, v)$，则每条这样的边要对 $d(v)$ 产生 $2$ 的贡献。

对于无向简单图，有 $d(v) = \left| N(v) \right|$。

握手定理（又称图论基本定理）：对于任何无向图 $G = (V, E)$，有 $\sum_{v \in V} d(v) = 2 \left| E \right|$。

推论：在任意图中，度数为奇数的点必然有偶数个。

若 $d(v) = 0$，则称 $v$ 为 **孤立点 (Isolated vertex)** 。

若 $d(v) = 1$，则称 $v$ 为 **叶节点 (Leaf vertex)**/**悬挂点 (Pendant vertex)** 。

若 $2 \mid d(v)$，则称 $v$ 为 **偶点 (Even vertex)** 。

若 $2 \nmid d(v)$，则称 $v$ 为 **奇点 (Odd vertex)** 。图中奇点的个数是偶数。

若 $d(v) = \left| V \right| - 1$，则称 $v$ 为 **支配点 (Universal vertex)** 。

对一张图，所有节点的度数的最小值称为 $G$ 的 **最小度 (Minimum degree)** ，记作 $\delta (G)$；最大值称为 **最大度 (Maximum degree)** ，记作 $\Delta (G)$。即：$\delta (G) = \min_{v \in G} d(v)$，$\Delta (G) = \max_{v \in G} d(v)$。

在有向图 $G = (V, E)$ 中，以一个顶点 $v$ 为起点的边的条数称为该顶点的 **出度 (Out-degree)** ，记作 $d^+(v)$。以一个顶点 $v$ 为终点的边的条数称为该节点的 **入度 (In-degree)** ，记作 $d^-(v)$。显然 $d^+(v)+d^-(v)=d(v)$。

对于任何有向图 $G = (V, E)$，有：

$$
\sum_{v \in V} d^+(v) = \sum_{v \in V} d^-(v) = \left| E \right|
$$

若对一张无向图 $G = (V, E)$，每个顶点的度数都是一个固定的常数 $k$，则称 $G$ 为 **$k$- 正则图 ($k$-Regular Graph)** 。

如果给定一个序列 a，可以找到一个图 G，以其为度数列，则称 a 是 **可图化** 的。

如果给定一个序列 a，可以找到一个简单图 G，以其为度数列，则称 a 是 **可简单图化** 的。

## 简单图

**自环 (Loop)** ：对 $E$ 中的边 $e = (u, v)$，若 $u = v$，则 $e$ 被称作一个自环。

**重边 (Multiple edge)** ：若 $E$ 中存在两个完全相同的元素（边）$e_1, e_2$，则它们被称作（一组）重边。

**简单图 (Simple graph)** ：若一个图中没有自环和重边，它被称为简单图。具有至少两个顶点的简单无向图中一定存在度相同的结点。（[鸽巢原理](math/combinatorics/drawer-principle.md)）

如果一张图中有自环或重边，则称它为 **多重图 (Multigraph)** 。

> [!WARNING]
> 
> 在无向图中 $(u, v)$ 和 $(v, u)$ 算一组重边，而在有向图中，$u \to v$ 和 $v \to u$ 不为重边。

> [!WARNING]
> 
> 在题目中，如果没有特殊说明，是可以存在自环和重边的，在做题时需特殊考虑。

## 路径

**途径 (Walk)** ：途径是一个将若干个点连接起来的边的集合。形式化地说，途径 $w$ 是一个边的集合 $\{e_1, e_2, \ldots, e_k\}$，这个边集需要满足条件：存在一个由点构成的序列 $v_0, v_1, \ldots, v_k$ 满足 $e_i$ 的两个端点分别为 $v_{i-1}$ 和 $v_i$。这样的路径可以简写为 $v_0 \to v_1 \to v_2 \to \cdots \to v_k$。通常来说，边的数量 $k$ 被称作这条途径的 **长度**（如果边是带权的，长度通常指路径上的边权之和，题目中也可能另有定义）。

**迹 (Trail)** ：对于一条途径 $w$，若 $e_1, e_2, \ldots, e_k$ 两两互不相同，则称 $w$ 是一条迹。

**路径 (Path)** ：（又称 **简单路径 (Simple path)**)：对于一条迹 $w$，若其连接的点的序列中点两两不同，则称 $w$ 是一条路径。

**回路 (Circuit)** ：对于一个迹 $w$，若 $v_0 = v_k$，则称 $w$ 是一个回路。

**环/圈 (Cycle)** ：（又称 **简单回路/简单环 (Simple circuit)**)：对于一个回路 $w$，若 $v_0 = v_k$ 是点序列中唯一重复出现的点对，则称 $w$ 是一个环。

> [!WARNING]
> 
> 关于路径的定义在不同地方可能有所不同，如，“路径”可能指本文中的“途径”，“环”可能指本文中的“回路”。如果在题目中看到类似的词汇，且没有“简单路径”/“非简单路径”（即本文中的“途径”）等特殊说明，最好询问一下具体指什么。

## 子图

对一张图 $G = (V, E)$，若存在另一张图 $H = (V', E')$ 满足 $V' \subseteq V$ 且 $E' \subseteq E$，则称 $H$ 是 $G$ 的 **子图 (Subgraph)** ，记作 $H \subseteq G$。

若对 $H \subseteq G$，满足 $\forall u, v \in V'$，只要 $(u, v) \in E$，均有 $(u, v) \in E'$，则称 $H$ 是 $G$ 的 **导出子图/诱导子图 (Induced subgraph)** 。

容易发现，一个图的导出子图仅由子图的点集决定，因此点集为 $V'$($V' \subseteq V$) 的导出子图称为 $V'$ 导出的子图，记作 $G \left[ V' \right]$。

若 $H \subseteq G$ 满足 $V' = V$，则称 $H$ 为 $G$ 的 **生成子图/支撑子图 (Spanning subgraph)** 。

显然，$G$ 是自身的子图，支撑子图，导出子图；空图是 $G$ 的支撑子图。原图 $G$ 和空图都是 $G$ 的平凡子图。

如果一张无向图 $G$ 的某个生成子图 $F$ 为 $k$- 正则图，则称 $F$ 为 $G$ 的一个 **$k$- 因子 ($k$-Factor)** 。

如果有向图 $G = (V, E)$ 的导出子图 $H = G \left[ V^\ast \right]$ 满足 $\forall v \in V^\ast, (v, u) \in E$，有 $u \in V^\ast$，则称 $H$ 为 $G$ 的一个 **闭合子图 (Closed subgraph)** 。

## 连通

### 无向图

对于一张无向图 $G = (V, E)$，对于 $u, v \in V$，若存在一条途径使得 $v_0 = u, v_k = v$，则称 $u$ 和 $v$ 是 **连通的 (Connected)** 。由定义，任意一个顶点和自身连通，任意一条边的两个端点连通。

若无向图 $G = (V, E)$，满足其中任意两个顶点均连通，则称 $G$ 是 **连通图 (Connected graph)** ，$G$ 的这一性质称作 **连通性 (Connectivity)** 。

若 $H$ 是 $G$ 的一个连通子图，且不存在 $F$ 满足 $H\subsetneq F \subseteq G$ 且 $F$ 为连通图，则 $H$ 是 $G$ 的一个 **连通块/连通分量 (Connected component)** （极大连通子图）。

### 有向图

对于一张有向图 $G = (V, E)$，对于 $u, v \in V$，若存在一条途径使得 $v_0 = u, v_k = v$，则称 $u$  **可达**  $v$。由定义，任意一个顶点可达自身，任意一条边的起点可达终点。（无向图中的连通也可以视作双向可达。）

若一张有向图的节点两两互相可达，则称这张图是 **强连通的 (Strongly connected)** 。

若一张有向图的边替换为无向边后可以得到一张连通图，则称原来这张有向图是 **弱连通的 (Weakly connected)** 。

与连通分量类似，也有 **弱连通分量 (Weakly connected component)** （极大弱连通子图）和 **强连通分量 (Strongly Connected component)** （极大强连通子图）。

相关算法请参见 [强连通分量](./scc.md)。

### 割

相关算法请参见 [割点和桥](./cut.md) 以及 [双连通分量](./bcc.md)。

在本部分中，有向图的“连通”一般指“强连通”。

对于连通图 $G = (V, E)$，若 $V'\subseteq V$ 且 $G\left[V\setminus V'\right]$（即从 $G$ 中删去 $V'$ 中的点）不是连通图，则 $V'$ 是图 $G$ 的一个 **点割集 (Vertex cut/Separating set)** 。大小为一的点割集又被称作 **割点 (Cut vertex)** 。

对于连通图 $G = (V, E)$ 和整数 $k$，若 $|V|\ge k+1$ 且 $G$ 不存在大小为 $k-1$ 的点割集，则称图 $G$ 是 **$k$- 点连通的 ($k$-vertex-connected)** ，而使得上式成立的最大的 $k$ 被称作图 $G$ 的 **点连通度 (Vertex connectivity)** ，记作 $\kappa(G)$。（对于非完全图，点连通度即为最小点割集的大小，而完全图 $K_n$ 的点连通度为 $n-1$。）

对于图 $G = (V, E)$ 以及 $u, v\in V$ 满足 $u\ne v$，$u$ 和 $v$ 不相邻，$u$ 可达 $v$，若 $V'\subseteq V$，$u, v\notin V'$，且在 $G\left[V\setminus V'\right]$ 中 $u$ 和 $v$ 不连通，则 $V'$ 被称作 $u$ 到 $v$ 的点割集。$u$ 到 $v$ 的最小点割集的大小被称作 $u$ 到 $v$ 的 **局部点连通度 (Local connectivity)** ，记作 $\kappa(u, v)$。

还可以在边上作类似的定义：

对于连通图 $G = (V, E)$，若 $E'\subseteq E$ 且 $G' = (V, E\setminus E')$（即从 $G$ 中删去 $E'$ 中的边）不是连通图，则 $E'$ 是图 $G$ 的一个 **边割集 (Edge cut)** 。大小为一的边割集又被称作 **桥 (Bridge)** 。

对于连通图 $G = (V, E)$ 和整数 $k$，若 $G$ 不存在大小为 $k-1$ 的边割集，则称图 $G$ 是 **$k$- 边连通的 ($k$-edge-connected)** ，而使得上式成立的最大的 $k$ 被称作图 $G$ 的 **边连通度 (Edge connectivity)** ，记作 $\lambda(G)$。（对于任何图，边连通度即为最小边割集的大小。）

对于图 $G = (V, E)$ 以及 $u, v\in V$ 满足 $u\ne v$，$u$ 可达 $v$，若 $E'\subseteq E$，且在 $G'=(V, E\setminus E')$ 中 $u$ 和 $v$ 不连通，则 $E'$ 被称作 $u$ 到 $v$ 的边割集。$u$ 到 $v$ 的最小边割集的大小被称作 $u$ 到 $v$ 的 **局部边连通度 (Local edge-connectivity)** ，记作 $\lambda(u, v)$。

**点双连通 (Biconnected)** 几乎与 $2$- 点连通完全一致，除了一条边连接两个点构成的图，它是点双连通的，但不是 $2$- 点连通的。换句话说，没有割点的连通图是点双连通的。

**边双连通 ($2$-edge-connected)** 与 $2$- 边双连通完全一致。换句话说，没有桥的连通图是边双连通的。

与连通分量类似，也有 **点双连通分量 (Biconnected component)** （极大点双连通子图）和 **边双连通分量 ($2$-edge-connected component)** （极大边双连通子图）。

**Whitney 定理**：对任意的图 $G$，有 $\kappa(G)\le \lambda(G)\le \delta(G)$。（不等式中的三项分别为点连通度、边连通度、最小度。）

## 稀疏图/稠密图

若一张图的边数远小于其点数的平方，那么它是一张 **稀疏图 (Sparse graph)** 。

若一张图的边数接近其点数的平方，那么它是一张 **稠密图 (Dense graph)** 。

这两个概念并没有严格的定义，一般用于讨论 [时间复杂度](basic/complexity.md) 为 $O(|V|^2)$ 的算法与 $O(|E|)$ 的算法的效率差异（在稠密图上这两种算法效率相当，而在稀疏图上 $O(|E|)$ 的算法效率明显更高）。

## 补图

对于无向简单图 $G = (V, E)$，它的 **补图 (Complement graph)** 指的是这样的一张图：记作 $\bar G$，满足 $V \left( \bar G \right) = V \left( G \right)$，且对任意节点对 $(u, v)$，$(u, v) \in E \left( \bar G \right)$ 当且仅当 $(u, v) \notin E \left( G \right)$。

## 反图

对于有向图 $G = (V, E)$，它的 **反图 (Transpose Graph)** 指的是点集不变，每条边反向得到的图，即：若 $G$ 的反图为 $G'=(V, E')$，则 $E'=\{(v, u)|(u, v)\in E\}$。

## 特殊的图

若无向简单图 $G$ 满足任意不同两点间均有边，则称 $G$ 为 **完全图 (Complete graph)** ，$n$ 阶完全图记作 $K_n$。若有向图 $G$ 满足任意不同两点间都有两条方向不同的边，则称 $G$ 为 **有向完全图 (Complete digraph)** 。

边集为空的图称为 **零图 (Null graph)** ，$n$ 阶零图记作 $N_n$。易知，$N_n$ 为 $K_n$ 互为补图。

若有向简单图 $G$ 满足任意不同两点间都有恰好一条边（单向），则称 $G$ 为 **竞赛图 (Tournament graph)** 。

若无向简单图 $G = \left( V, E \right)$ 的所有边恰好构成一个圈，则称 $G$ 为 **环图/圈图 (Cycle graph)** ，$n$($n \geq 3$) 阶圈图记作 $C_n$。易知，一张图为圈图的充分必要条件是，它是 $2$- 正则连通图。

若无向简单图 $G = \left( V, E \right)$ 满足，存在一个点 $v$ 为支配点，其余点之间没有边相连，则称 $G$ 为 **星图/菊花图 (Star graph)** ，$n + 1$($n \geq 1$) 阶星图记作 $S_n$。

若无向简单图 $G = \left( V, E \right)$ 满足，存在一个点 $v$ 为支配点，其它点之间构成一个圈，则称 $G$ 为 **轮图 (Wheel Graph)** ，$n + 1$($n \geq 3$) 阶轮图记作 $W_n$。

若无向简单图 $G = \left( V, E \right)$ 的所有边恰好构成一条简单路径，则称 $G$ 为 **链 (Chain/Path Graph)** ，$n$ 阶的链记作 $P_n$。易知，一条链由一个圈图删去一条边而得。

如果一张无向连通图不含环，则称它是一棵 **树 (Tree)** 。相关内容详见 [树基础](./tree-basic.md)。

如果一张无向连通图包含恰好一个环，则称它是一棵 **基环树 (Pseudotree)** 。

如果一张有向弱连通图每个点的入度都为 $1$，则称它是一棵 **基环外向树** 。

如果一张有向弱连通图每个点的出度都为 $1$，则称它是一棵 **基环内向树** 。

多棵树可以组成一个 **森林 (Forest)** ，多棵基环树可以组成 **基环森林 (Pseudoforest)** ，多棵基环外向树可以组成 **基环外向树森林** ，多棵基环内向树可以组成 **基环内向森林 (Functional graph)** 。

如果一张无向连通图的每条边最多在一个环内，则称它是一棵 **仙人掌 (Cactus)** 。多棵仙人掌可以组成 **沙漠** 。

如果一张图的点集可以被分为两部分，每一部分的内部都没有连边，那么这张图是一张 **二分图 (Bipartite graph)** 。如果二分图中任何两个不在同一部分的点之间都有连边，那么这张图是一张 **完全二分图 (Complete bipartite graph/Biclique)** ，一张两部分分别有 $n$ 个点和 $m$ 个点的完全二分图记作 $K_{n, m}$。相关内容详见 [二分图](graph/bi-graph.md)。

如果一张图可以画在一个平面上，且没有两条边在非端点处相交，那么这张图是一张 **平面图 (Planar graph)** 。一张图的任何子图都不是 $K_5$ 或 $K_{3, 3}$ 是其为一张平面图的充要条件。对于简单连通平面图 $G=(V, E)$ 且 $V\ge 3$，$|E|\le 3|V|-6$。

## 同构

两个图 $G$ 和 $H$，如果存在一个双射 $f : V(G) \to V(H)$，且满足 $(u,v)\in E(G)$，当且仅当 $(f(u),f(v))\in E(H)$，则我们称 $f$ 为 $G$ 到 $H$ 的一个 **同构 (Isomorphism)** ，且图 $G$ 与图 $H$ 是 **同构的 (Isomorphic)** ，记作 $G \cong H$。

从定义可知，若 $G \cong H$，必须满足：

- $|V(G)|=|V(H)|,|E(G)|=|E(H)|$
- $G$ 和 $H$ 结点度的非增序列相同
- $G$ 和 $H$ 存在同构的导出子图

## 无向简单图的二元运算

对于无向简单图，我们可以定义如下二元运算：

**交 (Intersection)** ：图 $G = \left( V_1, E_1 \right), H = \left( V_2, E_2 \right)$ 的交定义成图 $G \cap H = \left( V_1 \cap V_2, E_1 \cap E_2 \right)$。

容易证明两个无向简单图的交还是无向简单图。

**并 (Union)** ：图 $G = \left( V_1, E_1 \right), H = \left( V_2, E_2 \right)$ 的并定义成图 $G \cup H = \left( V_1 \cup V_2, E_1 \cup E_2 \right)$。

**和 (Sum)/直和 (Direct sum)** ：对于 $G = \left( V_1, E_1 \right), H = \left( V_2, E_2 \right)$，任意构造 $H' \cong H$ 使得 $V \left( H' \right) \cap V_1 = \varnothing$($H'$ 可以等于 $H$)。此时与 $G \cup H'$ 同构的任何图称为 $G$ 和 $H$ 的和/直和/不交并，记作 $G + H$ 或 $G \oplus H$。

若 $G$ 与 $H$ 的点集本身不相交，则 $G \cup H = G + H$。

比如，森林可以定义成若干棵树的和。

> [!NOTE] **并与和的区别**
> 
> 可以理解为，“并”会让两张图中“名字相同”的点、边合并，而“和”则不会。

## 特殊的点集/边集

### 支配集

对于无向图 $G=(V, E)$，若 $V'\subseteq V$ 且 $\forall v\in(V\setminus V')$ 存在边 $(u, v)\in E$ 满足 $u\in V'$，则 $V'$ 是图 $G$ 的一个 **支配集 (Dominating set)** 。

无向图 $G$ 最小的支配集的大小记作 $\gamma(G)$。求一张图的最小支配集是 [NP 困难](misc/cc-basic.md#np-hard) 的。

对于有向图 $G=(V, E)$，若 $V'\subseteq V$ 且 $\forall v\in(V\setminus V')$ 存在边 $(u, v)\in E$ 满足 $u\in V'$，则 $V'$ 是图 $G$ 的一个 **出 - 支配集 (Out-dominating set)** 。类似地，可以定义有向图的 **入 - 支配集 (In-dominating set)** 。

有向图 $G$ 最小的出 - 支配集大小记作 $\gamma^+(G)$，最小的入 - 支配集大小记作 $\gamma^-(G)$。

### 边支配集

对于图 $G=(V, E)$，若 $E'\subseteq E$ 且 $\forall e\in(E\setminus E')$ 存在 $E'$ 中的边与其有公共点，则称 $E'$ 是图 $G$ 的一个 **边支配集 (Edge dominating set)** 。

求一张图的最小边支配集是 [NP 困难](misc/cc-basic.md#np-hard) 的。

### 独立集

对于图 $G=(V, E)$，若 $V'\subseteq V$ 且 $V'$ 中任意两点都不相邻，则 $V'$ 是图 $G$ 的一个 **独立集 (Independent set)** 。

图 $G$ 最大的独立集的大小记作 $\alpha(G)$。求一张图的最大独立集是 [NP 困难](misc/cc-basic.md#np-hard) 的。

### 匹配

对于图 $G=(V, E)$，若 $E'\in E$ 且 $E'$ 中任意两条不同的边都没有公共的端点，且 $E'$ 中任意一条边都不是自环，则 $E'$ 是图 $G$ 的一个 **匹配 (Matching)** ，也可以叫作 **边独立集 (Independent edge set)** 。如果一个点是匹配中某条边的一个端点，则称这个点是 **被匹配的 (matched)/饱和的 (saturated)** ，否则称这个点是 **不被匹配的 (unmatched)** 。

边数最多的匹配被称作一张图的 **最大匹配 (Maximum-cardinality matching)** 。图 $G$ 的最大匹配的大小记作 $\nu(G)$。

如果边带权，那么权重之和最大的匹配被称作一张图的 **最大权匹配 (Maximum-weight matching)** 。

如果一个匹配在加入任何一条边后都不再是一个匹配，那么这个匹配是一个 **极大匹配 (Maximal matching)** 。最大的极大匹配就是最大匹配，任何最大匹配都是极大匹配。极大匹配一定是边支配集，但边支配集不一定是匹配。最小极大匹配和最小边支配集大小相等，但最小边支配集不一定是匹配。求最小极大匹配是 NP 困难的。

如果在一个匹配中所有点都是被匹配的，那么这个匹配是一个 **完美匹配 (Perfect matching)** 。如果在一个匹配中只有一个点不被匹配，那么这个匹配是一个 **准完美匹配 (Near-perfect matching)** 。

求一张普通图或二分图的匹配或完美匹配个数都是 [#P 完全](misc/cc-basic.md#p_1) 的。

对于一个匹配 $M$，若一条路径以非匹配点为起点，每相邻两条边的其中一条在匹配中而另一条不在匹配中，则这条路径被称作一条 **交替路径 (Alternating path)** ；一条在非匹配点终止的交替路径，被称作一条 **增广路径 (Augmenting path)** 。

**托特定理** ：$n$ 阶无向图 $G$ 有完美匹配当且仅当对于任意的 $V' \subset V(G)$，$p_{\text{奇}}(G-V')\leq |V'|$，其中 $p_{\text{奇}}$ 表示奇数阶连通分支数。

**托特定理（推论）** ：任何无桥 3 - 正则图都有完美匹配。

### 点覆盖

对于图 $G=(V, E)$，若 $V'\subseteq V$ 且 $\forall e\in E$ 满足 $e$ 的至少一个端点在 $V'$ 中，则称 $V'$ 是图 $G$ 的一个 **点覆盖 (Vertex cover)** 。

点覆盖集必为支配集，但极小点覆盖集不一定是极小支配集。

一个点集是点覆盖的充要条件是其补集是独立集，因此最小点覆盖的补集是最大独立集。求一张图的最小点覆盖是 [NP 困难](misc/cc-basic.md#np-hard) 的。

一张图的任何一个匹配的大小都不超过其任何一个点覆盖的大小。完全二分图 $K_{n, m}$ 的最大匹配和最小点覆盖大小都为 $\min(n, m)$。

### 边覆盖

对于图 $G=(V, E)$，若 $E'\subseteq E$ 且 $\forall v\in V$ 满足 $v$ 与 $E'$ 中的至少一条边相邻，则称 $E'$ 是图 $G$ 的一个 **边覆盖 (Edge cover)** 。

最小边覆盖的大小记作 $\rho(G)$，可以由最大匹配贪心扩展求得：对于所有非匹配点，将其一条邻边加入最大匹配中，即得到了一个最小边覆盖。

最大匹配也可以由最小边覆盖求得：对于最小边覆盖中每对有公共点的边删去其中一条。

一张图的最小边覆盖的大小加上最大匹配的大小等于图的点数，即 $\rho(G)+\nu(G)=|V(G)|$。

一张图的最大匹配的大小不超过最小边覆盖的大小，即 $\nu(G)\le\rho(G)$。特别地，完美匹配一定是一个最小边覆盖，这也是上式取到等号的唯一情况。

一张图的任何一个独立集的大小都不超过其任何一个边覆盖的大小。完全二分图 $K_{n, m}$ 的最大独立集和最小边覆盖大小都为 $\max(n, m)$。

### 团

对于图 $G=(V, E)$，若 $V'\subseteq V$ 且 $V'$ 中任意两个不同的顶点都相邻，则 $V'$ 是图 $G$ 的一个 **团 (Clique)** 。团的导出子图是完全图。

如果一个团在加入任何一个顶点后都不再是一个团，则这个团是一个 **极大团 (Maximal clique)** 。

一张图的最大团的大小记作 $\omega(G)$，最大团的大小等于其补图最大独立集的大小，即 $\omega(G)=\alpha(\bar{G})$。求一张图的最大团是 [NP 困难](misc/cc-basic.md#np-hard) 的。

## 参考资料

[OI 中转站 - 图论概念梳理](https://yhx-12243.github.io/OI-transit/memos/14.html)

[Wikipedia](https://en.wikipedia.org/wiki/Glossary_of_graph_theory_terms)（以及相关概念的对应词条）

离散数学（修订版），田文成 周禄新 编著，天津文学出版社，P184-187

戴一奇，胡冠章，陈卫。图论与代数结构[M]. 北京：清华大学出版社，1995.


## 习题

> [!NOTE] **[AcWing 1352. 虫洞](https://www.acwing.com/problem/content/1354/)**
> 
> 题意: 基环树（森林）找环

> [!TIP] **思路**
> 
> **经典算法 背过**
> 
> 将每个点拆为两个点
> 
> 入点 出点
> 
> 本题每个点的出边只有一条 可以不用tarjan 直接dfs判环
> 
> or 基环树(森林)找环 需要背过 经典算法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 12;

int n;
int to1[N], to2[N];
bool st[N], used[N][2], cur[N][2];

struct Point {
    int x, y;
    bool operator< (const Point & t) const {
        if (y != t.y) return y < t.y;
        return x < t.x;
    }
}q[N];
int ans;

// 判断当前的这部分图中是否存在环
bool dfs_c(int a, int b) {
    if (cur[a][b]) return true;
    if (used[a][b]) return false;
    // cur[a][b] = true
    // 当前选择的这个走法的序列中走过，整个dfs函数结束到时候要标记为false因为不能影响到其他的dfs情况
    // used[a][b] = true;
    // 已经走过这个点了（历史情况中一定走过这个点，标记为true整个dfs过程中都不能变了）
    cur[a][b] = used[a][b] = true;
    bool res = false;
    
    if (!b) {
        // a 的入点
        // 是入点，要传送，也就是要走ver2, 0 -> 1
        if (dfs_c(to2[a], 1)) res = true;
    } else {
        // a 的出点
        // 是出点，要向右走，也就是要走ver1, 1 -> 0
        // a点可以向右走并且向右走的这条路存在环
        if (to1[a] != -1 && dfs_c(to1[a], 0)) res = true;
    }
    cur[a][b] = false;  //已经dfs完了，这里是回溯到的，要恢复现场
    return res;
}

// 检查当前方案是否合法
bool check() {
    memset(used, 0, sizeof used);
    memset(cur, 0, sizeof cur);
    
    for (int i = 0; i < n; ++ i )
        for (int j = 0; j < 2; ++ j )
            if (!used[i][j])
                // 没有走过的话就走一下，因为我们要把整张图的所有的点都遍历一遍
                // 所以只要used = false ，我们从这儿就走（因为可能存在孤立点）
                if (dfs_c(i, j))
                    return true;
    return false;
}

// 分配方案（将n个点分为n/2组，每组2个点）
void dfs(int u) {
    if (u == n / 2) {
        if (check()) ++ ans;
        return ;
    }
    for (int i = 0; i < n; ++ i )
        // 当前方案中没有选到这个点，也就意味着可以选
        if (!st[i]) {
            for (int j = i + 1; j < n; ++ j )
                if (!st[j]) {
                    st[i] = st[j] = true;
                    to2[i] = j, to2[j] = i;
                    dfs(u + 1);
                    to2[i] = to2[j] = -1;
                    st[i] = st[j] = false;
                }
            break;// 选完了可以退了，不然会多选
        }
}

int main() {
    cin >> n;
    for (int i = 0; i < n; ++ i ) cin >> q[i].x >> q[i].y;
    sort(q, q + n);
    
    memset(to1, -1, sizeof to1);
    memset(to2, -1, sizeof to2);
    for (int i = 1; i < n; ++ i )
        if (q[i].y == q[i - 1].y)   // 可以从i-1走到i
            to1[i - 1] = i;
    dfs(0);
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

> [!NOTE] **[LeetCode 2127. 参加会议的最多员工数](https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 内向基环树找环 + 分情况讨论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10;
    
    int deg[N], depth[N];
    int q[N], hh, tt;
    bool st[N];
    
    // 所有数据可以看作多个【内向基环树（树中只有一个有向环）】
    // 每个内向基环树中 找到该最大环
    int maximumInvitations(vector<int>& favorite) {
        int n = favorite.size();
        
        // 统计入度
        memset(deg, 0, sizeof deg);
        for (int i = 0; i < n; ++ i )
            deg[favorite[i]] ++ ;
        
        // topo 求最长链
        for (int i = 0; i < n; ++ i )
            depth[i] = 1;
        hh = 0, tt = -1;
        for (int i = 0; i < n; ++ i )
            if (deg[i] == 0)
                q[ ++ tt] = i;
        while (hh <= tt) {
            int u = q[hh ++ ];
            int v = favorite[u];
            depth[v] = max(depth[v], depth[u] + 1);
            if ( -- deg[v] == 0)
                q[ ++ tt] = v;
        }
        
        // 求解
        // res1 : 一个【最长的环】
        // res2 : 多个【长度为2且各自带有一条链的环】
        int res1 = 0, res2 = 0;
        memset(st, 0, sizeof st);
        for (int i = 0; i < n; ++ i )
            // 选择环上的点
            if (deg[i] && !st[i]) {
                st[i] = true;
                int j = favorite[i], c = 1;
                while (j != i) {
                    st[j] = true;
                    j = favorite[j];
                    c ++ ;
                }
                // now: i = j
                if (c > 2)
                    // case 1: A longest circle
                    res1 = max(res1, c);
                else
                    // case 2: c == 2 (因为不会有长度为 1 的环)
                    // 此时 一个长度为2的环外加各自连上的最长链
                    res2 += depth[i] + depth[favorite[i]];
            }
        return max(res1, res2);
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