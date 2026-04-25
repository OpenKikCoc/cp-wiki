# 图论 (Codeforces 教程索引)

原帖第 7 节，13 篇教程。聚焦最短路变体、2-SAT、连通性 / DFS 树、欧拉序、二分图、最小生成树。

> [!NOTE] **本节定位**
>
> 本 wiki 的 `graph/` 章节覆盖极完整（60+ 文件，含 `flow/`、`graph-matching/` 子目录）。本节作为英文社区视角的索引层。

## 综合 / 题集

> [!NOTE] **[Algorithm Gym :: Graph Algorithms](https://codeforces.com/blog/entry/16221)**
>
> 难度：综合题集

> [!TIP] **要点**
>
> - 图论训练题集：基础 + 进阶
> - 配套 Gym 比赛
> - 适合按主题刷题

> 本站对应：[图论概念](../graph/concept.md)、[图的存储](../graph/save.md)

* * *

## 最短路 / 搜索变体

> [!NOTE] **[0-1 BFS [Tutorial]](https://codeforces.com/blog/entry/22276)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 0-1 BFS：边权只有 0 / 1 的图最短路用 deque 即可 $O(V+E)$
> - 是 Dijkstra 的特殊高效实现
> - 前置：BFS、双端队列

> 本站对应：[BFS](../graph/bfs.md)、[最短路](../graph/shortest-path.md)

* * *

> [!NOTE] **["Meet in the middle" with shortest path problems of unweighted graph](https://codeforces.com/blog/entry/58894)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 双向搜索 + 最短路结合
> - 在状态空间巨大时折中两端搜索
> - 前置：BFS、状态搜索

> 本站对应：[双向搜索](../search/bidirectional.md)、[BFS](../graph/bfs.md)

* * *

## 2-SAT / 满足性问题

> [!NOTE] **[2-SAT](https://codeforces.com/blog/entry/16205)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 2-SAT：通过蕴含图 + SCC 求解
> - $O(V+E)$ 判定可满足性并构造一组解
> - 前置：SCC、Tarjan / Kosaraju

> 本站对应：[2-SAT](../graph/2-sat.md)、[强连通分量](../graph/scc.md)

* * *

> [!NOTE] **[Vertex cover and 2-SAT](https://codeforces.com/blog/entry/63164)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 把特殊形式的最小点覆盖归约成 2-SAT
> - 拓宽 2-SAT 应用面
> - 前置：2-SAT、二分图

> 本站对应：[2-SAT](../graph/2-sat.md)、[二分图](../graph/bi-graph.md)、[二分图最大匹配](../graph/graph-matching/bigraph-match.md)

* * *

## 连通性 / DFS 树

> [!NOTE] **[Add edges to a digraph to make it strongly connected](https://codeforces.com/blog/entry/15102)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - SCC 缩点 → DAG → 计算「入度 0 / 出度 0 节点数」
> - 经典结论：$\max(\text{src}, \text{dst})$ 条边可使 DAG 变强连通
> - 前置：SCC、DAG

> 本站对应：[强连通分量](../graph/scc.md)、[DAG](../graph/dag.md)、[连通性相关](../graph/connected.md)

* * *

> [!NOTE] **[The DFS tree and its applications](https://codeforces.com/blog/entry/68138)**
>
> 难度：中等（必读）

> [!TIP] **要点**
>
> - DFS 树：把任意无向 / 有向图组织为树边 + 返祖 / 横叉边
> - 在 DFS 树上理解割点 / 桥 / 环 / SCC 的结构
> - 是连通性算法的统一视角
> - 前置：DFS

> 本站对应：[DFS](../graph/dfs.md)、[欧拉序（DFS 序）](../graph/dfs-timestamp.md)

* * *

> [!NOTE] **[The "Bridge Tree" of a graph](https://tanujkhattar.wordpress.com/2016/01/10/the-bridge-tree-of-a-graph/#more-7)**
>
> 难度：高级 / 外站资源

> [!TIP] **要点**
>
> - 桥边缩点得到的树 ＝ Bridge Tree（边双连通分量缩点）
> - 用于路径上桥的数量、必经边等
> - 前置：边双连通分量、桥
> - 来源：Tanuj Khattar 的博客

> 本站对应：[双连通分量](../graph/bcc.md)、[割点和桥](../graph/cut.md)、[圆方树](../graph/block-forest.md)

* * *

> [!NOTE] **[Articulation points and bridges (Tarjan's Algorithm)](https://codeforces.com/blog/entry/71146)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - Tarjan 求割点 / 桥：在 DFS 时维护 `dfn` 与 `low`
> - 经典模板
> - 前置：DFS

> 本站对应：[割点和桥](../graph/cut.md)、[无向图必经点 / 边](../graph/essential-point-edge.md)、[双连通分量](../graph/bcc.md)

* * *

## 树上问题

> [!NOTE] **[On Euler tour trees](https://codeforces.com/blog/entry/18369)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 欧拉序展开：把树问题转化为序列问题
> - 与 LCA、子树查询、树上分治紧密相关
> - 前置：DFS

> 本站对应：[欧拉序（DFS 序）](../graph/dfs-timestamp.md)、[最近公共祖先](../graph/lca.md)

* * *

> [!NOTE] **[Number of Topological Orderings of a Directed Tree](https://codeforces.com/blog/entry/75627)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 有向树的拓扑序数公式：$\frac{n!}{\prod_v \mathrm{size}(v)}$
> - 配合组合恒等式推导
> - 前置：拓扑排序、计数

> 本站对应：[拓扑排序](../graph/topo.md)、[计数 DP](../dp/count.md)

* * *

## 二分图与匹配

> [!NOTE] **[Maximum Independent Set in Bipartite Graphs](https://codeforces.com/blog/entry/72751)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 二分图最大独立集 = 总点数 - 最大匹配（König 定理）
> - 与最小点覆盖互补
> - 前置：匈牙利 / Hopcroft-Karp

> 本站对应：[二分图](../graph/bi-graph.md)、[二分图最大匹配](../graph/graph-matching/bigraph-match.md)、[图匹配](../graph/graph-matching/graph-match.md)

* * *

## 最小生成树

> [!NOTE] **[Boruvka's Algorithm](https://codeforces.com/blog/entry/77760)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - Borůvka：最古老的 MST 算法之一，每轮所有连通块同时挑最小出边
> - 适合并行化与「隐式图」（大量点 + 距离公式）的 MST
> - 前置：并查集、MST 概念

> 本站对应：[最小生成树](../graph/mst.md)
