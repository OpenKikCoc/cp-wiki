## 定义

边有向，无环。

英文名叫 Directed Acyclic Graph，缩写是 DAG。

## 性质

-   能 [拓扑排序](./topo.md) 的图，一定是有向无环图；

    如果有环，那么环上的任意两个节点在任意序列中都不满足条件了。

-   有向无环图，一定能拓扑排序；

    （归纳法）假设节点数不超过 $k$ 的 有向无环图都能拓扑排序，那么对于节点数等于 $k$ 的，考虑执行拓扑排序第一步之后的情形即可。

## 判定

如何判定一个图是否是有向无环图呢？

检验它是否可以进行 [拓扑排序](./topo.md) 即可。

当然也有另外的方法，可以对图进行一遍 [DFS](search/dfs.md)，在得到的 DFS 树上看看有没有连向祖先的非树边（返祖边）。如果有的话，那就有环了。