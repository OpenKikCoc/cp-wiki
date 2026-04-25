# 数据结构 (Codeforces 教程索引)

原帖第 2 节，34 篇教程。本节按本站常用分类组织：综合 / 题集、线段树族、树上分解、链树 / 平衡树形、高维 / 区间查询结构、Fenwick (BIT)、莫队算法、特殊 / 专用结构。

> [!NOTE] **本节定位**
>
> 数据结构是本 wiki 覆盖最深的章节之一（`ds/` 下 60+ 文件）。本页主要起跨链作用，让读者从 CF 教程一键跳到本站对应深入页。

## 综合 / 题集

> [!NOTE] **[Algorithm Gym :: Data Structures](https://codeforces.com/blog/entry/15729)**
>
> 难度：综合题集

> [!TIP] **要点**
>
> - 数据结构系统训练题集
> - 链接到 Gym 比赛 + 题目分类
> - 适合按结构刷题

* * *

> [!NOTE] **[Top 10 optimizations 2017 (collectors edition)](https://codeforces.com/blog/entry/53168)**
>
> 难度：综合 / 优化技巧

> [!TIP] **要点**
>
> - 2017 年度社区评选的 10 大优化技巧
> - 涵盖 Fenwick、bitset、CHT、根号分块等多类
> - 跨主题入门读物

* * *

## 线段树族

> [!NOTE] **[Everything about Segment Trees](https://codeforces.com/blog/entry/15890)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 线段树体系化教程：基本结构、单点 / 区间修改、懒标记
> - 各类经典应用（区间加 + 区间和、区间最大）
> - 前置：树的基本概念

> 本站对应：[线段树](../ds/seg.md)

* * *

> [!NOTE] **[Efficient and easy Segtree](https://codeforces.com/blog/entry/18051)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 自底向上、迭代式（非递归）线段树写法
> - 代码短、常数小，适合非懒线段树
> - 替代「正统递归 + 懒」写法的轻量方案

> 本站对应：[线段树](../ds/seg.md)

* * *

> [!NOTE] **[A simple introduction to "Segment tree beats"](https://codeforces.com/blog/entry/57319)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 吉司机线段树（Segment Tree Beats）
> - 处理「区间取 min/max + 区间求和 / 区间历史最值」类问题
> - 关键思想：把「不需要递归到底」的修改截在 max-seg / min-seg 上
> - 前置：线段树、懒标记

> 本站对应：[Segment tree beats](../ds/seg-beats.md)

* * *

> [!NOTE] **[Easy and (Semi)Efficient Dynamic Segment Trees](https://codeforces.com/blog/entry/60837)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 动态开点线段树（按需分配节点）
> - 处理值域极大但实际只有少量插入的场景
> - 前置：线段树、指针 / 池式分配

> 本站对应：[线段树](../ds/seg.md)、[可持久化线段树](../ds/persistent-seg.md)

* * *

> [!NOTE] **[Compressed segment trees and merging sets in O(N logU)](https://codeforces.com/blog/entry/83170)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 坐标压缩 + 启发式合并的线段树合并
> - 复杂度 $O(N \log U)$，$U$ 为值域
> - 适用「树上每点维护值集合 + 自底向上合并」类问题（树上启发式合并的近亲）

> 本站对应：[线段树](../ds/seg.md)、[树上启发式合并](../graph/dsu-on-tree.md)

* * *

> [!NOTE] **[Generalizing Segment Trees with Rust](https://codeforces.com/blog/entry/68419)**
>
> 难度：高级 / 工程

> [!TIP] **要点**
>
> - 用 Rust 的 trait 把线段树抽象成「半群 + lazy 单子」的通用模板
> - 适合理解线段树代数本质（monoid + monoid action）

> 本站对应：[线段树](../ds/seg.md)

* * *

## 树上分解 / 子树技巧

> [!NOTE] **[Sack (DSU on trees)](https://codeforces.com/blog/entry/44351)**
>
> 难度：高级（必读）

> [!TIP] **要点**
>
> - 「轻重儿子保留」式启发式合并：复杂度 $O(N \log N)$
> - 解决静态子树查询：每个子树需要全集统计的题型
> - 前置：DFS、并查集思想（虽然名字带 DSU，本质是合并思想）

> 本站对应：[树上启发式合并](../graph/dsu-on-tree.md)

* * *

> [!NOTE] **[dsu on trees](https://codeforces.com/blog/entry/67696)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 同一技巧的另一种讲法，配合更直观的图与代码模板
> - 与上一篇互为补充

> 本站对应：[树上启发式合并](../graph/dsu-on-tree.md)

* * *

> [!NOTE] **[Two ways to apply Mo's Algorithm on Trees](https://codeforces.com/blog/entry/68271)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 树上莫队的两种做法：欧拉序展开 vs 路径分解
> - 比较两者复杂度、实现难度、适用题型

> 本站对应：[树上莫队](../misc/mo-algo-on-tree.md)

* * *

> [!NOTE] **[Centroid Decomposition on a tree (Beginner)](https://codeforces.com/blog/entry/73707)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 点分治：每次取重心，统计跨重心的路径，递归子树
> - 经典应用：树上路径长度计数、距离 ≤ k 路径数
> - 前置：DFS、树的重心

> 本站对应：[树的重心](../graph/tree-centroid.md)、[树分治](../graph/tree-divide.md)

* * *

> [!NOTE] **[Online Query Based Rerooting Technique](https://codeforces.com/blog/entry/76150)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 树形 DP 换根技术（rerooting）的在线版本
> - 思路：预处理子树 DP + 父亲贡献，把答案分两部分合并
> - 前置：树形 DP

> 本站对应：[树形 DP](../dp/tree.md)

* * *

> [!NOTE] **[Mo's Algorithm on Trees](https://codeforces.com/blog/entry/43230)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 把树上路径展开成 DFS 序的「有效区间」，再跑标准莫队
> - 与点分治形成互补
> - 前置：莫队、欧拉序

> 本站对应：[树上莫队](../misc/mo-algo-on-tree.md)

* * *

## 链树 / 平衡树形

> [!NOTE] **[Link-Cut tree](https://codeforces.com/blog/entry/11241)**
>
> 难度：高级

> [!TIP] **要点**
>
> - LCT：动态森林上 link/cut/查链上信息的核心结构
> - 用 splay 维护「preferred path」森林
> - 前置：splay、均摊分析

> 本站对应：[Link-Cut Tree](../ds/lct.md)、[动态树](../ds/dynamic-tree.md)

* * *

> [!NOTE] **[Maintaining subtree information using link/cut trees](https://codeforces.com/blog/entry/67637)**
>
> 难度：高级

> [!TIP] **要点**
>
> - LCT 拓展：在虚子树（virtual subtree）上维护额外信息
> - 解决「LCT 形态下需要子树聚合」的题型
> - 前置：LCT 基础

> 本站对应：[Link-Cut Tree](../ds/lct.md)

* * *

> [!NOTE] **[Link Cut Tree implementation](https://codeforces.com/blog/entry/75885)**
>
> 难度：高级 / 工程

> [!TIP] **要点**
>
> - 实操向：清晰可读的 LCT 模板代码与调试经验
> - 与理论文章配合阅读

> 本站对应：[Link-Cut Tree](../ds/lct.md)

* * *

> [!NOTE] **[Tutorial on Permutation Tree (析合树)](https://codeforces.com/blog/entry/78898)**
>
> 难度：高级（专精）

> [!TIP] **要点**
>
> - 析合树（Permutation Tree）：把排列按「连续段」结构分解为「合点 / 析点」组成的树
> - 用于排列上区间询问 / 区间合法子段统计
> - 国内多见于省选 / 集训队水平
> - 前置：单调栈、笛卡尔树

> 本站对应：本 wiki 暂无独立页，可参考 [笛卡尔树](../ds/cartesian-tree.md) 类比

* * *

## 高维 / 区间查询结构

> [!NOTE] **[SQRT Tree](https://codeforces.com/blog/entry/57046)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 根号树：在 $\sqrt{N}$ 块上再做一层根号，做到「$O(1)$ 查询 + $O(\sqrt{N})$ 修改」
> - 适合静态区间结合律（min、gcd 等）多次查询
> - 前置：根号分块、稀疏表

> 本站对应：[根号树](../ds/sqrt-tree.md)、[Sparse Table](../ds/sparse-table.md)

* * *

> [!NOTE] **[Introduction to New Data Structure: Wavelet Trees](https://codeforces.com/blog/entry/52854)**
>
> 难度：高级（专精）

> [!TIP] **要点**
>
> - 小波树：基于值域二分的递归结构
> - 支持区间第 k 大、区间内值域计数等
> - 复杂度 $O(\log V)$ 每次查询
> - 国内 OI 主流改用「主席树（可持久化线段树）」做同类问题

> 本站对应：本 wiki 暂无 wavelet 单独页；可参考 [可持久化线段树](../ds/persistent-seg.md) 解决同类问题

* * *

> [!NOTE] **[Sparse table](https://codeforces.com/blog/entry/66643)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 稀疏表：$O(N \log N)$ 预处理、$O(1)$ 查询满足幂等结合律的区间询问（min / gcd / max）
> - 经典 RMQ 实现

> 本站对应：[Sparse Table](../ds/sparse-table.md)、[RMQ](../topic/rmq.md)

* * *

> [!NOTE] **[On Multidimensional Range Queries](https://codeforces.com/blog/entry/71038)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 二维 / 多维区间查询：嵌套 BIT、二维线段树、k-d 树等思路
> - 各方案的复杂度与适用场景对照

> 本站对应：[k-d 树](../ds/kdt.md)、[嵌套结构（树套树）](../ds/tree-in-tree.md)、[BIT 套块状数组](../ds/bit-in-block-array.md)

* * *

> [!NOTE] **[Enumerating all Binary Trees to build O(n)/O(1) RMQ](https://codeforces.com/blog/entry/71706)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 笛卡尔树 + 块状 + 预表得到 $O(N)/O(1)$ RMQ
> - 解决经典 RMQ 的渐近最优结构
> - 前置：笛卡尔树、ST 表

> 本站对应：[笛卡尔树](../ds/cartesian-tree.md)、[Sparse Table](../ds/sparse-table.md)

* * *

> [!NOTE] **[Path max queries on a tree in O(1)](https://codeforces.com/blog/entry/71568)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 树上路径最大值的 $O(1)$ 查询：DFS 序 + LCA + RMQ
> - 与上一篇 RMQ 思路结合
> - 前置：LCA、RMQ

> 本站对应：[最近公共祖先](../graph/lca.md)、[RMQ](../topic/rmq.md)

* * *

> [!NOTE] **[2D Range Minimum Query in O(1)](https://codeforces.com/blog/entry/45485)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 静态二维 RMQ：$O(N M \log N \log M)$ 预处理 + $O(1)$ 查询
> - 适合矩阵范围最值题

> 本站对应：[Sparse Table](../ds/sparse-table.md)

* * *

## Fenwick (BIT)

> [!NOTE] **[Understanding Fenwick Trees / Binary Indexed Trees](https://codeforces.com/blog/entry/57292)**
>
> 难度：中等

> [!TIP] **要点**
>
> - BIT 的 lowbit 结构、单点修改 + 前缀和的 $O(\log N)$ 实现
> - 推广到差分（区间修改 + 单点查询）

> 本站对应：[树状数组](../ds/fenwick.md)

* * *

> [!NOTE] **[Searching Binary Indexed Tree in O(log(N)) using Binary Lift](https://codeforces.com/blog/entry/61364)**
>
> 难度：高级

> [!TIP] **要点**
>
> - BIT 上倍增二分：$O(\log N)$ 找前缀和首次 ≥ k 的位置
> - 替代「线段树二分」的更轻量写法
> - 前置：BIT、倍增

> 本站对应：[树状数组](../ds/fenwick.md)、[倍增](../basic/binary-lifting.md)

* * *

> [!NOTE] **[Easy implementation of Compressed 2D BIT for grid queries](https://codeforces.com/blog/entry/52094)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 二维 BIT + 坐标压缩：处理稀疏二维点 + 矩形和
> - 工程上常作为模板使用

> 本站对应：[树状数组](../ds/fenwick.md)、[BIT 套块状数组](../ds/bit-in-block-array.md)、[离散化](../misc/discrete.md)

* * *

> [!NOTE] **[Nifty implementation of multi-dimensional Binary Indexed Trees](https://codeforces.com/blog/entry/64914)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 多维 BIT 的递归 / 模板写法
> - 维度数 $D$、长度 $N$，每次操作 $O(\log^D N)$

> 本站对应：[树状数组](../ds/fenwick.md)

* * *

## 莫队算法

> [!NOTE] **[Mo's Algorithm (with update and without update)](https://codeforces.com/blog/entry/72690)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 标准莫队 + 带修莫队
> - 离线区间问题的根号分块解法
> - 前置：根号分块

> 本站对应：[莫队算法](../misc/mo-algo.md)、[带修改莫队](../misc/modifiable-mo-algo.md)

* * *

> [!NOTE] **[An alternative sorting order for Mo's algorithm](https://codeforces.com/blog/entry/61203)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 莫队的「希尔伯特曲线 / 折线排序」优化
> - 实测能再降一些常数
> - 前置：莫队

> 本站对应：[莫队算法简介](../misc/mo-algo-intro.md)、[莫队算法](../misc/mo-algo.md)

* * *

## 特殊 / 专用结构

> [!NOTE] **[How can we perform segment queries with Palindromic Tree?](https://codeforces.com/blog/entry/63149)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 在回文树（PAM）上挂区间询问
> - 经典玩法：动态加字符 + 回文统计
> - 前置：PAM、Manacher

> 本站对应：[回文树](../string/pam.md)、[Manacher](../string/manacher.md)

* * *

> [!NOTE] **[A powerful representation of integer sets](https://codeforces.com/blog/entry/83969)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 用「按 popcount 分层 + bitset」的方式高效维护整数集合
> - 适合处理子集枚举 / 子集和 / 高维偏序
> - 与 SOS DP 思想接近

> 本站对应：[bitset](../lang/bitset.md)

* * *

> [!NOTE] **[Square root decomposition and applications](https://codeforces.com/blog/entry/83248)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 根号分块系统化教程：原理 + 经典题型
> - 包括块状数组、操作分块、值域分块的不同变体
> - 前置：基础数据结构

> 本站对应：[分块/根号分解](../ds/decompose.md)、[分块算法简介](../ds/decompose-intro.md)、[块状数组](../ds/block-array.md)、[块状链表](../ds/block-list.md)
