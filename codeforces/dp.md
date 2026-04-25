# 动态规划 (Codeforces 教程索引)

原帖第 3 节，8 篇教程。本节聚焦树形 DP、子集 DP、数位 DP、组合数学与 DP 的结合。

> [!NOTE] **本节定位**
>
> 本 wiki 的 `dp/` 章节系统覆盖了几乎所有竞赛 DP 范式（线性 / 区间 / 树 / 状压 / 数位 / 计数 / 概率 / 插头 / 动态 / 基环树）。本节优势在于「跨类」与「英文社区视角」的补充。

## 入门 / 题集

> [!NOTE] **[DP Tutorial and Problem List](https://codeforces.com/blog/entry/67679)**
>
> 难度：入门-中等

> [!TIP] **要点**
>
> - 系统化的 DP 入门索引：从 LIS 到背包到状态压缩
> - 配套精选题单，按类型分组
> - 适合刚开始接触 DP 的学习者

> 本站对应：[动态规划基础](../dp/basic.md)、[状态定义](../dp/definition.md)

* * *

## 经典模型

> [!NOTE] **[Optimized solution for Knapsack problem](https://codeforces.com/blog/entry/59606)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 0/1 背包、完全背包的空间压缩
> - 单调队列优化多重背包
> - 二进制拆分与分组背包
> - 前置：基础 DP

> 本站对应：[背包 DP](../dp/knapsack.md)、[单调队列优化](../dp/opt/monotonous-queue-stack.md)

* * *

> [!NOTE] **[Digit DP](https://codeforces.com/blog/entry/53960)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 数位 DP：在数字的十进制（或其他进制）位上做 DP
> - 状态通常包含「当前位、是否贴上界、是否前导零、其它附加信息」
> - 经典题：区间内满足某种数字性质的数的个数
> - 前置：DP、记忆化搜索

> 本站对应：[数位 DP](../dp/number.md)、[记忆化搜索](../dp/memo.md)

* * *

## 树形 DP

> [!NOTE] **[DP on Trees](https://codeforces.com/blog/entry/20935)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 树形 DP：从叶子向根 / 从根向叶传递信息
> - 经典：子树和、最大独立集、树的直径变体
> - 前置：DFS、基础 DP

> 本站对应：[树形 DP](../dp/tree.md)

* * *

> [!NOTE] **[Dp On Trees](https://codeforces.com/blog/entry/63257)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 上一篇的实战补充：5 类常见树形 DP 模式 + 配套例题
> - 包括「换根 DP（rerooting）」的清晰拆解
> - 前置：树形 DP 基础

> 本站对应：[树形 DP](../dp/tree.md)

* * *

## 状压 / 子集

> [!NOTE] **[SOS DP (sum over subsets)](https://codeforces.com/blog/entry/45223)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 子集和 DP：在 $2^N$ 个状态上，按位逐个「合并」子集
> - 复杂度 $O(N \cdot 2^N)$，把朴素 $O(3^N)$ 子集枚举降一档
> - 经典应用：高维前缀和、子集 OR 卷积、子集枚举优化
> - 前置：状压 DP、位运算

> 本站对应：[状压 DP](../dp/state.md)、[状态设计优化](../dp/opt/state.md)

* * *

## 进阶技巧

> [!NOTE] **[Non-trivial DP tricks & Techniques](https://codeforces.com/blog/entry/47764)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 一系列高级 DP 技巧汇总：状态合并、轮廓线 DP、决策单调性、四边形不等式、CHT
> - 适合在掌握基础 DP 后查漏补缺
> - 前置：成体系的 DP 基础

> 本站对应：[DP 优化总览](../dp/opt/README.md)、[斜率优化](../dp/opt/slope.md)、[四边形不等式](../dp/opt/quadrangle.md)、[数据结构优化](../dp/opt/datastruct.md)

* * *

> [!NOTE] **[Recurrent Sequences — Application of combinatorics in DP](https://codeforces.com/blog/entry/54154)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 用组合数学化简 DP 转移：通过递推关系闭式或矩阵快速幂加速
> - 涉及 Catalan、Fibonacci 等典型数列
> - 前置：DP、组合恒等式、矩阵快速幂

> 本站对应：[计数 DP](../dp/count.md)、[组合数学](../math/combinatorics/README.md)
