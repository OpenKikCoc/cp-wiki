# Codeforces 教程索引

本节是对 Codeforces 经典聚合贴 [**All the good tutorials found for Competitive Programming**](https://codeforces.com/blog/entry/57282)（作者 *vaibhav1997*，2017-07-14）的中文整理与本站交叉索引。原帖按主题归类收录了 **123** 篇高质量算法教程链接，是社区公认的体系化学习入口之一。

本节定位为 **桥接索引**，而非二次教学：

- 每条教程保留 **原英文标题 + Codeforces 原文链接**，便于检索；
- 给出 **2–4 条中文要点摘要 + 前置知识**；
- 若本 wiki 已有对应主题的深入页面，附 **「本站对应」** 跨链；若没有，则按需补充自洽的简介；
- 整体目录与原帖 9 大类一一对应，但在各类之内根据教程主题做了二级聚类（例如「数据结构」拆分为线段树族 / 树上分解 / 高维查询 / BIT / 莫队 / 特殊结构等）。

> [!NOTE] **致谢与版权**
>
> 原帖与所引教程版权归各自作者所有。本节仅做索引、摘要、跨链使用，遵循 [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/deed.zh)。如需深入学习，请直接访问原文链接。

## 目录

| # | 类别 | 教程数 | 本站对应文件 |
|---|---|---|---|
| 1 | [C++ 技巧](cpp.md) | 10 | `lang/`, `ds/hash.md` |
| 2 | [数据结构](data-structures.md) | 34 | `ds/`, `misc/mo-algo*.md`, `graph/dsu-on-tree.md` |
| 3 | [动态规划](dp.md) | 8 | `dp/` |
| 4 | [数学](math.md) | 35 | `math/`（数论 / 多项式 / 生成函数 / 组合 / 线性代数 / 博弈论） |
| 5 | [字符串处理](string.md) | 5 | `string/`（含 1 篇 Trie 外站资源） |
| 6 | [计算几何](geometry.md) | 7 | `geometry/`, `dp/opt/slope.md`, `ds/li-chao-tree.md` |
| 7 | [图论](graphs.md) | 13 | `graph/` |
| 8 | [其它技巧](others.md) | 10 | `basic/binary.md`, `misc/parallel-binsearch.md` |
| 9 | [拓展资源](resources.md) | 1 | — |
| **合计** |  | **123** |  |

> [!TIP] **教程总数说明**
>
> 原帖正文共 123 条教程链接（120 篇 CF 博客 + 3 篇外站：Quora Trie / FusharBlog 线性递推 / Tanuj Khattar Bridge Tree）。本节按原帖 9 大类一比一映射，不漏一条；「拓展资源」补充了若干同类元索引贴，方便延伸阅读。

## 学习路径建议

原帖是按主题平铺的，缺少难度坡度的指引。结合本 wiki 的覆盖与 CP 一般规律，建议按下面四层顺序阅读：

### Tier 1 — 基础 (Foundations)
入门或新接触 CP 的读者从此层开始。重点是 C++ 工具链与基础范式。

- [Competitive C++ Manifesto: A Style Guide](https://codeforces.com/blog/entry/64218) — 风格规范
- [C++ STL: map and set](https://codeforces.com/blog/entry/9702) — STL 容器入门
- [C++ Tricks](https://codeforces.com/blog/entry/15643) — 常用语言技巧
- [Modular Arithmetic for Beginners](https://codeforces.com/blog/entry/72527) — 取模运算
- [DP Tutorial and Problem List](https://codeforces.com/blog/entry/67679) — DP 入门题单
- [Number Theory in Competitive Programming](https://codeforces.com/blog/entry/46620) — 数论速览
- [Geometry: 2D points and lines](https://codeforces.com/blog/entry/48122) — 2D 几何

### Tier 2 — 核心进阶 (Core Intermediate)
熟练掌握分治、单调队列、典型 DP、基础图论；进入主流难题区。

- [Everything about Segment Trees](https://codeforces.com/blog/entry/15890) — 线段树
- [Understanding Fenwick Trees](https://codeforces.com/blog/entry/57292) — BIT
- [Sparse table](https://codeforces.com/blog/entry/66643) — RMQ
- [DP on Trees](https://codeforces.com/blog/entry/20935) — 树形 DP
- [SOS DP](https://codeforces.com/blog/entry/45223) — 子集和 DP
- [Digit DP](https://codeforces.com/blog/entry/53960) — 数位 DP
- [Z algorithm](https://codeforces.com/blog/entry/3107) — Z 函数
- [0-1 BFS](https://codeforces.com/blog/entry/22276) — 双端队列 BFS
- [The DFS tree and its applications](https://codeforces.com/blog/entry/68138) — DFS 树
- [Articulation points and bridges](https://codeforces.com/blog/entry/71146) — Tarjan 割点 / 桥

### Tier 3 — 高级数据结构与图论 (Advanced DS & Graph)
进入区域赛 / 网络赛主力题型。

- [Segment tree beats](https://codeforces.com/blog/entry/57319) — 吉司机线段树
- [Mo's Algorithm](https://codeforces.com/blog/entry/72690) + 树上 / 回滚 / bitset 变种
- [Sack (DSU on trees)](https://codeforces.com/blog/entry/44351) — 树上启发式合并
- [Centroid Decomposition](https://codeforces.com/blog/entry/73707) — 点分治
- [Link-Cut Tree](https://codeforces.com/blog/entry/11241) — LCT
- [Wavelet Trees](https://codeforces.com/blog/entry/52854) — 小波树
- [SQRT Tree](https://codeforces.com/blog/entry/57046) — 根号树
- [2-SAT](https://codeforces.com/blog/entry/16205) — 2-SAT
- [Convex Hull trick and Li Chao tree](https://codeforces.com/blog/entry/56994) — CHT/李超树

### Tier 4 — 专精数学 (Specialised Math)
长链条数学：多项式、生成函数、群论、博弈、数论高级。

- [Tutorial for FFT/NTT Part 1](https://codeforces.com/blog/entry/43499) + [Part 2](https://codeforces.com/blog/entry/48798)
- [Mobius Inversion](https://codeforces.com/blog/entry/53925) / [多重函数 + Möbius](https://codeforces.com/blog/entry/67693)
- [Generating Functions in CP — Part 1](https://codeforces.com/blog/entry/77468) + [Part 2](https://codeforces.com/blog/entry/77551)
- [Linear Recurrence and Berlekamp-Massey](https://codeforces.com/blog/entry/61306)
- [Burnside Lemma](https://codeforces.com/blog/entry/51272)
- [Inclusion-Exclusion Principle](https://codeforces.com/blog/entry/64625)
- [Sprague–Grundy](https://codeforces.com/blog/entry/63054)
- [Slope trick explained](https://codeforces.com/blog/entry/77298)
- [Matroid intersection](https://codeforces.com/blog/entry/69287)

## 覆盖度矩阵

下表记录原帖每个类别中：(a) 已链接到本 wiki 现有页面的教程数，(b) 仅保留外链摘要的教程数。后者代表本 wiki 当前内容的潜在补充方向（教程难度高 / 主题偏专精 / 工程性较强）。

| 类别 | 教程总数 | 已链本站 | 仅外链或参考 | 本站尚缺的代表性主题 |
|---|---|---|---|---|
| C++ | 10 | 5 | 5 | PBDS、unordered_map 攻击防御、faster lower_bound、64-bit bitset、GCC 调试技巧、CP 风格指南、STL 通用 I/O 模板 |
| Data Structures | 34 | 30 | 4 | 析合树（permutation tree）、Wavelet tree、Algorithm Gym 题集、Top 10 优化 |
| DP | 8 | 8 | 0 | — |
| Math | 35 | 31 | 4 | Schönhage-Strassen、Barrett reduction、ODE 技巧、Matroid intersection |
| String | 5 | 5 | 0 | — |
| Geometry | 7 | 6 | 1 | 四元数与三维旋转工程化 |
| Graphs | 13 | 13 | 0 | — |
| Others | 10 | 5 | 5 | 通用问题模式总结、面试导向、视频资源、Gym 题集 |
| Resources | 1 | 0 | 1 | 元索引贴本身 |
| **合计** | **123** | **103** | **20** | 约 84% 已与本站交叉链接 |

> [!NOTE] **使用方式**
>
> - 想体系化复习某个大主题：从 `_sidebar` 直接进入 wiki 对应章节（`ds/`、`graph/` 等），更深入。
> - 想找特定教程或对照英文社区资料：来 `codeforces/` 这一节按类别展开。
> - 想拓展 wiki 内容：参考上表「仅外链」这部分，是天然的写作清单。
