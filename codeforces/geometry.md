# 计算几何 (Codeforces 教程索引)

原帖第 6 节，7 篇教程。覆盖 2D 基础、多边形、四元数、Slope trick、最近邻、CHT。

> [!NOTE] **本节定位**
>
> 注意原帖把「Slope trick」与「Convex Hull Trick」放在了几何章节（因其几何直觉），但在本 wiki 中两者属于 DP 优化范畴（见 `dp/opt/slope.md`）。本节保留原分类并跨链。

## 2D / 3D 基础

> [!NOTE] **[Geometry: 2D points and lines [Tutorial]](https://codeforces.com/blog/entry/48122)**
>
> 难度：入门-中等

> [!TIP] **要点**
>
> - 二维向量基本运算：点积、叉积、旋转、投影
> - 直线与点的距离 / 共线 / 平行 / 相交判定
> - 前置：解析几何

> 本站对应：[二维计算几何基础](../geometry/2d.md)、[距离](../geometry/distance.md)

* * *

> [!NOTE] **[Geometry: Polygon algorithms](https://codeforces.com/blog/entry/48868)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 多边形面积（Shoelace 公式）、点是否在多边形内（射线法 / winding number）
> - 凸包基础引出
> - 前置：2D 基础

> 本站对应：[凸包](../geometry/convex-hull.md)、[Pick 定理](../geometry/pick.md)、[二维计算几何基础](../geometry/2d.md)

* * *

> [!NOTE] **[Quaternion algebra and geometry](https://codeforces.com/blog/entry/46744)**
>
> 难度：高级（专精）

> [!TIP] **要点**
>
> - 四元数：$\mathbb{H}$ 上的代数 + 用于 3D 旋转
> - 比矩阵旋转更紧凑、避免万向锁
> - 前置：线性代数、3D 几何

> 本站对应：本 wiki 暂无四元数独立页；可参考 [3D 计算几何](../geometry/3d.md)

* * *

## 邻近搜索 / 几何专题

> [!NOTE] **[Nearest Neighbor Search](https://codeforces.com/blog/entry/54080)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 平面 / 高维最近邻搜索：k-d 树、分治
> - 静态 / 动态版本对比
> - 前置：分治、空间索引

> 本站对应：[平面最近点对](../geometry/nearest-points.md)、[k-d 树](../ds/kdt.md)

* * *

## DP / 几何交界（Slope Trick & CHT）

> [!NOTE] **[Slope Trick](https://codeforces.com/blog/entry/47821)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Slope trick：维护「分段线性凸函数」的转折点集合
> - 解决一类带绝对值 / 凸性的 DP（与 `math.md` 中的另一篇互补）
> - 前置：DP、凸函数

> 本站对应：[斜率优化](../dp/opt/slope.md)

* * *

> [!NOTE] **[Convex Hull trick and Li chao tree](https://codeforces.com/blog/entry/56994)**
>
> 难度：高级

> [!TIP] **要点**
>
> - CHT：维护「直线下凸壳」加速 DP 中的 $\min/\max(ax_i + b)$
> - 李超树：在线段树上维护「下凸最优直线」
> - 前置：单调队列、线段树、凸性

> 本站对应：[斜率优化](../dp/opt/slope.md)、[李超树](../ds/li-chao-tree.md)、[四边形不等式](../dp/opt/quadrangle.md)

* * *

> [!NOTE] **[Convex Hull Trick — Geometry being useful](https://codeforces.com/blog/entry/63823)**
>
> 难度：高级

> [!TIP] **要点**
>
> - CHT 的几何视角讲解 + 例题串讲
> - 与上一篇互补
> - 前置：CHT

> 本站对应：[斜率优化](../dp/opt/slope.md)、[李超树](../ds/li-chao-tree.md)
