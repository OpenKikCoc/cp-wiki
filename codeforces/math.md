# 数学 (Codeforces 教程索引)

原帖第 4 节，35 篇教程，是体量最大的一节。涵盖：博弈论、数论（筛 / 素 / 模 / CRT / 反演）、多项式与变换（FFT / NTT / FWT）、组合恒等与容斥、生成函数、线性代数、群论、各类专题。

> [!NOTE] **本节定位**
>
> 本 wiki 的 `math/` 章节是覆盖最深、最完整的章节（100+ 文件，含 7 个子目录）。本节大部分链接都能对应到本站的精写页面。

## 博弈论 / 不变量

> [!NOTE] **[Invariants and monovariants](https://codeforces.com/blog/entry/57216)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 解题方法论：寻找「不变量」（不会变的量）和「半不变量」（单调变化）
> - 用以证明无解 / 终态唯一
> - 适合构造 / 博弈题前的思维训练

> 本站对应：[推导](../topic/deduction.md)、[博弈论思维](../math/game-theory/thinking.md)

* * *

> [!NOTE] **[A blog on the Sprague-Grundy Theorem](https://codeforces.com/blog/entry/63054)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - SG 定理：任意有限公平博弈等价于 Nim
> - SG 函数计算与组合（异或）
> - 前置：博弈论入门

> 本站对应：[公平博弈](../math/game-theory/impartial-game.md)、[博弈论入门](../math/game-theory/intro.md)

* * *

> [!NOTE] **[The Intuition Behind NIM and Grundy Numbers](https://codeforces.com/blog/entry/66040)**
>
> 难度：中等

> [!TIP] **要点**
>
> - Nim 与 Grundy 数的直觉化讲解
> - 配合上一篇形式化定理一起读
> - 前置：异或运算、博弈论

> 本站对应：[公平博弈](../math/game-theory/impartial-game.md)

* * *

> [!NOTE] **[A Beautiful Technique for Some XOR Related Problems](https://codeforces.com/blog/entry/68953)**
>
> 难度：高级

> [!TIP] **要点**
>
> - XOR 在博弈、最优化、构造中的若干高级用法
> - 经典：线性基、子集异或、Nim 变种
> - 前置：位运算、博弈论、线性基

> 本站对应：[公平博弈](../math/game-theory/impartial-game.md)、[线性基](../math/linear-algebra/basis.md)

* * *

## 数论 — 筛与素数

> [!NOTE] **[Extended Eratosthenes Sieve](https://codeforces.com/blog/entry/57212)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 在线性筛中同时筛出最小质因子 / 欧拉函数 / Möbius 函数 / 约数和
> - 是「积性函数线性筛」模板的英文阐述
> - 前置：素数筛

> 本站对应：[筛法](../math/number-theory/sieve.md)、[素数](../math/number-theory/prime.md)

* * *

> [!NOTE] **[Number Theory in Competitive Programming [Tutorial]](https://codeforces.com/blog/entry/46620)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 数论速览：模运算、GCD、扩展欧几里得、欧拉定理、费马小定理
> - 适合数论入门串讲
> - 前置：基础代数

> 本站对应：[数论基础](../math/number-theory/basic.md)、[GCD](../math/number-theory/gcd.md)、[Euler 定理](../math/number-theory/euler.md)、[Fermat 小定理](../math/number-theory/fermat.md)

* * *

## 数论 — 模运算 / 同余 / 丢番图

> [!NOTE] **[Avoid overflow in linear diophantine equation](https://codeforces.com/blog/entry/59842)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 扩展欧几里得求解 $ax + by = c$
> - 关注溢出陷阱与有效求解范围
> - 前置：扩展 GCD

> 本站对应：[Bezout 定理](../math/number-theory/bezouts.md)、[线性方程](../math/number-theory/linear-equation.md)

* * *

> [!NOTE] **[Chinese Remainder Theorem](https://codeforces.com/blog/entry/61290)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - CRT：解一组同余方程组
> - 模数互质 / 不互质两个版本（后者称扩展 CRT）
> - 前置：模运算、扩展 GCD

> 本站对应：[CRT](../math/number-theory/crt.md)

* * *

> [!NOTE] **[Modular Arithmetic for Beginners](https://codeforces.com/blog/entry/72527)**
>
> 难度：入门-中等

> [!TIP] **要点**
>
> - 模运算的基本性质 + 模逆元（费马 / 扩展 GCD / 线性递推）
> - 处理负数取模、防溢出
> - 前置：基本算术

> 本站对应：[逆元](../math/number-theory/inverse.md)、[Fermat 小定理](../math/number-theory/fermat.md)

* * *

## 数论 — 高级（反演 / Berlekamp-Massey）

> [!NOTE] **[Mobius Inversion](https://codeforces.com/blog/entry/53925)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Möbius 反演：$f(n) = \sum_{d|n} g(d) \Leftrightarrow g(n) = \sum_{d|n} \mu(d) f(n/d)$
> - 数论函数求和的核心工具
> - 前置：积性函数、约数枚举

> 本站对应：[Möbius 反演](../math/number-theory/mobius.md)

* * *

> [!NOTE] **[Mobius Inversion and Multiplicative functions : Tutorial](https://codeforces.com/blog/entry/67693)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Möbius + Dirichlet 卷积 + 积性函数体系
> - 配套经典推导题
> - 前置：上一篇

> 本站对应：[Möbius 反演](../math/number-theory/mobius.md)、[杜教筛](../math/number-theory/du.md)、[Min_25 筛](../math/number-theory/min-25.md)、[powerful number 筛](../math/number-theory/powerful-number.md)

* * *

> [!NOTE] **[Linear Recurrence and Berlekamp-Massey Algorithm](https://codeforces.com/blog/entry/61306)**
>
> 难度：高级（专精）

> [!TIP] **要点**
>
> - Berlekamp-Massey：从前缀求最短线性递推关系
> - 配合常系数线性递推 / Cayley-Hamilton 做高阶矩阵幂加速
> - 前置：线性代数、矩阵快速幂

> 本站对应：[线性递推](../math/linear-recurrence.md)、[特征多项式](../math/linear-algebra/char-poly.md)

* * *

## 多项式与卷积 (FFT / NTT / FWT)

> [!NOTE] **[Tutorial for FFT/NTT Part 1](https://codeforces.com/blog/entry/43499)**
>
> 难度：高级

> [!TIP] **要点**
>
> - FFT 原理：单位根、分治蝶形、迭代 vs 递归
> - 用于多项式乘法 $O(N \log N)$
> - 前置：复数、多项式

> 本站对应：[FFT](../math/poly/fft.md)、[多项式入门](../math/poly/intro.md)

* * *

> [!NOTE] **[Tutorial for FFT/NTT Part 2](https://codeforces.com/blog/entry/48798)**
>
> 难度：高级

> [!TIP] **要点**
>
> - NTT：模意义下的 FFT
> - 处理大整数 / 模数下的卷积
> - 前置：FFT、原根

> 本站对应：[NTT](../math/poly/ntt.md)、[原根](../math/number-theory/primitive-root.md)

* * *

> [!NOTE] **[On Fast Fourier Transform](https://codeforces.com/blog/entry/55572)**
>
> 难度：高级

> [!TIP] **要点**
>
> - FFT 的工程实现：常数优化、合并实部 / 虚部
> - 实测对比版

> 本站对应：[FFT](../math/poly/fft.md)

* * *

> [!NOTE] **[Dirichlet convolution](https://codeforces.com/blog/entry/54150)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Dirichlet 卷积 $(f * g)(n) = \sum_{d|n} f(d) g(n/d)$
> - 数论函数代数结构（积性函数下的卷积）
> - 与 Möbius 反演紧密相关

> 本站对应：[Möbius 反演](../math/number-theory/mobius.md)、[杜教筛](../math/number-theory/du.md)、[DGF 生成函数](../math/gen-func/dgf.md)

* * *

> [!NOTE] **[Fast convolution for 64-bit integers](https://codeforces.com/blog/entry/45298)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 64-bit 卷积：多个 NTT 模数 + CRT 合成
> - 工程上常见做法
> - 前置：NTT、CRT

> 本站对应：[NTT](../math/poly/ntt.md)、[CRT](../math/number-theory/crt.md)

* * *

> [!NOTE] **[Schonhage-Strassen (FFT-based integer multiplication)](https://codeforces.com/blog/entry/63446)**
>
> 难度：高级（专精）

> [!TIP] **要点**
>
> - Schönhage–Strassen：基于 FFT 的大整数乘法 $O(N \log N \log \log N)$
> - 大整数库底层算法
> - 前置：FFT、数论变换、复杂度分析

> 本站对应：本 wiki 暂无独立页；[大整数](../math/bignum.md) + [FFT](../math/poly/fft.md) 可作为参考

* * *

> [!NOTE] **[A Bitwise Convolution Tutorial](https://codeforces.com/blog/entry/65154)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 位运算卷积（OR / AND / XOR）的统一框架
> - SOS DP 即 OR 卷积；XOR 卷积即 FWT
> - 前置：FFT 思想、位运算

> 本站对应：[FWT](../math/poly/fwt.md)、[状压 DP（SOS）](../dp/state.md)

* * *

> [!NOTE] **[Fast Walsh Hadamard Transforms and its inner workings](https://codeforces.com/blog/entry/71899)**
>
> 难度：高级

> [!TIP] **要点**
>
> - FWHT 的代数推导：$2 \times 2$ 蝶形矩阵分解
> - OR / AND / XOR 三种 FWT 的统一视角
> - 前置：线性代数、卷积

> 本站对应：[FWT](../math/poly/fwt.md)

* * *

## 组合 / 容斥 / 排列

> [!NOTE] **[Inclusion-Exclusion Principle, Part 1](https://codeforces.com/blog/entry/64625)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - 容斥原理基础与典型应用
> - 至少 / 至多类计数转换
> - 前置：组合恒等式

> 本站对应：[容斥原理](../math/combinatorics/inclusion-exclusion-principle.md)

* * *

> [!NOTE] **[Derangement Generation of an Array [Tutorial]](https://codeforces.com/blog/entry/66176)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 错排数（derangement）的递推与生成
> - 容斥推导 / 生成函数推导
> - 前置：容斥、阶乘

> 本站对应：[排列与组合](../math/combinatorics/combination.md)、[容斥原理](../math/combinatorics/inclusion-exclusion-principle.md)

* * *

## 群论 / 计数

> [!NOTE] **[Burnside Lemma](https://codeforces.com/blog/entry/51272)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Burnside 引理：等价类计数 = 平均不动点数
> - Polya 计数定理是其特例
> - 前置：群作用、置换群

> 本站对应：[置换群](../math/permutation-group.md)、[群论](../math/group-theory.md)

* * *

> [!NOTE] **[On burnside (again)](https://codeforces.com/blog/entry/64860)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Burnside 的另一种讲法 + 题型补充
> - 与上一篇互补
> - 前置：Burnside

> 本站对应：[置换群](../math/permutation-group.md)、[群论](../math/group-theory.md)

* * *

## 线性代数 / 高斯消元

> [!NOTE] **[2 Special cases of Gaussian elimination](https://codeforces.com/blog/entry/60003)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 高斯消元的两个特殊场景：模意义、bool（异或方程组）
> - bool 高斯消元用 bitset 加速
> - 前置：线性代数

> 本站对应：[高斯消元](../math/linear-algebra/gauss.md)

* * *

> [!NOTE] **[Number of Solutions to a Linear Algebraic Equation](https://codeforces.com/blog/entry/54111)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 由秩判定线性方程组解的个数（无解 / 唯一 / 无穷）
> - 配合高斯消元使用
> - 前置：矩阵秩

> 本站对应：[高斯消元](../math/linear-algebra/gauss.md)、[矩阵](../math/linear-algebra/matrix.md)

* * *

> [!NOTE] **[The Fear of Gaussian Elimination](https://codeforces.com/blog/entry/65787)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 工程化教程：浮点稳定性、主元选取、模数下消元
> - 实战调试经验

> 本站对应：[高斯消元](../math/linear-algebra/gauss.md)

* * *

## 哈希 / 字符串数学

> [!NOTE] **[Rolling hash and 8 interesting problems](https://codeforces.com/blog/entry/60445)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 滚动哈希原理与 8 个经典题
> - 双模数 / 防 hack 注意点
> - 前置：哈希、字符串

> 本站对应：[字符串哈希](../string/hash.md)、[hash 技巧](../topic/hash.md)

* * *

## 高级专题（拟阵 / 矩阵 / ODE / Slope trick）

> [!NOTE] **[Matroid intersection in simple words](https://codeforces.com/blog/entry/69287)**
>
> 难度：高级（专精）

> [!TIP] **要点**
>
> - 拟阵交（matroid intersection）：在两个拟阵交集上求最大独立集
> - 推广二分图最大匹配的强力框架
> - 前置：图论、线性代数

> 本站对应：本 wiki 暂无独立页；可参考 [图匹配](../graph/graph-matching/graph-match.md)

* * *

> [!NOTE] **[Solving Linear Recurrence for Programming Contest](http://fusharblog.com/solving-linear-recurrence-for-programming-contest/)**
>
> 难度：高级 / 外站资源

> [!TIP] **要点**
>
> - 常系数线性递推的矩阵快速幂解法
> - FusharBlog 上的经典英文教程
> - 前置：矩阵快速幂

> 本站对应：[线性递推](../math/linear-recurrence.md)、[矩阵](../math/linear-algebra/matrix.md)

* * *

> [!NOTE] **[A problem collection of ODE and differential technique](https://codeforces.com/blog/entry/76447)**
>
> 难度：高级（专精）

> [!TIP] **要点**
>
> - 把 OI 题套用 ODE / 差分方程方法
> - 适合处理「无穷状态系统的稳态」类题
> - 前置：微积分、生成函数

> 本站对应：本 wiki 暂无独立页；可参考 [生成函数](../math/gen-func/intro.md)

* * *

> [!NOTE] **[Slope trick explained](https://codeforces.com/blog/entry/77298)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Slope trick：把分段线性凸函数维护为「转折点 + 当前最小值」
> - 处理一类带绝对值 / 单调性约束的 DP
> - 前置：DP、凸性

> 本站对应：[斜率优化](../dp/opt/slope.md)

* * *

## 生成函数

> [!NOTE] **[Generating Functions in Competitive Programming (Part 1)](https://codeforces.com/blog/entry/77468)**
>
> 难度：高级

> [!TIP] **要点**
>
> - OGF / EGF 入门：定义、运算、与计数题对应
> - 用形式幂级数解递推
> - 前置：组合数学、级数

> 本站对应：[生成函数入门](../math/gen-func/intro.md)、[OGF](../math/gen-func/ogf.md)、[EGF](../math/gen-func/egf.md)

* * *

> [!NOTE] **[Generating Functions in Competitive Programming (Part 2)](https://codeforces.com/blog/entry/77551)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Part 1 的进阶：复合、求逆、微分方程化简
> - 配合多项式操作模板使用
> - 前置：Part 1

> 本站对应：[生成函数入门](../math/gen-func/intro.md)、[多项式求逆](../math/poly/inv.md)、[多项式 ln/exp](../math/poly/ln-exp.md)、[DGF](../math/gen-func/dgf.md)

* * *

## 模数运算优化

> [!NOTE] **[Addendum: Optimized variant of Barrett reduction](https://codeforces.com/blog/entry/75406)**
>
> 难度：高级（工程）

> [!TIP] **要点**
>
> - Barrett reduction：用乘法 + 移位代替整数除法的取模优化
> - 模数非编译时常量时显著加速
> - 工程化模板（多见于 AtCoder Library）

> 本站对应：本 wiki 暂无独立页；可参考 [取模 / 逆元](../math/number-theory/inverse.md)
