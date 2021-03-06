数学是信息学竞赛所在的场所，数学知识是信息学竞赛知识的主体。本文概述信息学竞赛中涉及的数学知识。

在计算科学和计算机科学技术，特别是算法学科和教学发展的早期，“算法工作者需要学习的数学知识”成为了一个问题。最早的算法是从数理逻辑（可计算性，符号演算和图灵机等计算模型）和“组合”（具体的算法问题，例如子集、图上的的最优化问题和计数等）等领域发展起来的。某种程度上，它们被概括为“离散数学”，也即和微积分、流形这样以连续性、极限、无穷等概念为核心的“连续数学”相对的一种数学视角和分支。在计算机体系结构发展起来之后，算法模型更加注重程序实现和现实问题。这样，在离散数学的基础上又产生了“具体数学”这一和计算机程序实践结合更加紧密的算法工作者必修课程。

信息学竞赛中涉及到的数学知识来自离散数学，这些知识同时具有“具体数学”和算法设计结合紧密的特征，以及“竞赛数学”追求基础知识和模型之简洁与注重演绎、迁移的特征。笔者大概将其分为以下几类：

1. **基本模型。** 组合集合论（集族、偏序理论等），组合图论，平面和立体几何关系，字符串理论。这部分理论是信息学竞赛算法问题的基本模型，是各种性质和算法的基础。
2. **算法，数据结构。** 组合判定、构造、优化、计数等算法问题模式，贪心、分治等算法技巧，搜索、动态规划、线性规划、凸规划等对应的算法设计模式，以及数据结构的性质、问题、结构和可持久化等理论。
3. 代数。以多项式、线性空间和矩阵理论等为代表，主要作为基本模型的代数表示而存在。同时包括处理这些对象的算法。例如生成函数、矩阵乘法表示、代数图论；以及高斯消元、快速傅立叶变换、有限微分求解算法等。
4. 数论。主要是同余类以及积性函数理论。
5. 抽象代数。由于线性代数和数论知识的需要，少量引入的群、环、域、格等抽象代数基础理论。
6. 形式语言和自动机理论，作为计算理论和字符串算法的根据与补充。
7. 其它数学模型。这些模型由于各种原因引入 OI，但大都为 1 和 2 的基础模型、算法、问题服务，因此通常只引入定义和知识基础以及和算法设计证明相关的部分。例如概率论、立体计算几何、组合博弈论、拟阵论、杨氏矩阵等。上述 3-6 条也可以看作如此，但它们在 OI 中引入得较多且较成体系。

笔者个人认为，信息竞赛的一种思维方式是，用第 3 条之后的数学知识得出性质指导基于 1、2 条模型的算法设计、优化。因此对于不希望深入学习数学的 OIer，上述第 3 点开始的数学知识可以只在涉及到相关题目时学习。

举几个例子（数字括号代表知识在上述列表中的分类）：

1. 多项式（3）可以优化卷积形式（3）的背包（2)，可以做一些字符串（1）题。
2. 很多递推（2）类型的题背景都是排列组合（2）/概率期望（7），它们又常常使用生成函数（3）推导和解决，并用基于 FFT(3）的分治（2）优化算法效率。
3. 利用同余（4）和环（5）分析图（1）上非简单路径（1）在模（4,5）意义下可能的权值和，并用带权并查集（2）维护。

可以明显发现，上述第 2 条的知识是 OI 作为算法竞赛所必须的。但是，为什么图论（1）就是基本模型，多项式（3）就不是呢？这是笔者基于自身学习、教学经验对现阶段 OI 教学实践做出的一个判断，而不是数学上的事实。在笔者看来，现阶段的教学实践中，信息学竞赛以图论、组合集合论等为主要基础，呈现出注重划分和组合的，注重本体和几何直观的倾向。数学符号的推导，对符号的直观和抽象思维方式在信息学竞赛教学现状中是相对次要的。本文不讨论这种现状的成因，但请读者注意信息学竞赛数学的这种特点。某种程度上它也是信息学竞赛数学的局限性。当然，从数学的角度上图论、组合集合论的知识不具有什么“优先性”或“本体性”，上述表中（1）和（3）以后其它条目某种程度上是可以交换的。该列表仅仅是笔者基于自身学习教学经验给 OIer 描绘的一种图景而已，仅供参考。

另外，高中数学是信息学竞赛数学的基础，至少对课标内、课本上的基本概念和性质请务必熟悉。
