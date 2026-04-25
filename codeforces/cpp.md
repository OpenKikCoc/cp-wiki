# C++ 技巧 (Codeforces 教程索引)

原帖第 1 节，10 篇教程，聚焦 C++ 语言层面的竞赛编程技巧、STL 用法、性能优化与防御性编程。

> [!NOTE] **本节定位**
>
> 这一节大多是「单点技巧」，与本 wiki `lang/` 章节有重叠但更偏 trick。读 `lang/` 建立体系，读这里查零散技巧。

## 风格与基础技巧

> [!NOTE] **[Competitive C++ Manifesto: A Style Guide](https://codeforces.com/blog/entry/64218)**
>
> 难度：入门 / 风格规范

> [!TIP] **要点**
>
> - 竞赛 C++ 代码风格指南：宏定义、类型缩写、命名习惯
> - 代码可读性、调试便利性的取舍
> - 适合在自己的 template 文件中借鉴的常见简写

* * *

> [!NOTE] **[C++ Tricks](https://codeforces.com/blog/entry/15643)**
>
> 难度：入门-中等

> [!TIP] **要点**
>
> - 常用语言层面的小技巧（`__builtin_*` 系列、`auto`、`decltype`、范围 for 等）
> - 常数优化思路
> - 前置：C++ 基础

> 本站对应：[新版 C++ 特性](../lang/new.md)、[STL 算法](../lang/algorithm.md)

* * *

> [!NOTE] **[C++ tips and tricks](https://codeforces.com/blog/entry/74684)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 上一篇的延伸版，更多实用片段
> - 涵盖位运算技巧、容器初始化与遍历、lambda 应用

* * *

## STL 与高级容器

> [!NOTE] **[C++ STL: map and set](https://codeforces.com/blog/entry/9702)**
>
> 难度：入门

> [!TIP] **要点**
>
> - `map` / `set` / `multimap` / `multiset` 的接口与典型用法
> - 复杂度对照、常见陷阱
> - 前置：C++ 基础

> 本站对应：[STL 算法](../lang/algorithm.md)

* * *

> [!NOTE] **[C++ STL: Policy based data structures](https://codeforces.com/blog/entry/11080)**
>
> 难度：高级

> [!TIP] **要点**
>
> - GCC 扩展的 PBDS（Policy-Based Data Structures）
> - `__gnu_pbds::tree` 实现 order-statistics tree（即「平衡树 + 第 k 大 / 排名」）
> - `__gnu_pbds::gp_hash_table` 替代 `unordered_map` 抗 hack
> - 前置：模板、STL 内核机制

* * *

> [!NOTE] **[About a general reader/writer for STL-Structures](https://codeforces.com/blog/entry/71075)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 用模板/重载为各种 STL 容器写通用 I/O
> - 让调试输出与读入一行写完
> - 适合放进个人 template

* * *

## 防御性编程与 hash 攻防

> [!NOTE] **[Blowing up unordered_map, and how to stop getting hacked](https://codeforces.com/blog/entry/62393)**
>
> 难度：中-高级（必读）

> [!TIP] **要点**
>
> - 解释为何默认 `unordered_map<int,int>` 在 Codeforces 上易被反 hash 数据 hack
> - 给出加随机 seed 的 custom hash 防御写法
> - 是 CF 上「unhacked」基础设置之一
> - 前置：哈希、STL

> 本站对应：[hash 表](../ds/hash.md)、[hash 技巧](../topic/hash.md)

* * *

> [!NOTE] **[Catching silly mistakes with GCC](https://codeforces.com/blog/entry/15547)**
>
> 难度：中等

> [!TIP] **要点**
>
> - 推荐启用的 GCC 警告与 sanitizer：`-Wall -Wextra -Wshadow -fsanitize=address,undefined`
> - 在比赛环境本地复现 UB / 越界 / 未初始化变量
> - 比赛后调试小作弊

* * *

## 性能极限优化

> [!NOTE] **[4-5x Faster Drop-in Replacement for std::lower_bound](https://codeforces.com/blog/entry/75421)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 利用分支预测友好的 branchless 写法替代标准库二分
> - 适用于热路径上数百万次 `lower_bound` 的极限优化
> - 前置：二分查找、CPU 流水线/分支预测概念

> 本站对应：[二分](../basic/binary.md)

* * *

> [!NOTE] **[How to get actual 64 bit bitsets on Codeforces](https://codeforces.com/blog/entry/77480)**
>
> 难度：高级

> [!TIP] **要点**
>
> - CF 评测机原本以 32-bit 模式编译 `std::bitset`，导致大 bitset 性能对半折扣
> - 通过 pragma / 编译选项强制 64-bit 字宽
> - 适合 bitset 优化为生命线的题（例如莫队 + bitset 题）

> 本站对应：[bitset](../lang/bitset.md)、[莫队配合 bitset](../misc/mo-algo-with-bitset.md)
