# 字符串处理 (Codeforces 教程索引)

原帖第 5 节，5 篇教程（含 1 篇 Quora 外站 Trie 教程）。聚焦字典树、字符串匹配的 Z 函数、KMP 失配函数、后缀自动机、Manacher。

> [!NOTE] **本节定位**
>
> 原帖只列了 4 篇，是覆盖最薄的一节。但本 wiki 的 `string/` 章节是覆盖最全的章节之一（KMP / Z / AC 自动机 / Trie / SA / SAM / 广义 SAM / SAM-on-tree / PAM / 后缀树 / Manacher / Lyndon / Main-Lorentz / 序列自动机 / 最小表示），把这 4 篇当作字符串学习的「锚点」即可。

## 字典树 / 入门

> [!NOTE] **[Tutorial on Trie and example problems](https://www.quora.com/q/threadsiiithyderabad/Tutorial-on-Trie-and-example-problems)**
>
> 难度：入门-中等 / 外站资源（Quora）

> [!TIP] **要点**
>
> - 字典树（Trie）的概念、构造、查询
> - 配套例题：前缀计数、最长公共前缀、异或最大值（01-Trie）
> - 是 IIIT Hyderabad 算法社区的入门讲解
> - 前置：基础数据结构

> 本站对应：[字典树 (Trie)](../string/trie.md)、[可持久化 Trie](../ds/persistent-trie.md)

* * *

## 后缀 / 前缀函数族

> [!NOTE] **[Z algorithm](https://codeforces.com/blog/entry/3107)**
>
> 难度：中等

> [!TIP] **要点**
>
> - Z 函数：$z[i]$ 为 $s[i..]$ 与 $s$ 的最长公共前缀
> - 线性时间求解，能处理「模式串匹配 / 周期 / 公共前缀」类问题
> - 与 KMP 失配函数相辅相成
> - 前置：字符串基础

> 本站对应：[Z 函数（扩展 KMP）](../string/z-func.md)

* * *

> [!NOTE] **[Transition between Z- and prefix functions](https://codeforces.com/blog/entry/9612)**
>
> 难度：高级

> [!TIP] **要点**
>
> - Z 函数 ↔ KMP 失配函数（前缀函数）的相互转换
> - 帮助理解二者的代数等价性
> - 前置：Z、KMP

> 本站对应：[KMP 算法](../string/kmp.md)、[Z 函数](../string/z-func.md)

* * *

## 自动机

> [!NOTE] **[Suffix Automata](https://codeforces.com/blog/entry/20861)**
>
> 难度：高级

> [!TIP] **要点**
>
> - 后缀自动机 SAM：以最少状态识别字符串所有后缀的 DFA
> - $O(N)$ 构造，处理子串计数 / 最长公共子串等
> - 前置：自动机基础、字典树

> 本站对应：[后缀自动机 (SAM)](../string/sam.md)、[自动机](../string/automaton.md)、[广义后缀自动机](../string/general-sam.md)

* * *

## 回文

> [!NOTE] **[Manacher's algorithm and code readability](https://codeforces.com/blog/entry/12143)**
>
> 难度：中-高级

> [!TIP] **要点**
>
> - Manacher：$O(N)$ 求字符串内所有回文子串
> - 一篇关于「写出可读 Manacher 代码」的反思博客
> - 前置：字符串、双指针

> 本站对应：[Manacher](../string/manacher.md)、[回文树](../string/pam.md)
