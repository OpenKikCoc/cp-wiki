# LC 2800 训练计划（12 个月教练版）

> [!NOTE] **本文档定位**
>
> 这是一份针对「**在职高级工程师 + LC 2200–2400 起点 + CF 未深入 + 每周 5–15h 投入**」场景定制的 12 个月 LC 2800 训练计划。删除了 LC 极少出现的高级冷门主题（网络流、FFT、SAM、杜教筛、圆方树等），把时间集中到 LC Q4 实证高频主题。配套约 200 道精选题 + 4 层回顾机制 + 季度 KPI 检查点。

## 总体框架

| 项 | 内容 |
|---|---|
| **主目标** | 12 个月内 LC contest rating 稳定 2700–2800 |
| **次目标** | CF rating 到 2200+（IM 边界）|
| **时间预算** | 5–15h / 周（中位数 10h）|
| **核心方法** | 周表固定 + 月度专题 + 季度审计 + 漏洞 burn-down |
| **总刷题量** | ~380–400 题（专题 ~200 + 补题 ~100 + vp ~80）|

> [!TIP] **目标可行性**
>
> 从 LC 2200–2400 起步、保持 ~10h/周，1 年内：60% 概率到 2600，30% 到 2700，15–20% 到 2800。**建议把 1 年硬指标定为 2700，2800 作 stretch goal**。如果时间预算稳定 15h/周以上，2800 概率显著提高。

## 为什么 LC 不能孤立训练

国内观察：LC 2700–2800 群体几乎都是 CF Master+（2200+）。原因：
1. **LC Q4 题型分布偏窄**——「构造 / DP / 数据结构 / 数学」组合反复出现，单一圈层深度不够；
2. **题量不够**——LC 一年 ~100 场 Q4，CF 一年 80+ 场 round × 4–6 题，CF 高难题量是 LC 的 4–5 倍；
3. **2800 是头部分位博弈**——比你强的人在练 CF，你不练等于把杠杆白送对手。

**结论：**
- LC 2700：纯 LC + 强补题纪律，18 个月可达；
- **LC 2800：CF（或 AtCoder Grand）必练**，不可绕过。

## LC Q4 实证主题分布（决定专题分配）

基于近 100 场 LC weekly + biweekly：

| 主题块 | LC Q4 频率 | 月份分配 |
|---|---|---|
| 思维 / 构造 / ad-hoc | 极高（每 2–3 周一题）| **M2 + M3 部分** |
| DP（线性/区间/树/状压/数位/概率/期望/SOS）| 极高 | **M4 + M5 两月** |
| 数据结构组合（段树多标记 / Fenwick + 二分 / 启发式合并）| 高 | M6 |
| 计数 / 容斥 / 期望线性性 | 高 | M7 上半 |
| 贪心 + 二分 + 拆位贡献 | 高 | M3 |
| 字符串基础（Trie / 0-1 Trie / hash / Z / KMP）| 中 | M9 |
| 图论实战（Dij 变种 / Tarjan SCC / 拓扑+DP / 二分图判定）| 中 | M8 |
| 位运算 + 0-1 Trie | 中 | M3 / M9 |
| 博弈基础（SG / Nim）| 低 | M7 末 |
| **网络流** | **< 1%** | 删除（不入主线）|
| **多项式 / FFT / NTT** | **< 0.5%** | 删除 |
| **数论高级（杜教筛 / Möbius）** | **极低** | 删除 |
| **SAM / PAM / 后缀数组** | **极低** | 删除 |
| **圆方树 / Blossom / 一般图匹配** | **几乎无** | 删除 |

被删的主题不是「不重要」，而是「对 LC 2800 的边际收益太低」。如果学有余力可放在 M12 之后作 bonus tier，不占用主线时间。

## 12 个月路线图（高层）

| 月 | 主题 | 训练目标 |
|---|---|---|
| **M1** | 诊断 + CF 入轨 | 摸清漏洞；CF 到 1700 |
| **M2** | 思维 / 构造 / ad-hoc（深度训练）| LC Q3 速度↑；非算法题不再卡 |
| **M3** | 贪心进阶 + 二分 + 拆位贡献 | 「按位 / 按对 / 按段拆贡献」类题秒懂 |
| **M4** | DP I — 经典模型 | 经典 DP（线性/背包/区间/树形/状压）闭眼出 |
| **M5** | DP II — 进阶（数位 / 概率 / 期望 / SOS / CHT 决策单调）| LC Q4 DP 题不再卡 |
| **M6** | 数据结构组合（段树多标记 / 离散化 / Fenwick + 二分 / 启发式合并）| 复杂结构 30 分钟内写对 |
| **M7** | 数学 + 计数 + 博弈基础 | 计数题不再蒙 |
| **M8** | 图论实战（Dijkstra 变种 / Tarjan SCC / 拓扑+DP / 二分图判定 / 0-1 BFS）| 图论建模快 |
| **M9** | 字符串 + 位运算 | 字符串 + 位运算题闭眼出 |
| **M10** | 综合 + 模考密集 | 跨主题识别；vp 训练 |
| **M11** | 弱点 burn-down | 漏洞修补 |
| **M12** | 临门一脚 | LC 稳冲 2700–2800 |

## 主要赛事的加州本地时间

> [!NOTE] **时区说明**
>
> 本节及周表均以 **加州本地时间（San Jose / PT）** 为准。PT 在 3–11 月为 PDT（UTC-7），11–次年 3 月为 PST（UTC-8）。下表给出 PDT 时间（夏令时），冬令时（PST）所有时间向前 1 小时。
>
> 北京 / Moscow / 东京时间到 CA 的换算：北京 -15h（PDT）/ -16h（PST）；Moscow -10h / -11h；东京 -16h / -17h。

| 赛事 | 周期 | 比赛源时间 | 加州 PDT | 加州 PST | 备注 |
|---|---|---|---|---|---|
| **LC Weekly** | 每周日 | 北京 10:30 周日 | **周六 19:30–21:00** | 周六 18:30–20:00 | 周末黄金时段，必打 |
| **LC Biweekly** | 隔周六 | 北京 22:30 周六 | **周六 07:30–09:00** | 周六 06:30–08:00 | 早晨脑子最清醒；要么硬扛要么 vp |
| **CF Round（晚 Russia 标准时段）** | 多数周末 | Moscow 19:35 | **周六 / 周日 09:35–11:35** | 周六 / 周日 08:35–10:35 | 周末上午最舒服 |
| **CF Educational Round** | 多数工作日 | Moscow 17:35 | **工作日 07:35–09:35** | 工作日 06:35–08:35 | 上班前；起得早可参赛 |
| **CF Round（深夜 Russia）** | 偶尔 | Moscow 22:35 | **周六 / 周日 12:35–14:35** | 周六 / 周日 11:35–13:35 | 午饭后档，质量参差 |
| **AtCoder ABC** | 每周六 | 东京 21:00 | **周六 05:00–06:40** | 周六 04:00–05:40 | 太早，**默认 vp 一次** |
| **AtCoder ARC / AGC** | 不定（多周日）| 东京 21:00 | **周日 05:00–06:40** | 周日 04:00–05:40 | 同上，默认 vp |

> [!TIP] **加州时区的最大利好**
>
> LC Weekly 落在周六晚 primetime（19:30 PDT）+ CF 主流场次落在周末上午（09:35 PDT），**两个核心赛事都在状态最稳的窗口**，比国内同行的「周日早 10:30 + 周六晚 22:30」节奏舒服很多。**别浪费这个时差红利**——LC Weekly 必打，CF 周末场尽量打。

## 周表（baseline 7–8h / 周，加州 PT）

> 在职 senior 真实可用时间 baseline = 7–8h（不是 10h）。10h 是 ceiling，留给状态好或休假周。

### 标准周（7–8h）

| 时段（PDT 示意）| 时长 | 内容 |
|---|---|---|
| 周一 19:30–20:30 | 1h | 周末 LC Weekly Q4 复盘 + L1 微回顾 |
| 周二 / 周四（任一晚）| 1h | 当周专题（2 题）|
| 周六 09:30–11:30 | 2h | CF Round / vp（按质量分级选）|
| 周六 19:30–21:00 | 1.5h | **LC Weekly**（必打）|
| 周日 09:00–10:30 | 1.5h | 赛事补题 + editorial |

合计 **~7h**。专题保 1 晚而非 2 晚，避免「两晚都打不齐」的常见现象。

### 周期化（每月 4 周轮转，避免 burnout）

| 周 | 模式 | 时长 | 关键差异 |
|---|---|---|---|
| 第 1–2 周 | 标准 | 7–8h | 上面那张表 |
| 第 3 周 | **加强** | 10–12h | 加 1 个专题晚 + 1 场 AtCoder vp + 周日下午 vp |
| 第 4 周 | **deload + L2 回顾** | 5h | 只保 LC Weekly + 周一复盘 + 老主题 vp（**不开新内容**）|

每月平均 ≈ 8h/周。**deload 周不是「划水」，是让训练适应固化。** 跳过 deload = 第 5 周开始效率下降。

### 弹性 / 降级

| 状况 | 模式 |
|---|---|
| 工作 crunch / 出差 / 家庭事务 | **最低维持**：只保 LC Weekly + 周一 1h 复盘 = 2.5h |
| 休假 / 周末空闲多 | 切 15h 全力档：加 ABC vp + CF Edu 历史 vp + 1 个专题晚 |

### 比赛锚点（详见上节「时区表」）

- **必打**：LC Weekly（周六 19:30 PDT）
- **优选**：CF Edu / Global / Goodbye（周末早 09:35 PDT）
- **vp**：AtCoder ABC（凌晨档，永远 vp）/ LC Biweekly（早 7:30 PDT，**默认 vp** 而非现场）

### 工作日窗口现实

工作日 1h 窗口对在职 senior 是 OPTIMISTIC。**保 1 晚专题 / 周比保 2 晚更可持续**。如果某周 2 晚都能保住 → 升级到加强周节奏。

> [!TIP] **冬令时（PST）所有时段前移 1 小时**
>
> LC Weekly 18:30 / CF 08:35 / LC Biweekly 06:30。

## 赛事质量分级与选择策略

CF 现役 round 质量参差极大——random Div3 / 弱 setter 的 Div2 经常出现 A-B 太水、C 跳 2000+、weak test 等问题。**有限的训练时间应倾斜到 quality floor 高的源**。

| 优先级 | 训练源 | 质量 floor | 备注 |
|---|---|---|---|
| ★ S | **AtCoder ABC F/G + ARC** | 极高（强制 testing）| 100 分钟短赛制；F ≈ CF 1900–2300，G ≈ CF 2300–2700 |
| ★ S | **CF Educational Round** | 高 | 月 1–2，setter 通常是 ICPC 老兵 + 强 testing |
| ★ S | **CF Global Round / Goodbye / Hello** | 极高 | 年 ~12 场，sponsored，强 tester 阵容；**全部 vp 一遍** |
| ★ A | **CSES Problem Set** | 极高 | https://cses.fi/problemset/ — 300 题按主题分组，无 round 时间约束 |
| ★ A | **ICPC Asia / NEERC / NWERC Gym mirror** | 顶级 | CF Gym 搜 `ICPC` + region 名；M10 模考期主力 |
| ★ B | **灵茶山艾府 LC 分类题单** | 高 | https://leetcode.cn/circle/discuss/RvFUtj/ — LC 训练国内金标准 |
| ◯ C | **CF Catalog** 高 contributor 的部分 | 中高 | https://codeforces.com/catalog — 看 adamant / -is-this-fft- 等 |
| ◯ C | Random CF Div2 | 中下 | 实在没 round 才用；**优先级最低** |
| ✕ | Random CF Div3 | 低 | 不打，对 LC 2800 价值低 |

**原则**：每周 contest 时段优先级从高到低 = LC Weekly（必）→ AtCoder ABC vp → CF Edu / Global / Goodbye → CF Div1+Div2 → 其它。**只有当 S/A 档源已耗尽时，才打 random Div2**。

> [!TIP] **CF round 筛选具体操作**
>
> 在 [CF 比赛列表](https://codeforces.com/contests?contestType=2) 倒序扫，标 `Educational` / `Global` / `Goodbye` / `Hello` / `Div. 1 + Div. 2` 的优先 vp。普通 `Codeforces Round (Div. 2)` 跳过非著名 setter（在 round 页面看 announcement，setter 是 [tier list](https://clist.by) 上 master+ 即可考虑）。

## 训练方法（5 条核心规则）

1. **想 30 分钟硬下限**：不到 30 分钟绝不查 hint。看一眼 editorial 直接练成「想到就查」。
2. **口述 idea 再写代码**：能说清「为什么对」+「复杂度」才动手。这条对工程师特别重要——你写代码飞快，反而容易「先写起来再说」，结果 WA 后回不来。
3. **Editorial > 单纯 AC**：每场 contest 后必读官方 editorial。新 trick 写进 wiki 对应主题 markdown。
4. **每月归因表复盘**：维护表格 4 列：排名 / AC 用时 / fail 归因（知识/思维/实现/心态）/ 新 trick。3 个月后这张表会暴露真正的弱点 pattern——往往跟自我感知偏差很大。
5. **节奏 > 强度**：稳定 8h/周 强于波动 ｛0,15,5,20｝。CP 训练是节奏运动。

## 4 层回顾机制（占总训练时间约 22%）

| 层级 | 周期 | 单次时长 | 内容 |
|---|---|---|---|
| **L1 微回顾** | 每周一晚 | 20–30 min | 重读上周 wiki 笔记 + 1 题旧主题 vp |
| **L2 月度回顾** | 每月最后一周 | 3–4h | 老主题 vp（4 题）+ 模板重写（1 个）|
| **L3 季度审计** | M3 / M6 / M9 末（替换该周专题）| 1 周 | 全回顾 + 漏洞修补 + 模考；不开新主题 |
| **L4 模板热重载** | 每 2 个月（M4/M6/M8/M10）| 30 min | 限时手敲最久没用的模板 |

> [!NOTE] **wiki 反向索引选择回顾主题**
>
> 每月 L2 回顾日，从仓库根目录跑下面命令找出最久没动的 5 个主题文件，作为本次 vp 主题（dashboard 也会自动展示这一节）：
>
> ```sh
> find {dp,ds,graph,string,math,topic} -name '*.md' -type f \
>   -exec stat -f '%m %N' {} \; \
>   | sort -n | head -10
> ```
>
> 让数据替你选要复习什么，比凭感觉准。

> [!TIP] **回顾 ≠ 整理笔记**
>
> 工程师容易陷入：把「回顾」做成「整理 wiki / 补排版 / 归类」。这些对训练价值接近 0。**回顾的本质是「在限时压力下独立调用旧知识」**。所以：
> - L1 微回顾 = 1 题 vp
> - L2 月度回顾 = 4 题 vp + 1 模板重写
> - L3 季度审计 = 模考 + 漏洞修补
> - L4 模板重载 = 限时手写

## 验证 KPI（每季度对照）

| 时点 | LC rating 目标 | CF rating 目标 | 不达标降级动作 |
|---|---|---|---|
| **M3 末** | 不掉（保 ≥ 2200）| 1700–1800 | 检查周表执行率；可能是时间不够 |
| **M6 末** | 2300–2400 | 1900–2000 | 不达标 → 把 M7–M9 图论/字符串压缩，给 M2 思维专题加一轮 |
| **M9 末** | 2500–2600 | 2000–2100 | 不达标 → 放弃 2800 期望，目标改为 M12 稳 2700 |
| **M12 末** | **2700–2800** | 2200+ | 主目标 |

## Plateau / Regression / Burnout Playbook

KPI 表只覆盖「季度末没达标」。日常更常见的 3 种状况各有明确 playbook：

| 状况 | 触发条件 | 行动 |
|---|---|---|
| **横盘** | 4 周 rating 浮动 < 30，无升势 | 不是知识不够，是速度 / 心态。砍专题日，加 1 场 vp / 周；强制每场赛后归因表填「fail 类型」即使全 AC（写「无 fail，但 Q4 用时 35 分」也算）|
| **回退** | rating 从近 8 周峰值跌 ≥ 100 | 暂停目标。当周切「最低维持」（2.5h）。下周用 L2 回顾日做归因表 4 周扫描，找 fail tag 集中点，下个月专题切到该 tag |
| **Burnout** | 连续 3 周没碰任何 contest | 不是计划问题，是 life 问题。**不要罪疚式补课**。直接降到「最低维持」2 周，仅打 LC Weekly。期间评估：是 plan 太重还是 life 临时挤压？前者 → 永久切 baseline 到 5h/周 + 12 个月延到 18 个月。后者 → 等 life 通过后回标准周 |

> [!TIP] **核心原则**
>
> 三种状况的共同应对 = **降强度 + 加诊断**，不是「咬牙加量」。Plateau / regression 90% 是训练适应到达 local optimum，不是不够努力——加量只会更快 burnout。

---

## 详细题单

每月给出**具体题号 + 链接 + 训练目的**。LC 题号可在 `leetcode.cn/problems/{slug}` 直接打开；CF 在 `codeforces.com/problemset/problem/{round}/{letter}`；AtCoder 在 `atcoder.jp/contests/{contest}/tasks/{contest}_{letter}`。

### M1（第 1–4 周）— 诊断 + CF 入轨

不做专题题单。整月动作：
- **冷打 5 场 LC weekly + 2 场 biweekly**（每场后归因表写入）
- **冷打 3 场 CF Div3 / Edu Round**（注册 handle，目标 CF 1700）
- **第 4 周（L2 月度回顾日）**：复盘归因表，确定个人弱点排序 → 用此排序微调 M2–M3 的专题次序

### M2（第 5–8 周）— 思维 / 构造 / ad-hoc（深度训练）

**LC 历史 Q4 思维题（必做 12 题）：**

- [ ] [2071. 你可以安排的最多任务数目](https://leetcode.cn/problems/maximum-number-of-tasks-you-can-assign/) — 二分 + 双端队列贪心
- [ ] [1851. 包含每个查询的最小区间](https://leetcode.cn/problems/minimum-interval-to-include-each-query/) — 离线 + 优先队列
- [ ] [1665. 完成所有任务的最少初始能量](https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/) — exchange argument 排序贪心
- [ ] [1383. 最大的团队表现值](https://leetcode.cn/problems/maximum-performance-of-a-team/) — 排序 + 优先队列
- [ ] [2589. 完成所有任务的最少时间](https://leetcode.cn/problems/minimum-time-to-complete-all-tasks/) — 区间贪心
- [ ] [2503. 矩阵查询可获得的最大分数](https://leetcode.cn/problems/maximum-number-of-points-from-grid-queries/) — 离线 + 优先队列 + 并查集
- [ ] [1840. 最高建筑高度](https://leetcode.cn/problems/maximum-building-height/) — 思维构造
- [ ] [2127. 参加会议的最多员工数](https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/) — 基环树思维
- [ ] [1799. N 次操作后的最大分数和](https://leetcode.cn/problems/maximize-score-after-n-operations/) — 状压 + GCD 思维
- [ ] [2218. 从栈中取出 K 个硬币的最大面值和](https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/) — 分组背包思维
- [ ] [2603. 收集树中金币](https://leetcode.cn/problems/collect-coins-in-a-tree/) — 拓扑式思维剪枝
- [ ] [2812. 找出最安全路径](https://leetcode.cn/problems/find-the-safest-path-in-a-grid/) — 多源 BFS + 二分

**CF 1700–2000 构造（必做 6 题）：**

按 [Codeforces problemset filter](https://codeforces.com/problemset?tags=constructive+algorithms) 取 rating 1700–2000，按 solved 数倒序选前 6 题。重点关注题目特征：
- 「输出任意一个满足条件的方案」类型
- 「证明存在一种构造使得...」
- 「经典 trick：极端情况 / 二分性 / 分块构造」

**训练目的**：把「读题 → 找极端 / 不变量 / 对称性 → 构造」这一链条压到 ≤ 15 分钟。

### M3（第 9–12 周）— 贪心进阶 + 二分 + 拆位贡献

**拆位 / 贡献法 LC（必做 8 题）：**

- [ ] [898. 子数组按位或操作](https://leetcode.cn/problems/bitwise-ors-of-subarrays/) — 拆位 OR 增长性
- [ ] [2871. 将数组分割成最多数目的子数组](https://leetcode.cn/problems/split-array-into-maximum-number-of-subarrays/) — 拆位 AND 贪心
- [ ] [2680. 最大或值](https://leetcode.cn/problems/maximum-or/) — 拆位思维 + 前后缀
- [ ] [2932. 找出强数对的最大异或值 I](https://leetcode.cn/problems/maximum-strong-pair-xor-i/) — 0-1 Trie 入门
- [ ] [421. 数组中两个数的最大异或值](https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/) — 0-1 Trie 经典
- [ ] [1707. 与数组中元素的最大异或值](https://leetcode.cn/problems/maximum-xor-with-an-element-from-array/) — 离线 0-1 Trie
- [ ] [2935. 找出强数对的最大异或值 II](https://leetcode.cn/problems/maximum-strong-pair-xor-ii/) — 0-1 Trie + 滑窗
- [ ] [1601. 最多可达成的换楼请求数目](https://leetcode.cn/problems/maximum-number-of-achievable-transfer-requests/) — 状压枚举 / 拆位

**二分答案 LC（必做 6 题）：**

- [ ] [1011. 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/) — 二分答案模板
- [ ] [2528. 最大化城市的最小电量](https://leetcode.cn/problems/maximize-the-minimum-powered-city/) — 二分 + 差分
- [ ] [2517. 礼盒的最大甜蜜度](https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/) — 二分 + 贪心 check
- [ ] [2616. 最小化数对的最大差值](https://leetcode.cn/problems/minimize-the-maximum-difference-of-pairs/) — 二分 + 贪心
- [ ] [2513. 最小化两个数组中的最大值](https://leetcode.cn/problems/minimize-the-maximum-of-two-arrays/) — 二分 + 容斥
- [ ] [2861. 最大合金数](https://leetcode.cn/problems/maximum-number-of-alloys/) — 多组二分

**反悔贪心 LC（必做 4 题）：**

- [ ] [630. 课程表 III](https://leetcode.cn/problems/course-schedule-iii/) — 反悔贪心模板
- [ ] [871. 最低加油次数](https://leetcode.cn/problems/minimum-number-of-refueling-stops/) — 反悔贪心
- [ ] [1642. 可以到达的最远建筑](https://leetcode.cn/problems/furthest-building-you-can-reach/) — 反悔贪心
- [ ] [LCP 30. 魔塔游戏](https://leetcode.cn/problems/p0NxJO/) — 反悔贪心

### M4（第 13–16 周）— DP I 经典模型

**AtCoder DP Contest（必做 13 题，A–P 段）：**

| # | 题名 | 主题 |
|---|---|---|
| A | [Frog 1](https://atcoder.jp/contests/dp/tasks/dp_a) | 线性 DP 入门 |
| B | [Frog 2](https://atcoder.jp/contests/dp/tasks/dp_b) | 滑窗 DP |
| C | [Vacation](https://atcoder.jp/contests/dp/tasks/dp_c) | 状态机 DP |
| D | [Knapsack 1](https://atcoder.jp/contests/dp/tasks/dp_d) | 0-1 背包 |
| E | [Knapsack 2](https://atcoder.jp/contests/dp/tasks/dp_e) | 0-1 背包（按价值反转）|
| F | [LCS](https://atcoder.jp/contests/dp/tasks/dp_f) | 最长公共子序列 |
| G | [Longest Path](https://atcoder.jp/contests/dp/tasks/dp_g) | DAG 上 DP |
| H | [Grid 1](https://atcoder.jp/contests/dp/tasks/dp_h) | 二维路径计数 |
| L | [Deque](https://atcoder.jp/contests/dp/tasks/dp_l) | 区间博弈 DP |
| M | [Candies](https://atcoder.jp/contests/dp/tasks/dp_m) | 多重背包 / 前缀和 DP |
| N | [Slimes](https://atcoder.jp/contests/dp/tasks/dp_n) | 区间 DP（石子合并）|
| O | [Matching](https://atcoder.jp/contests/dp/tasks/dp_o) | 状压 DP（二分图匹配计数）|
| P | [Independent Set](https://atcoder.jp/contests/dp/tasks/dp_p) | 树形 DP |

**LC 经典 DP（必做 8 题）：**

- [ ] [1483. 树节点的第 K 个祖先](https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/) — 倍增 LCA + DP
- [ ] [1335. 工作计划的最低难度](https://leetcode.cn/problems/minimum-difficulty-of-a-job-schedule/) — 区间分割 DP
- [ ] [1473. 粉刷房子 III](https://leetcode.cn/problems/paint-house-iii/) — 多维状态机 DP
- [ ] [1547. 切棍子的最小成本](https://leetcode.cn/problems/minimum-cost-to-cut-a-stick/) — 区间 DP
- [ ] [1690. 石子游戏 VII](https://leetcode.cn/problems/stone-game-vii/) — 区间 DP（博弈）
- [ ] [1745. 分割回文串 IV](https://leetcode.cn/problems/palindrome-partitioning-iv/) — 双 DP（回文 + 分割）
- [ ] [2547. 拆分数组的最小代价](https://leetcode.cn/problems/minimum-cost-to-split-an-array/) — 分段 DP
- [ ] [2167. 移除所有载有违禁货物车厢所需的最少时间](https://leetcode.cn/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/) — 前后缀 DP

### M5（第 17–20 周）— DP II 进阶（数位 / 概率 / 期望 / SOS / CHT）

**AtCoder DP Contest 进阶（必做 10 题，I–Z 选段）：**

| # | 题名 | 主题 |
|---|---|---|
| I | [Coins](https://atcoder.jp/contests/dp/tasks/dp_i) | 概率 DP |
| J | [Sushi](https://atcoder.jp/contests/dp/tasks/dp_j) | 期望 DP |
| K | [Stones](https://atcoder.jp/contests/dp/tasks/dp_k) | 博弈 DP |
| Q | [Flowers](https://atcoder.jp/contests/dp/tasks/dp_q) | DP + 段树 |
| R | [Walk](https://atcoder.jp/contests/dp/tasks/dp_r) | 矩阵快速幂 |
| S | [Digit Sum](https://atcoder.jp/contests/dp/tasks/dp_s) | 数位 DP |
| T | [Permutation](https://atcoder.jp/contests/dp/tasks/dp_t) | 排列计数 DP |
| U | [Grouping](https://atcoder.jp/contests/dp/tasks/dp_u) | 子集枚举 DP |
| V | [Subtree](https://atcoder.jp/contests/dp/tasks/dp_v) | 树形换根 DP |
| Z | [Frog 3](https://atcoder.jp/contests/dp/tasks/dp_z) | 斜率优化（CHT）|

**LC 数位 / 状压 / 期望 DP（必做 10 题）：**

- [ ] [1799. N 次操作后的最大分数和](https://leetcode.cn/problems/maximize-score-after-n-operations/) — 状压 DP + GCD
- [ ] [1986. 完成任务的最少工作时间段](https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/) — 状压 DP
- [ ] [1879. 两个数组最小的异或值之和](https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/) — 状压 DP（二分图匹配）
- [ ] [2741. 特别的排列](https://leetcode.cn/problems/special-permutations/) — 状压 DP
- [ ] [1067. 范围内的数字计数](https://leetcode.cn/problems/digit-count-in-range/) — 数位 DP
- [ ] [600. 不含连续 1 的非负整数](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/) — 数位 DP
- [ ] [902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/) — 数位 DP
- [ ] [1289. 下降路径最小和 II](https://leetcode.cn/problems/minimum-falling-path-sum-ii/) — 拆维优化
- [ ] [2538. 最大价值和与最小价值和的差值](https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/) — 树形换根 DP
- [ ] [1227. 飞机座位分配概率](https://leetcode.cn/problems/airplane-seat-assignment-probability/) — 期望思维

**SOS DP / 子集枚举（必做 3 题）：**

- [ ] [1255. 得分最高的单词集合](https://leetcode.cn/problems/maximum-score-words-formed-by-letters/) — 状压子集枚举
- [ ] [2002. 两个回文子序列长度的最大乘积](https://leetcode.cn/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/) — 状压 + 回文
- [ ] [1494. 安排考试日期](https://leetcode.cn/problems/parallel-courses-ii/) — 状压子集枚举

**CSES DP 进阶（必做 5 题）：**

- [ ] [Counting Numbers](https://cses.fi/problemset/task/2220) — 数位 DP 模板
- [ ] [Two Sets II](https://cses.fi/problemset/task/1093) — 子集 DP 计数
- [ ] [Counting Tilings](https://cses.fi/problemset/task/2181) — 插头 / 状压 DP
- [ ] [Projects](https://cses.fi/problemset/task/1140) — DP + 二分
- [ ] [Elevator Rides](https://cses.fi/problemset/task/1653) — 状压 DP（背包变种）

### M6（第 21–24 周）— 数据结构组合

**段树 LC（必做 8 题）：**

- [ ] [307. 区域和检索 - 数组可修改](https://leetcode.cn/problems/range-sum-query-mutable/) — 段树/BIT 模板
- [ ] [732. 我的日程安排表 III](https://leetcode.cn/problems/my-calendar-iii/) — 动态开点段树
- [ ] [2407. 最长递增子序列 II](https://leetcode.cn/problems/longest-increasing-subsequence-ii/) — 段树 + DP
- [ ] [2569. 更新数组后处理求和查询](https://leetcode.cn/problems/handling-sum-queries-after-update/) — 段树 lazy
- [ ] [2926. 平衡子序列的最大和](https://leetcode.cn/problems/maximum-balanced-subsequence-sum/) — 段树 + DP
- [ ] [2940. 找到 Alice 和 Bob 可以相遇的建筑](https://leetcode.cn/problems/find-building-where-alice-and-bob-can-meet/) — 段树二分
- [ ] [2945. 找到最大非递减数组的长度](https://leetcode.cn/problems/find-maximum-non-decreasing-array-length/) — 段树 + DP
- [ ] [3187. 数组中的峰值](https://leetcode.cn/problems/peaks-in-array/) — 段树点修改

**Fenwick / 离散化 / 启发式合并 LC（必做 6 题）：**

- [ ] [315. 计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/) — 离散化 + BIT
- [ ] [1505. 最多 K 次交换相邻数位后得到的最小整数](https://leetcode.cn/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/) — BIT 贪心
- [ ] [2659. 将数组清空](https://leetcode.cn/problems/make-array-empty/) — BIT 模拟
- [ ] [3072. 将元素分配到两个数组中 II](https://leetcode.cn/problems/distribute-elements-into-two-arrays-ii/) — BIT
- [ ] [2421. 好路径的数目](https://leetcode.cn/problems/number-of-good-paths/) — 启发式合并 / 离线并查集
- [ ] [2092. 找出知晓秘密的所有专家](https://leetcode.cn/problems/find-all-people-with-secret/) — 并查集 + 离散事件

**单调栈 / 单调队列 LC（必做 4 题）：**

- [ ] [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/) — 单调栈
- [ ] [1944. 队列中可以看到的人数](https://leetcode.cn/problems/number-of-visible-people-in-a-queue/) — 单调栈
- [ ] [2289. 使数组按非递减顺序排列](https://leetcode.cn/problems/steps-to-make-array-non-decreasing/) — 单调栈
- [ ] [2818. 操作使得分最大](https://leetcode.cn/problems/apply-operations-to-maximize-score/) — 单调栈 + 贪心

**CSES Range Queries（必做 5 题）：**

- [ ] [Hotel Queries](https://cses.fi/problemset/task/1143) — 段树二分
- [ ] [List Removals](https://cses.fi/problemset/task/1749) — BIT + 二分（动态第 k 大）
- [ ] [Salary Queries](https://cses.fi/problemset/task/1144) — 离散化 + BIT / 段树
- [ ] [Subarray Sum Queries](https://cses.fi/problemset/task/1190) — 段树维护最大子段和
- [ ] [Range Updates and Sums](https://cses.fi/problemset/task/1735) — 段树双 lazy（区间加 + 区间赋值）

### M7（第 25–28 周）— 数学 + 计数 + 博弈基础

**组合 / 容斥 LC（必做 8 题）：**

- [ ] [62. 不同路径](https://leetcode.cn/problems/unique-paths/) — 组合数模板
- [ ] [1359. 有效的快递序列数目](https://leetcode.cn/problems/count-all-valid-pickup-and-delivery-options/) — 组合推导
- [ ] [1735. 生成乘积数组的方案数](https://leetcode.cn/problems/count-ways-to-make-array-with-product/) — 质因数分解 + 隔板
- [ ] [2514. 统计同位异构字符串数目](https://leetcode.cn/problems/count-anagrams/) — 多重集排列
- [ ] [2147. 分隔长廊的方案数](https://leetcode.cn/problems/number-of-ways-to-divide-a-long-corridor/) — 组合计数
- [ ] [2930. 重新排列后包含指定子字符串的字符串数目](https://leetcode.cn/problems/number-of-strings-which-can-be-rearranged-to-contain-substring/) — 容斥
- [ ] [3007. 价值和小于等于 K 的最大数字](https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/) — 数位 + 二分
- [ ] [1922. 统计好数字的数目](https://leetcode.cn/problems/count-good-numbers/) — 快速幂 + 计数

**期望 / 概率 LC（必做 3 题）：**

- [ ] [837. 新 21 点](https://leetcode.cn/problems/new-21-game/) — 概率 DP / 滑窗
- [ ] [688. 骑士在棋盘上的概率](https://leetcode.cn/problems/knight-probability-in-chessboard/) — 概率 DP
- [ ] [808. 分汤](https://leetcode.cn/problems/soup-servings/) — 期望 DP

**博弈 SG / Nim（必做 5 题）：**

- [ ] [292. Nim 游戏](https://leetcode.cn/problems/nim-game/) — Nim 入门
- [ ] [464. 我能赢吗](https://leetcode.cn/problems/can-i-win/) — 状压博弈
- [ ] [486. 预测赢家](https://leetcode.cn/problems/predict-the-winner/) — 区间博弈 DP
- [ ] [877. 石子游戏](https://leetcode.cn/problems/stone-game/) — 博弈 DP
- [ ] [1690. 石子游戏 VII](https://leetcode.cn/problems/stone-game-vii/) — 博弈 DP

**基础数论速览（必做 3 题）：**

- [ ] [29. 两数相除](https://leetcode.cn/problems/divide-two-integers/) — 位运算除法
- [ ] [1622. 奇妙序列](https://leetcode.cn/problems/fancy-sequence/) — 模逆元
- [ ] [2607. 使子数组元素和相等](https://leetcode.cn/problems/make-k-subarray-sums-equal/) — GCD + 环上分组

### M8（第 29–32 周）— 图论实战

**Dijkstra 变种（必做 6 题）：**

- [ ] [743. 网络延迟时间](https://leetcode.cn/problems/network-delay-time/) — Dijkstra 模板
- [ ] [787. K 站中转内最便宜的航班](https://leetcode.cn/problems/cheapest-flights-within-k-stops/) — Bellman-Ford / 分层
- [ ] [1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/) — Dijkstra 变种（最小化最大）
- [ ] [1976. 到达目的地的方案数](https://leetcode.cn/problems/number-of-ways-to-arrive-at-destination/) — Dijkstra + 计数
- [ ] [2203. 得到要求路径的最小带权子图](https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/) — 三次 Dijkstra
- [ ] [2577. 在网格图中访问一个格子的最少时间](https://leetcode.cn/problems/minimum-time-to-visit-a-cell-in-a-grid/) — Dijkstra 思维

**Tarjan / 拓扑 / 二分图 LC（必做 6 题）：**

- [ ] [1192. 查找集群内的关键连接](https://leetcode.cn/problems/critical-connections-in-a-network/) — Tarjan 求桥
- [ ] [1559. 二维网格图中探测环](https://leetcode.cn/problems/detect-cycles-in-2d-grid/) — 并查集 / DFS 找环
- [ ] [2127. 参加会议的最多员工数](https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/) — 基环树
- [ ] [2876. 有向图访问计数](https://leetcode.cn/problems/count-visited-nodes-in-a-directed-graph/) — 基环树 + DFS
- [ ] [2493. 将节点分成尽可能多的组](https://leetcode.cn/problems/divide-nodes-into-the-maximum-number-of-groups/) — BFS 染色 + 二分图判定
- [ ] [785. 判断二分图](https://leetcode.cn/problems/is-graph-bipartite/) — DFS / 并查集

**树论 LC（必做 6 题）：**

- [ ] [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/) — 树上换根思维
- [ ] [1483. 树节点的第 K 个祖先](https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/) — 倍增 LCA
- [ ] [2846. 边权重均等查询](https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree/) — LCA + 数组
- [ ] [2867. 统计树中的合法路径数目](https://leetcode.cn/problems/count-valid-paths-in-a-tree/) — 树形 + 组合
- [ ] [2646. 最小化旅行的价格总和](https://leetcode.cn/problems/minimize-the-total-price-of-the-trips/) — LCA + 树形 DP
- [ ] [2920. 收集所有金币可获得的最大积分](https://leetcode.cn/problems/maximum-points-after-collecting-coins-from-all-nodes/) — 树形 DP + 状态

**0-1 BFS / 多源 BFS（必做 3 题）：**

- [ ] [2290. 到达角落需要移除障碍物的最小数目](https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/) — 0-1 BFS
- [ ] [1162. 地图分析](https://leetcode.cn/problems/as-far-from-land-as-possible/) — 多源 BFS
- [ ] [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/) — 多源 BFS

**CSES Tree Algorithms（必做 5 题）：**

- [ ] [Subordinates](https://cses.fi/problemset/task/1674) — DFS 计数子树
- [ ] [Tree Distances I](https://cses.fi/problemset/task/1132) — 树形 DP（最远点）
- [ ] [Tree Distances II](https://cses.fi/problemset/task/1133) — 换根 DP
- [ ] [Company Queries II](https://cses.fi/problemset/task/1688) — LCA 倍增模板
- [ ] [Distinct Colors](https://cses.fi/problemset/task/1139) — 启发式合并 / 树上 set 合并

### M9（第 33–36 周）— 字符串 + 位运算

**字符串匹配（必做 6 题）：**

- [ ] [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/) — KMP 模板
- [ ] [3008. 找出数组中的美丽下标 II](https://leetcode.cn/problems/find-beautiful-indices-in-the-given-array-ii/) — Z 函数 / KMP
- [ ] [3036. 匹配模式数组的子数组数目 II](https://leetcode.cn/problems/number-of-subarrays-that-match-a-pattern-ii/) — KMP / Z 函数
- [ ] [796. 旋转字符串](https://leetcode.cn/problems/rotate-string/) — 字符串拼接 + KMP
- [ ] [214. 最短回文串](https://leetcode.cn/problems/shortest-palindrome/) — KMP / 哈希
- [ ] [1392. 最长快乐前缀](https://leetcode.cn/problems/longest-happy-prefix/) — KMP failure 数组

**0-1 Trie 异或题（必做 5 题）：**

- [ ] [421. 数组中两个数的最大异或值](https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/) — 0-1 Trie 经典
- [ ] [1707. 与数组中元素的最大异或值](https://leetcode.cn/problems/maximum-xor-with-an-element-from-array/) — 离线 Trie
- [ ] [1803. 统计异或值在范围内的数对有多少](https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/) — Trie 计数
- [ ] [1938. 查询最大基因差](https://leetcode.cn/problems/maximum-genetic-difference-query/) — 离线 Trie + DFS
- [ ] [2935. 找出强数对的最大异或值 II](https://leetcode.cn/problems/maximum-strong-pair-xor-ii/) — Trie + 滑窗

**字符串哈希（必做 4 题）：**

- [ ] [1044. 最长重复子串](https://leetcode.cn/problems/longest-duplicate-substring/) — 二分 + 哈希
- [ ] [1316. 不同的循环子字符串](https://leetcode.cn/problems/distinct-echo-substrings/) — 字符串哈希
- [ ] [2156. 查找给定哈希值的子串](https://leetcode.cn/problems/find-substring-with-given-hash-value/) — 滑动哈希
- [ ] [3213. 最小代价构造字符串](https://leetcode.cn/problems/construct-string-with-minimum-cost/) — Trie / KMP / Z 函数 + DP

**位运算技巧（必做 5 题）：**

- [ ] [137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/) — 位运算 + 状态机
- [ ] [201. 数字范围按位与](https://leetcode.cn/problems/bitwise-and-of-numbers-range/) — 位运算思维
- [ ] [338. 比特位计数](https://leetcode.cn/problems/counting-bits/) — DP + 位运算
- [ ] [2354. 优质数对的数目](https://leetcode.cn/problems/number-of-excellent-pairs/) — popcount + 计数
- [ ] [2939. 最大异或乘积](https://leetcode.cn/problems/maximum-xor-product/) — 位贪心

**CSES String（必做 5 题）：**

- [ ] [String Matching](https://cses.fi/problemset/task/1753) — KMP / Z 函数模板
- [ ] [Finding Borders](https://cses.fi/problemset/task/1732) — KMP failure 数组
- [ ] [Finding Periods](https://cses.fi/problemset/task/1733) — KMP / Z 函数应用
- [ ] [Minimal Rotation](https://cses.fi/problemset/task/1110) — Booth / 最小表示法
- [ ] [Required Substrings](https://cses.fi/problemset/task/1112) — 字符串 hash + 状压

### M10（第 37–40 周）— 综合训练 + 模考密集

**不开新主题。** 全部时间用于：
- **每周 3 场 vp**（按本节质量分层选源）
- **每场 vp 后 24h 内 100% 补题**（不允许带题入下周）
- 用归因表统计：vp 中识别错的主题 → 写进「漏洞清单」给 M11

**vp 选择策略（按优先级 S → C 排，选满 3 场为止）：**

| 优先级 | 源 | 选取规则 |
|---|---|---|
| ★ S | **AtCoder ABC F/G + ARC C/D**（vp）| 按 difficulty 倒序刷未做过的；https://kenkoooo.com/atcoder/ 按 difficulty 筛选 |
| ★ S | **CF Educational Round** 历史档 | 全部 Edu Round 100+ 倒序 vp，跳过你已 AC ≥ 5 题的场次 |
| ★ S | **CF Global / Goodbye / Hello** 历史 | 一年 ~12 场，**全部 vp 一遍** 才进入下个优先级 |
| ★ A | **LC 历史 weekly Q3+Q4** | 从你 wiki 未记录的场次倒推（Q4 中等以上）|
| ★ A | **ICPC Asia East / NEERC / Petrozavodsk Camp Gym** | CF Gym 搜 region 名；选 single-author setter 强的 mirror |
| ◯ B | **CF Div1+Div2** 历史 | 选 5 ≤ 题数 ≤ 8 的 round，rating bracket 1900–2400 |
| ✕ | random Div2 / Div3 | 不选；除非 S/A/B 全部用完 |

**目标节奏**：每周 3 场 vp = 至少 2 场 S 档 + 1 场 A 档。完全不打 random Div3。

### M11（第 41–44 周）— 弱点 burn-down

按 M10 末统计的漏洞清单**逐主题清算**。每个弱点主题：
1. 重读你 wiki 对应文件 + 重写最关键 1 个模板
2. 从该主题专门刷 10 题（先做你过去 fail 过的题，再做新题）
3. 限时 vp 该主题专项题（CF problemset filter by tag + rating + 你没做过）

题源：
- **CF problemset filter**：`https://codeforces.com/problemset?tags={tag}`
- **LC tag filter**：`leetcode.cn/tag/{tag}/`
- **灵茶分类题单**：在 leetcode.cn 用户搜索「灵茶山艾府」

### M12（第 45–48 周）— 临门一脚

- **前 2 周（45–46）**：每周 4 场 vp（密度拉到极限），不学新东西。
- **后 2 周（47–48）**：降低强度（每周 2 场 vp + 多睡），保持手感。

> [!TIP] **比赛前一天**
>
> 不刷题。只重读 wiki 的「易错点 / trick」节选。强行刷题会让大脑进入「过度处理」状态，正赛反而下降。

---

## 题量汇总（年度）

| 月 | 专题题量 | 累计 |
|---|---|---|
| M1 | 0（contests only）| 0 |
| M2 | 18 | 18 |
| M3 | 18 | 36 |
| M4 | 21 | 57 |
| M5 | 23 | 80 |
| M6 | 18 | 98 |
| M7 | 19 | 117 |
| M8 | 21 | 138 |
| M9 | 20 | 158 |
| M10 | 0（vp only，约 12 场 vp ≈ 36 题）| 158 |
| M11 | ~30（漏洞清算）| ~188 |
| M12 | ~10（强化）| ~198 |

**+ contest 补题约 100 题（LC 周赛 ≈ 50 + CF rounds ≈ 50）**  
**+ vp 中的题约 80 题**  
**= 全年总刷题量约 380–400 题**

与每周 6h 实操时间（专题 4h + 补题 2h）匹配，平均每题 ~30–40 分钟。

## 与本 wiki 的协同

每场 contest / 每场 vp / 每读一篇 editorial → **新 trick 必须立即写入 wiki 对应主题的 markdown 文件**。一年下来这是你最大的复利资产，远比刷过的题号有用。具体做法：

- 题号、链接、思路一句话、AC 代码
- **一句话「我没第一时间想到的 key insight」**（最关键的一行）

跨链：
- [Codeforces 教程索引](../codeforces/README.md) — 4-tier 学习路径中的 Tier 1–3 列表对应本计划 M1–M9 的 readings
- 各主题 wiki 章节（[dp/](../dp/README.md)、[ds/](../ds/README.md)、[graph/](../graph/README.md)、[string/](../string/README.md)、[math/](../math/README.md)、[topic/](../topic/README.md)）作为复习索引和 wiki 写入靶点

## 一个最重要的提醒

**LC 2800 不是「比 LC 2400 选手强 400 分」，而是「比 LC 2700 选手稳定再快 5 分钟」的概率结果。** 从 2400 到 2800，**80% 的提升不在新知识，而在执行力 + 比赛心态 + 速度**。

所以：

- 别再囤新主题了（你已经囤够了）
- 比赛里出错的根因 70% 是「读题马虎 / 边界 / WA 后慌」，**不是不会**
- 解决这些靠的是「严格自律的训练流程」，不是更多题
