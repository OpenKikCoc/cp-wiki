# CONTEXT — 给未来 Claude 会话的接续指引

> [!NOTE] **怎么用这个文档**
>
> 你（未来的 Claude 实例）来到这个仓库帮用户追踪训练时，**先读这个文件，再看 `dashboard.md`**。这里保留的是不会出现在 git 历史里、但又决定接下来怎么做决策的「软上下文」——用户 profile、设计决策的 why、已经达成共识的 conventions、容易被新会话反射性踩坑的雷区。

最后更新：**2026-04-25**（系统初次落地日）。

## 1. 用户 profile（校准你的所有建议）

| 维度 | 状态 | 含义 |
|---|---|---|
| 职业 | 在职 5+ 年高级工程师 | 工程能力强，时间碎片化；周中只有 3 个 1h 窗口 |
| LC contest rating | 起点 **2200–2400** | Guardian 中段；离 2800 还有 400–600 分 |
| CF rating | 起点 **<1900** / 不打 | 有过经验但近年荒疏；这是 1 年内 LC 2800 的最大杠杆 |
| ICPC 参赛资格 | **已不可参赛** | 在职。**不要再把 ICPC 金作为目标**，「金牌水平」只是质量描述符 |
| 时间预算 | **真实 baseline 7–8h / 周**，5h 是最低维持档，10h 是 ceiling | 周表按 7–8h 排；4 周轮转（2 标准 + 1 加强 + 1 deload）；不要把 baseline 设回 10h |
| **位置 / 时区** | **加州 San Jose / Pacific Time（PDT 3–11 月，PST 11–次年 3 月）**| **所有比赛锚点必须按 PT 计算**——LC Weekly = 周六 19:30 PDT，CF 主流场 = 周末 09:35 PDT，LC Biweekly = 周六 07:30 PDT，AtCoder = 凌晨（必 vp）|
| 母语 | 中文 | 文档全部中文；保留英文术语（CF / Dijkstra / DP 等）|
| 既有资产 | cp-wiki 主题覆盖度顶满（300+ 主题文件，含 `math/` 100+、`ds/` 67、`graph/` 61）| **不要再让用户「补基础知识」**——他知识储备超出大多数选手 |

## 2. 目标定义（明确的、有降级阶梯）

- **主目标**：12 个月内 LC contest rating **稳定 2700–2800**
- **次目标**：CF rating 到 **2200+**（IM 边界）
- **降级阶梯**：
  - 如果 M6 末 LC < 2300 → 把 M7–M9 图论/字符串压缩，给 M2 思维专题加一轮
  - 如果 M9 末 LC < 2500 → **降目标到 M12 稳 2700**，2800 留作下一年
  - 如果时间预算稳定 < 5h/周 → 全部时间表延长到 18 个月，主目标降为 2700

KPI 节点：M3 末 / M6 末 / M9 末 / M12 末（dashboard 的「当前状态」表自动对齐）。

## 3. 重大设计决策 + why（防反射性 over-engineer）

### 3.1 为什么不在主线包含 FFT / 网络流 / SAM / 杜教筛 / 圆方树

这些主题在原始版本 plan 里被排进了 M7–M10。**用户明确反驳后被全部删除**：

> "LC 好像很少用到特别高级冷门的数据结构或算法，你写的这些内容再考虑一下"

实证：近 100 场 LC weekly + biweekly 中这五块知识合计 Q4 出现率 **< 2%**。它们对 ICPC 区域金 / CF Master+ 是必需，但对 **LC 2800 这个具体目标** 边际收益太低。

**容易踩的坑**：用户在职、有积累，新会话可能反射性建议「补一下 FFT 模板」「SAM 也得学」——**不要这样建议**，除非用户主动问。如果用户后期完成 M12 主线后想拓展，再放在 bonus tier。

### 3.2 为什么 CF 是必需而非可选

观察：LC 2700–2800 群体几乎都是 CF Master+（CF 2200+），且这不是巧合——LC Q4 题型分布偏窄（构造 / DP / DS / 数学的有限组合反复出现），而 CF 的题型宽度（计数 / 博弈 / 构造 / DP 优化等）覆盖了 LC Q4 缺失的训练面。**用户从 LC 2200–2400 的瓶颈不在算法知识（他都会），而在「思维 / 构造类题型的肌肉」**——只有 CF / AtCoder 能练出来。

**容易踩的坑**：未来的 Claude 不要被「在职没空」说服去掉 CF；如果用户时间紧张，先砍专题日，**别砍 CF round**。

### 3.3 为什么是 4 层回顾（L1–L4）+ 22% 时间占比

- L1 微回顾（每周 30min）：日级遗忘曲线
- L2 月度回顾（每月 1 次 3–4h）：周级遗忘曲线
- L3 季度审计（每 3 月 1 周）：月级遗忘曲线
- L4 模板热重载（每 2 月 30min）：模板代码记忆衰减最快，单独维护

22% 是 spaced-repetition 文献中「介绍新材料 vs 复习」的有效区间下限。再低就遗忘，再高就训练量不够。**用户已认可这个比例**，不要随便调。

### 3.4 dashboard 是自动生成 vs 手动维护

选了「自动生成 + 手工编辑会被覆盖」的方向，因为：
- 用户是工程师，对「源数据 + 派生」的 pattern 接受度高
- 比手动维护 dashboard 摩擦小（log 一次 → dashboard 5 行命令刷新）
- 单一可信来源（contest-log.md 是唯一真相，dashboard 只是 view）

**容易踩的坑**：未来 Claude 不要把 dashboard 改成手工维护、不要把数据塞进 dashboard.md（必然被下次 `make dashboard` 覆盖）。**新数据加到 contest-log.md，新指标加到 gen-dashboard.py。**

## 4. 用户已认可的 conventions（不要重新讨论）

- ✅ 用 markdown checkbox `- [ ]` 追踪刷题进度（不用 issue tracker / Trello / 第三方工具）
- ✅ contest-log.md 的 8 列字段（Date / Type / Round / Rating / Solved / Time / Fail / Trick）
- ✅ Fail 列的 tag 规范：`知识/<主题>` `思维` `实现` `心态` `读题`
- ✅ Type 列的简写：`LCW` `LCB` `CF` `AC` `vp-LCW` `vp-CF` `vp-AC`
- ✅ Plan 里的题号引用 LC 中文站（`leetcode.cn/problems/{slug}`）
- ✅ 所有训练相关文件集中在 `training/`，**侧栏在末尾**（不要挪回顶部）
- ❌ 不要把回顾做成「整理 wiki / 补排版 / 归类」（这是反 pattern；回顾 = 在限时压力下独立调用旧知识）
- ❌ 不要追求「先把所有基础打牢再打比赛」（用户已经过这个阶段）
- ❌ 不要再囤新主题（用户主 wiki 已经过载，再加边际收益接近零）
- ❌ **不要假设北京时间** —— 用户在加州，所有比赛锚点按 PT 算（LC Weekly = 周六 19:30 PDT，不是周日 10:30）。改周表 / 改专题日 / 排 vp 时都要先核对 PT 时间。这条已经踩过一次坑。
- ❌ **不要让用户打 random Div3 / 弱 setter Div2** —— CF round 质量参差，对 LC 2800 价值低。
- ✅ **赛事 / vp 源优先级（plan「赛事质量分级」节有完整表）：**
  - **S 档**（首选）：AtCoder ABC F/G + ARC、CF Educational Round、CF Global / Goodbye / Hello
  - **A 档**：CSES Problem Set（无 round 约束，按主题刷）、灵茶山艾府 LC 题单、ICPC region Gym mirror
  - **B 档**：CF Div1+Div2、CF Div1（用户 CF 上 1900 后才能打）
  - **C 档及以下**：random Div2 / 弱 setter / Div3 —— 只有 S/A/B 全用完才轮到

## 5. 系统架构（数据流）

```
                        ┌──────────────────┐
                        │  人手输入        │
                        ├──────────────────┤
                        │  contest-log.md  │  每场比赛追加 1 行
                        │  plan.md         │  每题改 [ ] → [x]
                        └────────┬─────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │  scripts/gen-dashboard.py   │  纯函数，无副作用
                  │  - parse log                │
                  │  - parse plan checkbox      │
                  │  - scan wiki mtime          │
                  └────────┬────────────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │  dashboard.md   │  纯派生，可重生成
                    └─────────────────┘
```

关键不变量：
- **contest-log.md 是唯一真相源**——dashboard 的 rating / 漏洞分布 / 最近比赛全来自这里
- **plan.md 的 checkbox 状态是唯一真相源**——dashboard 的「月度专题进度」全来自这里
- **wiki 文件的 mtime 是唯一真相源**——dashboard 的「久未触碰主题」全来自这里
- dashboard.md 永远可以从前三者重生成；删掉无所谓，`make dashboard` 即恢复

## 6. 当前 snapshot（系统初次落地时）

| 项 | 值 |
|---|---|
| 落地日期 | 2026-04-25 |
| Plan 中题数 | 129（M2 + M3 + M4 + M5 + M6 + M7 + M8 + M9）|
| Plan 中已完成 | 0 |
| Contest log 行数 | 0（用户尚未开始第一场记录）|
| Wiki 主题文件总数 | ~330（按 `_sidebar.md` 估）|
| 最新主 wiki commit | `cebf35e add: contest weekly 475`（2026-01-10）|
| LC 当前 rating | 用户自报 2200–2400（precise 待用户在 contest-log 第一行填入）|
| CF 当前 rating | <1900 / 未启用 |
| Docker 验证 | ✓（http://localhost:3001/ 渲染 OK）|

## 7. 接续指引

如果用户在新会话里说类似「看一下我的训练进度」或「调整一下计划」：

1. **第一步**：读这个 CONTEXT.md（你现在做的事）
2. **第二步**：读 [`dashboard.md`](dashboard.md) 看当前数据
3. **第三步**：看 git log 最近的 training-related commits（`git log --oneline training/ scripts/gen-dashboard.py`）
4. **第四步**：跑 `make dashboard` 确保 dashboard 是最新的（避免读到陈旧数据）
5. **第五步**：再决定具体动作

如果用户提出 **修改计划主线**（加主题 / 删主题 / 改月份）：
- 先回顾本文档的 §3 设计决策，确认动机一致
- 任何「加 ICPC-only 主题」的提议都要 **明确确认这是用户主动提出的，而非你的反射性建议**
- 修改后更新本文档的 §3 / §6

如果用户提出 **改系统**（dashboard 字段 / log 格式 / 自动化）：
- §5 的不变量不能动；如果一定要动，先在本文档记录变更原因
- 任何修改都跑 `make dashboard` 验证不破坏现有数据

## 8. 一些可能的演进方向（用户未来可能要做的）

记下来给未来 Claude 当线索，不要主动提议：

- 把 contest-log.md 升级为 CSV 或 SQLite，dashboard 改为 Python pandas 处理（数据量超 200 行后性能可能要紧）
- 加一个 `training/notes/` 目录存读 editorial 时的洞察笔记（与主 wiki 的「主题永久知识」分开——这里是「时间锚定的洞察」）
- 加一个「每月归因表自动生成 issue list」的脚本，把高频 fail tag 自动转为下个月的专题题源
- 接入 LC 中文站的 GraphQL API（如果可行）自动拉 rating + 题目 AC 状态，免去手填 plan checkbox

但 **当前阶段保持简单**——用户的最大风险不是工具不够强，是工具太强导致维护成本超过训练价值。
