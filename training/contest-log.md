# 比赛 Log（归因表）

> [!NOTE] **使用方式**
>
> 每场 contest（含 vp）后立刻新增一行。**不要积压**——超过 24 小时再记，归因准确度暴跌。
> dashboard 会从这张表自动汇总：rating 曲线、漏洞分布、最近 8 场摘要。
>
> 字段含义：
> - **Date** — `YYYY-MM-DD`，与日历对齐
> - **Type** — `LCW` (LC weekly) / `LCB` (LC biweekly) / `CF` (Codeforces round) / `AC` (AtCoder) / `vp-LCW` / `vp-CF` / `vp-AC`
> - **Round** — 比赛编号或 round 名（如 `weekly-484`、`Div2-1942`、`ABC-345`）
> - **Rating** — 赛后 rating（绝对值，不是 delta）
> - **Solved** — 形如 `4/4` 或 `3/6`
> - **Time** — 各题用时分钟数，逗号分隔（`Q1:5,Q2:8,Q3:18,Q4:35`，未 AC 写 `WA` 或 `-`）
> - **Fail** — fail 原因 tag，逗号分隔。规范 tag：`知识/<主题>` `思维` `实现` `心态` `读题`。例：`思维(Q4),知识/段树(Q3)`
> - **Trick** — 一句话 key insight；写进 wiki 的 trick 备份在这里

> [!TIP] **填写示例**
>
> ```
> | 2026-04-25 | LCW | weekly-484 | 2235 | 4/4 | Q1:5,Q2:8,Q3:18,Q4:35 | -- | 拆位贡献：按 30 个 bit 独立看每位贡献，bit 间无干扰 |
> | 2026-04-26 | CF | Div2-1942 | 1750 | 4/6 | A:4,B:8,C:25,D:40,E:WA,F:- | 思维(E),知识/段树(F) | 单调队列 + 段树二分一起用，先处理出线性序再上段树 |
> ```

| Date | Type | Round | Rating | Solved | Time | Fail | Trick |
|---|---|---|---|---|---|---|---|
<!-- 在这条线下面追加新行 -->
| | | | | | | | |
