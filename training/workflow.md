# 工作流

## 每周 3 个动作

```sh
# 1. 比赛后 5 分钟内 —— 在 contest-log.md 追加一行
#    | YYYY-MM-DD | LCW | round | rating | solved | time | fail | trick |

# 2. 专题做完一题 —— 在 plan.md 把 [ ] 改成 [x]
#    （编辑器手改即可；批量可用 sed）

# 3. 想看进度时（每周或赛后）
make dashboard          # 0.1s 重生成 dashboard.md
make serve              # 起 docsify 容器，浏览器看 http://localhost:3001/
make stop               # 用完关容器
```

## 每月最后一周（L2 月度回顾）

- 看 dashboard 的「漏洞分布 top 10」选下次专题方向
- 看 dashboard 的「久未触碰 wiki 主题」选下次 vp 主题
- 不开新主题，做老主题的 vp + 1 个模板手敲重写

## 与主 wiki 的协同

每场 contest / 每场 vp / 每读一篇 editorial → **新 trick 必须立即写入主 wiki 对应主题的 markdown 文件**（如 `../dp/line.md`、`../topic/trick.md`）。一年下来这是最大的复利资产。

主 wiki 文件的 `mtime` 自动喂给 dashboard 的 staleness ranking——你越久没碰某主题，它越靠前提醒你回顾。

## 文件清单

| 文件 | 类型 | 用途 | 更新方式 |
|---|---|---|---|
| [`plan.md`](plan.md) | 静态计划 | 12 个月路线图 + 129 道 checkbox 题 | 做完一题手改 `[ ]` → `[x]` |
| [`contest-log.md`](contest-log.md) | 行为日志 | 每场比赛 / vp 一行的归因表 | 比赛后 5 分钟内追加一行 |
| [`dashboard.md`](dashboard.md) | 自动报表 | 当前 rating / 月度进度 / 漏洞分布 / wiki staleness | `make dashboard` 自动生成（**勿手改**）|
| [`CONTEXT.md`](CONTEXT.md) | 上下文存档 | 用户 profile + 设计决策 + 给未来 Claude 会话的接续指引 | 重大决策变更时 |
| `../scripts/gen-dashboard.py` | 工具 | dashboard 生成器（无外部依赖）| 不常改 |
| `../Makefile` | 入口 | `make dashboard` / `make serve` / `make stop` | 不常改 |
