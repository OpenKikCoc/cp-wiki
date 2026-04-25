# 训练 Dashboard

> 最近一次生成：**2026-04-25**　·　[训练计划](plan.md)　·　[比赛 Log](contest-log.md)

> [!NOTE] **此页面由 `make dashboard` 自动生成**
>
> 数据源：`training/contest-log.md` + `training/plan.md` + wiki 文件 mtime。手工编辑会被下次生成覆盖。

## 当前状态

| 指标 | 当前 | 目标（M3 末 / M6 末 / M9 末 / M12 末）|
|---|---|---|
| LC rating | **--** | ≥2200 / 2300 / 2500 / **2700–2800** |
| CF rating | **--** | 1700 / 1900 / 2000 / 2200+ |
| 已打 contest | 0（LC 0 · CF 0 · AC 0 · vp 0）| ~120 总（每周 ≥2 场）|
| Plan 题完成 | **0 / 149** (0.0%) | 全部 |

## 月度专题进度

| 月 | 完成 / 计划 | 进度 |
|---|---|---|
| M1 | 0 / 0 | `—（无专题题单）` |
| M2 | 0 / 12 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M3 | 0 / 18 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M4 | 0 / 8 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M5 | 0 / 18 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M6 | 0 / 23 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M7 | 0 / 19 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M8 | 0 / 26 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M9 | 0 / 25 | `░░░░░░░░░░░░░░░░░░░░ 0%` |
| M10 | 0 / 0 | `—（无专题题单）` |
| M11 | 0 / 0 | `—（无专题题单）` |
| M12 | 0 / 0 | `—（无专题题单）` |

## Rating 曲线（最近 12 场）

**LC：**
```
（数据不足；至少 2 场比赛后此处会显示曲线）
```

**CF：**
```
（数据不足；至少 2 场比赛后此处会显示曲线）
```

## 漏洞分布（按 fail 归因 tag）

（暂无数据；fail 列填 `知识/<主题>` `思维` `实现` `心态` `读题` 后会自动统计）

## 最近 8 场 contest

（log 还是空的；去 `contest-log.md` 追加第一行）

## 久未触碰的 wiki 主题（L2 月度回顾推荐）

文件 mtime 越老越优先回顾——把它们当作下次 vp 的主题选择器。

| 文件 | 上次修改 |
|---|---|
| [basic/basic-sort.md](../basic/basic-sort.md) | 511 天前 |
| [basic/binary-lifting.md](../basic/binary-lifting.md) | 511 天前 |
| [basic/binary.md](../basic/binary.md) | 511 天前 |
| [basic/construction.md](../basic/construction.md) | 511 天前 |
| [basic/deduction.md](../basic/deduction.md) | 511 天前 |
| [basic/divide-and-conquer.md](../basic/divide-and-conquer.md) | 511 天前 |
| [basic/prefix-sum.md](../basic/prefix-sum.md) | 511 天前 |
| [basic/sort.md](../basic/sort.md) | 511 天前 |

## 工作流

1. **每场 contest 后**：往 `training/contest-log.md` 表格末尾追加一行（< 5 分钟）。
2. **专题做完一题**：在 `training/plan.md` 把 `- [ ]` 改为 `- [x]`。
3. **每周 / 每次想看进度**：跑 `make dashboard`（或 `python3 scripts/gen-dashboard.py`），刷新本页。
4. **commit**：把 log + plan 的改动一起 commit，git 自然成为审计日志。

