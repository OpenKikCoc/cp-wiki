#!/usr/bin/env python3
"""Regenerate dashboard.md from contest-log.md + training-plan-lc-2800.md + filesystem.

Run: python3 scripts/gen-dashboard.py  (or `make dashboard`)
"""

from __future__ import annotations
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
LOG_PATH = TRAINING / "contest-log.md"
PLAN_PATH = TRAINING / "plan.md"
OUT_PATH = TRAINING / "dashboard.md"

WIKI_DIRS = ["dp", "ds", "graph", "string", "math", "topic", "basic", "search", "geometry", "lang", "misc"]


def parse_log(text: str) -> list[dict]:
    rows = []
    header_seen = False
    for line in text.splitlines():
        if line.startswith("| Date "):
            header_seen = True
            continue
        if not header_seen:
            continue
        if not line.startswith("|"):
            continue
        if "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        date = cells[0]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            continue
        rows.append(
            {
                "date": date,
                "type": cells[1],
                "round": cells[2],
                "rating": cells[3],
                "solved": cells[4],
                "time": cells[5],
                "fail": cells[6],
                "trick": cells[7],
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def parse_plan(text: str) -> "OrderedDict[str, dict]":
    sections: OrderedDict[str, dict] = OrderedDict()
    current = None
    for line in text.splitlines():
        m = re.match(r"^### (M\d+)", line)
        if m:
            current = m.group(1)
            sections[current] = {"done": 0, "total": 0, "title": line[4:].strip()}
            continue
        if current and re.match(r"^- \[[xX ]\] \[", line):
            sections[current]["total"] += 1
            if "[x]" in line[:6].lower():
                sections[current]["done"] += 1
    return sections


def latest_rating_by_platform(rows: list[dict]) -> dict[str, str]:
    """Return last seen rating for each rough platform (LC vs CF vs AC)."""

    def platform(t: str) -> str:
        if t.startswith("vp"):
            return ""  # vp doesn't change rating
        if t in ("LCW", "LCB"):
            return "LC"
        if t == "CF":
            return "CF"
        if t == "AC":
            return "AC"
        return ""

    latest: dict[str, str] = {}
    for r in rows:
        p = platform(r["type"])
        if p and r["rating"] and r["rating"] != "-":
            latest[p] = (r["date"], r["rating"], r["round"])
    return latest


def fail_distribution(rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        if not r["fail"] or r["fail"].strip() in ("--", "-", ""):
            continue
        for tag in re.split(r"[,，]", r["fail"]):
            tag = tag.strip()
            tag = re.sub(r"\([^)]*\)", "", tag).strip()  # strip "(Q4)" detail
            if tag:
                c[tag] += 1
    return c


def stale_wiki_files(n: int = 10) -> list[tuple[str, str]]:
    """Return n most stale .md files in WIKI_DIRS by mtime, formatted (relpath, days_ago)."""
    out = []
    for d in WIKI_DIRS:
        dp = ROOT / d
        if not dp.is_dir():
            continue
        for f in dp.rglob("*.md"):
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            if "READMD" in f.name or f.name in ("README.md", "index.md"):
                continue
            out.append((f.relative_to(ROOT).as_posix(), mtime))
    out.sort(key=lambda x: x[1])
    now = datetime.now().timestamp()
    return [(p, f"{int((now - m) / 86400)} 天前") for p, m in out[:n]]


def ascii_rating_curve(rows: list[dict], platform_filter: tuple[str, ...]) -> str:
    points: list[tuple[str, int]] = []
    for r in rows:
        if r["type"] not in platform_filter:
            continue
        try:
            v = int(r["rating"])
        except (ValueError, TypeError):
            continue
        points.append((r["date"], v))
    if len(points) < 2:
        return "（数据不足；至少 2 场比赛后此处会显示曲线）"
    lo, hi = min(p[1] for p in points), max(p[1] for p in points)
    rng = max(hi - lo, 1)
    width = min(60, len(points) * 4)
    out = []
    last = points[-1]
    out.append(f"  最新：{last[1]} ({last[0]}) | 区间 [{lo}, {hi}] | {len(points)} 场")
    # ascii bar of last 12
    last_n = points[-12:]
    height = 6
    for h in range(height, 0, -1):
        line = "  "
        threshold = lo + (h / height) * rng
        for _, v in last_n:
            line += "█ " if v >= threshold else "  "
        out.append(line)
    out.append("  " + "─" * (len(last_n) * 2))
    return "\n".join(out)


def render(rows: list[dict], plan_secs: "OrderedDict[str, dict]") -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    n_total = len(rows)
    n_lc = sum(1 for r in rows if r["type"] in ("LCW", "LCB"))
    n_cf = sum(1 for r in rows if r["type"] == "CF")
    n_ac = sum(1 for r in rows if r["type"] == "AC")
    n_vp = sum(1 for r in rows if r["type"].startswith("vp"))

    plan_done = sum(s["done"] for s in plan_secs.values())
    plan_total = sum(s["total"] for s in plan_secs.values())
    plan_pct = (plan_done / plan_total * 100) if plan_total else 0

    latest = latest_rating_by_platform(rows)
    fails = fail_distribution(rows)
    stale = stale_wiki_files(8)

    lines: list[str] = []
    lines.append("# 训练 Dashboard")
    lines.append("")
    lines.append(f"> 最近一次生成：**{today}**　·　[训练计划](plan.md)　·　[比赛 Log](contest-log.md)")
    lines.append("")
    lines.append("> [!NOTE] **此页面由 `make dashboard` 自动生成**")
    lines.append(">")
    lines.append("> 数据源：`training/contest-log.md` + `training/plan.md` + wiki 文件 mtime。手工编辑会被下次生成覆盖。")
    lines.append("")

    # --- KPI summary ---
    lines.append("## 当前状态")
    lines.append("")
    lines.append("| 指标 | 当前 | 目标（M3 末 / M6 末 / M9 末 / M12 末）|")
    lines.append("|---|---|---|")
    lc_now = latest.get("LC", ("--", "--", ""))[1]
    cf_now = latest.get("CF", ("--", "--", ""))[1]
    lines.append(f"| LC rating | **{lc_now}** | ≥2200 / 2300 / 2500 / **2700–2800** |")
    lines.append(f"| CF rating | **{cf_now}** | 1700 / 1900 / 2000 / 2200+ |")
    lines.append(f"| 已打 contest | {n_total}（LC {n_lc} · CF {n_cf} · AC {n_ac} · vp {n_vp}）| ~120 总（每周 ≥2 场）|")
    lines.append(f"| Plan 题完成 | **{plan_done} / {plan_total}** ({plan_pct:.1f}%) | 全部 |")
    lines.append("")

    # --- plan progress per month ---
    lines.append("## 月度专题进度")
    lines.append("")
    lines.append("| 月 | 完成 / 计划 | 进度 |")
    lines.append("|---|---|---|")
    for k, v in plan_secs.items():
        if v["total"] == 0:
            bar = "—（无专题题单）"
        else:
            pct = v["done"] / v["total"]
            filled = int(pct * 20)
            bar = "█" * filled + "░" * (20 - filled) + f" {pct*100:.0f}%"
        lines.append(f"| {k} | {v['done']} / {v['total']} | `{bar}` |")
    lines.append("")

    # --- rating curves ---
    lines.append("## Rating 曲线（最近 12 场）")
    lines.append("")
    lines.append("**LC：**")
    lines.append("```")
    lines.append(ascii_rating_curve(rows, ("LCW", "LCB")))
    lines.append("```")
    lines.append("")
    lines.append("**CF：**")
    lines.append("```")
    lines.append(ascii_rating_curve(rows, ("CF",)))
    lines.append("```")
    lines.append("")

    # --- fail distribution ---
    lines.append("## 漏洞分布（按 fail 归因 tag）")
    lines.append("")
    if not fails:
        lines.append("（暂无数据；fail 列填 `知识/<主题>` `思维` `实现` `心态` `读题` 后会自动统计）")
    else:
        lines.append("| Tag | 出现次数 | 占比 |")
        lines.append("|---|---|---|")
        total = sum(fails.values())
        for tag, count in fails.most_common(10):
            lines.append(f"| {tag} | {count} | {count/total*100:.1f}% |")
    lines.append("")

    # --- recent contests ---
    lines.append("## 最近 8 场 contest")
    lines.append("")
    if not rows:
        lines.append("（log 还是空的；去 `contest-log.md` 追加第一行）")
    else:
        lines.append("| Date | Type | Round | Rating | Solved | 关键 trick |")
        lines.append("|---|---|---|---|---|---|")
        for r in reversed(rows[-8:]):
            trick = (r["trick"] or "")[:40] + ("…" if len(r["trick"]) > 40 else "")
            lines.append(f"| {r['date']} | {r['type']} | {r['round']} | {r['rating']} | {r['solved']} | {trick} |")
    lines.append("")

    # --- stale wiki files ---
    lines.append("## 久未触碰的 wiki 主题（L2 月度回顾推荐）")
    lines.append("")
    lines.append("文件 mtime 越老越优先回顾——把它们当作下次 vp 的主题选择器。")
    lines.append("")
    lines.append("| 文件 | 上次修改 |")
    lines.append("|---|---|")
    for path, days in stale:
        # dashboard.md lives under training/, so wiki files need a ../ prefix
        lines.append(f"| [{path}](../{path}) | {days} |")
    lines.append("")

    # --- workflow ---
    lines.append("## 工作流")
    lines.append("")
    lines.append("1. **每场 contest 后**：往 `training/contest-log.md` 表格末尾追加一行（< 5 分钟）。")
    lines.append("2. **专题做完一题**：在 `training/plan.md` 把 `- [ ]` 改为 `- [x]`。")
    lines.append("3. **每周 / 每次想看进度**：跑 `make dashboard`（或 `python3 scripts/gen-dashboard.py`），刷新本页。")
    lines.append("4. **commit**：把 log + plan 的改动一起 commit，git 自然成为审计日志。")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    if not LOG_PATH.exists():
        print(f"missing {LOG_PATH}", file=sys.stderr)
        return 1
    if not PLAN_PATH.exists():
        print(f"missing {PLAN_PATH}", file=sys.stderr)
        return 1
    rows = parse_log(LOG_PATH.read_text(encoding="utf-8"))
    plan_secs = parse_plan(PLAN_PATH.read_text(encoding="utf-8"))
    OUT_PATH.write_text(render(rows, plan_secs), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}: {len(rows)} contests · {sum(s['total'] for s in plan_secs.values())} plan items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
