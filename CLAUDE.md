# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal competitive-programming knowledge wiki rendered by [docsify](https://docsify.js.org/) — there is no build step, no package manager, no tests. Every page is a Markdown file served verbatim and rendered client-side. The structure mirrors OI-wiki (see `README.md`), but content is curated for personal review and includes a long-running stream of solved LeetCode contest problems.

All prose is in Chinese (题意 / 思路 / 详细代码). Match this when adding content.

## Serve locally

```sh
docker-compose up        # serves the repo at http://localhost:3000 via docsify-cli
```

The image (`binacslee/cp-wiki:latest`) is built and pushed by `.github/workflows/docker-buildx.yml` on every push to `main` (multi-arch: linux/amd64, linux/arm64/v8). The `Dockerfile` itself is just `npm i -g docsify-cli` + `docsify serve /docs`.

## Architecture you can't see from a single file

### Sidebar is hand-maintained

`_sidebar.md` is the canonical table of contents that docsify reads. **It is not auto-generated from the file tree.** When you add a new `.md` page (a new topic, not a problem inside an existing topic), you must add a corresponding `* [Title](path/to/file.md)` entry in `_sidebar.md` at the right indentation, otherwise the page is unreachable from navigation.

When you only append a problem to an existing topic file (the common case for contest additions — see below), do **not** touch the sidebar.

### Content domains (top-level dirs)

Each is a docsify section with its own `README.md` (and often `index.md`) plus topic files:

- `lang/` C++ language basics · `basic/` algorithm fundamentals · `search/` search · `dp/` dynamic programming · `string/` strings · `math/` math · `ds/` data structures · `graph/` graph theory · `geometry/` computational geometry · `misc/` miscellaneous · `topic/` cross-cutting techniques (思维, RMQ, 模拟, trick, …)

Note the pre-existing typo files `dp/READMD.md` and `ds/READMD.md` — both coexist with the correct `README.md`. Leave them alone unless asked.

### Problem-entry conventions

Problems are not stored in a separate contest tree. They are **appended into the topic file that matches the algorithm used**. Recent commits show the pattern: `add: contest weekly 475` modifies `dp/line.md`; biweekly/weekly additions land in `dp/`, `ds/`, `graph/`, etc., depending on tag.

Every problem follows this exact block — copy it from `template-leetcode.md` / `template-codeforces.md` / `template-acwing.md` / `template-luogu.md` / `template-swordoffer.md` and fill in:

```markdown
> [!NOTE] **[LeetCode <id>. <title>](<url>)**
>
> 题意: TODO

> [!TIP] **思路**
>
> <prose explanation>

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// solution
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *
```

Several pieces here are load-bearing for docsify plugins — don't drop them:

- `> [!NOTE]` / `> [!TIP]` — rendered by `docsify-plugin-flexible-alerts`.
- `<!-- tabs:start -->` / `<!-- tabs:end -->` with `##### **C++**` / `##### **Python**` — rendered by `docsify-tabs`. The `#####` heading level and bold-text language label are how the plugin identifies tabs.
- `<details><summary>详细代码</summary> … </details>` — collapsible code; keep it around the tabs block.
- Trailing `<br>` and `* * *` — visual separator between problems. Always include both.

It is normal and expected for the Python tab to be empty (`​`) when only a C++ solution exists. Do not delete the empty Python tab.

### Math, diagrams, links

- Math is rendered by `docsify-katex` (v1.4.5 pinned in `index.html` because newer versions had issues). Use `$…$` for inline and `$$…$$` for display.
- Mermaid diagrams render via `docsify-mermaid` — use a fenced `mermaid` block.
- LeetCode CN URLs use the `leetcode.cn` host (a prior commit migrated away from the old `leetcode-cn.com`). Match this when adding new problems.
- The repo link in `index.html` (`OpenKikCoc/cp-wiki`) points to the upstream/origin project, not this fork's GitHub remote — don't "fix" it unless asked.

## Editing protocol

- Adding a contest problem: identify the algorithmic tag → open the matching topic file (e.g., `dp/line.md`, `ds/fenwick.md`) → append a new problem block at the appropriate sub-section, using the platform's template verbatim. No sidebar edit needed.
- Adding a brand-new topic page: create the `.md`, then add it to `_sidebar.md` under the right parent.
- Don't introduce a build system, linter, or formatter — every file is meant to be hand-edited and rendered as-is.
