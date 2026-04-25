.PHONY: dashboard serve stop

# Regenerate dashboard.md from contest-log.md + training-plan + wiki mtime.
dashboard:
	@python3 scripts/gen-dashboard.py

# Start docsify dev server on :3001 (3000 may collide with other tools).
serve:
	@docker run -d --name cp-wiki-verify --rm -v $(CURDIR):/docs -p 3001:3000 binacslee/cp-wiki:latest >/dev/null && echo "→ http://localhost:3001/"

stop:
	@docker stop cp-wiki-verify >/dev/null 2>&1 || true
