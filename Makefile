.DEFAULT_GOAL := help

.PHONY: dev api worker frontend-dev frontend-install build test test-watch lint typecheck playwright help

# ── Local development ──────────────────────────────────────────────────────────
dev: ## Start API + worker + frontend dev server (colored output, Ctrl+C stops all)
	uv run honcho start -f Procfile.dev

api: ## Run API server only (port 8000; set HUMPBACK_API_PORT to change)
	uv run humpback-api

worker: ## Run worker process only
	uv run humpback-worker

# ── Frontend ───────────────────────────────────────────────────────────────────
frontend-dev: ## Run frontend dev server only (port 5173, proxies API to :8000)
	cd frontend && npm run dev

frontend-install: ## Install frontend node_modules (run once after clone)
	cd frontend && npm install

build: ## Production build: frontend → static/dist/, then serve via API on :8000
	cd frontend && npm run build

# ── Testing & quality ──────────────────────────────────────────────────────────
test: ## Run all backend tests
	uv run pytest tests/

test-watch: ## Run backend tests in watch mode (pytest-watch)
	uv run ptw

lint: ## Run Ruff + Pyright via pre-commit
	uv run pre-commit run --all-files

typecheck: ## Run Pyright type checker only
	uv run pyright

playwright: ## Run frontend Playwright tests (requires backend + dev server running)
	cd frontend && npx playwright test

# ── Help ───────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
