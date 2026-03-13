#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/workspace/development/humpback-acoustic-embed

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

cd "$APP_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

TF_EXTRA="${TF_EXTRA:-tf-linux-cpu}"

git fetch --prune origin
git checkout main
git reset --hard origin/main

uv sync --locked --extra "$TF_EXTRA"

cd frontend
npm ci
npm run build
cd "$APP_DIR"

mkdir -p logs

if ! command -v supervisord >/dev/null 2>&1; then
  uv tool install supervisor
  export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
fi

if supervisorctl -c supervisord.conf status >/dev/null 2>&1; then
  supervisorctl -c supervisord.conf reread
  supervisorctl -c supervisord.conf update
  supervisorctl -c supervisord.conf restart humpback-api humpback-worker
else
  supervisord -c supervisord.conf
fi

supervisorctl -c supervisord.conf status
