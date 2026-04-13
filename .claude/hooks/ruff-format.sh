#!/bin/bash
# Run ruff format on Python files after edits.
# Only formatting — lint auto-fix (ruff check --fix) runs at commit time
# via pre-commit, when all imports and usage are present together.

FILE_PATH=$(jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

if [[ "$FILE_PATH" == *.py ]]; then
  uv run ruff format "$FILE_PATH" 2>/dev/null
fi

exit 0
