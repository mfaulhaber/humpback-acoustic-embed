#!/bin/bash
# Run ruff check and format on Python files after edits

FILE_PATH=$(jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

if [[ "$FILE_PATH" == *.py ]]; then
  uv run ruff check --fix "$FILE_PATH" 2>/dev/null
  uv run ruff format "$FILE_PATH" 2>/dev/null
fi

exit 0
