#!/usr/bin/env bash
# Embedding Space Viewer'ı başlatır: http://127.0.0.1:8765
cd "$(dirname "$0")"
exec uv run uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8765}" "$@"
