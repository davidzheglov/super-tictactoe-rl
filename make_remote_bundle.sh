#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-super_tictactoe_rl_remote.zip}"
rm -f "$OUT"
zip -r "$OUT" super_tictactoe_rl \
  -x "super_tictactoe_rl/.venv/*" \
  -x "super_tictactoe_rl/__pycache__/*" \
  -x "super_tictactoe_rl/*/__pycache__/*" \
  -x "super_tictactoe_rl/tic-tac-toe/*" \
  -x "super_tictactoe_rl/req.txt" \
  -x "super_tictactoe_rl/.DS_Store" \
  -x "super_tictactoe_rl/*/.DS_Store" \
  -x "super_tictactoe_rl/runs/*" \
  -x "super_tictactoe_rl/models/*.pt*" \
  -x "super_tictactoe_rl/models/*.pkl" \
  -x "super_tictactoe_rl/models/*.csv" \
  -x "super_tictactoe_rl/models/*.done"
echo "Wrote $OUT"
