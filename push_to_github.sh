#!/usr/bin/env bash
# One-time: turn this folder into the ml-from-scratch repo and push it to GitHub (sid2983).
# Run from ~/Documents/learn:   bash push_to_github.sh
set -euo pipefail
cd "$(dirname "$0")"

# the cleaned notebook lives in notebooks/; the original scratch copy is no longer needed
[ -f logistic_reg.ipynb ] && rm -f logistic_reg.ipynb

git init -b main 2>/dev/null || git init
git add -A
git commit -m "Logistic regression from scratch: sigmoid, log-loss, GD, gradient check, tests" \
  -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_014QibJjVFazbCiYjbqNBD3J" || true

if command -v gh >/dev/null 2>&1; then
  gh repo create sid2983/ml-from-scratch --public --source=. --remote=origin --push \
    --description "Core ML algorithms from scratch in NumPy, verified against sklearn"
else
  echo "gh CLI not found. Create an empty public repo named ml-from-scratch at https://github.com/new then run:"
  echo "  git remote add origin git@github.com:sid2983/ml-from-scratch.git && git push -u origin main"
fi
