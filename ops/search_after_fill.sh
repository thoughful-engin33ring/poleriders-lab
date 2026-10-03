#!/bin/bash
# Start search1 once the gauntlet round robin (gauntlet.py fill) has finished.
cd "$(dirname "$0")/.."
while pgrep -f "gauntlet.py fill" >/dev/null; do sleep 30; done
mkdir -p runs/search1
echo "$(date) gauntlet filled; starting search1" >> runs/search1/search.out
nohup .venv/bin/python -u search.py --name search1 >> runs/search1/search.out 2>&1 &
