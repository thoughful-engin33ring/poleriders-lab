#!/bin/bash
# 1) round robin for the new top members (t20 snapshots), 2) 4 more seeds of t20's recipe to 90M (seed variance).
cd "$(dirname "$0")/.."
.venv/bin/python -u gauntlet.py fill --games 64 --procs 12 >> runs/gauntlet/fill.out 2>&1
.venv/bin/python - <<'PY' >> runs/overnight.out 2>&1
import re
s = open("ops/overnight.py").read()
s = re.sub(r'RUNS = \[.*?\]\nEVERY', '''RUNS = [("n1_t20_seed2", 90e6, [], 2), ("n1_t20_seed3", 90e6, [], 3), ("n1_t20_seed4", 90e6, [], 4),
        ("n1_t20_seed5", 90e6, [], 5)]
EVERY''', s, flags=re.S)
open("ops/seeds.py", "w").write(s)
print("wrote ops/seeds.py")
PY
.venv/bin/python -u ops/seeds.py >> runs/overnight.out 2>&1
