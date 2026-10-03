#!/bin/bash
# Wait for pbt1's running generation-0 tournament to finish (state.json generation >= 1), then restart the controller
# so generations follow --gen-steps (25M env steps per member) — user, 2026-10-02.
cd "$(dirname "$0")/.."
until [ "$(.venv/bin/python -c "import json;print(json.load(open('runs/pbt1/state.json'))['generation'])")" -ge 1 ]; do sleep 20; done
sleep 30  # let it restart its members
pkill -f "pbt.py --name pbt1"; sleep 20
echo "===== $(date '+%F %T') controller restarted: generations every 25M env steps per member =====" >> runs/pbt1/pbt.out
nohup .venv/bin/python -u pbt.py --name pbt1 --resume --gen-steps 25e6 >> runs/pbt1/pbt.out 2>&1 &
