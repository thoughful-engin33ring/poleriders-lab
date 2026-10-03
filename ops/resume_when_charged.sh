#!/bin/bash
# Resume farm3 (+ its evaluator) once the battery is charged enough: training at full load draws more than
# the 100 W adapter supplies, and below ~10% macOS throttles the CPU (2026-10-02, after a power-loss restart).
cd "$(dirname "$0")/.."
until [ "$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)" -ge "${1:-25}" ]; do sleep 60; done
echo "===== $(date '+%F %T') resumed: battery $(pmset -g batt | grep -o '[0-9]*%') =====" >> runs/farm3/train.out
nohup .venv/bin/python -u train_farm.py --run farm3 --resume --native --hosts 12 --matches 38 --procs 6 --arena 2 \
  --threads 18 --torch-threads 1 --min-std 0.1 --pit-frac 0 --no-sudden-death --format 7,300 --random-pole \
  --rotation-bonus 0.005 --rotation-anneal 100e6 >> runs/farm3/train.out 2>&1 &
nohup .venv/bin/python -u eval_h2h.py --run farm3 --ref farm2 --kinds same,latest --legacy "" --games 32 --threads 2 \
  --every 600 >> runs/farm3/h2h.out 2>&1 &
