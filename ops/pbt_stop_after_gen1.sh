#!/bin/bash
# Wait for PBT generation 1 (the 2nd tournament) to be recorded, then stop the controller and its members.
cd "$(dirname "$0")/.."
until python3 -c "import json,sys;sys.exit(0 if len(json.load(open('runs/pbt1/state.json'))['generations'])>=2 else 1)" 2>/dev/null; do sleep 20; done
sleep 5
pkill -f "pbt.py --name pbt1"
sleep 2
pkill -f "train_farm.py --run pbt1_m"
sleep 5
echo "$(date) PBT stopped after generation 1" | tee -a runs/pbt1/pbt.out
