#!/bin/bash
# Double-click: opens the Pole Riders patcher in your browser (keep this Terminal window open while using it).
cd "$(dirname "$0")"
pkill -f "patcher_gui.py\|prmod.py gui" 2>/dev/null
exec python3 prmod.py gui
