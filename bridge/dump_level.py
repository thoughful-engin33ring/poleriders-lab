#!/usr/bin/env python3
"""Dump an arena's static geometry (Box2D polygons, world meters, +y down) to arenas/arena<N>.json.

    python3 bridge/dump_level.py 2        # Flatirons

Starts a throwaway game instance on its own socket (training instances are untouched)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spr_env import PoleRidersEnv, SOCK_DIR, launch_game  # noqa: E402

arena = int(sys.argv[1]) if len(sys.argv) > 1 else 2
sock = os.path.join(SOCK_DIR, "spr-dump.sock")
proc = launch_game(sock)
try:
    time.sleep(16)
    env = PoleRidersEnv(sock=sock)
    env.start_match(arena=arena)
    geo = json.loads(env.b.cmd("level"))
    st = env.b.state()
    geo["arena"] = arena
    geo["spawn"] = {p["id"]: [p["x"], p["y"]] for p in st["players"]}
    geo["ball"] = [st["ball"]["x"], st["ball"]["y"]]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arenas", f"arena{arena}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(geo, open(out, "w"))
    n = sum(len(b["polys"]) for b in geo["bodies"])
    tags = sorted({b["tag"] for b in geo["bodies"]})
    print(f"arena {arena}: {len(geo['bodies'])} static bodies, {n} polygons, tags {tags} -> {os.path.normpath(out)}")
finally:
    proc.kill()
