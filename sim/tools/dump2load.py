#!/usr/bin/env python3
"""Convert a `farm dump` JSON (bridge/farm.c) into the loader format of src/b2dump.cpp
b2load_world: bodies in creation order (reverse of the world list), fixtures in creation order
(= proxy id order at construction), joints in creation order, plus the broadphase tree.

    dump2load.py dump.json out.load"""
import json, struct, sys


def main(src, dst):
    d = json.load(open(src))
    bodies = d["bodies"][::-1]                      # creation order
    idx = {b["a"]: i for i, b in enumerate(bodies)}
    by_addr = {int(b["a"], 16): i for i, b in enumerate(bodies)}
    out = [f"WORLD {d['world']['hdr']}"]
    out += [f"BODY {b['raw']}" for b in bodies]
    fx = []
    for i, b in enumerate(bodies):
        prev = None
        for f in b["fixtures"]:                     # list order = newest first
            pid = struct.unpack_from("<i", bytes.fromhex(f["raw"]), 0x38)[0]
            if prev is not None and pid > prev:
                sys.exit(f"body {i}: fixture proxy ids not in creation order ({pid} after {prev})")
            prev = pid
            fx.append((pid, i, f))
    for pid, i, f in sorted(fx, key=lambda t: t[0]):
        out.append(f"FIX {i} {f['raw']} {f['shape_type']} {f['shape']}")
    for j in d["joints"][::-1]:
        raw = bytes.fromhex(j["raw"])
        a, b = struct.unpack_from("<QQ", raw, 0x60)
        out.append(f"JOINT {j['type']} {by_addr[a]} {by_addr[b]} {j['raw']}")
    out.append(f"TREE {d['world']['tree']}")
    out.append(f"MOVES {d['world']['moves']}")
    open(dst, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
