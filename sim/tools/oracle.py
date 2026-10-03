#!/usr/bin/env python3
"""The real game as an oracle for the native sim: one game instance (own socket, separate from
training) running shadow matches we can dump raw.

    o = Oracle(); o.boot()            # launch game, start a level-1 match, freeze it, 1 shadow
    d = o.dump(0)                     # raw world + objects of shadow 0 (see bridge/farm.c)
    o.physstep(0, 100)                # bare b2World::Step x100, no game logic
    o.reset(0); o.act([lx0, rx0, ry0, lx1, rx1, ry1], n=1)

Level: `Oracle(level=1)` (default; 1 = Kalgoorlie, the training map; 2 = Flatirons). The arena menu
cell -> level mapping is not what the menu script assumes (menu "arena 2" gives level id 1), so boot()
reads PlayState+0x214 from a `farm dump` and retries other menu cells until it gets the requested
level id, or raises. boot(launch=False) raises if the running oracle is on another level.

CLI: oracle.py dump OUT.json [LEVEL]  (boot if needed, reset shadow 0, dump tick 0)
"""
import json, os, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bridge"))
from farm_env import FarmHost  # noqa: E402
from spr_env import SOCK_DIR, PoleRidersEnv, launch_game  # noqa: E402

SOCK = os.path.join(SOCK_DIR, "spr-oracle.sock")
PS_LEVEL = 0x214  # PlayState::level (1..5)
# menu cell (spr_env.start_match(arena=...)) that gave each level id, as observed 2026-09-30; boot()
# tries this one first, then the others
MENU_CELL = {1: 2, 2: 4, 3: 3, 4: 1}


def dump_level(d):
    """Level id (PlayState+0x214) of a `farm dump`."""
    return struct.unpack_from("<i", bytes.fromhex(d["objects"]["playstate"]["raw"]), PS_LEVEL)[0]


def kill_oracles(sock=SOCK):
    """SIGKILL every game started on the oracle socket (the game ignores SIGTERM). Never touches the
    training games (they use other sockets)."""
    ps = subprocess.run(["ps", "axeww", "-o", "pid=,command="], capture_output=True, text=True).stdout
    for ln in ps.splitlines():
        if "MacOS/Sportsfriends" in ln and f"SPRBRIDGE_SOCK={sock}" in ln:
            os.kill(int(ln.split()[0]), 9)


class Oracle:
    def __init__(self, sock=SOCK, n=1, arena=2, level=1):
        # arena: terrain file for FarmHost observations (arenas/arenaN.json); level: required level id
        self.sock, self.n, self.arena, self.level = sock, n, arena, level
        self.proc = None
        self.host = None

    def _attach(self):
        self.host = FarmHost(self.sock, self.n, self.arena, frame_skip=1)
        self.host.start()
        self.reset(0)
        return dump_level(self.dump(0))

    def boot(self, launch=True):
        if not launch:
            got = self._attach()
            if got != self.level:
                raise RuntimeError(f"oracle on {self.sock} runs level {got}, want {self.level}")
            return self
        cells = [MENU_CELL.get(self.level, 1)] + [c for c in (1, 2, 3, 4) if c != MENU_CELL.get(self.level, 1)]
        seen = []
        for cell in cells:
            kill_oracles(self.sock)
            self.proc = launch_game(self.sock, size=(320, 180))
            time.sleep(16)
            PoleRidersEnv(frame_skip=1, sock=self.sock).start_match(arena=cell)
            got = self._attach()
            seen.append((cell, got))
            if got == self.level:
                return self
        kill_oracles(self.sock)
        raise RuntimeError(f"no menu cell gave level {self.level} (menu cell -> level: {seen})")

    def cmd(self, s):
        r = self.host.b.cmd(s)
        j = json.loads(r)
        if isinstance(j, dict) and "error" in j:
            raise RuntimeError(j["error"])
        return j

    def reset(self, k=0, clock=None):
        return self.cmd(f"farm reset {k}" + (f" {clock}" if clock else ""))

    def dump(self, k=0):
        return self.cmd(f"farm dump {k}")

    def physstep(self, k=0, n=1):
        return self.cmd(f"farm physstep {k} {n}")

    def act(self, acts, n=1):
        """acts: 6 floats per shadow (lx0 rx0 ry0 lx1 rx1 ry1, SDL convention, +y down)."""
        return self.cmd(f"farm act {n} " + " ".join(f"{v:.6f}" for v in acts))

    def close(self):
        kill_oracles(self.sock)


if __name__ == "__main__":
    if sys.argv[1:2] == ["dump"]:
        o = Oracle(level=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
        try:
            o.boot(launch=False)
        except Exception:  # noqa: BLE001 — no oracle game running yet
            o.boot(launch=True)
        o.reset(0)
        json.dump(o.dump(0), open(sys.argv[2], "w"))
