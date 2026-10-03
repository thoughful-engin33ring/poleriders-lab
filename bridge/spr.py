#!/usr/bin/env python3
"""Client for the sprbridge socket (game must be started with `prmod.py run --bridge`).

    python3 spr.py settings     # limits the game actually loaded
    python3 spr.py state        # one state snapshot
    python3 spr.py watch [hz]   # stream snapshots (default 10 Hz)
    python3 spr.py pads         # controller slots
    python3 spr.py pad connect 0 | pad tap 0 a | pad set 0 lx=1 rt=1 | pad clear 0

As a library:
    from spr import Bridge
    b = Bridge(); s = b.state()   # dict: frame, in_match, score, bodies[{tag,x,y,a,vx,vy,w}]
"""
import json, os, socket, sys, time

SOCK = os.environ.get("SPRBRIDGE_SOCK",
                      os.path.expanduser("~/Library/Application Support/sportsfriends/sprbridge.sock"))


class Bridge:
    def __init__(self, path=SOCK):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.rf = self.sock.makefile("r")

    def cmd(self, line):
        self.sock.sendall((line + "\n").encode())
        return self.rf.readline().strip()

    def state(self):
        return json.loads(self.cmd("state"))

    def settings(self):
        return json.loads(self.cmd("settings"))

    def frame(self):
        return json.loads(self.cmd("frame"))["frame"]

    def pad(self, slot, *args):
        """pad(0, "connect") / pad(0, "tap", "a") / pad(0, "set", lx=0.5, a=1)"""
        words, kw = list(args), {}
        if words and isinstance(words[-1], dict):
            kw = words.pop()
        return json.loads(self.cmd(" ".join(["pad", words[0], str(slot), *map(str, words[1:])]
                                            + [f"{k}={v}" for k, v in kw.items()])))

    def set_pad(self, slot, **inputs):
        """Hold inputs until changed: buttons a b x y back guide start ls rs lb rb up down left
        right (0/1), triggers lt rt (0..1), sticks lx ly rx ry (-1..1, +y = down)."""
        return self.pad(slot, "set", inputs)

    def wait_frames(self, n):
        target = self.frame() + n
        while self.frame() < target:
            time.sleep(0.005)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "state"
    b = Bridge()
    if cmd == "watch":
        period = 1 / float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        while True:
            s = b.state()
            if s.get("in_match"):
                tags = {}
                for body in s["bodies"]:
                    tags[body["tag"]] = tags.get(body["tag"], 0) + 1
                ball = next((x for x in s["bodies"] if x["tag"] == "ball"), None)
                where = f"ball=({ball['x']:.2f},{ball['y']:.2f})" if ball else "no ball"
                print(f"frame {s['frame']} score {s['score']} {where} bodies {tags}")
            else:
                print(f"frame {s['frame']} (not in a match)")
            time.sleep(period)
    elif cmd in ("state", "settings", "ping", "frame", "pads"):
        print(b.cmd(cmd))
    elif cmd == "pad":  # raw passthrough: spr.py pad connect 0 / pad tap 0 a / pad set 0 lx=1
        print(b.cmd(" ".join(sys.argv[1:])))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
