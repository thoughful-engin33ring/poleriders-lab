#!/usr/bin/env python3
"""Live training watcher: real-time exhibition matches of the latest policy + training charts.

    .venv/bin/python dashboard.py --run farm1     # then open http://localhost:8765

Live matches: by default 4 exhibition matches (Spectator) that the dashboard itself runs in the
native simulator in real time with the run's latest checkpoint — on the Farm tab: vs --ref (farm1) and
vs --hier-opp (manager2 with its own worker) at 1x, one league and one self-play game at 2x; each rider's
name is shown. Head-to-head numbers come from eval_h2h.py (runs/<run>/h2h.jsonl); /?train=1 shows the training matches instead (sampled from the trainer's read-only
`farm state` sockets, which also feed the heatmaps). It never disturbs training. Matches are drawn
from the physics state.
"""
import argparse, collections, csv, glob, json, math, os, random, socket, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK_DIR = os.path.expanduser("~/Library/Application Support/sportsfriends")
X_BINS, X_RANGE = 60, 30.0  # ball/rider x histograms over [-30, 30] m
RECENT_AGE = 125  # updates: "recent" league snapshots (the 5 newest at --pool-every 25) for the vs-recent chart


class Instance:
    def __init__(self, idx, sock):
        self.idx, self.sock, self.conn, self.rf = idx, sock, None, None
        self.state, self.alive, self.last_ok = None, False, 0.0
        self.ball_hist = [0] * X_BINS
        self.rider_hist = {1: [0] * X_BINS, 2: [0] * X_BINS}
        self.airborne = {1: 0, 2: 0}
        self.samples = 0

    def poll(self):
        try:
            if self.conn is None:
                self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.conn.settimeout(1.0)
                self.conn.connect(self.sock)
                self.rf = self.conn.makefile("r")
            self.conn.sendall(b"state\n")
            st = json.loads(self.rf.readline())
            self.state, self.alive, self.last_ok = st, True, time.time()
            if st.get("in_match") and st.get("players"):
                self.samples += 1
                b = st.get("ball")
                if b:
                    self.ball_hist[_bin(b["x"])] += 1
                for p in st["players"]:
                    if p["id"] in (1, 2):
                        self.rider_hist[p["id"]][_bin(p["x"])] += 1
                        if not p["on_ground"]:
                            self.airborne[p["id"]] += 1
        except (OSError, ValueError):
            self.alive = False
            try:
                self.conn and self.conn.close()
            except OSError:
                pass
            self.conn = None

    def compact(self):
        st = self.state or {}
        out = {"idx": self.idx, "alive": self.alive, "in_match": bool(st.get("in_match")),
               "score": st.get("score"), "time_left": st.get("time_left"),
               "frame": st.get("frame"), "match_id": st.get("match_id")}
        if st.get("players"):
            out["players"] = [{"id": p["id"], "x": p["x"], "y": p["y"], "a": p["a"],
                               "ghost": p["ghost"], "dead": p.get("dead", False), "on_ground": p["on_ground"],
                               "on_pole": p["on_pole"], "stats": p.get("stats"),
                               "pole": [[s["x"], s["y"]] for s in p["pole"]]} for p in st["players"]]
            out["ball"] = st.get("ball")
            out["walls"] = sorted(b["x"] for b in st.get("bodies", []) if b["tag"] == "wall"
                                  and abs(b["x"]) < 45)
            out["trolley"] = [[b["x"], b["y"]] for b in st.get("bodies", []) if b["tag"] == "ball"]
        return out


def _sort_key(k):
    return tuple(int(p) if p.isdigit() else 0 for p in str(k).split("."))


def _bin(x):
    return max(0, min(X_BINS - 1, int((x + X_RANGE) / (2 * X_RANGE) * X_BINS)))


class FarmMatch(Instance):
    """One shadow match of a farm host; its state is filled from the host's `farm state`."""

    def feed(self, st, alive):
        self.state, self.alive = st, alive
        if alive and st and st.get("in_match") and st.get("players"):
            self.samples += 1
            b = st.get("ball")
            if b:
                self.ball_hist[_bin(b["x"])] += 1
            for p in st["players"]:
                if p["id"] in (1, 2):
                    self.rider_hist[p["id"]][_bin(p["x"])] += 1
                    if not p["on_ground"]:
                        self.airborne[p["id"]] += 1

    def compact(self):
        out = super().compact()
        st = self.state or {}
        out["trolley"] = [st["trolley"]] if st.get("trolley") else []
        out["sudden_death"] = bool(st.get("sudden_death"))
        return out


def target_view(g, O, limit):
    """Both riders' current worker targets in world coordinates (x right, y down) for drawing."""
    targets = []
    bok, tok = g.parts_ok(O)
    for i, a in enumerate((1, 2)):
        sign = 1 if a == 1 else -1
        targets.append({"id": a, "body_x": sign * float(g.bx[i]) if g.body_on[i] else None,
                        "body_y": -float(g.by[i]) if g.body_on[i] and getattr(g, "air", None) is not None and g.air[i] else None,
                        "tip": [sign * float(g.tx[i]), -float(g.ty[i])] if g.tip_on[i] else None,
                        "body_ok": bool(bok[i]), "tip_ok": bool(tok[i]),
                        "left": round(max(0.0, float(g.left[i])) / limit, 2)})
    return targets


class Spectator:
    """Exhibition matches for watching, in the native simulator in real time, independent of training
    (whose matches run far too fast to follow). Each rider has its own controller: a flat policy (sticks
    every 4th frame, as in training) or a manager run driving its own frozen worker. Default slots: two
    league games at 1x (latest policy vs a pool snapshot, sides random) and two self-play games at 2x.
    With comparison runs (ref = a flat run, hier_opp = a manager run) the 1x slots become latest vs the
    ref's latest and latest vs the manager run, and one league and one self-play game run at 2x.
    Riders of runs in `legacy` (trained before the dead-rider observation fix) see dead riders the way
    they were trained. When a match ends the next one reloads the latest checkpoints and league games
    draw a new opponent the way the trainer does (half the time from the 5 newest snapshots).
    Needs the native sim (make -C sim env) and torch/numpy (run the dashboard with .venv/bin/python)."""

    SLOTS = [("L1", "league", 1.0), ("L2", "league", 1.0), ("S1", "self", 2.0), ("S2", "self", 2.0)]
    FRAME_SKIP = 4
    # Replays: a background thread plays each slot's next match at full speed and records it (REC_FPS frames
    # per second of game time); the page is served the recording at real time (x speed) — one match stale,
    # but never stutters (live simulation in this Python process froze whenever anything else held the GIL)
    REPLAY, REC_FPS = True, 30

    def __init__(self, run_dir, level=1, ref=None, hier_opp=None, legacy=()):
        import numpy as np
        import torch
        sys.path.insert(0, HERE)
        sys.path.insert(0, os.path.join(HERE, "bridge"))
        from native_env import NativeHost
        from train_ppo import ActorCritic, RunningNorm, net_from_ckpt
        torch.set_num_threads(1)
        self.np, self.torch, self.AC, self.RN = np, torch, ActorCritic, RunningNorm
        self.net_from_ckpt = net_from_ckpt
        self.run_dir, self.rng = run_dir, random.Random()
        self.run = os.path.basename(run_dir.rstrip("/"))
        self.cache = {}   # checkpoint path -> (mtime, (net, norm, size))
        self.hruns = {}   # run dir -> None (flat run) or {"worker": ..., "macro": ...} (manager run)
        self.legacy = set(legacy)
        self.ref = os.path.join(HERE, "runs", ref) if ref and os.path.isdir(os.path.join(HERE, "runs", ref)) else None
        self.hier_opp = (os.path.join(HERE, "runs", hier_opp)
                         if hier_opp and os.path.isdir(os.path.join(HERE, "runs", hier_opp)) else None)
        slots = list(self.SLOTS)
        if (self.ref or self.hier_opp) and slots[0][1] == "league":
            slots = (([("X1", "ref", 1.0)] if self.ref else []) + ([("X2", "hier", 1.0)] if self.hier_opp else [])
                     + [("L1", "league", 2.0), ("S1", "self", 2.0)])
        self.hier = self._hrun(run_dir)  # this run is a manager run: kept for WorkerSpectator checks
        self.viewed_t = 0.0  # last time a page showed these matches: they only run while someone watches
        self.slots = []
        for i, (idx, kind, speed) in enumerate(slots):
            # the training rule (user): no sudden death, a tie at the end of time is a draw
            env = NativeHost(1, 1, 1, sd_frac=0.0, max_steps=10 ** 9, seed=1000 + i, threads=1, level=level,
                             end_on_sd=True, match_format=(7, 300.0))  # training format: 7 goals / 5 min
            self.slots.append({"idx": idx, "kind": kind, "speed": speed, "env": env, "t": 0,
                               "act": np.zeros((1, 2, 3), np.float32), "view": None, "last": None,
                               "view_t": 0.0})
        for s in self.slots:
            s["obs"] = s["env"].reset_all()[0].copy()
            s["rec"] = s["next"] = None
            self._new_game(s)
        threading.Thread(target=self._produce if self.REPLAY else self._loop, daemon=True).start()

    def _load(self, path):
        try:
            mt = os.path.getmtime(path)
            hit = self.cache.get(path)
            if hit and hit[0] == mt:
                return hit[1]
            ck = self.torch.load(path, weights_only=False)
            net, norm, size = self.net_from_ckpt(ck)  # any architecture / action head
            self.cache[path] = (mt, (net, norm, size))
            return net, norm, size
        except Exception:  # noqa: BLE001 — mid-write or missing: keep what we had
            hit = self.cache.get(path)
            return hit[1] if hit else None

    def _hrun(self, run_dir):
        """None for a flat run; for a manager run (train_farm.py --hier) its frozen worker and macro."""
        if run_dir in self.hruns:
            return self.hruns[run_dir]
        info = None
        try:
            ha = self.torch.load(os.path.join(run_dir, "ckpt.pt"), weights_only=False).get("args", {})
            if ha.get("hier"):
                import hier_env
                import train_worker as tw
                wpath = ha["hier"] if os.path.isabs(ha["hier"]) else os.path.join(HERE, ha["hier"])
                info = {"worker": hier_env.load_worker(wpath), "macro": ha.get("macro", 8), "tw": tw, "he": hier_env}
        except Exception as e:  # noqa: BLE001
            print(f"spectator: no hierarchy info for {run_dir} ({e!r})", flush=True)
        self.hruns[run_dir] = info
        return info

    def _ctl(self, run_dir, path=None, name=None):
        """A rider controller: run_dir's latest checkpoint (or `path`, e.g. a pool snapshot)."""
        m = self._load(path or os.path.join(run_dir, "ckpt.pt"))
        if not m:
            return None
        run = os.path.basename(run_dir.rstrip("/"))
        return {"m": m, "h": self._hrun(run_dir), "name": name or run, "legacy": run in self.legacy}

    def _new_game(self, s):
        cur = self._ctl(self.run_dir)
        s["ctl"], s["opp"], s["opp_agent"] = {1: cur, 2: cur}, None, 0
        opp = None
        if s["kind"] == "league":
            pool = os.path.join(self.run_dir, "pool")
            files = sorted(f for f in os.listdir(pool) if f.endswith(".pt")) if os.path.isdir(pool) else []
            if files:
                name = self.rng.choice(files[-5:]) if self.rng.random() < 0.5 else self.rng.choice(files)
                opp = self._ctl(self.run_dir, os.path.join(pool, name), f"{self.run} {name[:-3]}")
        elif s["kind"] == "ref":
            opp = self._ctl(self.ref)
        elif s["kind"] == "hier":
            opp = self._ctl(self.hier_opp)
        if opp:
            a = self.rng.choice((1, 2))
            s["ctl"][a], s["opp"], s["opp_agent"] = opp, opp["name"], a
        s["env"].set_legacy_obs(0, sum(1 << (a - 1) for a in (1, 2) if s["ctl"][a] and s["ctl"][a]["legacy"]))
        s["goals"] = None
        hs = [s["ctl"][a]["h"] for a in (1, 2) if s["ctl"][a] and s["ctl"][a]["h"]]
        if hs:
            s["goals"] = hs[0]["tw"].Goals(2, self.np.random.default_rng())
        s["t"] = 0
        s["started"] = time.time()

    def _sample(self, m, O):
        net, norm, size = m
        x = self.torch.as_tensor(norm(O[:, :size]), dtype=self.torch.float32)
        with self.torch.no_grad():
            return net.sample_env(x).numpy()

    def _act(self, s):
        """Every FRAME_SKIP frames: flat riders set their sticks; manager riders set a worker target every
        FRAME_SKIP * macro frames and their worker plays the sticks."""
        np, O = self.np, s["obs"]
        hier = [a for a in (1, 2) if s["ctl"][a] and s["ctl"][a]["h"]]
        if hier:
            g, H0 = s["goals"], s["ctl"][hier[0]]["h"]
            if s["t"] % (self.FRAME_SKIP * H0["macro"]) == 0:
                ma = np.zeros((2, 4), np.float32)
                for a in hier:
                    ma[a - 1] = self._sample(s["ctl"][a]["m"], O[a - 1:a])[0]
                H0["he"].set_goals(g, ma, O)
            tw = H0["tw"]
            for a in hier:
                wnet, wnorm, wmirror = s["ctl"][a]["h"]["worker"]
                x = np.concatenate([O, g.feats(O)], 1)
                flip = tw.target_behind(g, O, wmirror == "goal") if wmirror else np.zeros(2, bool)
                x = tw.mirror(x, flip)[:, :len(wnorm.mean)]
                with self.torch.no_grad():
                    w = np.clip(wnet.mean_env(self.torch.as_tensor(wnorm(x), dtype=self.torch.float32)).numpy(), -1, 1)
                w[flip] *= tw.ACT_MIRROR
                s["act"][0, a - 1] = w[a - 1]
            g.left -= 1
        for a in (1, 2):
            c = s["ctl"][a]
            if c and not c["h"]:
                s["act"][0, a - 1] = np.clip(self._sample(c["m"], O[a - 1:a])[0], -1, 1)

    def _frame_view(self, s):
        """The page's view of slot s right now (players, ball, names, targets of manager riders)."""
        st = json.loads(s["env"].json(0))
        m = FarmMatch(s["idx"], None)
        m.feed(st, True)
        v = m.compact()
        names = {a: (s["ctl"][a] or {}).get("name", "?") for a in (1, 2)}
        v.update({"league": s["kind"] == "league", "kind": s["kind"], "opp": s["opp"], "opp_agent": s["opp_agent"],
                  "names": names, "run": self.run, "speed": s["speed"], "last": s["last"], "spectator": True})
        hier = [a for a in (1, 2) if s["ctl"][a] and s["ctl"][a]["h"]]
        if hier and s["goals"] is not None:
            tw = s["ctl"][hier[0]]["h"]["tw"]
            v["targets"] = [t for t in target_view(s["goals"], s["obs"], tw.LIMIT) if t["id"] in hier]
            v["hier"] = True
        return v

    def _record(self, s):
        """Play slot s's current match to the end as fast as possible; returns its recording."""
        frames, actions, every = [], [], 60 // self.REC_FPS
        names = {a: (s["ctl"][a] or {}).get("name", "?") for a in (1, 2)}
        opp, opp_agent = s["opp"], s["opp_agent"]
        frames.append(self._frame_view(s))
        while True:
            if s["t"] % self.FRAME_SKIP == 0:
                self._act(s)
                actions.append(s["act"][0].copy())  # the inputs: the match can be replayed exactly (real game)
            out = s["env"].step(s["act"])
            s["t"] += 1
            s["obs"] = out["obs"][0].copy()
            if out["reset"][0]:  # over (the env already restarted it): the result; pick the next players
                res = {"score": out["score"][0].tolist(), "opp": opp, "opp_agent": opp_agent, "names": names,
                       "secs": round(s["t"] / 60)}
                self._new_game(s)
                return {"frames": frames, "result": res, "actions": np.array(actions, np.float32)}
            if s["t"] % every == 0:
                frames.append(self._frame_view(s))
                if len(frames) % 300 == 0:
                    time.sleep(0.001)  # let the HTTP threads in

    def _produce(self):
        """Keep one recorded match ready per slot (while the tab is watched, or for a slot with nothing yet)."""
        while True:
            busy = False
            for s in self.slots:
                if s["next"] is None and (time.time() - self.viewed_t < 120 or s["rec"] is None):
                    try:
                        s["next"] = self._record(s)
                    except Exception as e:  # noqa: BLE001 — keep the other slots going
                        print(f"spectator {s['idx']}: {e!r}", flush=True)
                        time.sleep(1)
                    busy = True
            if not busy:
                time.sleep(0.5)

    def _replay_views(self):
        out, now = [], time.time()
        for s in self.slots:
            rec = s["rec"]
            done = rec is not None and (now - s["t0"]) * self.REC_FPS * s["speed"] >= len(rec["frames"]) + 2 * self.REC_FPS
            if (rec is None or done) and s["next"] is not None:  # the next recorded match starts playing
                if rec is not None:
                    s["last"] = rec["result"]
                s["rec"], s["next"], s["t0"] = s["next"], None, now
                rec = s["rec"]
            if rec is None:
                continue
            i = min(int((now - s["t0"]) * self.REC_FPS * s["speed"]), len(rec["frames"]) - 1)
            out.append(dict(rec["frames"][i], last=s["last"], replay=True))
        return out

    def _tick(self, s):
        if s["t"] % self.FRAME_SKIP == 0:  # policies act every 4th frame, as in training
            self._act(s)
        out = s["env"].step(s["act"])
        s["t"] += 1
        if out["reset"][0]:  # match over (the env already restarted it): record, pick new players
            names = {a: (s["ctl"][a] or {}).get("name", "?") for a in (1, 2)}
            s["last"] = {"score": out["score"][0].tolist(), "opp": s["opp"], "opp_agent": s["opp_agent"],
                         "names": names, "secs": round(time.time() - s["started"])}
            self._new_game(s)
        s["obs"] = out["obs"][0].copy()
        now = time.time()
        if now - s["view_t"] < 0.025 and not out["reset"][0]:  # the page polls at ~30 Hz
            return
        s["view_t"] = now
        st = json.loads(s["env"].json(0))
        m = FarmMatch(s["idx"], None)
        m.feed(st, True)
        v = m.compact()
        names = {a: (s["ctl"][a] or {}).get("name", "?") for a in (1, 2)}
        v.update({"league": s["kind"] == "league", "kind": s["kind"], "opp": s["opp"], "opp_agent": s["opp_agent"],
                  "names": names, "run": self.run, "speed": s["speed"], "last": s["last"], "spectator": True})
        hier = [a for a in (1, 2) if s["ctl"][a] and s["ctl"][a]["h"]]
        if hier and s["goals"] is not None:
            tw = s["ctl"][hier[0]]["h"]["tw"]
            v["targets"] = [t for t in target_view(s["goals"], s["obs"], tw.LIMIT) if t["id"] in hier]
            v["hier"] = True
        s["view"] = v

    def _loop(self):
        nxt = [time.time()] * len(self.slots)
        rep_t, rep_n, drops = time.time(), [0] * len(self.slots), 0
        while True:
            if time.time() - self.viewed_t > 5:  # nobody watching this tab: pause (CPU and GIL for the rest)
                time.sleep(0.25)
                nxt = [time.time()] * len(self.slots)
                continue
            if time.time() - rep_t > 10 and os.environ.get("SPECTATOR_DEBUG"):
                dt = time.time() - rep_t
                print(f"spectator fps {[round(n / dt) for n in rep_n]} clamp drops {drops}", flush=True)
                rep_t, rep_n, drops = time.time(), [0] * len(self.slots), 0
            i = min(range(len(self.slots)), key=lambda k: nxt[k])
            d = nxt[i] - time.time()
            if d > 0:
                time.sleep(d)
            s = self.slots[i]
            try:
                self._tick(s)
            except Exception as e:  # noqa: BLE001 — keep the other matches running
                print(f"spectator {s['idx']}: {e!r}", flush=True)
                time.sleep(1)
            rep_n[i] += 1
            want = nxt[i] + 1 / (60 * s["speed"])
            drops += want < time.time() - 0.5
            nxt[i] = max(want, time.time() - 0.5)  # don't burst after a stall

    def export(self, idx):
        """Save the inputs of the match slot `idx` is showing (runs/replays/*.npz) so it can be rendered in the
        real game (render_replay.py): actions [policy steps, 2 riders, 3] from tick 0 of a fresh match."""
        s = next((x for x in self.slots if x["idx"] == idx), None)
        rec = s and s["rec"]
        if not rec or "actions" not in rec:
            return {"error": f"no recorded match with inputs in slot {idx} yet"}
        out = os.path.join(HERE, "runs", "replays")
        os.makedirs(out, exist_ok=True)
        res = rec["result"]
        name = time.strftime("%Y%m%d-%H%M%S") + f"_{res['names'][1]}_vs_{res['names'][2]}".replace(" ", "-")
        path = os.path.join(out, name + ".npz")
        np.savez(path, actions=rec["actions"], score=np.array(res["score"]), names=np.array([res["names"][1], res["names"][2]]),
                 frames=len(rec["frames"]), level=1, score_limit=7, minutes=5.0)
        return {"file": path, "score": res["score"], "names": res["names"], "policy_steps": len(rec["actions"])}

    def views(self):
        if self.REPLAY:
            return self._replay_views()
        return [s["view"] for s in self.slots if s["view"]]


class WorkerSpectator(Spectator):
    """Two real-time (1x) matches of the goal-conditioned worker (train_worker.py): both riders chase
    random targets drawn exactly as in its training; the view carries each rider's current target in
    world coordinates so the page can draw it."""

    SLOTS = [("W1", "worker", 1.0), ("W2", "worker", 1.0)]
    REPLAY = False  # live: its targets are redrawn on the fly

    def __init__(self, run_dir, level=1):
        import train_worker as tw
        self.tw, self.Goals, self.LIMIT = tw, tw.Goals, tw.LIMIT
        st = _read_json(os.path.join(run_dir, "status.json")) or {}
        self.mirror = st.get("mirror", False)  # act in the target-ahead frame, as the run was trained
        self.flip_goal = st.get("flip") == "goal"
        self.tally = {"reached": 0, "missed": 0}
        super().__init__(run_dir, level)

    def _new_game(self, s):
        s["nets"] = {1: self._load(os.path.join(self.run_dir, "ckpt.pt"))}
        s["opp"], s["opp_agent"] = None, 0
        if "goals" not in s:
            s["goals"] = self.Goals(2, self.np.random.default_rng())
        s["goals"].sample(self.np.arange(2), s["obs"])
        s["t"], s["started"] = 0, time.time()

    def _tick(self, s):
        np, torch = self.np, self.torch
        g = s["goals"]
        if s["t"] % self.FRAME_SKIP == 0:
            m = s["nets"][1]
            if m:
                net, norm, size = m
                x = np.concatenate([s["obs"], g.feats(s["obs"])], 1)
                flip = self.tw.target_behind(g, s["obs"], self.flip_goal) if self.mirror else np.zeros(2, bool)
                x = self.tw.mirror(x, flip)[:, :size]
                with torch.no_grad():
                    a = np.clip(net.sample_env(torch.as_tensor(norm(x), dtype=torch.float32)).numpy(), -1, 1)
                a[flip] *= self.tw.ACT_MIRROR
                s["act"][0] = a
        out = s["env"].step(s["act"])
        s["t"] += 1
        O2 = out["obs"][0].copy()
        if s["t"] % self.FRAME_SKIP == 0 or out["reset"][0]:  # targets advance per policy step, as in training
            valid = (O2[:, 9] < 0.5) & ~out["reset"][0]
            ok = g.reached(O2) & valid
            g.left -= 1
            g.age += 1
            miss = (g.left <= 0) & ~ok
            self.tally["reached"] += int(ok.sum()); self.tally["missed"] += int(miss.sum())
            redo = np.nonzero(ok | miss | out["reset"][0])[0]
            if out["reset"][0]:
                self._new_game(s)
                redo = []
            g.sample(redo, O2)
        s["obs"] = O2
        now = time.time()
        if now - s["view_t"] < 0.025:
            return
        s["view_t"] = now
        st = json.loads(s["env"].json(0))
        fm = FarmMatch(s["idx"], None)
        fm.feed(st, True)
        v = fm.compact()
        targets = target_view(g, s["obs"], self.LIMIT)
        v.update({"worker": True, "speed": s["speed"], "targets": targets, "tally": dict(self.tally),
                  "spectator": True})
        s["view"] = v


class Hub:
    def __init__(self, run, spectate=True, worker_run="worker1", ref=None, hier_opp=None, legacy=(), overlay=True,
                 metric="goal"):
        self.run_dir = os.path.join(HERE, "runs", run)
        self.run, self.ref, self.overlay, self.metric = run, ref, overlay, metric
        self.worker_dir = os.path.join(HERE, "runs", worker_run) if worker_run else None
        self.worker_run = worker_run
        self.worker_spectator = None
        if spectate and self.worker_dir and os.path.exists(os.path.join(self.worker_dir, "ckpt.pt")):
            try:
                self.worker_spectator = WorkerSpectator(self.worker_dir)
            except Exception as e:  # noqa: BLE001
                print(f"worker spectator off: {e!r}", flush=True)
        self.spectator = None
        self.train_view_t = 0.0  # last time a page asked for the training matches
        self.ev_lock = threading.Lock()
        self.ev = {"off": 0, "counts": {}, "buckets": {}, "recent": collections.deque(maxlen=200),
                   "cur": 0, "vs_recent": {}}
        if spectate:
            try:
                self.spectator = Spectator(self.run_dir, ref=ref, hier_opp=hier_opp, legacy=legacy)
            except Exception as e:  # noqa: BLE001 — no native sim / torch: training matches only
                print(f"spectator matches off: {e!r}", flush=True)
        self.instances = {}
        self.host_conns = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _socks(self):
        try:
            with open(os.path.join(self.run_dir, "status.json")) as f:
                return [(i["idx"], i["sock"]) for i in json.load(f)["instances"]]
        except (OSError, ValueError, KeyError):
            out = []
            for p in sorted(glob.glob(os.path.join(SOCK_DIR, "spr-*.sock"))):
                name = os.path.basename(p)[4:-5]
                if name.isdigit():  # skip ad-hoc sockets like spr-test.sock
                    out.append((int(name), p))
            return out

    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 — never let the poller die
                print("poll error:", repr(e), flush=True)
                time.sleep(1)

    def _tick(self):
        status = _read_json(os.path.join(self.run_dir, "status.json")) or {}
        mode = status.get("mode", "instances")
        if mode != getattr(self, "_mode", None):  # run restarted or switched: drop stale entries
            with self.lock:
                self.instances.clear()
            self._mode = mode
        if mode == "farm":
            return self._tick_farm(status)
        if True:
            t0 = time.time()
            for idx, sock in self._socks():
                with self.lock:
                    inst = self.instances.setdefault(idx, Instance(idx, sock))
                inst.sock = sock
                inst.poll()
            time.sleep(max(0.0, 1 / 15 - (time.time() - t0)))

    def _tick_farm(self, status):
        t0 = time.time()
        valid = {f"{h['idx']}.{k}" for h in status.get("hosts", []) for k in range(h.get("matches", 0))}
        with self.lock:  # the run was restarted with fewer hosts/matches: forget the extras
            for key in [k for k in self.instances if k not in valid]:
                del self.instances[key]
        for host in status.get("hosts", []):
            h, sock = host["idx"], host["sock"]
            conn = self.host_conns.get(h)
            try:
                if conn is None:
                    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    c.settimeout(2.0)
                    c.connect(sock)
                    conn = self.host_conns[h] = (c, c.makefile("r"))
                conn[0].sendall(b"farm state\n")
                matches, alive = json.loads(conn[1].readline()).get("matches", []), True
            except (OSError, ValueError):
                self.host_conns.pop(h, None)
                matches, alive = [None] * host.get("matches", 0), False
            for k, st in enumerate(matches):
                key = f"{h}.{k}"
                with self.lock:
                    m = self.instances.setdefault(key, FarmMatch(key, sock))
                m.feed(st, alive)
        # 10 Hz while someone watches the training matches (?train=1); otherwise they only feed the
        # heatmaps, and every 10 s keeps the parsing (all matches of every host) from stalling the
        # real-time exhibition matches (GIL): at 1 Hz they froze for > 0.5 s several times a minute
        hz = 10 if time.time() - self.train_view_t < 5 or not self.spectator else 0.1
        time.sleep(max(0.0, 1 / hz - (time.time() - t0)))

    def live(self, show_all=False, train=False):
        status = _read_json(os.path.join(self.run_dir, "status.json")) or {}
        farm = status.get("mode") == "farm"
        if train:
            self.train_view_t = time.time()
        if self.spectator and not train:
            self.spectator.viewed_t = time.time()
            return {"t": time.time(), "run": self.run, "farm": farm, "spectator": True,
                    "arena": status.get("arena"),
                    "status": {k: v for k, v in status.items() if k not in ("instances", "matches", "hosts")},
                    "n_matches": len(self.instances), "instances": self.spectator.views()}
        if farm:
            info = {f"{m['host']}.{m['k']}": m for m in status.get("matches", [])}
        else:
            info = {i["idx"]: i for i in status.get("instances", [])}
        with self.lock:
            insts = [self.instances[k] for k in sorted(self.instances, key=_sort_key)]
        # never mix modes: farm pages show farm matches only, instance pages plain instances only
        insts = [i for i in insts if isinstance(i, FarmMatch) == farm]
        if farm and not show_all:  # two matches per host (one self-play, one league) stay readable
            insts = [i for i in insts if str(i.idx).split(".")[-1] in ("0", "1")]
        out = []
        for inst in insts:
            c = inst.compact()
            o = info.get(inst.idx, {})
            c["league"], c["opp"], c["opp_agent"] = o.get("league"), o.get("opp"), o.get("opp_agent")
            c["sd_start"] = o.get("sd_start")
            out.append(c)
        return {"t": time.time(), "run": self.run, "farm": farm, "arena": status.get("arena"),
                "status": {k: v for k, v in status.items() if k not in ("instances", "matches", "hosts")},
                "n_matches": len(self.instances), "instances": out}

    def heat(self):
        with self.lock:
            insts = list(self.instances.values())
        ball = [sum(i.ball_hist[b] for i in insts) for b in range(X_BINS)]
        r1 = [sum(i.rider_hist[1][b] for i in insts) for b in range(X_BINS)]
        r2 = [sum(i.rider_hist[2][b] for i in insts) for b in range(X_BINS)]
        n = sum(i.samples for i in insts) or 1
        air = {a: sum(i.airborne[a] for i in insts) / n for a in (1, 2)}
        return {"range": X_RANGE, "ball": ball, "rider1": r1, "rider2": r2, "airborne": air,
                "samples": n}

    def worker_live(self):
        st = _read_json(os.path.join(self.worker_dir, "status.json")) if self.worker_dir else None
        if self.worker_spectator:
            self.worker_spectator.viewed_t = time.time()
        views = self.worker_spectator.views() if self.worker_spectator else []
        return {"t": time.time(), "run": self.worker_run, "status": st or {}, "instances": views}

    def worker_log(self, max_rows=1500):
        """The worker logs every update (~1/s): thin to at most max_rows points (keep the last row)."""
        rows = []
        try:
            with open(os.path.join(self.worker_dir, "log.csv")) as f:
                rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
        except (OSError, TypeError):
            return []
        if len(rows) > max_rows:
            step = len(rows) / max_rows
            rows = [rows[int(i * step)] for i in range(max_rows - 1)] + [rows[-1]]
        return rows

    def log(self, run_dir=None):
        path = os.path.join(run_dir or self.run_dir, "log.csv")
        rows = []
        try:
            with open(path) as f:
                for r in csv.DictReader(f):
                    rows.append({k: _num(v) for k, v in r.items()})
        except OSError:
            pass
        return rows

    def ref_log(self, max_rows=1500):
        """The reference run's log (e.g. farm1) up to 1.25x this run's env steps, for same-step overlays."""
        if not self.ref or not self.overlay:
            return {"run": None, "rows": []}
        mine = self.log()
        lim = 1.25 * max((r.get("steps") or 0) for r in mine) if mine else 0
        keep = ("steps", "goals_per_ep", "kills_per_ep", "ep_len", "good_side", "enemy_pit", "std", "vs_pool")
        rows = [{k: r.get(k) for k in keep} for r in self.log(os.path.join(HERE, "runs", self.ref))
                if (r.get("steps") or 0) <= lim]
        if len(rows) > max_rows:
            step = len(rows) / max_rows
            rows = [rows[int(i * step)] for i in range(max_rows)]
        return {"run": self.ref, "rows": rows}

    def h2h(self):
        """eval_h2h.py results for this run (runs/<run>/h2h.jsonl), and which number matters for it."""
        out = []
        try:
            with open(os.path.join(self.run_dir, "h2h.jsonl")) as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        return {"metric": self.metric, "rows": out}

    def vs_recent(self):
        """[[update, goals by the latest policy, goals in total]] vs snapshots <= RECENT_AGE updates old."""
        self.events(limit=0)  # brings the incremental parse up to date
        with self.ev_lock:
            return {"age": RECENT_AGE, "metric": self.metric,
                    "rows": [[u, g, n] for u, (g, n) in sorted(self.ev["vs_recent"].items())]}

    def events(self, limit=80):
        """Counts, 10-minute rates and the most recent events. The file grows by hundreds of
        thousands of lines per run, so it is read incrementally (only the bytes appended since the
        last call) — re-parsing it whole would hold the GIL for seconds and stall the real-time
        exhibition matches."""
        path = os.path.join(self.run_dir, "events.jsonl")
        with self.ev_lock:
            ec = self.ev
            try:
                size = os.path.getsize(path)
                if size < ec["off"]:  # truncated / new run: start over
                    ec.update(off=0, counts={}, buckets={}, recent=collections.deque(maxlen=200), cur=0, vs_recent={})
                with open(path, "rb") as f:
                    f.seek(ec["off"])
                    chunk = f.read(size - ec["off"])
                keep = chunk.rfind(b"\n") + 1  # a line still being written waits for the next call
                ec["off"] += keep
                for line in chunk[:keep].splitlines():
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    ec["counts"][e["type"]] = ec["counts"].get(e["type"], 0) + 1
                    b = ec["buckets"].setdefault(int(e["t"] // 600) * 600, {"goal": 0, "kill": 0, "crash": 0})
                    b[e["type"]] = b.get(e["type"], 0) + 1
                    ec["recent"].append(e)
                    opp = e.get("opp")
                    if e["type"] == self.metric and opp and opp.startswith("u") and opp[1:7].isdigit():
                        # goals in league games vs snapshots at most RECENT_AGE updates old, bucketed by the
                        # current update (proxy: the newest snapshot seen so far; snapshots every 25 updates)
                        u = int(opp[1:7])
                        ec["cur"] = max(ec["cur"], u)
                        if ec["cur"] - u <= RECENT_AGE:
                            b = ec["vs_recent"].setdefault(ec["cur"], [0, 0])
                            b[0] += e["winner"] != e.get("opp_agent")  # scored by the latest policy
                            b[1] += 1
            except OSError:
                pass
            rate = [{"t": k, **v} for k, v in sorted(ec["buckets"].items())]
            recent = list(ec["recent"])[-limit:][::-1]
            return {"recent": recent, "counts": dict(ec["counts"]), "rate10m": rate}


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return v


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pbt_state(name):
    """A PBT population (pbt.py): its state.json plus each member's live status (steps, sps, last log row)."""
    st = _read_json(os.path.join(HERE, "runs", name, "state.json"))
    if not st:
        return {}
    for m in st["members"]:
        s = _read_json(os.path.join(HERE, "runs", m["run"], "status.json")) or {}
        m["status"] = {k: s.get(k) for k in ("steps", "sps", "update", "t")}
    return st


# ---------------------------------------------------------------- Gauntlet and Search tabs (gauntlet.py, search.py)
GAUNTLET_DIR = os.path.join(HERE, "runs", "gauntlet")
_rating_cache = {"key": None, "val": None}
_rating_lock = threading.Lock()


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _jsonl(path):
    rows = []
    try:
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # a line being appended right now
    except OSError:
        pass
    return rows


def _tail(path, n=4096):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def gauntlet_ratings():
    """Member Elo (Bradley-Terry over the round robin) and every candidate's Elo with the members held fixed.
    The fit takes ~1 s (torch LBFGS), so it is cached until games/members/candidates change."""
    files = [os.path.join(GAUNTLET_DIR, f) for f in ("games.jsonl", "members.json", "candidates.json")]
    key = tuple(_mtime(f) for f in files)
    with _rating_lock:
        if _rating_cache["key"] != key:
            sys.path.insert(0, HERE)
            import gauntlet
            mem = gauntlet.jload(files[1], {})
            ratings = gauntlet.member_ratings() if mem else {}
            # candidates: the rated ones (candidates.json) and any being rated right now (rows, no entry yet)
            names = set(gauntlet.jload(files[2], {})) | {r["a"] for r in gauntlet.games_log() if r["a"] not in mem}
            cands = {n: gauntlet.rate_candidate(n, ratings)[0] for n in names - set(mem)} if ratings else {}
            _rating_cache.update(key=key, val=(ratings, cands))
        return _rating_cache["val"]


def gauntlet_state():
    """runs/gauntlet: members + Elo, the latest result per pairing (a's view), candidates, exploitability."""
    mem = _read_json(os.path.join(GAUNTLET_DIR, "members.json")) or {}
    if not mem:
        return {}
    ratings, cand_elo = gauntlet_ratings()
    last = {}
    for r in _jsonl(os.path.join(GAUNTLET_DIR, "games.jsonl")):
        last[(r["a"], r["b"])] = r
    keep = ("games", "w", "d", "l", "goal_diff", "kill_diff", "kills_per_min", "deaths_per_min", "t")
    games = [{"a": a, "b": b, **{k: r.get(k) for k in keep}} for (a, b), r in last.items()]
    names = list(mem)
    total = sum(1 for i, x in enumerate(names) for y in names[i + 1:]
                if not (mem[x]["kind"] == "hier" and mem[y]["kind"] == "hier"))
    played = len({frozenset(p) for p in last if p[0] in mem and p[1] in mem})
    cfile = _read_json(os.path.join(GAUNTLET_DIR, "candidates.json")) or {}
    cands = {}
    for n, e in cand_elo.items():
        c = cfile.get(n, {})
        cands[n] = {"steps": c.get("steps"), "elo": e, "t": c.get("t"), "rating": n not in cfile,
                    "played": sum(1 for (a, b) in last if a == n and b in mem)}
    return {"members": {n: {"kind": m["kind"], "steps": m.get("steps"), "run": m.get("run"), "elo": ratings.get(n)}
                        for n, m in mem.items()},
            "games": games, "candidates": cands, "pairings": {"played": played, "total": total},
            "exploit": _jsonl(os.path.join(GAUNTLET_DIR, "exploit.jsonl")),
            "filling": subprocess.run(["pgrep", "-f", "gauntlet.py fill"], capture_output=True).returncode == 0}


def _search_args(name):
    """The running search's --rungs (M env steps) and --eta, or search.py's defaults."""
    out = {"--rungs": "10,30,90", "--eta": "3"}
    for line in subprocess.run(["pgrep", "-fl", "search.py"], capture_output=True, text=True).stdout.splitlines():
        a = line.split()
        if "--name" in a and a.index("--name") + 1 < len(a) and a[a.index("--name") + 1] != name:
            continue
        out.update({k: a[a.index(k) + 1] for k in out if k in a and a.index(k) + 1 < len(a)})
    return [float(x) * 1e6 for x in out["--rungs"].split(",")], float(out["--eta"])


def search_state(name):
    """A hyperparameter search (search.py): its state.json, each trial's status, the driver log's tail."""
    sdir = os.path.join(HERE, "runs", name)
    st = _read_json(os.path.join(sdir, "state.json"))
    try:
        tail = open(os.path.join(sdir, "search.out")).read().splitlines()[-30:]
    except OSError:
        tail = []
    if not st:
        return {"name": name, "started": False, "log": tail}
    rungs, eta = _search_args(name)
    now = time.time()
    played = collections.Counter()  # gauntlet rows per candidate: a rating's progress
    for r in _jsonl(os.path.join(GAUNTLET_DIR, "games.jsonl")):
        played[r["a"]] += 1
    n_members = len(_read_json(os.path.join(GAUNTLET_DIR, "members.json")) or {})
    trials = st.get("trials", {})
    # successive halving as search.py does it: rung r+1 = the best ceil(n/eta) of rung r, once all of rung r is rated
    entered, decided = {}, set()  # trial key -> highest rung it was promoted into; (batch, rung) fully rated
    for b in {t["batch"] for t in trials.values()}:
        keys = [k for k, t in trials.items() if t["batch"] == b]
        for r in range(len(rungs)):
            for k in keys:
                entered[k] = r
            if not all(str(r) in trials[k]["elo"] for k in keys):
                break
            decided.add((b, r))
            done = sorted((k for k in keys if trials[k]["elo"][str(r)] is not None), key=lambda k: -trials[k]["elo"][str(r)])
            keys = done[:max(1, math.ceil(len(done) / eta))]
    for k, t in trials.items():
        rdir = os.path.join(HERE, "runs", t["run"])
        s = _read_json(os.path.join(rdir, "status.json")) or {}
        live = now - _mtime(os.path.join(rdir, "status.json")) < 60  # its trainer is writing status
        t["live"] = {"steps": s.get("steps"), "sps": s.get("sps") if live else None}
        nxt = next((r for r in range(len(rungs)) if str(r) not in t["elo"]), None)
        cand = f"{name}_t{k}@{rungs[nxt] / 1e6:.0f}M" if nxt is not None else None  # its gauntlet name there
        if any(v is None for v in t["elo"].values()):
            t["state"] = "failed"
        elif nxt is None:
            t["state"] = "done"
        elif nxt > entered.get(k, 0):  # rated at its rung: out if the rung is complete, else waiting for the rest
            t["state"] = "pruned" if (t["batch"], nxt - 1) in decided else "waiting"
        elif live:
            t["state"] = "training"
        elif cand in _tail(os.path.join(rdir, "rate.out")):  # gauntlet.py rate prints the name as it starts
            t["state"] = "rating"
            t["rating"] = {"name": cand, "played": played.get(cand, 0), "of": n_members}
        else:
            t["state"] = "queued"  # waiting for a free slot
    mem = _read_json(os.path.join(GAUNTLET_DIR, "members.json")) or {}
    ratings = gauntlet_ratings()[0] if mem else {}
    ref = {m: {"steps": mem[m].get("steps"), "elo": e} for m, e in ratings.items() if m.startswith("farm2@")}
    return {"name": name, "started": True, "rungs": rungs, "eta": eta, "batches": st.get("batches", 0),
            "trials": trials, "ref": ref, "log": tail}


# ---------------------------------------------------------------- Mods tab (prmod.py) and playing against a bot
MODES = ["force_snake", "force_pencil", "force_christmas", "force_chinese_new_year", "unlock_hidden_character",
         "physics_debug"]


def _prmod(*args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "prmod.py"), *args], cwd=HERE, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def mods_state():
    import tomllib
    cfg = tomllib.load(open(os.path.join(HERE, "mods.toml"), "rb"))
    ini = {}
    try:
        sec = None
        for ln in open(os.path.expanduser("~/Library/Application Support/sportsfriends/config.ini")):
            ln = ln.strip()
            if ln.startswith("["):
                sec = ln[1:-1]
            elif sec == "poleriders" and "=" in ln:
                k, v = (x.strip() for x in ln.split("=", 1))
                ini[k] = v
    except OSError:
        pass
    logs = sorted(glob.glob(os.path.join(HERE, "runs", "human", "*.jsonl")))
    frames = 0
    for f in logs:
        try:
            frames += sum(1 for _ in open(f))
        except OSError:
            pass
    bots = sorted(r for r in os.listdir(os.path.join(HERE, "runs"))
                  if os.path.exists(os.path.join(HERE, "runs", r, "ckpt.pt")) and not r.startswith("_"))
    return {"modes": {m: bool(cfg.get("modes", {}).get(m, False)) for m in MODES},
            "score": int(ini.get("scorelimit", 3)), "minutes": int(float(ini.get("duration", 3))),
            "binary": _prmod("status").splitlines()[-1] if True else "",
            "recordings": {"matches": len(logs), "minutes": round(frames / 3600, 1)}, "bots": bots,
            "playing": subprocess.run(["pgrep", "-f", "play_human.py"], capture_output=True).returncode == 0}


def mods_apply(q):
    sys.path.insert(0, HERE)
    import prmod
    path = os.path.join(HERE, "mods.toml")
    for m in MODES:
        prmod.toml_set(path, "modes", m, q.get(m, ["0"])[0] == "1")
    out = [_prmod("apply")]
    out.append(_prmod("match", q.get("score", ["7"])[0], q.get("minutes", ["5"])[0]))
    return {"output": "\n".join(o for o in out if o)}


def play_start(q):
    if subprocess.run(["pgrep", "-f", "play_human.py"], capture_output=True).returncode == 0:
        return {"error": "a game against a bot is already running"}
    cmd = [os.path.join(HERE, ".venv", "bin", "python"), "-u", os.path.join(HERE, "play_human.py"),
           "--bot", q.get("bot", ["farm2"])[0], "--side", q.get("side", ["left"])[0]]
    if q.get("windowed", [""])[0]:
        cmd += ["--windowed", q["windowed"][0]]
    if q.get("record", ["1"])[0] != "1":
        cmd.append("--no-record")
    log = open(os.path.join(HERE, "runs", "play_human.out"), "w")
    subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"ok": True, "cmd": " ".join(cmd[2:])}


def make_handler(hubs, pbt=None, search=None):
    """hubs: {"farm": Hub, "manager": Hub (optional)}; API calls pick one with ?h=farm|manager."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            hub = hubs.get(q.get("h", ["farm"])[0], hubs["farm"])
            if url.path == "/api/runs":
                extra = {"pbt": pbt, "search": search, "gauntlet": os.path.isdir(GAUNTLET_DIR) or None}
                self._send(json.dumps(dict({k: h.run for k, h in hubs.items()}, **{k: v for k, v in extra.items() if v})))
                return
            if url.path in ("/api/mods", "/api/mods/apply", "/api/mods/restore", "/api/play/start", "/api/play/stop"):
                if url.path == "/api/mods/apply":
                    res = mods_apply(q)
                elif url.path == "/api/mods/restore":
                    res = {"output": _prmod("restore")}
                elif url.path == "/api/play/start":
                    res = play_start(q)
                elif url.path == "/api/play/stop":
                    subprocess.run(["pkill", "-f", "play_human.py"]); subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
                    res = {"ok": True}
                else:
                    res = {}
                res["state"] = mods_state()
                self._send(json.dumps(res))
                return
            if url.path == "/api/pbt":
                self._send(json.dumps(pbt_state(pbt) if pbt else {}))
                return
            if url.path == "/api/gauntlet":
                self._send(json.dumps(gauntlet_state()))
                return
            if url.path == "/api/search":
                self._send(json.dumps(search_state(search) if search else {}))
                return
            route = {"/api/heat": hub.heat, "/api/log": hub.log, "/api/events": hub.events,
                     "/api/ref_log": hub.ref_log, "/api/h2h": hub.h2h, "/api/vs_recent": hub.vs_recent,
                     "/api/worker/live": hubs["farm"].worker_live, "/api/worker/log": hubs["farm"].worker_log}
            if url.path == "/api/export":
                sp = hub.spectator
                self._send(json.dumps(sp.export(q.get("slot", ["X1"])[0]) if sp else {"error": "no exhibition matches"}))
                return
            if url.path == "/api/live":
                self._send(json.dumps(hub.live(show_all="all" in q, train="train" in q)))
            elif self.path.startswith("/api/terrain"):
                arena = self.path.split("arena=")[-1] if "arena=" in self.path else "2"
                path = os.path.join(HERE, "arenas", f"arena{int(arena)}.json")
                self._send(open(path).read() if os.path.exists(path) else "{}")
            elif url.path in route:
                self._send(json.dumps(route[url.path]()))
            elif url.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                    self._send(f.read(), "text/html; charset=utf-8")
            else:
                self.send_error(404)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="flatirons")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--worker-run", default="worker2", help="goal-conditioned worker run for the Worker tab")
    p.add_argument("--manager-run", default="manager2", help="hierarchical manager run for the Manager tab")
    p.add_argument("--killer-run", default="killer1", help="kill-only specialist for the Killer tab ('' = none)")
    p.add_argument("--killer-ref", default="farm2", help="the Killer tab's exhibition opponent")
    p.add_argument("--ref", default="farm1",
                   help="flat reference run: its curves overlay the Farm tab's, and an exhibition game plays it")
    p.add_argument("--pbt", default=None, metavar="NAME", help="a PBT population (pbt.py): PBT tab + a tab per member")
    p.add_argument("--search", default=None, metavar="NAME",
                   help="a hyperparameter search (search.py): Search tab (default: search1 if runs/search1 exists)")
    p.add_argument("--tab", action="append", default=[], metavar="KEY=RUN[:REF]",
                   help="an extra game-view tab for run RUN (optionally with a --ref-style comparison run REF)")
    p.add_argument("--no-overlay", dest="overlay", action="store_false",
                   help="don't overlay the --ref run's curves (e.g. a specialist started from it)")
    p.add_argument("--hier-opp", default="manager2",
                   help="manager run (with its own frozen worker) that the Farm tab's latest policy plays")
    p.add_argument("--legacy", default="farm1,manager2",
                   help="runs trained before the dead-rider observation fix (they see dead riders the old way)")
    p.add_argument("--no-spectate", dest="spectate", action="store_false",
                   help="don't run the real-time exhibition matches (show training matches only)")
    a = p.parse_args()
    if a.search is None and os.path.isdir(os.path.join(HERE, "runs", "search1")):
        a.search = "search1"
    legacy = tuple(filter(None, a.legacy.split(",")))
    hubs = {"farm": Hub(a.run, a.spectate, a.worker_run, ref=a.ref if a.ref != a.run else None,
                        hier_opp=a.hier_opp or None,
                        legacy=legacy, overlay=a.overlay)}
    if a.killer_run and os.path.isdir(os.path.join(HERE, "runs", a.killer_run)):
        hubs["killer"] = Hub(a.killer_run, a.spectate, worker_run=None, ref=a.killer_ref, overlay=False,
                             legacy=legacy, metric="kill")
    if a.pbt:  # a game-view tab per member (training charts, exhibitions vs farm2)
        st = _read_json(os.path.join(HERE, "runs", a.pbt, "state.json")) or {}
        a.tab = [f"{m['id']}={m['run']}:farm2" for m in st.get("members", [])] + a.tab
    for spec in a.tab:
        key, run = spec.split("=", 1)
        run, ref = (run.split(":", 1) + [None])[:2]
        hubs[key] = Hub(run, a.spectate, worker_run=None, ref=ref, overlay=bool(ref), legacy=legacy)
    if a.manager_run and os.path.isdir(os.path.join(HERE, "runs", a.manager_run)):
        hubs["manager"] = Hub(a.manager_run, a.spectate, worker_run=None, legacy=legacy)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(hubs, a.pbt, a.search))
    print(f"dashboard on http://localhost:{a.port}: " + ", ".join(f"{k}={h.run}" for k, h in hubs.items())
          + f", worker={a.worker_run}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
