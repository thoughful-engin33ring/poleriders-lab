#!/usr/bin/env python3
"""Bit-for-bit comparison of two builds of the native env library (e.g. the installed sim/build and a
test build in sim/build-dev) over many env configurations.

    compare_libs.py OLD.dylib NEW.dylib [--steps 3000] [--matches 32] [--policy CKPT] [--new-kw JSON]

Each library runs in its own subprocess (ctypes loads one copy per process). Per configuration and step the
SHA-1 of every output array (obs, final, rew, terms, reset, trunc, sd, score, sd_start) is recorded, plus
the snapshot JSON (minus match_id, which the thread pool hands out in any order) of a few matches every 50
steps; the first differing step / key is reported. Actions:
sticky random actions with ball chasing (test_env_equiv.Actions), or a checkpoint's stochastic policy
(--policy; real play with many goals and kills), seeded identically on both sides. --new-kw adds NativeHost
kwargs on the NEW side only (e.g. '{"goal_refund": true}' to show which outputs a feature changes).
"""
import argparse, hashlib, json, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bridge"))
sys.path.insert(0, HERE)

KEYS = ["obs", "final", "rew", "terms", "reset", "trunc", "sd", "score", "sd_start", "touch", "touch_final"]
# (touch rows: only on builds that have them; an empty hash on both sides otherwise)
# (name, NativeHost kwargs, per-step extras) — every training/eval switch the trainers use
CONFIGS = [
    ("default", dict(sd_frac=0.15), {}),
    ("train", dict(end_on_sd=True, match_format=(7, 300)), {}),
    ("death_w", dict(end_on_sd=True, match_format=(7, 300), reward={"kill": 2.0, "death": 1.0}), {}),
    ("rot_pole_spawn", dict(end_on_sd=True, match_format=(7, 300), random_pole=True, goal_spawn=0.5,
                            reward={"rotation": 0.005}), {}),
    ("scramble_pit", dict(sd_frac=0.1, scramble=True, pit_frac=0.3), {}),
    ("legacy_ghost", dict(end_on_sd=True), {"legacy": True}),
]


def run(lib, steps, matches, policy, seed, extra_kw, out):
    import native_env
    native_env.load_lib(lib)
    from native_env import NativeHost
    from test_env_equiv import Actions, PolicyActions
    res = {}
    for name, kw, ext in CONFIGS:
        kw = dict(kw, **extra_kw)
        env = NativeHost(2, matches // 2, 4, max_steps=1500, seed=seed, threads=4, **kw)
        if ext.get("legacy"):
            for k in range(0, env.n, 3):
                env.set_legacy_obs(k, 1 + k % 3)
                env.set_ghost_lock(k + 1, 2) if k + 1 < env.n else None
        obs = env.reset_all()
        act = PolicyActions(policy, seed) if policy else Actions(env.n, seed)
        H, J, ev = [], [], {"goals": 0, "kills": 0, "resets": 0}
        for t in range(steps):
            out_ = env.step(act(obs))
            H.append([hashlib.sha1(np.ascontiguousarray(out_.get(k, np.zeros(0))).tobytes()).hexdigest() for k in KEYS])
            ev["goals"] += int((out_["terms"][:, 0, 0] != 0).sum())
            ev["kills"] += int((out_["terms"][:, :, 1] > 0).sum())
            ev["resets"] += int(out_["reset"].sum())
            if t % 50 == 0:
                js = "".join(env.json(k) for k in range(0, env.n, 7))
                js = re.sub(r'"match_id":\d+', "", js)  # ids are handed out by the thread pool (any order)
                J.append(hashlib.sha1(js.encode()).hexdigest())
            obs = out_["obs"]
        env.close()
        res[name] = {"h": H, "j": J, "ev": ev}
    json.dump(res, open(out, "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old"); p.add_argument("new")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--matches", type=int, default=32)
    p.add_argument("--policy", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--new-kw", default="{}")
    p.add_argument("--worker", nargs=2, metavar=("LIB", "OUT"), help=argparse.SUPPRESS)
    p.add_argument("--extra", default="{}", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        return run(a.worker[0], a.steps, a.matches, a.policy, a.seed, json.loads(a.extra), a.worker[1])
    import tempfile
    d = tempfile.mkdtemp()
    outs = []
    procs = []
    for i, (lib, extra) in enumerate(((a.old, "{}"), (a.new, a.new_kw))):
        o = os.path.join(d, f"{i}.json")
        cmd = [sys.executable, __file__, a.old, a.new, "--worker", os.path.abspath(lib), o, "--steps", str(a.steps),
               "--matches", str(a.matches), "--seed", str(a.seed), "--extra", extra]
        if a.policy:
            cmd += ["--policy", a.policy]
        procs.append(subprocess.Popen(cmd)); outs.append(o)
    for pr in procs:
        if pr.wait():
            raise SystemExit("worker failed")
    A, B = (json.load(open(o)) for o in outs)
    ok = True
    for name, _, _ in CONFIGS:
        ha, hb = A[name]["h"], B[name]["h"]
        bad = [(t, KEYS[i]) for t in range(len(ha)) for i in range(len(KEYS)) if ha[t][i] != hb[t][i]]
        jbad = sum(x != y for x, y in zip(A[name]["j"], B[name]["j"]))
        ev = A[name]["ev"]
        state = "IDENTICAL" if not bad and not jbad else f"DIFFER: first {bad[:3]}, {len(bad)} arrays, {jbad} json"
        ok &= not bad and not jbad
        keys = sorted({k for _, k in bad})
        print(f"{name:15s} {len(ha)} steps x {a.matches} matches, {ev['goals']} goals, {ev['kills']} kill terms, "
              f"{ev['resets']} resets: {state}" + (f" keys {keys}" if keys else ""), flush=True)
    print("ALL IDENTICAL" if ok else "SOME DIFFER")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
