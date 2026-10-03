"""One farm host per process: owns a game instance (K shadow matches) and does all the per-step
Python work for it — the socket round trip, JSON parsing, observations, rewards, resets and crash
recovery — so the trainer process only runs the policy and PPO. Talks over a multiprocessing Pipe:

    -> ("step", actions float32[K, 2, 3])
    <- ("ok", {"obs", "final", "rew", "terms", "reset", "trunc", "sd", "score", "sd_start"})
    <- ("crash", {"obs", "sd_start", "why", "returncode"})      # relaunched; every match restarted
    -> ("close", None)
"""
import os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from farm_env import OBS2_SIZE, FarmHost  # noqa: E402
from spr_env import launch_game  # noqa: E402

SIZE = (320, 180)


def host_worker(conn, i, sock, cfg):
    K, D = cfg["matches"], OBS2_SIZE
    time.sleep(1.5 * i)  # stagger game launches
    proc = [launch_game(sock, size=SIZE)]
    time.sleep(16)
    host = FarmHost(sock, K, cfg["arena"], cfg["frame_skip"], sudden_death_frac=cfg["sd_frac"],
                    max_steps=cfg["episode_steps"], seed=i, goal_refund=cfg.get("goal_refund", False))

    def relaunch():
        proc[0].kill()
        proc[0] = launch_game(sock, size=SIZE)
        time.sleep(16)
        host.env.b, host.env.synced = None, False

    def boot():
        for attempt in range(3):
            try:
                host.start()
                return np.array([[o[1], o[2]] for o in host.reset_all()], np.float32)
            except Exception:  # noqa: BLE001
                relaunch()
        raise RuntimeError(f"host {i} failed to boot")

    conn.send(("ready", {"obs": boot(), "sd_start": np.array(host.sd_start)}))
    while True:
        cmd, acts = conn.recv()
        if cmd == "close":
            proc[0].kill()
            return
        try:
            out = step_out(host, acts)
            conn.send(("ok", out))
        except Exception as e:  # noqa: BLE001 — dead or hung game: relaunch, restart every match
            rc = proc[0].poll()
            relaunch()
            obs = boot()
            conn.send(("crash", {"obs": obs, "sd_start": np.array(host.sd_start), "why": repr(e)[:300],
                                 "returncode": rc}))


def step_out(host, acts):
    """One FarmHost step for actions float32[K, 2, 3] -> the `out` dict sent to the trainer (matches
    that ended or hit max_steps are reset: their `obs` is the new match's, `final` the last state's)."""
    K, D = host.n, OBS2_SIZE
    o_list, r_list, d_list, i_list = host.step(
        [(tuple(acts[k, 0]), tuple(acts[k, 1])) for k in range(K)])
    out = {"obs": np.zeros((K, 2, D), np.float32), "final": np.zeros((K, 2, D), np.float32),
           "rew": np.zeros((K, 2), np.float32), "terms": np.zeros((K, 2, 2), np.float32),
           "reset": np.zeros(K, bool), "trunc": np.zeros(K, bool), "sd": np.zeros(K, bool),
           "score": np.zeros((K, 2), np.int32), "sd_start": np.zeros(K, bool)}
    for k in range(K):
        o, r, d, info = o_list[k], r_list[k], d_list[k], i_list[k]
        st = info["state"]
        out["rew"][k] = (r[1], r[2])
        out["terms"][k] = [[info["terms"][a]["goal"], info["terms"][a]["kill"]] for a in (1, 2)]
        out["sd"][k] = bool(st.get("sudden_death"))
        if st.get("score"):
            out["score"][k] = st["score"]
        if o is not None:
            out["final"][k] = (o[1], o[2])  # for time-limit bootstrapping
        if d or info["truncated"] or o is None:
            out["reset"][k] = True
            out["trunc"][k] = bool(info["truncated"]) and o is not None
            n = host.reset(k)
            out["obs"][k] = (n[1], n[2])
        else:
            out["obs"][k] = (o[1], o[2])
        out["sd_start"][k] = host.sd_start[k]
    return out
