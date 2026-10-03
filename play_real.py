#!/usr/bin/env python3
"""Two trained flat policies play each other in the real game (Sportsfriends + bridge), in real time.

    .venv/bin/python play_real.py --left farm2 --right farm3              # full screen, 7 goals / 5 min
    .venv/bin/python play_real.py --left farm2 --right farm3 --windowed 1280x720

Rider 1 (starts on the left, attacks right) is --left, rider 2 --right. The game runs in real time; each
policy reads the latest state and sets its virtual pad 15 times a second (bridge/farm_env.observe2,
sampled actions as in training). The match format
is set through the bridge (`settings set`) before the match starts; the real game keeps its own rules
(a tie at the end of time goes to sudden death — the agents were trained without it). New matches follow
until ctrl-c; each result is printed.
"""
import argparse, os, subprocess, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))
from farm_env import OBS2_SIZE, Terrain, observe2  # noqa: E402
from spr_env import SOCK_DIR, PoleRidersEnv, launch_game  # noqa: E402
from train_ppo import net_from_ckpt  # noqa: E402


def load(run):
    ck = torch.load(os.path.join(HERE, "runs", run, "ckpt.pt"), weights_only=False)
    net, norm, _ = net_from_ckpt(ck)  # any architecture / action head
    print(f"{run}: {ck['steps'] / 1e6:.0f}M env steps", flush=True)
    return net, norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", default="farm2")
    p.add_argument("--right", default="farm3")
    p.add_argument("--score", type=int, default=7)
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--windowed", default=None, metavar="WxH", help="a window instead of full screen")
    p.add_argument("--size", default="3024x1964", help="full-screen resolution")
    p.add_argument("--arena", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(2)
    pols = {1: load(a.left), 2: load(a.right)}
    names = {1: a.left, 2: a.right}
    subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
    time.sleep(1)
    subprocess.run([sys.executable, os.path.join(HERE, "prmod.py"), "exhibition", "on"], check=True, capture_output=True)
    w, h = (a.windowed or a.size).split("x")
    sock = os.path.join(SOCK_DIR, "spr-show.sock")
    proc = launch_game(sock, mute=False, size=(int(w), int(h)), fullscreen=a.windowed is None)
    time.sleep(16)
    env = PoleRidersEnv(frame_skip=4, fast=0, sock=sock)
    env._connect()
    print("settings:", env.b.cmd(f"settings set {a.score} {a.minutes}"), flush=True)
    env.start_match(arena=a.arena)
    print("in match:", env.b.cmd("settings"), flush=True)
    terrain = Terrain(a.arena)
    env.reset()
    # Real time: the game runs at its own 60 fps (sync off); the policies read the latest state and set the
    # virtual pads 15 times a second (their training rate, frame skip 4). Lockstep (env.step) made the
    # game wait for every observation + both networks, so it ran slow and stuttered.
    env._sync(False)
    st = env.b.state()
    score, period = list(st.get("score", [0, 0])), 4 / 60
    nxt = time.time()
    try:
        while True:
            st = env.b.state()
            if st.get("in_match") and st.get("players"):
                act = {}
                for ag in (1, 2):
                    net, norm = pols[ag]
                    x = torch.as_tensor(norm(np.array([observe2(st, ag, terrain)], np.float32)[:, :len(norm.mean)]),
                                        dtype=torch.float32)
                    with torch.no_grad():
                        lx, rx, ry = env._stick(ag, tuple(net.sample_env(x).clamp(-1, 1).numpy()[0]))
                    env.b.cmd(f"pad set {ag - 1} lx={lx:.4f} rx={rx:.4f} ry={ry:.4f}")
                sc = st.get("score", score)
                if list(sc) != score and sum(sc) > sum(score):
                    print(f"goal: {names[1]} {sc[0]} - {sc[1]} {names[2]}", flush=True)
                score = list(sc)
            if st.get("game_over") or (st.get("in_match") is False):
                print(f"=== match over: {names[1]} {score[0]} - {score[1]} {names[2]} ===", flush=True)
                time.sleep(5)
                env.reset()
                env._sync(False)
                score = [0, 0]
            nxt += period
            time.sleep(max(0.0, nxt - time.time()))
            nxt = max(nxt, time.time() - period)  # don't burst after a stall
    finally:
        env.close()
        proc.kill()


if __name__ == "__main__":
    main()
