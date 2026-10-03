#!/usr/bin/env python3
"""Render a recorded exhibition match (dashboard /api/export -> runs/replays/*.npz) in the real game to MP4.

The native simulator is bit-exact with the game, so the match's inputs replayed from tick 0 of a fresh match
reproduce it exactly. The game runs in lockstep (sync mode): every simulated frame is rendered once and
captured by the bridge (`record start`: back buffer -> ffmpeg, H.264 at 60 fps), so the video is smooth and
real time whatever the machine's speed. At the end the final score is checked against the recording's.

    .venv/bin/python render_replay.py runs/replays/<name>.npz [more.npz ...] [--size 1920x1080] [--out file.mp4]
        [--realtime] [--hud "text"]

--realtime paces the lockstep to the game clock (60 ticks/s), so it can be watched live; several replays play one
after another in one game session (each to its own MP4).
"""
import argparse, json, os, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))
from spr_env import SOCK_DIR, PoleRidersEnv, launch_game  # noqa: E402


def render(b, env, replay, a, n):
    R = np.load(replay)
    acts, names = R["actions"], [str(x) for x in R["names"]]
    out = os.path.abspath(a.out if a.out and len(a.replay) == 1 else replay.replace(".npz", ".mp4"))
    if a.hud:
        b.cmd("hud " + a.hud.format(i=n, n=len(a.replay)))
    # a fresh match caught at tick 0 (the creation frame doesn't tick), as the native recording started
    old = json.loads(b.cmd("step 1"))["match_id"]
    b.cmd("reset")
    for _ in range(3000):
        st = json.loads(b.cmd("step 1"))
        if st.get("in_match") and st["match_id"] != old and st.get("players"):
            break
    print(f"{names[0]} vs {names[1]}: {len(acts)} policy steps, recording -> {out}", flush=True)
    print(b.cmd(f"record start {out}"), flush=True)
    score = st.get("score", [0, 0])
    key = lambda st: [(p["x"], p["y"], p["vx"], p["vy"]) for p in st["players"]] + [st["ball"]["x"], st["ball"]["y"]]
    sticks = lambda a1, a2: " ".join(f"{v:.4f}" for v in env._stick(1, tuple(a1)) + env._stick(2, tuple(a2)))
    # After a restart the real match is frozen while the loading screen fades in (the simulator's match isn't):
    # feed the first input frame by frame until the state changes — that frame was its first real tick, with
    # input 0 as in the recording — then the rest of that 4-tick policy step.
    k0, frozen = key(st), 0
    while frozen < 600:
        st = json.loads(b.cmd("act 1 " + sticks(*acts[0])))
        if key(st) != k0:
            break
        frozen += 1
        if a.realtime:
            time.sleep(1 / 60)
    st = json.loads(b.cmd("act 3 " + sticks(*acts[0])))
    print(f"kickoff after {frozen} frozen frames", flush=True)
    t_next = time.time()
    for i, (a1, a2) in enumerate(acts[1:], start=1):
        st = json.loads(b.cmd("act 4 " + sticks(a1, a2)))
        if a.realtime:  # 4 ticks = 4/60 s of game time
            t_next += 4 / 60
            time.sleep(max(0.0, t_next - time.time()))
        if st.get("score") and st["score"] != score:
            score = st["score"]
            print(f"  {i * 4 / 60:6.1f} s  {names[0]} {score[0]} - {score[1]} {names[1]}", flush=True)
        if st.get("game_over"):
            break
    for _ in range(90):  # a moment of the end screen
        b.cmd("step 2")
        if a.realtime:
            time.sleep(2 / 60)
    print(b.cmd("record stop"), flush=True)
    exp = R["score"].tolist()
    print(f"final {score} vs recorded {exp}: {'IDENTICAL' if list(score) == exp else 'DIFFERENT'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("replay", nargs="+")
    p.add_argument("--size", default="3024x1964", help="full-screen resolution (the video is scaled to 1920 wide)")
    p.add_argument("--windowed", default=None, metavar="WxH", help="a window instead of full screen")
    p.add_argument("--out", default=None, help="(one replay only)")
    p.add_argument("--realtime", action="store_true", help="pace to the game clock: watchable live")
    p.add_argument("--hud", default=None, help="a label over the bottom of the screen ({i}/{n}: replay index/count)")
    a = p.parse_args()
    w, h = (int(v) for v in (a.windowed or a.size).split("x"))
    subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
    time.sleep(1)
    subprocess.run([sys.executable, os.path.join(HERE, "prmod.py"), "exhibition", "on"], check=True, capture_output=True)
    sock = os.path.join(SOCK_DIR, "spr-render.sock")
    proc = launch_game(sock, mute=False, size=(w, h), fullscreen=a.windowed is None)
    try:
        time.sleep(16)
        env = PoleRidersEnv(frame_skip=4, fast=0, sock=sock)
        env._connect()
        R = np.load(a.replay[0])
        for q in json.loads(env.b.cmd("pads"))["pads"][:2]:  # a real controller in slot 0/1: scripted, its input ignored
            if q["connected"] and not q["virtual"]:
                print(env.b.cmd(f"pad override {q['slot']} on"), flush=True)
        env.b.cmd(f"settings set {int(R['score_limit'])} {float(R['minutes'])}")
        env.start_match(arena=2)
        b = env.b
        for i in (0, 1):
            b.cmd(f"pad clear {i}")
        b.cmd("sync on")
        for n, replay in enumerate(a.replay, start=1):
            render(b, env, replay, a, n)
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
