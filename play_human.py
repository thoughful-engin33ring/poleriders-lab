#!/usr/bin/env python3
"""A match in the real game between any two players: you on a controller, you (or a friend) on the keyboard, or
trained bots — with human play recorded as training data.

    .venv/bin/python play_human.py --left pad --right bot:farm2           # you (controller) vs farm2
    .venv/bin/python play_human.py --left wasd --right arrows             # two people on one keyboard
    .venv/bin/python play_human.py --left wasd --right bot:pbt1_m1 --windowed 1280x720
    .venv/bin/python play_human.py --bot farm2 --side right               # (old form) controller vs farm2

Player kinds: `pad` / `pad:<n>` (a real controller: the first / n-th one found), `wasd`, `arrows` (the keyboard,
while the game window has focus: A/D or left/right run, W/S or up/down rotate the pole counter-clockwise /
clockwise; see bridge/sprbridge.c apply_keyboard), `bot:<run>` (runs/<run>/ckpt.pt, or a checkpoint path).
The menus are scripted: a human's controller is driven by the script until the match starts (its own input is
ignored meanwhile), bots and keyboard players get virtual pads (`pad keyboard` for the keyboard); bots set theirs
15 times a second in real time (their training rate). After each match a new one starts; close the game (or
ctrl-c) to stop. The game's own mods apply (the patcher: snake, pencil, ...).

Recording (default on): every frame of matches with a human goes to runs/human/<time>_m<match>.jsonl — all pads'
sticks every frame and the state every 4th frame; analysis/playlog.py turns them into (observation, action) pairs.
"""
import argparse, json, math, os, subprocess, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))
from farm_env import Terrain, observe2  # noqa: E402
from spr import Bridge  # noqa: E402
from spr_env import SOCK_DIR, PoleRidersEnv, launch_game  # noqa: E402
from train_ppo import net_from_ckpt  # noqa: E402


def load(name):
    path = name if name.endswith(".pt") else os.path.join(HERE, "runs", name, "ckpt.pt")
    ck = torch.load(path, weights_only=False)
    net, norm, size = net_from_ckpt(ck)  # any architecture / action head
    print(f"bot {name}: {ck.get('steps', 0) / 1e6:.0f}M env steps", flush=True)
    return net, norm, size


def pads(b):
    return json.loads(b.cmd("pads"))["pads"]


def scripted_menus(b, va, vb, arena, slow=False):
    """PoleRidersEnv.start_match's menu script on virtual pads va (left team) and vb (right team). The default
    timings get from team select into a match in ~5 s (tested); slow = the training env's (more margin)."""
    tap, step = (0.8, 2.5) if slow else (0.4, 1.2)  # between stick moves / between menu screens
    for s in (va, vb):
        b.cmd(f"pad connect {s}")
    time.sleep(tap)
    b.cmd(f"pad set {va} lx=-1"); b.cmd(f"pad set {vb} lx=1"); time.sleep(tap)
    b.cmd(f"pad clear {va}"); b.cmd(f"pad clear {vb}"); time.sleep(tap)
    for slots in ((va,), (va, vb)):  # teams, players
        for s in slots:
            b.cmd(f"pad tap {s} a")
        time.sleep(step)
    col, row = (arena - 1) % 2, (arena - 1) // 2
    for btn in ["right"] * col + ["down"] * row:
        b.cmd(f"pad tap {va} {btn}"); time.sleep(tap)
    b.cmd(f"pad tap {va} a")
    for _ in range(100):
        if b.state().get("in_match"):
            return True
        time.sleep(0.1)
    return False


def boot(sock, size, fullscreen, settle):
    """Start the game and wait until its main loop runs (the bridge's frame counter jumps past ~1000 once the
    title is loaded, ~1.5 s after launch) plus `settle` seconds for team select to take input. Returns
    (Popen, Bridge)."""
    proc = launch_game(sock, mute=False, size=size, fullscreen=fullscreen, canon=False)
    t0 = time.time()
    while True:
        if proc.poll() is not None or time.time() - t0 > 60:
            sys.exit("the game didn't start (see ~/Library/Application Support/sportsfriends/logs)")
        try:
            b = Bridge(sock)
            if json.loads(b.cmd("frame"))["frame"] >= 1000:
                break
            b.sock.close()
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.1)
    time.sleep(settle)
    return proc, b


class Placement:
    """`gauntlet` as a player: you play gauntlet members one match at a time and get an Elo on the gauntlet's scale.
    Each match is against the member rated closest to your current estimate (most informative; never the same one
    twice in a row), starting from the strongest. Results go to runs/gauntlet/human.jsonl (one line per match, so a
    placement can be continued later), and an aggregate per opponent to games.jsonl + candidates.json as
    "human:<name>" (the dashboard's Gauntlet tab shows it). A tie at the end of time is a draw (no sudden death)."""

    def __init__(self, player, matches):
        import gauntlet
        self.g, self.name, self.matches = gauntlet, f"human:{player}", matches
        mem = gauntlet.jload(os.path.join(gauntlet.DIR, "members.json"), {})
        self.ratings = gauntlet.member_ratings()
        self.pool = {n: m for n, m in mem.items() if m["kind"] == "flat" and n in self.ratings}
        self.path = os.path.join(gauntlet.DIR, "human.jsonl")
        self.results = []
        try:
            self.results = [r for r in map(json.loads, open(self.path)) if r["player"] == self.name]
        except OSError:
            pass
        self.start = len(self.results)
        self.est, self.se = self.fit()

    def agg(self):
        by = {}
        for r in self.results:
            d = by.setdefault(r["opp"], {"a": self.name, "b": r["opp"], "games": 0, "w": 0, "d": 0, "l": 0, "gd": 0.0})
            d["games"] += 1
            d["gd"] += r["mine"] - r["theirs"]
            d["w" if r["mine"] > r["theirs"] else "d" if r["mine"] == r["theirs"] else "l"] += 1
        for d in by.values():
            d["goal_diff"] = d.pop("gd") / d["games"]
        return list(by.values())

    def fit(self):
        rows = self.agg()
        if not rows:
            return max(self.ratings[n] for n in self.pool), None
        fixed = {r["b"]: self.ratings[r["b"]] for r in rows}
        elo = self.g.fit(rows, [self.name] + list(fixed), fixed=fixed)[self.name]
        k = math.log(10) / 400  # standard error from the fit's curvature (Fisher information + the prior)
        info = 1 / self.g.PRIOR_SD ** 2 + sum(r["games"] * k * k * (p := 1 / (1 + 10 ** ((fixed[r["b"]] - elo) / 400))) * (1 - p)
                                             for r in rows)
        return elo, 1 / math.sqrt(info)

    def next_opponent(self):
        last = self.results[-1]["opp"] if self.results else None
        return min((n for n in self.pool if n != last), key=lambda n: abs(self.ratings[n] - self.est))

    def done(self):
        return len(self.results) - self.start >= self.matches

    def record(self, opp, mine, theirs):
        r = {"player": self.name, "opp": opp, "opp_elo": self.ratings[opp], "mine": mine, "theirs": theirs,
             "t": time.time()}
        self.results.append(r)
        with open(self.path, "a") as f:
            f.write(json.dumps(r) + "\n")
        self.est, self.se = self.fit()
        row = next(x for x in self.agg() if x["b"] == opp)
        with open(os.path.join(self.g.DIR, "games.jsonl"), "a") as f:  # cumulative per opponent (latest wins)
            f.write(json.dumps(dict(row, seed=0, t=time.time())) + "\n")
        entry = {"name": self.name, "kind": "human", "path": "", "steps": 0, "elo": self.est, "se": self.se,
                 "matches": len(self.results), "t": time.time()}
        self.g.jupdate(os.path.join(self.g.DIR, "candidates.json"), lambda c: c.__setitem__(self.name, entry))
        print(f"  {self.name} {mine}-{theirs} vs {opp} (Elo {self.ratings[opp]:.0f}) -> your Elo {self.est:.0f} "
              f"± {self.se:.0f} after {len(self.results)} matches", flush=True)


def freeze(run):
    """A series opponent: `t27` = runs/search1_t27 (else runs/<run> or a .pt path); its current checkpoint is copied to
    runs/human/opponents/ so the session plays (and the log names) exactly that version. Returns (label, path)."""
    import shutil
    if not run.endswith(".pt") and not os.path.isdir(os.path.join(HERE, "runs", run)) and \
            os.path.isdir(os.path.join(HERE, "runs", "search1_" + run)):
        run = "search1_" + run
    src = run if run.endswith(".pt") else os.path.join(HERE, "runs", run, "ckpt.pt")
    steps = int(torch.load(src, weights_only=False).get("steps", 0))
    short = os.path.basename(os.path.dirname(src)).replace("search1_", "") if not run.endswith(".pt") else \
        os.path.basename(src)[:-3]
    dst = os.path.join(HERE, "runs", "human", "opponents", f"{short}_{steps // 1_000_000}M.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
    return f"{short} ({steps / 1e6:.0f}M steps)", dst


def hud_names(names, spec, extra=""):
    who = {"pad": "you", "wasd": "you (WASD)", "arrows": "you (arrows)"}
    lbl = {ag: who.get(spec[ag], "you" if spec[ag].startswith("pad") else names[ag]) for ag in (1, 2)}
    return f"{lbl[1]}   vs   {lbl[2]}" + (f"   ·   {extra}" if extra else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", default=None, help="pad | pad:<n> | wasd | arrows | bot:<run> (left rider, attacks right)")
    p.add_argument("--right", default=None, help="the same for the right rider (attacks left)")
    p.add_argument("--bot", default=None, help="(old form) the bot's run; you play --side on your controller")
    p.add_argument("--side", choices=["left", "right"], default="left", help="(old form) your side")
    p.add_argument("--windowed", default=None, metavar="WxH")
    p.add_argument("--size", default="3024x1964", help="full-screen resolution")
    p.add_argument("--score", type=int, default=None, help="override the score limit for this session")
    p.add_argument("--minutes", type=float, default=None, help="override the round length for this session")
    p.add_argument("--no-record", dest="record", action="store_false")
    p.add_argument("--arena", type=int, default=2)
    p.add_argument("--matches", type=int, default=8, help="gauntlet: matches to play (placement)")
    p.add_argument("--player", default="you", help="gauntlet: your name in the ratings")
    a = p.parse_args()
    if not (a.left or a.right):  # the old form
        bot = f"bot:{a.bot or 'farm2'}"
        a.left, a.right = ("pad", bot) if a.side == "left" else (bot, "pad")
    spec = {1: a.left or "bot:farm2", 2: a.right or "bot:farm2"}
    for sp in spec.values():
        if not (sp.startswith(("bot:", "series:")) or sp in ("wasd", "arrows", "gauntlet") or sp == "pad"
                or sp.startswith("pad:")):
            sys.exit(f"unknown player {sp!r} (pad | pad:<n> | wasd | arrows | bot:<run> | series:<run>,<run>.. | gauntlet)")
    torch.set_num_threads(2)
    sys.path.insert(0, HERE)
    place, g_ag = None, next((ag for ag, sp in spec.items() if sp == "gauntlet"), None)
    if g_ag:  # rate the human on the other side against the gauntlet
        if spec[3 - g_ag].startswith("bot:") or spec[3 - g_ag] == "gauntlet":
            sys.exit("gauntlet: the other side must be a human (pad / wasd / arrows)")
        place = Placement(a.player, a.matches)
        opp = place.next_opponent()
        spec[g_ag] = "bot:" + opp
        a.score, a.minutes, a.record = 7, 5, True  # the gauntlet's format; your games are training data too
        print(f"placement: {a.matches} matches; start estimate {place.est:.0f}; first opponent {opp}", flush=True)
    series, s_ag, s_i = None, next((ag for ag, sp in spec.items() if sp.startswith("series:")), None), 0
    if s_ag:  # one match against each bot in turn (then round again); their checkpoints frozen now
        series = [freeze(r) for r in spec[s_ag][7:].split(",") if r]
        spec[s_ag] = "bot:" + series[0][1]
        print("series: " + ", ".join(lbl for lbl, _ in series), flush=True)
    bots = {ag: load(place.pool[sp[4:]]["path"] if place and ag == g_ag else sp[4:])
            for ag, sp in spec.items() if sp.startswith("bot:")}
    names = {ag: sp[4:] if sp.startswith("bot:") else sp for ag, sp in spec.items()}
    if series:
        names[s_ag] = series[0][0]
    logdir = os.path.join(HERE, "runs", "human")
    os.makedirs(logdir, exist_ok=True)
    subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
    time.sleep(1)
    subprocess.run([sys.executable, os.path.join(HERE, "prmod.py"), "exhibition", "on"], check=True, capture_output=True)
    w, h = (int(v) for v in (a.windowed or a.size).split("x"))
    sock = os.path.join(SOCK_DIR, "spr-human.sock")
    proc = None
    try:
        for slow in (False, True):  # fast menu timings; if they miss, relaunch once with the safe ones
            proc, b = boot(sock, (w, h), a.windowed is None, settle=4.0 if slow else 2.0 + (a.windowed is None))
            want = max([int(sp[4:]) if sp.startswith("pad:") else 1 for sp in spec.values() if sp.startswith("pad")] + [0])
            real = [q["slot"] for q in pads(b) if q["connected"] and not q["virtual"]]
            while len(real) < want:
                print("waiting for your controller — plug it in (or turn it on)…", flush=True)
                time.sleep(2)
                real = [q["slot"] for q in pads(b) if q["connected"] and not q["virtual"]]
            # a human on a controller plays on that controller itself; the menu script drives it meanwhile
            # (`pad override`: the human's own input is ignored until the match starts, so a touched button or a
            # drifting stick can't put a third controller on a team); bots and keyboards get virtual pads
            slot = {ag: real[int(sp[4:]) - 1 if sp.startswith("pad:") else 0]
                    for ag, sp in spec.items() if sp.startswith("pad")}
            free = [q["slot"] for q in pads(b) if not q["connected"]]
            need = [ag for ag in (1, 2) if ag not in slot]
            if len(free) < len(need):
                sys.exit("not enough free controller slots for the bots / keyboard players")
            slot.update(zip(need, free))
            for ag, sp in spec.items():
                if sp.startswith("pad"):
                    b.cmd(f"pad override {slot[ag]} on")
            print(f"left: {names[1]} (pad {slot[1]}) vs right: {names[2]} (pad {slot[2]})", flush=True)
            if a.score or a.minutes:
                print(b.cmd(f"settings set {a.score or 7} {a.minutes or 5}"), flush=True)
            if scripted_menus(b, slot[1], slot[2], a.arena, slow):
                break
            if slow:
                sys.exit("the menu script didn't reach a match (take a screenshot and adjust)")
            print("the menus didn't reach a match: relaunching with slower menu timings", flush=True)
            proc.kill()
            proc.wait()
        for ag, sp in spec.items():  # humans take over their pads
            if sp.startswith("pad"):
                print(b.cmd(f"pad override {slot[ag]} off"), flush=True)
            elif sp in ("wasd", "arrows"):
                print(b.cmd(f"pad keyboard {slot[ag]} {sp}"), flush=True)
        if a.record and len(bots) < 2:
            print(b.cmd(f"playlog on {logdir}"), flush=True)
        terrain = Terrain(a.arena)
        series_log = os.path.join(HERE, "runs", "human", "series.jsonl")

        def show():
            extra = f"match {s_i + 1} of {len(series)}" if series else \
                (f"rating match {len(place.results) - place.start + 1} of {place.matches}" if place else "")
            b.cmd("hud " + hud_names(names, spec, extra))
        show()
        print(f"{names[1]} vs {names[2]}. Go!", flush=True)
        period, nxt = 4 / 60, time.time()
        # match end: the game's game_over flag, or (it may never be seen: the game-over menu is not a match) the match
        # disappearing; in a placement or a series a tie at the end of time (sudden death) also ends it, as a draw
        last, ended, over_t = None, set(), None
        bot_slots = [slot[ag] for ag in bots]
        while proc.poll() is None:
            st = b.state()
            playing = st.get("in_match") and st.get("players")
            sd = bool((place or series) and st.get("sudden_death"))
            if playing and not st.get("game_over") and not sd:
                for ag, (net, norm, size) in bots.items():
                    x = torch.as_tensor(norm(np.array([observe2(st, ag, terrain)], np.float32)[:, :size]),
                                        dtype=torch.float32)
                    with torch.no_grad():
                        act = net.sample_env(x).clamp(-1, 1).numpy()[0]
                    lx, rx, ry = PoleRidersEnv._stick(ag, tuple(act))
                    b.cmd(f"pad set {slot[ag]} lx={lx:.4f} rx={rx:.4f} ry={ry:.4f}")
            if playing:
                last = (st.get("match_id"), list(st.get("score", [0, 0])))
                if last[0] not in ended:
                    over_t = None  # the next match is on
            over = playing and (st.get("game_over") or sd) or (not st.get("in_match") and last is not None)
            if over and last and last[0] not in ended:
                ended.add(last[0])
                over_t = time.time()
                sc = last[1]
                print(f"match over: {names[1]} {sc[0]} - {sc[1]} {names[2]}" + (" (draw at time)" if sd else ""),
                      flush=True)
                for s_ in bot_slots:
                    b.cmd(f"pad clear {s_}")
                if series:
                    with open(series_log, "a") as f:
                        f.write(json.dumps({"opp": names[s_ag], "path": series[s_i % len(series)][1],
                                            "mine": sc[2 - s_ag], "theirs": sc[s_ag - 1], "t": time.time()}) + "\n")
                    s_i += 1
                    lbl, path = series[s_i % len(series)]
                    bots[s_ag] = load(path)
                    names[s_ag] = lbl
                    print(f"next: {lbl}", flush=True)
                    show()
                if place:
                    place.record(names[g_ag], sc[2 - g_ag], sc[g_ag - 1])
                    if place.done():
                        print(f"placement done: {place.name} Elo {place.est:.0f} ± {place.se:.0f} "
                              f"(farm2 latest = 1500)", flush=True)
                        time.sleep(5)
                        break
                    opp = place.next_opponent()
                    bots[g_ag] = load(place.pool[opp]["path"])
                    names[g_ag] = opp
                    print(f"next: {opp} (Elo {place.ratings[opp]:.0f})", flush=True)
                    show()
            if over_t is not None and time.time() - over_t > 5:  # then the next match
                if playing:  # sudden death (a placement draw): restart the match in place
                    b.cmd("reset")
                elif bot_slots:  # the game-over menu: its first item is REMATCH (same teams, same arena)
                    b.cmd(f"pad tap {bot_slots[0]} a")
                over_t = time.time()  # again in 5 s unless the next match has started
            nxt += period
            time.sleep(max(0.0, nxt - time.time()))
            nxt = max(nxt, time.time() - period)
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None:
            proc.kill()


if __name__ == "__main__":
    main()
