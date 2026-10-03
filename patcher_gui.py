#!/usr/bin/env python3
"""Pole Riders patcher GUI: every prmod.py setting in a window (your browser), plus launching the game and playing
against a trained bot. Python standard library only.

    python3 prmod.py gui            # or double-click "Pole Riders Patcher.command"

Edits mods.toml (only values that differ from vanilla are written uncommented) and runs the same apply / restore as
the command line; the match format goes to the game's config.ini (prmod.py match).
"""
import contextlib, io, json, os, subprocess, sys, threading, tomllib, urllib.parse, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prmod  # noqa: E402

CFG = os.path.join(HERE, "mods.toml")
PORT = 8766
VENV_PY = os.path.join(HERE, ".venv", "bin", "python")
SECTION_TITLES = {"gameplay": "Gameplay", "ball": "Ball", "player": "Riders", "pole": "Poles",
                  "level1": "Level 1 (menu arena 2, the training map)", "level2": "Level 2 (menu arena 4)",
                  "level3": "Level 3", "level4": "Level 4", "level5": "Level 5 (Rainbow Highway)"}


def capture(fn, *a):
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            fn(*a)
    except SystemExit as e:  # prmod reports errors through sys.exit(message)
        out.write(str(e))
    return out.getvalue().strip()


def current():
    """The settings in mods.toml (absent = vanilla) and the match format in config.ini."""
    floats, flags, limits, gravity = prmod.load_config(CFG) if os.path.exists(CFG) else ({}, {}, {}, None)
    ini = {}
    try:
        sec = None
        for ln in open(prmod.CONFIG_INI):
            ln = ln.strip()
            if ln.startswith("["):
                sec = ln[1:-1]
            elif sec == "poleriders" and "=" in ln:
                k, v = (x.strip() for x in ln.split("=", 1))
                ini[k] = v
    except OSError:
        pass
    params = [{"key": "gameplay.gravity", "section": "gameplay", "name": "gravity", "vanilla": prmod.GRAVITY_DEFAULT,
               "value": gravity if gravity is not None else prmod.GRAVITY_DEFAULT,
               "desc": "downward acceleration (Box2D units)"}]
    for key, (_, default, desc) in prmod.FLOATS.items():
        sec, name = key.split(".")
        params.append({"key": key, "section": sec, "name": name, "vanilla": default,
                       "value": floats.get(key, default), "desc": desc})
    modes = [{"key": k, "desc": desc, "on": bool(flags.get(k, False))} for k, (desc, _) in prmod.FLAGS.items()]
    bots = []
    runs = os.path.join(HERE, "runs")
    if os.path.isdir(runs) and os.path.exists(VENV_PY):
        bots = sorted(r for r in os.listdir(runs) if not r.startswith("_") and os.path.exists(os.path.join(runs, r, "ckpt.pt")))
    human = os.path.join(runs, "human")
    recs = sorted(f for f in os.listdir(human) if f.endswith(".jsonl")) if os.path.isdir(human) else []
    frames = sum(sum(1 for _ in open(os.path.join(human, f))) for f in recs)
    return {"params": params, "modes": modes, "titles": SECTION_TITLES,
            "score": int(float(ini.get("scorelimit", 3))), "minutes": int(float(ini.get("duration", 3))),
            "binary": next((ln for ln in capture(prmod.cmd_status).splitlines() if "installed binary" in ln), "?")
                      if os.path.exists(prmod.BIN) else "game not found",
            "game_running": subprocess.run(["pgrep", "-f", "MacOS/Sportsfriends"], capture_output=True).returncode == 0,
            "bots": bots, "recordings": {"matches": len(recs), "minutes": round(frames / 3600, 1)}}


def write_config(params, modes):
    """mods.toml from scratch: changed values uncommented, everything else commented out at its vanilla value."""
    out = ["# Super Pole Riders mods (written by the patcher GUI; editing by hand works too). Commented = vanilla.",
           "# Apply: python3 prmod.py apply   (or the GUI)", ""]
    lim = prmod.LIMITS
    out.append("[limits]   # raised automatically by the match format when you go above 7 goals / 5 min")
    for key, (_, mdef, *_r, desc) in lim.items():
        out.append(f"# {key} = {mdef:g}   # {desc}")
    sec = None
    for p in params:
        if p["section"] != sec:
            sec = p["section"]
            out += ["", f"[{sec}]"]
        changed = abs(float(p["value"]) - p["vanilla"]) > 1e-6
        line = f"{p['name']} = {float(p['value']):g}" if changed else f"# {p['name']} = {p['vanilla']:g}"
        out.append(line + (f"   # {p['desc']}" if p["desc"] else ""))
    out += ["", "[modes]"]
    for k, (desc, _) in prmod.FLAGS.items():
        on = bool(modes.get(k))
        out.append(f"{k} = true   # {desc}" if on else f"# {k} = false   # {desc}")
    open(CFG, "w").write("\n".join(out) + "\n")


def launch(q):
    w = q.get("window", ["1280x720"])[0]
    args = ["-windowed", "0", "-width", "3024", "-height", "1964"] if w == "full" else \
        ["-windowed", "1", "-width", w.split("x")[0], "-height", w.split("x")[1]]
    args.append("--bridge")  # so keyboard gamepads can be added (and the state read)
    subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
    return capture(prmod.cmd_run, args)


def keyboard_pad(q):
    """Add a keyboard-driven virtual gamepad to the running game (launched with the bridge)."""
    sys.path.insert(0, os.path.join(HERE, "bridge"))
    from spr import SOCK, Bridge
    try:
        b = Bridge(SOCK)
    except OSError:
        return "start the game with Launch first (it loads the bridge the keyboard pads need)"
    free = [p["slot"] for p in json.loads(b.cmd("pads"))["pads"] if not p["connected"]]
    if not free:
        return "all 4 controller slots are taken"
    mode = q.get("mode", ["wasd"])[0]
    r = json.loads(b.cmd(f"pad keyboard {free[0]} {mode}"))
    return r.get("error") or f"keyboard pad ({mode}) added as controller {free[0] + 1} — click into the game window"


def play(q):
    if not os.path.exists(VENV_PY):
        return "matches with bots need the training environment (.venv) of this project"
    subprocess.run(["pkill", "-f", "play_human.py"])
    left, right = q.get("left", ["pad"])[0], q.get("right", ["bot:farm2"])[0]
    cmd = [VENV_PY, "-u", os.path.join(HERE, "play_human.py"), "--left", left, "--right", right]
    w = q.get("window", ["full"])[0]
    if w != "full":
        cmd += ["--windowed", w]
    if q.get("record", ["1"])[0] != "1":
        cmd.append("--no-record")
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    subprocess.Popen(cmd, cwd=HERE, stdout=open(os.path.join(HERE, "runs", "play_human.out"), "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    who = lambda x: x[4:] if x.startswith("bot:") else {"pad": "your controller", "wasd": "keyboard WASD",
                                                         "arrows": "keyboard arrows",
                                                         "gauntlet": "the gauntlet (8 rating matches)"}.get(x, x)
    return f"starting {who(left)} vs {who(right)} — the menus are scripted, don't press anything until the match starts"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json"):
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
        msg = None
        if url.path in ("/", "/index.html"):
            return self.send(PAGE, "text/html; charset=utf-8")
        if url.path == "/api/apply":
            d = json.loads(q["json"][0])
            write_config(d["params"], d["modes"])
            msg = capture(prmod.cmd_apply, CFG) + "\n" + capture(prmod.cmd_match, d["score"], d["minutes"])
        elif url.path == "/api/restore":
            msg = capture(prmod.cmd_restore)
        elif url.path == "/api/launch":
            msg = launch(q)
        elif url.path == "/api/play":
            msg = play(q)
        elif url.path == "/api/kbd":
            msg = keyboard_pad(q)
        elif url.path == "/api/stop":
            subprocess.run(["pkill", "-f", "play_human.py"])
            subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
            msg = "game closed"
        elif url.path != "/api/state":
            return self.send_error(404)
        self.send(json.dumps({"state": current(), "message": msg}))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pole Riders Patcher</title>
<style>
:root { --bg: #f6f4ef; --panel: #ffffff; --line: #e3ded4; --text: #23201b; --muted: #7a7366; --accent: #d9480f;
        --accent2: #0b7285; --ok: #2b8a3e; --chip: #f1ece2; }
@media (prefers-color-scheme: dark) { :root { --bg: #16161a; --panel: #1f1f25; --line: #2e2e36; --text: #ecebe7;
        --muted: #9a978f; --accent: #ff8a3d; --accent2: #4dd0e1; --ok: #69db7c; --chip: #2a2a32; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { padding: 22px 28px 10px; display: flex; flex-wrap: wrap; align-items: baseline; gap: 14px; }
h1 { margin: 0; font-size: 22px; letter-spacing: -0.01em; }
.status { color: var(--muted); }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; background: var(--chip); font-size: 12px; }
.badge.modded { background: var(--accent); color: #fff; }
main { padding: 0 28px 40px; display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
section.wide { grid-column: 1 / -1; }
h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.row { display: grid; grid-template-columns: 1fr 96px 22px; align-items: center; gap: 8px; padding: 5px 0; border-top: 1px solid var(--line); }
.row:first-of-type { border-top: 0; }
.row .n { font-weight: 550; } .row .d { color: var(--muted); font-size: 12px; }
.row.changed .n { color: var(--accent); }
input[type=number], select { width: 100%; padding: 5px 7px; border: 1px solid var(--line); border-radius: 7px; background: var(--bg); color: var(--text); font: inherit; }
.reset { cursor: pointer; color: var(--muted); border: 0; background: none; font-size: 15px; padding: 0; }
.reset:hover { color: var(--accent); }
.mode { display: flex; gap: 10px; align-items: flex-start; padding: 7px 0; border-top: 1px solid var(--line); }
.mode:first-of-type { border-top: 0; } .mode input { margin-top: 3px; }
.mode b { display: block; }
.bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
button.main { padding: 9px 16px; border-radius: 9px; border: 0; background: var(--accent); color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
button.alt { padding: 9px 14px; border-radius: 9px; border: 1px solid var(--line); background: var(--panel); color: var(--text); font: inherit; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
.match { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.match label { color: var(--muted); font-size: 12px; } .match input { margin-top: 3px; }
pre#msg { white-space: pre-wrap; margin: 12px 0 0; padding: 10px 12px; background: var(--chip); border-radius: 8px; font: 12px ui-monospace, monospace; display: none; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; } .grid2 label { color: var(--muted); font-size: 12px; }
@media (max-width: 600px) { header, main { padding-left: 16px; padding-right: 16px; } }
</style></head><body>
<header><h1>Pole Riders Patcher</h1><span class="status" id="status">loading…</span></header>
<main>
  <section class="wide">
    <div class="bar">
      <button class="main" id="apply">Apply mods</button>
      <button class="alt" id="restore">Restore vanilla game</button>
      <button class="alt" id="resetall">Reset all fields to vanilla</button>
      <span class="note" style="margin:0">Apply patches the game (the original is backed up once) — restart the game to see changes.</span>
    </div>
    <pre id="msg"></pre>
  </section>
  <section>
    <h2>Match</h2>
    <div class="match">
      <div><label>Score limit (goals)</label><input type="number" id="score" min="1" max="99"></div>
      <div><label>Round length (minutes)</label><input type="number" id="minutes" min="1" max="60"></div>
    </div>
    <div class="note">Above 7 goals / 5 min the menu limits are raised automatically.</div>
    <h2 style="margin-top:18px">Modes</h2><div id="modes"></div>
  </section>
  <section>
    <h2>Launch the game</h2>
    <div class="grid2">
      <div><label>Window</label><select id="window"><option value="1280x720">1280 × 720 window</option><option value="full">full screen</option></select></div>
      <div><label>&nbsp;</label><button class="alt" id="launch" style="width:100%">Launch game</button></div>
    </div>
    <div class="bar" style="margin-top:10px"><button class="alt" id="kbd_wasd">+ WASD keyboard pad</button><button class="alt" id="kbd_arrows">+ arrows keyboard pad</button></div>
    <div class="note">Keyboard pads work in the menus and in matches while the game window has focus.<br>
      <b>WASD</b>: A/D run · W/S rotate the pole counter-clockwise / clockwise · Space or E = confirm · Q = back · R = start<br>
      <b>Arrows</b>: ←/→ run · ↑/↓ rotate the pole counter-clockwise / clockwise · Return = confirm · right Shift = back · Backspace = start</div>
  </section>
  <section>
    <h2>Match (scripted set-up)</h2>
    <div class="grid2">
      <div><label>Left rider (attacks →)</label><select id="left"></select></div>
      <div><label>Right rider (attacks ←)</label><select id="right"></select></div>
    </div>
    <label style="display:block;margin:10px 0"><input type="checkbox" id="record" checked> record human play as training data</label>
    <div class="bar"><button class="main" id="play">Start match</button><button class="main" id="rateme" title="8 matches vs adaptively chosen gauntlet bots, on your controller (or the keyboard picked on the left)">Rate me vs the gauntlet</button><button class="alt" id="stop">Close game</button></div>
    <div class="note">Uses the Window setting above. Controller: plug it in first. Keyboard: click into the game window once the match starts. <span id="recs"></span></div>
  </section>
  <div id="params" style="display:contents"></div>
</main>
<script>
const $ = id => document.getElementById(id);
let S = null;
const api = async (path) => (await fetch(path)).json();
function msg(t) { $('msg').style.display = t ? 'block' : 'none'; $('msg').textContent = t || ''; }
function status(st) {
  const mod = st.binary.includes('MODDED');
  $('status').innerHTML = `<span class="badge ${mod ? 'modded' : ''}">${mod ? 'game modded' : 'vanilla game'}</span> &nbsp;` +
    (st.game_running ? '<span class="badge">game running</span>' : '');
  $('recs').textContent = st.bots.length ? `Your recorded play so far: ${st.recordings.matches} matches, ${st.recordings.minutes} min.` : '';
}
function render(st) {
  S = st; status(st);
  $('score').value = st.score; $('minutes').value = st.minutes;
  $('modes').innerHTML = st.modes.map(m => `<label class="mode"><input type="checkbox" data-mode="${m.key}" ${m.on ? 'checked' : ''}>
    <span><b>${m.key.replace(/_/g, ' ')}</b><span class="note" style="margin:0">${m.desc}</span></span></label>`).join('');
  const secs = [...new Set(st.params.map(p => p.section))];
  $('params').innerHTML = secs.map(sec => `<section><h2>${st.titles[sec] || sec}</h2>` + st.params.filter(p => p.section === sec).map(p =>
    `<div class="row ${Math.abs(p.value - p.vanilla) > 1e-6 ? 'changed' : ''}" data-key="${p.key}">
      <div><div class="n">${p.name.replace(/_/g, ' ')}</div><div class="d">${p.desc || ''} · vanilla ${p.vanilla}</div></div>
      <input type="number" step="any" value="${p.value}"><button class="reset" title="back to vanilla">↺</button></div>`).join('') + '</section>').join('');
  document.querySelectorAll('.row').forEach(r => {
    const p = st.params.find(x => x.key === r.dataset.key), inp = r.querySelector('input');
    inp.oninput = () => r.classList.toggle('changed', Math.abs(+inp.value - p.vanilla) > 1e-6);
    r.querySelector('.reset').onclick = () => { inp.value = p.vanilla; inp.oninput(); };
  });
  const opts = sel => [['pad', 'your controller'], ['wasd', 'keyboard WASD'], ['arrows', 'keyboard arrows'], ['gauntlet', 'gauntlet: rate me (8 matches)']]
    .map(([v, t]) => `<option value="${v}" ${v === sel ? 'selected' : ''}>${t}</option>`).join('') +
    st.bots.map(b => `<option value="bot:${b}" ${'bot:' + b === sel ? 'selected' : ''}>bot: ${b}</option>`).join('');
  $('left').innerHTML = opts('pad'); $('right').innerHTML = opts(st.bots.includes('farm2') ? 'bot:farm2' : 'arrows');
}
function collect() {
  return { score: +$('score').value, minutes: +$('minutes').value,
    params: S.params.map(p => ({ ...p, value: +document.querySelector(`.row[data-key="${p.key}"] input`).value })),
    modes: Object.fromEntries([...document.querySelectorAll('[data-mode]')].map(c => [c.dataset.mode, c.checked])) };
}
async function act(path, label) {
  msg(label + '…'); const r = await api(path); msg(r.message); status(r.state); return r;
}
$('apply').onclick = async () => { const r = await act('/api/apply?json=' + encodeURIComponent(JSON.stringify(collect())), 'Applying'); render(r.state); msg(r.message); };
$('restore').onclick = () => act('/api/restore', 'Restoring the original game');
$('resetall').onclick = () => { document.querySelectorAll('.row').forEach(r => r.querySelector('.reset').click());
  document.querySelectorAll('[data-mode]').forEach(c => c.checked = false); msg('All fields set to vanilla — press Apply to write them to the game.'); };
$('launch').onclick = () => act('/api/launch?window=' + $('window').value, 'Launching');
$('play').onclick = () => act(`/api/play?left=${encodeURIComponent($('left').value)}&right=${encodeURIComponent($('right').value)}&window=${$('window').value}&record=${$('record').checked ? 1 : 0}`, 'Starting');
$('kbd_wasd').onclick = () => act('/api/kbd?mode=wasd', 'Adding a keyboard pad');
$('kbd_arrows').onclick = () => act('/api/kbd?mode=arrows', 'Adding a keyboard pad');
$('stop').onclick = () => act('/api/stop', 'Closing');
$('rateme').onclick = () => { const h = ['pad', 'wasd', 'arrows'].includes($('left').value) ? $('left').value : 'pad';
  act(`/api/play?left=${h}&right=gauntlet&window=${$('window').value}&record=1`, 'Starting your rating matches'); };
api('/api/state').then(r => render(r.state));
setInterval(async () => { try { status((await api('/api/state')).state); } catch (e) {} }, 5000);
</script></body></html>
"""


def main(open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://localhost:{PORT}/"
    print(f"Pole Riders Patcher on {url} (ctrl-c to quit)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
