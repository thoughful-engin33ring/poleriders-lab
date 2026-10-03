#!/usr/bin/env python3
"""Super Pole Riders mod patcher for Sportsfriends (macOS Steam build, 2015-09-17).

Patches the game binary in place from mods.toml. The original is backed up once
to Sportsfriends.orig next to it, and every `apply` starts from that pristine
copy, so settings never stack and `restore` always gets you back to vanilla.

    python3 prmod.py list              # every tunable with its vanilla value
    python3 prmod.py apply [mods.toml] # patch the game from the config
    python3 prmod.py status            # what the installed binary currently has
    python3 prmod.py restore           # put the original binary back
    python3 prmod.py run [--bridge]    # launch the game directly (windowed 1280x720 by default)
    python3 prmod.py exhibition on|off # boot straight into Pole Riders (used for bot training)
    python3 prmod.py gui               # all of this in a window (your browser): patcher_gui.py
    python3 prmod.py match 7 5         # Pole Riders score limit / round minutes (config.ini; raises the
                                       # menu limits in mods.toml + re-applies when above 7 / 5)
"""
import hashlib, os, shutil, struct, subprocess, sys, tomllib

GAME = os.path.expanduser(
    "~/Library/Application Support/Steam/steamapps/common/Sportsfriends/Sportsfriends.app/Contents/MacOS")
BIN = os.path.join(GAME, "Sportsfriends")
ORIG = BIN + ".orig"
ORIG_SHA256 = "e24a10b2dbb4569a22faa1b6a237b5414b4b59ac1f46d19a80deacf297db1612"
HERE = os.path.dirname(os.path.abspath(__file__))

# section.key -> (file offset of float32, vanilla value, description)
# Offsets are the PoleRiders::* const symbols in __TEXT,__const; the game reads
# them by address (never constant-folded), so changing the bytes changes the game.
FLOATS = {
    "gameplay.kick_ball_strength":   (0x28e0dc, 1400.0, "impulse applied to the ball on a kick / hammer hit"),
    "gameplay.kick_player_strength": (0x28e0d8, 700.0,  "knockback when a player is kicked / hammered"),
    "gameplay.pole_swing_torque":    (0x28e0d0, 440.0,  "how hard the stick rotates the pole"),
    "gameplay.pole_push_force":      (0x28e0cc, 10.0,   "push force along the pole"),
    "gameplay.spin_speed_factor":    (0x28e0d4, 12.0,   "rider spin speed multiplier"),
    "gameplay.standard_pole_length": (0x28e0e0, 780.4,  "pole length in pixels (levels 1-4)"),
    "gameplay.rainbow_pole_length":  (0x28e0e4, 740.4,  "pole length in pixels (Rainbow Highway)"),
    "gameplay.death_delay":          (0x28e0b8, 3.0,    "seconds before a knocked-out player respawns"),
    "gameplay.player_linear_damping":(0x28e0bc, 0.33,   "air drag on riders"),
    "gameplay.ball_damping":         (0x28e0c0, 1.1,    "air drag on the ball"),
    "gameplay.pole_angular_damping": (0x28e0c4, 0.4,    "rotational drag on poles"),
    "gameplay.pole_linear_damping":  (0x28e0c8, 0.2,    "drag on poles"),
    "ball.density":     (0x28e480, 4.2, "ball mass per area (heavier = harder to knock)"),
    "ball.friction":    (0x28e478, 0.1, ""),
    "ball.restitution": (0x28e47c, 0.9, "bounciness (1.0 = perfectly elastic)"),
    "player.density":       (0x28e780, 1.1,  ""),
    "player.friction":      (0x28e778, 0.8,  ""),
    "player.restitution":   (0x28e77c, 0.0,  "bounciness"),
    "player.hitbox_width":  (0x28e788, 48.0, "pixels"),
    "player.hitbox_height": (0x28e784, 100.0,"pixels"),
    "player.stick_deadzone":(0x28e774, 0.2,  "analog stick dead zone"),
    "pole.width":       (0x290074, 16.0, "pixels"),
    "pole.density":     (0x290080, 0.2,  ""),
    "pole.friction":    (0x29007c, 0.4,  ""),
    "pole.restitution": (0x290078, 0.0,  "bounciness"),
}
_LEVELS = {  # name, levelWidth, targetWidth, ballHeight, spawnHeight, spawnBoost, friction, restitution
    1: (0x28e598, 50.0, 0.5), 2: (0x28e60c, 80.0, 0.5), 3: (0x28e65c, 55.0, 0.5),
    4: (0x28e698, 150.0, 0.5), 5: (0x28e6ec, 70.0, 0.6)}
for _n, (_base, _boost, _fric) in _LEVELS.items():
    FLOATS[f"level{_n}.spawn_boost"] = (_base + 0x10, _boost, "upward kick given to the ball when it spawns")
    FLOATS[f"level{_n}.ground_friction"] = (_base + 0x14, _fric, "")
    FLOATS[f"level{_n}.ground_restitution"] = (_base + 0x18, 0.3, "ground bounciness")

# b2World is constructed with gravity (0, 20.0) as an immediate in two places:
# PlayState::PlayState and PlayState::_SetupWorld (used when a match resets).
GRAVITY_SITES = [0x13436a, 0x13b04a]
GRAVITY_DEFAULT = 20.0

# Menu/config limits. Common::Save::kSettingMeta entries are {name[32], type, min, max, step};
# the options-menu slider reads max from there. Save::Load clamps config.ini values against
# shared literal-pool constants instead, so those two `movss lit(%rip)` loads get redirected
# at the meta entry's own max field (disp32 rewrite) and one patch covers both paths.
LIMITS = {
    # key: (meta max offset, vanilla max, clamp-load disp32 offset, vanilla disp, redirected disp, desc)
    "score_limit_max":      (0x289664, 7.0, 0x664e5, 0x2203bf, 0x22317b,
                             "highest selectable Pole Riders score limit"),
    "duration_max_minutes": (0x289798, 5.0, 0x6653e, 0x2205ae, 0x223256,
                             "highest selectable round length, in minutes"),
}

# Byte patches: flag -> [(offset, vanilla bytes, patched bytes)]
FLAGS = {
    # PlayState ctor: `seta (%rax)` on random() < 0.0001  ->  `movb $1,(%rax)`
    "force_snake": ("snake poles (vanilla: 1 in 10,000 matches)",
                    [(0x134306, "0f9700", "c60001")]),
    # the "not pencil" path `xorb %al,%al` -> `movb $1,%al` (vanilla: random() < 0.0011)
    "force_pencil": ("pencil poles (vanilla: ~1 in 900 matches)",
                     [(0x134355, "30c0", "b001")]),
    # Dec-25 check: both result paths forced to true
    "force_christmas": ("Christmas mode (vanilla: only on Dec 25)",
                        [(0x1342b1, "30c0", "b001"), (0x1342b9, "0f94c0", "b00190")]),
    # replace `call _CalculateChineseNewYear` with `mov al,1`
    "force_chinese_new_year": ("Chinese New Year mode (vanilla: only around the lunar new year)",
                               [(0x134289, "e87e120000", "b001909090")]),
    # PlayState::debug is zero-filled BSS, so instead skip the two `if (debug)` jumps:
    # one creates the Box2D DebugRender, the other draws it each frame.
    # Pole Riders' character menu offers NumCharacters()-1 = 7 of its 8 characters until a player scrolls past
    # the last one three times (Menu::Players::Box::Update: `cmovel` keeps 7 while box+0x44 is unset). The 8th
    # (head 7) is the hidden short-pole rider (shorter pole, stiffer arms). Drop the cmov: all 8 always offered.
    "unlock_hidden_character": ("the hidden 8th Pole Riders character (short pole) is always selectable "
                                "(vanilla: scroll past the last character three times)",
                                [(0xf5e39, "0f44d8", "909090")]),
    "physics_debug": ("draw Box2D collision shapes over the game (developer overlay)",
                      [(0x1351c7, "7449", "9090"), (0x138864, "0f84b7000000", "909090909090")]),
}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def ensure_backup():
    if os.path.exists(ORIG):
        if sha256(ORIG) != ORIG_SHA256:
            sys.exit(f"{ORIG} exists but is not the known original build; refusing to continue.")
        return
    if sha256(BIN) != ORIG_SHA256:
        sys.exit("Installed binary isn't the build this tool was made for (and no backup exists).\n"
                 "Verify game files in Steam to get a clean copy, then retry.")
    shutil.copy2(BIN, ORIG)
    print(f"backed up original -> {ORIG}")


def get_f32(buf, off):
    return struct.unpack_from("<f", buf, off)[0]


def verify_sites(buf):
    """Sanity-check every patch site against the original so a different build can't be corrupted."""
    for key, (off, default, _) in FLOATS.items():
        assert abs(get_f32(buf, off) - default) < 1e-3, f"{key}: unexpected original value"
    for off in GRAVITY_SITES:
        assert abs(get_f32(buf, off) - GRAVITY_DEFAULT) < 1e-6, "gravity site mismatch"
    for key, (moff, mdef, doff, ddef, _, _) in LIMITS.items():
        assert get_f32(buf, moff) == mdef, f"{key}: meta max mismatch"
        assert struct.unpack_from("<i", buf, doff)[0] == ddef, f"{key}: clamp site mismatch"
    for _, (_, sites) in FLAGS.items():
        for off, orig, _ in sites:
            assert buf[off:off + len(orig) // 2].hex() == orig, f"flag site {off:#x} mismatch"


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    floats, flags, limits, gravity = {}, {}, {}, None
    for section, table in cfg.items():
        if not isinstance(table, dict):
            sys.exit(f"top-level key '{section}' must be a [section]")
        for key, val in table.items():
            full = f"{section}.{key}"
            if section == "limits":
                if key not in LIMITS:
                    sys.exit(f"unknown limit '{key}' (known: {', '.join(LIMITS)})")
                if not 1 <= int(val) <= 999:
                    sys.exit(f"limits.{key} must be between 1 and 999")
                limits[key] = float(int(val))
            elif full == "gameplay.gravity":
                gravity = float(val)
            elif section == "modes":
                if key not in FLAGS:
                    sys.exit(f"unknown mode '{key}' (known: {', '.join(FLAGS)})")
                flags[key] = bool(val)
            elif full in FLOATS:
                floats[full] = float(val)
            else:
                sys.exit(f"unknown setting '{full}' — run `prmod.py list` for valid names")
    return floats, flags, limits, gravity


def cmd_apply(path):
    floats, flags, limits, gravity = load_config(path)
    ensure_backup()
    buf = bytearray(open(ORIG, "rb").read())
    verify_sites(buf)
    changed = []
    for key, val in floats.items():
        off, default, _ = FLOATS[key]
        struct.pack_into("<f", buf, off, val)
        if abs(val - default) > 1e-6:
            changed.append(f"{key} = {val:g}  (vanilla {default:g})")
    if gravity is not None:
        for off in GRAVITY_SITES:
            struct.pack_into("<f", buf, off, gravity)
        if gravity != GRAVITY_DEFAULT:
            changed.append(f"gameplay.gravity = {gravity:g}  (vanilla {GRAVITY_DEFAULT:g})")
    for key, val in limits.items():
        moff, mdef, doff, _, dnew, _ = LIMITS[key]
        if val != mdef:
            struct.pack_into("<f", buf, moff, val)
            struct.pack_into("<i", buf, doff, dnew)
            changed.append(f"limits.{key} = {val:g}  (vanilla {mdef:g})")
    for name, on in flags.items():
        if on:
            for off, _, new in FLAGS[name][1]:
                buf[off:off + len(new) // 2] = bytes.fromhex(new)
            changed.append(f"modes.{name} = true")
    tmp = BIN + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    shutil.copymode(ORIG, tmp)
    os.replace(tmp, BIN)
    print("patched" if changed else "no differences from vanilla; wrote clean binary")
    for c in changed:
        print("  " + c)


def cmd_status():
    buf = open(BIN, "rb").read()
    if sha256(BIN) == ORIG_SHA256:
        print("installed binary: VANILLA")
        return
    print("installed binary: MODDED" + ("" if os.path.exists(ORIG) else " (no backup found!)"))
    for key, (off, default, _) in FLOATS.items():
        v = get_f32(buf, off)
        if abs(v - default) > 1e-4:
            print(f"  {key} = {v:g}  (vanilla {default:g})")
    g = get_f32(buf, GRAVITY_SITES[0])
    if g != GRAVITY_DEFAULT:
        print(f"  gameplay.gravity = {g:g}  (vanilla {GRAVITY_DEFAULT:g})")
    for key, (moff, mdef, *_) in LIMITS.items():
        if get_f32(buf, moff) != mdef:
            print(f"  limits.{key} = {get_f32(buf, moff):g}  (vanilla {mdef:g})")
    for name, (_, sites) in FLAGS.items():
        off, _, new = sites[0]
        if buf[off:off + len(new) // 2].hex() == new:
            print(f"  modes.{name} = true")


def cmd_restore():
    if not os.path.exists(ORIG):
        sys.exit("no backup to restore from (binary was never patched by this tool)")
    shutil.copy2(ORIG, BIN)
    print("restored vanilla binary")


def cmd_list():
    print("[limits]   # then pick the value in the in-game options, or config.ini [poleriders]")
    for key, (_, mdef, *_, desc) in LIMITS.items():
        print(f"{key} = {mdef:g}   # {desc}")
    print("\n[gameplay]")
    print(f"gravity = {GRAVITY_DEFAULT:g}   # downward acceleration in Box2D units")
    section = "gameplay"
    for key, (_, default, desc) in FLOATS.items():
        s, k = key.split(".")
        if s != section:
            print(f"\n[{s}]")
            section = s
        print(f"{k} = {default:g}" + (f"   # {desc}" if desc else ""))
    print("\n[modes]")
    for name, (desc, _) in FLAGS.items():
        print(f"{name} = false   # {desc}")


def pad_env(env):
    """Controller mappings the game's old SDL lacks (DualSense...): bridge/pad_mappings.py."""
    sys.path.insert(0, os.path.join(HERE, "bridge"))
    from pad_mappings import PAD_MAPPINGS
    env["SDL_GAMECONTROLLERCONFIG"] = "\n".join(filter(None, [env.get("SDL_GAMECONTROLLERCONFIG"), PAD_MAPPINGS]))
    return env


def cmd_run(args):
    env = pad_env(dict(os.environ))
    if "--bridge" in args:
        args = [a for a in args if a != "--bridge"]
        dylib = os.path.join(HERE, "bridge", "sprbridge.dylib")
        if not os.path.exists(dylib):
            sys.exit("bridge/sprbridge.dylib missing — build it with: make -C bridge")
        env["DYLD_INSERT_LIBRARIES"] = dylib
        # the game polls a PS Move daemon every frame; when it isn't reachable each poll stalls the frame
        # (menus at ~1/7 speed): the bridge stubs PS Move out (Pole Riders never uses it)
        env.setdefault("SPRBRIDGE_NOMOVE", "1")
    args = args or ["-windowed", "1", "-width", "1280", "-height", "720"]
    log = open(os.path.join(HERE, "last-run.log"), "w")
    subprocess.Popen([BIN, *args], cwd=GAME, env=env, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    print("launched" + (" with sprbridge" if "DYLD_INSERT_LIBRARIES" in env else "")
          + " (stdout: last-run.log, game log: ~/Library/Application Support/sportsfriends/log.txt)")


CONFIG_INI = os.path.expanduser("~/Library/Application Support/sportsfriends/config.ini")


def cmd_exhibition(arg):
    """Exhibition mode boots straight into Pole Riders team select and never leaves the game."""
    import re
    if not os.path.exists(CONFIG_INI):
        sys.exit("config.ini not found — launch the game once first")
    text = open(CONFIG_INI).read()
    value = {"on": "PoleRiders", "off": "No"}.get(arg)
    if value is None:
        cur = re.search(r"^exhibition = (.*)$", text, re.M)
        print(f"exhibition = {cur.group(1) if cur else '?'}")
        return
    new = re.sub(r"^exhibition = .*$", f"exhibition = {value}", text, flags=re.M)
    open(CONFIG_INI, "w").write(new)
    print(f"exhibition = {value}")


def ini_set(section, key, value):
    """Set `key = value` in [section] of the game's config.ini (added if missing)."""
    lines = open(CONFIG_INI).read().splitlines() if os.path.exists(CONFIG_INI) else []
    out, sec, done = [], None, False
    for ln in lines:
        st = ln.strip()
        if st.startswith("[") and st.endswith("]"):
            if sec == section and not done:
                out.append(f"{key} = {value}"); done = True
            sec = st[1:-1]
        elif sec == section and st.split("=")[0].strip() == key:
            out.append(f"{key} = {value}"); done = True
            continue
        out.append(ln)
    if not done:
        if sec != section:
            out += ["", f"[{section}]"]
        out.append(f"{key} = {value}")
    open(CONFIG_INI, "w").write("\n".join(out) + "\n")


def toml_set(path, section, key, value):
    """Uncomment/set `key = value` in [section] of a mods.toml-style file, keeping its comments."""
    lines = open(path).read().splitlines()
    sec, done = None, False
    v = ("true" if value else "false") if isinstance(value, bool) else str(value)
    for i, ln in enumerate(lines):
        st = ln.strip()
        if st.startswith("[") and "]" in st:
            sec = st[1:st.index("]")]
        elif sec == section and st.lstrip("# ").split("=")[0].strip() == key:
            comment = ln[ln.index("#", ln.index("=")):] if "#" in ln[ln.index("=") if "=" in ln else 0:] else ""
            lines[i] = f"{key} = {v}" + (f"   {comment}" if comment else "")
            done = True
    if not done:
        sys.exit(f"{section}.{key} not found in {path}")
    open(path, "w").write("\n".join(lines) + "\n")


def cmd_match(score, minutes):
    """Pole Riders match format in config.ini; above vanilla's 7 goals / 5 min the menu limits are raised too."""
    score, minutes = int(score), int(minutes)
    cfg = os.path.join(HERE, "mods.toml")
    if score > 7:
        toml_set(cfg, "limits", "score_limit_max", max(score, 7))
    if minutes > 5:
        toml_set(cfg, "limits", "duration_max_minutes", max(minutes, 5))
    if score > 7 or minutes > 5:
        cmd_apply(cfg)
    ini_set("poleriders", "scorelimit", score)
    ini_set("poleriders", "duration", minutes)
    print(f"Pole Riders: first to {score} goals, {minutes} min rounds (config.ini; restart the game)")


def main():
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "help"
    if cmd == "apply":
        cmd_apply(argv[1] if len(argv) > 1 else os.path.join(HERE, "mods.toml"))
    elif cmd == "status":
        cmd_status()
    elif cmd == "restore":
        cmd_restore()
    elif cmd == "list":
        cmd_list()
    elif cmd == "run":
        cmd_run(argv[1:])
    elif cmd == "gui":
        import patcher_gui
        patcher_gui.main()
    elif cmd == "match" and len(argv) == 3:
        cmd_match(argv[1], argv[2])
    elif cmd == "exhibition":
        cmd_exhibition(argv[1] if len(argv) > 1 else "")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
