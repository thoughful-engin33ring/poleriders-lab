# Super Pole Riders lab

Bots for Super Pole Riders (Sportsfriends, macOS Steam build), trained with self-play reinforcement learning in a
bit-exact re-implementation of the game's physics (`sim/`), plus a mod kit / patcher and tools to play the bots in
the real game. `HANDOFF.md` is the detailed lab notebook; `docs/strategy.md` is the strategy guide the bots are
measured against.

## Play the bots
`bots/` has frozen bots, strongest first: `t20_300M.pt` (beats everything else here by ~6 goals per match),
`farm2_427M.pt` (the old reference bot), `t27_20M.pt` (medium), `t24_20M.pt` (easy).

```
python3 -m venv .venv && .venv/bin/pip install torch numpy optuna
make -C bridge                 # the game bridge (x86_64 dylib, needs Xcode command line tools)
make -C sim env                # the native simulator (training / evaluation only)
python3 patcher_gui.py         # the patcher + match launcher (http://localhost:8766)
.venv/bin/python play_human.py --left pad --right bot:bots/t20_300M.pt     # you (controller) vs t20
.venv/bin/python play_human.py --left wasd --right bot:bots/t27_20M.pt --windowed 1280x720
```
Needs Sportsfriends installed through Steam on a Mac. Controllers: any SDL game controller, incl. DualSense.

## Mod kit

Patcher for Super Pole Riders in Sportsfriends: the macOS Steam build (Mac binary built 2015-09-17), x86_64, run through Rosetta.

```
python3 prmod.py list                     # all tunables + vanilla values
python3 prmod.py apply                    # apply mods.toml
python3 prmod.py apply presets/moon.toml  # or a preset
python3 prmod.py status
python3 prmod.py restore                  # back to vanilla
python3 prmod.py run                      # launch windowed, no Steam needed
```

The first `apply` backs up the original binary to `Sportsfriends.orig` and checks its SHA-256. Every later `apply` starts again from that original, so patches never stack on top of each other. If Steam updates the game or you run "Verify files", it puts vanilla back. Run `apply` again after that.

## What's moddable
- **Physics and gameplay** (`[gameplay]`, `[ball]`, `[player]`, `[pole]`, `[levelN]`): gravity, kick strengths, pole torque, pole length, damping, friction, bounciness, hitboxes, respawn delay and ball spawn boost.
- **Hidden modes** (`[modes]`): the game already has these, but they rarely show up. Snake poles appear in 1 of 10,000 matches, pencil poles in about 1 of 900, Christmas only on Dec 25, and Chinese New Year only around the lunar new year. Each one can be forced on here.
- **`unlock_hidden_character`** (`[modes]`): Pole Riders has 8 characters but its menu offers 7 until a player scrolls past the last one three times; the 8th (head 7) has a shorter pole and stiffer arms. The patch removes that gate (`Menu::Players::Box::Update`, a 3-byte `cmov`).
- **`physics_debug`**: turns on the developers' leftover Box2D debug renderer, which draws the collision shapes.
- **Match format**: `python3 prmod.py match 7 5` sets the Pole Riders score limit / round minutes in config.ini (and raises the menu limits + re-applies when you go above 7 / 5).
- **Dashboard "Mods & play" tab** (dashboard.py): the same as a form (score, minutes, snake, pencil, Christmas, Chinese New Year, hidden character, physics overlay; Apply / Restore vanilla), plus *Play against a bot* (play_human.py).
- **Score limit and round time above 7 / 5 min** (`[limits]`, see `presets/long-matches.toml`): raises the maximum in the game's settings table (`Common::Save::kSettingMeta`), which the options-menu slider uses. It also redirects the `config.ini` loader's clamp to that maximum. After applying, pick the value in the options menu or set `[poleriders] scorelimit` / `duration` in `~/Library/Application Support/sportsfriends/config.ini`. Tested: with the patch, `config.ini` values of 15 and 12 load unchanged; vanilla clamps them to 7 and 5.
- **Sounds**: `Sportsfriends.app/Contents/Resources/Sound/poleriders/*.ogg` are plain OGG files, so you can replace them in place.

## Game state bridge (for bots / RL)
`bridge/sprbridge.dylib` is injected with `DYLD_INSERT_LIBRARIES`. This works because the binary is unsigned. The dylib hooks `PoleRiders::Update()` and, after every frame, snapshots the match: the scores plus every Box2D body with its tag (`player1-4`, `pole`, `ball`, `ground`, `wall`, ...), position, angle and velocity, in Box2D meters (64 px = 1 m). The snapshot is served on a Unix socket.

```
make -C bridge
python3 prmod.py run --bridge
python3 bridge/spr.py settings   # loaded score limit / duration
python3 bridge/spr.py watch      # live state stream during a match
```
In Python: `from spr import Bridge; Bridge().state()`. The snapshot has `score`, `time_left`, `players[]` (pose, velocity, on_pole, on_ground, ghost, 10 pole segments each), `ball`, and every body tagged.

### Virtual controllers (tested)
The bridge plugs fake pads into the engine's own gamepad slots (`atrophy::sGamepadList`), so the menus and gameplay treat them as real controllers.
```
python3 bridge/spr.py pad connect 0          # slots 0-3
python3 bridge/spr.py pad tap 0 a            # a b x y back guide start ls rs lb rb up down left right lt rt
python3 bridge/spr.py pad set 0 lx=1 rt=1    # held until changed; sticks -1..1 (+y = down)
python3 bridge/spr.py pad clear 0
```
Cold start into a match, all done with the pads: connect 2 pads → A (warning) → A (quote) → down, down, A (Pole Riders) → A (Start Game) → pad 0 lx=-1, pad 1 lx=1 (pick teams) → A → A on both pads (players) → A (arena).

## RL environment (`bridge/spr_env.py`)
Self-play 1v1 with no dependencies. Pad 0 is player 1 (Vaulters), pad 1 is player 2 (Benders).
```python
from spr_env import PoleRidersEnv
env = PoleRidersEnv(frame_skip=4, fast=20)
env.start_match(launch=True)     # kill + relaunch the game in exhibition mode, script into a match
obs = env.reset(seed=0)          # {1: [37 floats], 2: [37 floats]}
obs, rew, done, info = env.step({1: (run, aim_x, aim_y), 2: (run, aim_x, aim_y)})
```
- **Controls in a match** (full controller): left stick X runs. The right stick sets the angle the rider's arms try to drive the pole toward, and the pole lags behind it, so this is a servo problem. Nothing else does anything, except LT + left stick up + R3 while on the pole, which is a cosmetic taunt (it makes another player fart). Two players sharing one controller get the D-pad + LT, or the face buttons + RT.
- **Observations** are agent-centric: mirrored so each agent attacks toward +x, with y pointing up and units in meters. Layout: `me[0:15] opp[15:30] ball[30:34] time, my_score, opp_score`. Each player block is position, velocity, body angle (sin, cos), angular velocity, on_pole, on_ground, ghost, pole angle (sin, cos), pole angular velocity, and the pole tip relative to the player. The raw snapshot is in `info["state"]`.
- **Rewards** are zero-sum between the agents. Weights are set with `PoleRidersEnv(reward={...})`, and the per-term breakdown is in `info["terms"]`.
  - `goal`: ±1 for scoring or conceding. Vaulters score when the ball, hanging from its zipline trolley, hits the +x end wall; Benders the −x wall. This was verified live by firing the ball at each wall.
  - `kill`: +0.3 per opponent death (killed or suicide), −0.3 per own death. It's read from the game's own Stats counters. In this game a kill only happens when both riders are on their poles and at least one is airborne: a stomp from above, or an aerial clash resolved by relative height. Pole "hammer" hits are a separate kill path.
  - `ball_progress`: 0.003 × the change in ball x toward the opponent's wall. It's potential-based, so it can't change the optimal policy, and it's skipped on goal and respawn steps.
  - `done` is set on game over or time up. `python3 bridge/test_reward.py` runs the offline tests.
- **Curriculum starts:** the bridge command `ball x y vx vy` places the ball and its zipline trolley, for example near a goal.
- **Lockstep:** in sync mode each stepped frame runs exactly one 1/60 s physics tick. The bridge zeroes the game's fixed-step accumulator and pins the frame delta. Rollouts from `reset` are bit-for-bit reproducible: 600 steps matched exactly.
- **Speed:** `fast=N` turns vsync off (through CGL) and draws only every N-th frame. That gives about 7× real time through the env (about 9.5× raw) under Rosetta on an M5 Pro. For more throughput, run several game instances, each with its own `SPRBRIDGE_SOCK`.
- **Exhibition mode:** `start_match(launch=True)` sets `[game] exhibition = PoleRiders` in config.ini, which boots straight into Pole Riders. Undo it with `python3 prmod.py exhibition off`.

Bridge commands used by the env: `sync on|off`, `step N` (replies with the state), `reset [seed]`, `fast on N|off`, `pad ...`.

## Training (`train_ppo.py`)
```
python3 -m venv .venv && .venv/bin/pip install torch numpy
.venv/bin/python train_ppo.py --run first --instances 4       # launches 4 games, trains
.venv/bin/python train_ppo.py --run first --resume
.venv/bin/python train_ppo.py --run first --watch             # watch the latest checkpoint play
```
- **Self-play PPO:** one shared policy plays both sides, using the mirrored observations. It uses a Gaussian policy, running observation normalization, GAE, and 60 s episodes with time-limit bootstrapping.
- **League:** every `--pool-every` updates a snapshot is frozen into `runs/<run>/pool/`. In `--league-frac` of the instances, one rider (side picked at random) is driven by a past snapshot, and those transitions aren't trained on.
- **What to watch:** `vs_pool`, the current policy's zero-sum return against past snapshots. Goals and kills per episode are not a skill measure in self-play; strong equal players may converge to few of either.
- **Throughput:** about 110 env steps/s per game instance, and instances scale roughly linearly (3 instances measured 285/s). Each game takes about 340 MB.
- **Curriculum:** `--random-ball 10` starts each episode with the ball at a random x within ±10 m.

## Watcher (`dashboard.py`)
```
python3 dashboard.py --run flatirons      # http://localhost:8765
```
- **Live matches:** every training game is drawn from its physics state in one grid (riders, pole chains, ball on the zipline, goal walls in the colour of the team that scores there). Each card also shows the kill count, and the snapshot opponent for league games.
- **Charts:** skill (`vs_pool`, the rolling return against past snapshots), goals and kills per episode, episode length, throughput, policy std, value loss, the event rate per 10 min, and pool size.
- **Heatmaps** of ball and rider x-position across all games, plus airborne time.
- **Event feed** of goals, kills and instance crashes/relaunches.
- It only sends read-only `state` requests, so it can't disturb training.

Training defaults to arena 2 (Flatirons, `--arena`). Game instances that die are relaunched automatically. Per-instance logs are in `~/Library/Application Support/sportsfriends/logs/`.

## Headless farm (`train_farm.py`, `bridge/farm.c`, `bridge/farm_env.py`)
Training no longer needs one full game per match. Each game process ("host") runs many **shadow matches**. These are extra `PoleRiders::PlayState` objects that the bridge builds and ticks directly from its game thread: no drawing, no sound, no per-tick engine frame.
- **Equivalence:** `bridge/test_farm_equivalence.py` steps a shadow and the real rendered match with identical random inputs from the same tick-0 kickoff. Three runs of 2,400 ticks each were bit-identical. The shadows are the game.
- **Inputs:** each shadow keeps private copies of the engine's pad state. Its sticks go through the real input chain (`Gamepad::setJoystickPosition` → `Common::Input::_UpdatePad`) before every tick.
- **Sound:** the game's FMOD C calls are interposed. While shadow code runs, sounds are fake handles and real sounds are never released. Fake sounds are also removed from the engine's 512-entry loading list, which it only drains in frames that farm mode never runs.
- **Throughput:** 12 hosts × 16 matches (192 matches) run at about 3,000–4,000 env steps/s, versus about 750 with 12 rendered games. One host is about 2,200 ticks/s at about 450 µs/tick.
- **Bridge commands:** `farm create N`, `farm act n <6 floats per match>`, `farm reset k [clock]` (replies with the tick-0 snapshot), `farm state`, `farm info`, `farm destroy`.

```
.venv/bin/python train_farm.py --run farm1 --hosts 12 --matches 16      # train
python3 dashboard.py --run farm1                                         # watch (farm mode)
.venv/bin/python train_farm.py --run farm1 --watch                       # policy in a real, rendered match
```

**Observation v2** (60 floats, mirrored per agent) adds to v1:
- a hammer flag per rider;
- a respawn countdown per rider;
- **layered terrain**: at −6…+6 m, the floor below the rider and the nearest overhead ledge (Flatirons has high ledges above the pits);
- a sudden-death flag.

While the opponent is dead, its corpse is hidden: its block is all zeros except the ghost flag and the respawn countdown.

**Sudden death:** when time runs out with the scores level, the game zeroes the score, turns poles into hammers, and a goal *or a hammer kill* wins. `--sd-frac` (default 0.15) starts that many matches with the clock at 0.05 s, so the game enters sudden death at once and the hammer phase gets practice.

**Training-instance guards** (set by the launcher):
- PS Move stubbed out (`SPRBRIDGE_NOMOVE`). Its daemons fought over one port and crashed games.
- Rainbow Highway unlock disabled (`SPRBRIDGE_NOUNLOCK`); every reset checks the match is on the expected level.
- Canonical rules (`SPRBRIDGE_CANON`): no random snake or pencil poles and no Christmas / Chinese New Year modes.

## How it works
The binary ships with full C++ symbols. It has `PoleRiders::*` classes and uses Box2D physics. Each tuning constant (for example `PoleRiders::Assets::kickBallStrength`) is a named `const float` that the code reads by address. That means changing those 4 bytes changes the game. Gravity is an immediate value `(0, 20.0)` given to `b2World` in two places. The mode flags are small instruction patches in `PlayState::PlayState`. Every patch site is checked against the original bytes before anything is written.

## Not done yet
- **New maps (parked):** each level is hard-coded C++. `Level1..5::CreateBody(x, y)` builds static Box2D bodies from a Level object that is just `{b2World*}`, so hooking it to build ground from JSON is straightforward. `bridge/dump_level.py N` already exports an arena's polygons to `arenas/arenaN.json`, which can serve as a template.
- **Textures** (`.aetex`): the format is a 4-byte size, then zlib, then a 48-byte header (dimensions and mip count), then a mip offset table, then a custom per-mip encoding that isn't decoded yet. `.ssht` files are plain sprite-sheet frame tables.
