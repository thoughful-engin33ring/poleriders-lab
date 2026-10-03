# Handoff — Super Pole Riders RL lab (2026-09-30, end of day 2)

Everything below is current. Read `docs/strategy.md` too: game knowledge from the user (self-described
best player in the world), backed by data where we could measure it.

## What's running / paused
| run | state | command (resume = same command; `--resume` for train_farm) |
|---|---|---|
| **pbt1** | **training** (population of 4 from scratch, started 2026-10-02 ~13:20; ~8k env steps/s per member) | controller: `.venv/bin/python -u pbt.py --name pbt1 --resume >> runs/pbt1/pbt.out 2>&1 &` (members = train_farm runs `runs/pbt1_m0..m3`, launched/stopped by the controller; stop it with `pkill -f pbt.py` — it stops its members) |
| farm3 | paused 2026-10-02 at 684M steps (still -1.3..-2.5 goals/match vs farm2) — was: (from scratch, started 2026-09-30 ~23:45; random pole at spawn + fading backward-rotation bonus) | `.venv/bin/python -u train_farm.py --run farm3 --resume --native --hosts 12 --matches 38 --procs 6 --arena 2 --threads 18 --torch-threads 1 --min-std 0.1 --pit-frac 0 --no-sudden-death --format 7,300 --random-pole --rotation-bonus 0.005 --rotation-anneal 100e6 --league-fixed farm2:runs/farm3/opp_farm2.pt:0.25 >> runs/farm3/train.out 2>&1` (farm2 frozen in 25% of league games since ~608M steps; user: nothing else until farm3 beats farm2) |
| worker3 | paused at 3.5M steps (user: farm3 only for now; from farm2; opponents 1/3 each self-play / farm2 / killer2; jump curriculum; started by hand 2026-09-30 ~23:25 when the user judged killer2 saturated) | `train_worker.py --run worker3 --init-std 0.3 --min-std 0.15 --hosts 12 --matches 16 --threads 16 --torch-threads 4 --solo-frac 0 --dead-frac 0 --opps farm2:runs/worker3/opp_farm2.pt:0.34,killer2:runs/worker3/opp_killer2.pt:0.33 --no-sudden-death --format 7,300 --air-curriculum` (resume = same without `--init`) |
| killer2 | paused (180M steps): vs farm2 ~3.5-3.8 kills / ~1.6 deaths per min (farm2 vs farm2 1.59 / 1.74); self-play degenerated into kill trading (66 kills/match) — expected with kill 2 / death 1 | see above (`--resume`) |
| killer1 | paused (zero-sum kill reward, 167M steps): vs farm2 2.06 kills / 1.21 deaths per min (farm2 vs farm2: 1.59 / 1.74); learned mostly not to die | same command with `--run killer1 --reward '{"goal": 0, "kill": 1, "ball_progress": 0, "position": 0}'` |
| farm2 | paused (412M env steps, update ~17k; plateauing: ~50.5% of goals vs its 5 newest snapshots) | `.venv/bin/python -u train_farm.py --run farm2 --resume --native --hosts 12 --matches 38 --procs 6 --arena 2 --threads 18 --torch-threads 1 --min-std 0.1 --pit-frac 0 --no-sudden-death --format 7,300 >> runs/farm2/train.out 2>&1` (+ `eval_h2h.py --run farm2 ...` for the head-to-head chart) |
| dashboard | running, http://localhost:8765 (PBT tab + a tab per member, Farm tab = farm3) | `.venv/bin/python -u dashboard.py --run farm3 --ref farm2 --hier-opp "" --killer-run "" --manager-run "" --worker-run "" --pbt pbt1 --port 8765` |
| farm1 | paused (159M steps, update ~11.1k) | `train_farm.py --run farm1 --resume --native --hosts 14 --matches 16 --arena 2 --pit-frac 0.3 --threads 12 --min-std 0.1` |
| manager2 | paused (46M env steps, update ~570) | `train_farm.py --run manager2 --resume --native --hier runs/manager2/worker.pt --macro 8 --hosts 14 --matches 16 --arena 2 --threads 16 --pit-frac 0.3 --flat-league runs/farm1/pool --flat-frac 0.5` |
| worker2 | paused (~215M steps, update ~18.3k) | `train_worker.py --run worker2 --hosts 12 --matches 16 --threads 16 --torch-threads 4 --min-std 0.15` |

- No game processes are needed for training any more (native sim). If a game-backed trainer is ever
  stopped, its worker processes relaunch games as orphans — kill them with `kill -9` (the game ignores
  SIGTERM); never blanket-`pkill` Sportsfriends while an oracle is in use.
- The dashboard imports train_worker/hier_env/native_env at start: **restart it after changing worker
  or env code** (otherwise e.g. the Worker tab crashes on a feature-size mismatch).
- `config.ini` still has `exhibition = PoleRiders` (only needed for game-backed runs / oracles).

**Match format (user, 2026-09-30): 7 goals to win, 5 min** — `train_farm.py --format 7,300` (the default
now), native `NativeHost(match_format=(7, 300))`, eval_h2h and the dashboard use it too; the safety cap
is twice the round time. The env's own default stays 3 goals / 180 s (the game config the oracle checks).
farm2 switched at update ~3790 (54M steps) by `--resume`. Obs time_left is still divided by 180 (so it
reaches 1.67 at kick-off) to keep farm2's inputs compatible.

**No sudden death (user's call, 2026-09-30):** "if the match ends 0-0 then it ends 0-0". `--no-sudden-death`
(native env `end_on_sd`): a match tied when the clock runs out ends as a draw (terminal, no bootstrap) and
no matches start in sudden death (`--sd-frac 0`). Use it for all future runs. Note the clock only starts on
the first ball touch; fresh policies play long 0-0 draws at first (the log's `sd` column now counts matches
that ended tied at time-out). A first farm2 attempt with sudden death is in `runs/old/`.

## worker3
User: from farm2's checkpoint, opponents a third each self-play (duo), farm2, killer2. train_worker.py gained
`--opps NAME:CKPT:FRAC,...` (several flat opponents; deaths/min logged per opponent), `--format`,
`--no-sudden-death`, PPO on the GPU, `real_sps` in the log, and `--air-curriculum`: jump-target heights drawn
per band in proportion to s(1-s)+0.02 of a running success rate (worker2 ended at 43% / 21% / 18% / 6% for
1.5-3 / 3-4.5 / 4.5-6 / 6-7.5 m with uniform heights). Still the single-process loop (~20k real steps/s). A multi-process port was started and stopped unfinished (user: farm3 only): `old/train_worker_mp_wip.py`, NOT verified.

## farm3: backward pole rotation (2026-09-30)
User: all agents struggle to turn the pole backwards (counter-clockwise on the left / clockwise on the right =
over the top toward their own goal), which is sometimes the right move. Data (analysis/pole_rotation.py,
farm2 self-play): passes through vertical backward 4/min vs forward 31/min (killer2 9 vs 22); pole in the
back quadrants goes with +0.09..+0.10 goals over the next 10 s vs -0.02..-0.15 forward (correlational).
farm3 = new run from scratch (user: farm2 has internalised the habit) with `--random-pole` (every (re)spawn
builds the pole at a random 0-150 deg, attack frame; engine `randomPole`, per-match seeded) and
`--rotation-bonus 0.005 --rotation-anneal 100e6` (reward per radian of backward rotation, native
`RewardWeights::rotation` = Python `backward_rotation`, fading to 0 over 100M steps). Log/dashboard:
`back_passes_per_min`, `fwd_passes_per_min`, `rotation_w`. Dashboard tab `farm3` (`--tab farm3=farm3:farm2`).

## Killer specialists (2026-09-30)
killer1 (zero-sum kill reward) plateaued after ~40M steps and learned mostly defence (self-play kills fell
4.1 -> 1.3/min). killer2 (user): kills worth 2x deaths — reward weight `death` (native `RewardWeights::death`,
Python `compute_reward`; unset = the zero-sum kill term, bit-identical to before), so trading kills pays.
eval_h2h now reports kills and deaths per match / per minute separately (its env's reward counts kills).
Dashboard: Killer tab (`--killer-run`, `--killer-ref farm2`), kills/deaths per minute vs farm2.

### killer1 details
Reward kill 1 only (zero-sum: suicides/deaths -1), goals 0. `--random-spawn` (native `SetScramble`): after every
kill both riders re-spawn at random ground positions between the walls (>= 4 m apart, pole up) and the ball
at a random zipline point (|x| <= 15 m); after a goal the same happens once the game's own 1.4 s ball reset is
done. The transition after a re-spawn starts fresh (no reward for the teleport). `--init-std 0.6` re-opened
exploration (farm2 was at the 0.1 floor); floor 0.15. Uses (agreed): worker opponent share, manager option.

## PBT (pbt.py, 2026-10-02)
User: a brand-new run with evolutionary hyperparameters and reward weights. 4 members from scratch; genome = lr,
entropy bonus, std floor, gamma (perturbed via its horizon), reward weights (kill, position, ball progress),
backward-rotation bonus + its fade, random spawn after GOALS ONLY (probability; user: never after kills),
random pole on/off, and whether the policy sees score/clock (`--mask-match-info`, pinned in RunningNorm: every
tool loading the checkpoint hides them; inherited, never mutated). All members: goal-to-goal episodes
(`--goal-episodes`, user: a match is perfectly subdivided by goals), 7 goals / 5 min, no sudden death; half the
league games vs the other members' frozen checkpoints. Generation every 45 min: tournament (all pairs + farm2 +
farm3, 32 matches each), fitness = mean goal difference; the worst copies the best (weights, optimizer, norm,
pool) with a perturbed genome if the best leads by > 0.25. State: runs/pbt1/state.json; dashboard PBT tab.

## Human play recording + mod kit (2026-10-02)
`play_human.py --bot <run> [--side left|right]` (or the dashboard's "Mods & play" tab): the real game, the match
scripted with two virtual pads, then the user's controller drives their rider via the bridge's `pad mirror`
(copies a real pad's sticks into the virtual pad every frame) and the bot plays the other rider in real time.
The bridge's play log (`playlog on <dir>` / SPRBRIDGE_PLAYLOG) writes runs/human/<time>_m<match>.jsonl: all pads'
sticks every frame + the state every 4th frame (snapshots now carry each player's `pad` and `head`);
`analysis/playlog.py` -> runs/human/demos.npz = (observe2 obs, action in the bots' terms) for the human rider.
Tested with a virtual "human" (no physical controller here); not yet with a real one. Next (agreed idea): use the
demos — behaviour-cloning pretraining / a PBT member, and correction mode (take over at flagged moments).
Keyboard gamepads: bridge `pad keyboard <slot> wasd|arrows` (CGEventSourceKeyState, only while Sportsfriends is the active app — the game draws its own Cocoa window and uses SDL only for controllers, so SDL_GetKeyboardState is always empty; `keys` = debug dump; A/D or ←/→ run, W/S or ↑/↓ rotate the pole aim CCW/CW at 1 turn/s, starting from the pole's angle; menu buttons Space/E/Q/R or Return/RShift/Backspace); logged as human play (kind 4). `play_human.py --left pad|pad:<n>|wasd|arrows|bot:<run> --right ...` = any mix (bot vs bot, two people on one keyboard). It boots by polling the bridge (frame counter > 1000 ≈ 1.5 s, + 2 s settle; was a fixed 16 s) and runs the menus at 0.4/1.2 s steps (relaunches once with the old 0.8/2.5 s if that misses): ~11 s from Start to playing.
Patcher GUI: `python3 prmod.py gui` / double-click `Pole Riders Patcher.command` -> http://localhost:8766 (patcher_gui.py, stdlib only): every prmod setting, match format, apply/restore, launch, play vs bot.
Mod kit: prmod.py gained `unlock_hidden_character` (3-byte patch at file 0xf5e39: the menu offers all 8 Pole
Riders characters; head 7 = short pole) and `match <score> <minutes>`; the game is currently VANILLA.

## Real-game replays to MP4
Dashboard replays store their inputs; `/api/export?slot=X1` saves the shown match (runs/replays/*.npz);
`render_replay.py <npz>` replays it in the real game (full screen) and records it (bridge `record start/stop`:
interposed CGLFlushDrawable -> glReadPixels -> ffmpeg). The real match is frozen ~33 frames after a reset
(loading fade): the first input is fed frame by frame until the first real tick — without that the replay
diverged ("trolling bots"). Verified: identical 6-3.

## farm3 vs farm2 (2026-10-02, analysis/h2h_breakdown.py -> analysis/h2h_farm3_farm2.txt)
farm3 has the technique (8.8 vs 3.2 backward passes/min, more kills: 2.1 vs 1.5/min) but loses on position:
22% of its time in situations A/B vs farm2's 10%, good side 72% vs 87%, and it converts the right side poorly
(goal diff next 10 s from right-far +0.03 vs farm2's +0.17); 38% of its goals come with farm2 dead (farm2: 19%).
Fix tried: frozen farm2 as 25% of farm3's league games (`--league-fixed`, rollout.Match `fixed`; logged as vs_flat).
Dashboard charts trimmed (2026-10-02): removed "skill vs past selves" (pool-composition artefact; its yellow line
duplicated the head-to-head), events/10 min, snapshot pool, episode length; goals+kills merged into "self-play
match style"; good-side chart without the near-zero pit lines (pit % in its title).

## Concept library (2026-10-02)
Step 1 of "bots that grasp concepts": docs/strategy.md turned into predicates. `concepts.py` (repo root, numpy only):
`Frames(O, live)` over observe2 rows [T, ..., 62] (attack frame) -> 26 state concepts (`state_concepts`: good side,
near ball, tip behind/past the ball, situations right-far/right-close/A/B/both-wrong, opp out = kill window, opp
wrong side, airborne, pole planted, pit, pole quadrants, backswing in flight, pole on opp / opp pole on me, pole over
opp's head, opp on my pole / me on his, tip peek, rhythmic pressing in B) and 13 events (`event_concepts`, int16 =
length on the step the event completes, no lookahead: goal for/against, kill, death, back/fwd pass, vault, pogo vault
(landing on the pole without touchdown -> vault), jump over the ball from A, pushed opp to the wrong side, tip got
behind the ball, catapult, peek touch). Each docstring = exact definition + thresholds + the strategy idea.
`obs_from_states(snapshots)` / `playlog_matches(runs/human/*.jsonl)` give the same rows from NativeHost.json or the
human play logs (for the hindsight labelling step). Tests: `test_concepts.py` (21, incl. snapshot->obs == native obs,
play-log parsing). Validated on full snapshots: kill = game kill stats 185/186; opp_on_my_pole 88% precision / 89%
recall vs the real pole polyline; `on_ground` is true on poles too (90% of grounded-at-height = on the opp's pole).
Report: `analysis/concept_report.py --self CKPT --h2h A:B --matches 128` -> `analysis/concepts_report.txt/.json`
(per policy: share / entries per min / rate, value = own goal diff next 10 s minus baseline; correlational).
Headlines (128 matches each, farm2 427M, farm3 684M, PBT members ~37-40M steps):
- Confirmed for every policy: tip behind the ball +0.06..+0.18 vs tip past -0.08..-0.24; A -0.24..-0.72, B
  -0.14..-0.39; kill +0.19..+0.35 (opp out +0.18..+0.38); pushed opp to the wrong side +0.08..+0.29 but rare
  (0.2-0.5/min; farm2 1.3/min vs pbt1_m1).
- farm3 vs farm2 (31-12-85, -1.51/match): farm3 spends 13% in B (farm2 8%), good side 78% vs 85%; it uses the
  backward technique ~3x more (back passes 7.7 vs 3.3/min, backswing in flight 12% vs 4% of time) but those carry
  ~0 value; more kills (2.0 vs 1.4/min); fewer pogo vaults (1.2 vs 2.0/min) and they are bad for it (-0.21).
- Never/weakly used: jump over the ball from A 0.1-0.4/min and negative for everyone (-0.06..-0.35: the user's
  "only with free tempo"); catapult ~0.5/min but <= 0 value — standing on the OTHER's pole is good for the stander
  (farm2/farm3 +0.07..+0.12), i.e. nobody throws yet; rhythmic pressing 0.3-0.6% of farm2/farm3 time (~4% of their B
  time) and negative for them; peek touch/tip peek negative (see caveats).
- PBT (early): m0-m2 press a lot (pole on opp 12-35% of time vs farm2 5-7%, rhythmic pressing up to 10% of time vs farm2,
  worth +0.15..+0.21 there). **pbt1_m3 farms the rotation bonus**: 52 back passes/min, never touches the ball (0 goals in 128
  self-play matches, 0-7 in all 128 vs farm2) — matches its -7.00 PBT fitness.
- Energy storage: pogo-vault apex = other vaults for farm2 (5.8 m), higher for farm3 (6.7 vs 6.1) and m1 (10.4 vs 5.9).
Caveats (not robust): the pole's bend is not observed (pogo = proxy); pole contact uses a straight body->tip line;
tip_peek does not separate "from below/above" (it also catches ordinary strides). Ball touches now come from the
native env's touch rows (below); play logs / snapshots still have none.

## Showing agents in the real game
`play_real.py --left farm2 --right farm3` (full screen 3024x1964, 7 goals / 5 min via the bridge's new `settings set`).
It runs the game in real time and sets the virtual pads 15x/s; lockstep (`env.step`) made the game slow and jerky.

## Trainer throughput (2026-09-30)
- Real rate (the log's `sps` now covers the whole iteration; it used to leave out PPO): **~19k -> ~35k env
  steps/s**. Native runs roll out in `--procs` worker processes (`rollout.py`: each has its hosts, native
  env threads, a policy copy and a league bank; the learner process only runs PPO, on the Apple GPU,
  `--ppo-device mps`, ~2.5x the CPU). Best measured: `--hosts 12 --matches 38 --procs 6 --threads 18
  --torch-threads 1` 35.7k; procs 4 34.4k; procs 8 31.5k (physics-starved). Remaining cap: the PPO update
  is synchronous (~40% of an iteration).
- **Don't overlap PPO with the rollout** (`--async-ppo`, now opt-in): acting one update behind learned
  about half as much per step from scratch (goals/match at 3.5M steps: 2.5-2.9 overlapped vs 8.8 not;
  the pre-procs overlapped trainer 5.3-5.9). farm2 trained with the overlap from update ~5420 to ~6600.
- Checked: with `--procs 0 --ppo-device cpu` (rollout in-process, CPU learner) the trainer reproduces the
  previous one bit for bit over 235 updates incl. episode ends and league games. With several workers
  each updates its copy of the observation normalizer per step and the learner merges their stats.
- The game-backed (non-native) farm path is unchanged (in train_farm.py).

## The native simulator (done, bit-exact)
`sim/` — arm64 reimplementation of the match; **identical to the real game tick for tick** (tested
~100k+ ticks incl. goals, kills, sudden death; env outputs identical to the real-game farm).
- `sim/README.md`, `PORTING.md` (rules), `BOX2D_DIFF.md` (the game's Box2D = 2.1.2 + 6 changed
  constants + `SetFilterData` early-out inverted), `re_notes/` (layouts, functions, globals).
- x86 libm `sinf/cosf` ported bit-exactly (`sim/src/x86math.c`, `tools/libm/`).
- Env: `sim/src/env/` (batched, thread pool, observe2/reward natively), C ABI `sim/build/libsprsim.dylib`
  (`make -C sim env`; builds to temp + rename so live trainers keep their copy; `make -C sim env B=build-dev`
  builds a test copy elsewhere, `SPRSIM_LIB=sim/build-dev/libsprsim.dylib` makes native_env load it), Python
  `bridge/native_env.py` (`NativeHost`, `NativeFarm` with groups/pipelining, pit drops, ghost lock).
- Validation tools: `sim/tools/oracle.py` (dedicated game instance, `farm dump`), `trace_compare.py`,
  `diff_dump.py`, `test_env_equiv.py unit|python|oracle|dead` (run `python` after any reward/obs change; it
  now runs with the goal refund off and on, `--policy CKPT` for real play), `bridge/test_reward.py` (17 tests),
  `sim/tools/compare_libs.py OLD.dylib NEW.dylib [--policy CKPT] [--new-kw JSON]` (two builds bit for bit over
  6 env configs: run before replacing the installed library).
- Speed: 57 µs/tick/core (game: 625 µs). Training: ~30–45k env steps/s wall clock (was 3.7k).

## Game / map facts that bit us
- Training map = menu cell "arena 2" = **level id 1** (`Level1`, levelWidth 2800) = `arenas/arena2.json`.
  (Old docs called it "Flatirons" — unreliable; Level2 is menu cell 4.)
- Goal-wall **face at x = ±20.3 m** (the `wall` bodies are solid blocks 20.3..36.3, top y = −7.8);
  pits 16.3–20.3 m (floor 1.5 m below the plateau).
- **Pole ≈ 6 m** (10 segments 0.61 m apart); the tip sits 4.6–5.5 m from the body.
- **Dead riders** are tracked by `PlayState::respawnTimer` (+0x1a0), not `Player::ghost` (that's the
  0.3 s spawn ghost). Snapshots now carry `dead`/`respawn_t`; observe2 hides dead opponents (was a bug:
  the corpse looked like a live opponent for 3 s — farm1 never learned to exploit kills).
- Obs (observe2, 62 floats) and reward: see `bridge/farm_env.py`, `bridge/spr_env.py`. Reward =
  goal 1, kill 0.3, ball progress 0.003, position 0.1 (+ the goal refund rule, training default since
  2026-10-02: see the last section) (potential in **[-1, 0]**: 0 anywhere behind the
  ball, down to -1 at 10 m past it; **within half a pole of the ball the pole tip's side replaces the
  body's**, blended over 1 m — from the user's rule, confirmed in farm1 data).
  **Pitfall (cost one farm2 restart):** the potential used to be [-1, 1] (+1 at 10 m behind). Riders
  spawn 6.25 m behind the ball, so approaching it cost reward and a fresh policy learned never to touch
  the ball (0 goals, 0 kills after 790k steps, archived in runs/old/farm2_barrier_*). Keep shaping
  flat on the good side.

## Findings (data)
- **Fake score/clock (analysis/fake_score.py, 256 matches each, farm2 vs farm2):** a copy fed a frozen fake
  match state plays worse, mostly because of the score: "6-6" costs -1.6 to -1.9 goals/match (15 s or 300 s
  left), "0-0" only -0.2 to -0.5 (frozen inputs alone); control +0.09 ± 0.20. The real score is best.
- **farm2 (flat, from scratch, all fixes) beats everything early** (eval_h2h, 32 matches, no sudden death):
  at 17.3M steps vs farm1 at the same steps 22-2-8 (+1.1 goals/match), vs farm1 latest 25-3-4 (+1.6),
  vs manager2 29-0-3 (+1.8). farm1 latest is barely better than its own early snapshots (+0.5 vs u700):
  its late training (pit drill etc.) didn't make it stronger. farm1/manager2 are evaluated with
  `set_legacy_obs` (they were trained before the dead-rider fix; with the new obs they lose even more).
- farm1 (flat PPO): excellent on the right side of the ball; weak near the ball and in wrong-side
  situations. Situation scoreboard (`analysis/situations.py`, goal diff next 10 s): right-far +0.19,
  right-close +0.10, **A you-opp-ball −0.58**, **B opp-you-ball −0.33**; kills worth only +0.13 (blind to
  dead opponents). Pole-tip rule: near the ball, tip behind the ball +0.11..+0.15 vs past it −0.07..−0.17.
- Pit drill (`--pit-frac`) made farm1's pit time **worse** (1.1% → 5.9%) — not used for farm2.
  10 m shaping + std floor helped past-the-ball only a little (38% → 29% time, recovery 29% → 31%).
- **manager2** (hierarchy, frozen OLD worker) reached **parity with farm1 head-to-head** after 46M env
  steps (7851 games: goal diff +0.05/game; last 800: +0.12).
- Distilling farm1 into manager targets (`distill_manager.py`, `runs/distill/`) gave weak play through
  the old worker (1.2 goals/min vs farm1 2.0–2.4). `runs/distill/farm1_rollout.npz` (405 MB, obs+actions)
  is ready for worker behaviour cloning if needed.

## Worker (goal-conditioned executor) — `train_worker.py`
- Targets v3: body on the ground (x) or **in the air** (pass within 1 m of (x, 1.5–7 m above the
  surface) = a jump, 40% of body targets), pole tip point (0–6 m high, reach 5.3 m), "both" (tip on the
  4.3–5.6 m reach ring around the body target; +0.02/step per satisfied part). All targets reachable
  (checked against arena polygons). Mirror frame fixed per target ("reverse the controls", user's idea).
- Opponent mix per match: solo 0.5 (other rider ghost-locked, shown alive), vs farm1 0.3, "dead" 0.1
  (opponent shown dead), duo 0.1.
- Status when paused: ground body ~92%, pole tip ~79%, both ~38%, **air 18%** (1.5–3 m ~35%, higher
  jumps low). Std floor 0.15 (exploration re-opened for jumps).
- Compare checkpoints: `analysis/worker_compare.py A B`; tip study: `analysis/tip_control.py`.
- Known: in failed "both" targets 34% were body/tip swapped (partial credit added to fix).

## Next steps (agreed direction)
1. **farm2**: let it run; compare with farm1 via the situation scoreboard and head-to-head.
2. **Killer specialist** (user's idea): kill-only reward, from farm1/farm2, e.g.
   `train_farm.py --run killer1 --native --init-from runs/farm2/ckpt.pt --reward '{"goal": 0, "kill": 1, "ball_progress": 0, "position": 0}' --min-std 0.15 ...`
   Uses: worker opponent (stop dying so stupidly) and a manager option (hunt).
3. **Worker**: resume; jumps (maybe a height curriculum); add killer as an opponent share.
4. **Manager v3**: options = {flat policy plays normally (farm1/farm2), worker target, killer}; decide
   every ~4 worker steps with an explicit "keep target" and a "null/hold" choice (user); update
   `bridge/hier_env.set_goals` ranges (still the OLD mapping: tip 1–9 m, body ±16 m, no height) to
   body ±8 m + height, tip 0–6 m, "both" on the ring. Train with `--flat-league` for the head-to-head.
5. Later: lookahead planning in the deterministic sim (AlphaZero-style), check policies in the real game
   (oracle farm), maps (parked).

## User preferences
Faithful to the real game; sparse rewards (goal 1, kill 0.3, tiny ball progress) + potential shaping for
diagnosed weaknesses; skill = head-to-head/vs_pool, not goal counts; train on menu arena 2; wants data,
the dashboard, and to be asked about game mechanics before designs are invented. `(pa)` = needs approval.

## Gauntlet + search (2026-10-02; replaces PBT, which stopped after generation 1)
- `gauntlet.py`: frozen members in runs/gauntlet (farm2 @5/10/25/50/100/200M + latest, farm3 @10/50/200/400M +
  latest, killer2, manager2 (hier), pbt1_m1 @25M/@50M). `fill` = round robin (64 matches/pair), Elo by
  Bradley-Terry (farm2 latest = 1500); `rate CKPT --name X` = X vs every member with member ratings held fixed
  (comparable over time); `exploit X --steps 30e6` = exploitability (copy of X trained only vs frozen X:
  `train_farm --league-frac 1 --league-fixed target:X:1.0`; league_frac > 0.5 now also turns even slots into league).
- `search.py --name search1`: Optuna TPE samples 24 trials (baselines farm2 recipe + pbt1_m1 genome first),
  successive halving 10M -> 30M -> 90M (best third promoted, after the whole rung is rated), value = gauntlet Elo.
  4 trials at a time (hosts 4 x matches 24). Runs: runs/search1_t<N>; state runs/search1/state.json + study.db;
  `search.py --name search1 --report`. Net size deliberately not searched (short screens punish bigger nets):
  test 256 vs 512 on the winner at long horizon instead. train_farm now saves its final weights at --total-steps.
- Real controllers: the game's 2014 SDL has no DualSense mapping (Steam Input normally hides that); launches now
  pass SDL_GAMECONTROLLERCONFIG from bridge/pad_mappings.py (prmod run + spr_env.launch_game). Bridge
  `joysticks` lists what SDL sees (name, GUID, mapped?). Patcher launches now set SPRBRIDGE_NOMOVE (the PS Move
  poll stalled menus to ~1/7 speed). play_human: a human's controller plays as itself (no mirror); during the
  scripted menus it is driven by the script via bridge `pad override <slot> on|off` (its own input ignored), so a
  touched button / stick drift can't add a third controller to a team.
- Human rating: `play_human.py --left pad --right gauntlet [--matches 8 --player you]` (patcher: "gauntlet: rate me").
  Adaptive placement: each match vs the flat member rated closest to the current estimate (start: the strongest),
  7 goals / 5 min, a tie at time = draw (sudden death skipped). runs/gauntlet/human.jsonl per match; aggregate rows
  as "human:<name>" in games.jsonl + candidates.json (dashboard Gauntlet tab). ~±120 Elo after 8 matches.

## Goal refund (training default), ball touches, concepts v2 (2026-10-02 evening)
**Goal refund — the user's reward rule, ON by default for all training** (native `GoalRefundState`,
sim/src/env/features.h; Python mirror `spr_env.GoalRefund`, used by FarmHost when `goal_refund=True`):
(a) from the goal step until the game's post-goal reset (ball rebuilt + both riders respawned in one tick, ~1.4 s;
seen as the ball jumping > max(5, 1.5 fs) m without a goal) only the goal is paid: no kill/death, ball progress,
position potential or rotation bonus for either rider. The goal and reset steps already had no shaping (teleport
rule) and the next step starts from the respawned state, so there is no potential jump. A match-deciding goal has
no reset: the ~5 s end phase (1 s clock, slow motion, fade) is all masked. (b) If a rider dies (penalty paid) and
its team scores while it is still dead, the death is refunded in full at the goal step and the killer's kill reward
taken back (zero-sum: +/-kill; death weights: victim +death, killer -kill). Only `rew` changes; `terms` stay raw.
With `--goal-episodes` (episode ends at the goal step, from terms) the refund is the last reward of the episode that
holds the death, and the masked window opens the next episode with zero rewards.
- `NativeHost(goal_refund=True)` (default), NativeFarm/HierHost via cfg; C library default off (`sprenv_set_goal_refund`),
  so an old Python keeps the old reward. `train_farm.py --no-goal-refund` = the old reward (saved in ckpt args as
  `goal_refund`). Statistics stay on the old counting: eval_h2h/gauntlet (`COUNT_KW`), fake_score pass
  `goal_refund=False`; concept_report too (it only reads terms). Other NativeHost users read obs/terms only.
- **Installed 2026-10-02 17:17:40** (library + Python). search1 trials t26-t28 (started 17:09-17:10) and everything
  before trained WITHOUT it; any search1 trial started or resumed (promoted) after that trains WITH it.
- Evidence: rule off = bit-identical to the previous library (compare_libs, 6 configs x 3000 steps x 32 matches,
  random + farm2/farm3 play, old and new Python); native == Python mirror with the rule on and off
  (test_env_equiv python, incl. 4000 steps x 16 matches of farm2 play: 159 goals, 243 kills); rule on vs off in farm2
  play (16 x 4500 steps): rew differs only in goal->respawn windows (150 goals, 3461 window steps, 34 of 267 kill
  steps masked) and at 3 refund goals. Tests: `bridge/test_goal_refund.py` (7: window mask incl. a kill inside it,
  refund while dead, no refund after respawn / when the other team scores, symmetric take-back with death weights,
  match-deciding goal, real-play classification; test-only hook `NativeHost.debug_event(k, "kill"|"goal", who)`).
**Ball touches** (read-only polling of the ball's Box2D contacts after every tick; sim unchanged): `out["touch"]` /
`out["touch_final"]` float32[N, 2 riders, 5] (native_env `TOUCH_*`): seconds since the rider last touched the ball,
what touched (1 body, 2 pole, 4 pole handle), seconds since its last kick (the game's kick impulse = its
ballNoCollideTimer jumping to 0.6), touching now, the game's own lastTouch is this rider; 999 = never this match.
Ages count ticks incl. hit-stop. `bridge/test_touch.py`: kicks == the game's kick stats (351/351), lastTouch == goal
credit. Game facts found: pole segments 0-2 carry the rider's own tag, so a ball contact on the pole's handle counts
as the RIDER (it kicks, sets lastTouch); a kick switches the kicker's ball collisions off 0.6 s (body + pole); one
kick per tick per match. The real-game bridge does not export touches yet (hook: PlayState+0x30 lastTouch,
Player+0x1cc ballNoCollideTimer in snapshot_ps; not done: untestable without launching the game).
**Concepts v2** (concepts.py, from the user's 2026-10-02 notes in docs/strategy.md; `Frames(..., touch=TC)`):
states committed, at_mercy, high_energy (E = 0.5 v^2 + 20 (y - 0.69) J/kg, >= 165 = farm2's top quarter); events
takeoff, fast_takeoff (>= 14 m/s), lunge_at_opp, lunge_from_pit, overcommit, overcommit_opp_still, got_flung,
flung_own_goal, second_touch, second_kick, impale_plant, impale_brake (touch ones empty without touch rows).
`analysis/concept_report.py` collects the touch rows and adds an "outcomes" block (lunge -> overcommit / goal against,
value by energy / energy lead / take-off speed quartile, second touch/kick consequences, impale brakes).
Report `analysis/concepts_report_v2.txt/.json` (128 matches per line-up; self-play farm2 427M, farm3 684M, search1 t2
@10M (best Elo, 1237; gauntlet frozen copies), h2h farm3-farm2 34-22-72 (-1.12/match), t2-farm2 16-14-98 (-2.09)):
- Second KICK (2.6-3.0/min): ball moving the second kicker's way 0.5 s later 90-97% (any second touch: 58-69%);
  P(goal for / against within 5 s) after it ~0.04-0.06 / 0.014-0.033 (baseline against 0.05-0.08); the kicker is
  thrown back ~8 m in 1 s (19.6 -> 21.5 m/s); value +0.04 (farm2), ~0 (farm3, t2).
- Lunge at the opponent ~0.1-0.2/min; 33-47% of lunges end in an overcommit (flip to [ball, opp, me]); P(goal against
  within 5 s) after an overcommit 0.19-0.42 vs 0.05-0.08 baseline; overcommit value -0.12..-0.52. The opponent was
  "still" in 0-2% of overcommits (bots never stand still: his median speed at the flip 6-14 m/s).
- Energy: value rises with energy (farm2 bottom quartile -0.08, upper half +0.03..+0.05) and more clearly with the energy
  LEAD over the opponent (farm2 -0.13 -> +0.07 across quartiles; same order for every policy). Take-off speed: top
  half +0.04, bottom half ~-0.01 (farm2; t2 only the top quartile helps).
- committed ~45% of time (t2 32%), value ~0; at_mercy 4-7%, value +0.06..+0.10 (correlational: mostly pressure moments).
- Impale plants: farm3 0.55-0.65/min (10x farm2), 60% shed speed, value -0.2..-0.26 (it happens when already flying the
  wrong way). Got flung ~0.5/min, value +0.08 for farm2 (riders use the other's pole as a springboard); flung own
  goals 1-12 per 128 matches.
Rejected by the user (not built): league opponents that randomly do nothing ("you need to expertly know when to
freeze, 99% of the time it's autolose").
- Bridge `hud <text>` / `hud off`: a text line drawn over the bottom of every frame (CoreText bitmap + glDrawPixels in
  the CGLFlushDrawable hook). play_human shows "you vs <bot> · match i of n" in every mode.
  `play_human.py --left pad --right series:t27,t2,t7,t24` = one match per bot in turn (tNN = runs/search1_tNN;
  checkpoints frozen to runs/human/opponents/<name>_<steps>M.pt; results in runs/human/series.jsonl).

## Network options: --arch / --hidden / --layers / --action-head (2026-10-02 evening)
All opt-in `train_farm.py` flags, saved in the checkpoint args (`arch`, `hidden`, `layers`, `res_expand`, `action_head`).
The default network (2x256 tanh MLP + Gaussian, keys pi.0/2/4, v.0/2/4, log_std) is unchanged **bit for bit**. Old vs new
code, fixed seed, tiny native runs, everything identical (log.csv minus sps, ckpt net/opt/norm, every pool snapshot, events):
procs 0 and 1, CPU and MPS learner, search-trial flags, `--resume`, `--init-from`, pool league + `--league-fixed` farm2.
eval_h2h.play_flat is also identical (seeded). `--hier` runs differ from run to run even with the old code, so they were not compared.
- `--arch mlp|resmlp`, `--hidden N` (256), `--layers L` (2), `--res-expand E` (4). resmlp = SimBa: Linear in->N, L blocks
  `x + fc2(relu(fc1(LN(x))))` (N -> E*N -> N), final LayerNorm, head. Policy and value towers are separate. Init: orthogonal,
  gain sqrt 2 (fc1/input) and 1 (fc2). The policy head is x0.01 as before, the value head has gain 0.3. Keys: pi.inp /
  pi.blocks.i.{ln,fc1,fc2} / pi.ln / pi.out.
- `--action-head gated` (`--native`, flat stick policies only): the policy also outputs 2 gate logits. For each stick
  (left = run; right = aim_x, aim_y) a Bernoulli "touch this stick" gate is drawn, plus the existing Gaussian for its
  value (`train_ppo.GatedNormal`). Stored action = [run, ax, ay, g_run, g_aim]: the rollout buffer and PPO data have 5
  columns. The env gets `value * gate`. log p = sum_k [log Bern(g_k) + g_k log N(stick k)]. The entropy is exact:
  sum_k [H(p_k) + p_k H_N(stick k)]. Gate 0 = exactly 0 on the wire. In the game that means the 0.1 pad dead zone gives 0,
  there is no run force (|lx| <= 0.2) and no airborne kick aim (len <= 0.8). **A neutral right stick turns the pole servo
  OFF** (magnitude <= 0.2, sim player.cpp): both hand-joint motors get speed 0 and max torque 0, so the pole swings
  freely. It "lets go" — it does not hold the angle. Gate init logit +2 (p = 0.88 touching; `GATE_INIT_LOGIT`).
  With 0 (p = 0.5), a 1.5M-step smoke run let go of both sticks 87% of the time: the early do-nothing attractor.
  farm2 has the servo off 1% of the time. stdout prints `gate on run / aim` (the mean gate probability).
- Loading: `train_ppo.net_from_ckpt(ck_or_path) -> (net, norm, obs_size)` rebuilds the network from the state_dict alone
  (`spec_from_state`), so pool snapshots without args load too. It draws random numbers exactly like the old
  `ActorCritic(size, act)`. Used by eval_h2h.load_policy (-> gauntlet, pbt, concept_report, h2h_breakdown, fake_score,
  pole_rotation), rollout Pool + `--league-fixed`, train_farm `--watch`, play_human, play_real, dashboard spectators,
  hier_env.load_worker, train_worker flat opponents, analysis situations / pit_stats / past_ball / ball_awareness.
  Env actions come from `net.sample_env(x)` / `net.mean_env(x)` (gauss: exactly `dist(x).sample()` / `pi(x)`).
  `net.act` returns the stored action for PPO, and `net.env_action(a)` maps it to the env action.
  `--resume` takes the checkpoint's network (a contradicting flag is an error). `--init-from` adopts the source's
  network (flags override; a mismatch is an error; `--hier` forces gauss).
- LeagueBank: plain opponents (tanh MLP, 2 layers, width <= 256, gauss) keep the batched bmm path. Narrower ones are
  zero-padded, which is exact (max |diff| < 2e-6 vs the module). Every other opponent (resmlp, deeper, wider, gated) runs
  its own module: one batched forward per distinct opponent over its slots. When all slots are plain, the code path is
  the original, bit for bit.
- Tests: `test_netopts.py` (8 tests). Default net = the original construction draw for draw. Spec round-trip for 7 specs.
  Real checkpoints load. Gated log-prob = float64 brute force (1.8e-15). Entropy = quadrature, and Monte Carlo agrees.
  Gate frequencies are right. Gate 0 = exact zeros on the wire (native stick). The bank's padded path, the bit-identical
  plain path and mixed slots are covered.
- Costs (2026-10-02 18:00-18:30, machine shared: load average ~40 on 18 cores, so read the ratios). Forward = one
  `net.act` call, 1 torch thread, batch 96 (one rollout group of the search config) / 76 (farm3 config), microbenchmark.
  Training = env steps/s for `train_farm --native --hosts 4 --matches 24 --procs 1 --threads 4` (the search-trial setup,
  MPS learner, median over 10-23 updates). Under normal load the default does ~13k steps/s there.

  | net | params | forward ms (B 96 / 76) | rollout policy ms/step | env steps/s | vs default |
  |---|---|---|---|---|---|
  | mlp 256x2 (default) | 0.16M | 0.73 / 0.66 | 3.2 | 6016 | 1.00 |
  | mlp 256x2 gated | 0.17M | 0.87 / 0.77 | 4.1 | 5848 | 0.97 |
  | mlp 512x2 | 0.59M | 1.6 / 1.4 | 7.4 | 5375 | 0.89 |
  | mlp 256x3 | 0.30M | 1.2 / 1.0 | 5.1 | 5659 | 0.94 |
  | resmlp 256x2 (E4) | 2.1M | 2.4 / 2.2 | 10.8 | 3823 | 0.64 |
  | resmlp 512x3 E2 | 6.4M | 6.2 / 5.3 | 29.2 | 2005 | 0.33 |
  | resmlp 512x3 E4 | 12.7M | 12.0 / 10.4 | 59.1 | 1046 | 0.17 |

  League opponents of a non-plain architecture cost one forward per distinct opponent per step. For 24 league slots:
  24 distinct 2x256 = 0.85 ms (bmm); 24 distinct mlp 512x2 = 2.2 ms; 24 distinct gated 2x256 = 2.2 ms; 24 distinct
  resmlp 512x3 = 16.6 ms; 6 distinct resmlp 512x3 = 10.4 ms. (Pool snapshots change only between matches, but a 512x3
  pool makes the league as expensive as the policy.) A CPU PPO minibatch of 4096 for resmlp 512x3 E4 took ~2.6 s, vs
  ~0.1 s for the default.
- Smoke runs (1.5M steps each, 2 hosts x 16 matches, procs 1, 25% frozen farm2 in the league, seed 1). Every update had
  finite losses, and std went 0.61 -> 0.56-0.58 for the default, gated and resmlp 512x3. Gated (gate init +2): vs_pool
  +2.2 over the last 20% of updates, 5.6 goals per match, gates 0.7-0.9. Rated in a scratch gauntlet copy
  (`rate --games 4 --only farm2@5M,farm3@10M`): every smoke checkpoint loaded and lost every match (as expected at
  1.5M steps). 1.5M steps is far too short to rank architectures.
- Not changed: train_worker's own worker net (always mlp), distill_manager, the legacy train_ppo.train, sim/tools.

## Snapshot / restore + lookahead planner (2026-10-02 evening)
**Per-match arenas (installed 18:20, `sim/build/libsprsim.dylib`):** every allocation a match's game code / Box2D makes
(operator new, b2Alloc) goes into that match's own arena (`sim/src/arena.*`; routed by `EngineContext::arena` of the
running `g_engine`, so PitDrop/Scramble/GhostLock/DebugEvent are covered automatically; the library's operator new is
not exported, torch keeps its own). `NativeHost.snapshot(k, slot)` / `restore(k, slot)` (16 slots) = memcpy of the
arena (~190 KB, ~5-10 us) + the env-side state (reward-rule state, touch counters, curriculum RNGs, the action log).
Not in a snapshot: the host's sudden-death coin stream (sd_frac > 0), config switches. If arenas ever run out
(16384 per process) new matches fall back to the heap (no snapshots) instead of crashing.
**Planning branches** (`plan_init(owner)`, `plan_begin(active)`, `plan_step(acts[B,2,3])`, `plan_end()`, `plan_json(b)`):
env-owned clones of match owner[b]. A clone is NOT a relocated copy: each branch is a replica kept in step by
re-playing its match's action log since its last reset (lazily at plan_begin: ~1 step per decision, a full replay
after the match was reset/restored) — exact by construction; cost ~1/H of the search. Then begin = save the
branch's root, step = one decision on every active branch (no auto-reset; a finished branch is frozen, done=1),
end = restore to root. Branches use the env's reward config (planner: the critic's).
- Evidence: with snapshots unused the library is bit-identical to the previous one (compare_libs, 6 configs x 3000
  steps x 32 matches, random + farm2 + farm3 play, now incl. touch rows); test_env_equiv unit/python/dead, test_goal_refund,
  test_touch, test_reward, test_concepts pass. `bridge/test_snapshot.py`: twin envs (one snapshots + wanders off 1-40
  steps with junk actions + scripted goals/kills + restores, ~430 restores per config) stay bit-identical in every
  output and the JSON over 2000 steps x 16 matches x 5 configs (training rules + refund, short format with sudden death
  and match ends, random pole + goal spawns, scramble + pit drops, ghost lock + legacy obs); planning branches
  reproduce the recorded future step for step (~80k branch steps per config, incl. match ends, root restores, begin
  without end). `check_arena(k)` (debug): game objects outside the arena = 0.
**planner.py** (prototype; numpy inference): per decision K candidate first actions (policy mean, "do nothing",
K-2 policy samples) x M opponent models (opp = the opponent's own policy sampled, oppmean, farm2, zero, random), held
--hold decisions then the policy; H decisions; score = discounted path reward + gamma^H V(leaf) with the critic's own
reward (farm2: default weights, no goal refund, gamma 0.998); aggregate over models (--agg mean|min|meanstd:L|softmin:T).
Eval vs plain farm2 (eval_h2h rules, 32 matches, sides swapped; runs/planner/*.json):

| planner (farm2 + ...) | W-D-L | goal diff (95% CI) | min/match | decisions/s (32 matches, 7 thr) | ms/decision, 1 match |
|---|---|---|---|---|---|
| none (plain farm2) | 13-2-17 | -0.38 (-1.42, +0.67) | 4.87 | — | — |
| K1 = policy mean only | 12-5-15 | -0.38 (-1.16, +0.41) | 4.80 | — | — |
| 8x4x15 (opp,oppmean,zero,random; meanstd:0.5) | 32-0-0 | +7.00 (all 7-0) | 1.55 | 22 | 31 |
| 8x4x15 agg min | 32-0-0 | +7.00 | 1.77 | 49 | — |
| 16x2x8 (opp,oppmean) | 32-0-0 | +7.00 | 1.75 | 69 | — |
| 8x2x15 hold 2 (opp,zero) | 32-0-0 | +6.97 | 1.84 | 89 | — |
| 8x1x15 (opp) | 32-0-0 | +7.00 | 1.64 | 128 | 11 |
| 4x2x8 (opp,oppmean) | 32-0-0 | +7.00 | 2.00 | 331 | 6.6 |
| 4x1x4 (opp) | 32-0-0 | +6.63 (6.43, 6.82) | 2.68 | 981 | — |
| 2x1x4 {mean, do nothing} | 32-0-0 | +6.56 (6.35, 6.78) | 2.96 | 1573 | 2.1 |
| 2x1x4 {mean, 1 sample} (--no-zero) | 20-2-10 | +1.19 (0.12, 2.26) | 4.82 | 1607 | — |
| 8x1x15 opponent modelled as "zero" (wrong model) | 1-0-31 | -4.06 (-4.74, -3.39) | 3.14 | 178 | — |
Single-match timings under the machine's training load (load avg 30-40). Readings: planning crushes farm2 even at
tiny budgets; most of it is knowing WHEN TO FREEZE (the do-nothing candidate wins ~35% of decisions at K=2; without it
+1.2) — matches the user's "expertly know when to freeze"; a correct opponent model is essential (wrong model: -4).
Caveat: the planner knew the opponent's exact policy (the realistic case is an unknown opponent).
Next: plan vs other opponents with farm2 as the model (farm3, gauntlet, humans) and with model ensembles / online
model selection; use planner play as targets (distill "plan -> policy", expert iteration); real-game play via the
bridge needs the sim state from the game (oracle snapshot -> native state), not built.

## Human vs bot concepts from the real-game logs (2026-10-02 evening, analysis/human_concepts.py)
`.venv/bin/python analysis/human_concepts.py` (any runs/human logs; ~25 s) -> analysis/human_concepts_report.txt/.json:
per match + pooled (human / bot, all / vs farm2-family / vs search trials) shares, rates, the human's 10 s goal-diff
values with n (`*` = n < 20), and human - bot differences ranked by a paired t over matches.
- **Play-log fix (concepts.py):** the logs hold a snapshot every 4 RENDERED frames and the user's display runs at
  120 Hz, so rows were 1/30 s apart (2 game ticks), not 1/15 s: the old `playlog_matches` halved every rate and
  window. The logs also never set `game_over` (the slow-motion end after the deciding goal was "live").
  `playlog_matches` now resamples by the match clock to one row per 4 ticks and cuts at the deciding goal / end of
  time (`playlog_read/ticks/end/resample`; `resample=False` = the old path); test added (test_concepts.py, 28 OK).
  analysis/playlog.py (BC demos) still uses every snapshot (1/30 s) and keeps the end-phase rows: fine for
  (obs, action) pairs, but don't read its rows as decision steps.
- The game's stats.kicks also count stomp kills (contacts.cpp PlayerCollision: AddKill + AddKick); ball kicks =
  kicks - kills. Ball kicks are exact from those counters; other touches are inferred (ball velocity jump that the
  zipline-spring model doesn't explain, a rider within 1.3 m): 95% of exact kicks show up as such a jump.
- Data: 13 matches (9 gauntlet vs farm2 / farm2@200M incl. the first plain one vs farm2, 4 series vs t27/t2/t7/t24
  @~20M); log-reconstructed scores = recorded results 13/13, concept kills = game kill stats 13/13. Excluded:
  20261002-150439 (3-min mirror-pad test match, 3-1 at time, no result row) and 20261002-181628 (2 s, aborted).
  Real-game farm2 rates ~ its sim self-play rates (airborne 72 vs 73%, vaults 23 vs 24/min, takeoffs 28 vs 27/min).
- Headlines (human vs bot, 13 matches, h>b = matches where the human is higher): pole carried low (fwd-up 24 vs 39%,
  0/13; back-down 18 vs 11%, 13/13); "still" (body < 2 m/s, pole < 1.5 rad/s) 9.3 vs 3.5% (13/13; most in B: 16%
  of B time); fewer but faster jumps (takeoffs 24 vs 28/min, >= 14 m/s 12.3 vs 7.4/min, median 14.2 vs 11.3 m/s);
  backward swing in flight 11 vs 6% (13/13); tip behind the ball 18.6 vs 13.7% (12/13); impale plants 0.78 vs
  0.07/min; less at mercy (5.9 vs 8.5%) and less on the opponent's pole (1.0 vs 2.0%, flung 0.39 vs 0.81/min);
  more catapults (0.80 vs 0.39/min, 9/13); A 3.7 vs 5.6% but B the same (11.8 vs 11.6%); kicks 16.7 vs 13.2/min.
  Second kicks: bots more per minute (4.1 vs 2.9) but per kick of the other side both reply ~22-24%: no evidence
  the human hits second more. Lunges / overcommits are too rare to say anything (10 vs 14 lunges, 2 vs 7 overcommits).

## Planner vs mismatched opponent models (2026-10-02 night; see "Snapshot / restore + lookahead planner" above)
runs/planner/mm_*.json, t20plan_*.json. farm2 + planner whose opponent
model is farm2's own policy (as against an unknown opponent), vs other policies (gauntlet frozen copies; search trials
copied to runs/planner/opps/*_now.pt at ~19:50, t20 = 52M steps). 32 matches each, sides swapped; goal diff (95% CI):

| opponent | plain farm2 | 8x1x15 {farm2} | 4x2x8 {farm2, zero} | 4x3x8 {farm2, zero, farm3} | 4x3x8 {farm2, zero, random} |
|---|---|---|---|---|---|
| farm3 | +1.31 (+0.27, +2.35) | +2.09 (+0.79, +3.40) | +4.59 (+3.91, +5.28) | +6.50 (+6.21, +6.79) | +4.94 (+4.44, +5.43) |
| t20 | -0.41 (-1.75, +0.94) | +4.75 (+4.13, +5.37) | +4.22 (+3.52, +4.92) | +5.94 (+5.61, +6.27) | +5.03 (+4.29, +5.78) |
| t27 | +1.25 (+0.09, +2.41) | +3.84 (+3.13, +4.56) | +5.50 (+4.98, +6.02) | | |
| t2 | +1.09 (-0.02, +2.21) | +4.97 (+4.51, +5.43) | +5.66 (+5.23, +6.08) | | |
| pbt1_m1@25M | +3.38 (+2.55, +4.20) | +5.97 (+5.56, +6.38) | +6.62 (+6.41, +6.84) | | |
| killer2 | +7.00 | +6.84 | +6.97 | | |
Decisions/s (32 matches, 7 threads): 8x1x15 165-260, 4x2x8 ~290, 4x3x8 ~320 (plain ~13k). Minutes/match: plain 4.2-4.8,
planner 1.9-3.8. Base t20 (own policy as model): t20 vs farm2 plain +0.31 (-1.09, +1.72); t20+planner 8x1x15 vs farm2
+6.78 (+6.64, +6.93), 4x2x8 {self, zero} +6.75; t20+planner vs plain t20 (exact model) 32-0-0 +7.00.
Readings: search still wins big with the wrong model (every opponent, every config, CIs far above plain), but less than with
the exact model (7.0), and model DIVERSITY matters more than budget: adding "do nothing" as an opponent model (robust
mean-0.5std) beats the bigger farm2-only search on 4 of 5 real opponents; a set containing the true opponent (farm3) is
best (+6.5 vs farm3; it also helps vs t20, +5.9); adding random instead helps a little. For humans: no policy -> use a
broad model set (several bot policies + do-nothing + random, robust aggregation), ideally a model fitted online to
the human's play / behaviour cloning on runs/human demos; expect less than vs bots (humans are further from every model).

## search1 result + overnight batch (2026-10-02 21:30)
- search1 done: t20 (lr 7.4e-5, ent 1.3e-4, min/init std 0.16/0.30, horizon 858, lam 0.87, kill 0.05, pos 0.02, bp 0.0035,
  rot 0.0015/21M, goal_spawn 0.17, league 0.23, score visible, no random pole) = 994 / 1402 / 1797 Elo at 10/30/90M;
  beats farm2 latest 37-2-9 (+2.67) and farm3 45-1-2 at 90M. Fast/exploratory recipes regressed (t27 1353 -> 834, t2 -> 1269).
- `ops/overnight.py` (log runs/overnight.out): search1_t20 continued to 300M; t20 recipe with seed 1 (n1_t20_seed1),
  with `--action-head gated` (n1_t20_gated), with `--arch resmlp --hidden 256 --layers 2` (n1_t20_res), each 90M;
  gauntlet-rated every 30M as "<run>@<N>M".

## t20 vs the user + concepts 2026-10-03 (analysis/t20_vs_user.py -> analysis/t20_vs_user_report.txt/.json)
The user played 3 real matches vs frozen t20_300M (runs/human/20261003-03*; 1-7, 5-5 at time, 4-7: first non-wins). Play logs
now end at the end of regulation when a match goes to sudden death (house rule: a tie at time is a draw;
`playlog_end(..., sudden_death=True)` = old behaviour). human_concepts.py: group "t20@300M" (+ a "t20@300M vs not" Welch
section), logged sticks -> action rows (`C.playlog_actions`) so the pole-servo concepts work on real games.
- **New concepts** (concepts.py, tests in test_concepts.py, 34 OK): `servo_off` (the game's pole servo is off when the stick,
  after the 0.1 per-axis dead zone, is <= 0.2 long: the pole hangs free = "letting go"), `mutual_pole_contact`, `interlocked`
  / `pole_interlock` (each pole on the other RIDER >= 1 s, gaps <= 1/3 s bridged — the two poles never collide, collision
  filter in pole_ctor.cpp), `interlock_let_go`, `let_go_escape` (servo off at the end, released <= 0.5 s before the last
  contact), `open_goal` / `open_goal_conversion` (opp dead or > 5.5 m on his wrong side), `aim_back_pass`.
  `Frames(..., act=[T, ..., 2, 3])` = actions (me, opp). Episode/outcome code: analysis/concept_outcomes.py (also used by
  concept_report.py, which now records actions; `analysis/concepts_report_v3.*` = t20_300M + farm2 self-play incl. these).
  Contact proxy vs the real polylines: precision 0.96 / recall 0.89 (sim rows), 0.97 / 0.86 (real t20 games).
- The user's named situation [t20, user, ball] = **situation A for t20** (sit_A). Headlines (n small for real games):
  t20 conceded within 10 s from A 62% (n=16; user in A 34%, n=38; t20 self-play 54%, n=2001); in A t20's stick drives the
  pole backward 7% of the time (sim 10%; the user 54%) — it does back-rotate elsewhere (11.7 back passes/min in sim vs farm2 3.9).
  Interlocks are a t20 trait: 0.91/match-min vs the user, 0.78 in t20 self-play (10% of play, some held 4.5 min to the end),
  farm2 0.29; bots never let go (1 of 447; servo off 1% of play vs the user 30%). Open-goal conversion: t20 30% vs the user
  (n=30), earlier bots 12% (n=113), t20 self-play 43%.
- Pole servo (player.cpp): absolute target angle, error wrapped to +-180 deg (always the short way) -> a backward full turn
  needs the stick swept round ahead of the pole; suggestion list (relative-aim action head, release bit, A start states,
  concept prediction targets, let-go league opponent (needs the user's OK)) at the end of the report. Nothing implemented.
- Sim caches: runs/concepts_cache/ (t20_300M / farm2 self-play 128 matches with actions + touch rows, ~380 MB each;
  calib_8_s777.jsonl = 8 t20 matches with full snapshots). `t20_vs_user.py gen|calib|report`.
