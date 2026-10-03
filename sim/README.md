# sim/ — native Super Pole Riders match (work in progress)

A native arm64 reimplementation of one Pole Riders match, bit-exact with the real game (which runs
x86_64 under Rosetta), for fast RL training. Porting rules: `PORTING.md`. Plan: `../HANDOFF.md`.

## Status
- **Box2D**: the game's Box2D is stock 2.1.2 with six changed `b2Settings.h` constants
  (`BOX2D_DIFF.md`); `Box2D/` has them applied. **Verified bit-exact**: the real Flatirons world
  rebuilt natively from a dump and stepped 600 times with bare `b2World::Step(1/60, 60, 18)` matches
  the game's own physics field for field (every body, fixture, joint, contact, broadphase node).
- **libm**: only `sinf/cosf` (Box2D rotations) differ between x86_64 and arm64 libm; `src/x86math.c`
  is a port of the x86 routine, verified over all 2^32 inputs (`tools/libm/`).
- **Game objects**: layouts recovered (`src/game/layout.h`, `re_notes/layout.md`); function
  inventory and globals in `re_notes/`.
- **Game logic port**: packages A (construction, Level1 = training map + Level2), B (player tick),
  D (input chain) done; C (PlayState::Update, contacts) finishing. With everything linked the native match
  is **identical to the real game tick for tick**: 12 runs / ~76k ticks (package B) + 2 x 5000 ticks on
  fresh seeds, incl. kicks, kills, respawns, goals, sudden death. The game's Box2D also has one code
  change: `b2Fixture::SetFilterData` never flags contacts (BOX2D_DIFF.md).
- **Speed** (`build/bench`): 57 us/tick on one core (real game: ~625 us/tick), ~178k ticks/s on 16
  threads while 14 training games share the machine.
- Level ids: the training map (menu cell 2) is **level id 1** (`Level1` class, levelWidth 2800);
  `Level2` (menu cell 4) is also ported. `tools/oracle.py` verifies the level it boots into.

## Oracle workflow
- `tools/oracle.py` — launches a dedicated game instance (socket `spr-oracle.sock`, separate from
  training), starts a Flatirons match, freezes it, creates shadow matches. `Oracle().boot()`;
  `.reset(k)`, `.dump(k)` (bridge `farm dump`), `.physstep(k, n)`, `.act(acts, n)`.
- `tools/dump2load.py` + `build/physcheck` — rebuild a world from a dump and step it natively.
- `tools/gen_layouts.py` — exact field layouts from clang (`build/layouts.json`; rerun after
  editing `src/game/layout.h`).
- `tools/diff_dump.py real.json native.json` — typed field-by-field diff (pointers by identity,
  padding/dead storage/presentation fields skipped).

## Build
`make -C sim build/gamecheck build/bench` (arm64, `-O2 -ffp-contract=off -fno-fast-math`).

## Batched RL env (`src/env/`, `build/libsprsim.dylib`, `../bridge/native_env.py`)
N native matches on a persistent C++ thread pool with farm_worker.py's exact step semantics
(`_stick` + "%.4f" wire format, snapshot rounding, observe2, compute_reward, resets incl. sudden-death
starts from `random.Random(seed + host)`, max_steps truncation). Level id 1 (menu "arena 2") by
default; its static geometry equals `arenas/arena2.json`.
- Build: `make -C sim env`. Python: `NativeHost(n_hosts, matches, frame_skip, sd_frac, max_steps)`,
  `reset_all()`, `step(actions[N,2,3])` -> farm_worker's `out` dict (zero-copy views).
- Trainer: `train_farm.py --native [--threads T]` (no game processes; `farm state` sockets
  `spr-native-<run>-<h>.sock` keep the dashboard working).
- Tests: `tools/test_env_equiv.py unit|python|oracle` (bit-identical vs the Python farm code and vs a
  real-game oracle on `spr-oracle-E.sock`). Throughput: `tools/bench_env.py`.
- Per-match arenas (`src/arena.*`): every allocation of a match's game code / Box2D goes into its own arena
  (routed by `EngineContext::arena` of the running `g_engine`), so `snapshot(k, slot)` / `restore(k, slot)` are
  memcpys (~5-10 us, exact). Planning branches (`plan_init/plan_begin/plan_step/plan_end`): env-owned clones of a
  match, kept in step by re-playing its action log (see `src/env/batch_env.h`). Tests: `../bridge/test_snapshot.py`.
