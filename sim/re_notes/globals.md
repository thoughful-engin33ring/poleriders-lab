# Globals, statics and constants read/written by gameplay code

Addresses are unslid VM addresses of `Sportsfriends.orig` (file offset = vmaddr - 0x100000000). Values were read from
the binary's bytes (`__TEXT,__const` for `S` constants, `__DATA,__data` for initialized mutables; `__bss`/`__common`
entries are 0 at load). "Folded" = the compiler emitted the value as an immediate/literal, so the code does not read the
symbol (the value is still the same). Readers/writers are restricted to the match code in `functions.md` unless noted.

## 1. Mutable globals

| address | symbol | type | value in binary | read by | written by (runtime) | sim treatment |
|---|---|---|---|---|---|---|
| 0x10062fd80 | `PoleRiders::Assets::currentLevel` | int | 3 (`__data`) | PlayState ctor (arena switch), Player::Update (`==2`: sets flare flag +0x191 only; `==4`: extra Dust), BeginContactInner | `PoleRiders::Begin()` = `Session._sessionData+0x90` + 1; `PoleRiders::Update()` (random arena on restart when the session asks for it, `random()%4/5`); BeginContactInner sets 5 on the Rainbow-Highway unlock (never with level5unlocked=1) | constant per match. **Flatirons = 2** (-> `Level2`, `PlayState+0x214 = 2`; the HANDOFF note "Flatirons reports 1" contradicts the ctor code - `+0x214` is set to 1 only for arena 1). Arenas: 1 Kalgoorlie, 2 Flatirons, 3 Dakhla, 4 Princeton, 5 Rainbow Highway. |
| 0x10062fd88 | `PoleRiders::Assets::maxScore` | int | 4 | PlayState::Update (decided-match check), `_ScoreGoal` | `PoleRiders::Begin()`: `(int)save[0x74b34]` (cvttss2si, truncation) | training config: 3 |
| 0x10062fd8c | `PoleRiders::Assets::roundTime` | float (s) | 180.0 | PlayState ctor (`+0x180 = roundTime`), GroundContact (sound gate, presentation) | `PoleRiders::Begin()`: `save[0x74b38] * 60.0f` (mulss) | training config: 3 min -> 180.0 |
| 0x10062fd84 | `PoleRiders::Assets::showIcons` | bool | 1 | (Draw only) | PlayState ctor: `= save[0x74b3d]` | ignore |
| 0x10062fd90 | `PoleRiders::Assets::shouldRumble` | bool | 1 | Player::Update (rumble only) | `PoleRiders::Begin()`: `= save[0x74b4a]` | ignore |
| 0x100d0f778 | `PoleRiders::Assets::debug` | bool | 0 (bss) | PlayState ctor (DebugRender at +0x198) | - | 0 |
| 0x100d0f77c | `PoleRiders::Assets::ballSoundIndex` | int | 0 | Player::PlayBallSound | Player::PlayBallSound | presentation |
| 0x100d0f780 | `PoleRiders::Assets::level5unlocked` | bool | 0 | BeginContactInner (unlock path guard), PoleRiders::Update | PlayState ctor: `= save[0x74b3c]`; BeginContactInner (unlock: =1, `save[0x74b3c]=1`, `Common::Save::Save()`, `currentLevel=5`, `PlayState+0x35=1`, `+0x200=0`); **bridge forces 1 before every tick** (`g_no_unlock`) | constant 1 -> the unlock branch never runs |
| 0x100d0f784 | `PoleRiders::Assets::winStreak` | int | 0 | VictoryScene only | - | ignore |
| 0x100d0f788 | `PoleRiders::Assets::chineseNewYear` | bool | 0 | Draw/sounds only | PlayState ctor `= _CalculateChineseNewYear()` (time/localtime); bridge `SPRBRIDGE_CANON` patches the call to `xor al,al` | 0. **Cosmetic only** (no gameplay reader). |
| 0x100d0f789 | `PoleRiders::Assets::christmas` | bool | 0 | Ball::Draw | PlayState ctor: `localtime().tm_mon==11 && tm_mday==25` (bridge patches `sete` -> `xor`) | 0, cosmetic |
| 0x100d0f78a | `PoleRiders::Assets::pencil` | bool | 0 | Pole::Draw | PlayState ctor: `!snake && r2 < 0.0011` with `r = random()*4.656613e-10f` (lerp 0..1), double compare (bridge patches `mov al,1` -> `mov al,0`) | 0, cosmetic. The `random()` call still happens (RNG advance). |
| 0x100d0f78b | `PoleRiders::Assets::snake` | bool | 0 | Pole::Draw, Player ctor (picks sound files) | PlayState ctor: `r1 < 0.0001` (bridge patches `seta` -> `movb $0`) | 0, cosmetic; `random()` still called |
| 0x100c57ee0 | `(anonymous)::_playState` | PlayState* | null | `PoleRiders::Update/Draw/End/LoadComplete` only | `PoleRiders::Begin()`, `PoleRiders::Update()` (restart) | not used by any PlayState/Player code - the shadow never touches it. Bridge reads it for the real match. |
| 0x100c57f54 | `(anonymous)::sElapsed` (PoleRiders) | float | 0 | PoleRiders::Update fixed-step accumulator | PoleRiders::Update / bridge | not in the shadow path |
| 0x100c57f51 | `(anonymous)::_resetting` (PoleRiders) | bool | 0 | PoleRiders::Update | bridge (reset) | not in the shadow path |
| 0x100c5c300 | `Common::Save::_saveData` | struct | bss (loaded from config) | fields below | menus; `Save()` | config input |
|  | `+0x74b34` score limit | float |  | PoleRiders::Begin -> maxScore |  | 3.0 |
|  | `+0x74b38` match duration (minutes) | float |  | PoleRiders::Begin -> roundTime |  | 3.0 |
|  | `+0x74b3c` level5 unlocked | bool |  | PlayState ctor -> Assets::level5unlocked | BeginContactInner (unlock) | 1 |
|  | `+0x74b3d` show icons | bool |  | PlayState ctor | | ignore |
|  | `+0x74b4a` rumble | bool |  | PoleRiders::Begin | | ignore |
| 0x100cd0e60 | `Common::Session::_sessionData` | struct | bss (menus) | PlayState ctor: player slot `i` record at `+0x1c*i` (7 ints; field 1 = side/active: 0 = empty slot, -1 / other = team side, drives spawn side and `+0x210` count); Player ctor copies the whole record to `Player+0x220..+0x238` (field 0 = control mode (0 full pad, 1/2 = shared-pad halves) and field 2 = pad index are what `Common::Input::Virtual*` use; field 2 -> `+0xa8` (also `ButtonHeld` pad index), field 3 -> `+0x22c` head number); `Common::Input::GetControllerNum`; `+0x90` arena index (-> currentLevel in Begin) | menus | the sim must be given the same 2 records the real game has (read them once from a live game via the bridge) |
| 0x100c5aca0 | `Common::Input::_pads[8]` | struct, stride 0x2c8 | bss | ButtonHeld (`byte[button]`), LeftStick (`+0xfc,+0x100`), RightStick (`+0x10c..+0x118`), VirtualLeftStick (`+0x0..+0xb` d-pad/buttons, `+0xfc/+0x100`), VirtualLeftTrigger, VirtualPRTaunt | `_UpdatePad` (every frame; also first clears 0x51 edge bytes), `VirtualUpdateVibration` (rumble `+0xf4/+0xf8`) | per-match copy (the bridge already keeps a private copy per shadow) |
| 0x10064c980 | `atrophy::sGamepadList[4]` | GamepadSDL, stride 0x1b8 | bss | `Input_GetGamepadAt/IsGamepadConnectedAt`, `_UpdatePad` (buttons `+0xe2..+0x162` step 8, sticks `+0xc..+0x18`, `+0x3c..+0x48`, triggers `+0x6c/+0x70/+0x9c`), setJoystickPosition (deadzone `+0x1c+0x30*stick`, response table `+0xd0/+0xd8`, emulated-button flag `+0x38+0x30*stick`) | setJoystickPosition, setButtonState, GamepadSDL::* | per-match copy |
| 0x10064d340 | `(anonymous)::gameData` (Common::Move) | struct | bss | Common::Move::Active/GetButtonHeld/GetAnyButtonHeld/GetVirtualDPad | PS Move daemon (stubbed by bridge `SPRBRIDGE_NOMOVE`) | treat as inactive |
| (libc) | `random()` state | BSD random | seeded by `srandom(Time_GetHighResTimer())` in PlayState ctor, Camera ctor and every Player ctor | see random_and_libm.md | | no gameplay effect |

## 2. `PoleRiders::Assets` constants (`__TEXT,__const`)

Body/fixture userData tags: code stores the *address* of these ints as `b2BodyDef.userData` / `b2FixtureDef.userData`;
the contact listener dereferences them (`ContactToInts`) to get the tag. The sim needs unique pointers per tag (or
an equivalent tag field) - the oracle masks pointers anyway.

| address | symbol | value | used by (gameplay) |
|---|---|---|---|
| 0x10028e06c..78 | `BODY_PLAYER1..4` | 0,1,2,3 | SpawnPlayerBody (userData of rider body) |
| 0x10028e07c | `BODY_GROUND` | 4 | Level*::CreateBody |
| 0x10028e080 | `BODY_WALL` | 5 | Level*::CreateBody (goal walls) |
| 0x10028e084 | `BODY_POLE` | 6 | Pole ctor, Player::SetGhost |
| 0x10028e088 | `BODY_BALL` | 7 | Ball::CreateBody (ball AND trolley) |
| 0x10028e08c | `BODY_ANCHOR` | 8 | Ball::CreateBody |
| 0x10028e090 | `BODY_NONE` | 9 | Level*::CreateBody |
| 0x10028e094 | `BODY_DELETER` | 10 | Level*::CreateBody, _MakeDeleters (kill zones) |
| 0x10028e098 | `BODY_HAMMER` | 11 | Pole ctor (hammer head) |
| 0x10028e09c | `BODY_SEA` | 12 | Level3::CreateBody |
| 0x10028e0a0 | `FIXTURE_SENSOR` | 13 | SpawnPlayerBody (foot sensor) |
| 0x10028e0a4 | `FIXTURE_NONE` | 14 | (folded) |
| 0x10028e0a8 | `FIXTURE_HAMMERBODY1` | 15 | (folded) |
| 0x10028e0ac | `FIXTURE_HAMMERBODY2` | 16 | (folded) |
| 0x10028e0b0 | `FIXTURE_IMPALE` | 17 | SpawnPlayerBody |
| 0x10028e0e8 | `HANDLE_START` | 2 (int) | Pole ctor, Player::SetGhost (index of first hand-held segment) |
| 0x10028e0b4 | `ratio` | 64.0 (px per m) | everywhere (`leaq` + `movss`, divisions `x / ratio`) |
| 0x10028e0b8 | `deathDelay` | 3.0 | BeginContactInner, PlayerCollision (`PlayState+0x1a0[i]` respawn timers) |
| 0x10028e0bc | `linearDamping` | 0.33 | SpawnPlayerBody |
| 0x10028e0c0 | `ballDamping` | 1.1 | Ball::CreateBody (ball; trolley uses `ballDamping*3`) |
| 0x10028e0c4 | `poleAngularDamping` | 0.4 | Pole ctor |
| 0x10028e0c8 | `poleLinearDamping` | 0.2 | Pole ctor |
| 0x10028e0cc | `polePushForce` | 10.0 | (folded; not referenced by symbol) |
| 0x10028e0d0 | `poleSwingTorque` | 440.0 | Player::Update (max motor torque = `|err| * 440 * 4`), Respawn |
| 0x10028e0d4 | `spinSpeedFactor` | 12.0 | Player::Update (motor speed = `err * 12 * 4`) |
| 0x10028e0d8 | `kickPlayerStrength` | 700.0 | PlayerCollision, HammerPlayerCollision (`/ ratio`) |
| 0x10028e0dc | `kickBallStrength` | 1400.0 | BallPlayerCollision, HammerBallCollision (`/ ratio`) |
| 0x10028e0e0 | `standardPoleLength` | 780.4 (0x4443199a) | PlayState ctor/Update (Respawn pole length; head #7 gets `standard - 147`) |
| 0x10028e0e4 | `rainbowPoleLength` | 740.4 (0x4439199a) | same, arena 5 |

## 3. Class statics (`__TEXT,__const`; values are folded into the code as literals)

| address | symbol | value | notes |
|---|---|---|---|
| 0x10028e478 | `Ball::_friction` | 0.1 | ball fixture |
| 0x10028e47c | `Ball::_restitution` | 0.9 | |
| 0x10028e480 | `Ball::_density` | 4.2 | |
| 0x10028e774 | `Player::_deadZone` | 0.2 | right-stick magnitude threshold for the pole servo (`> 0.2` literal in Player::Update) |
| 0x10028e778 | `Player::_friction` | 0.8 | |
| 0x10028e77c | `Player::_restitution` | 0.0 | |
| 0x10028e780 | `Player::_density` | 1.1 | |
| 0x10028e784 | `Player::_hitBoxHeight` | 100.0 px | |
| 0x10028e788 | `Player::_hitBoxWidth` | 48.0 px | |
| 0x100290074 | `Pole::_poleWidth` | 16.0 px | |
| 0x100290078 | `Pole::_restitution` | 0.0 | |
| 0x10029007c | `Pole::_friction` | 0.4 | |
| 0x100290080 | `Pole::_density` | 0.2 | |

Level statics (`Level<N>::_x`, px unless noted). The PlayState ctor copies `_levelWidth/_ballHeight/_targetWidth/_spawnHeight`
of the chosen arena to `PlayState+0x8/+0x10/+0xc/+0x14` and **always `Level1::_spawnBoost` (50.0) to `+0x18`** - the
per-level `_spawnBoost` values are never read (copy/paste bug in the original; reproduce it). `_friction/_restitution/_density/_angle`
are folded into `Level<N>::CreateBody`.

| level | addr (`_levelWidth`) | levelWidth | ballHeight | targetWidth | spawnHeight | spawnBoost (unused except L1) | friction | restitution | density | angle | player spawn x base (ctor jump table) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 Kalgoorlie | 0x10028e598 | 2800 | -394 | 2606 | -612 | 50 | 0.5 | 0.3 | 0.7 | 0 | 400 |
| **2 Flatirons** | 0x10028e60c | **3520** | **-394** | **2600** | **-782** | 80 (unused) | 0.5 | 0.3 | 0.7 | 0 | **800** |
| 3 Dakhla | 0x10028e65c | 2930 | -364 | 2294 | -652 | 55 (unused) | 0.5 | 0.3 | 0.7 | 0 | 800 |
| 4 Princeton | 0x10028e698 | 4652 | -421 | 3600 | -512 | 150 (unused) | 0.5 | 0.3 | 0.7 | 0 | 1000 |
| 5 Rainbow Hwy | 0x10028e6ec | 2926 | -394 | 2600 | -652 | 70 (unused) | 0.6 | 0.3 | 0.7 | 0 | 400 |

Spawn in the ctor: `x = side * (base + ((i << 7) & 0x100))` (i.e. +256 for player slots 2/3), `y = -5`, pole length as above;
`Respawn(x, -5, false, poleLen, false, false)` (head #7: `Respawn(x, -5, false, standard-147, false, true)`). Other ctor constants:
world gravity (0, 20) m/s^2 (+y down), `PlayState+0x18c` dt = 0x3c888889 (1/60f), `+0x200` = -1, `+0x204` = 1, `+0x170` = 15,
`+0x174` = 1, `+0x1e8..+0x1f4` = -1, `+0x220/+0x221` = 1; Camera/ Target positions derive from `levelWidth/targetWidth/ballHeight`.

## 4. Animation tables (`(anonymous)::*`, `__TEXT,__const`) - gameplay-relevant

`AnimationData` layout: `int frames[32]` @0, `int count` @0x80, `float fps` @0x84, `bool loop` @0x88 (stride 0xa0).
`Animation` object: `const AnimationData* data` @0, `float pos` @8 (0..1), `float rate` @0xc, `bool playing` @0x10.
`Animation::Update(dt)`: if `pos < 1`: `pos += dt*rate*fps / (float)count`; then `while (pos >= 1 && loop) pos += -1`.
`IsFinished` = `pos >= 1`. Player::Update compares `Player+0x68 anim->data` against the `kickAnim*_hold` addresses and uses
`GetPosition/SetPosition` when switching kick directions, so these tables (count/fps/loop) **are gameplay** for the rider's
main animation (+0x68). Pointer identity only needs to be consistent inside the sim.

| address | symbol | count | fps | loop | used by |
|---|---|---|---|---|---|
| 0x10028dfe0 | nullAnimation | 1 | 1 | 0 | Animation ctor |
| 0x10028ed40 / 0x10028ede0 | kickAnimNE_hold / kickAnimNE | 6 / 8 | 30 | 0 | Kick, Update |
| 0x10028ee80 / 0x10028ef20 | kickAnimE_hold / kickAnimE | 6 / 8 | 30 | 0 | Kick, Update |
| 0x10028efc0 / 0x10028f060 | kickAnimSE_hold / kickAnimSE | 6 / 8 | 30 | 0 | Kick, Update |
| 0x10028f100 | kickAnimS | 5 | 30 | 0 | Kick, Update |
| 0x10028f1a0 / 0x10028f240 | kickAnimSW_hold / kickAnimSW | 6 / 8 | 30 | 0 | Kick, Update |
| 0x10028f2e0 / 0x10028f380 | kickAnimW_hold / kickAnimW | 6 / 8 | 30 | 0 | Kick, Update |
| 0x10028f420 / 0x10028f4c0 | kickAnimNW_hold / kickAnimNW | 6 / 8 | 30 | 0 | Kick, Update |
| 0x10028f560 | landingAnim | 5 | 15 | 0 | Land, Update |
| 0x10028f600 | turnAnim | 3 | 15 | 0 | Update |
| 0x10028f6a0 | walkAnim | 8 | 30 | 1 | SpawnPlayerBody, Update |
| 0x10028f740 | idleAnim | 1 | 0 | 0 | Update |
| 0x10028f7e0 | poleIdleAnim | 1 | 0 | 0 | Update |
| 0x10028f880 | fellAnim | 3 | 10 | 0 | Update |
| 0x10028f920 | fallingAnim | 4 | 15 | 0 | Update |
| 0x10028fa60 | liftoffAnim | 2 | 15 | 0 | LiftOff |
| 0x10028f9c0 | smashAnimData | 8 | 30 | 0 | ShowSmash (+0x70, presentation) |
| 0x10028fd60/0x10028fe00/0x10028fea0/0x10028ff40/0x10028ffe0 | scoreArrow2/1, scoreArrow2idle/1idle, gullIdle | 24/24/1/1/1 | 15 | 0 | PlayState +0x38/+0x40/+0x48 (presentation) |
| 0x100290220 / 0x1002902c0 | animHit / animWait | 23 / 1 | 15 | 0 | Target (presentation) |

## 5. Input constants

| address | symbol | value | used by |
|---|---|---|---|
| 0x100286600 | `atrophy::kEmulatedButtonAngles` | {pi/2, -pi/2, pi, 0} | setJoystickPosition (emulated d-pad sectors) |
| 0x100286a20 | `_kButtonMap.17882` | literal pool area | _UpdatePad / VirtualLeftStick literals (0.4 / 0.8 / 0.2 stick-flick thresholds, 7.0 hold-rate) |
| - | stick deadzone | per gamepad `+0x1c+0x30*stick` (runtime, from GamepadSDL::connect) | setJoystickPosition |

## 6. Box2D-related globals touched by game code

| address | symbol | notes |
|---|---|---|
| 0x10063fd00 | `vtable for b2PolygonShape` | stack shapes in CreateBody functions |
| 0x10063fc90 | `vtable for b2CircleShape` | Ball, Player, Pole |
| 0x10062fe00 | `vtable for b2FixtureDef` (via GOT 0x100318230) | every stack b2FixtureDef (patched Box2D: b2FixtureDef has a vtable) |
| 0x10062fec0 | `vtable for b2ContactListener` (via GOT 0x100318238) | PlayState base (ctor unwind path) |

## 7. How config reaches the match (and what the bridge overrides)

1. Menu/config -> `Common::Save::_saveData` (+0x74b34 score limit 3.0, +0x74b38 duration 3.0 min) and `Common::Session::_sessionData`.
2. `PoleRiders::Begin()` @0x10013f880 (real game only): `currentLevel = session+0x90 + 1`, `roundTime = duration*60`,
   `maxScore = (int)scoreLimit`, `shouldRumble = save+0x74b4a`, then `new PlayState(false)`.
3. The shadow calls `PlayState::PlayState(0)` directly, so it inherits whatever `Assets::*` the real game set in Begin
   (maxScore 3, roundTime 180, currentLevel 2 for Flatirons).
4. PlayState ctor: `+0x180 = roundTime`; `level5unlocked = save+0x74b3c`; snake/pencil/Christmas/CNY computed (bridge
   `SPRBRIDGE_CANON=1` patches the 4 sites so all are 0 - cosmetic anyway); bridge sets `level5unlocked = 1` before each tick.
5. Per tick the bridge writes pad state (`sGamepadList[i]` axes + `setJoystickPosition(slot, 0, lx, -ly)` /
   `(slot, 1, rx, -ry)`, then `_UpdatePad(i, 1/60 + 0.0002)`) and calls `PlayState::Update()`.
