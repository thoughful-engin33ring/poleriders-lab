# random() and libm calls

All call sites of libc `random/srandom/time/localtime/mach_absolute_time` and libm inside the functions reachable from the match roots (see functions.md), with what the result feeds. Box2D is not included (Box2D 2.1.2 calls `sqrtf` x3, `cosf/sinf` x2 (b2Mat22::Set / b2Body transforms), `floorf` x1 - that is the Box2D port's concern; it never calls `random`).

## Summary

- **No `random()` result reaches gameplay.** Every `random()` in reachable code feeds a particle system (Dust/Sparks/Flare/Droplet/Splash), a sound choice, or the cosmetic snake/pencil flags. The libc RNG is additionally **re-seeded from `mach_absolute_time()` three+ times per construction** (PlayState ctor start, Camera ctor, each Player ctor) - so its state is wall-clock dependent anyway. The sim does not need a per-match RNG for gameplay; if you want byte-equal presentation-only fields you would also need the Dust/Flare internals, which we skip.
- The hidden modes (snake 1/10000, pencil ~1/900, Christmas, Chinese New Year) are **purely cosmetic**: their flags are read only by `Pole::Draw`, `Ball::Draw`, `PlayState::Draw` and the Player ctor's sound-file choice. Forcing them off changes nothing physical.
- libm in gameplay: `atan2f`, `fmodf`, `fminf`, `fmaxf` (identical natively), `cosf`/`sinf` (use `x86_cosf/x86_sinf`) in `Pole::Pole` and `PlayState::PlayerCollision`, and one **double `pow(x, 3.0)`** in `Ball::Update` - replaceable by `d*d*d` (proven bit-identical at float output for all inputs). No acosf/asinf/expf/logf/powf/tanf/double sin/cos anywhere in gameplay.
- Other double-precision spots worth copying exactly (not libm): `ucomisd` angle-window compares in HammerPlayerCollision/HammerBallCollision/Kick, the snake/pencil compares, `sqrt` of a double-computed length in setJoystickPosition (`cvtss2sd`, `mulsd`, `addsd`, `cvtsd2ss`, `sqrtss`), `_UpdatePad` stick thresholds (`cvtss2sd` vs 0.8/0.2).

## libc random / seeding / time call sites

| call addr | function | fn class | call | result feeds |
|---|---|---|---|---|
| 0x10004c9a5 | `atrophy::Time_GetHighResTimer()` | MIXED | `mach_absolute_time` | seed source |
| 0x10011d0e4 | `Background_Level3::Splash(b2Vec2, b2Vec2)` | PRES | `random` | particle params |
| 0x100124707 | `Camera::Camera(float, float, float)` | PRES | `srandom` | Camera ctor tail: srandom(Time_GetHighResTimer()) (reseed during PlayState ctor) |
| 0x10012587e | `Droplet::AddParticle(b2Vec2, float, b2Vec2)` | PRES | `random` | particle params |
| 0x100125fda | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126023 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x10012605c | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x1001261a6 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x1001261dc | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126215 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126248 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126281 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x1001262ba | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x1001262ed | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126312 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126337 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x10012636b | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x10012639d | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x1001263c4 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x1001263fa | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126426 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126452 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `random` | particle params |
| 0x100126ae6 | `Sparks::AddParticle(b2Vec2, float, int)` | PRES | `random` | particle params |
| 0x100126b1d | `Sparks::AddParticle(b2Vec2, float, int)` | PRES | `random` | particle params |
| 0x100126cf0 | `Flare::Flare(SpriteSheet*)` | PRES | `random` | particle init |
| 0x100126d28 | `Flare::Flare(SpriteSheet*)` | PRES | `random` | particle init |
| 0x100126d65 | `Flare::Flare(SpriteSheet*)` | PRES | `random` | particle init |
| 0x100126d9c | `Flare::Flare(SpriteSheet*)` | PRES | `random` | particle init |
| 0x100127332 | `Flare::AddParticle(b2Vec2, float, b2Vec2)` | PRES | `random` | particle params |
| 0x10012736e | `Flare::AddParticle(b2Vec2, float, b2Vec2)` | PRES | `random` | particle params |
| 0x1001273a3 | `Flare::AddParticle(b2Vec2, float, b2Vec2)` | PRES | `random` | particle params |
| 0x1001273e9 | `Flare::AddParticle(b2Vec2, float, b2Vec2)` | PRES | `random` | particle params |
| 0x100127470 | `Flare::Update(float)` | PRES | `random` | particle flicker |
| 0x100127478 | `Flare::Update(float)` | PRES | `random` | particle flicker |
| 0x100127480 | `Flare::Update(float)` | PRES | `random` | particle flicker |
| 0x100127535 | `Flare::Update(float)` | PRES | `random` | particle flicker |
| 0x1001275c1 | `Flare::Update(float)` | PRES | `random` | particle flicker |
| 0x10012c5aa | `Player::PlayTap(float)` | PRES* | `random` | random()%3 -> tap sound. Presentation. |
| 0x10012da4c | `Player::PlayDeathSound(float)` | PRES* | `random` | random()%2 -> death sound. Presentation. |
| 0x10012daeb | `Player::PlayKickSound(float)` | PRES* | `random` | random()%n -> kick sound. Presentation. |
| 0x10012db83 | `Player::PlayHitSound(float)` | PRES* | `random` | random()%n -> hit sound. Presentation. |
| 0x10012dc35 | `Player::PlayHammerSound(float, bool)` | PRES* | `random` | random()%3 -> hammer sound. Presentation. |
| 0x10013237c | `Player::LiftOff(bool)` | MIXED | `random` | random()%2 -> which jump sound (only if +0x1c0 > 0). Presentation. |
| 0x1001336c1 | `Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*)` | MIXED | `srandom` | Player ctor: srandom(Time_GetHighResTimer()) (reseed, once per player) |
| 0x1001341bf | `PlayState::PlayState(bool)` | MIXED | `srandom` | srandom(Time_GetHighResTimer()) at the very start of the ctor |
| 0x100134299 | `PlayState::PlayState(bool)` | MIXED | `time` | Christmas/CNY date check (cosmetic) |
| 0x1001342a6 | `PlayState::PlayState(bool)` | MIXED | `localtime` | Christmas/CNY date check (cosmetic) |
| 0x1001342c5 | `PlayState::PlayState(bool)` | MIXED | `random` | r = random()*2^-31 (float lerp 0..1) -> `Assets::snake = r < 0.0001` (double compare). Cosmetic (Pole/Ball Draw, sounds). |
| 0x100134309 | `PlayState::PlayState(bool)` | MIXED | `random` | r -> `Assets::pencil = !snake && r < 0.0011`. Cosmetic. |
| 0x100134e73 | `PlayState::PlayState(bool)` | MIXED | `time` | Boxing-Day (Dec 26, arena 4) achievement check (presentation) |
| 0x100134e80 | `PlayState::PlayState(bool)` | MIXED | `localtime` | same |
| 0x100135516 | `PlayState::_CalculateChineseNewYear()` | MIXED | `time` | Christmas/CNY date check (cosmetic) |
| 0x100135523 | `PlayState::_CalculateChineseNewYear()` | MIXED | `localtime` | Christmas/CNY date check (cosmetic) |
| 0x100138a0a | `PlayState::_ScoreGoal(int)` | MIXED | `random` | random()%2 -> which goal sound (Goal1A/1B or 2A/2B). Presentation. |
| 0x100138e8e | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | `random` | random()*2^-31 lerp -> Dust particle velocity. Presentation. |
| 0x100138f2e | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | `random` | same (second Dust burst). Presentation. |
| 0x1001390f8 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x100139141 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x100139194 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x1001391dd | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x100139230 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x100139279 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x1001392cc | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x100139315 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x100139368 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x1001393b1 | `PlayState::Splash(int)` | PRES | `random` | Background_Level3::Splash particle params (Level3 water only). Presentation. |
| 0x1001394e3 | `PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2)` | MIXED | `random` | random()%51 + 40 -> number of Sparks particles. Presentation. |
| 0x10013c945 | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |
| 0x10013c94d | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |
| 0x10013c9b7 | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |
| 0x10013c9ff | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |
| 0x10013caa4 | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |
| 0x10013caac | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |
| 0x10013caec | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |
| 0x10013cb34 | `PlayState::Update()` | MIXED | `random` | random() pairs -> Flare::AddParticle position/velocity (gated by OnGround && ShouldAddFlare). Presentation. |

Order inside `PlayState::PlayState` (for anyone who wants to replay the RNG): srandom(t0) -> random (snake) -> random (pencil) -> [Camera ctor: srandom(t1)] -> [Flare ctor: 4x random] -> per player [Player ctor: srandom(tN) ... Respawn -> LiftOff: random()%2 only if +0x1c0 > 0 (not at spawn)]. `PoleRiders::Update` (real game restart path, not the shadow) also calls random() for a random arena / heads.

## libm call sites

| call addr | function | fn class | call | native arm64 | result feeds |
|---|---|---|---|---|---|
| 0x100011724 | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `fmaxf` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x10001174b | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `fminf` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x1000117ab | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `fmaxf` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x1000117c0 | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `fminf` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x100011930 | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `atan2f` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x100011a5f | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `fmodf` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x100011a89 | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `fmodf` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x100011ab2 | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | `fmodf` | identical | **INPUT**: deadzone rescale (fmaxf/fminf), emulated d-pad (atan2f, fmodf) |
| 0x10004db35 | `atrophy::Input_SetMousePosition(atrophy::Vector2 const&)` | INFRA | `fmaxf` | identical | engine input polling (INFRA; not in the shadow path) |
| 0x10004db49 | `atrophy::Input_SetMousePosition(atrophy::Vector2 const&)` | INFRA | `fmaxf` | identical | engine input polling (INFRA; not in the shadow path) |
| 0x10004db62 | `atrophy::Input_SetMousePosition(atrophy::Vector2 const&)` | INFRA | `fminf` | identical | engine input polling (INFRA; not in the shadow path) |
| 0x10004db7b | `atrophy::Input_SetMousePosition(atrophy::Vector2 const&)` | INFRA | `fminf` | identical | engine input polling (INFRA; not in the shadow path) |
| 0x10004eb80 | `atrophy::Input_Update()` | INFRA | `fmodf` | identical | engine input polling (INFRA; not in the shadow path) |
| 0x10005e757 | `Common::Input::VirtualUpdateVibration(float, float, Common::Session::Player)` | PRES | `fmaxf` | identical | rumble (presentation) |
| 0x10005e7e0 | `Common::Input::VirtualUpdateVibration(float, float, Common::Session::Player)` | PRES | `fminf` | identical | rumble (presentation) |
| 0x100124145 | `Ball::Update(float)` | GAMEPLAY | `pow` | NOT identical in general - see note | **GAMEPLAY**: `(float)(pow((double)(x / (levelWidth/ratio)), 3.0) * -35.0)` = zipline centering force on the ball (ApplyForce inlined). Replace by `d*d*d` in double: verified bit-identical after the `*-35.0` + cvtsd2ss for all 2^32 float inputs on x86_64 (Rosetta, the game's libm) and arm64. |
| 0x10012450a | `Camera::Update(float)` | PRES | `fmaxf` | identical | camera (presentation) |
| 0x100124552 | `Camera::Update(float)` | PRES | `fminf` | identical | camera (presentation) |
| 0x100124619 | `Camera::AddFocus(float, float)` | PRES | `fmaxf` | identical | camera (presentation) |
| 0x100124635 | `Camera::AddFocus(float, float)` | PRES | `fminf` | identical | camera (presentation) |
| 0x100124649 | `Camera::AddFocus(float, float)` | PRES | `fminf` | identical | camera (presentation) |
| 0x10012465d | `Camera::AddFocus(float, float)` | PRES | `fmaxf` | identical | camera (presentation) |
| 0x100124679 | `Camera::AddFocus(float, float)` | PRES | `fminf` | identical | camera (presentation) |
| 0x100124689 | `Camera::AddFocus(float, float)` | PRES | `fmaxf` | identical | camera (presentation) |
| 0x100125dc0 | `Dust::Update(float)` | PRES | `fmaxf` | identical | presentation |
| 0x100125ec3 | `Dust::Fade()` | PRES | `fmaxf` | identical | presentation |
| 0x10012610b | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `fmaxf` | identical | presentation |
| 0x100126118 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `fminf` | identical | presentation |
| 0x100126130 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | `pow` | NOT identical in general - see note | presentation |
| 0x100126b2a | `Sparks::AddParticle(b2Vec2, float, int)` | PRES | `cosf` | use x86_cosf | presentation |
| 0x100126b39 | `Sparks::AddParticle(b2Vec2, float, int)` | PRES | `sinf` | use x86_sinf | presentation |
| 0x100126b4d | `Sparks::AddParticle(b2Vec2, float, int)` | PRES | `sinf` | use x86_sinf | presentation |
| 0x100126b5e | `Sparks::AddParticle(b2Vec2, float, int)` | PRES | `pow` | NOT identical in general - see note | presentation |
| 0x100127671 | `Flare::Update(float)` | PRES | `fmaxf` | identical | presentation |
| 0x10012c473 | `Player::Taunting()` | PRES | `atan2f` | identical | taunt gesture detection (presentation) |
| 0x10012da05 | `Player::RumbleBump(float, float)` | PRES* | `fminf` | identical | rumble clamp (presentation) |
| 0x10012daa6 | `Player::PlayDeathSound(float)` | PRES* | `fminf` | identical | rumble clamp (presentation) |
| 0x10012db45 | `Player::PlayKickSound(float)` | PRES* | `fminf` | identical | rumble clamp (presentation) |
| 0x10012dbdd | `Player::PlayHitSound(float)` | PRES* | `fminf` | identical | rumble clamp (presentation) |
| 0x10012dcc5 | `Player::PlayHammerSound(float, bool)` | PRES* | `fminf` | identical | rumble clamp (presentation) |
| 0x10012dd6d | `Player::PlayLandSound(float, bool)` | PRES* | `fminf` | identical | rumble clamp (presentation) |
| 0x100130578 | `Player::_OffPole()` | GAMEPLAY | `fminf` | identical | rumble (presentation) |
| 0x100130956 | `Player::Kick(b2Vec2)` | GAMEPLAY | `atan2f` | identical | **GAMEPLAY**: kick direction -> kick animation (8 sectors via double compares vs 1.1781 etc.) and foot-box position |
| 0x100130d38 | `Player::Update(float)` | MIXED | `fmaxf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x100130d62 | `Player::Update(float)` | MIXED | `fmaxf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x100130e74 | `Player::Update(float)` | MIXED | `fminf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x1001310a8 | `Player::Update(float)` | MIXED | `fminf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x100131369 | `Player::Update(float)` | MIXED | `fmaxf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x100131379 | `Player::Update(float)` | MIXED | `fminf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x10013147f | `Player::Update(float)` | MIXED | `fmaxf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x1001315d5 | `Player::Update(float)` | MIXED | `fmaxf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x1001315e5 | `Player::Update(float)` | MIXED | `fminf` | identical | rumble/sound volumes (presentation) - check each when porting |
| 0x100131711 | `Player::Update(float)` | MIXED | `atan2f` | identical | **GAMEPLAY**: left/right stick angle -> kick-hold direction sector (+-2.74889, +-1.5708, +-0.392699) -> kickAnim*_hold + _FootBoxTo |
| 0x100131ad6 | `Player::Update(float)` | MIXED | `atan2f` | identical | **GAMEPLAY**: right-stick setpoint angle for the pole servo |
| 0x100131b69 | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131b9d | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131bf2 | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131c3b | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131c64 | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131c9c | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131cc3 | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131cec | `Player::Update(float)` | MIXED | `fmodf` | identical | **GAMEPLAY**: pole-servo angle wrapping (x-fmodf(x,2pi) normalisation of segment angles and setpoint difference) |
| 0x100131dd0 | `Player::Update(float)` | MIXED | `fmaxf` | identical | **GAMEPLAY**: clamp servo error to [-0.4, 0.4] |
| 0x100131de4 | `Player::Update(float)` | MIXED | `fminf` | identical | **GAMEPLAY**: clamp servo error to [-0.4, 0.4] |
| 0x100131e05 | `Player::Update(float)` | MIXED | `fmaxf` | identical | **GAMEPLAY**: clamp servo error to [-0.4, 0.4] |
| 0x100131e19 | `Player::Update(float)` | MIXED | `fminf` | identical | **GAMEPLAY**: clamp servo error to [-0.4, 0.4] |
| 0x100138dcf | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | `fminf` | identical | dust count / sound volumes (presentation) |
| 0x100138fe1 | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | `fminf` | identical | dust count / sound volumes (presentation) |
| 0x100139009 | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | `fminf` | identical | dust count / sound volumes (presentation) |
| 0x100139055 | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | `fminf` | identical | dust count / sound volumes (presentation) |
| 0x100139096 | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | `fminf` | identical | dust count / sound volumes (presentation) |
| 0x100139515 | `PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2)` | MIXED | `atan2f` | identical | spark direction (presentation) |
| 0x1001396df | `PlayState::PlayerCollision(b2Contact*, int, int)` | MIXED | `atan2f` | identical | contact-normal angle `a = atan2f(n)`; -> `fabsf(cosf(a)) > 0.5` below |
| 0x10013979f | `PlayState::PlayerCollision(b2Contact*, int, int)` | MIXED | `cosf` | use x86_cosf | **GAMEPLAY**: `fabsf(cosf(a)) > 0.5` selects the vertical (stomp: upper rider kills lower, -32px offset) vs sideways (needs both airborne) branch of PlayerCollision. Use `x86_cosf`. |
| 0x100139e9b | `PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int)` | MIXED | `atan2f` | identical | **GAMEPLAY**: angle window (`< -2`, `(-1.15, 1.15)` as double, `> 2`) decides whether the hammer hit kills |
| 0x10013a26c | `PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int)` | MIXED | `atan2f` | identical | **GAMEPLAY**: angle window (+-0.95 / +-2.2, double compares) decides hammer->ball hit |
| 0x10013ae4d | `PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int)` | MIXED | `fminf` | identical | ball-hit sound volume (presentation) |
| 0x10013ae96 | `PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int)` | MIXED | `fminf` | identical | tap sound volume (presentation) |
| 0x10013e969 | `Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool)` | GAMEPLAY | `cosf` | use x86_cosf | **GAMEPLAY**: pole segment i x position `x0 + t` with `t = (cosf(a) * (float)i) * k; t = t + t` (Pole ctor, k = segment half-length). Use `x86_cosf`. |
| 0x10013e9a3 | `Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool)` | GAMEPLAY | `sinf` | use x86_sinf | **GAMEPLAY**: pole segment y positions (Pole ctor). Use `x86_sinf`. |
| 0x100141096 | `ShockWave::Trigger(b2Vec2, b2Vec2)` | PRES | `atan2f` | identical | presentation |

## Notes for porters

- `x86_cosf`/`x86_sinf` are needed at exactly three call sites: 0x10013e969 / 0x10013e9a3 (Pole ctor) and 0x10013979f (PlayerCollision).
- `Ball::Update` @0x100124145: game computes `double d = (float)(ball.x / (Ball+0x40 / ratio)); double p = pow(d, 3.0); force.x += (float)(p * -35.0)`. The exhaustive test (`/private/tmp/claude-501/-Users-tron-Projekty/9d4f10d9-e553-4286-b237-b9c53c84cb1c/scratchpad/re-inventory/powtest2.c`, all 4,278,190,080 finite floats) found 0 differences between `(float)(pow(d,3)*-35)` and `(float)((d*d*d)*-35)` on both x86_64 libm (Rosetta) and arm64 libm; raw `pow` vs `d*d*d` differ in ~10M doubles (1 ulp) but never after the float conversion. So write `d*d*d` (keep the double type and the multiply order `(d*d)*d`).
- `fminf/fmaxf` follow IEEE (NaN-ignoring) semantics on both platforms; keep argument order as in the binary anyway.
- `atan2f` results are compared against float and double literals; keep the exact literal and its precision (e.g. Kick compares the atan2f result, converted to double, with double literals such as 1.1781 = 3*pi/8) - use `re/dz.py` to get them.
