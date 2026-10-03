# Function inventory: one Super Pole Riders match (non-Box2D code)

Generated from `re/dis.txt` + symbols of `Sportsfriends.orig` (x86_64, unslid VM addresses; file offset = vmaddr - 0x100000000).
Scripts: `/private/tmp/claude-501/-Users-tron-Projekty/9d4f10d9-e553-4286-b237-b9c53c84cb1c/scratchpad/re-inventory/` (temporary) (parse.py, graph.py, bfs.py, classes.py, gen.py) - regenerate with `python3 bfs.py && python3 gen.py`.

## Scope and method

- Roots: `PlayState::PlayState(bool)` C1 @0x10013d64c (arg 0), `PlayState::LoadComplete()` @0x10013d574, `PlayState::Update()` @0x10013b8c8, deleting dtor @0x10013b098 (+ D1 @0x10013b4b0), the four `b2ContactListener` slots of PlayState, and the input chain `atrophy::Gamepad::setJoystickPosition` @0x1000116ce + `Common::Input::_UpdatePad` @0x10005f660.
- Edges: direct `callq`/tail `jmp` to other functions, RIP-relative `leaq` of function addresses (none found in game code), virtual calls resolved by hand (below). Box2D (`b2*`) callees are listed but not descended into.
- Classes: **GAMEPLAY** (port), **MIXED** (mostly presentation, but writes something gameplay reads - port that part; see note), **PRES*** (presentation that writes fields of a game object which only presentation reads - skip, but mask those bytes in the oracle diff or port trivially), **PRES** (skip), **INFRA** (engine/resource/SDL/FMOD/file I/O - skip).
- Packages: **A** construction+teardown, **B** Player per-tick & state, **C** PlayState::Update + contacts + scoring, **D** input chain.

## Virtual dispatch

- Only two PoleRiders vtables exist (`PlayState` @0x10063f4c0, `DebugRender`). **Player, Pole, Ball, Level*, Stats, Animation are non-virtual** (no vtable at +0).
- `vtable for PoleRiders::PlayState` (single base `b2ContactListener`), slots after offset-to-top/typeinfo:
  0: `~PlayState()` D1 @0x10013b4b0 | 1: `~PlayState()` D0 @0x10013b098 | 2: `BeginContact(b2Contact*)` @0x10013a93c | 3: `EndContact(b2Contact*)` @0x100136348 | 4: `PreSolve(b2Contact*, b2Manifold const*)` @0x10013544c | 5: `PostSolve(b2Contact*, b2ContactImpulse const*)` @0x10013af72.
- PlayState is registered with `b2World::SetContactListener(this)` in the ctor. **No** `SetContactFilter` / `SetDestructionListener` call is reachable: the world uses Box2D's default `b2ContactFilter` (group/category/mask) and no destruction listener.
- Indirect calls in reachable game code: `~PlayState` D2 `callq *0x8(%rax)` on `+0x198` (DebugRender deleting dtor; null unless `Assets::debug`); Background*/SpriteSheet dtors release atrophy resources (presentation). Everything else flagged by the scanner is jump-table data decoded as code.
- Note: in this build `b2FixtureDef` has a vtable (`__ZTV12b2FixtureDef` @0x10062fe00, loaded via GOT) - game code stores it into every stack b2FixtureDef it builds (Box2D-side patch; see BOX2D_DIFF.md).

## Inlined helpers (out-of-line copies are NOT called; port the inlined code in the caller)

| helper | out-of-line copy | inlined into | what |
|---|---|---|---|
| `PlayState::_SetupWorld()` | 0x10013b032 (30 insns) | PlayState ctor | new b2World(gravity (0,20), doSleep=true); SetContactListener(this); world -> +0x20 |
| `PlayState::_PunchTime()` | 0x1001353b2 (12 insns) | _ShockwaveAt tail | +0x21c += 2; if (+0x21c < 5) +0x218 = 7 |
| `PlayState::_GoSuddenDeath()` | 0x100135896 (22 insns) | PlayState::Update | if !+0x175: scores=0, +0x175=1, +0x180=0, +0x1ec=0, sound, +0x110=1 |
| `PlayState::EnableTimer()` | 0x10013538c (7 insns) | BeginContactInner | +0x188 = 1 |
| `PlayState::ContactToInts(...)` | 0x1001353e8 (36 insns) | Begin/End/PostSolve | userData -> int tags (0 if null) |
| `PlayState::PreSolveInner(...)` | 0x100135452 (4 insns) | PreSolve | empty |
| `PlayState::BallContact(...)` | 0x100136248 (24 insns) | PostSolveInner | sound only (fminf volume) |
| `PlayState::PoleGroundCollision(...)` | 0x10013acce (27 insns) | PostSolveInner | sound only (PlayTap) |
| `Player::_UpdateSticks()` | 0x100130bee (52 insns) | Player::Update | if +0x1d0: VirtualLeftStick -> +0x1ac..+0x1b8, VirtualRightStick -> +0x19c..+0x1a8; else zero them |
| `Player::_ParalyzeBody()` | 0x10012c374 (9 insns) | Player::Update | +0x22=0, +0x208=0, +0x190=0, run sound volume 0 |
| `Player::_UpdateVibration(float,float)` | 0x100130bae (15 insns) | Player::Update | rumble (presentation) |
| `Player::_Slerp(...)` | 0x10012c62a (93 insns) | (Draw) | presentation |
| `Ball::Frozen()` | 0x100123546 (5 insns) | - | accessor +0x44 |

## Porting work packages (GAMEPLAY + MIXED only)

| pkg | scope | functions | instructions (all / excluding Level1/3/4/5) |
|---|---|---|---|
| A | Construction + teardown (PlayState ctor, Level2 terrain, Player/Pole/Ball construction, Respawn-with-new-pole, dtors) | 32 | 6595 / 4268 |
| B | Player per-tick & state (Player::Update, pole servo, kick/land/lift-off, ghosting, filters, Animation, Pole helpers, Stats counters) | 51 | 2587 / 2587 |
| C | Match logic (PlayState::Update, Ball::Update, contact listener + all collision handlers, scoring, sudden death, punch time) | 27 | 3921 / 3921 |
| D | Input chain (setJoystickPosition -> _UpdatePad -> Virtual*Stick / ButtonHeld) | 14 | 1750 / 1750 |

Instruction counts are whole-function counts (MIXED functions include their presentation parts, which are skipped when porting). The Level1/3/4/5 terrain builders (~2.3k insns) are only needed for other arenas. Coupling: A<->B share Player/Pole construction (SpawnPlayerBody, Pole ctor are in A but called from B's Respawn paths); C calls into B only through small accessors plus Player::Update/Kick/Land/LiftOff/SetGhost/Teleport/Respawn/Boost; D is standalone (feeds Player+0x19c..+0x1b8 through Virtual*Stick and PlayState via ButtonHeld).

## Package A: Construction + teardown (PlayState ctor, Level2 terrain, Player/Pole/Ball construction, Respawn-with-new-pole, dtors)

| addr | bytes / insns | function | class | callers | callees (non-Box2D) | Box2D API | libc/libm | notes |
|---|---|---|---|---|---|---|---|---|
| 0x10004c9a0 | 16 / 10 | `atrophy::Time_GetHighResTimer()` | MIXED | Camera::Camera(float, float, float); PlayState::PlayState(bool); Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*); atrophy::Input_Update() |  |  | _mach_absolute_time | mach_absolute_time(); only use in the match: srandom() seeds (PlayState ctor, Camera ctor, Player ctor) |
| 0x100064ca0 | 16 / 6 | `Common::Save::Get()` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::PlayState(bool) |  |  |  | returns &Common::Save::_saveData (+0x74b3c level5unlocked, +0x74b3d showIcons) |
| 0x1000671d0 | 16 / 6 | `Common::Session::Get()` | GAMEPLAY | Common::Input::GetControllerNum(int); PlayState::PlayState(bool); Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) |  |  |  | returns &Common::Session::_sessionData (player records, +0x90 level index) |
| 0x10012350e | 46 / 12 | `Ball::Ball(float, float, b2World*, SpriteSheet*, SpriteSheet*, float)` | GAMEPLAY | PlayState::PlayState(bool); PlayState::Update() |  |  |  | C1 (same body as C2 @0x1001234e0): +0x18 world, +0x20 ballHeight px (PlayState+0x10), +0x28/+0x30 sheets, +0x38/+0x3c spin/anim = 0, +0x40 levelWidth px (PlayState+0x8), +0x44 frozen = 0. x arg (0) unused. |
| 0x1001235e0 | 1026 / 172 | `Ball::CreateBody()` | GAMEPLAY | PlayState::PlayState(bool); PlayState::Update() |  | b2Body::CreateFixture(b2FixtureDef const*) x3, b2PolygonShape::SetAsBox(float, float), b2World::CreateBody(b2BodyDef const*) x3, b2World::CreateJoint(b2JointDef const*) x2 |  | Ball body (+0, circle r=24/ratio, BODY_BALL, ballDamping, density 4.2 fric .1 rest .9 filter cat 0x20), trolley body (+8, circle r=10/ratio, linearDamping ballDamping*3, filter cat 0x100 mask 0), distance joint ball<->trolley (+0x10, e_distanceJoint, length 0 then driven by Ball::Update, frequencyHz 2, dampingRatio 0.2), anchor body BODY_ANCHOR (box 50/ratio, at x=-3000/ratio, y=ball y), prismatic joint (e_prismaticJoint, axis (1,0)) trolley<->anchor. Body creation order: ball, trolley, [distance joint], anchor, [prismatic joint]. |
| 0x1001276a6 | 4574 / 735 | `Level1::CreateBody(float, float)` | GAMEPLAY | Level1::Level1(float, float, b2World*) |  | b2Body::CreateFixture(b2FixtureDef const*) x19, b2Body::SetType(b2BodyType) x7, b2Fixture::SetFilterData(b2Filter const&) x12, b2PolygonShape::Set(b2Vec2 const*, int) x6, b2PolygonShape::SetAsBox(float, float) x6, b2PolygonShape::SetAsBox(float, float, b2Vec2 const&, float) x5, b2World::CreateBody(b2BodyDef const*) x9 |  | Level1 terrain - later |
| 0x100128884 | 14 / 6 | `Level1::Level1(float, float, b2World*)` | GAMEPLAY | PlayState::PlayState(bool) | Level1::CreateBody(float, float) |  |  | Level1 C1 (Kalgoorlie) - later |
| 0x1001288a0 | 4166 / 665 | `Level2::CreateBody(float, float)` | GAMEPLAY | Level2::Level2(float, float, b2World*) |  | b2Body::CreateFixture(b2FixtureDef const*) x17, b2Body::SetType(b2BodyType) x9, b2Fixture::SetFilterData(b2Filter const&) x6, b2PolygonShape::Set(b2Vec2 const*, int) x4, b2PolygonShape::SetAsBox(float, float) x6, b2PolygonShape::SetAsBox(float, float, b2Vec2 const&, float) x4, b2World::CreateBody(b2BodyDef const*) x11 |  | Flatirons terrain: static bodies, polygon/box fixtures, filters, BODY_GROUND/WALL userData (Level2::_friction etc.) |
| 0x1001298e6 | 14 / 6 | `Level2::Level2(float, float, b2World*)` | GAMEPLAY | PlayState::PlayState(bool) | Level2::CreateBody(float, float) |  |  | C1 thunk: +0 world; -> CreateBody |
| 0x100129902 | 4596 / 735 | `Level3::CreateBody(float, float)` | GAMEPLAY | Level3::Level3(float, float, b2World*) |  | b2Body::CreateFixture(b2FixtureDef const*) x16, b2Body::SetType(b2BodyType) x9, b2Fixture::SetFilterData(b2Filter const&) x5, b2PolygonShape::Set(b2Vec2 const*, int) x4, b2PolygonShape::SetAsBox(float, float) x8, b2PolygonShape::SetAsBox(float, float, b2Vec2 const&, float) x6, b2World::CreateBody(b2BodyDef const*) x13 |  | Level3 terrain - later |
| 0x10012aaf6 | 14 / 6 | `Level3::Level3(float, float, b2World*)` | GAMEPLAY | PlayState::PlayState(bool) | Level3::CreateBody(float, float) |  |  | Level3 C1 (Dakhla) - later |
| 0x10012ab12 | 2278 / 352 | `Level4::CreateBody(float, float)` | GAMEPLAY | Level4::Level4(float, float, b2World*) |  | b2Body::CreateFixture(b2FixtureDef const*) x6, b2Body::SetType(b2BodyType) x5, b2Fixture::SetFilterData(b2Filter const&), b2PolygonShape::SetAsBox(float, float) x5, b2PolygonShape::SetAsBox(float, float, b2Vec2 const&, float), b2World::CreateBody(b2BodyDef const*) x7 |  | Level4 terrain - later |
| 0x10012b3f8 | 14 / 6 | `Level4::Level4(float, float, b2World*)` | GAMEPLAY | PlayState::PlayState(bool) | Level4::CreateBody(float, float) |  |  | Level4 C1 (Princeton) - later |
| 0x10012b414 | 3010 / 481 | `Level5::CreateBody(float, float)` | GAMEPLAY | Level5::Level5(float, float, b2World*) |  | b2Body::CreateFixture(b2FixtureDef const*) x11, b2Body::SetType(b2BodyType) x7, b2Fixture::SetFilterData(b2Filter const&) x4, b2PolygonShape::Set(b2Vec2 const*, int) x3, b2PolygonShape::SetAsBox(float, float) x7, b2World::CreateBody(b2BodyDef const*) x9 |  | Level5 terrain - later |
| 0x10012bfd6 | 14 / 6 | `Level5::Level5(float, float, b2World*)` | GAMEPLAY | PlayState::PlayState(bool) | Level5::CreateBody(float, float) |  |  | Level5 C1 (Rainbow Highway) - later |
| 0x100132406 | 738 / 197 | `Player::~Player()` | GAMEPLAY | Player::~Player() | Common::Sound::Effect::~Effect(); Common::Sound::Persistent::~Persistent(); Pole::~Pole(); SpriteSheet::~SpriteSheet(); Stats::~Stats(); operator delete(void*) | b2World::DestroyBody(b2Body*) |  | D2: DestroyBody(body), ~Pole, sounds/sheets freed. |
| 0x1001326e8 | 10 / 4 | `Player::~Player()` | GAMEPLAY | PlayState::~PlayState() | Player::~Player() |  |  | D1 thunk |
| 0x1001326f2 | 1928 / 306 | `Player::SpawnPlayerBody(float, float)` | GAMEPLAY | Player::Respawn(float, float, bool, Pole*); Player::Respawn(float, float, bool, float, bool, bool) | Animation::Animation(); Animation::Play(AnimationData const*); operator delete(void*); operator new(unsigned long) | b2Body::CreateFixture(b2FixtureDef const*) x4, b2PolygonShape::Set(b2Vec2 const*, int), b2PolygonShape::SetAsBox(float, float, b2Vec2 const&, float), b2World::CreateBody(b2BodyDef const*), b2World::DestroyBody(b2Body*) |  | Creates rider body (+0x30) with hitbox fixtures (Player::_hitBoxWidth/Height, _friction/_density/_restitution, FIXTURE_SENSOR/IMPALE, BODY_PLAYERn userData) + walk Animation (+0x68). Destroys the previous body if any (DestroyBody). |
| 0x1001331a2 | 992 / 195 | `Player::Respawn(float, float, bool, float, bool, bool)` | GAMEPLAY | PlayState::PlayState(bool); PlayState::Update() | Player::LiftOff(bool); Player::SetGhost(bool, float); Player::SpawnPlayerBody(float, float); Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool); Pole::Segment(int); Pole::~Pole(); operator delete(void*); operator new(unsigned long) | b2World::CreateJoint(b2JointDef const*) x2 |  | Respawn with a NEW pole: deletes old pole, SpawnPlayerBody, new Pole(...) ctor, CreateJoint x2 (hands -> segments; revolute joints +0x78/+0x80), LiftOff, SetGhost. |
| 0x100133582 | 3070 / 694 | `Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*)` | MIXED | Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) | Common::Input::GetControllerNum(int); Common::Session::Get(); Common::Sound::Effect::Effect(char const*, Common::Sound::Priority); Common::Sound::Persistent::Persistent(char const*, bool, bool, Common::Sound::Priority, bool); SpriteSheet::SetBlendMode(atrophy::FragmentBlendMode); SpriteSheet::SetShader(char const*); SpriteSheet::SpriteSheet(char const*); Stats::Stats(int); atrophy::FilePath::FilePath(); atrophy::FilePath::printf(char const*, ...); atrophy::Resource_GetBaseDirectory(); atrophy::Time_GetHighResTimer(); operator delete(void*); operator new(unsigned long) |  | _srandom | Copies Common::Session record (7 ints) to +0x220..+0x238 (head# +0x22c, session player +0x228 -> +0xa8), world +0x10, player index +0x9c (0/2 = side -1 team, 1/3 = other), +0xa0=(idx+2)&3, team flag +0x98 (session side != -1), sheets/dust ptrs +0x58/+0x60, +0x200=-1, srandom(Time_GetHighResTimer()) (RNG reseed only), controller +0xa4=GetControllerNum; rest = SpriteSheets/sounds (Assets::snake picks sound set). Port the field init; skip resources. |
| 0x100134180 | 16 / 10 | `Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*)` | MIXED | PlayState::PlayState(bool) | Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) |  |  | C1 thunk |
| 0x100134190 | 4604 / 931 | `PlayState::PlayState(bool)` | MIXED | PlayState::PlayState(bool) | Animation::Animation(); Animation::Play(AnimationData const*); Background::Background(); Background_Level2::Background_Level2(); Background_Level3::Background_Level3(); Background_Level4::Background_Level4(); Background_Level5::Background_Level5(); Ball::Ball(float, float, b2World*, SpriteSheet*, SpriteSheet*, float); Ball::CreateBody(); Camera::Camera(float, float, float); Common::Achievements::UnlockAchievement(Common::Achievements::Achievements, int); Common::Save::Get(); Common::Session::Get(); Common::Sound::Effect::Effect(char const*, Common::Sound::Priority); Common::Sound::SetCompressor(bool, float, float); Common::Sound::SetReverb(bool, float, float); DebugRender::DebugRender(); Dust::Dust(SpriteSheet*); Flare::Flare(SpriteSheet*); Level1::Level1(float, float, b2World*); Level2::Level2(float, float, b2World*); Level3::Level3(float, float, b2World*); Level4::Level4(float, float, b2World*); Level5::Level5(float, float, b2World*); PlayState::_CalculateChineseNewYear(); Player::GetHeadNumber(); Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*); Player::Respawn(float, float, bool, float, bool, bool); ShockWave::ShockWave(SpriteSheet*); Sparks::Sparks(SpriteSheet*); SpriteSheet::SetBlendMode(atrophy::FragmentBlendMode); SpriteSheet::SetShader(char const*); SpriteSheet::SpriteSheet(char const*); Target::Target(float, float, bool); atrophy::FilePath::FilePath(); atrophy::FilePath::printf(char const*, ...); atrophy::Resource_GetBaseDirectory(); atrophy::Time_GetHighResTimer(); operator delete(void*); operator new(unsigned long) | b2DebugDraw::SetFlags(unsigned int), b2World::SetContactListener(b2ContactListener*), b2World::SetDebugDraw(b2DebugDraw*), b2World::b2World(b2Vec2 const&, bool) | _localtime x2, _random x2, _srandom, _time x2 | Whole match setup; _SetupWorld inlined (b2World gravity (0,20) doSleep=1, SetContactListener(this)). srandom(Time_GetHighResTimer()) first; 2x random() for snake/pencil; time()/localtime for Christmas; _CalculateChineseNewYear. Level switch on Assets::currentLevel (Flatirons = 2 -> Level2, +0x214=2); copies level statics to +0x8..+0x18 (NB: +0x18 spawnBoost always reads Level1::_spawnBoost=50, copy/paste bug in the original). Players from Common::Session (+0x4 of each 0x1c record = active), team assignment, Player ctor(world, slot, idx, team, 0, sheet, dust) + Respawn(x,-5,false,poleLen,false,false) (pole len = standardPoleLength, rainbowPoleLength on L5; head #7: Respawn(x,-5,false,standard-147,false,true)); +0x1a0[idx]=0. Ball ctor + CreateBody. SKIP: SpriteSheets, Dust/Sparks/Flare/ShockWave/Camera/Target/Background/Animation(+0x38/+0x40/+0x48), sounds, reverb/compressor, DebugRender (Assets::debug=0). Keep order of Box2D creation: Level body -> per player (Player ctor, Respawn -> SpawnPlayerBody + Pole) -> Ball::CreateBody. |
| 0x10013550c | 128 / 47 | `PlayState::_CalculateChineseNewYear()` | MIXED | PlayState::PlayState(bool) |  |  | _localtime, _time | time()/localtime -> Assets::chineseNewYear (bridge patches the call away: forced false). |
| 0x10013b098 | 1048 / 277 | `PlayState::~PlayState()` | GAMEPLAY | (root) | Background::~Background(); Background_Level2::~Background_Level2(); Background_Level3::~Background_Level3(); Background_Level4::~Background_Level4(); Background_Level5::~Background_Level5(); Ball::~Ball(); Camera::~Camera(); Common::Sound::Effect::~Effect(); Dust::~Dust(); Flare::~Flare(); Player::~Player(); ShockWave::~ShockWave(); Sparks::~Sparks(); SpriteSheet::~SpriteSheet(); Target::~Target(); operator delete(void*) | b2World::~b2World() |  | Deleting dtor (D0): ~PlayState body + operator delete. Teardown only (order of DestroyBody irrelevant once the world is deleted). |
| 0x10013b4b0 | 10 / 4 | `PlayState::~PlayState()` | GAMEPLAY | (root) | PlayState::~PlayState() |  |  | D1 thunk |
| 0x10013b4ba | 1038 / 276 | `PlayState::~PlayState()` | GAMEPLAY | PlayState::~PlayState() | Background::~Background(); Background_Level2::~Background_Level2(); Background_Level3::~Background_Level3(); Background_Level4::~Background_Level4(); Background_Level5::~Background_Level5(); Ball::~Ball(); Camera::~Camera(); Common::Sound::Effect::~Effect(); Dust::~Dust(); Flare::~Flare(); Player::~Player(); ShockWave::~ShockWave(); Sparks::~Sparks(); SpriteSheet::~SpriteSheet(); Target::~Target(); operator delete(void*) | b2World::~b2World() |  | D2 (complete) dtor: deletes Players (~Player destroys pole/body), Ball, b2World, presentation objects; 1 virtual call *0x8 on +0x198 (DebugRender deleting dtor, null unless Assets::debug). |
| 0x10013d64c | 20 / 14 | `PlayState::PlayState(bool)` | GAMEPLAY | (root) | PlayState::PlayState(bool) |  |  | C1 thunk -> C2 @0x100134190 (entry point, arg 0) |
| 0x10013e5c2 | 74 / 24 | `Pole::~Pole()` | GAMEPLAY | Player::Respawn(float, float, bool, Pole*); Player::Respawn(float, float, bool, float, bool, bool); Player::~Player() |  | b2World::DestroyBody(b2Body*) x2 |  | if hammer (+0): DestroyBody(+0x68) first; then DestroyBody(segment 0..9 = +0x8..+0x50); world ptr at +0x70. Order matters when a pole is replaced mid-match. |
| 0x10013e64c | 2318 / 379 | `Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool)` | GAMEPLAY | Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool) |  | b2Body::CreateFixture(b2FixtureDef const*) x3, b2PolygonShape::SetAsBox(float, float) x2, b2PolygonShape::SetAsBox(float, float, b2Vec2 const&, float) x2, b2WeldJointDef::Initialize(b2Body*, b2Body*, b2Vec2 const&) x2, b2World::CreateBody(b2BodyDef const*) x2, b2World::CreateJoint(b2JointDef const*) x2 | _cosf, _sinf | 10 segment bodies (+0x8..+0x50) along angle (cosf/sinf!), fixtures (Pole::_poleWidth/_friction/_density, damping Assets::pole*Damping), BODY_POLE/BODY_HAMMER userData, weld joints (b2WeldJointDef::Initialize), hammer head body +0x68 in hammer mode (+0 flag), circle shapes. |
| 0x10013ef5a | 22 / 16 | `Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool)` | GAMEPLAY | Player::Respawn(float, float, bool, float, bool, bool) | Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool) |  |  | C1 thunk |
| 0x100141144 | 68 / 13 | `Stats::Stats(int)` | GAMEPLAY | Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) |  |  |  | C1 thunk: player#, counters=0, +0x10=-1 |
| 0x10014118e | 6 / 4 | `Stats::~Stats()` | GAMEPLAY | Player::~Player() |  |  |  | dtor (empty) |

## Package B: Player per-tick & state (Player::Update, pole servo, kick/land/lift-off, ghosting, filters, Animation, Pole helpers, Stats counters)

| addr | bytes / insns | function | class | callers | callees (non-Box2D) | Box2D API | libc/libm | notes |
|---|---|---|---|---|---|---|---|---|
| 0x100119b14 | 20 / 7 | `Animation::Animation()` | GAMEPLAY | Background::Background(); Background_Level2::Background_Level2(); Background_Level4::Background_Level4(); PlayState::PlayState(bool); Player::SpawnPlayerBody(float, float); ShockWave::ShockWave(SpriteSheet*); Target::Target(float, float, bool) |  |  |  | data=&nullAnimation, playing=0 |
| 0x100119b28 | 32 / 10 | `Animation::Play(AnimationData const*)` | GAMEPLAY | Background::Background(); Background_Level2::Background_Level2(); Background_Level4::Background_Level4(); PlayState::PlayState(bool); PlayState::Update(); PlayState::_ScoreGoal(int); Player::Kick(b2Vec2); Player::Land(bool); Player::LiftOff(bool); Player::SpawnPlayerBody(float, float); Player::Update(float); ShockWave::ShockWave(SpriteSheet*); ShockWave::Trigger(b2Vec2, b2Vec2); ShockWave::Update(float); Target::GoalScored(); Target::Target(float, float, bool); Target::Update(float) |  |  |  | if data!=d: data=d, pos=0, rate=1, playing=1 |
| 0x100119b48 | 48 / 15 | `Animation::Play(AnimationData const*, bool)` | GAMEPLAY | Player::ShowSmash(b2Vec2) |  |  |  | restart flag variant |
| 0x100119b78 | 104 / 25 | `Animation::Update(float)` | GAMEPLAY | PlayState::Update(); Player::Update(float); ShockWave::Update(float); Target::Update(float) |  |  |  | pos += dt*rate*fps/count; loop wrap |
| 0x100119bea | 12 / 6 | `Animation::SetRate(float)` | GAMEPLAY | Player::Update(float) |  |  |  | rate=+0xc |
| 0x100119c36 | 24 / 8 | `Animation::IsFinished()` | GAMEPLAY | PlayState::Update(); Player::Update(float); ShockWave::Update(float); Target::Update(float) |  |  |  | pos >= 1 |
| 0x100119c4e | 10 / 6 | `Animation::GetData()` | GAMEPLAY | Player::Kick(b2Vec2); Player::Land(bool); Player::Update(float); Target::Update(float) |  |  |  | accessor data |
| 0x100119c58 | 38 / 13 | `Animation::SetPosition(float)` | GAMEPLAY | Player::Update(float) |  |  |  | pos= |
| 0x100119c7e | 18 / 12 | `Animation::GetPosition()` | GAMEPLAY | Player::Update(float) |  |  |  | accessor pos |
| 0x10012c000 | 10 / 5 | `Player::GetPole()` | GAMEPLAY | PlayState::Update() |  |  |  | accessor +0x18 Pole* |
| 0x10012c00a | 10 / 5 | `Player::SharingPole()` | GAMEPLAY | PlayState::Update() |  |  |  | accessor +0x20 sharing |
| 0x10012c014 | 12 / 6 | `Player::SetRunSpeed(float)` | GAMEPLAY | PlayState::Update() |  |  |  | +0x24 = run speed |
| 0x10012c0a8 | 26 / 8 | `Player::OffPole()` | GAMEPLAY | PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int) |  |  |  | +0x1c7=1; if sharing +0x1c8=1 |
| 0x10012c0c2 | 14 / 6 | `Player::ShouldUnshare()` | GAMEPLAY | PlayState::Update() |  |  |  | accessor +0x1c8 |
| 0x10012c0d0 | 14 / 6 | `Player::ResetUnshare()` | GAMEPLAY | PlayState::Update() |  |  |  | +0x1c8=0 |
| 0x10012c0ec | 32 / 9 | `Player::PoleContactAdjust(int)` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::EndContactInner(int, int, int, int) |  |  |  | +0x1c0 += n, clamp >=0 |
| 0x10012c10c | 238 / 54 | `Player::WallCollision()` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int) |  |  |  | Inlined ApplyLinearImpulse (+-x) on rider body when on pole and airborne. |
| 0x10012c1fa | 10 / 5 | `Player::Body()` | GAMEPLAY | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int); PlayState::Splash(int); PlayState::Update() |  |  |  | accessor +0x30 body |
| 0x10012c204 | 24 / 6 | `Player::Boost(float)` | GAMEPLAY | PlayState::Update() |  |  |  | +0xb4=speed, +0xb8=0.8 |
| 0x10012c21c | 10 / 5 | `Player::OnPole()` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int); PlayState::Update() |  |  |  | accessor +0x28 on pole |
| 0x10012c226 | 10 / 5 | `Player::OnGround()` | GAMEPLAY | PlayState::PlayerCollision(b2Contact*, int, int); PlayState::Update() |  |  |  | accessor +0x29 on ground |
| 0x10012c23e | 14 / 6 | `Player::JustUnghosted()` | GAMEPLAY | PlayState::Update() |  |  |  | accessor +0x204 |
| 0x10012c256 | 24 / 9 | `Player::ShouldBallCollide()` | GAMEPLAY | PlayState::BallPlayerCollision(b2ContactImpulse const*, int) |  |  |  | +0x1cc <= 0 |
| 0x10012c26e | 14 / 6 | `Player::SetActive(bool)` | GAMEPLAY | PlayState::Update() |  |  |  | +0x1d0 = active (gates stick reading) |
| 0x10012c27c | 12 / 5 | `Player::GetHeadNumber()` | GAMEPLAY | PlayState::PlayState(bool); PlayState::Update() |  |  |  | accessor +0x22c head number (7 -> short pole) |
| 0x10012c30a | 10 / 6 | `Player::GetStats()` | GAMEPLAY | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int); PlayState::Update(); PlayState::_ScoreGoal(int) |  |  |  | accessor +0 Stats* |
| 0x10012c354 | 32 / 8 | `Player::GameOverBody()` | MIXED | PlayState::Update() | Common::Sound::Persistent::SetVolume(float) |  |  | +0x22=0, +0x190=0 (run-sound flags, also read by Update logic) + sound volume 0 |
| 0x10012c61c | 14 / 5 | `Player::GetHammer()` | GAMEPLAY | PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int) | Pole::Hammer() |  |  | Pole::Hammer(+0x18) |
| 0x10012fdb4 | 376 / 95 | `Player::_SetBallCollisionFilters()` | GAMEPLAY | Player::Land(bool); Player::SetBallCollisions(bool); Player::Update(float) | Pole::NumSegments(); Pole::Segment(int) | b2Fixture::SetFilterData(b2Filter const&) x3 |  | Filter data for ball collisions on body+pole segments (depends on +0x1cc). |
| 0x10012ff2c | 40 / 11 | `Player::SetBallCollisions(bool)` | GAMEPLAY | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int) | Player::_SetBallCollisionFilters() |  |  | +0x1cc = table value; _SetBallCollisionFilters |
| 0x10012ff54 | 1194 / 280 | `Player::SetGhost(bool, float)` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int); PlayState::Update(); Player::Respawn(float, float, bool, float, bool, bool); Player::UnsharePole(); Player::UpdateGhostTimer(float) | Pole::NumSegments(); Pole::Segment(int) | b2Fixture::SetFilterData(b2Filter const&) x13 |  | Ghost on/off: +0x1fd/+0x200 timer, filter data on body/pole fixtures. |
| 0x1001303fe | 74 / 21 | `Player::UpdateGhostTimer(float)` | GAMEPLAY | PlayState::Update() | Player::SetGhost(bool, float) |  |  | +0x204=0; ghost timer countdown (only while on pole) -> SetGhost(false). |
| 0x100130448 | 28 / 8 | `Player::UnsharePole()` | GAMEPLAY | PlayState::Update() | Player::SetGhost(bool, float) |  |  | +0x20=0; SetGhost(true, ...) |
| 0x100130464 | 226 / 63 | `Player::Teleport(float, bool)` | GAMEPLAY | PlayState::Update() | Pole::Hammer(); Pole::NumSegments(); Pole::Segment(int) | b2Body::SetTransform(b2Vec2 const&, float) x3 |  | SetTransform on body and all pole segments (+hammer). |
| 0x100130546 | 466 / 103 | `Player::_OffPole()` | GAMEPLAY | Player::Update(float) | Pole::Detach(bool) | b2Body::DestroyFixture(b2Fixture*), b2Fixture::SetFilterData(b2Filter const&) x3, b2World::DestroyJoint(b2Joint*) x2 | _fminf | Leave pole: rumble (skip), +0x28=0, DestroyJoint x2 (+0x78,+0x80), Pole::Detach, DestroyFixture, SetFilterData. |
| 0x100130718 | 426 / 108 | `Player::_FootBoxTo(float, float)` | GAMEPLAY | Player::Kick(b2Vec2); Player::Land(bool); Player::Update(float) |  | b2TestOverlap(b2Shape const*, b2Shape const*, b2Transform const&, b2Transform const&) |  | Moves foot/kick box fixture (polygon verts) and b2TestOverlap check. |
| 0x1001308c2 | 582 / 128 | `Player::Kick(b2Vec2)` | GAMEPLAY | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::PlayerCollision(b2Contact*, int, int) | Animation::GetData(); Animation::Play(AnimationData const*); Player::_FootBoxTo(float, float) |  | _atan2f | Kick direction: atan2f, picks kickAnim* (Animation::Play), _FootBoxTo. |
| 0x100130b08 | 166 / 44 | `Player::Land(bool)` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int) | Animation::GetData(); Animation::Play(AnimationData const*); Player::_FootBoxTo(float, float); Player::_SetBallCollisionFilters() |  |  | landingAnim, _FootBoxTo(0,0), _SetBallCollisionFilters, flags. |
| 0x100130d10 | 5342 / 1108 | `Player::Update(float)` | MIXED | PlayState::Update() | Animation::GetData(); Animation::GetPosition(); Animation::IsFinished(); Animation::Play(AnimationData const*); Animation::SetPosition(float); Animation::SetRate(float); Animation::Update(float); Common::Input::VirtualLeftStick(Common::Session::Player); Common::Input::VirtualRightStick(Common::Session::Player); Common::Input::VirtualUpdateVibration(float, float, Common::Session::Player); Common::Sound::Persistent::SetFrequency(float); Common::Sound::Persistent::SetVolume(float); Dust::AddParticle(b2Vec2, b2Vec2, bool); Player::_FootBoxTo(float, float); Player::_OffPole(); Player::_SetBallCollisionFilters(); Pole::NumSegments(); Pole::Segment(int) | b2RevoluteJoint::SetMaxMotorTorque(float) x3, b2RevoluteJoint::SetMotorSpeed(float) x4 | _atan2f x2, _fmaxf x7, _fminf x6, _fmodf x8 | Per-tick rider logic: rumble fade (+0x180/+0x184, fmaxf) + VirtualUpdateVibration (skip), inlined _UpdateSticks (VirtualLeftStick->+0x1ac.., VirtualRightStick->+0x19c.. when active +0x1d0), run force/walk, OffPole checks, Boost force (+0xb4/+0xb8), footbox (_FootBoxTo), kick-hold animation state machine (Animation data ptr compared with kickAnim*_hold tables -> gameplay!), fall/land anims, pole servo: atan2f(stick) vs segment angles (fmodf wraps) -> RevoluteJoint SetMotorSpeed/SetMaxMotorTorque (spinSpeedFactor, poleSwingTorque), _SetBallCollisionFilters, Animation::Update(+0x68). Dust::AddParticle, Persistent SetVolume/SetFrequency, ShouldAddFlare(+0x191) are presentation. Reads Assets::currentLevel (==2: flare flag only; ==4: extra dust) and shouldRumble. |
| 0x1001321ee | 200 / 46 | `Player::UpdateNormal(b2Contact*)` | GAMEPLAY | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) |  | b2WorldManifold::Initialize(b2Manifold const*, b2Transform const&, float, b2Transform const&, float) |  | b2WorldManifold::Initialize -> stores contact normal (on pole only). |
| 0x10013231e | 158 / 43 | `Player::LiftOff(bool)` | MIXED | PlayState::EndContactInner(int, int, int, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); Player::Respawn(float, float, bool, Pole*); Player::Respawn(float, float, bool, float, bool, bool) | Animation::Play(AnimationData const*); Common::Sound::Persistent::Play(); Stats::AddJump() |  | _random | liftoffAnim, Stats::AddJump, +0x1c4=1; if +0x1c0>0 random()%2 picks a jump sound (RNG advance only). |
| 0x100132e7a | 808 / 161 | `Player::Respawn(float, float, bool, Pole*)` | GAMEPLAY | PlayState::Update() | Common::Sound::Persistent::SetVolume(float); Player::LiftOff(bool); Player::SpawnPlayerBody(float, float); Pole::NumSegments(); Pole::Segment(int); Pole::~Pole(); operator delete(void*) | b2World::CreateJoint(b2JointDef const*) x2 |  | Respawn keeping/taking an existing Pole* (sharing): SpawnPlayerBody, joints to that pole, LiftOff. |
| 0x10013d660 | 20 / 11 | `Pole::Hammer()` | GAMEPLAY | Player::GetHammer(); Player::Teleport(float, bool) |  |  |  | +0 ? +0x68 : null |
| 0x10013d674 | 12 / 6 | `Pole::NumSegments()` | GAMEPLAY | PlayState::Update(); Player::Respawn(float, float, bool, Pole*); Player::SetGhost(bool, float); Player::Teleport(float, bool); Player::Update(float); Player::_SetBallCollisionFilters() |  |  |  | returns 10 |
| 0x10013d7da | 114 / 37 | `Pole::Detach(bool)` | GAMEPLAY | Player::_OffPole() |  | b2Fixture::SetFilterData(b2Filter const&) |  | SetFilterData on segments 1..10 when detaching. |
| 0x10013e596 | 44 / 17 | `Pole::Segment(int)` | GAMEPLAY | PlayState::Update(); Player::Respawn(float, float, bool, Pole*); Player::Respawn(float, float, bool, float, bool, bool); Player::SetGhost(bool, float); Player::Teleport(float, bool); Player::Update(float); Player::_SetBallCollisionFilters() | _Kernel_Panic |  |  | +0x8+8*i (bounds check -> _Kernel_Panic) |
| 0x1001411a6 | 10 / 6 | `Stats::AddKick()` | GAMEPLAY | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::PlayerCollision(b2Contact*, int, int) |  |  |  | +4++ kicks |
| 0x1001411ba | 10 / 6 | `Stats::AddJump()` | GAMEPLAY | Player::LiftOff(bool) |  |  |  | +8++ jumps |
| 0x1001411ce | 10 / 6 | `Stats::AddSuicide()` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int) |  |  |  | +0xc++ suicides |
| 0x1001411fa | 10 / 6 | `Stats::AddKill()` | GAMEPLAY | PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int) |  |  |  | +0x14++ kills |
| 0x10014120e | 10 / 6 | `Stats::AddDeath()` | GAMEPLAY | PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int) |  |  |  | +0x18++ deaths |

## Package C: Match logic (PlayState::Update, Ball::Update, contact listener + all collision handlers, scoring, sudden death, punch time)

| addr | bytes / insns | function | class | callers | callees (non-Box2D) | Box2D API | libc/libm | notes |
|---|---|---|---|---|---|---|---|---|
| 0x10012353c | 10 / 6 | `Ball::Body()` | GAMEPLAY | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::BallWallCollision(); PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int); PlayState::Update() |  |  |  | accessor +0 |
| 0x100123550 | 144 / 43 | `Ball::Ghost()` | GAMEPLAY | PlayState::BallWallCollision() |  | b2Fixture::SetFilterData(b2Filter const&) x2 |  | +0x44=1, filter {cat 0x20, mask 0, group 0} on both ball bodies |
| 0x100124080 | 324 / 76 | `Ball::Update(float)` | GAMEPLAY | PlayState::Update() |  | b2Body::SetType(b2BodyType) | _pow | +0x38 spin (wrap 2pi, presentation), +0x3c; if frozen SetType(static); distance-joint length (+0x10)->+0xb8 = |x|*ratio/levelWidth + 2; ApplyForce(-35*pow(x/(levelWidth/ratio),3), 0) on ball body (inlined, wakes body). pow(double,3.0) == d*d*d bit-exactly here (tested all 2^32 floats on x86 & arm64). |
| 0x1001241c4 | 44 / 15 | `Ball::~Ball()` | GAMEPLAY | PlayState::Update(); PlayState::~PlayState() |  | b2World::DestroyBody(b2Body*) x2 |  | DestroyBody(ball +0) then DestroyBody(trolley +8) (their joints die with them). The static BODY_ANCHOR body is NOT destroyed: every ball respawn after a goal leaves one more anchor body in the world (body list order/count matters for bit-exactness). Called on ball respawn in Update and in ~PlayState. |
| 0x10013544c | 6 / 4 | `PlayState::PreSolve(b2Contact*, b2Manifold const*)` | GAMEPLAY | (root) |  |  |  | b2ContactListener slot 4: empty (PreSolveInner is `ret`). Must still exist (Box2D calls it). |
| 0x10013560a | 652 / 107 | `PlayState::_MakeDeleters()` | GAMEPLAY | PlayState::Update() |  | b2Body::CreateFixture(b2FixtureDef const*) x2, b2Body::SetType(b2BodyType) x2, b2PolygonShape::SetAsBox(float, float), b2World::CreateBody(b2BodyDef const*) x2 |  | Creates static BODY_DELETER boxes (called from Update when level==4 only). |
| 0x100136298 | 176 / 61 | `PlayState::EndContactInner(int, int, int, int)` | GAMEPLAY | PlayState::EndContact(b2Contact*) | Player::HammerLiftoff(); Player::LiftOff(bool); Player::PoleContactAdjust(int) |  |  | LiftOff / HammerLiftoff / PoleContactAdjust(-1) |
| 0x100136348 | 164 / 56 | `PlayState::EndContact(b2Contact*)` | GAMEPLAY | (root) | PlayState::EndContactInner(int, int, int, int) |  |  | b2ContactListener slot 3; ContactToInts inlined -> EndContactInner |
| 0x1001389a4 | 860 / 186 | `PlayState::_ScoreGoal(int)` | MIXED | PlayState::BallWallCollision(); PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int) | Animation::Play(AnimationData const*); Common::Sound::Effect::Play(float, float, float); Player::GetStats(); Player::RumbleBump(float, float); Stats::AddGoal(); Stats::AddOwnGoal(); Stats::SetTimeLeft(float); Target::GoalScored() |  | _random | +0x118 = (team==0); team 0 -> +0x28++, else +0x2c++; random()%2 picks goal sound only; Stats AddGoal/AddOwnGoal for last toucher +0x30; if sudden death or a score reaches Assets::maxScore: +0x184 = time left, clock +0x180 = 1.0, +0x188 = 1, +0x1f4 = 0, Stats::SetTimeLeft for every player; +0x114 = 1.4 (ball-respawn/message timer). Skipped: score-arrow Animation (+0x38/+0x40), Target::GoalScored, RumbleBump, +0x1e8/+0x1f4 fx. |
| 0x10013942c | 358 / 82 | `PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2)` | MIXED | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::BallWallCollision(); PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int) | Camera::Shake(float, float, float); ShockWave::Trigger(b2Vec2, b2Vec2); Sparks::AddParticle(b2Vec2, float, int) |  | _atan2f, _random | Presentation (ShockWave::Trigger, Sparks::AddParticle w/ random()%51+40 count, atan2f for spark angle, Camera::Shake) BUT tail is inlined _PunchTime: +0x21c += 2; if +0x21c < 5 then +0x218 = 7 (7 frozen ticks). Port the tail only. |
| 0x100139592 | 2012 / 445 | `PlayState::PlayerCollision(b2Contact*, int, int)` | MIXED | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) | PlayState::_ScoreGoal(int); PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2); Player::Body(); Player::GetStats(); Player::Kick(b2Vec2); Player::OffPole(); Player::OnGround(); Player::OnPole(); Player::PlayKickSound(float); Stats::AddDeath(); Stats::AddKick(); Stats::AddKill() | b2WorldManifold::Initialize(b2Manifold const*, b2Transform const&, float, b2Transform const&, float) | _atan2f, _cosf | Rider-vs-rider (both on poles, bitmask +0x170 of alive riders): starts clock (+0x188), b2WorldManifold normal, impulse kickPlayerStrength/ratio, a=atan2f(n); |cosf(a)|>0.5 -> vertical (stomp, upper y wins, -32/ratio offset) else sideways (both airborne). Kill: contact FlagForFiltering inlined (flags|=8), victim OffPole, Stats AddDeath/AddKill/AddKick, killer Kick(dir), victim death timer +0x1a0[i]=deathDelay (normal) or sudden death: +0x34=1, +0x30=killer, _ScoreGoal, +0x178/+0x17c; +0x170 &= ~bit; inline ApplyLinearImpulse on both bodies (v +0x44/+0x48, w +0x4c, wake); _ShockwaveAt (punch time). PlayKickSound skipped. |
| 0x100139d6e | 1114 / 246 | `PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int)` | MIXED | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) | PlayState::_ScoreGoal(int); PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2); Player::Body(); Player::GetHammer(); Player::GetStats(); Player::LiftOff(bool); Player::OffPole(); Player::OnPole(); Player::PlayKickSound(float); Stats::AddDeath(); Stats::AddKill() | b2WorldManifold::Initialize(b2Manifold const*, b2Transform const&, float, b2Transform const&, float) | _atan2f | Hammer (sudden-death pole head) hits rider: atan2f window +-1.15..2 rad, kill + _ScoreGoal, LiftOff, _ShockwaveAt. |
| 0x10013a1c8 | 810 / 182 | `PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int)` | MIXED | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) | Ball::Body(); PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2); Player::GetHammer(); Player::PlayHitSound(float); Player::SetBallCollisions(bool) | b2WorldManifold::Initialize(b2Manifold const*, b2Transform const&, float, b2Transform const&, float) | _atan2f | Hammer hits ball: atan2f window, impulse on ball, SetBallCollisions(false), _ShockwaveAt; PlayHitSound skipped. |
| 0x10013a4f2 | 178 / 53 | `PlayState::BallWallCollision()` | GAMEPLAY | PlayState::BeginContactInner(b2Contact*, int, int, int, int) | Ball::Body(); Ball::Ghost(); PlayState::_ScoreGoal(int); PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2) |  |  | Ball hits goal wall: Ball::Ghost, _ScoreGoal(ball.x <= 0), _ShockwaveAt. |
| 0x10013a5a4 | 920 / 238 | `PlayState::BeginContactInner(b2Contact*, int, int, int, int)` | MIXED | PlayState::BeginContact(b2Contact*) | Common::Save::Get(); Common::Save::Save(); Common::Sound::Effect::Play(float, float, float); PlayState::BallWallCollision(); PlayState::Splash(int); PlayState::_ScoreGoal(int); Player::Body(); Player::GetStats(); Player::HammerLand(); Player::Land(bool); Player::OnPole(); Player::PlayDeathSound(float); Player::PoleContactAdjust(int); Player::SetGhost(bool, float); Player::WallCollision(); Stats::AddSuicide() |  |  | Land/Land(true) on ground/pole contacts, PoleContactAdjust(+1), HammerLand, WallCollision, BallWallCollision (only while +0x200<0), timer start +0x188, last toucher +0x30, DELETER contact -> suicide/ghost/death timers +0x1a0[] (deathDelay), sudden-death own-goal via _ScoreGoal; Rainbow-Highway unlock path (level5unlocked==0 only: Save(), currentLevel=5) - never taken with level5unlocked forced 1. PlayDeathSound (skip). SEA (L3) -> Splash. |
| 0x10013a93c | 176 / 59 | `PlayState::BeginContact(b2Contact*)` | GAMEPLAY | (root) | PlayState::BeginContactInner(b2Contact*, int, int, int, int) |  |  | b2ContactListener slot 2. ContactToInts inlined (tag = *(int*)userData or 0 if null; body userData @b2Body+0xb8, fixture userData @b2Fixture+0x48). Calls BeginContactInner(c, bodyTagA, fixTagA, bodyTagB, fixTagB) then again with A/B swapped (tail call). |
| 0x10013a9ec | 738 / 166 | `PlayState::BallPlayerCollision(b2ContactImpulse const*, int)` | MIXED | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) | Ball::Body(); PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2); Player::Body(); Player::GetStats(); Player::Kick(b2Vec2); Player::PlayBallSound(float); Player::PlayHitSound(float); Player::SetBallCollisions(bool); Player::ShouldBallCollide(); Stats::AddKick() |  |  | Rider kicks ball: ShouldBallCollide gate, Kick(dir), ball impulse kickBallStrength/ratio, SetBallCollisions, Stats::AddKick, _ShockwaveAt; sounds skipped. |
| 0x10013ad2e | 580 / 171 | `PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int)` | MIXED | PlayState::PostSolve(b2Contact*, b2ContactImpulse const*) | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int); PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int); PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int); Player::PlayBallSound(float); Player::PlayTap(float); Player::SetGhost(bool, float); Player::UpdateNormal(b2Contact*) |  | _fminf x2 | Dispatch by tags: UpdateNormal, BallPlayerCollision, PlayerCollision, HammerPlayerCollision, HammerBallCollision, SetGhost, GroundContact (presentation). BallContact/PoleGroundCollision inlined = sounds only (PlayBallSound/PlayTap). |
| 0x10013af72 | 192 / 64 | `PlayState::PostSolve(b2Contact*, b2ContactImpulse const*)` | GAMEPLAY | (root) | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) |  |  | b2ContactListener slot 5; ContactToInts inlined -> PostSolveInner |
| 0x10013b8c8 | 7340 / 1619 | `PlayState::Update()` | MIXED | (root) | Animation::IsFinished(); Animation::Play(AnimationData const*); Animation::Update(float); Background::SetVolume(float); Background_Level2::SetVolume(float); Background_Level3::SetVolume(float); Background_Level4::SetVolume(float); Background_Level5::SetVolume(float); Ball::Ball(float, float, b2World*, SpriteSheet*, SpriteSheet*, float); Ball::Body(); Ball::CreateBody(); Ball::Update(float); Ball::~Ball(); Camera::AddFocus(float, float); Camera::ClearFocus(); Camera::Shake(float, float, float); Camera::Update(float); Common::Achievements::UnlockAchievement(Common::Achievements::Achievements, int); Common::Input::ButtonHeld(int, Common::Input::Button); Common::Sound::Effect::Play(float, float, float); Dust::Fade(); Dust::Update(float); Flare::AddParticle(b2Vec2, float, b2Vec2); Flare::Update(float); PlayState::_MakeDeleters(); Player::Body(); Player::Boost(float); Player::GameOverBody(); Player::GetHeadNumber(); Player::GetPole(); Player::GetStats(); Player::JustUnghosted(); Player::OnGround(); Player::OnPole(); Player::ResetUnshare(); Player::Respawn(float, float, bool, Pole*); Player::Respawn(float, float, bool, float, bool, bool); Player::SetActive(bool); Player::SetGhost(bool, float); Player::SetRunSpeed(float); Player::SharingPole(); Player::ShouldAddFlare(); Player::ShouldUnshare(); Player::Taunting(); Player::Teleport(float, bool); Player::TriggerFart(); Player::UnsharePole(); Player::Update(float); Player::UpdateGhostTimer(float); Pole::NumSegments(); Pole::Segment(int); ShockWave::Update(float); Sparks::Fade(); Sparks::Update(float); Stats::IncrementOffenseRatio(float); Stats::IncrementTimeInEndzone(float); Stats::IncrementTimeInGoal(float); Stats::IncrementTimeNearestBall(float); Target::Update(float); operator delete(void*); operator new(unsigned long) | b2TestOverlap(b2Shape const*, b2Shape const*, b2Transform const&, b2Transform const&) x4, b2World::ClearForces(), b2World::Step(float, int, int) | _random x8 | Per-tick. dt=+0x18c (1/60); punch/hit-stop: +0x21c decays by dt, +0x218>0 -> decrement and dt=0 for this tick. +0x204/+0x200/+0x208 fades (+0x200 fade-out drives +0x1fc game over), Background SetVolume (skip). Sudden death via Common::Input::ButtonHeld(ctrl,0x4a) on both teams (inlined _GoSuddenDeath). Clock +0x180 runs when +0x188; at 0 -> tie: sudden death (+0x175), else +0x194 decided (maxScore compare), players SetActive(0)+GameOverBody, +0x190=3. UpdateGhostTimer for all; JustUnghosted -> b2TestOverlap vs other poles -> SetGhost. world->Step(dt,60,18); ClearForces; Player::Update x N; Ball::Update; Target/Animation/ShockWave/Dust/Sparks/Flare (skip); +0x110/+0x114 message timer; ball respawn after goal (~Ball, new Ball, CreateBody); Taunting->TriggerFart (presentation); out-of-bounds Teleport; Respawn logic (+0x1a0[i] death timers, Respawn(float,float,bool,float,bool,bool) or Respawn(..,Pole*), Boost, SetRunSpeed, +0x178/+0x17c); _MakeDeleters (Level4 only); game-over slow-mo dt/=1.5; Camera (skip); Stats Increment* (time nearest ball, offense ratio, endzone, goal). 8x random() only feed Flare::AddParticle. |
| 0x1001411e4 | 12 / 6 | `Stats::SetTimeLeft(float)` | GAMEPLAY | PlayState::_ScoreGoal(int) |  |  |  | +0x10 = time left |
| 0x100141224 | 16 / 6 | `Stats::IncrementTimeNearestBall(float)` | GAMEPLAY | PlayState::Update() |  |  |  | +0x1c += dt |
| 0x100141234 | 10 / 6 | `Stats::AddGoal()` | GAMEPLAY | PlayState::_ScoreGoal(int) |  |  |  | +0x20++ goals |
| 0x10014123e | 10 / 6 | `Stats::AddOwnGoal()` | GAMEPLAY | PlayState::_ScoreGoal(int) |  |  |  | +0x24++ own goals |
| 0x100141268 | 16 / 6 | `Stats::IncrementOffenseRatio(float)` | GAMEPLAY | PlayState::Update() |  |  |  | +0x28 += x |
| 0x100141290 | 16 / 6 | `Stats::IncrementTimeInGoal(float)` | GAMEPLAY | PlayState::Update() |  |  |  | +0x2c += dt |
| 0x1001412a0 | 16 / 6 | `Stats::IncrementTimeInEndzone(float)` | GAMEPLAY | PlayState::Update() |  |  |  | +0x30 += dt |

## Package D: Input chain (setJoystickPosition -> _UpdatePad -> Virtual*Stick / ButtonHeld)

| addr | bytes / insns | function | class | callers | callees (non-Box2D) | Box2D API | libc/libm | notes |
|---|---|---|---|---|---|---|---|---|
| 0x1000116ce | 1352 / 312 | `atrophy::Gamepad::setJoystickPosition(int, float, float)` | GAMEPLAY | atrophy::Input_Update() | atrophy::Event_DispatchEvent(atrophy::InputEvent const*) |  | _atan2f, _fmaxf x2, _fminf x2, _fmodf x3 | Per axis: v = sign(v)*fminf(1, fmaxf(0,|v|-dz)/(1-dz)) with dz = gp[+0x1c+0x30*stick]; if x*x+y*y > 1 both are divided by x*x+y*y (sic, squared length); optional piecewise-linear response table (gp+0xd0 ptr, +0xd8 count); stores gp[+0xc/+0x10 + 0x30*stick]. If emulated-button flag gp[+0x38+0x30*stick]: double-precision length > 0.5 -> atan2f + fmodf sectors (kEmulatedButtonAngles) -> button states + Event_DispatchEvent (menus, INFRA). |
| 0x10004d8e0 | 32 / 9 | `atrophy::Input_IsGamepadConnectedAt(int)` | GAMEPLAY | Common::Input::LeftStick(int); Common::Input::RightStick(int); Common::Input::VirtualLeftStick(Common::Session::Player); Common::Input::VirtualLeftTrigger(Common::Session::Player); Common::Input::VirtualPRTaunt(Common::Session::Player); Common::Input::_UpdatePad(int, float) |  |  |  | sGamepadList[i].connected |
| 0x10004d940 | 48 / 15 | `atrophy::Input_GetGamepadAt(int)` | GAMEPLAY | Common::Input::VirtualLeftTrigger(Common::Session::Player); Common::Input::VirtualPRTaunt(Common::Session::Player); Common::Input::_UpdatePad(int, float) |  |  |  | &sGamepadList[i] |
| 0x10005e340 | 48 / 12 | `Common::Input::ButtonHeld(int, Common::Input::Button)` | GAMEPLAY | PlayState::Update() |  |  |  | _pads[c].held[button] - used by Update for the both-teams sudden-death button (0x4a) |
| 0x10005e810 | 304 / 101 | `Common::Input::GetControllerNum(int)` | GAMEPLAY | Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) | Common::Session::Get() |  |  | Session -> controller index (Player ctor +0xa4) |
| 0x10005f0a0 | 256 / 57 | `Common::Input::RightStick(int)` | GAMEPLAY | Common::Input::VirtualRightStick(Common::Session::Player) | atrophy::Input_IsGamepadConnectedAt(int) |  |  | _pads[c]+0x10c/+0x110 (right stick) or digital fallback |
| 0x10005f1a0 | 240 / 55 | `Common::Input::LeftStick(int)` | GAMEPLAY | Common::Input::VirtualRightStick(Common::Session::Player) | atrophy::Input_IsGamepadConnectedAt(int) |  |  | _pads[c]+0xfc/+0x100 (left stick) or digital fallback |
| 0x10005f290 | 48 / 17 | `Common::Input::VirtualRightStick(Common::Session::Player)` | GAMEPLAY | Player::Update(float) | Common::Input::LeftStick(int); Common::Input::RightStick(int) |  |  | mode 1 -> LeftStick(ctrl) else RightStick(ctrl) |
| 0x10005f2c0 | 928 / 200 | `Common::Input::VirtualLeftStick(Common::Session::Player)` | GAMEPLAY | Player::Taunting(); Player::Update(float) | atrophy::Input_IsGamepadConnectedAt(int) |  |  | mode 1/2: shared-controller d-pad/buttons; else _pads[c]+0xfc/+0x100 with d-pad override (bigger |v| wins) |
| 0x10005f660 | 4384 / 862 | `Common::Input::_UpdatePad(int, float)` | GAMEPLAY | (root) | Common::Move::Active(int); Common::Move::GetAnyButtonHeld(int); Common::Move::GetButtonHeld(int, Common::Move::Button); Common::Move::GetVirtualDPad(int); atrophy::Gamepad::setMotorAt(int, float); atrophy::Input_GetGamepadAt(int); atrophy::Input_IsGamepadConnectedAt(int) |  |  | Per pad per frame (bridge calls it with dt=1/60+0.0002): clears 0x51 edge bytes, maps gamepad buttons -> _pads flags, copies sticks gp+0xc..0x18 -> _pads+0xfc..0x108 and gp+0x3c..0x48 -> +0x10c..0x118, stick-flick thresholds (0.4/0.8/0.2), triggers, hold timers (dt*7), rumble -> Gamepad::setMotorAt. Common::Move::* (PS Move, stubbed) consulted. |
| 0x100060e90 | 48 / 15 | `Common::Move::Active(int)` | GAMEPLAY | Common::Input::_UpdatePad(int, float) |  |  |  | PS Move (daemon stubbed in bridge) -> inactive |
| 0x100060fa0 | 64 / 21 | `Common::Move::GetButtonHeld(int, Common::Move::Button)` | GAMEPLAY | Common::Input::_UpdatePad(int, float) |  |  |  | PS Move |
| 0x100061070 | 208 / 56 | `Common::Move::GetAnyButtonHeld(int)` | GAMEPLAY | Common::Input::_UpdatePad(int, float) |  |  |  | PS Move |
| 0x100061140 | 64 / 18 | `Common::Move::GetVirtualDPad(int)` | GAMEPLAY | Common::Input::_UpdatePad(int, float) |  |  |  | PS Move |

## Presentation (skip) - PRES* first

PRES* functions write fields of Player/PlayState objects that only presentation reads (rumble +0x180/+0x184, sound gates +0x188/+0x189/+0x18a/+0x1d4, fart +0x1e4..+0x1ec/+0x1fc, smash +0x1f0..+0x1f8, flare +0x191, hammer counter +0x198). If the oracle diffs raw object bytes, either mask these or port them (they are tiny).

| addr | bytes / insns | function | class | callers | libc/libm | notes |
|---|---|---|---|---|---|---|
| 0x10012c020 | 14 / 6 | `Player::CanPlayLandSound()` | PRES* | PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int) |  | accessor +0x188 (land-sound gate) |
| 0x10012c02e | 14 / 6 | `Player::CanPlayHammerSound()` | PRES* | PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int) |  | accessor +0x189 |
| 0x10012c03c | 52 / 15 | `Player::HammerLand()` | PRES* | PlayState::BeginContactInner(b2Contact*, int, int, int, int) |  | +0x198 hammer-contact counter / +0x189 sound gate (only sound/ShowSmash read them) |
| 0x10012c070 | 42 / 12 | `Player::HammerLiftoff()` | PRES* | PlayState::EndContactInner(int, int, int, int) |  | +0x198-- ; +0x189=1 when 0 |
| 0x10012c09a | 14 / 6 | `Player::DisableLandSound()` | PRES* | PlayState::Splash(int) |  | +0x188=0 |
| 0x10012c288 | 130 / 27 | `Player::TriggerFart()` | PRES* | PlayState::Update() |  | Fart visual: +0x1e4 timer, +0x1e8/+0x1ec pos, +0x1fc |
| 0x10012c314 | 14 / 6 | `Player::ShouldAddFlare()` | PRES* | PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int); PlayState::Update() |  | accessor +0x191 (flare flag) |
| 0x10012c580 | 156 / 43 | `Player::PlayTap(float)` | PRES* | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) | _random | random()%3 sound pick; +0x1d4 cooldown |
| 0x10012d9e0 | 78 / 18 | `Player::RumbleBump(float, float)` | PRES* | PlayState::_ScoreGoal(int) | _fminf | rumble +0x180/+0x184 (fminf) |
| 0x10012da2e | 172 / 39 | `Player::PlayDeathSound(float)` | PRES* | PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::Splash(int) | _fminf, _random | random()%2 sound; rumble; +0x18a |
| 0x10012dada | 152 / 36 | `Player::PlayKickSound(float)` | PRES* | PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int); PlayState::PlayerCollision(b2Contact*, int, int) | _fminf, _random | random() sound; rumble |
| 0x10012db72 | 152 / 36 | `Player::PlayHitSound(float)` | PRES* | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int) | _fminf, _random | random() sound; rumble |
| 0x10012dc0a | 234 / 57 | `Player::PlayHammerSound(float, bool)` | PRES* | PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int) | _fminf, _random | random()%3 sound; rumble; +0x189 |
| 0x10012dcf4 | 166 / 35 | `Player::PlayLandSound(float, bool)` | PRES* | PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int) | _fminf | sound; rumble; +0x188 |
| 0x1001322b6 | 104 / 23 | `Player::ShowSmash(b2Vec2)` | PRES* | PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int) |  | +0x1f0..+0x1f8 + smash Animation (+0x70) |
| 0x10004d78e | 114 / 33 | `atrophy::Gamepad::setMotorAt(int, float)` | PRES | Common::Input::_UpdatePad(int, float) |  | rumble motor |
| 0x10005e2f0 | 16 / 5 | `Common::Achievements::UnlockAchievement(Common::Achievements::Achievements, int)` | PRES | PlayState::PlayState(bool); PlayState::Update() |  | no-op stub (5 instrs) |
| 0x10005e740 | 208 / 43 | `Common::Input::VirtualUpdateVibration(float, float, Common::Session::Player)` | PRES | Player::Update(float) | _fmaxf, _fminf | rumble output into _pads |
| 0x10005ed30 | 144 / 43 | `Common::Input::VirtualLeftTrigger(Common::Session::Player)` | PRES | Player::Taunting() |  | LT for taunt |
| 0x10005eea0 | 96 / 31 | `Common::Input::VirtualPRTaunt(Common::Session::Player)` | PRES | Player::Taunting() |  | R3 for taunt |
| 0x100067c5a | 276 / 79 | `Common::Sound::Persistent::Initiate()` | PRES | Common::Sound::Persistent::Play(); Common::Sound::Persistent::SetFrequency(float); Common::Sound::Persistent::SetVolume(float) |  |  |
| 0x100067ec2 | 112 / 33 | `Common::Sound::Persistent::SetFrequency(float)` | PRES | Player::Update(float) |  |  |
| 0x100067f9e | 108 / 33 | `Common::Sound::Persistent::SetVolume(float)` | PRES | Background::Background(); Background_Level2::Background_Level2(); Background_Level3::Background_Level3(); Background_Level4::Background_Level4(); Background_Lev |  |  |
| 0x1000680ce | 112 / 36 | `Common::Sound::Persistent::Play()` | PRES | Background::Background(); Background_Level2::Background_Level2(); Background_Level3::Background_Level3(); Background_Level4::Background_Level4(); Background_Lev |  |  |
| 0x10006813e | 62 / 23 | `Common::Sound::Persistent::~Persistent()` | PRES | Background::~Background(); Background_Level2::~Background_Level2(); Background_Level3::~Background_Level3(); Background_Level4::~Background_Level4(); Background |  |  |
| 0x1000681ba | 212 / 63 | `Common::Sound::Effect::Initiate()` | PRES | Common::Sound::Effect::PlayInternal(FMOD_CHANNELGROUP*, float, float, float) |  |  |
| 0x10006828e | 62 / 23 | `Common::Sound::Effect::~Effect()` | PRES | Background_Level5::~Background_Level5(); PlayState::~PlayState(); Player::~Player() |  |  |
| 0x10006830a | 372 / 108 | `Common::Sound::Effect::PlayInternal(FMOD_CHANNELGROUP*, float, float, float)` | PRES | Common::Sound::Effect::Play(float, float, float) |  |  |
| 0x1000684d2 | 30 / 10 | `Common::Sound::Effect::Play(float, float, float)` | PRES | PlayState::BeginContactInner(b2Contact*, int, int, int, int); PlayState::LoadComplete(); PlayState::Update(); PlayState::_ScoreGoal(int); Player::PlayBallSound( |  |  |
| 0x100068552 | 458 / 117 | `Common::Sound::Persistent::Persistent(char const*, bool, bool, Common::Sound::Priority, bool)` | PRES | Common::Sound::Persistent::Persistent(char const*, bool, bool, Common::Sound::Priority, bool) |  |  |
| 0x10006871c | 10 / 4 | `Common::Sound::Persistent::Persistent(char const*, bool, bool, Common::Sound::Priority, bool)` | PRES | Background::Background(); Background_Level2::Background_Level2(); Background_Level3::Background_Level3(); Background_Level4::Background_Level4(); Background_Lev |  |  |
| 0x100068726 | 342 / 90 | `Common::Sound::Effect::Effect(char const*, Common::Sound::Priority)` | PRES | Common::Sound::Effect::Effect(char const*, Common::Sound::Priority) |  |  |
| 0x10006887c | 20 / 5 | `Common::Sound::Effect::Effect(char const*, Common::Sound::Priority)` | PRES | Background_Level5::Background_Level5(); PlayState::PlayState(bool); Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) |  |  |
| 0x100068890 | 448 / 117 | `Common::Sound::SetCompressor(bool, float, float)` | PRES | PlayState::PlayState(bool) |  |  |
| 0x100068a50 | 496 / 130 | `Common::Sound::SetReverb(bool, float, float)` | PRES | PlayState::PlayState(bool) |  |  |
| 0x100069766 | 8 / 5 | `SpriteSheet::SetBlendMode(atrophy::FragmentBlendMode)` | PRES | Background_Level3::Background_Level3(); Background_Level5::Background_Level5(); PlayState::PlayState(bool); Player::Player(b2World*, int, int, int, int, SpriteS |  |  |
| 0x10006987e | 20 / 7 | `SpriteSheet::SetShader(char const*)` | PRES | PlayState::PlayState(bool); Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) |  |  |
| 0x1000698c4 | 196 / 62 | `SpriteSheet::~SpriteSheet()` | PRES | Background::~Background(); Background_Level2::~Background_Level2(); Background_Level3::~Background_Level3(); Background_Level4::~Background_Level4(); Background |  |  |
| 0x100069a4c | 180 / 46 | `SpriteSheet::SpriteSheet(char const*)` | PRES | Background::Background(); Background_Level2::Background_Level2(); Background_Level3::Background_Level3(); Background_Level4::Background_Level4(); Background_Lev |  |  |
| 0x100119c90 | 354 / 79 | `Background::Background()` | PRES | Background::Background() |  |  |
| 0x100119df2 | 12 / 6 | `Background::SetVolume(float)` | PRES | PlayState::Update() |  |  |
| 0x10011bc06 | 52 / 18 | `Background::LoadComplete()` | PRES | PlayState::LoadComplete() |  |  |
| 0x10011bc3a | 230 / 80 | `Background::~Background()` | PRES | Background::~Background() |  |  |
| 0x10011bd20 | 10 / 4 | `Background::~Background()` | PRES | PlayState::~PlayState() |  |  |
| 0x10011bd2a | 22 / 16 | `Background::Background()` | PRES | PlayState::PlayState(bool) |  |  |
| 0x10011bd40 | 268 / 64 | `Background_Level2::Background_Level2()` | PRES | Background_Level2::Background_Level2() |  |  |
| 0x10011be4c | 12 / 6 | `Background_Level2::SetVolume(float)` | PRES | PlayState::Update() |  |  |
| 0x10011ce44 | 52 / 18 | `Background_Level2::LoadComplete()` | PRES | PlayState::LoadComplete() |  |  |
| 0x10011ce78 | 228 / 78 | `Background_Level2::~Background_Level2()` | PRES | Background_Level2::~Background_Level2() |  |  |
| 0x10011cf5c | 10 / 4 | `Background_Level2::~Background_Level2()` | PRES | PlayState::~PlayState() |  |  |
| 0x10011cf66 | 10 / 4 | `Background_Level2::Background_Level2()` | PRES | PlayState::PlayState(bool) |  |  |
| 0x10011cf70 | 338 / 80 | `Background_Level3::Background_Level3()` | PRES | Background_Level3::Background_Level3() |  |  |
| 0x10011d0c2 | 12 / 6 | `Background_Level3::SetVolume(float)` | PRES | PlayState::Update() |  |  |
| 0x10011d0ce | 98 / 23 | `Background_Level3::Splash(b2Vec2, b2Vec2)` | PRES | PlayState::Splash(int) | _random |  |
| 0x10011e5ea | 52 / 18 | `Background_Level3::LoadComplete()` | PRES | PlayState::LoadComplete() |  |  |
| 0x10011e61e | 270 / 91 | `Background_Level3::~Background_Level3()` | PRES | Background_Level3::~Background_Level3() |  |  |
| 0x10011e72c | 10 / 4 | `Background_Level3::~Background_Level3()` | PRES | PlayState::~PlayState() |  |  |
| 0x10011e736 | 10 / 4 | `Background_Level3::Background_Level3()` | PRES | PlayState::PlayState(bool) |  |  |
| 0x10011e740 | 354 / 76 | `Background_Level4::Background_Level4()` | PRES | Background_Level4::Background_Level4() |  |  |
| 0x10011e8a2 | 14 / 5 | `Background_Level4::SetVolume(float)` | PRES | PlayState::Update() |  |  |
| 0x100121b56 | 86 / 26 | `Background_Level4::LoadComplete()` | PRES | PlayState::LoadComplete() |  |  |
| 0x100121bac | 352 / 125 | `Background_Level4::~Background_Level4()` | PRES | Background_Level4::~Background_Level4() |  |  |
| 0x100121d0c | 10 / 4 | `Background_Level4::~Background_Level4()` | PRES | PlayState::~PlayState() |  |  |
| 0x100121d16 | 10 / 4 | `Background_Level4::Background_Level4()` | PRES | PlayState::PlayState(bool) |  |  |
| 0x100121d20 | 448 / 101 | `Background_Level5::Background_Level5()` | PRES | Background_Level5::Background_Level5() |  |  |
| 0x100121ee0 | 12 / 6 | `Background_Level5::SetVolume(float)` | PRES | PlayState::Update() |  |  |
| 0x100123278 | 102 / 30 | `Background_Level5::LoadComplete()` | PRES | PlayState::LoadComplete() |  |  |
| 0x1001232de | 484 / 167 | `Background_Level5::~Background_Level5()` | PRES | Background_Level5::~Background_Level5() |  |  |
| 0x1001234c2 | 10 / 4 | `Background_Level5::~Background_Level5()` | PRES | PlayState::~PlayState() |  |  |
| 0x1001234cc | 20 / 14 | `Background_Level5::Background_Level5()` | PRES | PlayState::PlayState(bool) |  |  |
| 0x100124294 | 6 / 4 | `Camera::~Camera()` | PRES | PlayState::~PlayState() |  |  |
| 0x10012429a | 32 / 9 | `Camera::ClearFocus()` | PRES | PlayState::Update() |  |  |
| 0x1001242ba | 22 / 8 | `Camera::Shake(float, float, float)` | PRES | PlayState::Update(); PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2) |  |  |
| 0x100124468 | 394 / 91 | `Camera::Update(float)` | PRES | PlayState::Update() | _fmaxf, _fminf |  |
| 0x1001245f2 | 168 / 37 | `Camera::AddFocus(float, float)` | PRES | PlayState::Update() | _fmaxf x3, _fminf x3 |  |
| 0x10012469a | 118 / 26 | `Camera::Camera(float, float, float)` | PRES | PlayState::PlayState(bool) | _srandom | NB: ends with srandom(Time_GetHighResTimer()) - reseeds libc RNG during PlayState ctor |
| 0x100124710 | 324 / 70 | `DebugRender::DebugRender()` | PRES | DebugRender::DebugRender() |  | only when Assets::debug |
| 0x100125268 | 24 / 18 | `DebugRender::DebugRender()` | PRES | PlayState::PlayState(bool) |  |  |
| 0x1001252f6 | 128 / 24 | `Droplet::Droplet(SpriteSheet*, SpriteSheet*)` | PRES | Background_Level3::Background_Level3() |  |  |
| 0x10012537c | 6 / 4 | `Droplet::~Droplet()` | PRES | Background_Level3::~Background_Level3() |  |  |
| 0x1001257ca | 246 / 67 | `Droplet::AddParticle(b2Vec2, float, b2Vec2)` | PRES | Background_Level3::Splash(b2Vec2, b2Vec2) | _random |  |
| 0x10012593a | 128 / 23 | `Dust::Dust(SpriteSheet*)` | PRES | PlayState::PlayState(bool) |  |  |
| 0x1001259c0 | 6 / 4 | `Dust::~Dust()` | PRES | PlayState::~PlayState() |  |  |
| 0x100125cfc | 260 / 54 | `Dust::Update(float)` | PRES | PlayState::Update() | _fmaxf |  |
| 0x100125e00 | 280 / 57 | `Dust::Fade()` | PRES | PlayState::Update() | _fmaxf |  |
| 0x100125f18 | 1432 / 360 | `Dust::AddParticle(b2Vec2, b2Vec2, bool)` | PRES | PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int); Player::Update(float) | _fmaxf, _fminf, _pow, _random x18 |  |
| 0x100126522 | 112 / 22 | `Sparks::Sparks(SpriteSheet*)` | PRES | PlayState::PlayState(bool) |  |  |
| 0x100126598 | 6 / 4 | `Sparks::~Sparks()` | PRES | PlayState::~PlayState() |  |  |
| 0x10012659e | 118 / 25 | `Sparks::Fade()` | PRES | PlayState::Update() |  |  |
| 0x100126614 | 144 / 31 | `Sparks::Update(float)` | PRES | PlayState::Update() |  |  |
| 0x100126a30 | 512 / 113 | `Sparks::AddParticle(b2Vec2, float, int)` | PRES | PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2) | _cosf, _pow, _random x2, _sinf x2 |  |
| 0x100126c30 | 480 / 96 | `Flare::Flare(SpriteSheet*)` | PRES | Flare::Flare(SpriteSheet*) | _random x4 |  |
| 0x100126e16 | 6 / 4 | `Flare::~Flare()` | PRES | PlayState::~PlayState() |  |  |
| 0x100127298 | 370 / 95 | `Flare::AddParticle(b2Vec2, float, b2Vec2)` | PRES | Flare::Flare(SpriteSheet*); Flare::Update(float); PlayState::Update() | _random x4 |  |
| 0x10012740a | 658 / 128 | `Flare::Update(float)` | PRES | Flare::Flare(SpriteSheet*); PlayState::Update() | _fmaxf, _random x5 |  |
| 0x10012769c | 10 / 4 | `Flare::Flare(SpriteSheet*)` | PRES | PlayState::PlayState(bool) |  |  |
| 0x10012c39e | 346 / 77 | `Player::Taunting()` | PRES | PlayState::Update() | _atan2f | Taunt detection: VirtualLeftTrigger + VirtualLeftStick + VirtualPRTaunt, atan2f; result only triggers TriggerFart/sound on another player. |
| 0x10012c4f8 | 136 / 40 | `Player::PlayBallSound(float)` | PRES | PlayState::BallPlayerCollision(b2ContactImpulse const*, int); PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) |  | ball sound; Assets::ballSoundIndex cycles |
| 0x1001323bc | 74 / 18 | `Player::LoadComplete()` | PRES | PlayState::LoadComplete() |  | starts looping sounds |
| 0x100138d00 | 932 / 206 | `PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int)` | PRES | PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int) | _fminf x5, _random x2 | Land/hammer dust, sounds, ShowSmash, _Kernel_Log("player %d impulse"). 2x random() for Dust. Reads Assets::roundTime (sound gating). No gameplay writes (ShowSmash/Play*Sound touch Player presentation fields). |
| 0x1001390a4 | 904 / 177 | `PlayState::Splash(int)` | PRES | PlayState::BeginContactInner(b2Contact*, int, int, int, int) | _random x10 | SEA contact (Level3 water only): 10x random() -> Background_Level3::Splash; DisableLandSound, PlayDeathSound. |
| 0x10013d574 | 216 / 71 | `PlayState::LoadComplete()` | PRES |  |  | Calls Background*::LoadComplete, Player::LoadComplete (music loops), plays a sound. No gameplay state (port as no-op). |
| 0x100140d5e | 14 / 5 | `ShockWave::~ShockWave()` | PRES | PlayState::~PlayState() |  |  |
| 0x100140fae | 74 / 25 | `ShockWave::Update(float)` | PRES | PlayState::Update() |  |  |
| 0x100140ff8 | 176 / 46 | `ShockWave::Trigger(b2Vec2, b2Vec2)` | PRES | PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2) | _atan2f |  |
| 0x1001410a8 | 88 / 25 | `ShockWave::ShockWave(SpriteSheet*)` | PRES | PlayState::PlayState(bool) |  |  |
| 0x1001412b0 | 160 / 46 | `Target::Target(float, float, bool)` | PRES | Target::Target(float, float, bool) |  | goal target sprite+Animation |
| 0x100141350 | 22 / 7 | `Target::GoalScored()` | PRES | PlayState::_ScoreGoal(int) |  |  |
| 0x100141484 | 84 / 26 | `Target::Update(float)` | PRES | PlayState::Update() |  |  |
| 0x1001414d8 | 48 / 17 | `Target::~Target()` | PRES | PlayState::~PlayState() |  |  |
| 0x100141538 | 24 / 18 | `Target::Target(float, float, bool)` | PRES | PlayState::PlayState(bool) |  | C1 thunk |

## Infrastructure (skip)

Reached only through resource loading (SpriteSheet/Sound ctors -> atrophy::Resource_* -> synchronous load loop `_Kernel_Update` -> `Input_Update`/`File_Update`/SDL event pump), FMOD sound, config saving (`Common::Save::Save` on the Rainbow-Highway unlock), memory allocation and logging. NB: `_Kernel_Update` -> `atrophy::Input_Update` during a synchronous resource load can run the real SDL input pipeline (incl. setJoystickPosition) inside the PlayState ctor; irrelevant for the sim (the bridge overwrites pad state before every tick).

| addr | bytes / insns | function | reached from |
|---|---|---|---|
| 0x10000f960 | 16 / 5 | `atrophy::Event_Update()` | _Kernel_Update |
| 0x10000fa70 | 288 / 105 | `atrophy::Event_DispatchEvent(atrophy::InputEvent const*)` | atrophy::Gamepad::connect; atrophy::Gamepad::disconnect; atrophy::Gamepad::setButtonState; atrophy::Gamepad::setJoystickPosition; atrophy::I |
| 0x10000fb90 | 84 / 19 | `atrophy::InputFile::InputFile(char const*, unsigned int)` | atrophy::InputFileCocoa::OpenLocal |
| 0x10000fd70 | 160 / 39 | `atrophy::File_Update()` | _Kernel_Update |
| 0x10000ffb4 | 268 / 67 | `atrophy::InputFile::addToLoadQueue(int)` | atrophy::InputFile::loadAsync |
| 0x1000100c0 | 24 / 7 | `atrophy::InputFile::loadAsync(int, atrophy::IFileDelegate*, atrophy::FileLoadFlags)` | atrophy::ResourceTypeManager::createFromFile |
| 0x100010636 | 110 / 32 | `atrophy::InputFile::initDataAligned(unsigned int, unsigned int)` | atrophy::InputFileCocoa::startLoading |
| 0x1000111e6 | 50 / 15 | `atrophy::Gamepad::connect(int, atrophy::GamepadType)` | atrophy::GamepadSDL::connect |
| 0x100011376 | 178 / 51 | `atrophy::Gamepad::setButtonState(int, unsigned char)` | atrophy::GamepadSDL::updateAxis; atrophy::Input_ProcessEvent |
| 0x1000114ea | 484 / 114 | `atrophy::Gamepad::disconnect()` | atrophy::GamepadSDL::disconnect |
| 0x100011eb0 | 16 / 6 | `atrophy::Memory_Copy(void*, void const*, unsigned long)` | atrophy::FilePath_GetDirectory; atrophy::String_GetUriScheme |
| 0x100011ec0 | 16 / 6 | `atrophy::Memory_Set(void*, unsigned char, unsigned long)` | atrophy::Gamepad::disconnect; atrophy::GamepadSDL::disconnect; atrophy::GamepadSDL::updateRumble |
| 0x100011ed0 | 16 / 6 | `atrophy::Memory_Zero(void*, unsigned long)` | atrophy::Gamepad::disconnect |
| 0x100011fc0 | 32 / 10 | `operator delete[](void*)` | atrophy::FilePath_GetRelative; atrophy::InputFile::addToLoadQueue; atrophy::ResourceDirectory::addResource; atrophy::tokenizePath |
| 0x100012000 | 32 / 10 | `operator delete(void*)` | Background::Background; Background::~Background; Background_Level2::Background_Level2; Background_Level2::~Background_Level2; Background_Lev |
| 0x100012020 | 144 / 38 | `atrophy::Memory_AllocAligned(unsigned long, unsigned long, char const*, int)` | atrophy::Data::CreateAligned |
| 0x100012200 | 116 / 33 | `operator new[](unsigned long)` | atrophy::InputFile::addToLoadQueue; atrophy::ResourceDirectory::addResource; atrophy::tokenizePath |
| 0x1000123f0 | 128 / 45 | `operator new(unsigned long)` | Background::Background; Background_Level2::Background_Level2; Background_Level3::Background_Level3; Background_Level4::Background_Level4; Ba |
| 0x10001268a | 136 / 39 | `atrophy::SystemDialog::Update()` | _Kernel_Update |
| 0x10001a9c6 | 6 / 4 | `atrophy::Transform::Transform()` | DebugRender::DebugRender |
| 0x10001aafc | 44 / 12 | `atrophy::Transform::setTranslation(atrophy::Vector3 const&)` | DebugRender::DebugRender |
| 0x10001ab28 | 46 / 13 | `atrophy::Transform::setOrientation(atrophy::Quaternion const&)` | DebugRender::DebugRender |
| 0x10001ab56 | 46 / 13 | `atrophy::Transform::setScale(atrophy::Vector3 const&)` | DebugRender::DebugRender |
| 0x10002662e | 416 / 63 | `atrophy::Material::Material(atrophy::ShaderResource*)` | atrophy::Material::CreateFromTexture |
| 0x100026ffc | 210 / 65 | `atrophy::Material::initTexturesFromShader()` | atrophy::Material::CreateFromTexture |
| 0x100027128 | 248 / 77 | `atrophy::Material::CreateFromTexture(atrophy::Texture*, atrophy::FragmentBlendMode)` | Background::LoadComplete; Background_Level2::LoadComplete; Background_Level3::LoadComplete; Background_Level4::LoadComplete; Background_Leve |
| 0x100033bda | 60 / 17 | `atrophy::Texture::LoadFromResource(atrophy::TextureResource*, int)` | Background::LoadComplete; Background_Level2::LoadComplete; Background_Level3::LoadComplete; Background_Level4::LoadComplete; Background_Leve |
| 0x1000346d0 | 32 / 12 | `atrophy::Resource_GetBaseDirectory()` | Common::Sound::Effect::Effect; Common::Sound::Persistent::Persistent; PlayState::PlayState; Player::Player |
| 0x1000348b0 | 32 / 7 | `atrophy::Resource_DumpLoadedResources()` | atrophy::Memory_AllocAligned; operator new; operator new[] |
| 0x1000349a0 | 160 / 52 | `atrophy::getTypeForExtension(char const*)` | atrophy::loadFromFileAsync |
| 0x100034a40 | 256 / 76 | `atrophy::getFileEntry(char const*)` | atrophy::loadFromFileAsync |
| 0x100034b40 | 272 / 74 | `atrophy::loadFromFileAsync(char const*, int)` | atrophy::Resource_LoadFromFile; atrophy::Resource_LoadFromFileAsync |
| 0x100034c50 | 16 / 6 | `atrophy::Resource_LoadFromFileAsync(char const*)` | Background::Background; Background_Level2::Background_Level2; Background_Level3::Background_Level3; Background_Level4::Background_Level4; Ba |
| 0x100034e30 | 144 / 45 | `atrophy::Resource_GetPathAtLocation(atrophy::FilePath*, char const*, char const*)` | atrophy::ResourceTypeManager::getPath |
| 0x100034ec0 | 224 / 60 | `atrophy::Resource_GetPackageForFile(char const*)` | atrophy::Resource_GetFilePath |
| 0x100034fa0 | 144 / 48 | `atrophy::Resource_GetFilePath(atrophy::FilePath*, char const*)` | atrophy::ResourceTypeManager::getPath |
| 0x100035280 | 128 / 36 | `atrophy::Resource_LoadFromFile(char const*)` | Background_Level5::Background_Level5 |
| 0x100035dac | 80 / 31 | `atrophy::ResourceCache::popBack(char const*)` | atrophy::ResourceCache::addResource |
| 0x100036084 | 266 / 86 | `atrophy::ResourceCache::addResource(atrophy::Resource*)` | atrophy::loadFromFileAsync |
| 0x10003666c | 82 / 31 | `atrophy::ResourceTypeManager::hasExtension(char const*) const` | atrophy::getTypeForExtension |
| 0x1000367aa | 138 / 49 | `atrophy::ResourceTypeManager::createFromFile(char const*, unsigned int, int)` | atrophy::loadFromFileAsync |
| 0x100036834 | 146 / 50 | `atrophy::ResourceTypeManager::getPath(atrophy::FilePath*, char const*) const` | atrophy::loadFromFileAsync |
| 0x1000429d8 | 14 / 6 | `atrophy::Data::clear()` | atrophy::InputFile::initDataAligned |
| 0x100042b92 | 170 / 41 | `atrophy::Data::CreateAligned(unsigned int, unsigned int)` | atrophy::InputFile::initDataAligned |
| 0x1000432ea | 22 / 7 | `atrophy::FilePath::FilePath()` | Common::Sound::Effect::Effect; Common::Sound::Persistent::Persistent; PlayState::PlayState; Player::Player; atrophy::loadFromFileAsync |
| 0x100043300 | 16 / 8 | `atrophy::FilePath_IsAbsolute(char const*)` | atrophy::ResourcePackage::getDirectoryForFile; atrophy::ResourceTypeManager::getPath |
| 0x100043340 | 48 / 22 | `atrophy::FilePath_GetName(char const*)` | atrophy::Resource_GetFilePath; atrophy::getFileEntry |
| 0x100043370 | 1264 / 361 | `atrophy::tokenizePath(atrophy::FlexArray<atrophy::PathComponent>*, char const*)` | atrophy::FilePath_GetRelative |
| 0x100043960 | 80 / 30 | `atrophy::FilePath_GetExtension(char const*)` | atrophy::getTypeForExtension |
| 0x100043af0 | 144 / 51 | `atrophy::FilePath_GetDirectory(char*, int, char const*)` | atrophy::FilePath_MustGetDirectory |
| 0x100043b80 | 32 / 11 | `atrophy::FilePath_MustGetDirectory(char*, int, char const*)` | atrophy::ResourcePackage::getDirectoryForFile |
| 0x100043ba0 | 432 / 110 | `atrophy::FilePath_GetRelative(char const*, char const*)` | atrophy::ResourcePackage::getDirectoryForFile; atrophy::Resource_GetPathAtLocation; atrophy::getFileEntry |
| 0x100043f30 | 16 / 6 | `atrophy::FilePath::set(char const*)` | atrophy::ResourceTypeManager::getPath |
| 0x100043f70 | 16 / 6 | `atrophy::FilePath::FilePath(char const*)` | atrophy::InputFile::InputFile |
| 0x100043f90 | 192 / 52 | `atrophy::FilePath::printf(char const*, ...)` | Common::Sound::Effect::Effect; Common::Sound::Persistent::Persistent; PlayState::PlayState; Player::Player; atrophy::Resource_GetFilePath; a |
| 0x100046710 | 32 / 10 | `atrophy::String_AreEqual(char const*, char const*)` | atrophy::Material::CreateFromTexture |
| 0x100046730 | 80 / 30 | `atrophy::String_Copy(char*, char const*, int)` | atrophy::FacilityLog::log; atrophy::FilePath_GetDirectory |
| 0x100046820 | 96 / 34 | `atrophy::String_GetUriScheme(char*, int, char const*)` | atrophy::Resource_GetPackageForFile |
| 0x100046960 | 112 / 42 | `atrophy::String_MustCopy(char*, char const*, int)` | Common::Sound::Effect::Effect; Common::Sound::Persistent::Persistent; SpriteSheet::SetShader; atrophy::FilePath::FilePath; atrophy::FilePath |
| 0x100046a70 | 64 / 23 | `atrophy::String_Vprintf(char*, int, char const*, __va_list_tag*)` | atrophy::FacilityLog::log; atrophy::FilePath::printf |
| 0x100046c70 | 32 / 11 | `atrophy::String_AreEqualWithinCI(char const*, char const*, int)` | atrophy::FilePath_GetRelative; atrophy::ResourceDirectory::getDirectoryAtRelativePath |
| 0x100046cd0 | 32 / 10 | `atrophy::String_AreEqualCI(char const*, char const*)` | atrophy::ResourceDirectory::getResourceWithName; atrophy::ResourceTypeManager::hasExtension; atrophy::Resource_GetPackageForFile |
| 0x100047f60 | 46 / 15 | `atrophy::LogManager::getLastLog() const` | _Kernel_Panic |
| 0x100048404 | 242 / 76 | `atrophy::FacilityLog::log(atrophy::SeverityLevel, char const*, __va_list_tag*)` | _Kernel_Log; _Kernel_Panic |
| 0x100048808 | 72 / 14 | `atrophy::ShaderStateBlock::ShaderStateBlock()` | atrophy::Material::Material |
| 0x100048f14 | 354 / 98 | `atrophy::ResourceDatabase::dumpLoadedResources() const` | atrophy::Resource_DumpLoadedResources |
| 0x100049076 | 290 / 84 | `atrophy::ResourceDatabase::dumpAllResources() const` | atrophy::ResourceDatabase::createEntry |
| 0x100049198 | 168 / 48 | `atrophy::ResourceDatabase::createEntry(char const*)` | atrophy::getFileEntry |
| 0x10004950c | 90 / 31 | `atrophy::ResourceDirectory::getResourceWithName(char const*) const` | atrophy::Resource_GetFilePath; atrophy::Resource_GetPathAtLocation; atrophy::getFileEntry |
| 0x10004962e | 180 / 50 | `atrophy::ResourceDirectory::addResource(atrophy::ResourceEntry*)` | atrophy::getFileEntry |
| 0x1000496e2 | 168 / 55 | `atrophy::ResourceDirectory::getDirectoryAtRelativePath(char const*) const` | atrophy::ResourcePackage::getDirectoryForFile; atrophy::Resource_GetPathAtLocation |
| 0x10004989c | 224 / 62 | `atrophy::ResourcePackage::getDirectoryForFile(char const*)` | atrophy::Resource_GetFilePath; atrophy::getFileEntry |
| 0x10004c266 | 92 / 24 | `atrophy::InputFile::update()` | atrophy::File_Update |
| 0x10004c36a | 232 / 61 | `atrophy::InputFileCocoa::startLoading()` | atrophy::InputFile::startLoading |
| 0x10004c452 | 50 / 17 | `atrophy::InputFile::startLoading()` | atrophy::File_Update |
| 0x10004c51e | 70 / 22 | `atrophy::InputFile::finishLoading()` | atrophy::File_Update |
| 0x10004c698 | 152 / 40 | `atrophy::InputFileCocoa::OpenLocal(char const*)` | atrophy::File_OpenLocalFile |
| 0x10004c730 | 16 / 10 | `atrophy::File_OpenLocalFile(char const*)` | atrophy::ResourceTypeManager::createFromFile |
| 0x10004c980 | 32 / 10 | `atrophy::Time_GetMicroseconds(unsigned long long)` | atrophy::Input_Update |
| 0x10004da90 | 272 / 56 | `atrophy::Input_SetMousePosition(atrophy::Vector2 const&)` | atrophy::Input_ProcessEvent |
| 0x10004dbda | 418 / 90 | `atrophy::GamepadSDL::updateRumble()` | atrophy::Input_Update |
| 0x10004dda2 | 166 / 46 | `atrophy::GamepadSDL::updateAxis(SDL_ControllerAxisEvent const*)` | atrophy::Input_ProcessEvent |
| 0x10004dee0 | 116 / 28 | `atrophy::GamepadSDL::disconnect()` | atrophy::Input_ProcessEvent |
| 0x10004df54 | 508 / 129 | `atrophy::GamepadSDL::connect(_SDL_GameController*, atrophy::GamepadType)` | atrophy::connectController |
| 0x10004e150 | 272 / 78 | `atrophy::connectController(int)` | atrophy::Input_ProcessEvent; atrophy::Input_Update |
| 0x10004e4a0 | 1536 / 409 | `atrophy::Input_ProcessEvent(SDL_Event const*)` | atrophy::Input_Update |
| 0x10004eaa0 | 462 / 104 | `atrophy::Input_Update()` | _Kernel_Update |
| 0x10004f4d0 | 176 / 37 | `_Kernel_Log` | Common::Save::_SaveConfig; Common::Save::_SaveSecret; Common::Sound::Effect::Effect; Common::Sound::Effect::Initiate; Common::Sound::Effect: |
| 0x10004f580 | 304 / 67 | `_Kernel_Panic` | Common::Sound::Effect::Effect; Common::Sound::Effect::Initiate; Common::Sound::Persistent::Initiate; Common::Sound::Persistent::Persistent;  |
| 0x10004f720 | 64 / 15 | `_Kernel_Update` | atrophy::Resource_LoadFromFile |
| 0x10004f8a6 | 166 / 41 | `atrophy::SystemDialogOSX::open()` | atrophy::SystemDialog::open |
| 0x10004f952 | 6 / 4 | `atrophy::SystemDialog::update()` | atrophy::SystemDialog::Update |
| 0x10004fa6c | 20 / 14 | `atrophy::SystemDialog::open()` | atrophy::SystemDialog::Update |
| 0x10004fc30 | 16 / 6 | `atrophy::Window_GetMainWindow()` | _Kernel_Panic |
| 0x10005c9c0 | 6 / 4 | `atrophy::ResourcePackage_Update()` | atrophy::Resource_LoadFromFile |
| 0x100064d30 | 464 / 99 | `Common::Save::_SaveSecret()` | Common::Save::Save |
| 0x100064f00 | 240 / 55 | `(anonymous namespace)::SetFloat(_dictionary_*, Common::Save::SettingMeta const*, float)` | Common::Save::_SaveConfig |
| 0x100064ff0 | 2336 / 480 | `Common::Save::_SaveConfig()` | Common::Save::Save |
| 0x100065910 | 16 / 6 | `Common::Save::Save()` | PlayState::BeginContactInner |
| 0x100067620 | 1296 / 518 | `FMOD_ErrorString(FMOD_RESULT)` | Common::Sound::Effect::Effect; Common::Sound::Effect::Initiate; Common::Sound::Effect::PlayInternal; Common::Sound::Effect::~Effect; Common: |
| 0x1001943f0 | 16 / 5 | `_iniparser_freedict` | Common::Save::_SaveConfig |
| 0x100194c70 | 160 / 47 | `_iniparser_set` | ; Common::Save::_SaveConfig |
| 0x100195490 | 448 / 114 | `_iniparser_dumpsection_ini` | _iniparser_dump_ini |
| 0x1001956e0 | 384 / 123 | `_iniparser_dump_ini` | Common::Save::_SaveConfig |
| 0x100195970 | 128 / 40 | `_dictionary_del` | _iniparser_freedict |
| 0x1001959f0 | 128 / 36 | `_dictionary_new` | Common::Save::_SaveConfig |
| 0x100195b80 | 192 / 67 | `_dictionary_get` | _iniparser_dumpsection_ini |
| 0x100195c40 | 784 / 231 | `_dictionary_set` | _iniparser_set |
| 0x1001bdb20 | 16 / 5 | `_SDL_GetError` | atrophy::GamepadSDL::connect |
| 0x1001bdb50 | 16 / 5 | `_SDL_PumpEvents` | atrophy::Input_Update |
| 0x1001bdb60 | 16 / 6 | `_SDL_PeepEvents` | atrophy::Input_Update |
| 0x1001bdca0 | 16 / 5 | `_SDL_GameControllerMapping` | atrophy::GamepadSDL::connect |
| 0x1001bdcb0 | 16 / 5 | `_SDL_IsGameController` | atrophy::connectController |
| 0x1001bdcd0 | 16 / 5 | `_SDL_GameControllerOpen` | atrophy::connectController |
| 0x1001bdd00 | 16 / 5 | `_SDL_GameControllerGetJoystick` | atrophy::GamepadSDL::connect; atrophy::connectController |
| 0x1001bddb0 | 16 / 5 | `_SDL_GameControllerClose` | atrophy::GamepadSDL::disconnect |
| 0x1001bde70 | 16 / 5 | `_SDL_JoystickIsHaptic` | atrophy::GamepadSDL::connect |
| 0x1001bde80 | 16 / 5 | `_SDL_HapticOpenFromJoystick` | atrophy::GamepadSDL::connect |
| 0x1001bde90 | 16 / 5 | `_SDL_HapticClose` | atrophy::GamepadSDL::connect; atrophy::GamepadSDL::disconnect |
| 0x1001bded0 | 16 / 5 | `_SDL_HapticNumAxes` | atrophy::GamepadSDL::connect |
| 0x1001bdee0 | 16 / 6 | `_SDL_HapticEffectSupported` | atrophy::GamepadSDL::updateRumble |
| 0x1001bdef0 | 16 / 6 | `_SDL_HapticNewEffect` | atrophy::GamepadSDL::updateRumble |
| 0x1001bdf10 | 16 / 6 | `_SDL_HapticRunEffect` | atrophy::GamepadSDL::updateRumble |
| 0x1001bdf30 | 16 / 6 | `_SDL_HapticDestroyEffect` | atrophy::GamepadSDL::updateRumble |
| 0x1001bdfa0 | 16 / 5 | `_SDL_HapticRumbleSupported` | atrophy::GamepadSDL::connect |
| 0x1001bdfb0 | 16 / 5 | `_SDL_HapticRumbleInit` | atrophy::GamepadSDL::connect |
| 0x1001be040 | 16 / 5 | `_SDL_NumJoysticks` | atrophy::Input_Update |
| 0x1001be050 | 16 / 5 | `_SDL_JoystickNameForIndex` | atrophy::connectController |
| 0x1001be070 | 16 / 5 | `_SDL_JoystickName` | atrophy::GamepadSDL::connect |
| 0x1001be080 | 16 / 6 | `_SDL_JoystickGetDeviceGUID` | atrophy::connectController |
| 0x1001be090 | 16 / 6 | `_SDL_JoystickGetGUID` | atrophy::GamepadSDL::connect |
| 0x1001be0a0 | 16 / 6 | `_SDL_JoystickGetGUIDString` | atrophy::GamepadSDL::connect; atrophy::connectController |
| 0x1001be0d0 | 16 / 5 | `_SDL_JoystickInstanceID` | atrophy::GamepadSDL::connect; atrophy::connectController |
| 0x1001be1b0 | 16 / 5 | `_SDL_GetModState` | atrophy::Input_ProcessEvent |
| 0x1001beba0 | 16 / 5 | `_SDL_free` | atrophy::GamepadSDL::connect |

## Box2D API called from GAMEPLAY/MIXED code (call sites)

- `b2Body::CreateFixture(b2FixtureDef const*)` x81
- `b2Body::DestroyFixture(b2Fixture*)` x1
- `b2Body::SetTransform(b2Vec2 const&, float)` x3
- `b2Body::SetType(b2BodyType)` x40
- `b2DebugDraw::SetFlags(unsigned int)` x1
- `b2Fixture::SetFilterData(b2Filter const&)` x50
- `b2PolygonShape::Set(b2Vec2 const*, int)` x18
- `b2PolygonShape::SetAsBox(float, float)` x36
- `b2PolygonShape::SetAsBox(float, float, b2Vec2 const&, float)` x19
- `b2RevoluteJoint::SetMaxMotorTorque(float)` x3
- `b2RevoluteJoint::SetMotorSpeed(float)` x4
- `b2TestOverlap(b2Shape const*, b2Shape const*, b2Transform const&, b2Transform const&)` x5
- `b2WeldJointDef::Initialize(b2Body*, b2Body*, b2Vec2 const&)` x2
- `b2World::ClearForces()` x1
- `b2World::CreateBody(b2BodyDef const*)` x57
- `b2World::CreateJoint(b2JointDef const*)` x8
- `b2World::DestroyBody(b2Body*)` x6
- `b2World::DestroyJoint(b2Joint*)` x2
- `b2World::SetContactListener(b2ContactListener*)` x1
- `b2World::SetDebugDraw(b2DebugDraw*)` x1
- `b2World::Step(float, int, int)` x1
- `b2World::b2World(b2Vec2 const&, bool)` x1
- `b2World::~b2World()` x2
- `b2WorldManifold::Initialize(b2Manifold const*, b2Transform const&, float, b2Transform const&, float)` x4

Inlined Box2D (no call): `b2Body::ApplyForce`/`ApplyLinearImpulse`/`SetAwake` bodies appear inline in Ball::Update, Player::WallCollision, Player::Update (Boost) - flags |= 2 (awake), sleepTime (+0xb0) = 0, then force/velocity += ...; b2Vec2/b2Mul math inline everywhere.

## Other namespaces read by gameplay

- **Common::Session** (`Session::Get()` -> `Common::Session::_sessionData`): per-player records of 7 ints (stride 0x1c; PlayState ctor tests +0x4 of each record = slot used; Player ctor copies the record to Player+0x220..+0x238, i.e. mode/controller/head#), `+0x90` = arena index (currentLevel = +0x90 + 1, set in `PoleRiders::Begin`). Read at construction only.
- **Common::Save** (`Save::Get()` -> `Common::Save::_saveData`): `+0x74b34` score limit (float, -> `Assets::maxScore` by truncation in `PoleRiders::Begin`), `+0x74b38` duration minutes (float, x60 -> `Assets::roundTime`), `+0x74b3c` level5 unlocked (-> `Assets::level5unlocked` in PlayState ctor), `+0x74b3d` show icons, `+0x74b4a` rumble. `Save::Save()` only on the RH-unlock path.
- **Common::Input**: `_pads[8]` (stride 0x2c8) filled by `_UpdatePad`; read by VirtualLeftStick/VirtualRightStick/LeftStick/RightStick/ButtonHeld/VirtualLeftTrigger/VirtualPRTaunt; `GetControllerNum` (Player ctor). Written back by `VirtualUpdateVibration` (rumble only).
- **atrophy input**: `sGamepadList[4]` (GamepadSDL, stride 0x1b8) via `Input_GetGamepadAt`/`Input_IsGamepadConnectedAt`; written by `Gamepad::setJoystickPosition` / `setButtonState`.
- **Common::Move** (PS Move): `Active/GetButtonHeld/GetAnyButtonHeld/GetVirtualDPad` read `(anonymous)::gameData`; daemon stubbed by the bridge -> inactive.
- **Time / DT**: gameplay never reads wall-clock time or Common::DT - its dt is the constant `PlayState+0x18c = 0x3c888889` (1/60 f), set in the ctor (only modified by the game-over slow-motion `dt /= 1.5`). `atrophy::Time_GetHighResTimer()` is used only as `srandom()` seed (PlayState ctor, Camera ctor, each Player ctor); `time()/localtime()` only for Christmas/Chinese-New-Year/Boxing-day checks (forced off / achievements).
- **Random**: libc `random()` - see random_and_libm.md (no gameplay effect).
- **Config**: `PoleRiders::Begin()` @0x10013f880 (not called by the shadow) maps save/session -> `Assets::currentLevel/roundTime/maxScore/shouldRumble`; see globals.md.
