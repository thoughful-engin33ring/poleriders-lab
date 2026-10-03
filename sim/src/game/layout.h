// layout.h -- memory layout of the Super Pole Riders game objects, recovered from the
// original binary (Sportsfriends.orig, x86_64, LP64). Every named field sits at the exact
// byte offset the game uses; unknown / unused bytes are explicit _pad_XXX arrays.
// See sim/re_notes/layout.md for per-field writers/readers, ownership and the Box2D
// userData conventions.
//
// Conventions:
//   - "px" = pixels; Box2D metres = px / Assets::ratio (64).
//   - "// presentation" = only drawing/sound/particles/rumble read it (the sim may leave it
//     zero, but keep the slot so raw object dumps line up).
//   - "// ?" = meaning uncertain.
//   - Opaque engine types (SpriteSheet, sound objects, particle systems, camera, backgrounds)
//     are only forward-declared; their sizes are given in comments for completeness.
//   - Classes other than PlayState have NO vtable (only PlayState and DebugRender have
//     vtables in the binary).
#pragma once
#pragma clang diagnostic ignored "-Winvalid-offsetof"  // PlayState has a vptr (b2ContactListener base)

#include <Box2D/Box2D.h>
#include <cstddef>
#include <cstdint>

class SpriteSheet;                  // engine sprite sheet, 0x250 bytes            // presentation
namespace Common {
namespace Sound {
class Effect;                       // one-shot sound, 0x18 bytes                   // presentation
class Persistent;                   // looping sound, 0x20 bytes                    // presentation
}  // namespace Sound
}  // namespace Common

namespace PoleRiders {

// ---- opaque presentation-only objects owned by PlayState / Player (sizes from operator new) ----
struct Camera;              // 0x44                                                  // presentation
struct Dust;                // 0x3410  (AddParticle calls random() 18x!)             // presentation
struct Sparks;              // 0x3010  (AddParticle calls random())                  // presentation
struct Flare;               // 0xd010  (ctor/AddParticle/Update call random())       // presentation
struct ShockWave;           // 0x20                                                  // presentation
struct Droplet;             // 0x1818  (owned by Background_Level3)                  // presentation
struct Background;          // 0x358   (Level1 background)                           // presentation
struct Background_Level2;   // 0x48                                                  // presentation
struct Background_Level3;   // 0x68                                                  // presentation
struct Background_Level4;   // 0x88                                                  // presentation
struct Background_Level5;   // 0x88                                                  // presentation
struct DebugRender;         // 0x40, b2DebugDraw subclass, only if Assets::debug     // presentation

struct Stats;
struct Pole;
struct Ball;
struct Player;
struct Target;
struct Animation;

// =====================================================================================
// AnimationData: static tables in __DATA ((anonymous namespace)::walkAnim, kickAnimNE, ...),
// spaced 0xa0 apart in the binary. Gameplay compares Animation::data against these
// addresses and depends on IsFinished() timing (numFrames, fps, loop).
// =====================================================================================
struct AnimationData {
    int32_t frames[32];     // 0x00 sprite frame indices                              // presentation
    int32_t numFrames;      // 0x80 frame count (Update: pos += dt*rate*fps/numFrames)
    float   fps;            // 0x84 frames per second
    bool    loop;           // 0x88 wraps position back below 1.0 when set
    uint8_t _pad_089[3];
};
static_assert(offsetof(AnimationData, frames) == 0x00, "");
static_assert(offsetof(AnimationData, numFrames) == 0x80, "");
static_assert(offsetof(AnimationData, fps) == 0x84, "");
static_assert(offsetof(AnimationData, loop) == 0x88, "");
static_assert(sizeof(AnimationData) == 0x8c, "");

// =====================================================================================
// Animation (0x18, operator new(0x18)). Player+0x68 drives gameplay (kick state machine).
// =====================================================================================
struct Animation {
    const AnimationData *data;  // 0x00 current clip (ctor: &nullAnimation); Play() swaps it
    float   position;           // 0x08 normalized 0..1 (>=1 => IsFinished)
    float   rate;               // 0x0c playback rate multiplier (Play resets to 1.0)
    bool    playing;            // 0x10 IsPlaying(); cleared by GetFrame() at the end   // presentation
    uint8_t _pad_011[7];

    // ---- methods (package B: src/game/animation.cpp) ----
    Animation();                                          // @0x100119b14
    void Play(const AnimationData *d);                    // @0x100119b28
    void Play(const AnimationData *d, bool restart);      // @0x100119b48
    void Update(float dt);                                // @0x100119b78
    void SetRate(float r);                                // @0x100119bea
    bool IsFinished();                                    // @0x100119c36
    const AnimationData *GetData();                       // @0x100119c4e
    void SetPosition(float p);                            // @0x100119c58
    float GetPosition();                                  // @0x100119c7e
};
static_assert(offsetof(Animation, data) == 0x00, "");
static_assert(offsetof(Animation, position) == 0x08, "");
static_assert(offsetof(Animation, rate) == 0x0c, "");
static_assert(offsetof(Animation, playing) == 0x10, "");
static_assert(sizeof(Animation) == 0x18, "");

// =====================================================================================
// Stats (0x34, operator new(0x34) in Player::Player). Ctor: playerNum, zeros, timeLeft=-1.
// =====================================================================================
struct Stats {
    int32_t playerNum;          // 0x00 Stats(int) arg = Player::index
    int32_t kicks;              // 0x04 AddKick
    int32_t jumps;              // 0x08 AddJump (Player::LiftOff)
    int32_t suicides;           // 0x0c AddSuicide (deleter contact while on pole)
    float   timeLeft;           // 0x10 SetTimeLeft (clock snapshot at match end); ctor -1
    int32_t kills;              // 0x14 AddKill
    int32_t deaths;             // 0x18 AddDeath
    float   timeNearestBall;    // 0x1c IncrementTimeNearestBall
    int32_t goals;              // 0x20 AddGoal
    int32_t ownGoals;           // 0x24 AddOwnGoal
    float   offenseRatio;       // 0x28 IncrementOffenseRatio (+-dt by ball side)
    float   timeInGoal;         // 0x2c IncrementTimeInGoal
    float   timeInEndzone;      // 0x30 IncrementTimeInEndzone

    // ---- methods ----
    explicit Stats(int player);                           // @0x100141144 (package A)
    ~Stats();                                             // @0x10014118e (package A)
    void AddKick();                                       // @0x1001411a6 (package B: src/game/stats.cpp, all counters)
    void AddJump();                                       // @0x1001411ba
    void AddSuicide();                                    // @0x1001411ce
    void AddKill();                                       // @0x1001411fa
    void AddDeath();                                      // @0x10014120e
    void SetTimeLeft(float t);                            // @0x1001411e4
    void IncrementTimeNearestBall(float dt);              // @0x100141224
    void AddGoal();                                       // @0x100141234
    void AddOwnGoal();                                    // @0x10014123e
    void IncrementOffenseRatio(float x);                  // @0x100141268
    void IncrementTimeInGoal(float dt);                   // @0x100141290
    void IncrementTimeInEndzone(float dt);                // @0x1001412a0
};
static_assert(offsetof(Stats, playerNum) == 0x00, "");
static_assert(offsetof(Stats, kicks) == 0x04, "");
static_assert(offsetof(Stats, jumps) == 0x08, "");
static_assert(offsetof(Stats, suicides) == 0x0c, "");
static_assert(offsetof(Stats, timeLeft) == 0x10, "");
static_assert(offsetof(Stats, kills) == 0x14, "");
static_assert(offsetof(Stats, deaths) == 0x18, "");
static_assert(offsetof(Stats, timeNearestBall) == 0x1c, "");
static_assert(offsetof(Stats, goals) == 0x20, "");
static_assert(offsetof(Stats, ownGoals) == 0x24, "");
static_assert(offsetof(Stats, offenseRatio) == 0x28, "");
static_assert(offsetof(Stats, timeInGoal) == 0x2c, "");
static_assert(offsetof(Stats, timeInEndzone) == 0x30, "");
static_assert(sizeof(Stats) == 0x34, "");

// =====================================================================================
// Level / Level1..Level5 (all 8 bytes, identical layout, no vtable). The ctor stores the
// world and tail-calls LevelN::CreateBody(0,0); the ground bodies are NOT kept anywhere
// (only reachable through b2World's body list, tagged by userData).
// =====================================================================================
struct Level {
    b2World *world;             // 0x00 ctor arg

    // (package A: src/game/level.cpp) LevelN(float, float, b2World*) stores the world and calls
    // LevelN::CreateBody(float, float); declared on the subclasses below.
};
static_assert(offsetof(Level, world) == 0x00, "");
static_assert(sizeof(Level) == 0x08, "");
struct Level1 : Level {         // Kalgoorlie (PlayState::level == 1; the training map, menu cell "arena 2")
    Level1(float x, float y, b2World *w);                 // @0x100128884 (package A)
    void CreateBody(float x, float y);                    // @0x1001276a6 (package A)
};
struct Level2 : Level {         // Flatirons (PlayState::level == 2)
    Level2(float x, float y, b2World *w);                 // @0x1001298e6 (package A)
    void CreateBody(float x, float y);                    // @0x1001288a0 (package A)
};
struct Level3 : Level {};
struct Level4 : Level {};
struct Level5 : Level {};       // Rainbow Highway
static_assert(sizeof(Level2) == 0x08, "");

// =====================================================================================
// Target (0x20): goal-post decoration; PlayState::_ScoreGoal calls GoalScored().
// =====================================================================================
struct Target {
    float        x;             // 0x00 px (ctor arg)                                   // presentation
    float        y;             // 0x04 px (ctor arg)                                   // presentation
    bool         mirrored;      // 0x08 ctor bool: false = left target (target.ssht), true = right (target2.ssht)  // presentation
    uint8_t      _pad_009[7];
    Animation   *anim;          // 0x10 animWait / animHit                              // presentation
    SpriteSheet *sprites;       // 0x18                                                 // presentation
};
static_assert(offsetof(Target, x) == 0x00, "");
static_assert(offsetof(Target, y) == 0x04, "");
static_assert(offsetof(Target, mirrored) == 0x08, "");
static_assert(offsetof(Target, anim) == 0x10, "");
static_assert(offsetof(Target, sprites) == 0x18, "");
static_assert(sizeof(Target) == 0x20, "");

// =====================================================================================
// Pole (0x78, operator new(0x78) in Player::Respawn(float,float,bool,float,bool,bool)).
// Pole(x, y, world, team, index, angle, sprites, hammer, lengthPx, shared, shortPole)
// =====================================================================================
struct Pole {
    bool         isHammer;      // 0x00 ctor arg 8 (sudden death): Hammer() returns hammerBody iff set
    bool         shared;        // 0x01 ctor stack arg 1: pole shared by two riders (mode 1/2); changes segment userData
    bool         shortPole;     // 0x02 ctor stack arg 2 (= Player::shortPole)          // presentation (Pole::Draw only)
    uint8_t      _pad_003[1];
    float        lengthPx;      // 0x04 pole length in px (Assets::standardPoleLength etc.); hammer head offset
    b2Body      *segments[10];  // 0x08 segment bodies, welded chain; Segment(i), NumSegments()==10
    SpriteSheet *sprites;       // 0x58 (= Player::sharedSprites)                        // presentation
    int32_t      team;          // 0x60 ctor arg (selects collision filter bits)
    int32_t      ownerIndex;    // 0x64 player index; &ownerIndex is the userData of every pole fixture
                                //      and of the handle segments' bodies
    b2Body      *hammerBody;    // 0x68 hammer head body (BODY_HAMMER), only if isHammer (else uninitialized)
    b2World     *world;         // 0x70 ctor arg

    // ---- methods ----
    Pole(float x, float y, b2World *world, int a, int b, float c, SpriteSheet *sprites, bool d, float e,
         bool f, bool g);                                 // @0x10013e64c (package A; name the params when ported)
    ~Pole();                                              // @0x10013e5c2 (package A)
    b2Body *Hammer();                                     // @0x10013d660 (package B: src/game/pole.cpp)
    int NumSegments();                                    // @0x10013d674 (B)
    void Detach(bool b);                                  // @0x10013d7da (B)
    b2Body *Segment(int i);                               // @0x10013e596 (B)
};
static_assert(offsetof(Pole, isHammer) == 0x00, "");
static_assert(offsetof(Pole, shared) == 0x01, "");
static_assert(offsetof(Pole, shortPole) == 0x02, "");
static_assert(offsetof(Pole, lengthPx) == 0x04, "");
static_assert(offsetof(Pole, segments) == 0x08, "");
static_assert(offsetof(Pole, sprites) == 0x58, "");
static_assert(offsetof(Pole, team) == 0x60, "");
static_assert(offsetof(Pole, ownerIndex) == 0x64, "");
static_assert(offsetof(Pole, hammerBody) == 0x68, "");
static_assert(offsetof(Pole, world) == 0x70, "");
static_assert(sizeof(Pole) == 0x78, "");

// =====================================================================================
// Ball (0x48). Ball(x=0, heightPx, world, sprites, additiveSprites, levelWidthPx) + CreateBody().
// CreateBody also makes a static BODY_ANCHOR body + prismatic joint that are NOT stored here.
// =====================================================================================
struct Ball {
    b2Body          *body;          // 0x00 the ball (BODY_BALL, circle r=24px); Body()
    b2Body          *trolley;       // 0x08 zipline trolley (BODY_BALL, r=10px, filter cat 0x100 mask 0)
    b2DistanceJoint *wire;          // 0x10 ball<->trolley distance joint; Update sets its frequencyHz (+0xb8)
    b2World         *world;         // 0x18 ctor arg
    float            heightPx;      // 0x20 ctor arg (PlayState::ballHeight): spawn y in px
    uint8_t          _pad_024[4];
    SpriteSheet     *sprites;       // 0x28                                              // presentation
    SpriteSheet     *additive;      // 0x30                                              // presentation
    float            spin;          // 0x38 += dt, wrapped to 2pi (Draw)                 // presentation
    float            fadeIn;        // 0x3c 0 -> 1 by dt (Draw)                          // presentation
    float            levelWidthPx;  // 0x40 ctor arg (PlayState::levelWidth); Update force/spring scale
    bool             frozen;        // 0x44 Ghost() sets; Update then SetType(static); Frozen()
    uint8_t          _pad_045[3];

    // ---- methods ----
    Ball(float x, float y, b2World *world, SpriteSheet *s1, SpriteSheet *s2, float levelWidth);  // @0x10012350e (A)
    void CreateBody();                                    // @0x1001235e0 (A: src/game/ball.cpp)
    b2Body *Body();                                       // @0x10012353c (C: src/game/ball_update.cpp)
    void Ghost();                                         // @0x100123550 (C)
    void Update(float dt);                                // @0x100124080 (C)
    ~Ball();                                              // @0x1001241c4 (C)
};
static_assert(offsetof(Ball, body) == 0x00, "");
static_assert(offsetof(Ball, trolley) == 0x08, "");
static_assert(offsetof(Ball, wire) == 0x10, "");
static_assert(offsetof(Ball, world) == 0x18, "");
static_assert(offsetof(Ball, heightPx) == 0x20, "");
static_assert(offsetof(Ball, sprites) == 0x28, "");
static_assert(offsetof(Ball, additive) == 0x30, "");
static_assert(offsetof(Ball, spin) == 0x38, "");
static_assert(offsetof(Ball, fadeIn) == 0x3c, "");
static_assert(offsetof(Ball, levelWidthPx) == 0x40, "");
static_assert(offsetof(Ball, frozen) == 0x44, "");
static_assert(sizeof(Ball) == 0x48, "");

// =====================================================================================
// Common::Session::Player (0x1c): copied by value from Common::Session::Get()[slot] in
// Player::Player; passed by value to Common::Input::Virtual{Left,Right}Stick etc.
// =====================================================================================
struct SessionPlayer {
    int32_t mode;           // 0x00 0 = whole pad; 1/2 = shared-controller halves (unused in training)
    int32_t team;           // 0x04 0 = empty slot, -1 = team 0, else team 1 (PlayState ctor reads it)
    int32_t pad;            // 0x08 gamepad index (Input::_pads[pad]); also Player::inputId
    int32_t head;           // 0x0c head/character number (GetHeadNumber; 7 = short pole)
    int32_t unk10;          // 0x10 ?
    int32_t unk14;          // 0x14 ? (default 7)
    int32_t unk18;          // 0x18 ?
};
static_assert(offsetof(SessionPlayer, mode) == 0x00, "");
static_assert(offsetof(SessionPlayer, team) == 0x04, "");
static_assert(offsetof(SessionPlayer, pad) == 0x08, "");
static_assert(offsetof(SessionPlayer, head) == 0x0c, "");
static_assert(offsetof(SessionPlayer, unk10) == 0x10, "");
static_assert(offsetof(SessionPlayer, unk14) == 0x14, "");
static_assert(offsetof(SessionPlayer, unk18) == 0x18, "");
static_assert(sizeof(SessionPlayer) == 0x1c, "");

// =====================================================================================
// Player (0x240). Player(world, sessionSlot, index, team, 0, sharedSprites, dust).
// Body is (re)created by SpawnPlayerBody; pole by Respawn.
// =====================================================================================
struct Player {
    Stats       *stats;             // 0x000 new Stats(index) (last thing the ctor does)
    uint8_t      _pad_008[8];       // 0x008 never accessed
    b2World     *world;             // 0x010 ctor arg
    Pole        *pole;              // 0x018 owned unless sharingPole; GetPole()
    bool         sharingPole;       // 0x020 SharingPole(); set by Respawn(bool), cleared by UnsharePole
    bool         reversedGrip;      // 0x021 hands on the far pole end (shared-pole partner); picks segments in Update/SetGhost
    bool         bodyActive;        // 0x022 set by SpawnPlayerBody; cleared by GameOverBody/_ParalyzeBody/when an off-pole body settles
    bool         shortPole;         // 0x023 head #7: shorter pole, stiffer arm servo gains (0.014/0.07 vs 0.004/0.06)
    float        runSpeed;          // 0x024 run force scale (Respawn: 2100/ratio; SetRunSpeed)
    bool         onPole;            // 0x028 OnPole(); SpawnPlayerBody=1, _OffPole=0
    bool         onGround;          // 0x029 OnGround(); Land/LiftOff
    bool         standingOnPole;    // 0x02a StandingOnPole(); Land(bool onPole)
    uint8_t      _pad_02b[1];
    int32_t      unk2c;             // 0x02c ctor=0, never read ?
    b2Body      *body;              // 0x030 rider body; userData = &index; Body()
    SpriteSheet *bodySprites;       // 0x038 body{1,2,1b,2b}_white.ssht                   // presentation
    SpriteSheet *headSprites;       // 0x040 heads_white.ssht                             // presentation
    SpriteSheet *fartSprites;       // 0x048 fart.ssht                                    // presentation
    SpriteSheet *numberSprites;     // 0x050 playernumbers.ssht                           // presentation
    SpriteSheet *sharedSprites;     // 0x058 ctor arg (PlayState::sprites), handed to Pole // presentation
    Dust        *dust;              // 0x060 ctor arg (PlayState::dust)                   // presentation
    Animation   *anim;              // 0x068 body animation; kick state machine compares anim->data
    Animation   *smashAnim;         // 0x070 ShowSmash                                    // presentation
    b2RevoluteJoint *handJoint0;    // 0x078 body<->pole segment 0 (or n-2 if reversed), motor = arm servo
    b2RevoluteJoint *handJoint1;    // 0x080 body<->pole segment 2 (or n-3 if reversed), motor = arm servo
    b2Vec2       drawHand0;         // 0x088 smoothed hand position for Draw              // presentation
    b2Vec2       drawHand1;         // 0x090                                              // presentation
    int32_t      team;              // 0x098 0 or 1 (ctor arg)
    int32_t      index;             // 0x09c player index 0..3; body userData points HERE
    int32_t      partnerIndex;      // 0x0a0 (index+2)%4 = teammate; userData of far handle segs of a shared pole
    int32_t      controllerNum;     // 0x0a4 Input::GetControllerNum(slot)                // presentation
    int32_t      inputId;           // 0x0a8 = session.pad; ButtonHeld(inputId,...), achievements
    b2Vec2       groundNormal;      // 0x0ac contact normal (UpdateNormal); (0,-1) default; run force is along its tangent
    float        boostStrength;     // 0x0b4 Boost(f)
    float        boostTimer;        // 0x0b8 Boost sets 0.8, Update: force while >0, -= dt
    uint8_t      _pad_0bc[12];      // 0x0bc never accessed
    b2Fixture   *hullFixture;       // 0x0c8 9-vertex body polygon (_OffPole sets its m_restitution (+0x34) to 0.2)
    b2Fixture   *footFixture;       // 0x0d0 circle "foot box" (moved by _FootBoxTo; destroyed by _OffPole)
    b2Fixture   *sensorFixture;     // 0x0d8 sensor circle, userData FIXTURE_SENSOR
    b2Fixture   *impaleFixture;     // 0x0e0 sensor box above head, userData FIXTURE_IMPALE
    Common::Sound::Persistent *bendSoundA;  // 0x0e8                                      // presentation
    Common::Sound::Persistent *bendSoundB;  // 0x0f0                                      // presentation
    Common::Sound::Effect *deathSoundA;     // 0x0f8                                      // presentation
    Common::Sound::Effect *deathSoundB;     // 0x100                                      // presentation
    Common::Sound::Effect *landHardSound;   // 0x108                                      // presentation
    Common::Sound::Effect *landSoftSound;   // 0x110                                      // presentation
    Common::Sound::Effect *hitSoundA;       // 0x118                                      // presentation
    Common::Sound::Effect *hitSoundB;       // 0x120                                      // presentation
    Common::Sound::Effect *kickSoundA;      // 0x128                                      // presentation
    Common::Sound::Effect *kickSoundB;      // 0x130                                      // presentation
    Common::Sound::Effect *tapSound[3];     // 0x138                                      // presentation
    Common::Sound::Effect *ballSound[4];    // 0x150                                      // presentation
    Common::Sound::Persistent *walkSound;   // 0x170                                      // presentation
    Common::Sound::Persistent *slideSound;  // 0x178                                      // presentation
    float        rumbleA;           // 0x180 rumble accumulator (-dt per Update)           // presentation
    float        rumbleB;           // 0x184 rumble level (x0.95 per Update)               // presentation
    bool         canPlayLandSound;  // 0x188 (sound gating; sound fns call random())       // presentation
    bool         canPlayHammerSound;// 0x189                                               // presentation
    bool         canPlayDeathSound; // 0x18a                                               // presentation
    uint8_t      _pad_18b[1];
    float        bendVolume;        // 0x18c pole-bend sound volume                        // presentation
    bool         emitDust;          // 0x190 ? dust/slide effects allowed (cleared with bodyActive)
    bool         rainbowFlare;      // 0x191 ShouldAddFlare(): on level 2 near centre, high up; PlayState::Update then calls random() 8x/player
    uint8_t      _pad_192[2];
    int32_t      groundContacts;    // 0x194 Land ++ / LiftOff -- (LiftOff at <=0 => airborne)
    int32_t      hammerGroundContacts; // 0x198 HammerLand / HammerLiftoff
    float        rightStick[4];     // 0x19c VirtualRightStick() result: [0]=x [1]=y (pole-angle setpoint), [2],[3] never read
    float        leftStick[4];      // 0x1ac VirtualLeftStick(): [0]=x (run) [1]=y, [2],[3] never read
    float        animClock;         // 0x1bc += dt, wraps at 200 s                         // presentation
    int32_t      poleGroundContacts;// 0x1c0 PoleContactAdjust(+-1), clamped >= 0
    bool         liftoffFlag;       // 0x1c4 ? written by LiftOff/Update/SpawnPlayerBody, never read
    bool         kicking;           // 0x1c5 set by Kick, cleared by Land / when kick anim finishes; blocks kick-aim
    bool         facingLeft;        // 0x1c6 init team!=0; flips on turns; mirrors kick/foot box
    bool         offPolePending;    // 0x1c7 OffPole(): _OffPole() runs at next Update
    bool         unsharePending;    // 0x1c8 OffPole() while sharing; ShouldUnshare()
    uint8_t      _pad_1c9[3];
    float        ballNoCollideTimer;// 0x1cc >0 => ball collisions disabled (SetBallCollisions); -= dt
    bool         active;            // 0x1d0 SetActive(); false => sticks read as 0
    uint8_t      _pad_1d1[3];
    float        tapTimer;          // 0x1d4 PlayTap cooldown                              // presentation
    float        walkVolume;        // 0x1d8 walk sound volume                             // presentation
    float        tauntTimer;        // 0x1dc Taunting() cooldown (0.8 s)
    bool         tauntFlag;         // 0x1e0 ? cleared every Update / Respawn
    uint8_t      _pad_1e1[3];
    float        fartTimer;         // 0x1e4 TriggerFart = 1                               // presentation
    b2Vec2       fartPos;           // 0x1e8                                               // presentation
    float        smashTimer;        // 0x1f0 ShowSmash = 1                                 // presentation
    b2Vec2       smashPos;          // 0x1f4                                               // presentation
    bool         fartFlip;          // 0x1fc                                               // presentation
    bool         ghost;             // 0x1fd GetGhost(); SetGhost(b, t)
    uint8_t      _pad_1fe[2];
    float        ghostTimer;        // 0x200 GetGhostTimer(); counts down only while onPole; ctor -1
    bool         justUnghosted;     // 0x204 JustUnghosted(); set by SetGhost(false), cleared by UpdateGhostTimer
    uint8_t      _pad_205[3];
    float        paralyzeFade;      // 0x208 -1 = alive; 0..1 (+0.5*dt) after _ParalyzeBody   // presentation
    float        unk20c[4];         // 0x20c ? zeroed by SpawnPlayerBody, never read
    bool         drawFlag21c;       // 0x21c ? Draw only                                   // presentation
    uint8_t      _pad_21d[3];
    SessionPlayer session;          // 0x220 copy of Common::Session::Get()[slot]
    uint8_t      _pad_23c[4];

    // ---- methods ----
    // package A (src/game/player_spawn.cpp): construction / spawning / teardown
    Player(b2World *world, int a, int b, int c, int d, SpriteSheet *sprites, Dust *dust);  // @0x100133582
    ~Player();                                            // @0x100132406
    void SpawnPlayerBody(float x, float y);               // @0x1001326f2
    void Respawn(float x, float y, bool a, float b, bool c, bool d);  // @0x1001331a2 (new pole)
    // package B (src/game/player.cpp): per-tick & state
    Pole *GetPole();                                      // @0x10012c000
    bool SharingPole();                                   // @0x10012c00a
    void SetRunSpeed(float s);                            // @0x10012c014
    void OffPole();                                       // @0x10012c0a8
    bool ShouldUnshare();                                 // @0x10012c0c2
    void ResetUnshare();                                  // @0x10012c0d0
    void PoleContactAdjust(int n);                        // @0x10012c0ec
    void WallCollision();                                 // @0x10012c10c
    b2Body *Body();                                       // @0x10012c1fa
    void Boost(float speed);                              // @0x10012c204
    bool OnPole();                                        // @0x10012c21c
    bool OnGround();                                      // @0x10012c226
    bool JustUnghosted();                                 // @0x10012c23e
    bool ShouldBallCollide();                             // @0x10012c256
    void SetActive(bool a);                               // @0x10012c26e
    int GetHeadNumber();                                  // @0x10012c27c
    Stats *GetStats();                                    // @0x10012c30a
    void GameOverBody();                                  // @0x10012c354
    b2Body *GetHammer();                                  // @0x10012c61c
    void _SetBallCollisionFilters();                      // @0x10012fdb4
    void SetBallCollisions(bool on);                      // @0x10012ff2c
    void SetGhost(bool ghost, float t);                   // @0x10012ff54
    void UpdateGhostTimer(float dt);                      // @0x1001303fe
    void UnsharePole();                                   // @0x100130448
    void Teleport(float x, bool b);                       // @0x100130464
    void _OffPole();                                      // @0x100130546
    void _FootBoxTo(float x, float y);                    // @0x100130718
    void Kick(b2Vec2 dir);                                // @0x1001308c2
    void Land(bool b);                                    // @0x100130b08
    void Update(float dt);                                // @0x100130d10 (inlines _UpdateSticks, _ParalyzeBody)
    void UpdateNormal(b2Contact *c);                      // @0x1001321ee
    void LiftOff(bool b);                                 // @0x10013231e
    void HammerLiftoff();                                 // @0x10012c070 (EndContactInner; hammerGroundContacts--)
    void HammerLand();                                    // @0x10012c03c (BeginContactInner; hammerGroundContacts++ capped at 1)
    // package C (src/game/contacts.cpp): PRES* sound helpers called by the contact handlers. Only
    // their rumble part is ported (rumbleA/B feed Common::Input::_pads[+0xf4/+0xf8] via
    // Player::Update -> VirtualUpdateVibration, which the pad dumps compare); sounds/random() skipped.
    void RumbleBump(float a, float b);                    // @0x10012d9e0 (_ScoreGoal)
    void PlayDeathSound(float v);                         // @0x10012da2e (BeginContactInner)
    void PlayKickSound(float v);                          // @0x10012dada (PlayerCollision, HammerPlayerCollision)
    void PlayHitSound(float v);                           // @0x10012db72 (BallPlayerCollision, HammerBallCollision)
    void PlayHammerSound(float v, bool hard);             // @0x10012dc0a (GroundContact)
    void PlayLandSound(float v, bool hard);               // @0x10012dcf4 (GroundContact)
    void Respawn(float x, float y, bool a, Pole *pole);   // @0x100132e7a (reuse pole)
};
static_assert(offsetof(Player, stats) == 0x000, "");
static_assert(offsetof(Player, world) == 0x010, "");
static_assert(offsetof(Player, pole) == 0x018, "");
static_assert(offsetof(Player, sharingPole) == 0x020, "");
static_assert(offsetof(Player, reversedGrip) == 0x021, "");
static_assert(offsetof(Player, bodyActive) == 0x022, "");
static_assert(offsetof(Player, shortPole) == 0x023, "");
static_assert(offsetof(Player, runSpeed) == 0x024, "");
static_assert(offsetof(Player, onPole) == 0x028, "");
static_assert(offsetof(Player, onGround) == 0x029, "");
static_assert(offsetof(Player, standingOnPole) == 0x02a, "");
static_assert(offsetof(Player, unk2c) == 0x02c, "");
static_assert(offsetof(Player, body) == 0x030, "");
static_assert(offsetof(Player, bodySprites) == 0x038, "");
static_assert(offsetof(Player, headSprites) == 0x040, "");
static_assert(offsetof(Player, fartSprites) == 0x048, "");
static_assert(offsetof(Player, numberSprites) == 0x050, "");
static_assert(offsetof(Player, sharedSprites) == 0x058, "");
static_assert(offsetof(Player, dust) == 0x060, "");
static_assert(offsetof(Player, anim) == 0x068, "");
static_assert(offsetof(Player, smashAnim) == 0x070, "");
static_assert(offsetof(Player, handJoint0) == 0x078, "");
static_assert(offsetof(Player, handJoint1) == 0x080, "");
static_assert(offsetof(Player, drawHand0) == 0x088, "");
static_assert(offsetof(Player, drawHand1) == 0x090, "");
static_assert(offsetof(Player, team) == 0x098, "");
static_assert(offsetof(Player, index) == 0x09c, "");
static_assert(offsetof(Player, partnerIndex) == 0x0a0, "");
static_assert(offsetof(Player, controllerNum) == 0x0a4, "");
static_assert(offsetof(Player, inputId) == 0x0a8, "");
static_assert(offsetof(Player, groundNormal) == 0x0ac, "");
static_assert(offsetof(Player, boostStrength) == 0x0b4, "");
static_assert(offsetof(Player, boostTimer) == 0x0b8, "");
static_assert(offsetof(Player, hullFixture) == 0x0c8, "");
static_assert(offsetof(Player, footFixture) == 0x0d0, "");
static_assert(offsetof(Player, sensorFixture) == 0x0d8, "");
static_assert(offsetof(Player, impaleFixture) == 0x0e0, "");
static_assert(offsetof(Player, bendSoundA) == 0x0e8, "");
static_assert(offsetof(Player, bendSoundB) == 0x0f0, "");
static_assert(offsetof(Player, deathSoundA) == 0x0f8, "");
static_assert(offsetof(Player, deathSoundB) == 0x100, "");
static_assert(offsetof(Player, landHardSound) == 0x108, "");
static_assert(offsetof(Player, landSoftSound) == 0x110, "");
static_assert(offsetof(Player, hitSoundA) == 0x118, "");
static_assert(offsetof(Player, hitSoundB) == 0x120, "");
static_assert(offsetof(Player, kickSoundA) == 0x128, "");
static_assert(offsetof(Player, kickSoundB) == 0x130, "");
static_assert(offsetof(Player, tapSound) == 0x138, "");
static_assert(offsetof(Player, ballSound) == 0x150, "");
static_assert(offsetof(Player, walkSound) == 0x170, "");
static_assert(offsetof(Player, slideSound) == 0x178, "");
static_assert(offsetof(Player, rumbleA) == 0x180, "");
static_assert(offsetof(Player, rumbleB) == 0x184, "");
static_assert(offsetof(Player, canPlayLandSound) == 0x188, "");
static_assert(offsetof(Player, canPlayHammerSound) == 0x189, "");
static_assert(offsetof(Player, canPlayDeathSound) == 0x18a, "");
static_assert(offsetof(Player, bendVolume) == 0x18c, "");
static_assert(offsetof(Player, emitDust) == 0x190, "");
static_assert(offsetof(Player, rainbowFlare) == 0x191, "");
static_assert(offsetof(Player, groundContacts) == 0x194, "");
static_assert(offsetof(Player, hammerGroundContacts) == 0x198, "");
static_assert(offsetof(Player, rightStick) == 0x19c, "");
static_assert(offsetof(Player, leftStick) == 0x1ac, "");
static_assert(offsetof(Player, animClock) == 0x1bc, "");
static_assert(offsetof(Player, poleGroundContacts) == 0x1c0, "");
static_assert(offsetof(Player, liftoffFlag) == 0x1c4, "");
static_assert(offsetof(Player, kicking) == 0x1c5, "");
static_assert(offsetof(Player, facingLeft) == 0x1c6, "");
static_assert(offsetof(Player, offPolePending) == 0x1c7, "");
static_assert(offsetof(Player, unsharePending) == 0x1c8, "");
static_assert(offsetof(Player, ballNoCollideTimer) == 0x1cc, "");
static_assert(offsetof(Player, active) == 0x1d0, "");
static_assert(offsetof(Player, tapTimer) == 0x1d4, "");
static_assert(offsetof(Player, walkVolume) == 0x1d8, "");
static_assert(offsetof(Player, tauntTimer) == 0x1dc, "");
static_assert(offsetof(Player, tauntFlag) == 0x1e0, "");
static_assert(offsetof(Player, fartTimer) == 0x1e4, "");
static_assert(offsetof(Player, fartPos) == 0x1e8, "");
static_assert(offsetof(Player, smashTimer) == 0x1f0, "");
static_assert(offsetof(Player, smashPos) == 0x1f4, "");
static_assert(offsetof(Player, fartFlip) == 0x1fc, "");
static_assert(offsetof(Player, ghost) == 0x1fd, "");
static_assert(offsetof(Player, ghostTimer) == 0x200, "");
static_assert(offsetof(Player, justUnghosted) == 0x204, "");
static_assert(offsetof(Player, paralyzeFade) == 0x208, "");
static_assert(offsetof(Player, unk20c) == 0x20c, "");
static_assert(offsetof(Player, drawFlag21c) == 0x21c, "");
static_assert(offsetof(Player, session) == 0x220, "");
static_assert(sizeof(Player) == 0x240, "");

// =====================================================================================
// PlayState (0x228, operator new(0x228) in PoleRiders::Begin() / PoleRiders::Update()).
// Single inheritance from b2ContactListener: vtable at 0, and `this` itself is passed to
// b2World::SetContactListener (no secondary vtable). PlayState(bool fromRainbowUnlock).
// =====================================================================================
struct PlayState : public b2ContactListener {  // vptr at 0x000 (b2ContactListener), as in the game
    float        levelWidth;        // 0x008 px, LevelN::_levelWidth (Flatirons 3520)
    float        targetWidth;       // 0x00c px, LevelN::_targetWidth (endzone stats: |x| > targetWidth/3)
    float        ballHeight;        // 0x010 px, LevelN::_ballHeight (Ball spawn y)
    float        spawnHeight;       // 0x014 px, LevelN::_spawnHeight (respawn y)
    float        spawnBoost;        // 0x018 px, ALWAYS Level1::_spawnBoost (50) -- binary bug, see notes
    uint8_t      _pad_01c[4];
    b2World     *world;             // 0x020 new b2World(gravity (0,20)), contact listener = this
    int32_t      score[2];          // 0x028 [team]; _ScoreGoal(team) increments; zeroed on sudden death
    int32_t      lastTouch;         // 0x030 player index that last touched the ball (-1 none): goal/own-goal credit
    bool         koWin;             // 0x034 sudden death decided by a kill/deleter (Draw "KOText")
    bool         unlockingRainbow;  // 0x035 Unlocking(): rainbow-highway unlock in progress
    bool         timeOver;          // 0x036 TimeOver()/SetTimeOver(); set when the clock ends with a winner
    uint8_t      _pad_037[1];
    Animation   *scoreArrowAnim[2]; // 0x038 [team] scoreArrow1/2                           // presentation
    Animation   *gullAnim;          // 0x048                                               // presentation
    Player      *players[4];        // 0x050 [index]; team 0 = 0,2  team 1 = 1,3; only numPlayers valid
    Camera      *camera;            // 0x070                                               // presentation
    Ball        *ball;              // 0x078 recreated after every goal (ballResetPending)
    Target      *targets[2];        // 0x080 [0] left, [1] right; targets[1-team]->GoalScored()   // presentation
    ShockWave   *shockwaves[4];     // 0x090 [player] (numPlayers allocated)               // presentation
    Dust        *dust;              // 0x0b0                                               // presentation
    Sparks      *sparks;            // 0x0b8                                               // presentation
    Flare       *flare;             // 0x0c0                                               // presentation
    uint8_t      _pad_0c8[8];       // 0x0c8 never accessed
    Common::Sound::Effect *sndFanfare;     // 0x0d0                                        // presentation
    Common::Sound::Effect *sndGoal[4];     // 0x0d8 Goal1A,Goal1B,Goal2A,Goal2B (random() picks) // presentation
    Common::Sound::Effect *sndGoldenGoal;  // 0x0f8                                        // presentation
    Common::Sound::Effect *sndFart;        // 0x100                                        // presentation
    Common::Sound::Effect *sndSecret;      // 0x108                                        // presentation
    bool         ballResetPending;  // 0x110 set when ballResetDelay expires / sudden death starts; Update rebuilds ball + respawns everyone
    uint8_t      _pad_111[3];
    float        ballResetDelay;    // 0x114 1.4 s after a goal; >0 blocks _ScoreGoal and BallWallCollision
    bool         lastGoalTeam0;     // 0x118 last goal was by team 0 (camera focus)          // presentation
    uint8_t      _pad_119[7];
    Level1      *level1;            // 0x120 only the slot for `level` is written (others uninitialized)
    Level2      *level2;            // 0x128
    Level3      *level3;            // 0x130
    Level4      *level4;            // 0x138
    Level5      *level5;            // 0x140
    Background        *background1; // 0x148                                               // presentation
    Background_Level2 *background2; // 0x150                                               // presentation
    Background_Level3 *background3; // 0x158 (Splash() uses it for water droplets)         // presentation
    Background_Level4 *background4; // 0x160                                               // presentation
    Background_Level5 *background5; // 0x168                                               // presentation
    int32_t      killMask;          // 0x170 bit i = player i may still die in PlayerCollision this tick; reset to 0xf each Update
    bool         ballKickAvailable; // 0x174 one BallPlayerCollision kick per tick; reset to 1 each Update
    bool         suddenDeath;       // 0x175 poles are hammers; next goal / kill wins
    uint8_t      _pad_176[2];
    int32_t      sdLives[2];        // 0x178 [team] sudden-death kills needed-1 (set 2 when 4 players respawn); 0 in 1v1
    float        timeLeft;          // 0x180 match clock (s), runs only while timerOn; forced to 1.0 on a winning goal
    float        timeLeftShown;     // 0x184 clock snapshot shown on the win screen          // presentation
    bool         timerOn;           // 0x188 set on first ball/pole touch (EnableTimer)
    uint8_t      _pad_189[3];
    float        dt;                // 0x18c 1/60; divided by 1.5 per tick once matchDecided (slow-mo)
    float        endSlowmoTimer;    // 0x190 3 s after matchDecided, then endFade starts
    bool         matchDecided;      // 0x194 clock ran out with a winner: players deactivated
    uint8_t      _pad_195[3];
    DebugRender *debugRender;       // 0x198 NULL unless Assets::debug                     // presentation
    float        respawnTimer[4];   // 0x1a0 [player] >0: counting down, respawn at <=0 (kill: deathDelay 3 s; deleter: 1/60; ball reset: 0.5*dt)
    SpriteSheet *sprites;           // 0x1b0 sprites.ssht (shared with players/ball/dust)  // presentation
    SpriteSheet *messageSprites;    // 0x1b8 messages.ssht                                 // presentation
    SpriteSheet *numberSprites;     // 0x1c0 numbers.ssht                                  // presentation
    SpriteSheet *scoreArrowSprites; // 0x1c8 scorearrow.ssht                               // presentation
    SpriteSheet *shockwaveSprites;  // 0x1d0 shockwave.ssht                                // presentation
    SpriteSheet *additiveSprites;   // 0x1d8 additive.ssht                                 // presentation
    SpriteSheet *wireSprites;       // 0x1e0 wire.ssht                                     // presentation
    float        goalBanner;        // 0x1e8 -1 idle; 0 on goal, +dt/1.7 to 1 then -1      // presentation
    float        sdBanner;          // 0x1ec -1 idle; 0 on sudden death, +dt/0.3 to 1      // presentation
    float        timeUpBanner;      // 0x1f0 -1 idle; 0 when decided on time, +1/112 per tick // presentation
    float        winBanner;         // 0x1f4 -1 idle; 0 on winning goal / decision, +1/208 per tick; >=0 BLOCKS _ScoreGoal
    int32_t      bannerFrame;       // 0x1f8 Draw-only frame counter (mod 18)              // presentation
    bool         gameOver;          // 0x1fc GameOver(): set when endFade passes 1
    uint8_t      _pad_1fd[3];
    float        endFade;           // 0x200 -1 = playing; Reset()/rainbow unlock/slow-mo end set 0; +1/84 per tick; >0 blocks goals
    float        fadeIn;            // 0x204 1 -> 0 at 2/s                                 // presentation
    float        volumeRamp;        // 0x208 music/background volume ramp                  // presentation
    bool         fromRainbowUnlock; // 0x20c ctor arg (_unlockingRainbowHighway): skip fade-in  // presentation
    uint8_t      _pad_20d[3];
    int32_t      numPlayers;        // 0x210 count of active session slots (2 or 4 in practice)
    int32_t      level;             // 0x214 1..5 (Flatirons/Level2 = 2); selects LevelN/Background slots
    int32_t      hitstopFrames;     // 0x218 >0: dt forced to 0 for this many ticks (_PunchTime / _ShockwaveAt set 7)
    float        punchHeat;         // 0x21c +2 per punch, -dt per tick; hit-stop only if < 5 after adding
    bool         reverbOn;          // 0x220 ctor 1                                        // presentation
    bool         compressorOn;      // 0x221 ctor 1, passed to Sound::SetCompressor         // presentation
    uint8_t      _pad_222[6];

    // ---- methods ----
    // package A (src/game/playstate_ctor.cpp)
    explicit PlayState(bool fromRainbowUnlock);           // @0x100134190 (inlines _SetupWorld)
    ~PlayState();                                         // @0x10013b098
    void LoadComplete();                                  // @0x10013d574 (bridge calls it right after the ctor)
    // package C (src/game/playstate_update.cpp, src/game/contacts.cpp)
    void Update();                                        // @0x10013b8c8 (inlines _GoSuddenDeath)
    void BeginContact(b2Contact *c) override;             // @0x10013a93c
    void EndContact(b2Contact *c) override;               // @0x100136348
    void PreSolve(b2Contact *c, const b2Manifold *m) override;          // @0x10013544c (empty)
    void PostSolve(b2Contact *c, const b2ContactImpulse *imp) override; // @0x10013af72
    void BeginContactInner(b2Contact *c, int a, int b, int c2, int d);  // @0x10013a5a4
    void EndContactInner(int a, int b, int c, int d);     // @0x100136298
    void PostSolveInner(b2Contact *c, const b2ContactImpulse *imp, int a, int b, int c2, int d);  // @0x10013ad2e
    void PlayerCollision(b2Contact *c, int a, int b);     // @0x100139592
    void HammerPlayerCollision(b2Contact *c, const b2ContactImpulse *imp, int a, int b);  // @0x100139d6e
    void HammerBallCollision(b2Contact *c, const b2ContactImpulse *imp, int a);           // @0x10013a1c8
    void BallWallCollision();                             // @0x10013a4f2
    void BallPlayerCollision(const b2ContactImpulse *imp, int a);                          // @0x10013a9ec
    void _ScoreGoal(int team);                            // @0x1001389a4
    void _ShockwaveAt(int a, b2Vec2 p, b2Vec2 v);         // @0x10013942c (tail = _PunchTime: port it)
    void _MakeDeleters();                                 // @0x10013560a
    void GroundContact(b2Contact *c, const b2ContactImpulse *imp, int a, int b);  // @0x100138d00 (PRES; rumble part ported)
};
static_assert(offsetof(PlayState, levelWidth) == 0x008, "");
static_assert(offsetof(PlayState, targetWidth) == 0x00c, "");
static_assert(offsetof(PlayState, ballHeight) == 0x010, "");
static_assert(offsetof(PlayState, spawnHeight) == 0x014, "");
static_assert(offsetof(PlayState, spawnBoost) == 0x018, "");
static_assert(offsetof(PlayState, world) == 0x020, "");
static_assert(offsetof(PlayState, score) == 0x028, "");
static_assert(offsetof(PlayState, lastTouch) == 0x030, "");
static_assert(offsetof(PlayState, koWin) == 0x034, "");
static_assert(offsetof(PlayState, unlockingRainbow) == 0x035, "");
static_assert(offsetof(PlayState, timeOver) == 0x036, "");
static_assert(offsetof(PlayState, scoreArrowAnim) == 0x038, "");
static_assert(offsetof(PlayState, gullAnim) == 0x048, "");
static_assert(offsetof(PlayState, players) == 0x050, "");
static_assert(offsetof(PlayState, camera) == 0x070, "");
static_assert(offsetof(PlayState, ball) == 0x078, "");
static_assert(offsetof(PlayState, targets) == 0x080, "");
static_assert(offsetof(PlayState, shockwaves) == 0x090, "");
static_assert(offsetof(PlayState, dust) == 0x0b0, "");
static_assert(offsetof(PlayState, sparks) == 0x0b8, "");
static_assert(offsetof(PlayState, flare) == 0x0c0, "");
static_assert(offsetof(PlayState, sndFanfare) == 0x0d0, "");
static_assert(offsetof(PlayState, sndGoal) == 0x0d8, "");
static_assert(offsetof(PlayState, sndGoldenGoal) == 0x0f8, "");
static_assert(offsetof(PlayState, sndFart) == 0x100, "");
static_assert(offsetof(PlayState, sndSecret) == 0x108, "");
static_assert(offsetof(PlayState, ballResetPending) == 0x110, "");
static_assert(offsetof(PlayState, ballResetDelay) == 0x114, "");
static_assert(offsetof(PlayState, lastGoalTeam0) == 0x118, "");
static_assert(offsetof(PlayState, level1) == 0x120, "");
static_assert(offsetof(PlayState, level2) == 0x128, "");
static_assert(offsetof(PlayState, level3) == 0x130, "");
static_assert(offsetof(PlayState, level4) == 0x138, "");
static_assert(offsetof(PlayState, level5) == 0x140, "");
static_assert(offsetof(PlayState, background1) == 0x148, "");
static_assert(offsetof(PlayState, background2) == 0x150, "");
static_assert(offsetof(PlayState, background3) == 0x158, "");
static_assert(offsetof(PlayState, background4) == 0x160, "");
static_assert(offsetof(PlayState, background5) == 0x168, "");
static_assert(offsetof(PlayState, killMask) == 0x170, "");
static_assert(offsetof(PlayState, ballKickAvailable) == 0x174, "");
static_assert(offsetof(PlayState, suddenDeath) == 0x175, "");
static_assert(offsetof(PlayState, sdLives) == 0x178, "");
static_assert(offsetof(PlayState, timeLeft) == 0x180, "");
static_assert(offsetof(PlayState, timeLeftShown) == 0x184, "");
static_assert(offsetof(PlayState, timerOn) == 0x188, "");
static_assert(offsetof(PlayState, dt) == 0x18c, "");
static_assert(offsetof(PlayState, endSlowmoTimer) == 0x190, "");
static_assert(offsetof(PlayState, matchDecided) == 0x194, "");
static_assert(offsetof(PlayState, debugRender) == 0x198, "");
static_assert(offsetof(PlayState, respawnTimer) == 0x1a0, "");
static_assert(offsetof(PlayState, sprites) == 0x1b0, "");
static_assert(offsetof(PlayState, messageSprites) == 0x1b8, "");
static_assert(offsetof(PlayState, numberSprites) == 0x1c0, "");
static_assert(offsetof(PlayState, scoreArrowSprites) == 0x1c8, "");
static_assert(offsetof(PlayState, shockwaveSprites) == 0x1d0, "");
static_assert(offsetof(PlayState, additiveSprites) == 0x1d8, "");
static_assert(offsetof(PlayState, wireSprites) == 0x1e0, "");
static_assert(offsetof(PlayState, goalBanner) == 0x1e8, "");
static_assert(offsetof(PlayState, sdBanner) == 0x1ec, "");
static_assert(offsetof(PlayState, timeUpBanner) == 0x1f0, "");
static_assert(offsetof(PlayState, winBanner) == 0x1f4, "");
static_assert(offsetof(PlayState, bannerFrame) == 0x1f8, "");
static_assert(offsetof(PlayState, gameOver) == 0x1fc, "");
static_assert(offsetof(PlayState, endFade) == 0x200, "");
static_assert(offsetof(PlayState, fadeIn) == 0x204, "");
static_assert(offsetof(PlayState, volumeRamp) == 0x208, "");
static_assert(offsetof(PlayState, fromRainbowUnlock) == 0x20c, "");
static_assert(offsetof(PlayState, numPlayers) == 0x210, "");
static_assert(offsetof(PlayState, level) == 0x214, "");
static_assert(offsetof(PlayState, hitstopFrames) == 0x218, "");
static_assert(offsetof(PlayState, punchHeat) == 0x21c, "");
static_assert(offsetof(PlayState, reverbOn) == 0x220, "");
static_assert(offsetof(PlayState, compressorOn) == 0x221, "");
static_assert(sizeof(PlayState) == 0x228, "");

// =====================================================================================
// Box2D userData tags: PoleRiders::Assets::BODY_* / FIXTURE_* are `static int` in __DATA
// (0x10028e06c..0x10028e0b0) holding their own ordinal; userData stores a POINTER to such
// an int (or to Player::index / Pole::ownerIndex). Contact code reads *(int*)userData,
// NULL -> 0 (which aliases BODY_PLAYER1!).
// =====================================================================================
enum Tag : int32_t {
    BODY_PLAYER1 = 0, BODY_PLAYER2 = 1, BODY_PLAYER3 = 2, BODY_PLAYER4 = 3,  // = player index
    BODY_GROUND = 4, BODY_WALL = 5, BODY_POLE = 6, BODY_BALL = 7, BODY_ANCHOR = 8,
    BODY_NONE = 9, BODY_DELETER = 10, BODY_HAMMER = 11, BODY_SEA = 12,
    FIXTURE_SENSOR = 13, FIXTURE_NONE = 14, FIXTURE_HAMMERBODY1 = 15, FIXTURE_HAMMERBODY2 = 16,
    FIXTURE_IMPALE = 17,
};
constexpr int32_t HANDLE_START = 2;  // Assets::HANDLE_START
constexpr int32_t POLE_SEGMENTS = 10;

// Box2D offsets the game code touches directly (2.1.2, LP64) -- sanity checks for the port.
static_assert(sizeof(b2World) == 0x19368, "b2World size (operator new(0x19368))");

}  // namespace PoleRiders
