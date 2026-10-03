// sprbridge — injected into Sportsfriends (x86_64, under Rosetta) via DYLD_INSERT_LIBRARIES.
//
// 1. Game state: hooks PoleRiders::Update() and after every Pole Riders frame snapshots the
//    match (scores + every Box2D body, tagged via its userData).
// 2. Virtual gamepads: hooks atrophy::Input_Update() (the engine's once-per-frame input pump)
//    and drives fake controllers in the engine's own gamepad slots, so menus, team select and
//    gameplay see them exactly like real pads. All engine calls happen on the game thread.
//
// A background thread serves a Unix socket, one command per line, one reply line each:
//
//   ping                       -> pong
//   state                      -> latest match snapshot (JSON)
//   record start <file.mp4>    -> every rendered frame from now on goes to ffmpeg (60 fps, H.264); use it in
//                                 sync mode so each simulated frame is rendered (and recorded) exactly once
//   record stop                -> finish the video
//   playlog on <dir> | off      -> log human play (sticks every frame + state every 4th) per match (see playlog_frame)
//   pad mirror <i> <src>|off   -> virtual pad i copies real controller src's sticks (a human plays that player)
//   pad override <i> on|off    -> real controller i is driven by set/tap/clear instead of its own input
//   hud <text> | hud off       -> a line of text over the bottom of the screen (names of who plays whom)
//   pad keyboard <i> wasd|arrows|off -> virtual pad i is played on the keyboard (see apply_keyboard)
//   settings                   -> score limit / duration the game loaded (JSON)
//   settings set <score> <min> -> set them (save data, read by PoleRiders::Begin at the next match)
//   frame                      -> {"frame":N}  (input frames since launch)
//   pads                       -> which slots are connected, and which are virtual
//   pad connect <i>            -> plug a virtual pad into slot i (0-3)
//   pad disconnect <i>
//   pad set <i> k=v ...        -> hold state until changed. Keys:
//        buttons (0/1): a b x y back guide start ls rs lb rb up down left right
//        analog: lt rt (0..1), lx ly rx ry (-1..1; SDL convention, +y = down)
//   pad tap <i> <button> [frames]  -> press for N frames (default 4), then release
//   pad clear <i>              -> release everything, center sticks
//   sync on                    -> lockstep: the game freezes before each frame until stepped;
//                                 every stepped frame runs exactly one 1/60 s physics tick
//   sync off                   -> back to real time
//   step [n]                   -> (sync mode) run n frames (default 1), reply with the state
//   act n lx0 rx0 ry0 lx1 rx1 ry1 -> set pad 0/1 sticks and step n frames in one round trip
//   ball x y [vx vy]           -> move the ball (and its zipline trolley) so the ball is at x,y
//                                 (Box2D meters, +y down) with velocity vx,vy; for tests and
//                                 curriculum starts. Applied before the next tick
//   mute on|off                -> silence all game audio (also: env SPRBRIDGE_MUTE=1 at launch)
//   fast on [n] | fast off     -> vsync off + draw only every n-th frame (default 10) for
//                                 faster-than-realtime training; `fast off` restores both
//   reset [seed]               -> restart the current match with the same teams (the game's own
//                                 pause-menu restart path); watch match_id to see it take effect.
//                                 With a seed, libc random() is reseeded for reproducible episodes
//
// Addresses are for the 2015-09-17 macOS build (sha256 e24a10b2…): unslid VM addresses from
// `nm`, slid at load, and every hooked prologue is byte-checked before patching.
#include <CoreGraphics/CoreGraphics.h>
#include <CoreText/CoreText.h>
#include <OpenGL/OpenGL.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <dispatch/dispatch.h>
#include <OpenGL/gl.h>
#include <errno.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <math.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define A_POLERIDERS_UPDATE 0x10013f070ULL  // PoleRiders::Update()
#define A_INPUT_UPDATE      0x10004eaa0ULL  // atrophy::Input_Update()
#define A_GAMEPAD_LIST      0x10064c980ULL  // atrophy::sGamepadList[4], GamepadSDL (0x1b8 each)
#define A_GP_CONNECT        0x1000111e6ULL  // atrophy::Gamepad::connect(int, GamepadType)
#define A_GP_DISCONNECT     0x1000114eaULL  // atrophy::Gamepad::disconnect()
#define A_GP_SETBUTTON      0x100011376ULL  // atrophy::Gamepad::setButtonState(int, uint8)
#define A_PLAYSTATE_PTR     0x100c57ee0ULL  // (anonymous)::_playState  (PlayState*)
#define A_SAVEDATA          0x100c5c300ULL  // Common::Save::_saveData
#define A_BODY_TAGS         0x10028e06cULL  // PoleRiders::Assets::BODY_PLAYER1 .. BODY_SEA
#define A_MAXSCORE          0x10062fd88ULL  // PoleRiders::Assets::maxScore (int)
#define A_ROUNDTIME         0x10062fd8cULL  // PoleRiders::Assets::roundTime (float, seconds)
#define A_RESETTING         0x100c57f51ULL  // (anonymous)::_resetting (bool)
#define A_NEXT_IS_VICTORY   0x100c57f40ULL  // (anonymous)::_nextIsVictoryScene (bool)
#define A_B2_SETTRANSFORM   0x10016eaf0ULL  // b2Body::SetTransform(b2Vec2 const&, float)
#define A_B2_SETAWAKE       0x10015f1a0ULL  // b2Body::SetAwake(bool)
#define A_SND_SETMASTER     0x100067b50ULL  // Common::Sound::SetMasterVolume(float)
#define A_PS_CTOR           0x10013d64cULL  // PoleRiders::PlayState::PlayState(bool) (C1)
#define A_PS_LOADCOMPLETE   0x10013d574ULL  // PoleRiders::PlayState::LoadComplete()
#define A_PS_UPDATE         0x10013b8c8ULL  // PoleRiders::PlayState::Update()  (one 1/60 s tick)
#define A_PS_DTOR_DELETE    0x10013b098ULL  // PoleRiders::PlayState::~PlayState() deleting (D0)
#define PS_SIZE             0x228
#define PL_SESSION_PAD      0x228    // int: Player.session (+0x220) .pad = gamepad index driving it
#define PL_SESSION_HEAD     0x22c    // int: Player.session .head = character (7 = the hidden short-pole one)
#define A_B2_FIXTURELIST    0x100175b80ULL  // b2Body::GetFixtureList()
#define A_B2_FIX_NEXT       0x100175ba0ULL  // b2Fixture::GetNext()
#define A_B2_FIX_SHAPE      0x1001689e0ULL  // b2Fixture::GetShape()
#define A_B2_FIX_SENSOR     0x1001695f0ULL  // b2Fixture::IsSensor() const
#define A_B2_SHAPE_TYPE     0x100168aa0ULL  // b2Shape::GetType() const  (1 = polygon here)
#define POLY_VERTS          0x18     // b2PolygonShape::m_vertices[8] (body-local)
#define POLY_COUNT          0xa8     // b2PolygonShape::m_count
#define A_MOVE_INIT         0x100061990ULL  // Common::Move::Init(bool)   (PS Move / `moved` daemon)
#define A_MOVE_UPDATE       0x100061b10ULL  // Common::Move::Update(float)
#define A_LEVEL5_UNLOCKED   0x100d0f780ULL  // PoleRiders::Assets::level5unlocked (bool)
#define A_PR_ELAPSED        0x100c57f54ULL  // PoleRiders (anonymous)::sElapsed, fixed-step accumulator
#define A_SHOULD_REDRAW     0x100c57ee9ULL  // PoleRiders (anonymous)::_shouldRedraw (bool)
#define A_DT_LAST_UPDATE    0x10064eb50ULL  // Common::DT (anonymous)::dtLastUpdateAt (timer ticks)
#define A_TIME_HIRES        0x10004c9a0ULL  // atrophy::Time_GetHighResTimer()
#define A_TIME_MICROS       0x10004c980ULL  // atrophy::Time_GetMicroseconds(ticks)

// Common::Save::_saveData
#define SAVE_PR_SCORELIMIT  0x74b34  // float
#define SAVE_PR_DURATION    0x74b38  // float, minutes
// PoleRiders::PlayState
#define PS_WORLD            0x20     // b2World*
#define PS_SCORE_A          0x28     // int, team 0 (Vaulters)
#define PS_SCORE_B          0x2c     // int, team 1 (Benders)
#define PS_PLAYERS          0x50     // Player*[4]
#define PS_BALL             0x78     // Ball*  (Ball+0 = b2Body*)
#define PS_TIME_LEFT        0x180    // float, seconds
#define PS_GAME_OVER        0x1fc    // bool (PlayState::GameOver)
#define PS_LEVEL            0x214    // int, which level this PlayState built (5 = Rainbow Highway)
#define PS_SUDDEN_DEATH     0x175    // bool: clock ran out level; scores reset, poles become hammers
#define PS_TIMER_ON         0x188    // bool: the match clock runs (set on the first touch)
#define POLE_IS_HAMMER      0x00     // bool on the Pole (Pole::Hammer() returns the head body)
#define PS_NUM_PLAYERS      0x210    // int
#define PS_RESPAWN_TIMER    0x1a0    // float[4] per player index: >0 = dead, seconds until respawn
                                     // (kill: deathDelay 3 s; deleter 1/60; ball reset 0.5*dt)
// PoleRiders::Player / Pole
#define PL_STATS            0x00     // Stats*: kicks +4, jumps +8, suicides +0xc, kills +0x14,
                                     //         deaths +0x18, goals +0x20, own goals +0x24
#define PL_POLE             0x18     // Pole*
#define PL_ON_POLE          0x28     // bool
#define PL_ON_GROUND        0x29     // bool
#define PL_BODY             0x30     // b2Body*
#define PL_GHOST            0x1fd    // bool (respawning / knocked out)
#define PL_GHOST_TIMER      0x200    // float: seconds until the ghost respawns
#define POLE_SEGMENTS       0x08     // b2Body*[10]
#define POLE_NSEG           10
// Box2D
#define W_BODYLIST          0x19320  // b2World::m_bodyList
#define B_TYPE              0x00     // b2BodyType
#define B_POS               0x0c     // b2Vec2 (m_xf.p)
#define B_ANGLE             0x40     // m_sweep.a
#define B_LINVEL            0x44     // b2Vec2
#define B_ANGVEL            0x4c     // float
#define B_NEXT              0x70     // b2Body*
#define B_USERDATA          0xb8     // void* -> &PoleRiders::Assets::BODY_*
// atrophy::GamepadSDL (layout from GamepadSDL::connect / Input_Update)
#define GP_STRIDE           0x1b8
#define GP_CONNECTED        0x08     // bool
#define GP_DEADZONE0        0x1c     // float, one per stick (stride 0x30)
#define GP_NUMSTICKS        0xcc     // int (2)
#define GP_NUMBUTTONS       0x170    // int (17)
#define GP_SDLCONTROLLER    0x188    // SDL_GameController*
#define GP_INSTANCEID       0x190    // SDL joystick instance id
#define GP_HAPTIC           0x198    // SDL_Haptic*
#define GP_AXES             0x1a4    // float[4]: SDL LX LY RX RY in -1..1, applied every frame
#define GAMEPAD_TYPE_SDL    1

static const char *kTagNames[] = {  // in address order starting at BODY_PLAYER1
    "player1", "player2", "player3", "player4", "ground", "wall", "pole",
    "ball", "anchor", "none", "deleter", "hammer", "sea"};

// SDL GameController button -> atrophy button index (atrophy's _kButtonMap), plus triggers.
enum { NBTN = 17 };
static const char *kBtnNames[NBTN] = {"a",    "b",     "x",    "y",  "back", "guide",
                                      "start", "ls",   "rs",   "lb", "rb",   "up",
                                      "down", "left",  "right", "lt", "rt"};
static const int kBtnAtrophy[NBTN] = {0, 1, 2, 3, 9, 10, 8, 6, 7, 4, 5, 11, 12, 13, 14, 15, 16};

typedef struct {
  int want_connected, is_virtual;
  float buttons[NBTN];  // 0..1
  int tap_frames[NBTN]; // >0: held for that many more frames
  float axes[4];
  int mirror;           // 1 + the real controller slot whose sticks this pad copies every frame (0 = none)
  int kbd;              // keyboard-driven: 1 = WASD, 2 = arrows (see apply_keyboard), 0 = off
  int kbd_aimed;        // the pole aim angle has been set (from the rider's pole at the first rotate key)
  float kbd_angle;      // pole aim, radians counter-clockwise from +x (screen), held when no key is down
  int override;         // a REAL controller's inputs replaced by this pad's (scripted menus on a human's own pad)
  int release;          // override just ended: release its buttons (on the game thread)
  int detached;         // override: the slot's SDL controller is unhooked (so its own stick events can't leak in)
  void *saved_ctrl;     // ... and restored from these when the override ends
  int saved_iid;
} VPad;

static intptr_t g_slide;
static int (*g_orig_pr_update)(void);
static void (*g_orig_input_update)(void);
static void (*gp_connect)(void *, int, int);
static void (*gp_disconnect)(void *);
static void (*gp_set_button)(void *, int, unsigned char);

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_snap[256 * 1024];
static unsigned long g_pr_frame, g_input_frame;
static VPad g_pads[4];

// Lockstep. The game thread parks at the top of each input frame while g_budget == 0.
static pthread_cond_t g_step_cv = PTHREAD_COND_INITIALIZER, g_idle_cv = PTHREAD_COND_INITIALIZER;
static int g_sync, g_idle, g_reset_pending;
static long g_sync_owner = -1, g_next_conn = 1;  // sync owner = connection id (fds get reused)
static __thread long t_conn;
static char *g_last_ps;
static int g_render_every = 1, g_vsync_pending = -1;  // -1: nothing to apply
static int g_mute, g_unmute_pending, g_no_unlock;
static void (*snd_set_master)(float);
static void (*ps_ctor)(void *, int), (*ps_load_complete)(void *), (*ps_update)(void *),
    (*ps_delete)(void *);
static void *(*b2_fixture_list)(void *), *(*b2_fix_next)(void *), *(*b2_fix_shape)(void *);
static int (*b2_fix_sensor)(void *), (*b2_shape_type)(void *);
static int g_level_req;               // pending `level` dump request
static char *g_level_out;
static pthread_cond_t g_farmtest_cv = PTHREAD_COND_INITIALIZER;  // game-thread job completion
static int farm_job_pending(void);  // farm.c (headless shadow matches)
static int farm_active(void);
static void farm_run_job(void);
static void farm_command(int fd, char *args);
static void farm_init(void);

static unsigned long g_match_id;
static long g_budget;
// Slightly over one 1/60 s tick: with the accumulator zeroed each frame this is always
// exactly one physics tick, whatever wall-clock jitter adds.
static float g_sync_dt = 1.0f / 60 + 0.0002f;
static long g_reset_seed = -1;
static int g_ball_pending;
static float g_ball_set[4];  // x y vx vy
static void (*b2_set_transform)(void *, const float *, float);
static void (*b2_set_awake)(void *, int);
static uint64_t (*time_hires)(void);
static long (*time_micros)(uint64_t);
static double g_ticks_per_us;

#define P(addr) ((uintptr_t)(addr) + g_slide)
#define SLOT(i) ((char *)P(A_GAMEPAD_LIST) + (i) * GP_STRIDE)

// ---------------------------------------------------------------- state snapshot

static const char *tag_name(void *ud) {
  for (unsigned i = 0; i < sizeof kTagNames / sizeof *kTagNames; i++)
    if ((uintptr_t)ud == P(A_BODY_TAGS + 4 * i)) return kTagNames[i];
  return NULL;
}

static size_t put_body(char *buf, size_t n, size_t cap, const char *pre, char *b) {
  float *p = (float *)(b + B_POS), *v = (float *)(b + B_LINVEL);
  return n + snprintf(buf + n, cap - n,
                      "%s\"x\":%.4f,\"y\":%.4f,\"a\":%.4f,\"vx\":%.4f,\"vy\":%.4f,\"w\":%.4f}",
                      pre, p[0], p[1], *(float *)(b + B_ANGLE), v[0], v[1],
                      *(float *)(b + B_ANGVEL));
}

// Label a body by what owns it: a player's torso, a player's pole, or its BODY_* userData tag.
static const char *label(char *ps, char *b, char *tmp) {
  int np = *(int *)(ps + PS_NUM_PLAYERS);
  for (int i = 0; i < np && i < 4; i++) {
    char *pl = *(char **)(ps + PS_PLAYERS + 8 * i);
    if (!pl) continue;
    if (*(char **)(pl + PL_BODY) == b) return sprintf(tmp, "player%d", i + 1), tmp;
    char *pole = *(char **)(pl + PL_POLE);
    for (int k = 0; pole && k < POLE_NSEG; k++)
      if (*(char **)(pole + POLE_SEGMENTS + 8 * k) == b) return sprintf(tmp, "pole%d", i + 1), tmp;
  }
  const char *tag = tag_name(*(void **)(b + B_USERDATA));
  return tag ? tag : "other";
}

// JSON snapshot of one PlayState (the real match or a farm shadow). with_bodies adds every Box2D
// body; without it, only the zipline trolley is added (enough to draw the match).
static size_t snapshot_ps(char *ps, char *buf, size_t cap, unsigned long frame,
                          unsigned long match_id, int with_bodies) {
  size_t n;
  char tmp[16];
  if (!ps) return snprintf(buf, cap, "{\"frame\":%lu,\"match_id\":%lu,\"in_match\":false}", frame,
                           match_id);
  n = snprintf(buf, cap,
               "{\"frame\":%lu,\"match_id\":%lu,\"in_match\":true,\"level\":%d,\"game_over\":%s,"
               "\"sudden_death\":%s,\"score\":[%d,%d],\"time_left\":%.3f,\"players\":[",
               frame, match_id, *(int *)(ps + PS_LEVEL), ps[PS_GAME_OVER] ? "true" : "false",
               ps[PS_SUDDEN_DEATH] ? "true" : "false",
               *(int *)(ps + PS_SCORE_A), *(int *)(ps + PS_SCORE_B), *(float *)(ps + PS_TIME_LEFT));
  int np = *(int *)(ps + PS_NUM_PLAYERS);
  for (int i = 0; i < np && i < 4; i++) {
    char *pl = *(char **)(ps + PS_PLAYERS + 8 * i), *body = pl ? *(char **)(pl + PL_BODY) : 0;
    if (!body) continue;
    char pre[224];
    char *polep = *(char **)(pl + PL_POLE);
    // dead: killed (or fell into a deleter) and waiting to respawn; the corpse body stays in the
    // world meanwhile. ghost is only the short post-spawn ghost.
    float rt = ((float *)(ps + PS_RESPAWN_TIMER))[i];
    int dead = rt > 0.0f;
    snprintf(pre, sizeof pre,
             "%s{\"id\":%d,\"on_pole\":%s,\"on_ground\":%s,\"ghost\":%s,\"ghost_t\":%.3f,"
             "\"dead\":%s,\"respawn_t\":%.3f,\"hammer\":%s,\"pad\":%d,\"head\":%d,",
             i ? "," : "", i + 1, pl[PL_ON_POLE] ? "true" : "false",
             pl[PL_ON_GROUND] ? "true" : "false", pl[PL_GHOST] ? "true" : "false",
             pl[PL_GHOST] ? *(float *)(pl + PL_GHOST_TIMER) : 0.0f, dead ? "true" : "false",
             dead ? rt : 0.0f, polep && polep[POLE_IS_HAMMER] ? "true" : "false",
             *(int *)(pl + PL_SESSION_PAD), *(int *)(pl + PL_SESSION_HEAD));
    n = put_body(buf, n, cap, pre, body) - 1;  // reopen the object to append stats + pole
    int *st = *(int **)(pl + PL_STATS);
    if (st)
      n += snprintf(buf + n, cap - n,
                    ",\"stats\":{\"kicks\":%d,\"jumps\":%d,\"suicides\":%d,\"kills\":%d,"
                    "\"deaths\":%d,\"goals\":%d,\"own_goals\":%d}",
                    st[1], st[2], st[3], st[5], st[6], st[8], st[9]);
    n += snprintf(buf + n, cap - n, ",\"pole\":[");
    char *pole = *(char **)(pl + PL_POLE);
    for (int k = 0; pole && k < POLE_NSEG; k++) {
      char *seg = *(char **)(pole + POLE_SEGMENTS + 8 * k);
      if (seg) n = put_body(buf, n, cap, k ? ",{" : "{", seg);
    }
    n += snprintf(buf + n, cap - n, "]}");
  }
  n += snprintf(buf + n, cap - n, "]");
  char *ball = *(char **)(ps + PS_BALL), *bb = ball ? *(char **)ball : NULL;
  if (bb) n = put_body(buf, n, cap, ",\"ball\":{", bb);
  char *world = *(char **)(ps + PS_WORLD);
  if (with_bodies) {
    n += snprintf(buf + n, cap - n, ",\"bodies\":[");
    int first = 1;
    for (char *b = world ? *(char **)(world + W_BODYLIST) : NULL; b && n < cap - 512;
         b = *(char **)(b + B_NEXT)) {
      char pre[64];
      snprintf(pre, sizeof pre, "%s{\"tag\":\"%s\",\"type\":%d,", first ? "" : ",",
               label(ps, b, tmp), *(int *)(b + B_TYPE));
      n = put_body(buf, n, cap, pre, b);
      first = 0;
    }
    n += snprintf(buf + n, cap - n, "]");
  } else {  // just the zipline trolley: the other BODY_BALL-tagged body
    for (char *b = world ? *(char **)(world + W_BODYLIST) : NULL; b; b = *(char **)(b + B_NEXT))
      if (b != bb && *(void **)(b + B_USERDATA) == (void *)P(A_BODY_TAGS + 0x1c)) {
        n += snprintf(buf + n, cap - n, ",\"trolley\":[%.4f,%.4f]", ((float *)(b + B_POS))[0],
                      ((float *)(b + B_POS))[1]);
        break;
      }
  }
  n += snprintf(buf + n, cap - n, "}");
  return n;
}

// ---------------------------------------------------------------- play log (human demonstrations)
// While on (`playlog on <dir>` or SPRBRIDGE_PLAYLOG=<dir>) and a human plays (a real controller is connected or a
// virtual pad mirrors one), every frame of a real match appends one JSON line to <dir>/<time>_m<match>.jsonl:
// {"f": frame, "ax": [[lx, ly, rx, ry] x 4 slots], "kind": [0 none / 1 real / 2 virtual / 3 virtual mirroring a
// human / 4 virtual on the keyboard] x 4} and, every 4th frame (the bots' decision rate), "s": the match snapshot (players incl. pad + head).
static char g_playlog_dir[400];
static FILE *g_playlog;
static unsigned long g_playlog_match;

static void playlog_frame(char *ps) {
  if (!g_playlog_dir[0] || !ps) return;
  int kind[4], human = 0;
  pthread_mutex_lock(&g_lock);
  for (int i = 0; i < 4; i++) {
    int conn = SLOT(i)[GP_CONNECTED];
    kind[i] = !conn ? 0 : !g_pads[i].is_virtual ? 1 : g_pads[i].mirror ? 3 : g_pads[i].kbd ? 4 : 2;
    human |= kind[i] == 1 || kind[i] == 3 || kind[i] == 4;
  }
  pthread_mutex_unlock(&g_lock);
  if (!human) return;
  if (!g_playlog || g_playlog_match != g_match_id) {
    if (g_playlog) fclose(g_playlog);
    char path[512], ts[32];
    time_t t = time(NULL);
    strftime(ts, sizeof ts, "%Y%m%d-%H%M%S", localtime(&t));
    snprintf(path, sizeof path, "%s/%s_m%lu.jsonl", g_playlog_dir, ts, g_match_id);
    g_playlog = fopen(path, "w");
    g_playlog_match = g_match_id;
    if (!g_playlog) return;
  }
  fprintf(g_playlog, "{\"f\":%lu,\"kind\":[%d,%d,%d,%d],\"ax\":[", g_pr_frame, kind[0], kind[1], kind[2], kind[3]);
  for (int i = 0; i < 4; i++) {
    float *a = (float *)(SLOT(i) + GP_AXES);
    fprintf(g_playlog, "%s[%.4f,%.4f,%.4f,%.4f]", i ? "," : "", a[0], a[1], a[2], a[3]);
  }
  fputc(']', g_playlog);
  if (g_pr_frame % 4 == 0) {
    static char buf[64 * 1024];
    snapshot_ps(ps, buf, sizeof buf, g_pr_frame, g_match_id, 0);
    fprintf(g_playlog, ",\"s\":%s", buf);
  }
  fputs("}\n", g_playlog);
  if (g_pr_frame % 60 == 0) fflush(g_playlog);
}

static void snapshot(void) {
  static char buf[sizeof g_snap];
  char *ps = *(char **)P(A_PLAYSTATE_PTR);
  g_pr_frame++;
  if (ps && ps != g_last_ps) g_match_id++;
  g_last_ps = ps;
  size_t n = snapshot_ps(ps, buf, sizeof buf, g_pr_frame, g_match_id, 1);
  pthread_mutex_lock(&g_lock);
  memcpy(g_snap, buf, n + 1);
  pthread_mutex_unlock(&g_lock);
  playlog_frame(ps);
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Static arena geometry in world meters (+y down): every static body's polygon fixtures.
static void dump_level(void) {
  size_t cap = 256 * 1024, n = 0;
  char *out = malloc(cap), tmp[16];
  char *ps = *(char **)P(A_PLAYSTATE_PTR), *world = ps ? *(char **)(ps + PS_WORLD) : NULL;
  n += snprintf(out + n, cap - n, "{\"bodies\":[");
  int firstb = 1;
  for (char *b = world ? *(char **)(world + W_BODYLIST) : NULL; b && n < cap - 4096;
       b = *(char **)(b + B_NEXT)) {
    if (*(int *)(b + B_TYPE) != 0) continue;  // static only
    float px = ((float *)(b + B_POS))[0], py = ((float *)(b + B_POS))[1], a = *(float *)(b + B_ANGLE);
    float c = cosf(a), sn = sinf(a);
    n += snprintf(out + n, cap - n, "%s{\"tag\":\"%s\",\"polys\":[", firstb ? "" : ",",
                  label(ps, b, tmp));
    firstb = 0;
    int firstf = 1;
    for (char *f = b2_fixture_list(b); f && n < cap - 1024; f = b2_fix_next(f)) {
      char *sh = b2_fix_shape(f);
      if (!sh || b2_shape_type(sh) != 1) continue;  // this Box2D: polygon = 1
      int cnt = *(int *)(sh + POLY_COUNT);
      if (cnt < 3 || cnt > 8) continue;
      n += snprintf(out + n, cap - n, "%s{\"sensor\":%s,\"v\":[", firstf ? "" : ",",
                    b2_fix_sensor(f) ? "true" : "false");
      firstf = 0;
      for (int k = 0; k < cnt; k++) {
        float vx = ((float *)(sh + POLY_VERTS))[2 * k], vy = ((float *)(sh + POLY_VERTS))[2 * k + 1];
        n += snprintf(out + n, cap - n, "%s[%.4f,%.4f]", k ? "," : "", px + c * vx - sn * vy,
                      py + sn * vx + c * vy);
      }
      n += snprintf(out + n, cap - n, "]}");
    }
    n += snprintf(out + n, cap - n, "]}");
  }
  snprintf(out + n, cap - n, "]}");
  g_level_out = out;
}

static int hooked_pr_update(void) {
  if (g_level_req && *(char **)P(A_PLAYSTATE_PTR)) {
    dump_level();
    pthread_mutex_lock(&g_lock);
    g_level_req = 0;
    pthread_cond_broadcast(&g_farmtest_cv);
    pthread_mutex_unlock(&g_lock);
  }
  if (__atomic_exchange_n(&g_reset_pending, 0, __ATOMIC_ACQ_REL)) {
    *(char *)P(A_RESETTING) = 1;
    *(char *)P(A_NEXT_IS_VICTORY) = 0;
    long seed = __atomic_exchange_n(&g_reset_seed, -1, __ATOMIC_ACQ_REL);
    if (seed >= 0) srandom((unsigned)seed);
  }
  // Jumping out over the terrain on the enemy side unlocks Rainbow Highway: the match ends,
  // the save file is written and the next match switches maps. Claim it's already unlocked.
  if (g_no_unlock) *(char *)P(A_LEVEL5_UNLOCKED) = 1;
  if (g_sync) *(float *)P(A_PR_ELAPSED) = 0;  // exactly one fixed tick this frame
  if (__atomic_exchange_n(&g_ball_pending, 0, __ATOMIC_ACQ_REL)) {
    // The ball hangs from a trolley on the zipline; both bodies are tagged BODY_BALL, so move
    // them together (same offset, same velocity) or the joint yanks the ball back.
    char *ps = *(char **)P(A_PLAYSTATE_PTR), *ball = ps ? *(char **)(ps + PS_BALL) : NULL;
    char *bb = ball ? *(char **)ball : NULL, *world = ps ? *(char **)(ps + PS_WORLD) : NULL;
    if (bb && world) {
      float dx = g_ball_set[0] - ((float *)(bb + B_POS))[0];
      float dy = g_ball_set[1] - ((float *)(bb + B_POS))[1];
      for (char *b = *(char **)(world + W_BODYLIST); b; b = *(char **)(b + B_NEXT)) {
        if (*(void **)(b + B_USERDATA) != (void *)P(A_BODY_TAGS + 0x1c)) continue;  // BODY_BALL
        float pos[2] = {((float *)(b + B_POS))[0] + dx, ((float *)(b + B_POS))[1] + dy};
        b2_set_transform(b, pos, *(float *)(b + B_ANGLE));
        ((float *)(b + B_LINVEL))[0] = g_ball_set[2];
        ((float *)(b + B_LINVEL))[1] = g_ball_set[3];
        *(float *)(b + B_ANGVEL) = 0;
        b2_set_awake(b, 1);
      }
    }
  }
  int r = g_orig_pr_update();
  if (g_render_every > 1 && g_pr_frame % g_render_every) *(char *)P(A_SHOULD_REDRAW) = 0;
  snapshot();
  return r;
}

// ---------------------------------------------------------------- virtual pads

// Runs on the game thread, once per frame, right before SDL events are pumped.
// ---------------------------------------------------------------- joysticks (debug)
// The game's statically linked SDL 2.0.x: a pad is only usable when SDL_IsGameController (it has a mapping, built
// in or from SDL_GAMECONTROLLERCONFIG; see prmod.py / spr_env.py PAD_MAPPINGS).
#define A_SDL_NUMJOY      0x1001be040ULL
#define A_SDL_JOYNAME     0x1001be050ULL
#define A_SDL_JOYGUID     0x1001be080ULL
#define A_SDL_GUIDSTR     0x1001be0a0ULL
#define A_SDL_ISGC        0x1001bdcb0ULL
typedef struct { uint8_t data[16]; } SdlGuid;
static void list_joysticks(void *ctx) {  // main thread
  struct { char *buf; size_t cap; } *q = ctx;
  int (*num)(void) = (void *)P(A_SDL_NUMJOY);
  const char *(*name)(int) = (void *)P(A_SDL_JOYNAME);
  SdlGuid (*guid)(int) = (void *)P(A_SDL_JOYGUID);
  void (*gstr)(SdlGuid, char *, int) = (void *)P(A_SDL_GUIDSTR);
  int (*isgc)(int) = (void *)P(A_SDL_ISGC);
  int n = num();
  size_t o = snprintf(q->buf, q->cap, "{\"joysticks\":[");
  for (int i = 0; i < n; i++) {
    char g[40];
    gstr(guid(i), g, sizeof g);
    const char *nm = name(i);
    o += snprintf(q->buf + o, q->cap - o, "%s{\"index\":%d,\"name\":\"%s\",\"guid\":\"%s\",\"controller\":%s}",
                  i ? "," : "", i, nm ? nm : "", g, isgc(i) ? "true" : "false");
  }
  snprintf(q->buf + o, q->cap - o, "]}");
}

// ---------------------------------------------------------------- keyboard gamepads
// A virtual pad driven by the keyboard while the game is the active app (`pad keyboard <slot> wasd|arrows`):
//   run: A/D | left/right  ->  left stick x (and the d-pad, for menus)
//   pole: W/S | up/down rotate the aim counter-clockwise / clockwise at KBD_SPIN rad/s; the right stick points at it
//         (so the pole follows and stays where it was left); the first press starts from the rider's pole angle
//   buttons: WASD: Space or E = A, Q = B, R = start;  arrows: Return = A, right Shift = B, Backspace = start
// The game draws its own Cocoa window and only uses SDL for controllers, so SDL's keyboard state stays empty: the
// keys are read from the window server instead (CGEventSourceKeyState, macOS virtual key codes), and only while
// Sportsfriends is the frontmost app (so typing elsewhere doesn't move a rider).
enum { VK_A = 0, VK_S = 1, VK_D = 2, VK_Q = 12, VK_W = 13, VK_E = 14, VK_R = 15, VK_RETURN = 36, VK_SPACE = 49,
       VK_BACKSPACE = 51, VK_RSHIFT = 60, VK_LEFT = 123, VK_RIGHT = 124, VK_DOWN = 125, VK_UP = 126 };
#define KBD_SPIN 6.2832f
enum { B_A = 0, B_B = 1, B_START = 6, B_UP = 11, B_DOWN = 12, B_LEFT = 13, B_RIGHT = 14 };  // indices into kBtnNames

// [NSApp isActive], refreshed on the main thread (AppKit isn't thread-safe: calling it from the game or socket
// thread crashed the game at start-up); app_is_active() returns the last value and queues a refresh
static volatile int g_app_active, g_active_queued;
static void refresh_active(void *unused) {
  (void)unused;
  id app = ((id (*)(id, SEL))objc_msgSend)((id)objc_getClass("NSApplication"), sel_registerName("sharedApplication"));
  g_app_active = app && ((BOOL (*)(id, SEL))objc_msgSend)(app, sel_registerName("isActive"));
  g_active_queued = 0;
}
static int app_is_active(void) {
  if (!g_active_queued) g_active_queued = 1, dispatch_async_f(dispatch_get_main_queue(), NULL, refresh_active);
  return g_app_active;
}

static int key_down(int active, int vk) {
  return active && CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, (CGKeyCode)vk);
}

static float pole_angle_of_pad(int slot) {  // the aim angle of the rider driven by gamepad `slot`, or 0
  char *ps = *(char **)P(A_PLAYSTATE_PTR);
  if (!ps) return 0;
  int np = *(int *)(ps + PS_NUM_PLAYERS);
  for (int i = 0; i < np && i < 4; i++) {
    char *pl = *(char **)(ps + PS_PLAYERS + 8 * i), *pole = pl ? *(char **)(pl + PL_POLE) : NULL;
    if (!pole || *(int *)(pl + PL_SESSION_PAD) != slot) continue;
    char *b0 = *(char **)(pole + POLE_SEGMENTS), *b9 = *(char **)(pole + POLE_SEGMENTS + 8 * (POLE_NSEG - 1));
    if (!b0 || !b9) break;
    float dx = ((float *)(b9 + B_POS))[0] - ((float *)(b0 + B_POS))[0];
    float dy = ((float *)(b9 + B_POS))[1] - ((float *)(b0 + B_POS))[1];
    return atan2f(-dy, dx);  // Box2D +y is down
  }
  return 0;
}

static void apply_keyboard(int slot, VPad *vp, int active) {  // called with g_lock held, game thread
  int w = vp->kbd == 1;
  int left = key_down(active, w ? VK_A : VK_LEFT), right = key_down(active, w ? VK_D : VK_RIGHT);
  int up = key_down(active, w ? VK_W : VK_UP), down = key_down(active, w ? VK_S : VK_DOWN);
  if ((up || down) && !vp->kbd_aimed) vp->kbd_angle = pole_angle_of_pad(slot), vp->kbd_aimed = 1;
  if (up != down) vp->kbd_angle += (up ? 1.0f : -1.0f) * KBD_SPIN / 60.0f;
  vp->axes[0] = (float)(right - left);
  vp->axes[1] = 0;
  vp->axes[2] = vp->kbd_aimed ? cosf(vp->kbd_angle) : 0;
  vp->axes[3] = vp->kbd_aimed ? -sinf(vp->kbd_angle) : 0;  // SDL stick +y is down
  vp->buttons[B_LEFT] = left, vp->buttons[B_RIGHT] = right, vp->buttons[B_UP] = up, vp->buttons[B_DOWN] = down;
  vp->buttons[B_A] = w ? (key_down(active, VK_SPACE) || key_down(active, VK_E)) : key_down(active, VK_RETURN);
  vp->buttons[B_B] = key_down(active, w ? VK_Q : VK_RSHIFT);
  vp->buttons[B_START] = key_down(active, w ? VK_R : VK_BACKSPACE);
}

static void apply_pads_pre(void) {
  int any_kbd = 0;
  for (int i = 0; i < 4; i++) any_kbd |= g_pads[i].kbd;
  int active = any_kbd && app_is_active();
  pthread_mutex_lock(&g_lock);
  for (int i = 0; i < 4; i++) {
    VPad *vp = &g_pads[i];
    char *s = SLOT(i);
    if (vp->want_connected && !vp->is_virtual && !s[GP_CONNECTED]) {
      *(int *)(s + GP_NUMSTICKS) = 2;
      *(int *)(s + GP_NUMBUTTONS) = NBTN;
      *(float *)(s + GP_DEADZONE0) = 0.1f;
      *(float *)(s + GP_DEADZONE0 + 0x30) = 0.1f;
      *(void **)(s + GP_SDLCONTROLLER) = NULL;
      *(int *)(s + GP_INSTANCEID) = -1000 - i;  // never matches a real SDL device
      *(void **)(s + GP_HAPTIC) = NULL;
      memset(s + GP_AXES, 0, 16);
      gp_connect(s, i, GAMEPAD_TYPE_SDL);
      vp->is_virtual = 1;
    } else if (!vp->want_connected && vp->is_virtual) {
      for (int b = 0; b < NBTN; b++) gp_set_button(s, kBtnAtrophy[b], 0);
      memset(s + GP_AXES, 0, 16);
      gp_disconnect(s);
      s[GP_CONNECTED] = 0;
      vp->is_virtual = 0;
    }
    // an overridden REAL controller: unhook its SDL controller for the duration (like a virtual pad's), else the
    // device's own axis updates overwrite the scripted sticks every frame (replays then diverge)
    if (vp->override && !vp->is_virtual && !vp->detached) {
      vp->saved_ctrl = *(void **)(s + GP_SDLCONTROLLER), vp->saved_iid = *(int *)(s + GP_INSTANCEID);
      *(void **)(s + GP_SDLCONTROLLER) = NULL, *(int *)(s + GP_INSTANCEID) = -2000 - i;
      vp->detached = 1;
    } else if (!vp->override && vp->detached) {
      *(void **)(s + GP_SDLCONTROLLER) = vp->saved_ctrl, *(int *)(s + GP_INSTANCEID) = vp->saved_iid;
      vp->detached = 0;
    }
    if (vp->is_virtual && vp->mirror) memcpy(vp->axes, SLOT(vp->mirror - 1) + GP_AXES, 16);  // a human's sticks
    if (vp->is_virtual && vp->kbd) apply_keyboard(i, vp, active);  // a human on the keyboard
    if (vp->is_virtual || vp->override) memcpy(s + GP_AXES, vp->axes, 16);  // Input_Update applies these
  }
  pthread_mutex_unlock(&g_lock);
}

static void apply_pads_post(void) {
  pthread_mutex_lock(&g_lock);
  for (int i = 0; i < 4; i++) {
    VPad *vp = &g_pads[i];
    if (vp->release && !vp->is_virtual) {
      for (int b = 0; b < NBTN; b++) gp_set_button(SLOT(i), kBtnAtrophy[b], 0);
      vp->release = 0;
    }
    if (!vp->is_virtual && !vp->override) continue;
    for (int b = 0; b < NBTN; b++) {
      float v = vp->buttons[b];
      if (vp->tap_frames[b] > 0) {
        v = 1;
        vp->tap_frames[b]--;
      }
      gp_set_button(SLOT(i), kBtnAtrophy[b], (unsigned char)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255));
    }
  }
  pthread_mutex_unlock(&g_lock);
}

// Park until stepped, then make Common::DT::NextDelta() see exactly g_sync_dt this frame.
static void lockstep_gate(void) {
  int keepalive = 0;  // a frame let through only to pump events: it must not simulate
  pthread_mutex_lock(&g_lock);
  if (!g_sync) {
    pthread_mutex_unlock(&g_lock);
    return;
  }
  while (g_sync && g_budget == 0) {
    if (farm_job_pending()) {  // farm mode: the parked game thread executes shadow jobs
      pthread_mutex_unlock(&g_lock);
      farm_run_job();
      pthread_mutex_lock(&g_lock);
      continue;
    }
    g_idle = 1;
    pthread_cond_broadcast(&g_idle_cv);
    if (!farm_active()) {
      pthread_cond_wait(&g_step_cv, &g_lock);
      continue;
    }
    // With shadows running the real match is irrelevant; let one real frame through every
    // 250 ms so the window keeps pumping events (otherwise macOS flags the app as hung).
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 250 * 1000000L;
    if (ts.tv_nsec >= 1000000000L) ts.tv_sec++, ts.tv_nsec -= 1000000000L;
    if (pthread_cond_timedwait(&g_step_cv, &g_lock, &ts) == ETIMEDOUT && !farm_job_pending() &&
        g_budget == 0) {
      keepalive = 1;
      break;
    }
  }
  g_idle = 0;
  if (g_budget > 0 && !keepalive) g_budget--;
  float dt = g_sync_dt;
  int sync = g_sync;
  pthread_mutex_unlock(&g_lock);
  if (!sync) return;
  if (g_ticks_per_us == 0)  // engine timebase is only initialized once the game is running
    g_ticks_per_us = 1e9 / (double)time_micros(1000000000ULL);
  // keepalive frames get dt ~ 0, so the fixed-step accumulator (zeroed each frame) never ticks
  *(uint64_t *)P(A_DT_LAST_UPDATE) = time_hires() - (keepalive ? 0 : (uint64_t)(dt * 1e6 * g_ticks_per_us));
}

static void hooked_input_update(void) {
  pthread_mutex_lock(&g_lock);
  int job = !g_sync && farm_job_pending();
  pthread_mutex_unlock(&g_lock);
  if (job) farm_run_job();
  int v = __atomic_exchange_n(&g_vsync_pending, -1, __ATOMIC_ACQ_REL);
  if (v >= 0) {  // needs the GL context, i.e. the game thread
    // the engine owns its NSOpenGLContext (SDL's GL API reports no context), so use CGL
    CGLContextObj ctx = CGLGetCurrentContext();
    GLint iv = v;
    int rc = ctx ? CGLSetParameter(ctx, kCGLCPSwapInterval, &iv) : -1;
    if (rc) fprintf(stderr, "sprbridge: swap interval %d failed (%d)\n", v, rc);
  }
  if (g_mute) snd_set_master(0);  // re-applied every frame: menus/options may restore it
  else if (__atomic_exchange_n(&g_unmute_pending, 0, __ATOMIC_ACQ_REL)) snd_set_master(1);
  lockstep_gate();
  apply_pads_pre();
  g_orig_input_update();
  apply_pads_post();
  g_input_frame++;
}

// ---------------------------------------------------------------- video recording
// The engine presents through NSOpenGLContext -flushBuffer -> CGLFlushDrawable: interposed so the frame
// about to be shown is read from the back buffer and piped (raw RGBA) to an ffmpeg started at the first frame.
static pthread_mutex_t g_rec_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_rec_path[512];
static int g_rec_on, g_rec_w, g_rec_h;
static long g_rec_frames;
static FILE *g_rec_pipe;
static unsigned char *g_rec_buf;

static void rec_frame(void) {
  pthread_mutex_lock(&g_rec_lock);
  if (g_rec_on) {
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int w = vp[2] & ~1, h = vp[3] & ~1;  // H.264 4:2:0 needs even sizes
    if (!g_rec_pipe && w > 0 && h > 0) {
      char cmd[1024];
      snprintf(cmd, sizeof cmd,
               "/opt/homebrew/bin/ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s %dx%d -r 60 -i - "
               "-vf 'vflip,scale=min(1920\\,iw):-2' -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p '%s'",
               w, h, g_rec_path);
      g_rec_pipe = popen(cmd, "w");
      g_rec_w = w, g_rec_h = h;
      g_rec_buf = malloc((size_t)w * h * 4);
    }
    if (g_rec_pipe && w == g_rec_w && h == g_rec_h) {
      glReadBuffer(GL_BACK);
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadPixels(vp[0], vp[1], w, h, GL_RGBA, GL_UNSIGNED_BYTE, g_rec_buf);  // the game's (letterboxed) area
      fwrite(g_rec_buf, 1, (size_t)w * h * 4, g_rec_pipe);
      g_rec_frames++;
    }
  }
  pthread_mutex_unlock(&g_rec_lock);
}

// ---------------------------------------------------------------- HUD text
// `hud <text>` / `hud off`: one line of text drawn over the bottom of every frame (e.g. who plays whom), rendered with
// CoreText into a bitmap (redone only when the text or the window height changes) and blended in with glDrawPixels
// just before the frame is shown. The game uses a legacy (2.1) GL context, so the fixed-function path is available.
static pthread_mutex_t g_hud_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_hud_text[256];
static int g_hud_dirty, g_hud_w, g_hud_h, g_hud_for_h;
static unsigned char *g_hud_px;

static void hud_render(int vh) {  // game thread, g_hud_lock held
  free(g_hud_px), g_hud_px = NULL, g_hud_w = g_hud_h = 0;
  g_hud_for_h = vh, g_hud_dirty = 0;
  if (!g_hud_text[0]) return;
  CGFloat size = 26.0 * vh / 1080.0 < 13 ? 13 : 26.0 * vh / 1080.0;
  CFStringRef str = CFStringCreateWithCString(NULL, g_hud_text, kCFStringEncodingUTF8);
  if (!str) return;
  CTFontRef font = CTFontCreateWithName(CFSTR("Helvetica-Bold"), size, NULL);
  const void *keys[] = {kCTFontAttributeName, kCTForegroundColorFromContextAttributeName};
  const void *vals[] = {font, kCFBooleanTrue};
  CFDictionaryRef attr = CFDictionaryCreate(NULL, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks,
                                            &kCFTypeDictionaryValueCallBacks);
  CFAttributedStringRef astr = CFAttributedStringCreate(NULL, str, attr);
  CTLineRef line = CTLineCreateWithAttributedString(astr);
  CGFloat asc, desc, lead;
  double tw = CTLineGetTypographicBounds(line, &asc, &desc, &lead);
  int pad = (int)(size * 0.45);
  g_hud_w = ((int)tw + 2 * pad + 3) & ~3, g_hud_h = (int)(asc + desc) + pad;
  g_hud_px = calloc((size_t)g_hud_w * g_hud_h, 4);
  CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
  CGContextRef c = CGBitmapContextCreate(g_hud_px, g_hud_w, g_hud_h, 8, g_hud_w * 4, cs,
                                         kCGImageAlphaPremultipliedLast);
  CGContextSetRGBFillColor(c, 0, 0, 0, 0.55);
  CGContextFillRect(c, CGRectMake(0, 0, g_hud_w, g_hud_h));
  CGContextSetRGBFillColor(c, 1, 1, 1, 1);
  CGContextSetTextPosition(c, pad, pad / 2 + desc);
  CTLineDraw(line, c);
  CGContextRelease(c), CGColorSpaceRelease(cs);
  CFRelease(line), CFRelease(astr), CFRelease(attr), CFRelease(font), CFRelease(str);
}

static void hud_draw(void) {  // game thread, inside the flush
  pthread_mutex_lock(&g_hud_lock);
  if (!g_hud_text[0] && !g_hud_px) { pthread_mutex_unlock(&g_hud_lock); return; }
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  if (g_hud_dirty || g_hud_for_h != vp[3]) hud_render(vp[3]);
  if (g_hud_px && g_hud_w <= vp[2]) {
    GLint prog = 0, pbo = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prog);
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
    glUseProgram(0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDisable(GL_DEPTH_TEST), glDisable(GL_SCISSOR_TEST), glDisable(GL_STENCIL_TEST), glDisable(GL_ALPHA_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  // premultiplied
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1), glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelZoom(1, -1);  // bitmap rows are top-down: draw downward from the label's top edge
    glWindowPos2i(vp[0] + (vp[2] - g_hud_w) / 2, vp[1] + vp[3] / 40 + g_hud_h);
    glDrawPixels(g_hud_w, g_hud_h, GL_RGBA, GL_UNSIGNED_BYTE, g_hud_px);
    glPopClientAttrib();
    glPopAttrib();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glUseProgram(prog);
  }
  pthread_mutex_unlock(&g_hud_lock);
}

static CGLError my_CGLFlushDrawable(CGLContextObj ctx) {
  hud_draw();
  rec_frame();
  return CGLFlushDrawable(ctx);
}
__attribute__((used)) static const struct { const void *repl, *orig; } g_interpose_flush
    __attribute__((section("__DATA,__interpose"))) = {(const void *)&my_CGLFlushDrawable, (const void *)&CGLFlushDrawable};

// ---------------------------------------------------------------- hooking

// Replace the first `len` bytes of `target` (which must equal `expect`) with an absolute
// jmp to `repl`; returns a trampoline that runs the displaced bytes and jumps back.
static void *hook(uintptr_t target, void *repl, const uint8_t *expect, size_t len) {
  if (len < 14 || memcmp((void *)target, expect, len) != 0) return NULL;
  uint8_t *tramp = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (tramp == MAP_FAILED) return NULL;
  uint8_t jmp[14] = {0xff, 0x25, 0, 0, 0, 0};  // jmp *0(%rip); .quad dest
  memcpy(tramp, (void *)target, len);
  uint64_t back = target + len;
  memcpy(jmp + 6, &back, 8);
  memcpy(tramp + len, jmp, 14);
  mprotect(tramp, 4096, PROT_READ | PROT_EXEC);

  uint64_t dest = (uint64_t)repl;
  memcpy(jmp + 6, &dest, 8);
  vm_address_t page = target & ~(vm_address_t)0xfff;
  if (vm_protect(mach_task_self(), page, 0x2000, 0, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY))
    return NULL;
  memcpy((void *)target, jmp, 14);
  for (size_t i = 14; i < len; i++) ((uint8_t *)target)[i] = 0x90;
  vm_protect(mach_task_self(), page, 0x2000, 0, VM_PROT_READ | VM_PROT_EXECUTE);
  return tramp;
}

// Overwrite code bytes after checking they are what we expect.
static int patch_code(uintptr_t addr, const uint8_t *expect, size_t elen, const uint8_t *with, size_t wlen) {
  if (memcmp((void *)addr, expect, elen)) return 0;
  vm_address_t page = addr & ~(vm_address_t)0xfff;
  if (vm_protect(mach_task_self(), page, 0x2000, 0, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY)) return 0;
  memcpy((void *)addr, with, wlen);
  vm_protect(mach_task_self(), page, 0x2000, 0, VM_PROT_READ | VM_PROT_EXECUTE);
  return 1;
}

// ---------------------------------------------------------------- socket commands

static int btn_index(const char *name) {
  for (int b = 0; b < NBTN; b++)
    if (!strcmp(name, kBtnNames[b])) return b;
  return -1;
}

static int axis_index(const char *name) {
  static const char *ax[4] = {"lx", "ly", "rx", "ry"};
  for (int a = 0; a < 4; a++)
    if (!strcmp(name, ax[a])) return a;
  return -1;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

#include "farm.c"

// Handles "pad ..." commands; returns reply text.
static void pad_cmd(char *args, char *out, size_t cap) {
  char *save, *verb = strtok_r(args, " ", &save), *idx = strtok_r(NULL, " ", &save);
  if (!verb || !idx || idx[1] || idx[0] < '0' || idx[0] > '3') {
    snprintf(out, cap, "{\"error\":\"usage: pad connect|disconnect|set|tap|clear <0-3> ...\"}");
    return;
  }
  int i = idx[0] - '0';
  VPad *vp = &g_pads[i];
  pthread_mutex_lock(&g_lock);
  int real = SLOT(i)[GP_CONNECTED] && !vp->is_virtual;
  if (!strcmp(verb, "connect")) {
    if (real) snprintf(out, cap, "{\"error\":\"slot %d has a real controller\"}", i);
    else vp->want_connected = 1, snprintf(out, cap, "{\"ok\":true}");
  } else if (!strcmp(verb, "disconnect")) {
    vp->want_connected = 0;
    memset(vp->buttons, 0, sizeof vp->buttons);
    memset(vp->tap_frames, 0, sizeof vp->tap_frames);
    memset(vp->axes, 0, sizeof vp->axes);
    snprintf(out, cap, "{\"ok\":true}");
  } else if (!strcmp(verb, "clear")) {
    memset(vp->buttons, 0, sizeof vp->buttons);
    memset(vp->tap_frames, 0, sizeof vp->tap_frames);
    memset(vp->axes, 0, sizeof vp->axes);
    snprintf(out, cap, "{\"ok\":true}");
  } else if (!strcmp(verb, "set")) {
    snprintf(out, cap, "{\"ok\":true}");
    for (char *kv; (kv = strtok_r(NULL, " ", &save));) {
      char *eq = strchr(kv, '=');
      if (!eq) { snprintf(out, cap, "{\"error\":\"expected key=value, got %s\"}", kv); break; }
      *eq = 0;
      float v = strtof(eq + 1, NULL);
      int b = btn_index(kv), a = axis_index(kv);
      if (b >= 0) vp->buttons[b] = clampf(v, 0, 1);
      else if (a >= 0) vp->axes[a] = clampf(v, -1, 1);
      else { snprintf(out, cap, "{\"error\":\"unknown input %s\"}", kv); break; }
    }
  } else if (!strcmp(verb, "keyboard")) {  // pad keyboard <i> wasd|arrows|off (connects the pad if needed)
    char *m = strtok_r(NULL, " ", &save);
    int mode = m && !strcmp(m, "wasd") ? 1 : m && !strcmp(m, "arrows") ? 2 : m && !strcmp(m, "off") ? 0 : -1;
    if (mode < 0) snprintf(out, cap, "{\"error\":\"usage: pad keyboard <0-3> wasd|arrows|off\"}");
    else if (real) snprintf(out, cap, "{\"error\":\"slot %d has a real controller\"}", i);
    else {
      vp->kbd = mode, vp->kbd_aimed = 0;
      if (mode) vp->want_connected = 1;
      else memset(vp->buttons, 0, sizeof vp->buttons), memset(vp->axes, 0, sizeof vp->axes);
      snprintf(out, cap, "{\"ok\":true}");
    }
  } else if (!strcmp(verb, "override")) {  // pad override <i> on|off: script a REAL controller (its own input ignored)
    char *m = strtok_r(NULL, " ", &save);
    if (!m || (strcmp(m, "on") && strcmp(m, "off"))) snprintf(out, cap, "{\"error\":\"usage: pad override <0-3> on|off\"}");
    else {
      vp->override = !strcmp(m, "on");
      memset(vp->buttons, 0, sizeof vp->buttons), memset(vp->tap_frames, 0, sizeof vp->tap_frames);
      memset(vp->axes, 0, sizeof vp->axes);
      vp->release = !vp->override;  // hand the controller back without stuck buttons (apply_pads_post)
      snprintf(out, cap, "{\"ok\":true}");
    }
  } else if (!strcmp(verb, "mirror")) {  // pad mirror <i> <real slot>|off: a human drives this pad's player
    char *src = strtok_r(NULL, " ", &save);
    if (src && !strcmp(src, "off")) vp->mirror = 0, snprintf(out, cap, "{\"ok\":true}");
    else if (src && src[0] >= '0' && src[0] <= '3' && !src[1]) vp->mirror = 1 + (src[0] - '0'), snprintf(out, cap, "{\"ok\":true}");
    else snprintf(out, cap, "{\"error\":\"usage: pad mirror <0-3> <0-3>|off\"}");
  } else if (!strcmp(verb, "tap")) {
    char *name = strtok_r(NULL, " ", &save), *nf = strtok_r(NULL, " ", &save);
    int b = name ? btn_index(name) : -1;
    if (b < 0) snprintf(out, cap, "{\"error\":\"unknown button\"}");
    else vp->tap_frames[b] = nf ? atoi(nf) : 4, snprintf(out, cap, "{\"ok\":true}");
  } else {
    snprintf(out, cap, "{\"error\":\"unknown pad command %s\"}", verb);
  }
  pthread_mutex_unlock(&g_lock);
}

static void reply(int fd, char *line) {
  char buf[1024];
  if (!strcmp(line, "ping")) {
    snprintf(buf, sizeof buf, "pong");
  } else if (!strcmp(line, "frame")) {
    snprintf(buf, sizeof buf, "{\"frame\":%lu,\"match_frame\":%lu}", g_input_frame, g_pr_frame);
  } else if (!strcmp(line, "settings")) {
    char *sd = (char *)P(A_SAVEDATA);
    snprintf(buf, sizeof buf,
             "{\"config_scorelimit\":%g,\"config_duration_min\":%g,"
             "\"match_max_score\":%d,\"match_round_time_s\":%g}",
             *(float *)(sd + SAVE_PR_SCORELIMIT), *(float *)(sd + SAVE_PR_DURATION),
             *(int *)P(A_MAXSCORE), *(float *)P(A_ROUNDTIME));
  } else if (!strncmp(line, "playlog on ", 11)) {
    snprintf(g_playlog_dir, sizeof g_playlog_dir, "%s", line + 11);
    snprintf(buf, sizeof buf, "{\"ok\":true,\"dir\":\"%.400s\"}", g_playlog_dir);
  } else if (!strcmp(line, "playlog off")) {
    g_playlog_dir[0] = 0;
    if (g_playlog) fclose(g_playlog), g_playlog = NULL;
    snprintf(buf, sizeof buf, "{\"ok\":true}");
  } else if (!strcmp(line, "playlog")) {
    snprintf(buf, sizeof buf, "{\"on\":%s,\"dir\":\"%.400s\"}", g_playlog_dir[0] ? "true" : "false", g_playlog_dir);
  } else if (!strncmp(line, "record start ", 13)) {
    pthread_mutex_lock(&g_rec_lock);
    snprintf(g_rec_path, sizeof g_rec_path, "%s", line + 13);
    g_rec_on = 1, g_rec_frames = 0;
    pthread_mutex_unlock(&g_rec_lock);
    snprintf(buf, sizeof buf, "{\"ok\":true,\"file\":\"%.400s\"}", g_rec_path);
  } else if (!strcmp(line, "record stop")) {
    pthread_mutex_lock(&g_rec_lock);
    long n = g_rec_frames;
    g_rec_on = 0;
    if (g_rec_pipe) pclose(g_rec_pipe);  // waits for ffmpeg to finish the file
    g_rec_pipe = NULL;
    free(g_rec_buf), g_rec_buf = NULL;
    pthread_mutex_unlock(&g_rec_lock);
    snprintf(buf, sizeof buf, "{\"ok\":true,\"frames\":%ld,\"size\":\"%dx%d\"}", n, g_rec_w, g_rec_h);
  } else if (!strncmp(line, "settings set ", 13)) {  // settings set <score limit> <minutes>: the save data
    float sc, mn;                                      // PoleRiders::Begin reads at the next match + the live
    if (sscanf(line + 13, "%f %f", &sc, &mn) == 2) {   // Assets values
      char *sd = (char *)P(A_SAVEDATA);
      *(float *)(sd + SAVE_PR_SCORELIMIT) = sc;
      *(float *)(sd + SAVE_PR_DURATION) = mn;
      *(int *)P(A_MAXSCORE) = (int)sc;
      *(float *)P(A_ROUNDTIME) = mn * 60.0f;
      snprintf(buf, sizeof buf, "{\"ok\":true,\"score_limit\":%g,\"minutes\":%g}", sc, mn);
    } else {
      snprintf(buf, sizeof buf, "{\"error\":\"usage: settings set <score limit> <minutes>\"}");
    }
  } else if (!strcmp(line, "pads")) {
    size_t n = snprintf(buf, sizeof buf, "{\"pads\":[");
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < 4; i++)
      n += snprintf(buf + n, sizeof buf - n, "%s{\"slot\":%d,\"connected\":%s,\"virtual\":%s}",
                    i ? "," : "", i, SLOT(i)[GP_CONNECTED] ? "true" : "false",
                    g_pads[i].is_virtual ? "true" : "false");
    pthread_mutex_unlock(&g_lock);
    snprintf(buf + n, sizeof buf - n, "]}");
  } else if (!strncmp(line, "hud", 3) && (!line[3] || line[3] == ' ')) {  // hud <text> | hud off
    const char *t = line[3] ? line + 4 : "";
    pthread_mutex_lock(&g_hud_lock);
    snprintf(g_hud_text, sizeof g_hud_text, "%s", strcmp(t, "off") ? t : "");
    g_hud_dirty = 1;
    pthread_mutex_unlock(&g_hud_lock);
    snprintf(buf, sizeof buf, "{\"ok\":true}");
  } else if (!strcmp(line, "joysticks")) {  // debug: the joysticks the game's SDL sees (index, name, GUID, mapped)
    struct { char *buf; size_t cap; } jq = {buf, sizeof buf};
    dispatch_sync_f(dispatch_get_main_queue(), &jq, list_joysticks);
  } else if (!strcmp(line, "keys")) {  // debug: the keys down (macOS key codes) + the keyboard pads' sticks
    size_t n = snprintf(buf, sizeof buf, "{\"active\":%s,\"down\":[", app_is_active() ? "true" : "false");
    for (int vk = 0, first = 1; vk < 128; vk++)
      if (CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, (CGKeyCode)vk))
        n += snprintf(buf + n, sizeof buf - n, "%s%d", first ? "" : ",", vk), first = 0;
    n += snprintf(buf + n, sizeof buf - n, "],\"kbd\":[");
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < 4; i++)
      n += snprintf(buf + n, sizeof buf - n, "%s[%d,%.2f,%.2f,%.2f]", i ? "," : "", g_pads[i].kbd,
                    g_pads[i].axes[0], g_pads[i].axes[2], g_pads[i].axes[3]);
    pthread_mutex_unlock(&g_lock);
    snprintf(buf + n, sizeof buf - n, "]}");
  } else if (!strncmp(line, "sync", 4)) {
    pthread_mutex_lock(&g_lock);
    if (!strncmp(line, "sync on", 7)) {
      g_sync = 1, g_budget = 0, g_sync_owner = t_conn;
      snprintf(buf, sizeof buf, "{\"ok\":true,\"dt\":%g}", g_sync_dt);
    } else if (!strcmp(line, "sync off")) {
      g_sync = 0;
      pthread_cond_broadcast(&g_step_cv);
      snprintf(buf, sizeof buf, "{\"ok\":true}");
    } else {
      snprintf(buf, sizeof buf, "{\"sync\":%s,\"dt\":%g}", g_sync ? "true" : "false", g_sync_dt);
    }
    pthread_mutex_unlock(&g_lock);
  } else if (!strncmp(line, "act ", 4)) {
    long n;
    float v[6];
    if (sscanf(line + 4, "%ld %f %f %f %f %f %f", &n, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 7) {
      snprintf(buf, sizeof buf, "{\"error\":\"usage: act n lx0 rx0 ry0 lx1 rx1 ry1\"}");
    } else {
      pthread_mutex_lock(&g_lock);
      for (int i = 0; i < 2; i++) {
        g_pads[i].axes[0] = clampf(v[3 * i], -1, 1);
        g_pads[i].axes[2] = clampf(v[3 * i + 1], -1, 1);
        g_pads[i].axes[3] = clampf(v[3 * i + 2], -1, 1);
      }
      pthread_mutex_unlock(&g_lock);
      snprintf(line, 32, "step %ld", n);
      reply(fd, line);
      return;
    }
  } else if (!strncmp(line, "step", 4) && (!line[4] || line[4] == ' ')) {
    long n = line[4] ? atol(line + 5) : 1;
    pthread_mutex_lock(&g_lock);
    if (!g_sync) {
      pthread_mutex_unlock(&g_lock);
      snprintf(buf, sizeof buf, "{\"error\":\"step needs sync on\"}");
    } else {
      g_budget += n > 0 ? n : 1;
      g_idle = 0;
      pthread_cond_broadcast(&g_step_cv);
      while (g_sync && !(g_idle && g_budget == 0)) pthread_cond_wait(&g_idle_cv, &g_lock);
      pthread_mutex_unlock(&g_lock);
      strcpy(line, "state");
      reply(fd, line);
      return;
    }
  } else if (!strncmp(line, "ball ", 5)) {
    float v[4] = {0, 0, 0, 0};
    int k = sscanf(line + 5, "%f %f %f %f", &v[0], &v[1], &v[2], &v[3]);
    if (k != 2 && k != 4) {
      snprintf(buf, sizeof buf, "{\"error\":\"usage: ball x y [vx vy]\"}");
    } else {
      memcpy(g_ball_set, v, sizeof v);
      __atomic_store_n(&g_ball_pending, 1, __ATOMIC_RELEASE);
      snprintf(buf, sizeof buf, "{\"ok\":true}");
    }
  } else if (!strcmp(line, "mute on") || !strcmp(line, "mute off")) {
    g_mute = line[6] == 'n';
    if (!g_mute) __atomic_store_n(&g_unmute_pending, 1, __ATOMIC_RELEASE);
    snprintf(buf, sizeof buf, "{\"ok\":true}");
  } else if (!strncmp(line, "farm ", 5)) {
    farm_command(fd, line + 5);
    return;
  } else if (!strcmp(line, "level")) {
    pthread_mutex_lock(&g_lock);
    g_level_req = 1;
    while (g_level_req) pthread_cond_wait(&g_farmtest_cv, &g_lock);
    char *out = g_level_out;
    g_level_out = NULL;
    pthread_mutex_unlock(&g_lock);
    size_t len = strlen(out);
    out[len] = '\n';
    for (size_t off = 0; off < len + 1;) {
      ssize_t w = write(fd, out + off, len + 1 - off);
      if (w <= 0) break;
      off += w;
    }
    free(out);
    return;
  } else if (!strncmp(line, "fast on", 7)) {
    int n = line[7] ? atoi(line + 8) : 10;
    g_render_every = n > 1 ? n : 10;
    __atomic_store_n(&g_vsync_pending, 0, __ATOMIC_RELEASE);
    snprintf(buf, sizeof buf, "{\"ok\":true,\"render_every\":%d}", g_render_every);
  } else if (!strcmp(line, "fast off")) {
    g_render_every = 1;
    __atomic_store_n(&g_vsync_pending, 1, __ATOMIC_RELEASE);
    snprintf(buf, sizeof buf, "{\"ok\":true}");
  } else if (!strncmp(line, "reset", 5) && (!line[5] || line[5] == ' ')) {
    if (line[5]) __atomic_store_n(&g_reset_seed, atol(line + 6), __ATOMIC_RELEASE);
    __atomic_store_n(&g_reset_pending, 1, __ATOMIC_RELEASE);
    snprintf(buf, sizeof buf, "{\"ok\":true,\"match_id\":%lu}", g_match_id);
  } else if (!strncmp(line, "pad ", 4)) {
    pad_cmd(line + 4, buf, sizeof buf);
  } else if (!strcmp(line, "state")) {  // per-call buffer: several clients may ask at once
    pthread_mutex_lock(&g_lock);
    size_t len = strlen(g_snap);
    char *out = malloc(len + 1);
    memcpy(out, g_snap, len);
    pthread_mutex_unlock(&g_lock);
    out[len] = '\n';
    for (size_t off = 0; off < len + 1;) {
      ssize_t w = write(fd, out + off, len + 1 - off);
      if (w <= 0) break;
      off += w;
    }
    free(out);
    return;
  } else {
    snprintf(buf, sizeof buf, "{\"error\":\"unknown command\"}");
  }
  size_t n = strlen(buf);
  buf[n] = '\n';
  write(fd, buf, n + 1);
}

static void *serve_client(void *arg) {
  int fd = (int)(intptr_t)arg;
  t_conn = __atomic_fetch_add(&g_next_conn, 1, __ATOMIC_RELAXED);
  char *line = malloc(1 << 20);  // `farm act` carries 6 floats per shadow
  size_t used = 0;
  for (;;) {
    ssize_t r = read(fd, line + used, (1 << 20) - 1 - used);
    if (r <= 0) break;
    used += r;
    char *nl;
    while ((nl = memchr(line, '\n', used))) {
      *nl = 0;
      if (nl > line && nl[-1] == '\r') nl[-1] = 0;
      reply(fd, line);
      used -= nl + 1 - line;
      memmove(line, nl + 1, used);
    }
    if (used == (1 << 20) - 1) used = 0;  // overlong line: drop it
  }
  pthread_mutex_lock(&g_lock);  // a vanished agent must not leave the game frozen
  if (g_sync && g_sync_owner == t_conn) {
    g_sync = 0;
    pthread_cond_broadcast(&g_step_cv);
  }
  pthread_mutex_unlock(&g_lock);
  free(line);
  close(fd);
  return NULL;
}

static void *server(void *path) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  strncpy(a.sun_path, path, sizeof a.sun_path - 1);
  unlink(a.sun_path);
  if (bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 4)) {
    fprintf(stderr, "sprbridge: cannot listen on %s\n", a.sun_path);
    return NULL;
  }
  for (;;) {
    int c = accept(s, NULL, NULL);
    if (c < 0) continue;
    pthread_t t;
    pthread_create(&t, NULL, serve_client, (void *)(intptr_t)c);
    pthread_detach(t);
  }
}

__attribute__((constructor)) static void init(void) {
  const char *img = _dyld_get_image_name(0);
  if (!img || !strstr(img, "/MacOS/Sportsfriends")) return;  // e.g. the `moved` helper
  g_slide = _dyld_get_image_vmaddr_slide(0);
  strcpy(g_snap, "{\"frame\":0,\"in_match\":false}");
  gp_connect = (void *)P(A_GP_CONNECT);
  gp_disconnect = (void *)P(A_GP_DISCONNECT);
  gp_set_button = (void *)P(A_GP_SETBUTTON);
  time_hires = (void *)P(A_TIME_HIRES);
  b2_set_transform = (void *)P(A_B2_SETTRANSFORM);
  b2_set_awake = (void *)P(A_B2_SETAWAKE);
  snd_set_master = (void *)P(A_SND_SETMASTER);
  b2_fixture_list = (void *)P(A_B2_FIXTURELIST);
  b2_fix_next = (void *)P(A_B2_FIX_NEXT);
  b2_fix_shape = (void *)P(A_B2_FIX_SHAPE);
  b2_fix_sensor = (void *)P(A_B2_FIX_SENSOR);
  b2_shape_type = (void *)P(A_B2_SHAPE_TYPE);
  ps_ctor = (void *)P(A_PS_CTOR);
  ps_load_complete = (void *)P(A_PS_LOADCOMPLETE);
  ps_update = (void *)P(A_PS_UPDATE);
  ps_delete = (void *)P(A_PS_DTOR_DELETE);
  g_mute = getenv("SPRBRIDGE_MUTE") && atoi(getenv("SPRBRIDGE_MUTE"));
  g_no_unlock = getenv("SPRBRIDGE_NOUNLOCK") && atoi(getenv("SPRBRIDGE_NOUNLOCK"));
  if (getenv("SPRBRIDGE_PLAYLOG")) snprintf(g_playlog_dir, sizeof g_playlog_dir, "%s", getenv("SPRBRIDGE_PLAYLOG"));
  time_micros = (void *)P(A_TIME_MICROS);

  // push rbp; mov rbp,rsp; push r15; push r14; push r12; push rbx; sub rsp,0x70
  static const uint8_t pr_prologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                        0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x70};
  // push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x708
  static const uint8_t in_prologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                        0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48,
                                        0x81, 0xec, 0x08, 0x07, 0x00, 0x00};
  // Every instance starts a PS Move daemon that binds the same port; all but one assert and get
  // respawned forever (and take games down with them). Pole Riders never uses PS Move: stub it.
  if (getenv("SPRBRIDGE_NOMOVE") && atoi(getenv("SPRBRIDGE_NOMOVE"))) {
    static const uint8_t init_pro[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56};
    static const uint8_t upd_pro[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
    static const uint8_t ret[] = {0xc3};
    int ok = patch_code(P(A_MOVE_INIT), init_pro, sizeof init_pro, ret, 1) &&
             patch_code(P(A_MOVE_UPDATE), upd_pro, sizeof upd_pro, ret, 1);
    fprintf(stderr, "sprbridge: PS Move %s\n", ok ? "disabled" : "stub FAILED (unexpected bytes)");
  }
  // Canonical rules for training: PlayState's constructor randomly enables snake (1/10000) or
  // pencil (~1/900) poles, and date-triggered Christmas / Chinese New Year modes. Force all off.
  if (getenv("SPRBRIDGE_CANON") && atoi(getenv("SPRBRIDGE_CANON"))) {
    static const struct { uintptr_t a; uint8_t len, from[5], to[5]; } pt[] = {
        {0x100134306, 3, {0x0f, 0x97, 0x00}, {0xc6, 0x00, 0x00}},  // snake: seta (%rax) -> movb $0
        {0x100134351, 2, {0xb0, 0x01}, {0xb0, 0x00}},              // pencil: mov al,1 -> mov al,0
        {0x1001342b9, 3, {0x0f, 0x94, 0xc0}, {0x30, 0xc0, 0x90}},  // christmas: sete al -> xor al
        {0x100134289, 5, {0xe8, 0x7e, 0x12, 0x00, 0x00}, {0x30, 0xc0, 0x90, 0x90, 0x90}}};  // CNY
    int ok = 1;
    for (unsigned i = 0; i < sizeof pt / sizeof *pt; i++)
      ok &= patch_code(P(pt[i].a), pt[i].from, pt[i].len, pt[i].to, pt[i].len);
    fprintf(stderr, "sprbridge: canonical rules %s\n", ok ? "on" : "patch FAILED");
  }
  farm_init();
  g_orig_pr_update = hook(P(A_POLERIDERS_UPDATE), hooked_pr_update, pr_prologue, sizeof pr_prologue);
  g_orig_input_update = hook(P(A_INPUT_UPDATE), hooked_input_update, in_prologue, sizeof in_prologue);
  if (!g_orig_pr_update || !g_orig_input_update) {
    fprintf(stderr, "sprbridge: prologue mismatch — wrong game build? (update:%p input:%p)\n",
            (void *)g_orig_pr_update, (void *)g_orig_input_update);
    return;
  }

  static char path[104];
  const char *env = getenv("SPRBRIDGE_SOCK");
  if (env) snprintf(path, sizeof path, "%s", env);
  else snprintf(path, sizeof path, "%s/Library/Application Support/sportsfriends/sprbridge.sock",
                getenv("HOME"));
  pthread_t t;
  pthread_create(&t, NULL, server, path);
  pthread_detach(t);
  fprintf(stderr, "sprbridge: loaded (slide %#lx), socket %s\n", (long)g_slide, path);
}
