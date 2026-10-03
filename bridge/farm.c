// farm.c — headless "shadow" matches inside one game process. #included by sprbridge.c.
//
// A shadow is an extra PoleRiders::PlayState built next to the real match and ticked directly
// (PlayState::Update = one 1/60 s tick) — the game's own code, no drawing, no sound, no frame.
// Each shadow keeps private copies of the engine's gamepad state for pads 0/1; before a tick
// they are swapped in and the shadow's sticks are pushed through the real input chain
// (Gamepad::setJoystickPosition -> Common::Input::_UpdatePad), exactly as a rendered frame does.
//
// Sound: shadows share the engine's FMOD objects with the real match, and a shadow's destructor
// would release sounds the real match still uses. So while the game thread runs shadow code,
// the FMOD C API (imported from libfmodex.dylib) is interposed: sound creation hands out fake
// handles, playback returns fake channels, calls on fake handles are no-ops, and real sounds
// are never released.
//
// Socket commands (all shadow work runs on the game thread; farm mode wants `sync on` so the
// real match stays frozen and the game thread just executes farm jobs):
//   farm create N            add N shadows (the real match must be running)
//   farm act n a...          set every shadow's sticks (6 floats each: lx0 rx0 ry0 lx1 rx1 ry1),
//                            tick every shadow n times, reply {"matches":[snapshot, ...]}
//   farm reset k [clock]     rebuild shadow k and reply with its tick-0 snapshot. With a clock
//                            (seconds), the match starts with that much time left and the
//                            timer running: e.g. 0.05 drops a 0-0 match into sudden death
//   farm destroy             delete all shadows
//   farm state               the last `farm act` reply (read-only; for dashboards)
//   farm info                shadow count, ticks run, microseconds per tick
//   farm dump k              raw bytes (hex) of shadow k's whole Box2D world — world header, every
//                            body/fixture/shape, joint and contact in list order — and of the game
//                            objects (PlayState, players, poles, ball): the oracle for the native sim
//                            (sim/). Pointers are reported so the diff tool can mask/relate them.
//   farm physstep k n        n bare b2World::Step(1/60, 60, 18) calls on shadow k's world with the
//                            game's contact listener/filter swapped for Box2D's defaults: no game
//                            logic at all (tests the native Box2D build in isolation)

#define A_GP_SETJOY  0x1000116ceULL  // atrophy::Gamepad::setJoystickPosition(int, float, float)
#define A_UPDATEPAD  0x10005f660ULL  // Common::Input::_UpdatePad(int, float)
#define A_PADS       0x100c5aca0ULL  // Common::Input::_pads[8] (0x2c8 each)
#define PAD_STRIDE   0x2c8
#define A_LOADING_SOUNDS 0x10064db40ULL  // Common::Sound (anonymous)::loadingSounds[512]
#define N_LOADING_SOUNDS 512
#define MAX_SHADOWS  256

typedef struct {
  char *ps;
  uint8_t slot[2][GP_STRIDE];  // private atrophy::Gamepad state for pads 0/1
  uint8_t pad[2][PAD_STRIDE];  // private Common::Input::_pads state for pads 0/1
  float act[6];                // lx0 rx0 ry0 lx1 rx1 ry1 (SDL stick convention, +y down)
  unsigned long frame, match_id;
} Shadow;

static Shadow g_sh[MAX_SHADOWS];
static int g_nsh;
static int g_in_shadow;  // game thread is inside shadow code: FMOD calls are neutralized
static unsigned long g_sh_next_match = 1, g_sh_ticks;
static double g_sh_tick_s, g_sh_step_s, g_sh_snap_s;  // profiling: ticks, Box2D steps, snapshot JSON
static void (*g_orig_b2step)(void *, float, int, int);

// b2World::Step wrapper: times Box2D's share of shadow ticks (farm info)
static void hooked_b2step(void *w, float dt, int vi, int pi) {
  if (!g_in_shadow) {
    g_orig_b2step(w, dt, vi, pi);
    return;
  }
  double t0 = now_s();
  g_orig_b2step(w, dt, vi, pi);
  g_sh_step_s += now_s() - t0;
}
static void (*gp_setjoy)(void *, int, float, float);
static void (*update_pad)(int, float);

enum { JOB_NONE, JOB_CREATE, JOB_ACT, JOB_RESET, JOB_DESTROY, JOB_DUMP, JOB_PHYSSTEP };
static struct {
  int type, n, k, done;
  float clock;
  float *acts;
  char *out;
} g_job;
static char *g_farm_last;  // copy of the last act reply, for `farm state` (guarded by g_lock)

// ---------------------------------------------------------------- FMOD interposition

typedef int FR;  // FMOD_RESULT; 0 = FMOD_OK
FR FMOD_System_CreateSound(void *, const char *, unsigned, void *, void **);
FR FMOD_System_CreateStream(void *, const char *, unsigned, void *, void **);
FR FMOD_System_PlaySound(void *, int, void *, int, void **);
FR FMOD_Sound_Release(void *);
FR FMOD_Sound_GetDefaults(void *, float *, float *, float *, int *);
FR FMOD_Sound_SetDefaults(void *, float, float, float, int);
FR FMOD_Sound_GetOpenState(void *, int *, unsigned *, int *, int *);
FR FMOD_Channel_SetVolume(void *, float);
FR FMOD_Channel_SetPan(void *, float);
FR FMOD_Channel_SetFrequency(void *, float);
FR FMOD_Channel_SetPaused(void *, int);
FR FMOD_Channel_GetPaused(void *, int *);
FR FMOD_Channel_IsPlaying(void *, int *);
FR FMOD_Channel_Stop(void *);
FR FMOD_Channel_SetPosition(void *, unsigned, unsigned);
FR FMOD_Channel_SetChannelGroup(void *, void *);

static char g_fake[1 << 16];
static unsigned g_fake_i;
static int is_fake(const void *p) { return (const char *)p >= g_fake && (const char *)p < g_fake + sizeof g_fake; }
static void *fake_handle(void) {
  g_fake_i = (g_fake_i + 16) % sizeof g_fake;
  return g_fake + g_fake_i;
}

static FR my_CreateSound(void *sys, const char *n, unsigned m, void *ex, void **s) {
  if (g_in_shadow) return *s = fake_handle(), 0;
  return FMOD_System_CreateSound(sys, n, m, ex, s);
}
static FR my_CreateStream(void *sys, const char *n, unsigned m, void *ex, void **s) {
  if (g_in_shadow) return *s = fake_handle(), 0;
  return FMOD_System_CreateStream(sys, n, m, ex, s);
}
static FR my_PlaySound(void *sys, int id, void *snd, int paused, void **ch) {
  if (g_in_shadow || is_fake(snd)) {
    if (ch) *ch = fake_handle();
    return 0;
  }
  return FMOD_System_PlaySound(sys, id, snd, paused, ch);
}
static FR my_SoundRelease(void *s) {
  if (is_fake(s) || g_in_shadow) return 0;  // never release a real sound from shadow code
  return FMOD_Sound_Release(s);
}
static FR my_GetDefaults(void *s, float *f, float *v, float *p, int *pr) {
  if (!is_fake(s)) return FMOD_Sound_GetDefaults(s, f, v, p, pr);
  if (f) *f = 44100;
  if (v) *v = 1;
  if (p) *p = 0;
  if (pr) *pr = 128;
  return 0;
}
static FR my_SetDefaults(void *s, float f, float v, float p, int pr) {
  if (is_fake(s) || g_in_shadow) return 0;
  return FMOD_Sound_SetDefaults(s, f, v, p, pr);
}
static FR my_GetOpenState(void *s, int *st, unsigned *pb, int *starving, int *busy) {
  if (!is_fake(s)) return FMOD_Sound_GetOpenState(s, st, pb, starving, busy);
  if (st) *st = 0;  // FMOD_OPENSTATE_READY
  if (pb) *pb = 100;
  if (starving) *starving = 0;
  if (busy) *busy = 0;
  return 0;
}
#define CH1(name, T)                                   \
  static FR my_##name(void *c, T v) {                  \
    if (is_fake(c) || g_in_shadow) return 0;           \
    return FMOD_Channel_##name(c, v);                  \
  }
CH1(SetVolume, float)
CH1(SetPan, float)
CH1(SetFrequency, float)
CH1(SetPaused, int)
CH1(SetChannelGroup, void *)
static FR my_GetPaused(void *c, int *v) {
  if (is_fake(c)) return *v = 0, 0;
  return FMOD_Channel_GetPaused(c, v);
}
static FR my_IsPlaying(void *c, int *v) {
  if (is_fake(c)) return *v = 0, 0;
  return FMOD_Channel_IsPlaying(c, v);
}
static FR my_Stop(void *c) {
  if (is_fake(c) || g_in_shadow) return 0;
  return FMOD_Channel_Stop(c);
}
static FR my_SetPosition(void *c, unsigned pos, unsigned unit) {
  if (is_fake(c) || g_in_shadow) return 0;
  return FMOD_Channel_SetPosition(c, pos, unit);
}

#define INTERPOSE(rep, orig)                                                       \
  __attribute__((used)) static const struct {                                      \
    const void *r, *o;                                                             \
  } _ip_##orig __attribute__((section("__DATA,__interpose"))) = {(const void *)&rep, \
                                                                  (const void *)&orig};
INTERPOSE(my_CreateSound, FMOD_System_CreateSound)
INTERPOSE(my_CreateStream, FMOD_System_CreateStream)
INTERPOSE(my_PlaySound, FMOD_System_PlaySound)
INTERPOSE(my_SoundRelease, FMOD_Sound_Release)
INTERPOSE(my_GetDefaults, FMOD_Sound_GetDefaults)
INTERPOSE(my_SetDefaults, FMOD_Sound_SetDefaults)
INTERPOSE(my_GetOpenState, FMOD_Sound_GetOpenState)
INTERPOSE(my_SetVolume, FMOD_Channel_SetVolume)
INTERPOSE(my_SetPan, FMOD_Channel_SetPan)
INTERPOSE(my_SetFrequency, FMOD_Channel_SetFrequency)
INTERPOSE(my_SetPaused, FMOD_Channel_SetPaused)
INTERPOSE(my_SetChannelGroup, FMOD_Channel_SetChannelGroup)
INTERPOSE(my_GetPaused, FMOD_Channel_GetPaused)
INTERPOSE(my_IsPlaying, FMOD_Channel_IsPlaying)
INTERPOSE(my_Stop, FMOD_Channel_Stop)
INTERPOSE(my_SetPosition, FMOD_Channel_SetPosition)

// ---------------------------------------------------------------- shadows (game thread only)

static char *real_pad(int i) { return (char *)P(A_PADS) + i * PAD_STRIDE; }

// Every new sound is registered in a 512-entry "still loading" list that the engine only drains
// in Common::Sound::Update — which never runs while the game thread is parked in farm mode. The
// shadows' sounds are fake anyway: drop them from the list, or it fills up and the engine aborts.
static void purge_fake_loading_sounds(void) {
  void **ls = (void **)P(A_LOADING_SOUNDS);
  for (int i = 0; i < N_LOADING_SOUNDS; i++)
    if (ls[i] && is_fake(ls[i])) ls[i] = NULL;
}

static void sh_create(Shadow *s) {
  g_in_shadow = 1;
  s->ps = calloc(1, PS_SIZE);
  ps_ctor(s->ps, 0);
  ps_load_complete(s->ps);
  g_in_shadow = 0;
  purge_fake_loading_sounds();
  for (int i = 0; i < 2; i++) {  // start from the engine's current (connected, idle) pad state
    memcpy(s->slot[i], SLOT(i), GP_STRIDE);
    memcpy(s->pad[i], real_pad(i), PAD_STRIDE);
    memset(s->slot[i] + GP_AXES, 0, 16);
  }
  memset(s->act, 0, sizeof s->act);
  s->frame = 0;
  s->match_id = g_sh_next_match++;
}

static void sh_destroy(Shadow *s) {
  if (!s->ps) return;
  g_in_shadow = 1;
  ps_delete(s->ps);
  g_in_shadow = 0;
  s->ps = NULL;
}

// One 1/60 s tick of shadow s, with its inputs pushed through the engine's real input chain.
static void sh_tick(Shadow *s) {
  for (int i = 0; i < 2; i++) {
    char *slot = SLOT(i);
    memcpy(slot, s->slot[i], GP_STRIDE);
    memcpy(real_pad(i), s->pad[i], PAD_STRIDE);
    float *ax = (float *)(slot + GP_AXES);  // what atrophy::Input_Update would apply
    ax[0] = s->act[3 * i];
    ax[1] = 0;
    ax[2] = s->act[3 * i + 1];
    ax[3] = s->act[3 * i + 2];
    gp_setjoy(slot, 0, ax[0], -ax[1]);
    gp_setjoy(slot, 1, ax[2], -ax[3]);
  }
  for (int i = 0; i < 2; i++) update_pad(i, g_sync_dt);  // as Common::Input::Update does
  if (g_no_unlock) *(char *)P(A_LEVEL5_UNLOCKED) = 1;
  g_in_shadow = 1;
  ps_update(s->ps);
  g_in_shadow = 0;
  for (int i = 0; i < 2; i++) {
    memcpy(s->slot[i], SLOT(i), GP_STRIDE);
    memcpy(s->pad[i], real_pad(i), PAD_STRIDE);
  }
  s->frame++;
}


// ---------------------------------------------------------------- raw dumps (oracle for sim/)

// Box2D 2.1.2 as built into the game (b2_maxPolygonVertices = 9): object sizes and list links
#define B2_WORLD_HDR     0x19298  // b2World fields after the allocators (flags .. continuousPhysics)
#define B2_WORLD_HDR_LEN 0xd0
#define B2_WORLD_CM      0x192a0  // b2ContactManager (contact list +0x58, count +0x60)
#define B2_WORLD_BODIES  0x19320
#define B2_WORLD_JOINTS  0x19328
#define B2_BODY_SIZE     0xc0
#define B2_FIXTURE_SIZE  0x50
#define B2_CONTACT_SIZE  0xb8
#define A_B2_DEFAULT_FILTER   0x100d0f808ULL  // b2_defaultFilter
#define A_B2_DEFAULT_LISTENER 0x100d0f810ULL  // b2_defaultListener
// game objects (sizes from sim/src/game/layout.h; 0 = not dumped yet)
#define DUMP_PLAYER_SIZE 0x240
#define DUMP_POLE_SIZE   0x78
#define DUMP_BALL_SIZE   0x48
#define DUMP_STATS_SIZE  0x34

typedef struct { char *p; size_t n, cap; } Buf;
static void bputs(Buf *b, const char *s) {
  size_t l = strlen(s);
  if (b->n + l + 1 > b->cap) {
    b->cap = (b->n + l + 1) * 2;
    b->p = realloc(b->p, b->cap);
  }
  memcpy(b->p + b->n, s, l + 1);
  b->n += l;
}
static void bprintf(Buf *b, const char *fmt, ...) {
  char tmp[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tmp, sizeof tmp, fmt, ap);
  va_end(ap);
  bputs(b, tmp);
}
static void bhex(Buf *b, const void *mem, size_t len) {
  static const char H[] = "0123456789abcdef";
  if (b->n + 2 * len + 3 > b->cap) {
    b->cap = (b->n + 2 * len + 3) * 2;
    b->p = realloc(b->p, b->cap);
  }
  const uint8_t *m = mem;
  b->p[b->n++] = '"';
  for (size_t i = 0; i < len; i++) b->p[b->n++] = H[m[i] >> 4], b->p[b->n++] = H[m[i] & 15];
  b->p[b->n++] = '"';
  b->p[b->n] = 0;
}
static size_t b2_joint_size(int type) {
  //  unknown revolute prismatic distance pulley mouse gear line weld friction
  static const size_t sz[] = {0, 0x108, 0x138, 0xd8, 0x108, 0xe8, 0x118, 0x118, 0xe8, 0xd8};
  return type > 0 && type < 10 ? sz[type] : 0;
}
static size_t b2_shape_size(int type) { return type == 0 ? 0x18 : type == 1 ? 0xb0 : 0; }

static void dump_obj(Buf *b, const char *name, const char *ptr, size_t len, int *first) {
  if (!ptr || !len) return;
  bprintf(b, "%s\"%s\":{\"a\":\"%p\",\"raw\":", *first ? "" : ",", name, ptr);
  bhex(b, ptr, len);
  bputs(b, "}");
  *first = 0;
}

static char *dump_shadow(Shadow *s) {
  Buf b = {0};
  char *ps = s->ps, *w = *(char **)(ps + PS_WORLD);
  bprintf(&b, "{\"frame\":%lu,\"match_id\":%lu,\"world\":{\"a\":\"%p\",\"hdr\":", s->frame, s->match_id, w);
  bhex(&b, w + B2_WORLD_HDR, B2_WORLD_HDR_LEN);
  char *cm = w + B2_WORLD_CM;  // broadphase dynamic tree (node array) and pending-move buffer
  bputs(&b, ",\"tree\":");
  bhex(&b, *(char **)(cm + 0x8), (size_t)*(int *)(cm + 0x14) * 0x28);
  bputs(&b, ",\"moves\":");
  bhex(&b, *(char **)(cm + 0x30), (size_t)*(int *)(cm + 0x3c) * 4);
  bputs(&b, "},\"bodies\":[");
  int nb = 0;
  for (char *body = *(char **)(w + B2_WORLD_BODIES); body; body = *(char **)(body + 0x70)) {
    bprintf(&b, "%s{\"a\":\"%p\",\"raw\":", nb++ ? "," : "", body);
    bhex(&b, body, B2_BODY_SIZE);
    bputs(&b, ",\"fixtures\":[");
    int nf = 0;
    for (char *f = *(char **)(body + 0x78); f; f = *(char **)(f + 0x18)) {
      char *shape = *(char **)(f + 0x28);
      int st = *(int *)(shape + 8);
      bprintf(&b, "%s{\"a\":\"%p\",\"raw\":", nf++ ? "," : "", f);
      bhex(&b, f, B2_FIXTURE_SIZE);
      bprintf(&b, ",\"shape_type\":%d,\"shape\":", st);
      bhex(&b, shape, b2_shape_size(st));
      bputs(&b, "}");
    }
    bputs(&b, "]}");
  }
  bputs(&b, "],\"joints\":[");
  int nj = 0;
  for (char *j = *(char **)(w + B2_WORLD_JOINTS); j; j = *(char **)(j + 0x18)) {
    int t = *(int *)(j + 8);
    bprintf(&b, "%s{\"a\":\"%p\",\"type\":%d,\"raw\":", nj++ ? "," : "", j, t);
    bhex(&b, j, b2_joint_size(t));
    bputs(&b, "}");
  }
  bputs(&b, "],\"contacts\":[");
  int nc = 0;
  for (char *c = *(char **)(w + B2_WORLD_CM + 0x58); c; c = *(char **)(c + 0x18)) {
    bprintf(&b, "%s{\"a\":\"%p\",\"raw\":", nc++ ? "," : "", c);
    bhex(&b, c, B2_CONTACT_SIZE);
    bputs(&b, "}");
  }
  bputs(&b, "],\"objects\":{");
  int first = 1;
  dump_obj(&b, "playstate", ps, PS_SIZE, &first);
  for (int i = 0; i < 4; i++) {
    char *pl = ((char **)(ps + PS_PLAYERS))[i], name[16];
    if (!pl) continue;
    snprintf(name, sizeof name, "player%d", i);
    dump_obj(&b, name, pl, DUMP_PLAYER_SIZE, &first);
    snprintf(name, sizeof name, "stats%d", i);
    dump_obj(&b, name, *(char **)(pl + PL_STATS), DUMP_STATS_SIZE, &first);
    snprintf(name, sizeof name, "pole%d", i);
    dump_obj(&b, name, *(char **)(pl + PL_POLE), DUMP_POLE_SIZE, &first);
  }
  dump_obj(&b, "ball", *(char **)(ps + PS_BALL), DUMP_BALL_SIZE, &first);
  for (int i = 0; i < 2; i++) {  // the shadow's private input state (atrophy::Gamepad, Common::Input pad)
    char name[16];
    snprintf(name, sizeof name, "gamepad%d", i);
    dump_obj(&b, name, (char *)s->slot[i], GP_STRIDE, &first);
    snprintf(name, sizeof name, "pad%d", i);
    dump_obj(&b, name, (char *)s->pad[i], PAD_STRIDE, &first);
  }
  bputs(&b, "}}");
  return b.p;
}

static char *farm_snapshots(void) {
  size_t cap = 64 + (size_t)g_nsh * 8192, n = 0;
  char *out = malloc(cap);
  n += snprintf(out + n, cap - n, "{\"matches\":[");
  for (int k = 0; k < g_nsh; k++) {
    if (k) out[n++] = ',';
    n += snapshot_ps(g_sh[k].ps, out + n, cap - n, g_sh[k].frame, g_sh[k].match_id, 0);
  }
  n += snprintf(out + n, cap - n, "]}");
  return out;
}

static void farm_run_job(void) {
  // Shadow ticks overwrite the engine's pad state for pads 0/1: keep the real match's copy safe.
  static uint8_t real_slot[2][GP_STRIDE], real_padv[2][PAD_STRIDE];
  for (int i = 0; i < 2; i++) {
    memcpy(real_slot[i], SLOT(i), GP_STRIDE);
    memcpy(real_padv[i], real_pad(i), PAD_STRIDE);
  }
  char *out = NULL, buf[128];
  switch (g_job.type) {
    case JOB_CREATE:
      for (int i = 0; i < g_job.n && g_nsh < MAX_SHADOWS; i++) sh_create(&g_sh[g_nsh++]);
      snprintf(buf, sizeof buf, "{\"ok\":true,\"n\":%d}", g_nsh);
      out = strdup(buf);
      break;
    case JOB_RESET:
      if (g_job.k >= 0 && g_job.k < g_nsh) {
        sh_destroy(&g_sh[g_job.k]);
        sh_create(&g_sh[g_job.k]);
        Shadow *sh = &g_sh[g_job.k];
        if (g_job.clock > 0) {
          *(float *)(sh->ps + PS_TIME_LEFT) = g_job.clock;
          sh->ps[PS_TIMER_ON] = 1;
        }
        out = malloc(16384);  // reply: the new match's tick-0 snapshot
        snapshot_ps(sh->ps, out, 16384, sh->frame, sh->match_id, 0);
        break;
      } else {
        snprintf(buf, sizeof buf, "{\"error\":\"no shadow %d\"}", g_job.k);
      }
      out = strdup(buf);
      break;
    case JOB_DUMP:
      if (g_job.k >= 0 && g_job.k < g_nsh) {
        out = dump_shadow(&g_sh[g_job.k]);
        break;
      }
      snprintf(buf, sizeof buf, "{\"error\":\"no shadow %d\"}", g_job.k);
      out = strdup(buf);
      break;
    case JOB_PHYSSTEP:
      if (g_job.k >= 0 && g_job.k < g_nsh) {
        char *w = *(char **)(g_sh[g_job.k].ps + PS_WORLD);
        void **filter = (void **)(w + B2_WORLD_CM + 0x68), **listener = (void **)(w + B2_WORLD_CM + 0x70);
        void *f0 = *filter, *l0 = *listener;
        *filter = (void *)P(A_B2_DEFAULT_FILTER), *listener = (void *)P(A_B2_DEFAULT_LISTENER);
        for (int t = 0; t < g_job.n; t++) g_orig_b2step(w, 1.0f / 60.0f, 60, 18);
        *filter = f0, *listener = l0;
        g_sh[g_job.k].frame += g_job.n;
        out = strdup("{\"ok\":true}");
        break;
      }
      snprintf(buf, sizeof buf, "{\"error\":\"no shadow %d\"}", g_job.k);
      out = strdup(buf);
      break;
    case JOB_DESTROY:
      for (int k = 0; k < g_nsh; k++) sh_destroy(&g_sh[k]);
      g_nsh = 0;
      out = strdup("{\"ok\":true}");
      break;
    case JOB_ACT: {
      double t0 = now_s();
      for (int k = 0; k < g_nsh; k++) {
        memcpy(g_sh[k].act, g_job.acts + 6 * k, sizeof g_sh[k].act);
        for (int t = 0; t < g_job.n; t++) sh_tick(&g_sh[k]);
      }
      double t1 = now_s();
      g_sh_tick_s += t1 - t0;
      g_sh_ticks += (unsigned long)g_nsh * g_job.n;
      out = farm_snapshots();
      g_sh_snap_s += now_s() - t1;
      break;
    }
  }
  for (int i = 0; i < 2; i++) {
    memcpy(SLOT(i), real_slot[i], GP_STRIDE);
    memcpy(real_pad(i), real_padv[i], PAD_STRIDE);
  }
  pthread_mutex_lock(&g_lock);
  if (g_job.type == JOB_ACT) {
    free(g_farm_last);
    g_farm_last = strdup(out);
  }
  g_job.out = out;
  g_job.done = 1;
  pthread_cond_broadcast(&g_farmtest_cv);
  pthread_mutex_unlock(&g_lock);
}

// Called with g_lock held.
static int farm_job_pending(void) { return g_job.type != JOB_NONE && !g_job.done; }
static int farm_active(void) { return g_nsh > 0; }

// ---------------------------------------------------------------- socket side

static void write_all(int fd, const char *s, size_t len) {
  for (size_t off = 0; off < len;) {
    ssize_t w = write(fd, s + off, len - off);
    if (w <= 0) return;
    off += w;
  }
}

static char *farm_submit(int type, int n, int k, float *acts, float clock) {
  pthread_mutex_lock(&g_lock);
  while (g_job.type != JOB_NONE) pthread_cond_wait(&g_farmtest_cv, &g_lock);  // one at a time
  g_job.type = type, g_job.n = n, g_job.k = k, g_job.acts = acts, g_job.out = NULL, g_job.done = 0;
  g_job.clock = clock;
  pthread_cond_broadcast(&g_step_cv);  // wake a game thread parked in the lockstep gate
  while (!g_job.done) pthread_cond_wait(&g_farmtest_cv, &g_lock);
  char *out = g_job.out;
  g_job.type = JOB_NONE;
  pthread_cond_broadcast(&g_farmtest_cv);
  pthread_mutex_unlock(&g_lock);
  return out;
}

// Handles "farm ..." (args = text after "farm "). Writes the reply line itself.
static void farm_command(int fd, char *args) {
  char *out = NULL;
  if (!strncmp(args, "create ", 7)) {
    if (!*(char **)P(A_PLAYSTATE_PTR)) out = strdup("{\"error\":\"start a real match first\"}");
    else out = farm_submit(JOB_CREATE, atoi(args + 7), 0, NULL, 0);
  } else if (!strncmp(args, "act ", 4)) {
    char *p = args + 4, *e;
    int n = (int)strtol(p, &e, 10), want = 6 * g_nsh, got = 0;
    float *acts = calloc(want ? want : 1, sizeof(float));
    for (p = e; got < want; p = e) {
      float v = strtof(p, &e);
      if (e == p) break;
      acts[got++] = clampf(v, -1, 1);
    }
    if (got != want || n < 1) {
      char msg[128];
      snprintf(msg, sizeof msg, "{\"error\":\"act wants n and %d floats, got %d\"}", want, got);
      out = strdup(msg);
    } else {
      out = farm_submit(JOB_ACT, n, 0, acts, 0);
    }
    free(acts);
  } else if (!strncmp(args, "reset ", 6)) {
    char *e;
    int k = (int)strtol(args + 6, &e, 10);
    out = farm_submit(JOB_RESET, 0, k, NULL, strtof(e, NULL));
  } else if (!strncmp(args, "dump ", 5)) {
    out = farm_submit(JOB_DUMP, 0, atoi(args + 5), NULL, 0);
  } else if (!strncmp(args, "physstep ", 9)) {
    char *e;
    int k = (int)strtol(args + 9, &e, 10);
    out = farm_submit(JOB_PHYSSTEP, atoi(e), k, NULL, 0);
  } else if (!strcmp(args, "destroy")) {
    out = farm_submit(JOB_DESTROY, 0, 0, NULL, 0);
  } else if (!strcmp(args, "state")) {
    pthread_mutex_lock(&g_lock);
    out = strdup(g_farm_last ? g_farm_last : "{\"matches\":[]}");
    pthread_mutex_unlock(&g_lock);
  } else if (!strcmp(args, "info")) {
    char msg[320];
    double tk = g_sh_ticks ? (double)g_sh_ticks : 1.0;
    snprintf(msg, sizeof msg,
             "{\"shadows\":%d,\"ticks\":%lu,\"us_per_tick\":%.1f,\"us_box2d_step_per_tick\":%.1f,"
             "\"us_game_logic_per_tick\":%.1f,\"us_snapshot_json_per_tick\":%.1f}",
             g_nsh, g_sh_ticks, g_sh_tick_s * 1e6 / tk, g_sh_step_s * 1e6 / tk,
             (g_sh_tick_s - g_sh_step_s) * 1e6 / tk, g_sh_snap_s * 1e6 / tk);
    out = strdup(msg);
  } else {
    out = strdup("{\"error\":\"farm create|act|reset|dump|physstep|destroy|state|info\"}");
  }
  write_all(fd, out, strlen(out));
  write_all(fd, "\n", 1);
  free(out);
}

static void farm_init(void) {
  // push rbp; mov rbp,rsp; sub rsp,0x40; mov [rbp-8],rdi; movss [rbp-0xc],xmm0
  static const uint8_t step_pro[] = {0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x40, 0x48,
                                     0x89, 0x7d, 0xf8, 0xf3, 0x0f, 0x11, 0x45, 0xf4};
  g_orig_b2step = hook(P(0x100174770ULL), hooked_b2step, step_pro, sizeof step_pro);  // b2World::Step
  if (!g_orig_b2step) fprintf(stderr, "sprbridge: b2World::Step hook failed (profiling off)\n");
  gp_setjoy = (void *)P(A_GP_SETJOY);
  update_pad = (void *)P(A_UPDATEPAD);
}
