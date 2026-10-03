// C ABI of the batched native env (build/libsprsim.dylib), used by bridge/native_env.py via ctypes.
// Output buffers are owned by the env and overwritten in place by every reset_all/step (zero-copy
// numpy views on the Python side).
#include <cstring>

#include "batch_env.h"

using sim::env::BatchEnv;
using sim::env::EnvConfig;

extern "C" {

struct SprEnvBuffers {  // pointers into the env's buffers (N = n_hosts * per_host)
  float *obs;           // [N][2][62]
  float *final_obs;     // [N][2][62]
  float *rew;           // [N][2]
  float *terms;         // [N][2][2]
  uint8_t *reset, *trunc, *sd, *sd_start;  // [N]
  int32_t *score;       // [N][2]
};

// weights: goal, kill, ball_progress, position; heads: session head numbers of players 1, 2
void *sprenv_create(int n_hosts, int per_host, int frame_skip, double sd_frac, int max_steps,
                    unsigned long long seed, int threads, const double *weights, int level, const int *heads,
                    float sd_clock) {
  EnvConfig c;
  c.n_hosts = n_hosts, c.per_host = per_host, c.frame_skip = frame_skip, c.sd_frac = sd_frac;
  c.max_steps = max_steps, c.seed = seed, c.threads = threads, c.level = level, c.sd_clock = sd_clock;
  if (weights) c.w = {weights[0], weights[1], weights[2], weights[3]};
  if (heads) c.head[0] = heads[0], c.head[1] = heads[1];
  if (n_hosts < 1 || per_host < 1 || frame_skip < 1) return nullptr;
  return new BatchEnv(c);
}

void sprenv_destroy(void *e) { delete static_cast<BatchEnv *>(e); }
int sprenv_n(void *e) { return static_cast<BatchEnv *>(e)->N(); }
int sprenv_obs_size() { return sim::env::kObsSize; }

void sprenv_buffers(void *e, SprEnvBuffers *out) {
  auto &b = static_cast<BatchEnv *>(e)->Buffers();
  *out = {b.obs.data(), b.final_obs.data(), b.rew.data(), b.terms.data(), b.reset.data(),
          b.trunc.data(), b.sd.data(), b.sd_start.data(), b.score.data()};
}

void sprenv_reset_all(void *e) { static_cast<BatchEnv *>(e)->ResetAll(); }
void sprenv_step(void *e, const float *actions) { static_cast<BatchEnv *>(e)->Step(actions); }
void sprenv_set_record(void *e, int on) { static_cast<BatchEnv *>(e)->SetRecord(on != 0); }

// Copy a string into buf (cap bytes incl. NUL); returns the full length (call again if >= cap).
static size_t out_str(const std::string &s, char *buf, size_t cap) {
  if (buf && cap) {
    size_t n = s.size() < cap - 1 ? s.size() : cap - 1;
    memcpy(buf, s.data(), n);
    buf[n] = 0;
  }
  return s.size();
}

// which: 0 = current state, 1 = post-step state of the last step (recording), 2 = the fresh match of
// the last reset (recording), 3 = raw oracle-format dump (Match::Dump, `farm dump`)
size_t sprenv_json(void *e, int k, int which, char *buf, size_t cap) {
  auto *env = static_cast<BatchEnv *>(e);
  if (k < 0 || k >= env->N()) return out_str("", buf, cap);
  switch (which) {
    case 1: return out_str(env->StepJson(k), buf, cap);
    case 2: return out_str(env->ResetJson(k), buf, cap);
    case 3: return out_str(env->GetMatch(k)->Dump(), buf, cap);
    default: return out_str(env->CurrentJson(k), buf, cap);
  }
}

// `farm state`-style reply for matches [k0, k1): {"matches":[snapshot, ...]}
size_t sprenv_state_json(void *e, int k0, int k1, char *buf, size_t cap) {
  auto *env = static_cast<BatchEnv *>(e);
  std::string s = "{\"matches\":[";
  for (int k = k0; k < k1 && k < env->N(); k++) {
    if (k > k0) s += ",";
    s += env->CurrentJson(k);
  }
  s += "]}";
  return out_str(s, buf, cap);
}

size_t sprenv_level_json(void *e, char *buf, size_t cap) {
  return out_str(static_cast<BatchEnv *>(e)->LevelJson(), buf, cap);
}

// farm_env.Terrain queries on the env's terrain: floor(x, y); above(x, y) (returns 0 for None)
double sprenv_terrain_floor(void *e, double x, double y) {
  return static_cast<BatchEnv *>(e)->GetTerrain().Floor(x, y);
}
int sprenv_terrain_above(void *e, double x, double y, double *out) {
  return static_cast<BatchEnv *>(e)->GetTerrain().Above(x, y, out);
}
// column i of the terrain surface table (Terrain.surf[i]); returns its length
int sprenv_terrain_column(void *e, int i, double *out, int cap) {
  const auto &t = static_cast<BatchEnv *>(e)->GetTerrain();
  if (i < 0 || i >= t.Columns()) return -1;
  int n;
  const double *c = t.Column(i, &n);
  for (int j = 0; j < n && j < cap; j++) out[j] = c[j];
  return n;
}

// fake-bridge backend (tests): `farm act n <sticks>` and `farm reset k [clock]` on the env's matches
void sprenv_raw_act(void *e, const float *sticks, int n) { static_cast<BatchEnv *>(e)->RawAct(sticks, n); }
size_t sprenv_raw_reset(void *e, int k, float clock, char *buf, size_t cap) {
  auto *env = static_cast<BatchEnv *>(e);
  if (k < 0 || k >= env->N()) return out_str("", buf, cap);
  return out_str(env->RawReset(k, clock), buf, cap);
}

// PoleRidersEnv._stick + "%.4f" + bridge strtof: the stick floats the game receives (tests)
void sprenv_stick(int agent, const float *act, float *out) { sim::env::StickFromAction(agent, act, out); }

// training curriculum: drop riders into the enemy pit (pole up) on a fraction of non-sd resets
void sprenv_set_pit(void *e, double frac, float x_px, float y_px) {
  static_cast<BatchEnv *>(e)->SetPit(frac, x_px, y_px);
}
// bitmask of riders dropped into the enemy pit at match k's last reset (1 = player 1, 2 = player 2)
int sprenv_pit_start(void *e, int k) { return static_cast<BatchEnv *>(e)->PitStart(k); }
// training rule: end a match that is tied when the clock runs out (no sudden death)
void sprenv_set_end_on_sd(void *e, int on) { static_cast<BatchEnv *>(e)->SetEndOnSuddenDeath(on != 0); }
void sprenv_set_scramble(void *e, int on) { static_cast<BatchEnv *>(e)->SetScramble(on != 0); }
void sprenv_set_death_weight(void *e, double w) { static_cast<BatchEnv *>(e)->SetDeathWeight(w); }
void sprenv_set_rotation_weight(void *e, double w) { static_cast<BatchEnv *>(e)->SetRotationWeight(w); }
void sprenv_set_random_pole(void *e, int on) { static_cast<BatchEnv *>(e)->SetRandomPole(on != 0); }
// training reward rule (GoalRefundState): no non-goal reward between a goal and the respawn, deaths refunded
// / kills taken back when the victim's team scores while it is dead. Off unless set (Python turns it on).
void sprenv_set_goal_refund(void *e, int on) { static_cast<BatchEnv *>(e)->SetGoalRefund(on != 0); }
// ball touch bookkeeping buffers float[N][2][sprenv_touch_size()] (see EnvBuffers::touch)
int sprenv_touch_size() { return sim::env::kTouchSize; }
void sprenv_touch_buffers(void *e, float **touch, float **touch_final) {
  auto &b = static_cast<BatchEnv *>(e)->Buffers();
  *touch = b.touch.data(), *touch_final = b.touch_final.data();
}
// tests only: scripted events that change match k (BatchEnv::DebugEvent); 0 = done
int sprenv_debug_event(void *e, int k, int kind, int who) {
  return static_cast<BatchEnv *>(e)->DebugEvent(k, kind, who);
}
void sprenv_set_goal_spawn(void *e, double p) { static_cast<BatchEnv *>(e)->SetGoalSpawn(p); }
void sprenv_set_format(void *e, int max_score, float round_time) {
  static_cast<BatchEnv *>(e)->SetFormat(max_score, round_time);
}
// training-only: keep riders of match k (bitmask 1 = player 1, 2 = player 2) permanently ghosted
void sprenv_set_ghost_lock(void *e, int k, int mask) { static_cast<BatchEnv *>(e)->SetGhostLock(k, mask); }
void sprenv_set_legacy_obs(void *e, int k, int mask) { static_cast<BatchEnv *>(e)->SetLegacyObs(k, mask); }

// ---- snapshot / restore + planning branches (2026-10-02; BatchEnv::Snapshot / PlanBegin docs)
int sprenv_snapshot(void *e, int k, int slot) { return static_cast<BatchEnv *>(e)->Snapshot(k, slot); }
int sprenv_restore(void *e, int k, int slot) { return static_cast<BatchEnv *>(e)->Restore(k, slot); }
int sprenv_max_snapshots() { return BatchEnv::kMaxSnapshots; }
size_t sprenv_arena_used(void *e, int k) { return static_cast<BatchEnv *>(e)->ArenaUsed(k); }
// debug: game objects of match k (b >= 0: of planning branch b instead) outside its arena (0 = ok)
int sprenv_check_arena(void *e, int k, int b) {
  auto *env = static_cast<BatchEnv *>(e);
  return b >= 0 ? env->PlanCheckArena(b) : env->CheckArena(k);
}
// debug: allocations routed to arenas / made by game code of matches without an arena (process-wide)
void sprenv_arena_counters(unsigned long long *out) {
  sim::ArenaCounters c = sim::GetArenaCounters();
  out[0] = c.arena_allocs, out[1] = c.heap_allocs;
}

struct SprPlanBuffers {  // B = branches
  float *obs;            // [B][2][62]
  float *rew;            // [B][2]
  float *terms;          // [B][2][2]
  uint8_t *done, *trunc; // [B]
  int32_t *score;        // [B][2]
  float *touch;          // [B][2][touch_size]
};
int sprenv_plan_init(void *e, const int *owner, int n) { return static_cast<BatchEnv *>(e)->PlanInit(owner, n); }
void sprenv_plan_buffers(void *e, SprPlanBuffers *out) {
  auto &p = static_cast<BatchEnv *>(e)->PlanBufs();
  *out = {p.obs.data(), p.rew.data(), p.terms.data(), p.done.data(), p.trunc.data(), p.score.data(), p.touch.data()};
}
int sprenv_plan_begin(void *e, const uint8_t *active) { return static_cast<BatchEnv *>(e)->PlanBegin(active); }
void sprenv_plan_step(void *e, const float *actions) { static_cast<BatchEnv *>(e)->PlanStep(actions); }
void sprenv_plan_end(void *e) { static_cast<BatchEnv *>(e)->PlanEnd(); }
size_t sprenv_plan_json(void *e, int b, char *buf, size_t cap) {
  auto *env = static_cast<BatchEnv *>(e);
  if (b < 0 || b >= env->PlanN()) return out_str("", buf, cap);
  return out_str(env->PlanJson(b), buf, cap);
}

}  // extern "C"
