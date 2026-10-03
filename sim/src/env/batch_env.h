// Batched native RL environment: N native matches stepped on a persistent thread pool, with exactly
// the per-step semantics of bridge/farm_worker.py host_worker (+ farm_env.FarmHost): actions ->
// PoleRidersEnv._stick -> `farm act frame_skip` -> compute_reward / observe2 / done / truncation ->
// auto-reset (sudden-death start with probability sd_frac via clock 0.05 = `farm reset k 0.05`).
//
// Matches are grouped into "hosts" of `per_host` matches (match m belongs to host m / per_host) only
// for the reset RNG: host h draws its sudden-death coin flips from Python random.Random(seed + h) in
// match order, like FarmHost(seed=h) — so with the same actions the native env reproduces a
// farm_worker host exactly.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../arena.h"
#include "../match.h"
#include "features.h"
#include "pyrandom.h"

namespace sim {
namespace env {

class ThreadPool {
 public:
  explicit ThreadPool(int threads);  // threads-1 workers + the calling thread
  ~ThreadPool();
  // f(i) for i in [0, n), dynamically balanced; returns when all are done
  void Run(int n, const std::function<void(int)> &f);
  int Size() const { return static_cast<int>(workers_.size()) + 1; }

 private:
  void Work(const std::function<void(int)> *f, int n);
  void Loop();
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int)> *job_ = nullptr;
  int n_ = 0, gen_ = 0, busy_ = 0;
  bool stop_ = false;
  std::atomic<int> next_{0};
};

struct EnvConfig {
  int n_hosts = 1, per_host = 16;
  int frame_skip = 4;
  double sd_frac = 0.0;
  int max_steps = 5400;
  uint64_t seed = 0;  // host h: random.Random(seed + h)
  int threads = 0;    // 0 = hardware concurrency
  RewardWeights w;
  int level = 1;          // PlayState level id (1 = the training map, menu "arena 2")
  int head[2] = {0, 0};   // session head numbers (7 = short pole)
  float sd_clock = 0.05f; // `farm reset k 0.05`
  // Training curriculum (not in the game): a fraction of non-sudden-death resets drop rider(s) into
  // the ENEMY pit, pole pointing up, ball in the middle as usual. Which riders: player 1 only,
  // player 2 only, or both (uniform). Uses its own RNG, so the sd-start stream stays FarmHost's.
  double pit_frac = 0.0;
  bool scramble = false;  // see BatchEnv::SetScramble
  double goal_spawn = 0.0;  // see BatchEnv::SetGoalSpawn
  bool random_pole = false;  // see BatchEnv::SetRandomPole
  bool end_on_sd = false;  // training rule: a tie at the end of time ends the match (no sudden death)
  bool goal_refund = false;  // training reward rule, see GoalRefundState (off here = bit-identical to before)
  float pit_x_px = 1152.0f;  // |x| of the drop (18 m: under open sky, the ledge starts at 20.5 m)
  float pit_y_px = 46.0f;    // Respawn y (body 50 px above it; the lowest pole segment clears the floor)
};

// Output buffers, laid out like farm_worker's `out` dict (host-major == match order)
struct EnvBuffers {
  std::vector<float> obs, final_obs;  // [N][2][62]
  std::vector<float> rew;             // [N][2]
  std::vector<float> terms;           // [N][2][2] (goal, kill)
  std::vector<uint8_t> reset, trunc, sd, sd_start;  // [N] (numpy bool)
  std::vector<int32_t> score;         // [N][2]
  // Ball touches (read-only bookkeeping, not observed by the policy), per match and rider (player 1, 2):
  // [0] seconds since the rider last touched the ball (a touching contact between a non-sensor fixture of its
  //     body or pole and the ball after a tick; 0 = in the step's last tick; kTouchNever = never this match)
  // [1] what touched then: bit 1 = body (hull / foot), bit 2 = pole (segments tagged BODY_POLE, or the hammer),
  //     bit 4 = a pole handle segment: the game tags segments 0-2 with the rider itself (Player::index /
  //     Pole::ownerIndex), so its contact handlers treat a ball contact there exactly like the body's (it kicks)
  // [2] seconds since the rider's last ball KICK (the game's kick impulse, PlayState::BallPlayerCollision /
  //     HammerBallCollision, seen as the rider's ballNoCollideTimer jumping up), kTouchNever = never
  // [3] touching the ball in the step's last tick (0/1)
  // [4] the game's own PlayState::lastTouch (goal / own-goal credit) is this rider (0/1)
  // `touch` goes with `obs` (a reset match: never touched), `touch_final` with `final_obs`.
  std::vector<float> touch, touch_final;  // [N][2][kTouchSize]
};
constexpr int kTouchSize = 5;
constexpr float kTouchNever = 999.0f;

// Env-side state of one match (trivially copyable: saved with the match's arena in snapshots)
struct SimState {
  StateV prev;
  int steps = 0;
  unsigned long match_id = 0;
  uint64_t pit_rng = 0;  // splitmix64 state for pit drops
  int pit_start = 0;     // bitmask of riders dropped into the enemy pit at the last reset
  bool scramble_wait = false;  // training-only: a goal/kill happened, re-spawn at random when the game allows
  uint64_t scr_rng = 0;        // splitmix64 state for the random re-spawns
  uint64_t resets = 0;         // matches started in this slot (seeds the random pole angles)
  GoalRefundState gr;          // EnvConfig::goal_refund
  // ball touch bookkeeping (TrackTouch), in ticks
  int touch_age[2] = {0, 0}, kick_age[2] = {0, 0};
  uint8_t touch_what[2] = {0, 0};
  bool touching[2] = {false, false};
  float prev_nc[2] = {0.0f, 0.0f};  // ballNoCollideTimer after the previous tick
};

// One native match in its own arena + its env-side state: a match of the batch, or a planning branch
// (a replica of match `k`).
struct MatchSim {
  Arena *arena = nullptr;
  Match *m = nullptr;
  SimState st;
  int k = 0;  // the batch match whose RNG streams it uses (seeds)
};

// Where one match's step / reset writes its outputs (a row of EnvBuffers, or of the planning buffers)
struct SimOut {
  float *obs, *final_obs, *rew, *terms, *touch, *touch_final;
  uint8_t *reset, *trunc, *sd, *sd_start;
  int32_t *score;
};

// Planning outputs, one row per branch (see BatchEnv::PlanBegin)
struct PlanBuffers {
  std::vector<float> obs;      // [B][2][62] the branch's current observation (after its last step)
  std::vector<float> rew;      // [B][2]
  std::vector<float> terms;    // [B][2][2]
  std::vector<uint8_t> done;   // [B] the branch's match ended in its last step (or earlier): frozen since
  std::vector<uint8_t> trunc;  // [B] ... by the max_steps cap (not a real terminal)
  std::vector<int32_t> score;  // [B][2]
  std::vector<float> touch;    // [B][2][kTouchSize]
  // scratch (written by the shared step code, not exposed)
  std::vector<float> s_obs, s_touch;
  std::vector<uint8_t> s_reset, s_sd, s_sd_start;
};

class BatchEnv {
 public:
  explicit BatchEnv(const EnvConfig &cfg);
  ~BatchEnv();
  int N() const { return n_; }
  EnvBuffers &Buffers() { return buf_; }
  const EnvConfig &Config() const { return cfg_; }

  void ResetAll();                   // farm_worker boot(): obs, sd_start
  void Step(const float *actions);   // actions float32[N][2][3]; fills every buffer

  // Debug / tests: when recording, Step() keeps the snapshot JSON (bridge format) of each match's
  // post-step state and, for matches that were reset, of the fresh match.
  void SetRecord(bool on) { record_ = on; }
  const std::string &StepJson(int k) const { return slots_[k].step_json; }
  const std::string &ResetJson(int k) const { return slots_[k].reset_json; }
  std::string CurrentJson(int k) const;
  std::string LevelJson() const;
  const Terrain &GetTerrain() const { return terrain_; }
  // Bare bridge-style control (a fake `farm act` / `farm reset k [clock]` backend for tests):
  // sticks[N][6] = lx0 rx0 ry0 lx1 rx1 ry1 as the game receives them, n ticks per match.
  void RawAct(const float *sticks, int n);
  std::string RawReset(int k, float clock);  // returns the fresh match's snapshot JSON
  Match *GetMatch(int k) { return slots_[k].sim.m; }

  // ---- snapshot / restore (per match; the match's arena + its env-side state)
  // Snapshot match k into its slot `slot` (0 .. kMaxSnapshots-1); restore puts match k back exactly into that
  // state (same trajectories for the same actions as if it had never left it) and rewrites its rows of obs and
  // touch (the other buffers keep the last step's values). Not part of a snapshot: the host's sudden-death coin
  // stream (EnvConfig::sd_frac; shared by the host's matches), config switches, ghost lock / legacy obs masks.
  // Returns 0, or < 0: -1 bad k / slot, -2 nothing saved in that slot, -3 the arena overflowed (never seen).
  static constexpr int kMaxSnapshots = 16;
  int Snapshot(int k, int slot);
  int Restore(int k, int slot);
  size_t ArenaUsed(int k) const { return slots_[k].sim.arena ? slots_[k].sim.arena->Used() : 0; }
  // debug: number of the match's game objects (PlayState, players, poles, ball, Box2D bodies / fixtures / shapes /
  // joints / contacts ...) that are NOT inside its arena (must be 0: else snapshots would miss them)
  int CheckArena(int k) const { return CheckSim(slots_[k].sim); }

  // ---- planning pool: B branch matches, branch b a clone of match owner[b]
  // Branches are replicas kept in step with their match by deterministic replay of its log (the actions and
  // test events since its last reset): PlanBegin brings every active branch up to its match's current state
  // (re-playing only what it has not seen yet: ~1 step per decision; a full replay after the match was
  // reset / restored), saves it as the branch's root, and writes the root observation; PlanStep steps every
  // active, unfinished branch one decision (no auto-reset: a branch whose match ends is frozen, done = 1);
  // PlanEnd restores every branch to its root (the next PlanBegin then only needs the match's new steps).
  // Branch steps use the env's reward weights / rules, like Step. PlanBegin returns 0, or -1 if a branch's
  // match cannot be replayed (after RawAct / RawReset; fixed by the match's next reset). PlanInit: 0, -1 bad
  // owner, -2 out of arenas.
  int PlanInit(const int *owner, int n_branches);
  int PlanN() const { return static_cast<int>(branches_.size()); }
  PlanBuffers &PlanBufs() { return pbuf_; }
  int PlanBegin(const uint8_t *active);  // active[B] (nullptr = all)
  void PlanStep(const float *actions);   // float32[B][2][3]; inactive branches are left alone
  void PlanEnd();
  std::string PlanJson(int b) const;
  int PlanCheckArena(int b) const { return CheckSim(branches_[b].sim); }

 private:
  // log of what happened to a match since its last reset (replayed by its planning branches)
  struct Op {
    uint8_t kind;   // 0 = Step with act, 1 = DebugEvent(a, b)
    uint8_t ghost;  // ghost-lock mask during the step
    int8_t a, b;
    float act[6];
  };
  struct History {
    uint64_t lineage = 0;  // changes whenever the log restarts (reset) or is replaced (restore, raw control)
    bool valid = false;    // replayable (false after RawAct / RawReset until the next reset)
    SimState reset_state;  // the env-side state right before the reset (its RNG streams)
    bool reset_sd = false;
    unsigned long reset_match_id = 0;
    // the log: ops[0, len) of a buffer shared with the snapshots taken in this generation (copy-on-write:
    // appending in place keeps every snapshot's prefix intact; after a restore to an earlier point the next
    // append copies the prefix first)
    std::shared_ptr<std::vector<Op>> ops;
    size_t len = 0;
    void Clear() { ops = std::make_shared<std::vector<Op>>(), len = 0; }
    void Push(const Op &op) {
      if (!ops) ops = std::make_shared<std::vector<Op>>();
      if (ops->size() != len) ops = std::make_shared<std::vector<Op>>(ops->begin(), ops->begin() + len);
      ops->push_back(op), len++;
    }
  };
  struct Snap {
    bool ok = false;
    std::vector<uint8_t> mem;
    Match *m = nullptr;
    SimState st;
    History hist;
  };
  struct Slot {
    MatchSim sim;
    bool need_reset = false, sd_next = false;
    int ghost_lock = 0;    // training-only: bitmask of riders kept in the ghost state (see GhostLock)
    int legacy_obs = 0;    // evaluation-only: bitmask of riders observing dead riders the pre-fix way
    History hist;
    uint64_t next_lineage = 1;
    std::vector<Snap> snaps;
    std::string step_json, reset_json;
  };
  struct Branch {
    MatchSim sim;
    int owner = 0;
    uint64_t lineage = 0;  // History::lineage / ops replayed so far
    size_t pos = 0;
    bool active = false, dirty = false, done = false;
    std::vector<uint8_t> root_mem;
    SimState root_st;
  };
  SimOut RowOut(int k);
  SimOut PlanOut(int b);
  void NewMatch(MatchSim &s, bool sd);  // the fresh match (arena cleared), RNG-dependent spawns
  void ResetSim(MatchSim &s, bool sd, unsigned long match_id, int legacy, const SimOut &o, std::string *json);
  // one decision of match s (ticks, reward, outputs, curricula); returns need_reset
  bool StepSim(MatchSim &s, const float *act6, int ghost_lock, int legacy, const SimOut &o, std::string *json);
  int DebugEventSim(MatchSim &s, int kind, int who);
  void ResetSlot(int k, bool sd);  // FarmHost.reset(k) minus the RNG draw
  void PitDrop(MatchSim &s);       // training curriculum: see EnvConfig::pit_frac
  void GhostLock(MatchSim &s, int mask);
  bool Scramble(MatchSim &s);      // training curriculum: see SetScramble
  void TrackTouch(MatchSim &s);    // after every tick: ball contacts / kicks (reads the match only)
  void ResetTouch(MatchSim &s);
  void WriteTouch(const MatchSim &s, float *dst) const;
  int CheckSim(const MatchSim &s) const;
  void Invalidate(Slot &s);
  bool SyncBranch(Branch &br);
 public:
  void SetPit(double frac, float x_px, float y_px) { cfg_.pit_frac = frac, cfg_.pit_x_px = x_px, cfg_.pit_y_px = y_px; }
  void SetEndOnSuddenDeath(bool on) { cfg_.end_on_sd = on; }
  // match format (the game's menu settings "score limit" / "duration"): applies from each match's next reset
  void SetFormat(int max_score, float round_time) { engine0_.cfg.maxScore = max_score, engine0_.cfg.roundTime = round_time; }
  int PitStart(int k) const { return slots_[k].sim.st.pit_start; }
  void SetGhostLock(int k, int mask) { slots_[k].ghost_lock = mask; }  // persists across resets
  void SetLegacyObs(int k, int mask) { slots_[k].legacy_obs = mask; }  // persists across resets
  // Training-only curriculum: after every goal and every kill, both riders re-spawn at random ground positions
  // (pole up) and the ball at a random point of the zipline — right away after a kill (the victim doesn't wait
  // out its death timer), after a goal once the game has done its own ball reset.
  void SetScramble(bool on) { cfg_.scramble = on; }
  // the same re-spawn, but only after goals (never kills), with probability p per goal (0 = off)
  void SetGoalSpawn(double p) { cfg_.goal_spawn = p; }
  void SetDeathWeight(double w) { cfg_.w.death = w; }  // see RewardWeights::death
  void SetRotationWeight(double w) { cfg_.w.rotation = w; }  // see RewardWeights::rotation (may change any time)
  void SetRandomPole(bool on) { cfg_.random_pole = on; }  // see EngineContext::randomPole
  void SetGoalRefund(bool on) { cfg_.goal_refund = on; }  // see GoalRefundState (applies from the next step)
  // Tests only (scripted scenarios; changes the match): kind 1 = rider `who` (0/1) is killed by the other one
  // (the bookkeeping of PlayState::PlayerCollision's kill branch: OffPole, death/kill stats, 3 s respawn
  // timer; no impulses), kind 2 = the ball is put against the goal wall team `who` attacks, moving into it,
  // so the game itself scores it within a tick or two. Returns 0 on success.
  int DebugEvent(int k, int kind, int who);
 private:
  void Observe(const StateV &s, float *dst, int legacy) const;

  EnvConfig cfg_;
  int n_;
  EngineContext engine0_;  // a fresh shadow's engine state (idle pads + session records)
  Terrain terrain_;
  std::vector<Slot> slots_;
  std::vector<Branch> branches_;
  PlanBuffers pbuf_;
  std::vector<PyRandom> rng_;
  EnvBuffers buf_;
  ThreadPool pool_;
  std::atomic<unsigned long> next_match_id_{1};
  bool record_ = false;
};

}  // namespace env
}  // namespace sim
