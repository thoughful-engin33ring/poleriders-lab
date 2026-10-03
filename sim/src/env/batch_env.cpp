#include "batch_env.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../game/assets.h"
#include "../game/layout.h"

namespace sim {
namespace env {

// ---------------------------------------------------------------- ThreadPool

ThreadPool::ThreadPool(int threads) {
  for (int i = 1; i < threads; i++) workers_.emplace_back([this] { Loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto &t : workers_) t.join();
}

void ThreadPool::Work(const std::function<void(int)> *f, int n) {
  for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n;) (*f)(i);
}

void ThreadPool::Loop() {
  int seen = 0;
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || gen_ != seen; });
    if (stop_) return;
    seen = gen_;
    const std::function<void(int)> *f = job_;
    int n = n_;
    l.unlock();
    Work(f, n);
    l.lock();
    if (--busy_ == 0) done_cv_.notify_one();
  }
}

void ThreadPool::Run(int n, const std::function<void(int)> &f) {
  if (workers_.empty() || n <= 1) {
    for (int i = 0; i < n; i++) f(i);
    return;
  }
  {
    std::lock_guard<std::mutex> l(mu_);
    job_ = &f, n_ = n, busy_ = static_cast<int>(workers_.size());
    next_.store(0, std::memory_order_relaxed);
    gen_++;
  }
  cv_.notify_all();
  Work(&f, n);
  std::unique_lock<std::mutex> l(mu_);
  done_cv_.wait(l, [&] { return busy_ == 0; });
  job_ = nullptr;
}

// ---------------------------------------------------------------- BatchEnv

namespace {
int ResolveThreads(int t) {
  if (t > 0) return t;
  unsigned h = std::thread::hardware_concurrency();
  return h ? static_cast<int>(h) : 1;
}
uint64_t splitmix(uint64_t &s) {
  uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
Arena *NewArena() {
  Arena *a = Arena::Create();
  static std::atomic<bool> warned{false};
  if (!a && !warned.exchange(true))
    fprintf(stderr, "sprsim: out of match arenas (%d alive): new matches use the heap (no snapshots)\n", kArenaSlots);
  return a;
}
// a match of the heap fallback (no arena) is deleted the normal way; an arena's match is dropped with the arena
void DropMatch(MatchSim &s) {
  if (s.arena)
    s.arena->Clear();
  else
    delete s.m;
  s.m = nullptr;
}
Match *MakeMatch(MatchSim &s, const EngineContext &e) {
  return s.arena ? Match::CreateIn(s.arena, e) : new Match(e);
}
}  // namespace

BatchEnv::BatchEnv(const EnvConfig &cfg)
    : cfg_(cfg), n_(cfg.n_hosts * cfg.per_host), engine0_{}, pool_(ResolveThreads(cfg.threads)) {
  DefaultInputState(engine0_);
  // Common::Session records as the menus leave them for a 1v1 (see sim/re_notes/globals.md):
  // {mode 0, team (-1 = Vaulters / 1 = Benders), pad, head, 0, 7, 0}
  engine0_.session[0] = PoleRiders::SessionPlayer{0, -1, 0, cfg.head[0], 0, 7, 0};
  engine0_.session[1] = PoleRiders::SessionPlayer{0, 1, 1, cfg.head[1], 0, 7, 0};
  engine0_.cfg.level = cfg.level;
  {
    Match probe(engine0_);  // built on this thread first (Box2D's lazily initialized statics); plain heap
    terrain_.Build(probe.ps);
  }
  slots_.resize(n_);
  for (int k = 0; k < n_; k++) {
    slots_[k].sim.arena = NewArena();
    slots_[k].sim.k = k;
  }
  for (int h = 0; h < cfg.n_hosts; h++) rng_.emplace_back(cfg.seed + h);
  buf_.obs.assign(n_ * 2 * kObsSize, 0.0f);
  buf_.final_obs.assign(n_ * 2 * kObsSize, 0.0f);
  buf_.rew.assign(n_ * 2, 0.0f);
  buf_.terms.assign(n_ * 4, 0.0f);
  buf_.reset.assign(n_, 0), buf_.trunc.assign(n_, 0), buf_.sd.assign(n_, 0), buf_.sd_start.assign(n_, 0);
  buf_.score.assign(n_ * 2, 0);
  buf_.touch.assign(n_ * 2 * kTouchSize, 0.0f);
  buf_.touch_final.assign(n_ * 2 * kTouchSize, 0.0f);
}

BatchEnv::~BatchEnv() {
  // the matches live entirely in their arenas: dropping the arenas frees everything (no destructors)
  for (Slot &s : slots_) {
    if (!s.sim.arena) delete s.sim.m;
    Arena::Destroy(s.sim.arena);
  }
  for (Branch &b : branches_) Arena::Destroy(b.sim.arena);
}

void BatchEnv::Observe(const StateV &s, float *dst, int legacy) const {
  if (!legacy) {
    Observe2(s, 1, terrain_, dst);
    Observe2(s, 2, terrain_, dst + kObsSize);
    return;
  }
  // evaluation only: riders in `legacy` see dead riders the way policies trained before the dead-rider
  // fix did (the corpse as a live rider; out of play = the spawn ghost only)
  StateV old = s;
  old.p[0].dead = old.p[1].dead = false;
  Observe2(legacy & 1 ? old : s, 1, terrain_, dst);
  Observe2(legacy & 2 ? old : s, 2, terrain_, dst + kObsSize);
}

SimOut BatchEnv::RowOut(int k) {
  EnvBuffers &b = buf_;
  return {&b.obs[k * 2 * kObsSize], &b.final_obs[k * 2 * kObsSize], &b.rew[2 * k], &b.terms[4 * k],
          &b.touch[k * 2 * kTouchSize], &b.touch_final[k * 2 * kTouchSize], &b.reset[k], &b.trunc[k], &b.sd[k],
          &b.sd_start[k], &b.score[2 * k]};
}

SimOut BatchEnv::PlanOut(int b) {
  PlanBuffers &p = pbuf_;
  // a branch's exposed observation / touch rows are the post-step ones (final_obs / touch_final: a branch
  // never auto-resets); obs / touch proper go to scratch rows
  return {&p.s_obs[b * 2 * kObsSize], &p.obs[b * 2 * kObsSize], &p.rew[2 * b], &p.terms[4 * b],
          &p.s_touch[b * 2 * kTouchSize], &p.touch[b * 2 * kTouchSize], &p.s_reset[b], &p.trunc[b], &p.s_sd[b],
          &p.s_sd_start[b], &p.score[2 * b]};
}

// sh_destroy + sh_create: a fresh match in the (cleared) arena
void BatchEnv::NewMatch(MatchSim &s, bool sd) {
  DropMatch(s);
  if (cfg_.random_pole) {  // this match's own, reproducible stream of random pole angles
    sim::EngineContext e = engine0_;
    e.randomPole = true;
    e.poleRng = cfg_.seed * 0xd1342543de82ef95ULL + 1000003ULL * (s.k + 1) + 7919ULL * ++s.st.resets;
    s.m = MakeMatch(s, e);
  } else {
    s.m = MakeMatch(s, engine0_);
  }
  PoleRiders::PlayState *ps = s.m->ps;
  if (sd) {  // farm reset k <clock>
    ps->timeLeft = cfg_.sd_clock;
    ps->timerOn = true;
  }
  s.st.pit_start = 0;
  if (!sd && cfg_.pit_frac > 0) PitDrop(s);
}

void BatchEnv::ResetSim(MatchSim &s, bool sd, unsigned long match_id, int legacy, const SimOut &o,
                        std::string *json) {
  NewMatch(s, sd);
  PoleRiders::PlayState *ps = s.m->ps;
  s.st.match_id = match_id;
  ExtractState(ps, &s.st.prev);
  s.st.steps = 0;
  *o.sd_start = sd;
  s.st.scramble_wait = false;
  s.st.gr.Reset();
  ResetTouch(s);
  Observe(s.st.prev, o.obs, legacy);
  WriteTouch(s, o.touch);
  if (json) *json = SnapshotJson(ps, s.m->frame, s.st.match_id);
}

void BatchEnv::ResetSlot(int k, bool sd) {
  Slot &s = slots_[k];
  History &h = s.hist;
  h.reset_state = s.sim.st;  // the RNG streams the reset draws from (replayed by planning branches)
  h.reset_sd = sd;
  h.reset_match_id = next_match_id_.fetch_add(1);
  h.Clear();
  h.lineage = s.next_lineage++;
  h.valid = true;
  ResetSim(s.sim, sd, h.reset_match_id, s.legacy_obs, RowOut(k), record_ ? &s.reset_json : nullptr);
}

void BatchEnv::Invalidate(Slot &s) {
  s.hist.valid = false;
  s.hist.Clear();
  s.hist.lineage = s.next_lineage++;
}

// ---------------------------------------------------------------- ball touches (read-only)

namespace {
constexpr int kNeverTicks = 59940;  // 999 s
}

void BatchEnv::ResetTouch(MatchSim &s) {
  for (int i = 0; i < 2; i++) {
    s.st.touch_age[i] = s.st.kick_age[i] = kNeverTicks;
    s.st.touch_what[i] = 0;
    s.st.touching[i] = false;
    const PoleRiders::Player *pl = s.m->ps->players[i];
    s.st.prev_nc[i] = pl ? pl->ballNoCollideTimer : 0.0f;
  }
}

// After a tick: which riders' bodies / poles are in touching contact with the ball (non-sensor fixtures on
// both sides, i.e. a real collision; the contact list after b2World::Step holds this tick's contacts, with
// bodies attributed to riders the way the game's contact handlers decode their tags), and
// which riders kicked it (the game's kick, BallPlayerCollision / HammerBallCollision, is the only writer of
// a positive ballNoCollideTimer: 0.6 s). Nothing in the match is written.
void BatchEnv::TrackTouch(MatchSim &s) {
  PoleRiders::PlayState *ps = s.m->ps;
  SimState &t = s.st;
  bool now[2] = {false, false};
  uint8_t what[2] = {0, 0};
  b2Body *bb = ps->ball ? ps->ball->body : nullptr;
  auto tag = [](const void *ud) { return ud ? *static_cast<const int *>(ud) : 0; };  // the game's TagOf
  for (b2ContactEdge *e = bb ? bb->GetContactList() : nullptr; e; e = e->next) {
    b2Contact *c = e->contact;
    if (!c->IsTouching() || c->GetFixtureA()->IsSensor() || c->GetFixtureB()->IsSensor()) continue;
    const b2Fixture *fx = c->GetFixtureA()->GetBody() == bb ? c->GetFixtureB() : c->GetFixtureA();
    const b2Body *o = e->other;
    // decoded like the game's contact handlers: a body tagged with a player index IS that rider (its hull/foot,
    // or a pole handle segment, which the game also tags with the rider: those contacts kick the ball); a pole
    // or hammer body belongs to the player its fixture is tagged with
    const int bt = tag(o->GetUserData());
    int rider;
    uint8_t bit;
    if (bt <= 3) {
      rider = bt;
      bit = rider < ps->numPlayers && ps->players[rider] && o == ps->players[rider]->body ? 1 : 4;
    } else if (bt == PoleRiders::Assets::BODY_POLE || bt == PoleRiders::Assets::BODY_HAMMER) {
      rider = tag(fx->GetUserData());
      bit = 2;
    } else {
      continue;
    }
    if (rider < 0 || rider > 1) continue;
    now[rider] = true;
    what[rider] |= bit;
  }
  for (int i = 0; i < 2; i++) {
    if (now[i]) {
      t.touch_age[i] = 0;
      t.touch_what[i] = what[i];
    } else if (t.touch_age[i] < kNeverTicks) {
      t.touch_age[i]++;
    }
    t.touching[i] = now[i];
    const PoleRiders::Player *pl = i < ps->numPlayers ? ps->players[i] : nullptr;
    const float nc = pl ? pl->ballNoCollideTimer : 0.0f;
    if (nc > t.prev_nc[i])
      t.kick_age[i] = 0;
    else if (t.kick_age[i] < kNeverTicks)
      t.kick_age[i]++;
    t.prev_nc[i] = nc;
  }
}

void BatchEnv::WriteTouch(const MatchSim &s, float *dst) const {
  const int lt = s.m->ps->lastTouch;
  for (int i = 0; i < 2; i++) {
    float *d = dst + i * kTouchSize;
    d[0] = s.st.touch_age[i] >= kNeverTicks ? kTouchNever : s.st.touch_age[i] / 60.0f;
    d[1] = s.st.touch_what[i];
    d[2] = s.st.kick_age[i] >= kNeverTicks ? kTouchNever : s.st.kick_age[i] / 60.0f;
    d[3] = s.st.touching[i];
    d[4] = lt == i;
  }
}

int BatchEnv::DebugEvent(int k, int kind, int who) {
  if (k < 0 || k >= n_ || who < 0 || who > 1) return -1;
  Slot &s = slots_[k];
  const int rc = DebugEventSim(s.sim, kind, who);
  if (rc == 0) {
    Op op{};
    op.kind = 1, op.a = static_cast<int8_t>(kind), op.b = static_cast<int8_t>(who);
    s.hist.Push(op);
  }
  return rc;
}

int BatchEnv::DebugEventSim(MatchSim &s, int kind, int who) {
  PoleRiders::PlayState *ps = s.m->ps;
  Match::Scope scope(s.m);  // game code: g_engine (+ the match's arena)
  int rc = 0;
  if (kind == 1) {  // rider `who` killed by the other rider
    PoleRiders::Player *v = ps->players[who], *kr = ps->players[1 - who];
    if (!v || !kr || ps->respawnTimer[who] > 0.0f || v->ghost) {
      rc = -2;
    } else {
      v->OffPole();
      v->GetStats()->AddDeath();
      kr->GetStats()->AddKill();
      kr->GetStats()->AddKick();
      ps->respawnTimer[who] = PoleRiders::Assets::deathDelay;
    }
  } else if (kind == 2) {  // the ball against the wall team `who` scores on (team 0: +x), moving into it
    PoleRiders::Ball *b = ps->ball;
    if (!b) {
      rc = -2;
    } else {
      const float sx = who == 0 ? 1.0f : -1.0f, bx = sx * 19.6f, ty = b->trolley->GetPosition().y;
      b->trolley->SetTransform(b2Vec2(bx, ty), 0.0f);
      b->body->SetTransform(b2Vec2(bx, b->body->GetPosition().y), 0.0f);
      for (b2Body *body : {b->trolley, b->body}) {
        body->SetLinearVelocity(b2Vec2(sx * 10.0f, 0.0f));
        body->SetAngularVelocity(0.0f);
      }
      ps->lastTouch = who;
    }
  } else {
    rc = -1;
  }
  return rc;
}

// Right after construction, respawn the chosen riders in their enemy pit (player 1 attacks +x) through
// the game's own Player::Respawn, with the pole built pointing up (EngineContext::spawnPoleUp).
void BatchEnv::PitDrop(MatchSim &s) {
  SimState &t = s.st;
  if (!t.pit_rng) t.pit_rng = cfg_.seed * 0x100000001b3ULL + 7919ULL * (s.k + 1);
  if ((splitmix(t.pit_rng) >> 11) * 0x1.0p-53 >= cfg_.pit_frac) return;
  int who = 1 + (int)(splitmix(t.pit_rng) % 3);  // 1 = player 1, 2 = player 2, 3 = both
  PoleRiders::PlayState *ps = s.m->ps;
  Match::Scope scope(s.m);  // Respawn allocates: into the match's arena
  s.m->engine.spawnPoleUp = true;
  for (int i = 0; i < 2; i++) {
    if (!(who >> i & 1) || !ps->players[i]) continue;
    float x = i == 0 ? cfg_.pit_x_px : -cfg_.pit_x_px;
    ps->players[i]->Respawn(x, cfg_.pit_y_px, false, PoleRiders::Assets::standardPoleLength, false, false);
    ps->respawnTimer[i] = 0.0f;
  }
  s.m->engine.spawnPoleUp = false;
  t.pit_start = who;
}

// Training-only (worker solo practice): keep the locked riders in the game's ghost state — their
// fixtures then only collide with the terrain (no rider or ball contact, can't kill or be killed).
// The spawn-ghost timer would clear it, so it is refreshed before every tick.
void BatchEnv::GhostLock(MatchSim &s, int mask) {
  PoleRiders::PlayState *ps = s.m->ps;
  Match::Scope scope(s.m);
  for (int i = 0; i < 2; i++) {
    PoleRiders::Player *p = ps->players[i];
    if (!(mask >> i & 1) || !p) continue;
    if (!p->ghost) p->SetGhost(true, 0.3f);
    p->ghostTimer = 0.3f;
  }
}

// Random re-spawn (SetScramble). Returns false while the game's own post-goal reset is still pending.
bool BatchEnv::Scramble(MatchSim &s) {
  SimState &t = s.st;
  PoleRiders::PlayState *ps = s.m->ps;
  if (ps->ballResetPending || ps->ballResetDelay > 0.0f) return false;  // a goal: wait for the game's reset
  if (!ps->ball || !ps->players[0] || !ps->players[1]) return false;
  if (!t.scr_rng) t.scr_rng = cfg_.seed * 0x9e3779b97f4a7c15ULL + 104729ULL * (s.k + 1);
  auto uni = [&](double lo, double hi) { return lo + (hi - lo) * ((splitmix(t.scr_rng) >> 11) * 0x1.0p-53); };
  const double r = PoleRiders::Assets::ratio;  // px per m
  // riders anywhere between the goal walls (faces at +-20.3 m), at least 4 m apart, 0.78 m above the
  // topmost ground there (the pit drop's clearance), pole straight up
  double x[2];
  x[0] = uni(-19.5, 19.5);
  do x[1] = uni(-19.5, 19.5);
  while (std::fabs(x[1] - x[0]) < 4.0);
  Match::Scope scope(s.m);  // Respawn allocates: into the match's arena
  s.m->engine.spawnPoleUp = true;
  for (int i = 0; i < 2; i++) {
    const double y = terrain_.Floor(x[i], -30.0) - 0.78;  // y down: the surface under a point high above
    ps->players[i]->Respawn(static_cast<float>(x[i] * r), static_cast<float>(y * r), false,
                            PoleRiders::Assets::standardPoleLength, false, false);
    ps->respawnTimer[i] = 0.0f;
  }
  s.m->engine.spawnPoleUp = false;
  // the ball hangs from its zipline trolley (a prismatic joint along x): slide both to a random x, at rest
  PoleRiders::Ball *b = ps->ball;
  const float bx = static_cast<float>(uni(-15.0, 15.0));
  const float ty = b->trolley->GetPosition().y;
  b->trolley->SetTransform(b2Vec2(bx, ty), 0.0f);
  b->body->SetTransform(b2Vec2(bx, ty), 0.0f);
  for (b2Body *body : {b->trolley, b->body}) {
    body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
    body->SetAngularVelocity(0.0f);
  }
  return true;
}

void BatchEnv::ResetAll() {
  for (int k = 0; k < n_; k++) slots_[k].sd_next = rng_[k / cfg_.per_host].Random() < cfg_.sd_frac;
  pool_.Run(n_, [&](int k) { ResetSlot(k, slots_[k].sd_next); });
}

bool BatchEnv::StepSim(MatchSim &s, const float *act6, int ghost_lock, int legacy, const SimOut &o,
                       std::string *json) {
  const int fs = cfg_.frame_skip;
  SimState &t = s.st;
  float st[6];
  StickFromAction(1, act6, st);
  StickFromAction(2, act6 + 3, st + 3);
  const PadAct pa[2] = {{st[0], st[1], st[2]}, {st[3], st[4], st[5]}};
  for (int i = 0; i < fs; i++) {
    if (ghost_lock) GhostLock(s, ghost_lock);
    s.m->Tick(pa);
    TrackTouch(s);
  }
  StateV cur;
  ExtractState(s.m->ps, &cur);
  double r[2];
  Terms tm[2];
  ComputeReward(t.prev, cur, cfg_.w, fs, r, tm);
  if (cfg_.goal_refund) t.gr.Apply(t.prev, cur, cfg_.w, fs, tm, r);
  t.steps++;
  // end_on_sd (training rule, not the game's): a match tied when the clock runs out ends as a draw
  // instead of going into sudden death
  bool done = cur.game_over || !cur.has_players || (cfg_.end_on_sd && cur.sudden_death);
  bool trunc = !done && t.steps >= cfg_.max_steps;
  if (cur.has_players) t.prev = cur;
  for (int a = 0; a < 2; a++) {
    o.rew[a] = static_cast<float>(r[a]);
    o.terms[2 * a] = static_cast<float>(tm[a].goal);
    o.terms[2 * a + 1] = static_cast<float>(tm[a].kill);
    o.score[a] = cur.score[a];
  }
  *o.sd = cur.sudden_death;
  if (cur.has_players)
    Observe(cur, o.final_obs, legacy);
  else
    memset(o.final_obs, 0, 2 * kObsSize * sizeof(float));
  if (json) *json = SnapshotJson(s.m->ps, s.m->frame, t.match_id);
  const bool need_reset = done || trunc || !cur.has_players;
  *o.reset = need_reset;
  *o.trunc = trunc && cur.has_players;
  WriteTouch(s, o.touch_final);
  if (!need_reset) {
    memcpy(o.obs, o.final_obs, 2 * kObsSize * sizeof(float));
    memcpy(o.touch, o.touch_final, 2 * kTouchSize * sizeof(float));
  }
  if ((cfg_.scramble || cfg_.goal_spawn > 0.0) && !need_reset) {
    if (cfg_.scramble && (tm[0].goal != 0.0 || tm[0].kill != 0.0)) t.scramble_wait = true;
    if (cfg_.goal_spawn > 0.0 && tm[0].goal != 0.0) {  // goals only, with probability goal_spawn
      if (!t.scr_rng) t.scr_rng = cfg_.seed * 0x9e3779b97f4a7c15ULL + 104729ULL * (s.k + 1);
      if ((splitmix(t.scr_rng) >> 11) * 0x1.0p-53 < cfg_.goal_spawn) t.scramble_wait = true;
    }
    if (t.scramble_wait && Scramble(s)) {
      // the next transition starts from the re-spawned positions (no reward for the teleport)
      t.scramble_wait = false;
      ExtractState(s.m->ps, &t.prev);
      Observe(t.prev, o.obs, legacy);
    }
  }
  return need_reset;
}

void BatchEnv::Step(const float *actions) {
  pool_.Run(n_, [&](int k) {
    Slot &s = slots_[k];
    Op op{};
    op.kind = 0, op.ghost = static_cast<uint8_t>(s.ghost_lock);
    memcpy(op.act, actions + 6 * k, sizeof op.act);
    s.hist.Push(op);
    s.need_reset = StepSim(s.sim, actions + 6 * k, s.ghost_lock, s.legacy_obs, RowOut(k),
                           record_ ? &s.step_json : nullptr);
  });
  // FarmHost.reset(k) draws its coin in match order, per host
  std::vector<int> todo;
  for (int k = 0; k < n_; k++)
    if (slots_[k].need_reset) {
      slots_[k].sd_next = rng_[k / cfg_.per_host].Random() < cfg_.sd_frac;
      todo.push_back(k);
    }
  if (!todo.empty()) pool_.Run(static_cast<int>(todo.size()), [&](int i) { ResetSlot(todo[i], slots_[todo[i]].sd_next); });
}

void BatchEnv::RawAct(const float *sticks, int n) {
  pool_.Run(n_, [&](int k) {
    const float *st = sticks + 6 * k;
    const PadAct pa[2] = {{st[0], st[1], st[2]}, {st[3], st[4], st[5]}};
    for (int t = 0; t < n; t++) slots_[k].sim.m->Tick(pa);
    Invalidate(slots_[k]);  // not replayable by planning branches
  });
}

std::string BatchEnv::RawReset(int k, float clock) {
  Slot &s = slots_[k];
  DropMatch(s.sim);
  s.sim.m = MakeMatch(s.sim, engine0_);
  if (clock > 0) {  // bridge JOB_RESET
    s.sim.m->ps->timeLeft = clock;
    s.sim.m->ps->timerOn = true;
  }
  s.sim.st.match_id = next_match_id_.fetch_add(1);
  Invalidate(s);
  return CurrentJson(k);
}

std::string BatchEnv::CurrentJson(int k) const {
  const MatchSim &s = slots_[k].sim;
  return SnapshotJson(s.m->ps, s.m->frame, s.st.match_id);
}

std::string BatchEnv::LevelJson() const { return env::LevelJson(slots_[0].sim.m->ps); }

// ---------------------------------------------------------------- snapshot / restore

int BatchEnv::Snapshot(int k, int slot) {
  if (k < 0 || k >= n_ || slot < 0 || slot >= kMaxSnapshots) return -1;
  Slot &s = slots_[k];
  if (!s.sim.arena || s.sim.arena->Overflowed()) return -3;
  if (static_cast<int>(s.snaps.size()) <= slot) s.snaps.resize(slot + 1);
  Snap &p = s.snaps[slot];
  s.sim.arena->Save(&p.mem);
  p.m = s.sim.m, p.st = s.sim.st, p.hist = s.hist;
  p.ok = true;
  return 0;
}

int BatchEnv::Restore(int k, int slot) {
  if (k < 0 || k >= n_ || slot < 0 || slot >= kMaxSnapshots) return -1;
  Slot &s = slots_[k];
  if (static_cast<int>(s.snaps.size()) <= slot || !s.snaps[slot].ok) return -2;
  const Snap &p = s.snaps[slot];
  s.sim.arena->Restore(p.mem);
  s.sim.m = p.m, s.sim.st = p.st, s.hist = p.hist;
  s.hist.lineage = s.next_lineage++;  // branches that followed the abandoned line re-play from the reset
  s.need_reset = false;
  // the outputs a policy needs to continue from here (obs after a step == the observation of st.prev)
  Observe(s.sim.st.prev, &buf_.obs[k * 2 * kObsSize], s.legacy_obs);
  WriteTouch(s.sim, &buf_.touch[k * 2 * kTouchSize]);
  return 0;
}

// every game object pointer of the match must point into its arena
int BatchEnv::CheckSim(const MatchSim &s) const {
  const Arena *a = s.arena;
  int bad = 0;
  auto chk = [&](const void *p) {
    if (p && !a->Contains(p)) bad++;
  };
  if (!s.m) return 0;  // a branch never synced yet
  if (!a) return -1;   // heap fallback (out of arenas)
  chk(s.m);
  const PoleRiders::PlayState *ps = s.m->ps;
  chk(ps);
  chk(ps->world);
  for (int i = 0; i < ps->numPlayers && i < 4; i++) {
    const PoleRiders::Player *pl = ps->players[i];
    chk(pl);
    if (!pl) continue;
    chk(pl->stats), chk(pl->body), chk(pl->pole), chk(pl->anim), chk(pl->smashAnim);
    if (pl->pole)
      for (b2Body *sg : pl->pole->segments) chk(sg);
  }
  if (ps->ball) chk(ps->ball), chk(ps->ball->body), chk(ps->ball->trolley), chk(ps->ball->wire);
  b2World *w = ps->world;
  for (b2Body *b = w->GetBodyList(); b; b = b->GetNext()) {
    chk(b);
    for (b2Fixture *f = b->GetFixtureList(); f; f = f->GetNext()) chk(f), chk(f->GetShape());
  }
  for (b2Joint *j = w->GetJointList(); j; j = j->GetNext()) chk(j);
  for (b2Contact *c = w->GetContactList(); c; c = c->GetNext()) chk(c);
  return bad;
}

// ---------------------------------------------------------------- planning branches

int BatchEnv::PlanInit(const int *owner, int n_branches) {
  for (Branch &b : branches_) Arena::Destroy(b.sim.arena);
  branches_.clear();
  for (int b = 0; b < n_branches; b++)
    if (owner[b] < 0 || owner[b] >= n_) return -1;
  branches_.resize(n_branches);
  for (int b = 0; b < n_branches; b++) {
    Branch &br = branches_[b];
    br.owner = owner[b];
    br.sim.k = owner[b];
    br.sim.arena = NewArena();
    br.lineage = 0;  // never synced (lineages start at 1)
    if (!br.sim.arena) {
      for (Branch &x : branches_) Arena::Destroy(x.sim.arena);
      branches_.clear();
      return -2;
    }
  }
  const int B = n_branches;
  PlanBuffers &p = pbuf_;
  p.obs.assign(B * 2 * kObsSize, 0.0f), p.s_obs.assign(B * 2 * kObsSize, 0.0f);
  p.rew.assign(B * 2, 0.0f), p.terms.assign(B * 4, 0.0f);
  p.done.assign(B, 0), p.trunc.assign(B, 0), p.s_reset.assign(B, 0), p.s_sd.assign(B, 0), p.s_sd_start.assign(B, 0);
  p.score.assign(B * 2, 0);
  p.touch.assign(B * 2 * kTouchSize, 0.0f), p.s_touch.assign(B * 2 * kTouchSize, 0.0f);
  return 0;
}

// bring branch br up to its match's current state by re-playing the match's log
bool BatchEnv::SyncBranch(Branch &br) {
  const Slot &r = slots_[br.owner];
  if (!r.hist.valid) return false;
  const int b = static_cast<int>(&br - branches_.data());
  const SimOut o = PlanOut(b);
  if (br.dirty) {  // left mid-search without PlanEnd: back to its root first
    br.sim.arena->Restore(br.root_mem);
    br.sim.st = br.root_st;
    br.dirty = false;
  }
  if (br.lineage != r.hist.lineage) {  // the match was reset / restored since: rebuild from its reset
    br.sim.st = r.hist.reset_state;
    ResetSim(br.sim, r.hist.reset_sd, r.hist.reset_match_id, r.legacy_obs, o, nullptr);
    br.lineage = r.hist.lineage;
    br.pos = 0;
  }
  for (; br.pos < r.hist.len; br.pos++) {
    const Op &op = (*r.hist.ops)[br.pos];
    if (op.kind == 0)
      StepSim(br.sim, op.act, op.ghost, r.legacy_obs, o, nullptr);
    else
      DebugEventSim(br.sim, op.a, op.b);
  }
  return true;
}

int BatchEnv::PlanBegin(const uint8_t *active) {
  const int B = PlanN();
  for (int b = 0; b < B; b++) branches_[b].active = active ? active[b] != 0 : true;
  std::atomic<int> bad{0};
  pool_.Run(B, [&](int b) {
    Branch &br = branches_[b];
    if (!br.active) return;
    if (!SyncBranch(br)) {
      bad++;
      br.active = false;
      return;
    }
    br.sim.arena->Save(&br.root_mem);
    br.root_st = br.sim.st;
    br.done = false;
    const Slot &r = slots_[br.owner];
    Observe(br.sim.st.prev, &pbuf_.obs[b * 2 * kObsSize], r.legacy_obs);
    WriteTouch(br.sim, &pbuf_.touch[b * 2 * kTouchSize]);
    for (int a = 0; a < 2; a++) {
      pbuf_.rew[2 * b + a] = 0.0f, pbuf_.terms[4 * b + 2 * a] = pbuf_.terms[4 * b + 2 * a + 1] = 0.0f;
      pbuf_.score[2 * b + a] = br.sim.st.prev.score[a];
    }
    pbuf_.done[b] = pbuf_.trunc[b] = 0;
  });
  return bad.load() ? -1 : 0;
}

void BatchEnv::PlanStep(const float *actions) {
  pool_.Run(PlanN(), [&](int b) {
    Branch &br = branches_[b];
    if (!br.active) return;
    if (br.done) {  // frozen at its end
      for (int a = 0; a < 4; a++) pbuf_.terms[4 * b + a] = 0.0f;
      pbuf_.rew[2 * b] = pbuf_.rew[2 * b + 1] = 0.0f;
      return;
    }
    const Slot &r = slots_[br.owner];
    br.dirty = true;
    br.done = StepSim(br.sim, actions + 6 * b, r.ghost_lock, r.legacy_obs, PlanOut(b), nullptr);
    pbuf_.done[b] = br.done;
  });
}

void BatchEnv::PlanEnd() {
  pool_.Run(PlanN(), [&](int b) {
    Branch &br = branches_[b];
    if (!br.dirty) return;
    br.sim.arena->Restore(br.root_mem);
    br.sim.st = br.root_st;
    br.dirty = false;
  });
}

std::string BatchEnv::PlanJson(int b) const {
  const MatchSim &s = branches_[b].sim;
  if (!s.m) return "";
  return SnapshotJson(s.m->ps, s.m->frame, s.st.match_id);
}

}  // namespace env
}  // namespace sim
