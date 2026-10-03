// See features.h. Every function here mirrors a Python / bridge function line by line; keep them in
// sync (sim/tools/test_env_equiv.py checks the values).
#include "features.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../game/assets.h"
#include "../game/layout.h"
#include "../x86math.h"

namespace sim {
namespace env {

using PoleRiders::Player;
using PoleRiders::PlayState;

const int kTerrainOffsets[kNumOffsets] = {-6, -4, -2, -1, 0, 1, 2, 4, 6};

namespace {

// printf("%.4f", f) parsed back as a double (see features.h)
inline double R4(float f) { return std::nearbyint(static_cast<double>(f) * 1e4) / 1e4; }
inline double R3(float f) { return std::nearbyint(static_cast<double>(f) * 1e3) / 1e3; }

// Python min(a, b) / max(a, b) for floats: the first argument unless the second is strictly better
inline double PyMin(double a, double b) { return b < a ? b : a; }
inline double PyMax(double a, double b) { return b > a ? b : a; }

void Body(const b2Body *b, BodyV *o) {
  const b2Vec2 &p = b->GetPosition();
  b2Vec2 v = b->GetLinearVelocity();
  o->x = R4(p.x), o->y = R4(p.y), o->a = R4(b->GetAngle());
  o->vx = R4(v.x), o->vy = R4(v.y), o->w = R4(b->GetAngularVelocity());
}

const char *TagName(const void *ud) {
  using namespace PoleRiders::Assets;
  static const struct {
    const int *tag;
    const char *name;
  } kTags[] = {{&BODY_PLAYER[0], "player1"}, {&BODY_PLAYER[1], "player2"}, {&BODY_PLAYER[2], "player3"},
               {&BODY_PLAYER[3], "player4"}, {&BODY_GROUND, "ground"},      {&BODY_WALL, "wall"},
               {&BODY_POLE, "pole"},         {&BODY_BALL, "ball"},          {&BODY_ANCHOR, "anchor"},
               {&BODY_NONE, "none"},         {&BODY_DELETER, "deleter"},    {&BODY_HAMMER, "hammer"},
               {&BODY_SEA, "sea"}};
  for (const auto &t : kTags)
    if (ud == t.tag) return t.name;
  return nullptr;
}

// bridge label(): a player's torso, a player's pole segment, or the body's BODY_* tag
const char *Label(const PlayState *ps, const b2Body *b, char *tmp) {
  int np = ps->numPlayers;
  for (int i = 0; i < np && i < 4; i++) {
    const Player *pl = ps->players[i];
    if (!pl) continue;
    if (pl->body == b) return snprintf(tmp, 16, "player%d", i + 1), tmp;
    const PoleRiders::Pole *pole = pl->pole;
    for (int k = 0; pole && k < 10; k++)
      if (pole->segments[k] == b) return snprintf(tmp, 16, "pole%d", i + 1), tmp;
  }
  const char *tag = TagName(b->GetUserData());
  return tag ? tag : "other";
}

void Put(std::string &s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void Put(std::string &s, const char *fmt, ...) {
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
  va_end(ap);
  s.append(tmp, n < (int)sizeof tmp ? n : (int)sizeof tmp - 1);
}

void PutBody(std::string &s, const char *pre, const b2Body *b) {
  const b2Vec2 &p = b->GetPosition();
  b2Vec2 v = b->GetLinearVelocity();
  Put(s, "%s\"x\":%.4f,\"y\":%.4f,\"a\":%.4f,\"vx\":%.4f,\"vy\":%.4f,\"w\":%.4f}", pre, p.x, p.y, b->GetAngle(),
      v.x, v.y, b->GetAngularVelocity());
}

// Static "ground" polygons in world meters, each vertex as the bridge's dump_level() computes it
// (float math, x86 cosf/sinf) and rounded like its %.4f output.
template <class F>
void ForEachStaticPoly(const PlayState *ps, F f) {
  char tmp[16];
  for (const b2Body *b = ps->world->GetBodyList(); b; b = b->GetNext()) {
    if (b->GetType() != b2_staticBody) continue;
    const char *label = Label(ps, b, tmp);
    float px = b->GetPosition().x, py = b->GetPosition().y, a = b->GetAngle();
    float c = x86_cosf(a), sn = x86_sinf(a);
    f(label, nullptr, 0, false);  // body start
    for (const b2Fixture *fx = b->GetFixtureList(); fx; fx = fx->GetNext()) {
      const b2Shape *sh = fx->GetShape();
      if (!sh || sh->GetType() != b2Shape::e_polygon) continue;
      const b2PolygonShape *poly = static_cast<const b2PolygonShape *>(sh);
      int cnt = poly->GetVertexCount();
      if (cnt < 3 || cnt > 8) continue;
      float v[16];
      for (int k = 0; k < cnt; k++) {
        float vx = poly->m_vertices[k].x, vy = poly->m_vertices[k].y;
        v[2 * k] = px + c * vx - sn * vy;
        v[2 * k + 1] = py + sn * vx + c * vy;
      }
      f(label, v, cnt, fx->IsSensor());
    }
  }
}

}  // namespace

// ---------------------------------------------------------------- snapshot (bridge snapshot_ps)

void ExtractState(const PlayState *ps, StateV *o) {
  o->has_players = false;
  o->p[0].present = o->p[1].present = false;
  o->level = ps->level;
  o->game_over = ps->gameOver;
  o->sudden_death = ps->suddenDeath;
  o->score[0] = ps->score[0], o->score[1] = ps->score[1];
  o->time_left = R3(ps->timeLeft);
  int np = ps->numPlayers;
  for (int i = 0; i < np && i < 4; i++) {
    const Player *pl = ps->players[i];
    const b2Body *body = pl ? pl->body : nullptr;
    if (!body) continue;
    o->has_players = true;
    if (i >= 2) continue;
    PlayerV &p = o->p[i];
    p.present = true;
    p.on_pole = pl->onPole, p.on_ground = pl->onGround, p.ghost = pl->ghost;
    p.ghost_t = pl->ghost ? R3(pl->ghostTimer) : 0.0;
    p.dead = ps->respawnTimer[i] > 0.0f;
    p.respawn_t = p.dead ? R3(ps->respawnTimer[i]) : 0.0;
    p.hammer = pl->pole && pl->pole->isHammer;
    Body(body, &p.b);
    p.has_stats = pl->stats != nullptr;
    p.deaths = p.has_stats ? pl->stats->deaths : 0;
    p.suicides = p.has_stats ? pl->stats->suicides : 0;
    p.npole = 0;
    for (int k = 0; pl->pole && k < 10; k++)
      if (const b2Body *seg = pl->pole->segments[k]) Body(seg, &p.pole[p.npole++]);
  }
  const b2Body *bb = ps->ball ? ps->ball->body : nullptr;
  o->has_ball = bb != nullptr;
  if (bb) Body(bb, &o->ball);
}

std::string SnapshotJson(const PlayState *ps, unsigned long frame, unsigned long match_id) {
  std::string s;
  s.reserve(4096);
  Put(s,
      "{\"frame\":%lu,\"match_id\":%lu,\"in_match\":true,\"level\":%d,\"game_over\":%s,"
      "\"sudden_death\":%s,\"score\":[%d,%d],\"time_left\":%.3f,\"players\":[",
      frame, match_id, ps->level, ps->gameOver ? "true" : "false", ps->suddenDeath ? "true" : "false",
      ps->score[0], ps->score[1], ps->timeLeft);
  int np = ps->numPlayers;
  for (int i = 0; i < np && i < 4; i++) {
    const Player *pl = ps->players[i];
    const b2Body *body = pl ? pl->body : nullptr;
    if (!body) continue;
    char pre[224];
    const float rt = ps->respawnTimer[i];
    const bool dead = rt > 0.0f;
    snprintf(pre, sizeof pre,
             "%s{\"id\":%d,\"on_pole\":%s,\"on_ground\":%s,\"ghost\":%s,\"ghost_t\":%.3f,"
             "\"dead\":%s,\"respawn_t\":%.3f,\"hammer\":%s,",
             i ? "," : "", i + 1, pl->onPole ? "true" : "false", pl->onGround ? "true" : "false",
             pl->ghost ? "true" : "false", pl->ghost ? pl->ghostTimer : 0.0f, dead ? "true" : "false",
             dead ? rt : 0.0f, pl->pole && pl->pole->isHammer ? "true" : "false");
    PutBody(s, pre, body);
    s.pop_back();  // reopen the object to append stats + pole
    if (const PoleRiders::Stats *st = pl->stats)
      Put(s,
          ",\"stats\":{\"kicks\":%d,\"jumps\":%d,\"suicides\":%d,\"kills\":%d,"
          "\"deaths\":%d,\"goals\":%d,\"own_goals\":%d}",
          st->kicks, st->jumps, st->suicides, st->kills, st->deaths, st->goals, st->ownGoals);
    s += ",\"pole\":[";
    for (int k = 0; pl->pole && k < 10; k++)
      if (const b2Body *seg = pl->pole->segments[k]) PutBody(s, k ? ",{" : "{", seg);
    s += "]}";
  }
  s += "]";
  const b2Body *bb = ps->ball ? ps->ball->body : nullptr;
  if (bb) PutBody(s, ",\"ball\":{", bb);
  for (const b2Body *b = ps->world ? ps->world->GetBodyList() : nullptr; b; b = b->GetNext())
    if (b != bb && b->GetUserData() == &PoleRiders::Assets::BODY_BALL) {  // the zipline trolley
      Put(s, ",\"trolley\":[%.4f,%.4f]", b->GetPosition().x, b->GetPosition().y);
      break;
    }
  s += "}";
  return s;
}

std::string LevelJson(const PlayState *ps) {
  std::string s = "{\"bodies\":[";
  bool firstb = true, firstf = true, open = false;
  ForEachStaticPoly(ps, [&](const char *label, const float *v, int cnt, bool sensor) {
    if (!v) {
      if (open) s += "]}";
      Put(s, "%s{\"tag\":\"%s\",\"polys\":[", firstb ? "" : ",", label);
      firstb = false, firstf = true, open = true;
      return;
    }
    Put(s, "%s{\"sensor\":%s,\"v\":[", firstf ? "" : ",", sensor ? "true" : "false");
    firstf = false;
    for (int k = 0; k < cnt; k++) Put(s, "%s[%.4f,%.4f]", k ? "," : "", v[2 * k], v[2 * k + 1]);
    s += "]}";
  });
  if (open) s += "]}";
  s += "]}";
  return s;
}

// ---------------------------------------------------------------- Terrain (farm_env.Terrain)

void Terrain::Build(const PlayState *ps) {
  const double x1 = 45.0;
  int n = static_cast<int>((x1 - x0_) / dx_) + 1;
  std::vector<std::vector<double>> surf(n);
  ForEachStaticPoly(ps, [&](const char *label, const float *vf, int cnt, bool sensor) {
    if (!vf || strcmp(label, "ground") != 0 || sensor) return;
    double v[16];
    for (int k = 0; k < 2 * cnt; k++) v[k] = R4(vf[k]);
    for (int i = 0; i < n; i++) {
      double x = x0_ + i * dx_;
      bool any = false;
      double best = 0;
      for (int e = 0; e < cnt; e++) {  // zip(v, v[1:] + v[:1])
        double ax = v[2 * e], ay = v[2 * e + 1];
        double bx = v[2 * ((e + 1) % cnt)], by = v[2 * ((e + 1) % cnt) + 1];
        if (!(ax != bx && PyMin(ax, bx) <= x && x <= PyMax(ax, bx))) continue;
        double y = ay + (by - ay) * (x - ax) / (bx - ax);
        if (!any || y < best) best = y;  // min(ys): first of the smallest
        any = true;
      }
      if (any) surf[i].push_back(best);
    }
  });
  start_.assign(1, 0);
  vals_.clear();
  for (auto &col : surf) {
    std::sort(col.begin(), col.end());
    vals_.insert(vals_.end(), col.begin(), col.end());
    start_.push_back(static_cast<int>(vals_.size()));
  }
}

int Terrain::Col(double x) const {
  double r = std::nearbyint((x - x0_) / dx_);  // Python round(): half to even
  int last = Columns() - 1;
  if (!(r > 0)) return 0;  // also NaN
  if (r > last) return last;
  return static_cast<int>(r);
}

double Terrain::Floor(double x, double y) const {
  int i = Col(x), n = start_[i + 1] - start_[i];
  const double *c = vals_.data() + start_[i];
  double lim = y - 0.2;
  for (int k = 0; k < n; k++)  // sorted ascending: the first match is min(below)
    if (c[k] >= lim) return c[k];
  return kNone;
}

bool Terrain::Above(double x, double y, double *out) const {
  int i = Col(x), n = start_[i + 1] - start_[i];
  const double *c = vals_.data() + start_[i];
  double lim = y - 1.0;
  for (int k = n - 1; k >= 0; k--)  // the last match is max(over)
    if (c[k] < lim) return *out = c[k], true;
  return false;
}

// ---------------------------------------------------------------- observe2 (farm_env.observe2)

namespace {

// _body_feats(b, sign) with origin 0: 7 values
inline void BodyFeats(const BodyV &b, double sign, double *o) {
  double phi = -sign * b.a;
  o[0] = sign * (b.x - 0.0), o[1] = -(b.y - 0.0), o[2] = sign * b.vx, o[3] = -b.vy;
  o[4] = std::sin(phi), o[5] = std::cos(phi), o[6] = -sign * b.w;
}

// _player_feats(p, sign): 15 values
inline void PlayerFeats(const PlayerV &p, double sign, double *o) {
  BodyFeats(p.b, sign, o);
  o[7] = p.on_pole, o[8] = p.on_ground, o[9] = p.ghost || p.dead;  // absent
  if (p.npole == 0) {  // Python would raise (no pole): never happens in a live match
    for (int k = 10; k < 15; k++) o[k] = 0.0;
    return;
  }
  const BodyV &base = p.pole[0], &tip = p.pole[p.npole - 1], &mid = p.pole[p.npole / 2];
  double dx = sign * (tip.x - base.x), dy = -(tip.y - base.y);
  double ang = std::atan2(dy, dx);
  o[10] = std::sin(ang), o[11] = std::cos(ang), o[12] = -sign * mid.w;
  o[13] = sign * (tip.x - p.b.x), o[14] = -(tip.y - p.b.y);
}

// respawn_in: (respawn_t if dead else ghost_t) / 3
inline double RespawnIn(const PlayerV &p) { return (p.dead ? p.respawn_t : p.ghost_t) / 3.0; }

}  // namespace

void Observe2(const StateV &s, int agent, const Terrain &t, float *out) {
  double sign = agent == 1 ? 1.0 : -1.0;
  const PlayerV &me = s.p[agent - 1], &opp = s.p[2 - agent];
  BodyV ball = s.has_ball ? s.ball : BodyV{0, 0, 0, 0, 0, 0};
  double o[kObsSize];
  PlayerFeats(me, sign, o);
  o[15] = me.hammer, o[16] = RespawnIn(me);
  if (opp.ghost || opp.dead) {
    for (int k = 17; k < 34; k++) o[k] = 0.0;
    o[17 + 9] = 1.0;
    o[17 + 16] = RespawnIn(opp);
  } else {
    PlayerFeats(opp, sign, o + 17);
    o[32] = opp.hammer, o[33] = RespawnIn(opp);
  }
  for (int k = 0; k < kNumOffsets; k++) {
    double x = me.b.x + sign * kTerrainOffsets[k];
    o[34 + k] = PyMax(-10.0, PyMin(10.0, me.b.y - t.Floor(x, me.b.y)));
    double a;
    o[43 + k] = t.Above(x, me.b.y, &a) ? PyMax(0.0, PyMin(10.0, me.b.y - a)) : 10.0;
  }
  double bf[7];
  BodyFeats(ball, sign, bf);
  o[52] = bf[0], o[53] = bf[1], o[54] = bf[2], o[55] = bf[3];
  if (!s.has_ball) o[54] = o[55] = 0.0;  // Python's stand-in ball has int 0 velocities: sign * 0 == +0
  o[56] = s.time_left / 180.0;
  o[57] = s.score[agent - 1], o[58] = s.score[2 - agent];
  o[59] = s.sudden_death;
  o[60] = sign * (ball.x - me.b.x), o[61] = -(ball.y - me.b.y);
  for (int k = 0; k < kObsSize; k++) out[k] = static_cast<float>(o[k]);
}

// ---------------------------------------------------------------- compute_reward (spr_env)

namespace {
constexpr double kPositionRange = 10.0;  // m; keep equal to POSITION_RANGE in bridge/spr_env.py

constexpr double kHalfPole = 3.0, kTipBlend = 1.0;  // m; keep equal to spr_env.HALF_POLE / TIP_BLEND

// spr_env._position_potential: body-based good-side potential; within half a pole of the ball the pole
// tip's side replaces the body's (blended over kTipBlend m) — docs/strategy.md
double PositionPotential(const StateV &s, int agent) {
  const PlayerV &me = s.p[agent - 1];
  if (!s.has_ball || !me.present || me.ghost || me.dead) return 0.0;
  double ahead = agent == 1 ? (s.ball.x - me.b.x) : (me.b.x - s.ball.x);
  double body = PyMax(-1.0, PyMin(0.0, ahead / kPositionRange));  // [-1, 0]: behind the ball is flat
  if (me.npole <= 0) return body;
  const BodyV &tip = me.pole[me.npole - 1];
  double tip_ahead = agent == 1 ? (s.ball.x - tip.x) : (tip.x - s.ball.x);
  double tipv = PyMax(-1.0, PyMin(0.0, tip_ahead / kPositionRange));
  double wgt = PyMax(0.0, PyMin(1.0, (kHalfPole - std::fabs(ahead)) / kTipBlend));
  return body + wgt * (tipv - body);
}
inline int Died(const StateV &s, int id) {
  const PlayerV &p = s.p[id - 1];
  return p.present ? p.deaths + p.suicides : 0;
}
}  // namespace

namespace {
constexpr double kPi = 3.141592653589793;
// Backward rotation of the rider's pole between two states: the base->tip angle in its attack frame (0 = toward
// the goal it attacks, pi/2 = up), change wrapped to (-pi, pi], positive part only; 0 unless alive in both.
double BackwardRotation(const StateV &a, const StateV &b, int agent) {
  const PlayerV &p = a.p[agent - 1], &q = b.p[agent - 1];
  if (!p.present || !q.present || p.ghost || q.ghost || p.dead || q.dead || p.npole <= 0 || q.npole <= 0) return 0.0;
  const double sign = agent == 1 ? 1.0 : -1.0;
  auto ang = [&](const PlayerV &r) {
    const BodyV &base = r.pole[0], &tip = r.pole[r.npole - 1];
    return std::atan2(-(tip.y - base.y), sign * (tip.x - base.x));
  };
  double d = ang(q) - ang(p);
  while (d > kPi) d -= 2.0 * kPi;
  while (d <= -kPi) d += 2.0 * kPi;
  return d > 0.0 ? d : 0.0;
}
}  // namespace

void ComputeReward(const StateV &a, const StateV &b, const RewardWeights &w, int frame_skip, double rew[2],
                   Terms terms[2]) {
  if (!(a.has_players && b.has_players)) {
    for (int i = 0; i < 2; i++) rew[i] = 0.0, terms[i] = Terms{0.0, 0.0, 0.0, 0.0};
    return;
  }
  int dscore[2] = {b.score[0] - a.score[0], b.score[1] - a.score[1]};
  if (b.sudden_death && !a.sudden_death) dscore[0] = dscore[1] = 0;
  int died[3] = {0, Died(b, 1) - Died(a, 1), Died(b, 2) - Died(a, 2)};
  double dx = (a.has_ball && b.has_ball) ? (b.ball.x - a.ball.x) : 0.0;
  bool teleport = dscore[0] || dscore[1] || std::fabs(dx) > PyMax(5.0, 1.5 * frame_skip);
  if (teleport) dx = 0.0;
  for (int agent = 1; agent <= 2; agent++) {
    int me = agent - 1, opp = 2 - agent;
    double dpos = teleport ? 0.0 : PositionPotential(b, agent) - PositionPotential(a, agent);
    Terms t{static_cast<double>(dscore[me] - dscore[opp]), static_cast<double>(died[3 - agent] - died[agent]),
            agent == 1 ? dx : -dx, dpos};
    terms[me] = t;
    double r = 0.0;  // sum(): 0 + goal + kill + ball_progress + position, left to right
    r += w.goal * t.goal;
    if (w.death < 0.0) {
      r += w.kill * t.kill;
    } else {
      r += w.kill * died[3 - agent] - w.death * died[agent];
    }
    r += w.ball_progress * t.ball_progress;
    r += w.position * t.position;
    if (w.rotation != 0.0) r += w.rotation * BackwardRotation(a, b, agent);
    rew[me] = r;
  }
}

// ---------------------------------------------------------------- goal refund (spr_env.GoalRefund)

void GoalRefundState::Apply(const StateV &a, const StateV &b, const RewardWeights &w, int frame_skip,
                            const Terms terms[2], double rew[2]) {
  if (!(a.has_players && b.has_players)) return;  // ComputeReward gave 0
  int dscore[2] = {b.score[0] - a.score[0], b.score[1] - a.score[1]};
  if (b.sudden_death && !a.sudden_death) dscore[0] = dscore[1] = 0;
  const bool goal = dscore[0] != 0 || dscore[1] != 0;
  const int died[2] = {Died(b, 1) - Died(a, 1), Died(b, 2) - Died(a, 2)};
  const double dx = (a.has_ball && b.has_ball) ? (b.ball.x - a.ball.x) : 0.0;
  const bool reset = !goal && std::fabs(dx) > PyMax(5.0, 1.5 * frame_skip);  // post-goal ball reset + respawn
  if (goal || window) {
    for (int i = 0; i < 2; i++) {
      double r = 0.0;
      r += w.goal * terms[i].goal;
      rew[i] = r;
    }
    if (goal) {
      for (int team = 0; team < 2; team++) {  // 1v1: rider index == team
        const int x = team;
        if (dscore[team] > 0 && rec[x] && b.p[x].present && b.p[x].dead) {
          rew[x] += pen[x];
          rew[1 - x] -= cred[x];
        }
      }
      rec[0] = rec[1] = false;
      window = true;
      steps = 0;
    } else {
      steps++;
      if (reset) window = false;  // (a match-deciding goal has no reset: the window lasts to the match's end)
    }
    return;
  }
  for (int x = 0; x < 2; x++) {
    if (died[x] > 0) {
      rec[x] = true;
      pen[x] = (w.death < 0.0 ? w.kill : w.death) * died[x];
      cred[x] = w.kill * died[x];
    } else if (rec[x] && b.p[x].present && !b.p[x].dead) {
      rec[x] = false;  // respawned
    }
  }
}

// ---------------------------------------------------------------- actions (PoleRidersEnv._stick)

namespace {
struct StickTable {  // strtof(printf("%.4f", n / 1e4)) for n = -10000..10000, plus -0.0000
  float v[20001];
  StickTable() {
    char buf[32];
    for (int n = -10000; n <= 10000; n++) {
      snprintf(buf, sizeof buf, "%.4f", n / 1e4);
      v[n + 10000] = strtof(buf, nullptr);
    }
  }
};
const StickTable &Table() {
  static const StickTable t;
  return t;
}

// clamp(v) = max(-1.0, min(1.0, float(v))) -> f"{v:.4f}" -> bridge strtof + clampf
inline float Wire(double v) {
  v = PyMax(-1.0, PyMin(1.0, v));
  if (std::isnan(v)) return 0.0f;
  double r = std::nearbyint(v * 1e4);  // v is a float32 value: v * 1e4 is exact
  if (r == 0) return std::signbit(r) ? -0.0f : 0.0f;
  return Table().v[static_cast<int>(r) + 10000];
}
}  // namespace

void StickFromAction(int agent, const float act[3], float out[3]) {
  float run = act[0], ax = act[1], ay = act[2];
  double m = std::hypot(static_cast<double>(ax), static_cast<double>(ay));
  if (m > 1) {  // numpy (NEP 50): np.float32 / python float is a float32 division
    float mf = static_cast<float>(m);
    ax = ax / mf, ay = ay / mf;
  }
  float sign = agent == 1 ? 1.0f : -1.0f;
  out[0] = Wire(sign * run), out[1] = Wire(sign * ax), out[2] = Wire(-ay);
}

}  // namespace env
}  // namespace sim
