// Native ports of the Python RL-side functions, value-exact with them:
//   snapshot_ps (bridge/sprbridge.c, farm `farm act` / `farm reset` replies) -> SnapshotJson, ExtractState
//   Terrain, observe2 (bridge/farm_env.py), _body_feats, _player_feats, compute_reward,
//   PoleRidersEnv._stick (bridge/spr_env.py) and the bridge's `farm act` float parsing -> StickFromAction
//
// Exactness: the Python side sees the match through the snapshot JSON, where every float is printed
// with %.4f (time_left, ghost_t, respawn_t: %.3f) and parsed back as a double. ExtractState reproduces that
// rounding (double(f) * 1e4 is exact for a float, so nearbyint() = printf's round-half-even, and
// n / 1e4 = strtod's correctly rounded value); observe2 / compute_reward then do the same double
// operations in the same order as the Python code, and the results are cast to float32 like numpy.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class b2World;
namespace PoleRiders {
struct PlayState;
}

namespace sim {
namespace env {

constexpr int kObsSize = 62;  // farm_env.OBS2_SIZE
constexpr int kNumOffsets = 9;
extern const int kTerrainOffsets[kNumOffsets];  // farm_env.TERRAIN_OFFSETS

struct BodyV {  // put_body(): x y a vx vy w, as parsed from %.4f
  double x, y, a, vx, vy, w;
};

struct PlayerV {
  bool present = false;  // listed in "players" (Player with a body)
  bool on_pole, on_ground, ghost, hammer;
  double ghost_t;  // %.3f, 0 unless ghost
  bool dead;        // PlayState::respawnTimer[i] > 0: killed, waiting to respawn (corpse stays in the world)
  double respawn_t;  // %.3f, respawnTimer[i] while dead, else 0
  BodyV b;
  bool has_stats;
  int deaths, suicides;
  int npole;
  BodyV pole[10];
};

struct StateV {
  bool has_players = false;  // st.get("players") is a non-empty list
  int level;
  bool game_over, sudden_death;
  int score[2];
  double time_left;  // %.3f
  PlayerV p[2];      // ids 1 and 2 (players 3/4 never appear in 1v1 training)
  bool has_ball;
  BodyV ball;
};

void ExtractState(const PoleRiders::PlayState *ps, StateV *out);

// bridge snapshot_ps(ps, ..., with_bodies = 0): the JSON the farm returns per match. Returns length.
std::string SnapshotJson(const PoleRiders::PlayState *ps, unsigned long frame, unsigned long match_id);

// bridge `level` command: static bodies' polygon fixtures in world meters (arenas/arenaN.json).
std::string LevelJson(const PoleRiders::PlayState *ps);

// farm_env.Terrain built from the world's static "ground" polygons (vertices rounded like the
// arena JSON), same sampling / queries.
class Terrain {
 public:
  static constexpr double kNone = 50.0;
  void Build(const PoleRiders::PlayState *ps);
  double Floor(double x, double y) const;
  bool Above(double x, double y, double *out) const;  // false = None
  int Columns() const { return static_cast<int>(start_.size()) - 1; }
  const double *Column(int i, int *n) const {
    *n = start_[i + 1] - start_[i];
    return vals_.data() + start_[i];
  }

 private:
  double x0_ = -45.0, dx_ = 0.05;
  std::vector<int> start_;
  std::vector<double> vals_;
  int Col(double x) const;
};

void Observe2(const StateV &s, int agent, const Terrain &t, float *out);

struct RewardWeights {
  double goal = 1.0, kill = 0.3, ball_progress = 0.003, position = 0.1;  // spr_env.DEFAULT_REWARD
  // own deaths: < 0 = the same as kill (zero-sum kill term, the default); else kills are paid `kill` each
  // and own deaths (incl. suicides) cost `death` each (e.g. a killer specialist: kill 2, death 1)
  double death = -1.0;
  // backward pole rotation (counter-clockwise in the rider's attack frame, radians per step, positive part only;
  // 0 while dead / ghost): a training incentive, 0 = off (then the reward is bit-identical to before)
  double rotation = 0.0;
};
struct Terms {
  double goal, kill, ball_progress, position;
};
// compute_reward(a, b, w, frame_skip): rew[0..1] for agents 1, 2; terms per agent.
void ComputeReward(const StateV &a, const StateV &b, const RewardWeights &w, int frame_skip, double rew[2],
                   Terms terms[2]);

// Goal refund (user, 2026-10-02; spr_env.GoalRefund mirrors it line by line). Per match, applied after
// ComputeReward to rew only (terms stay the raw events):
//  (a) from the step a goal is scored until the step in which the game's post-goal reset happens (the ball is
//      rebuilt at the centre and both riders respawn in the same tick, ~1.4 s later: seen as the ball jumping
//      more than max(5, 1.5 * frame_skip) m in one step without a goal) every reward term except the goal is 0
//      for both riders (kill/death, ball progress, position potential, rotation bonus). The goal step and the
//      reset step already have no position/ball-progress term (ComputeReward's teleport rule), and the step
//      after the reset starts from the respawned state, so masking the potential differences in between
//      leaves no jump: it is as if the goal state were terminal and the potential restarted at the respawn.
//      A match-deciding goal (score limit / sudden death) has no reset: the match plays on ~5 s (1 s of clock,
//      slow motion, fade) until game_over, all of it inside the window. A new match starts with no window.
//  (b) a death (killed or suicide) whose penalty was paid (outside a window) is refunded in full at the goal
//      step if the victim's own team scores while the victim is still dead (before its respawn), and the kill
//      reward the other rider got for it is taken back at the same step. Records are cleared when the victim
//      is seen alive again and at every goal.
// Zero-sum kill term (death < 0): penalty = credit = kill * deaths; with a death weight: penalty = death *
// deaths, credit = kill * deaths.
struct GoalRefundState {
  bool window = false;  // inside a goal -> respawn window
  int steps = 0;        // steps since the goal (window only)
  bool rec[2] = {false, false};  // rider i's last death was paid and i has not respawned since
  double pen[2] = {0.0, 0.0}, cred[2] = {0.0, 0.0};
  void Reset() { *this = GoalRefundState(); }
  // a -> b with the step's rewards rew (from ComputeReward, same a, b, w); rewrites rew in place
  void Apply(const StateV &a, const StateV &b, const RewardWeights &w, int frame_skip, const Terms terms[2],
             double rew[2]);
};

// PoleRidersEnv._stick(agent, (run, aim_x, aim_y)) for float32 actions (numpy semantics), formatted
// "%.4f" and parsed by the bridge (strtof + clamp): the exact stick floats the game receives.
void StickFromAction(int agent, const float act[3], float out[3]);

}  // namespace env
}  // namespace sim
