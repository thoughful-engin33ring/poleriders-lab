// Stats counters (package B). Ctor/dtor are package A.
#include "layout.h"

namespace PoleRiders {

// PoleRiders::Stats::AddKick() @0x1001411a6
void Stats::AddKick() { kicks++; }
// PoleRiders::Stats::AddJump() @0x1001411ba
void Stats::AddJump() { jumps++; }
// PoleRiders::Stats::AddSuicide() @0x1001411ce
void Stats::AddSuicide() { suicides++; }
// PoleRiders::Stats::SetTimeLeft(float) @0x1001411e4
void Stats::SetTimeLeft(float t) { timeLeft = t; }
// PoleRiders::Stats::AddKill() @0x1001411fa
void Stats::AddKill() { kills++; }
// PoleRiders::Stats::AddDeath() @0x10014120e
void Stats::AddDeath() { deaths++; }
// PoleRiders::Stats::IncrementTimeNearestBall(float) @0x100141224
void Stats::IncrementTimeNearestBall(float dt) { timeNearestBall = dt + timeNearestBall; }
// PoleRiders::Stats::AddGoal() @0x100141234
void Stats::AddGoal() { goals++; }
// PoleRiders::Stats::AddOwnGoal() @0x10014123e
void Stats::AddOwnGoal() { ownGoals++; }
// PoleRiders::Stats::IncrementOffenseRatio(float) @0x100141268
void Stats::IncrementOffenseRatio(float x) { offenseRatio = x + offenseRatio; }
// PoleRiders::Stats::IncrementTimeInGoal(float) @0x100141290
void Stats::IncrementTimeInGoal(float dt) { timeInGoal = dt + timeInGoal; }
// PoleRiders::Stats::IncrementTimeInEndzone(float) @0x1001412a0
void Stats::IncrementTimeInEndzone(float dt) { timeInEndzone = dt + timeInEndzone; }

}  // namespace PoleRiders
