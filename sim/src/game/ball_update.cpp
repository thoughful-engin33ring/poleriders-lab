// Ball per-tick logic and teardown (package C). Ball ctor + CreateBody are package A.
#include <cmath>
#include "assets.h"
#include "layout.h"

namespace PoleRiders {

// PoleRiders::Ball::Body() @0x10012353c
b2Body *Ball::Body() { return body; }

// PoleRiders::Ball::Ghost() @0x100123550
// frozen = 1; every fixture of the ball and of the trolley gets filter {cat 0x20, mask 0, group 0}.
void Ball::Ghost() {
  frozen = true;
  b2Filter f;
  f.categoryBits = 0x20;
  f.maskBits = 0;
  f.groupIndex = 0;
  for (b2Fixture *fx = body->GetFixtureList(); fx; fx = fx->GetNext()) fx->SetFilterData(f);
  for (b2Fixture *fx = trolley->GetFixtureList(); fx; fx = fx->GetNext()) fx->SetFilterData(f);
}

// PoleRiders::Ball::Update(float) @0x100124080
void Ball::Update(float dt) {
  // spin (presentation, but it lives in the dumped object): += dt, wrapped below 2pi
  float s = spin + dt;
  spin = s;
  if (s > 0x1.921fb6p+2f) {  // ucomiss 2pi ; jbe
    do s = s + -0x1.921fb6p+2f;
    while (s > 0x1.921fb6p+2f);
    spin = s;
  }
  if (1.0f > fadeIn) fadeIn = fadeIn + dt;  // ucomiss fadeIn, 1.0 ; jbe
  if (frozen) body->SetType(b2_staticBody);

  const float r = Assets::ratio;
  // zipline spring: frequencyHz = |x| * ratio / levelWidth + 2
  wire->SetFrequency(fabsf(body->GetPosition().x) * r / levelWidthPx + 2.0f);
  // centring force: (float)(pow((double)(x / (levelWidth / ratio)), 3.0) * -35.0); pow(d, 3.0) is
  // bit-identical to (d*d)*d after this conversion (verified for all 2^32 float inputs).
  const float w = levelWidthPx / r;
  const double d = (double)(body->GetPosition().x / w);
  const float f = (float)(((d * d) * d) * -35.0);
  body->ApplyForce(b2Vec2(f, 0.0f), body->GetWorldCenter());  // inlined in the game (wakes the body)
}

// PoleRiders::Ball::~Ball() @0x1001241c4
// Destroys only the ball and the trolley: the static BODY_ANCHOR body made by CreateBody leaks
// (stays in the world), exactly as in the game.
Ball::~Ball() {
  world->DestroyBody(body);
  world->DestroyBody(trolley);
}

}  // namespace PoleRiders
