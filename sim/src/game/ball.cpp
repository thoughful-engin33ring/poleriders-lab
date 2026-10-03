// Ball construction (package A). Ball::Update/Ghost/Body/~Ball are package C (ball_update.cpp).
#include "assets.h"
#include "layout.h"

namespace PoleRiders {

// PoleRiders::Ball::Ball(float, float, b2World*, SpriteSheet*, SpriteSheet*, float) @0x10012350e
// (x is unused; the body is created by CreateBody())
Ball::Ball(float x, float y, b2World *w, SpriteSheet *s1, SpriteSheet *s2, float levelWidth) {
  (void)x;
  spin = 0.0f;
  fadeIn = 0.0f;
  heightPx = y;
  sprites = s1;   // presentation (null in the sim)
  additive = s2;
  world = w;
  frozen = false;
  levelWidthPx = levelWidth;
}

// PoleRiders::Ball::CreateBody() @0x1001235e0
// Ball body, zipline trolley (distance joint ball<->trolley), and a static BODY_ANCHOR body far left
// with a prismatic joint anchor<->trolley along x. The anchor body is not stored anywhere.
void Ball::CreateBody() {
  const float r = Assets::ratio;
  b2CircleShape circle;
  circle.m_radius = 24.0f / r;
  circle.m_p.SetZero();

  b2BodyDef bd;
  bd.position.x = 0.0f;
  bd.position.y = heightPx / r;
  bd.type = b2_dynamicBody;
  bd.userData = const_cast<int *>(&Assets::BODY_BALL);
  bd.linearDamping = Assets::ballDamping;
  bd.fixedRotation = false;

  b2FixtureDef fd;
  fd.userData = nullptr;
  fd.filter.groupIndex = 0;
  fd.isSensor = false;
  fd.density = Ball_::_density;          // 0x40866666 = 4.2
  fd.friction = Ball_::_friction;        // 0x3dcccccd = 0.1
  fd.restitution = Ball_::_restitution;  // 0x3f666666 = 0.9
  fd.shape = &circle;
  fd.filter.categoryBits = 0x20;
  fd.filter.maskBits = 0xffff;
  body = world->CreateBody(&bd);
  body->CreateFixture(&fd);

  circle.m_radius = 10.0f / r;
  bd.linearDamping = Assets::ballDamping * 3.0f;
  fd.shape = &circle;
  fd.filter.categoryBits = 0x100;
  fd.filter.maskBits = 0;
  trolley = world->CreateBody(&bd);
  trolley->CreateFixture(&fd);

  b2DistanceJointDef dj;
  dj.userData = nullptr;
  dj.localAnchorB.SetZero();
  dj.localAnchorA.SetZero();
  dj.bodyB = body;
  dj.bodyA = trolley;
  dj.collideConnected = false;
  dj.frequencyHz = 2.0f;
  dj.length = 0.0f;
  dj.dampingRatio = 0x1.99999ap-3f;  // 0x3e4ccccd = 0.2
  wire = static_cast<b2DistanceJoint *>(world->CreateJoint(&dj));

  b2PolygonShape box;
  float h = 50.0f / r;
  box.SetAsBox(h, h);
  fd.shape = &box;
  float ax = -3000.0f / r;
  fd.filter.categoryBits = 0x100;
  fd.filter.maskBits = 0;
  bd.position.x = ax;
  bd.position.y = body->GetPosition().y;
  bd.type = b2_staticBody;
  bd.userData = const_cast<int *>(&Assets::BODY_ANCHOR);
  bd.bullet = true;
  bd.allowSleep = false;
  bd.fixedRotation = true;
  b2Body *anchor = world->CreateBody(&bd);
  anchor->CreateFixture(&fd);

  b2PrismaticJointDef pj;
  pj.userData = nullptr;
  pj.maxMotorForce = 0.0f;
  pj.motorSpeed = 0.0f;
  pj.enableLimit = false;
  pj.enableMotor = false;
  pj.lowerTranslation = 0.0f;
  pj.upperTranslation = 0.0f;
  pj.bodyB = trolley;
  pj.bodyA = anchor;
  pj.collideConnected = true;
  pj.localAxis1.Set(1.0f, 0.0f);
  pj.referenceAngle = 0.0f;
  pj.localAnchorA = body->GetLocalCenter();  // sic: the ball's local center, not the anchor's
  pj.localAnchorB = anchor->GetLocalCenter();
  world->CreateJoint(&pj);
}

}  // namespace PoleRiders
