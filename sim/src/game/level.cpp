// Level terrain builders (package A). Level1 (Kalgoorlie, the training map) and Level2 (Flatirons)
// are ported; Level3/4/5 later.
// Every body/fixture is created in the original order (list order and broadphase proxy ids depend
// on it). The original reuses stack b2PolygonShape / b2FixtureDef / b2BodyDef objects between
// creations; the values carried over are reproduced here by reusing the same objects.
#include "assets.h"
#include "layout.h"

namespace PoleRiders {

// PoleRiders::Level1::Level1(float, float, b2World*) @0x100128884
Level1::Level1(float x, float y, b2World *w) {
  world = w;
  CreateBody(x, y);
}

// PoleRiders::Level1::CreateBody(float, float) @0x1001276a6
void Level1::CreateBody(float x, float y) {
  const float r = Assets::ratio;
  b2PolygonShape s1;  // -0xf8

  // central ground slab (BODY_GROUND) + its twin BODY_NONE body at the same place
  s1.SetAsBox(1876.0f / (r + r), 256.0f / (r + r));
  b2FixtureDef fd;  // -0x1a8
  fd.userData = nullptr;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xffff;
  fd.filter.groupIndex = 0;
  fd.isSensor = false;
  fd.density = 0x1.4p+2f;         // 5.0
  fd.friction = 0x1.333334p+0f;   // 0x3f99999a = 1.2
  fd.restitution = 0.0f;
  fd.shape = &s1;

  b2BodyDef bd1;  // -0x138
  bd1.position.x = x / r;
  bd1.position.y = (y + 128.0f) / r;
  bd1.userData = const_cast<int *>(&Assets::BODY_GROUND);
  bd1.type = b2_staticBody;
  b2BodyDef bd2;  // -0x178
  bd2.position = bd1.position;
  bd2.userData = const_cast<int *>(&Assets::BODY_NONE);
  bd2.type = b2_staticBody;
  b2Body *ground = world->CreateBody(&bd1);
  b2Body *none = world->CreateBody(&bd2);

  b2Filter f;  // -0x1b0
  f.groupIndex = 0;
  f.categoryBits = 1;
  f.maskBits = 0xfffe;
  ground->CreateFixture(&fd)->SetFilterData(f);

  s1.SetAsBox(1876.0f / (r + r), 256.0f / (r + r), b2Vec2(0.0f, 0.25f), 0.0f);
  fd.shape = &s1;
  none->CreateFixture(&fd)->SetFilterData(f);

  // side slabs at +-1440 px (twins at +-1432, 0.25 m lower)
  s1 = b2PolygonShape();
  s1.SetAsBox(788.0f / (r + r), 256.0f / (r + r), b2Vec2(-1440.0f / r, 96.0f / r), 0.0f);
  fd.shape = &s1;
  ground->CreateFixture(&fd)->SetFilterData(f);
  s1.SetAsBox(788.0f / (r + r), 256.0f / (r + r), b2Vec2(-1432.0f / r, 96.0f / r + 0.25f), 0.0f);
  fd.shape = &s1;
  none->CreateFixture(&fd)->SetFilterData(f);
  s1 = b2PolygonShape();
  s1.SetAsBox(788.0f / (r + r), 256.0f / (r + r), b2Vec2(1440.0f / r, 96.0f / r), 0.0f);
  fd.shape = &s1;
  ground->CreateFixture(&fd)->SetFilterData(f);
  s1.SetAsBox(788.0f / (r + r), 256.0f / (r + r), b2Vec2(1432.0f / r, 96.0f / r + 0.25f), 0.0f);
  fd.shape = &s1;
  none->CreateFixture(&fd)->SetFilterData(f);

  // quads between the slabs (-0x1d0: v[0..3])
  b2Vec2 v[4];
  {
    float a = -938.0f / r, b = -128.0f / r;
    v[3].Set(a, b);
    v[2].x = -950.0f / r;
    v[2].y = 6.0f / r + b;
    v[1].x = -1034.0f / r;
    v[1].y = 90.0f / r + b;
    v[0].x = a;
    v[0].y = 256.0f / r + b;
    s1 = b2PolygonShape();
    s1.Set(v, 4);
    fd.shape = &s1;
    ground->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = -1046.0f / r, b = -128.0f / r;
    v[3].x = a;
    v[3].y = 96.0f / r + b;
    v[2].x = a;
    float c = 256.0f / r + b;
    v[2].y = c;
    v[1].x = -938.0f / r;
    v[1].y = c;
    v[0].x = -1034.0f / r;
    v[0].y = 90.0f / r + b;
    s1 = b2PolygonShape();
    s1.Set(v, 4);
    fd.shape = &s1;
    ground->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = 938.0f / r, b = -128.0f / r;
    v[3].Set(a, b);
    v[0].x = 950.0f / r;
    v[0].y = 6.0f / r + b;
    v[1].x = 1034.0f / r;
    v[1].y = 90.0f / r + b;
    v[2].x = a;
    v[2].y = 256.0f / r + b;
    s1 = b2PolygonShape();
    s1.Set(v, 4);
    fd.shape = &s1;
    ground->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = 1046.0f / r, b = -128.0f / r;
    v[3].x = a;
    v[3].y = 96.0f / r + b;
    v[0].x = a;
    float c = 256.0f / r + b;
    v[0].y = c;
    v[1].x = 938.0f / r;
    v[1].y = c;
    v[2].x = 1034.0f / r;
    v[2].y = 90.0f / r + b;
    s1 = b2PolygonShape();
    s1.Set(v, 4);
    fd.shape = &s1;
    ground->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = -938.0f / r, b = -128.0f / r;
    v[0].x = a;
    v[0].y = 16.0f / r + b;
    v[1].x = a;
    float c = 272.0f / r + b;
    v[1].y = c;
    float e = -1040.0f / r;
    v[2].Set(e, c);
    v[3].x = e;
    v[3].y = 112.0f / r + b;
    s1 = b2PolygonShape();
    s1.Set(v, 4);
    fd.shape = &s1;
    none->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = 938.0f / r, b = -128.0f / r;
    v[2].x = a;
    v[2].y = 16.0f / r + b;
    float e = 1040.0f / r;
    v[3].x = e;
    v[3].y = 112.0f / r + b;
    v[0].x = e;
    float c = 272.0f / r + b;
    v[0].y = c;
    v[1].Set(a, c);
    s1 = b2PolygonShape();
    s1.Set(v, 4);
    fd.shape = &s1;
    none->CreateFixture(&fd)->SetFilterData(f);
  }

  // walls, goal ledges, deleters: fresh shapes/def, one body per piece
  b2PolygonShape s2;  // -0x280
  b2PolygonShape s3;  // -0x330
  b2BodyDef bd3;      // -0x370
  fd.userData = nullptr;
  fd.isSensor = false;
  fd.restitution = 0.0f;
  fd.density = 0.0f;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xffff;
  fd.filter.groupIndex = 0;
  fd.density = 1.0f;
  fd.friction = 0.5f;
  fd.restitution = 0.0f;
  fd.shape = &s2;
  s2.SetAsBox(512.0f / r, 360.0f / r);
  f.maskBits = 0xffff;
  f.groupIndex = 0;
  f.categoryBits = 1;

  b2Body *b;
  bd3.userData = const_cast<int *>(&Assets::BODY_WALL);
  bd3.position.x = -1812.0f / r;
  bd3.position.y = -138.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  s2.SetAsBox(512.0f / r, 360.0f / r);
  bd3.userData = const_cast<int *>(&Assets::BODY_WALL);
  bd3.position.x = 1812.0f / r;
  bd3.position.y = -138.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  fd.shape = &s3;
  s3.SetAsBox(512.0f / r, 32.0f / r);
  bd3.position.x = -1812.0f / r;
  bd3.position.y = -480.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetUserData(const_cast<int *>(&Assets::BODY_GROUND));
  b->SetType(b2_staticBody);

  fd.shape = &s3;
  bd3.position.x = 1812.0f / r;
  bd3.position.y = -480.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetUserData(const_cast<int *>(&Assets::BODY_GROUND));
  b->SetType(b2_staticBody);

  s2.SetAsBox(512.0f / r, 2000.0f / r);
  bd3.userData = const_cast<int *>(&Assets::BODY_DELETER);
  bd3.position.x = 2168.0f / r;
  bd3.position.y = 0.0f / r;
  b = world->CreateBody(&bd3);
  fd.isSensor = true;
  f.maskBits = 0xfe7e;
  f.categoryBits = 1;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xfe7e;
  fd.filter.groupIndex = f.groupIndex;
  fd.shape = &s2;
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  bd3.position.x = -2168.0f / r;
  bd3.position.y = 0.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  s2.SetAsBox(3000.0f / r, 256.0f / r);
  bd3.position.x = 0.0f;
  bd3.position.y = 900.0f / r;
  fd.shape = &s2;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);
}

// PoleRiders::Level2::Level2(float, float, b2World*) @0x1001298e6
Level2::Level2(float x, float y, b2World *w) {
  world = w;
  CreateBody(x, y);
}

// PoleRiders::Level2::CreateBody(float, float) @0x1001288a0
void Level2::CreateBody(float x, float y) {
  const float r = Assets::ratio;
  b2PolygonShape s1;  // -0xf0 (re-constructed before most uses; Set/SetAsBox overwrite what matters)

  // central ground slab (BODY_GROUND) + its twin BODY_NONE body at the same place
  s1.SetAsBox(1024.0f / (r + r), 256.0f / (r + r), b2Vec2(0.0f / r, 72.0f / r), 0.0f);
  b2FixtureDef fd;  // -0x1a0
  fd.shape = &s1;
  fd.userData = nullptr;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xffff;
  fd.filter.groupIndex = 0;
  fd.isSensor = false;
  fd.density = 0x1.4p+2f;         // 5.0
  fd.friction = 0x1.333334p+0f;   // 0x3f99999a = 1.2
  fd.restitution = 0.0f;

  b2BodyDef bd1;  // -0x130
  bd1.position.x = x / r;
  bd1.position.y = (y + 128.0f) / r;
  bd1.userData = const_cast<int *>(&Assets::BODY_GROUND);
  bd1.type = b2_staticBody;
  b2BodyDef bd2;  // -0x170
  bd2.position = bd1.position;
  bd2.userData = const_cast<int *>(&Assets::BODY_NONE);
  bd2.type = b2_staticBody;
  b2Body *ground = world->CreateBody(&bd1);
  b2Body *none = world->CreateBody(&bd2);

  b2Filter f;  // -0x1a8
  f.groupIndex = 0;
  f.categoryBits = 1;
  f.maskBits = 0xfffe;
  ground->CreateFixture(&fd)->SetFilterData(f);

  s1.SetAsBox(1088.0f / (r + r), 256.0f / (r + r), b2Vec2(0.0f, 72.0f / r + 0.25f), 0.0f);
  fd.shape = &s1;
  none->CreateFixture(&fd)->SetFilterData(f);

  {  // small blocks at +-1584 px (no filter change)
    float h = 32.0f / (r + r);
    s1 = b2PolygonShape();
    s1.SetAsBox(h, h, b2Vec2(1584.0f / r, -144.0f / r), 0.0f);
    fd.shape = &s1;
    ground->CreateFixture(&fd);
    h = 32.0f / (r + r);
    s1 = b2PolygonShape();
    s1.SetAsBox(h, h, b2Vec2(-1584.0f / r, -144.0f / r), 0.0f);
    fd.shape = &s1;
    ground->CreateFixture(&fd);
  }

  b2Vec2 v[5];  // -0x1d0
  {
    float a = -512.0f / r, b = -128.0f / r;
    float c = 72.0f / r + b;
    v[0].Set(a, c);
    v[1].x = a;
    float d = 256.0f / r + b;
    v[1].y = d;
    float e = -2504.0f / r;
    v[2].Set(e, d);
    v[3].Set(e, b);
    v[4].Set(-704.0f / r, b);
    s1 = b2PolygonShape();
    s1.Set(v, 5);
    fd.shape = &s1;
    ground->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = -544.0f / r, b = -128.0f / r;
    float c = 88.0f / r + b;
    v[0].Set(a, c);
    v[1].x = a;
    float d = 272.0f / r + b;
    v[1].y = d;
    float e = -2536.0f / r;
    v[2].Set(e, d);
    float g = -112.0f / r;
    v[3].Set(e, g);
    v[4].Set(-736.0f / r, g);
    s1 = b2PolygonShape();
    s1.Set(v, 5);
    fd.shape = &s1;
    none->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = 512.0f / r, b = -128.0f / r;
    float c = 256.0f / r + b;
    v[0].Set(a, c);
    v[1].Set(a, 72.0f / r + b);
    v[2].Set(704.0f / r, b);
    float e = 2504.0f / r;
    v[3].Set(e, b);
    v[4].Set(e, c);
    s1 = b2PolygonShape();
    s1.Set(v, 5);
    fd.shape = &s1;
    ground->CreateFixture(&fd)->SetFilterData(f);
  }
  {
    float a = 544.0f / r, b = -128.0f / r;
    float c = 272.0f / r + b;
    v[0].Set(a, c);
    v[1].Set(a, 88.0f / r + b);
    float g = -112.0f / r;
    v[2].Set(736.0f / r, g);
    float e = 2536.0f / r;
    v[3].Set(e, g);
    v[4].Set(e, c);
    s1 = b2PolygonShape();
    s1.Set(v, 5);
    fd.shape = &s1;
    none->CreateFixture(&fd)->SetFilterData(f);
  }

  // walls, goal ledges, deleters: fresh shapes/def, one body per piece
  b2PolygonShape s2;  // -0x280
  b2PolygonShape s3;  // -0x330
  b2BodyDef bd3;      // -0x370
  fd.userData = nullptr;
  fd.isSensor = false;
  fd.restitution = 0.0f;
  fd.density = 0.0f;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xffff;
  fd.filter.groupIndex = 0;
  fd.density = 1.0f;
  fd.friction = 0.5f;
  fd.restitution = 0.0f;
  fd.shape = &s2;
  s2.SetAsBox(512.0f / r, 360.0f / r);
  f.maskBits = 0xffff;
  f.groupIndex = 0;
  f.categoryBits = 1;

  b2Body *b;
  bd3.userData = const_cast<int *>(&Assets::BODY_WALL);
  bd3.position.x = -2112.0f / r;
  bd3.position.y = -238.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  s2.SetAsBox(512.0f / r, 360.0f / r);
  bd3.userData = const_cast<int *>(&Assets::BODY_WALL);
  bd3.position.x = 2112.0f / r;
  bd3.position.y = -238.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  fd.shape = &s3;
  s3.SetAsBox(512.0f / r, 32.0f / r);
  bd3.position.x = -2112.0f / r;
  bd3.position.y = -580.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetUserData(const_cast<int *>(&Assets::BODY_GROUND));
  b->SetType(b2_staticBody);

  fd.shape = &s3;
  bd3.position.x = 2112.0f / r;
  bd3.position.y = -580.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetUserData(const_cast<int *>(&Assets::BODY_GROUND));
  b->SetType(b2_staticBody);

  bd3.position.x = 1364.0f / r;
  bd3.position.y = -380.0f / r;
  f.groupIndex = 0;
  f.categoryBits = 1;
  f.maskBits = 0x20;
  s1 = b2PolygonShape();
  s1.SetAsBox(128.0f / (r + r), 1024.0f / (r + r));
  fd.shape = &s1;
  fd.filter = f;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetUserData(const_cast<int *>(&Assets::BODY_WALL));
  b->SetType(b2_staticBody);

  bd3.position.x = -1364.0f / r;
  bd3.position.y = -380.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetUserData(const_cast<int *>(&Assets::BODY_WALL));
  b->SetType(b2_staticBody);

  s2.SetAsBox(512.0f / r, 2000.0f / r);
  bd3.userData = const_cast<int *>(&Assets::BODY_DELETER);
  bd3.position.x = 2528.0f / r;
  bd3.position.y = 0.0f / r;
  b = world->CreateBody(&bd3);
  fd.isSensor = true;
  f.maskBits = 0xfe7e;
  f.categoryBits = 1;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xfe7e;
  fd.filter.groupIndex = f.groupIndex;
  fd.shape = &s2;
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  bd3.position.x = -2528.0f / r;
  bd3.position.y = 0.0f / r;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);

  s2.SetAsBox(3000.0f / r, 256.0f / r);
  bd3.position.x = 0.0f;
  bd3.position.y = 900.0f / r;
  fd.shape = &s2;
  b = world->CreateBody(&bd3);
  b->CreateFixture(&fd);
  b->SetType(b2_staticBody);
}

}  // namespace PoleRiders
