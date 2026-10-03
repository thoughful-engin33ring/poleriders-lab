// Box2D world dumps in the bridge/farm.c `farm dump` format, and the reverse (rebuild a world from a
// dump). Built with -Dprivate=public to reach Box2D internals.
#include "b2dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

void hex(std::string &s, const void *mem, size_t len) {
  static const char H[] = "0123456789abcdef";
  const uint8_t *m = static_cast<const uint8_t *>(mem);
  s += '"';
  for (size_t i = 0; i < len; i++) s += H[m[i] >> 4], s += H[m[i] & 15];
  s += '"';
}

std::string ptr(const void *p) {
  char b[32];
  snprintf(b, sizeof b, "\"%p\"", p);
  return b;
}

size_t joint_size(int type) {
  switch (type) {
    case e_revoluteJoint: return sizeof(b2RevoluteJoint);
    case e_prismaticJoint: return sizeof(b2PrismaticJoint);
    case e_distanceJoint: return sizeof(b2DistanceJoint);
    case e_pulleyJoint: return sizeof(b2PulleyJoint);
    case e_mouseJoint: return sizeof(b2MouseJoint);
    case e_gearJoint: return sizeof(b2GearJoint);
    case e_lineJoint: return sizeof(b2LineJoint);
    case e_weldJoint: return sizeof(b2WeldJoint);
    case e_frictionJoint: return sizeof(b2FrictionJoint);
    default: return 0;
  }
}

size_t shape_size(int type) {
  return type == b2Shape::e_circle ? sizeof(b2CircleShape) : type == b2Shape::e_polygon ? sizeof(b2PolygonShape) : 0;
}

std::vector<uint8_t> unhex(const std::string &h) {
  std::vector<uint8_t> v(h.size() / 2);
  for (size_t i = 0; i < v.size(); i++) v[i] = (uint8_t)strtoul(h.substr(2 * i, 2).c_str(), nullptr, 16);
  return v;
}

}  // namespace

static_assert(sizeof(b2Body) == 0xc0 && sizeof(b2Fixture) == 0x50 && sizeof(b2Contact) == 0xb8,
              "Box2D layout differs from the game's");
static_assert(sizeof(b2PolygonShape) == 0xb0, "b2_maxPolygonVertices must be 9 (game build)");

std::string b2dump_world(b2World *w, unsigned long frame, const std::vector<DumpObject> &objects) {
  std::string s;
  s.reserve(1 << 16);
  char buf[128];
  snprintf(buf, sizeof buf, "{\"frame\":%lu,\"match_id\":0,\"world\":{\"a\":", frame);
  s += buf;
  s += ptr(w);
  s += ",\"hdr\":";
  hex(s, &w->m_flags, sizeof(b2World) - offsetof(b2World, m_flags));
  b2BroadPhase &bp = w->m_contactManager.m_broadPhase;
  s += ",\"tree\":";
  hex(s, bp.m_tree.m_nodes, (size_t)bp.m_tree.m_nodeCapacity * sizeof(b2DynamicTreeNode));
  s += ",\"moves\":";
  hex(s, bp.m_moveBuffer, (size_t)bp.m_moveCount * sizeof(int32));
  s += "},\"bodies\":[";
  int nb = 0;
  for (b2Body *b = w->m_bodyList; b; b = b->m_next) {
    s += nb++ ? ",{\"a\":" : "{\"a\":";
    s += ptr(b);
    s += ",\"raw\":";
    hex(s, b, sizeof(b2Body));
    s += ",\"fixtures\":[";
    int nf = 0;
    for (b2Fixture *f = b->m_fixtureList; f; f = f->m_next) {
      s += nf++ ? ",{\"a\":" : "{\"a\":";
      s += ptr(f);
      s += ",\"raw\":";
      hex(s, f, sizeof(b2Fixture));
      snprintf(buf, sizeof buf, ",\"shape_type\":%d,\"shape\":", f->m_shape->m_type);
      s += buf;
      hex(s, f->m_shape, shape_size(f->m_shape->m_type));
      s += "}";
    }
    s += "]}";
  }
  s += "],\"joints\":[";
  int nj = 0;
  for (b2Joint *j = w->m_jointList; j; j = j->m_next) {
    s += nj++ ? ",{\"a\":" : "{\"a\":";
    s += ptr(j);
    snprintf(buf, sizeof buf, ",\"type\":%d,\"raw\":", j->m_type);
    s += buf;
    hex(s, j, joint_size(j->m_type));
    s += "}";
  }
  s += "],\"contacts\":[";
  int nc = 0;
  for (b2Contact *c = w->m_contactManager.m_contactList; c; c = c->m_next) {
    s += nc++ ? ",{\"a\":" : "{\"a\":";
    s += ptr(c);
    s += ",\"raw\":";
    hex(s, c, sizeof(b2Contact));
    s += "}";
  }
  s += "],\"objects\":{";
  int no = 0;
  for (const DumpObject &o : objects) {
    if (!o.ptr || !o.len) continue;
    s += no++ ? ",\"" : "\"";
    s += o.name + "\":{\"a\":" + ptr(o.ptr) + ",\"raw\":";
    hex(s, o.ptr, o.len);
    s += "}";
  }
  s += "}}";
  return s;
}

// Loader format (tools/dump2load.py): one record per line
//   WORLD <hdr hex>
//   BODY <raw hex>                          bodies in creation order (reverse of the world list)
//   FIX <body index> <raw hex> <shape type> <shape hex>   fixtures in creation (proxy id) order
//   JOINT <type> <body index A> <body index B> <raw hex>  in creation order
//   TREE <node array hex>                   broadphase tree (restored verbatim, fixtures remapped)
//   MOVES <move buffer hex>
b2World *b2load_world(const char *path) {
  std::ifstream in(path);
  if (!in) return nullptr;
  std::string line;
  b2World *w = nullptr;
  std::vector<b2Body *> bodies;
  std::vector<uint8_t> tree, moves;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string tag;
    ls >> tag;
    if (tag == "WORLD") {
      std::string h;
      ls >> h;
      std::vector<uint8_t> hdr = unhex(h);
      b2World tmp(b2Vec2(0, 0), true);  // just for field offsets within the header
      auto at = [&](const void *field) { return hdr.data() + ((const char *)field - (const char *)&tmp.m_flags); };
      b2Vec2 g;
      bool sleep;
      memcpy(&g, at(&tmp.m_gravity), sizeof g);
      memcpy(&sleep, at(&tmp.m_allowSleep), 1);
      w = new b2World(g, sleep);
      memcpy(&w->m_warmStarting, at(&tmp.m_warmStarting), 1);
      memcpy(&w->m_continuousPhysics, at(&tmp.m_continuousPhysics), 1);
    } else if (tag == "BODY") {
      std::string h;
      ls >> h;
      std::vector<uint8_t> raw = unhex(h);
      const b2Body *r = reinterpret_cast<const b2Body *>(raw.data());
      b2BodyDef d;
      d.type = (b2BodyType)r->m_type;
      d.position = r->m_xf.position;
      d.angle = r->m_sweep.a;
      d.linearVelocity = r->m_linearVelocity;
      d.angularVelocity = r->m_angularVelocity;
      d.linearDamping = r->m_linearDamping;
      d.angularDamping = r->m_angularDamping;
      d.allowSleep = r->m_flags & b2Body::e_autoSleepFlag;
      d.awake = r->m_flags & b2Body::e_awakeFlag;
      d.fixedRotation = r->m_flags & b2Body::e_fixedRotationFlag;
      d.bullet = r->m_flags & b2Body::e_bulletFlag;
      d.active = r->m_flags & b2Body::e_activeFlag;
      d.userData = r->m_userData;
      bodies.push_back(w->CreateBody(&d));
    } else if (tag == "FIX") {
      int bi, st;
      std::string h, sh;
      ls >> bi >> h >> st >> sh;
      std::vector<uint8_t> raw = unhex(h), shape = unhex(sh);
      const b2Fixture *r = reinterpret_cast<const b2Fixture *>(raw.data());
      b2PolygonShape poly;
      b2CircleShape circle;
      b2Shape *s = st == b2Shape::e_polygon ? (b2Shape *)&poly : (b2Shape *)&circle;
      memcpy((char *)s + 8, shape.data() + 8, shape.size() - 8);  // everything but the vtable
      b2FixtureDef d;
      d.shape = s;
      d.userData = r->m_userData;
      d.friction = r->m_friction;
      d.restitution = r->m_restitution;
      d.density = r->m_density;
      d.isSensor = r->m_isSensor;
      d.filter = r->m_filter;
      bodies[bi]->CreateFixture(&d);
    } else if (tag == "JOINT") {
      int type, ia, ib;
      std::string h;
      ls >> type >> ia >> ib >> h;
      std::vector<uint8_t> raw = unhex(h);
      const b2Joint *rj = reinterpret_cast<const b2Joint *>(raw.data());
      b2Body *A = bodies[ia], *B = bodies[ib];
      if (type == e_revoluteJoint) {
        const b2RevoluteJoint *r = reinterpret_cast<const b2RevoluteJoint *>(raw.data());
        b2RevoluteJointDef d;
        d.bodyA = A, d.bodyB = B, d.collideConnected = rj->m_collideConnected, d.userData = rj->m_userData;
        d.localAnchorA = r->m_localAnchor1, d.localAnchorB = r->m_localAnchor2;
        d.referenceAngle = r->m_referenceAngle;
        d.enableLimit = r->m_enableLimit, d.lowerAngle = r->m_lowerAngle, d.upperAngle = r->m_upperAngle;
        d.enableMotor = r->m_enableMotor, d.motorSpeed = r->m_motorSpeed, d.maxMotorTorque = r->m_maxMotorTorque;
        w->CreateJoint(&d);
      } else if (type == e_prismaticJoint) {
        const b2PrismaticJoint *r = reinterpret_cast<const b2PrismaticJoint *>(raw.data());
        b2PrismaticJointDef d;
        d.bodyA = A, d.bodyB = B, d.collideConnected = rj->m_collideConnected, d.userData = rj->m_userData;
        d.localAnchorA = r->m_localAnchor1, d.localAnchorB = r->m_localAnchor2;
        d.localAxis1 = r->m_localXAxis1;
        d.referenceAngle = r->m_refAngle;
        d.enableLimit = r->m_enableLimit, d.lowerTranslation = r->m_lowerTranslation;
        d.upperTranslation = r->m_upperTranslation;
        d.enableMotor = r->m_enableMotor, d.maxMotorForce = r->m_maxMotorForce, d.motorSpeed = r->m_motorSpeed;
        w->CreateJoint(&d);
      } else if (type == e_distanceJoint) {
        const b2DistanceJoint *r = reinterpret_cast<const b2DistanceJoint *>(raw.data());
        b2DistanceJointDef d;
        d.bodyA = A, d.bodyB = B, d.collideConnected = rj->m_collideConnected, d.userData = rj->m_userData;
        d.localAnchorA = r->m_localAnchor1, d.localAnchorB = r->m_localAnchor2;
        d.length = r->m_length, d.frequencyHz = r->m_frequencyHz, d.dampingRatio = r->m_dampingRatio;
        w->CreateJoint(&d);
      } else if (type == e_weldJoint) {
        const b2WeldJoint *r = reinterpret_cast<const b2WeldJoint *>(raw.data());
        b2WeldJointDef d;
        d.bodyA = A, d.bodyB = B, d.collideConnected = rj->m_collideConnected, d.userData = rj->m_userData;
        d.localAnchorA = r->m_localAnchorA, d.localAnchorB = r->m_localAnchorB;
        d.referenceAngle = r->m_referenceAngle;
        w->CreateJoint(&d);
      } else {
        fprintf(stderr, "b2load_world: joint type %d not supported\n", type);
        return nullptr;
      }
    } else if (tag == "TREE") {
      std::string h;
      ls >> h;
      tree = unhex(h);
    } else if (tag == "MOVES") {
      std::string h;
      ls >> h;
      moves = unhex(h);
    }
  }
  if (!w) return nullptr;
  // Restore the broadphase verbatim: fat AABBs depend on the objects' movement history and decide
  // when contacts are created (and so the contact list order). Node user data points at fixtures.
  b2DynamicTree &t = w->m_contactManager.m_broadPhase.m_tree;
  if (!tree.empty()) {
    if (tree.size() != (size_t)t.m_nodeCapacity * sizeof(b2DynamicTreeNode)) {
      fprintf(stderr, "b2load_world: tree capacity %zu vs %d\n", tree.size() / sizeof(b2DynamicTreeNode),
              t.m_nodeCapacity);
      return nullptr;
    }
    memcpy(t.m_nodes, tree.data(), tree.size());
    for (int i = 0; i < t.m_nodeCapacity; i++) t.m_nodes[i].userData = nullptr;
    for (b2Body *b = w->m_bodyList; b; b = b->m_next)  // leaves: same proxy ids, native fixtures
      for (b2Fixture *f = b->m_fixtureList; f; f = f->m_next) t.m_nodes[f->m_proxyId].userData = f;
  }
  b2BroadPhase &bp = w->m_contactManager.m_broadPhase;
  if (moves.size() / 4 != (size_t)bp.m_moveCount) {
    fprintf(stderr, "b2load_world: move buffer %zu vs %d\n", moves.size() / 4, bp.m_moveCount);
  } else {
    memcpy(bp.m_moveBuffer, moves.data(), moves.size());
  }
  return w;
}
