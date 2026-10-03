#!/usr/bin/env python3
"""Typed diff of two raw world dumps (bridge `farm dump` format; real game vs native sim).

Every struct is compared field by field using exact layouts from clang (gen_layouts.py ->
build/layouts.json), so padding and dead storage (unused polygon vertex slots, manifold points past
pointCount, free broadphase nodes) are ignored. Objects are matched by list position (bodies,
fixtures, joints, contacts) and by name (game objects). Pointers compare by identity: a pointer into
a dumped object becomes ("body", 3, +0x0) etc.; other pointers (vtables, allocator memory) must map
one-to-one between the two dumps.

    diff_dump.py real.json native.json [--max 40] [--ignore-objects] [--fresh]
    (python) count, lines = diff(real, native)
"""
import argparse, json, os, struct, sys

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LAYOUTS = os.path.join(SIM, "build", "layouts.json")
JOINT_CLASS = {1: "b2RevoluteJoint", 2: "b2PrismaticJoint", 3: "b2DistanceJoint", 8: "b2WeldJoint"}
SHAPE_CLASS = {0: "b2CircleShape", 1: "b2PolygonShape"}
OBJECT_CLASS = {"playstate": "PlayState", "player": "Player", "pole": "Pole", "ball": "Ball", "stats": "Stats"}
WORLD_HDR = 0x19298  # dumps carry b2World from m_flags on
# never meaningful between steps (uninitialized or rewritten before every use)
ALWAYS_SKIP = {"m_islandIndex", "m_groundBody", "m_contactManager.m_broadPhase.m_queryProxyId",
               "m_contactManager.m_broadPhase.m_pairBuffer", "m_contactManager.m_broadPhase.m_pairCapacity",
               "m_contactManager.m_broadPhase.m_pairCount"}
# joint solver temporaries: uninitialized until the joint is first solved, recomputed from the bodies
# on every solve (a real divergence shows up in the bodies)
JOINT_SCRATCH = ("m_localCenterA", "m_localCenterB", "m_invMassA", "m_invMassB", "m_invIA", "m_invIB",
                 "m_mass", "m_motorMass", "m_axis", "m_perp", "m_s1", "m_s2", "m_a1", "m_a2", "m_K", "m_u",
                 "m_gamma", "m_bias")
FMT = {"f32": "<f", "f64": "<d", "i32": "<i", "u32": "<I", "i16": "<h", "u16": "<H", "i8": "<b", "u8": "<B",
       "bool": "<B", "i64": "<q", "u64": "<Q", "ptr": "<Q"}


def load_layouts():
    if not os.path.exists(LAYOUTS):
        sys.exit(f"{LAYOUTS} missing: run sim/tools/gen_layouts.py")
    return json.load(open(LAYOUTS))


class Space:
    """Address -> (kind, index..., offset) for every object in one dump."""

    def __init__(self, d):
        self.ranges = []
        add = lambda a, n, key: self.ranges.append((int(a, 16), n, key))
        add(d["world"]["a"], 0x19368, ("world",))
        for i, b in enumerate(d["bodies"]):
            add(b["a"], len(b["raw"]) // 2, ("body", i))
            for j, f in enumerate(b["fixtures"]):
                add(f["a"], len(f["raw"]) // 2, ("fixture", i, j))
        for i, j in enumerate(d["joints"]):
            add(j["a"], len(j["raw"]) // 2, ("joint", i))
        for i, c in enumerate(d["contacts"]):
            add(c["a"], len(c["raw"]) // 2, ("contact", i))
        for k, o in d.get("objects", {}).items():
            add(o["a"], len(o["raw"]) // 2, ("object", k))

    def name(self, v):
        if v == 0:
            return "null"
        for a, n, key in self.ranges:
            if a <= v < a + n:
                return key + (v - a,)
        return None


class Differ:
    def __init__(self, A, B, limit, layouts, fresh=False, objects=True):
        self.L, self.fresh, self.objects = layouts, fresh, objects
        self.sa, self.sb = Space(A), Space(B)
        self.fwd, self.back = {}, {}   # bijection of pointers to memory outside the dump
        self.out, self.limit, self.count = [], limit, 0

    def report(self, path, msg):
        self.count += 1
        if len(self.out) < self.limit:
            self.out.append(f"{path}: {msg}")

    def ptr(self, path, fld, va, vb):
        na, nb = self.sa.name(va), self.sb.name(vb)
        if na or nb:
            if na != nb:
                self.report(path, f"{fld} -> {na} vs {nb}")
        elif self.fwd.setdefault(va, vb) != vb or self.back.setdefault(vb, va) != va:
            self.report(path, f"{fld} -> unmatched pointer {va:#x} vs {vb:#x}")

    def struct(self, path, cls, a, b, base=0, skip=None):
        """Compare the fields of class `cls` found at byte `base` of blobs a/b."""
        lay = self.L.get(cls)
        if lay is None:
            if a != b:
                self.report(path, f"raw bytes differ (no layout for {cls})")
            return
        for off, size, kind, name, *flags in lay["fields"]:
            if flags and flags[0]:  # presentation-only (layout.h `// presentation`)
                continue
            o = off - base
            if o < 0 or o + size > len(a):
                continue
            if name in ALWAYS_SKIP or (skip and skip(name, a)):
                continue
            if cls in JOINT_CLASS.values() and name.startswith(JOINT_SCRATCH):
                continue
            if not self.objects and name in ("m_contactManager.m_contactListener", "m_contactManager.m_contactFilter",
                                             "m_userData", "userData"):
                continue
            if kind == "ptr":
                self.ptr(path, name, struct.unpack_from("<Q", a, o)[0], struct.unpack_from("<Q", b, o)[0])
            elif a[o:o + size] != b[o:o + size]:
                va, vb = struct.unpack_from(FMT[kind], a, o)[0], struct.unpack_from(FMT[kind], b, o)[0]
                self.report(path, f"{name}: {va!r} vs {vb!r}")

    def run(self, A, B, objects=True):
        wa, wb = bytes.fromhex(A["world"]["hdr"]), bytes.fromhex(B["world"]["hdr"])
        self.struct("world", "b2World", wa, wb, base=WORLD_HDR)
        if "tree" in A["world"] and "tree" in B["world"]:
            free_head = struct.unpack_from("<i", wa, 0x20)[0]
            self.tree(bytes.fromhex(A["world"]["tree"]), bytes.fromhex(B["world"]["tree"]), free_head)
            if A["world"]["moves"] != B["world"]["moves"]:
                self.report("world.moves", f"{A['world']['moves'][:64]} vs {B['world']['moves'][:64]}")
        for key in ("bodies", "joints", "contacts"):
            if len(A[key]) != len(B[key]):
                self.report(key, f"count {len(A[key])} vs {len(B[key])}")
        for i, (ba, bb) in enumerate(zip(A["bodies"], B["bodies"])):
            self.struct(f"body[{i}]", "b2Body", bytes.fromhex(ba["raw"]), bytes.fromhex(bb["raw"]))
            if len(ba["fixtures"]) != len(bb["fixtures"]):
                self.report(f"body[{i}].fixtures", f"count {len(ba['fixtures'])} vs {len(bb['fixtures'])}")
            for j, (fa, fb) in enumerate(zip(ba["fixtures"], bb["fixtures"])):
                self.struct(f"body[{i}].fixture[{j}]", "b2Fixture", bytes.fromhex(fa["raw"]), bytes.fromhex(fb["raw"]))
                sa, sb = bytes.fromhex(fa["shape"]), bytes.fromhex(fb["shape"])
                self.struct(f"body[{i}].fixture[{j}].shape", SHAPE_CLASS.get(fa["shape_type"], "?"), sa, sb,
                            skip=self.poly_skip)
        for i, (ja, jb) in enumerate(zip(A["joints"], B["joints"])):
            if ja["type"] != jb["type"]:
                self.report(f"joint[{i}]", f"type {ja['type']} vs {jb['type']}")
                continue
            self.struct(f"joint[{i}]", JOINT_CLASS.get(ja["type"], "?"), bytes.fromhex(ja["raw"]), bytes.fromhex(jb["raw"]))
        for i, (ca, cb) in enumerate(zip(A["contacts"], B["contacts"])):
            self.struct(f"contact[{i}]", "b2Contact", bytes.fromhex(ca["raw"]), bytes.fromhex(cb["raw"]),
                        skip=self.manifold_skip)
        if objects:
            for k in sorted(set(A.get("objects", {})) | set(B.get("objects", {}))):
                oa, ob = A["objects"].get(k), B["objects"].get(k)
                if not oa or not ob:
                    self.report(f"objects.{k}", "missing on one side")
                    continue
                cls = OBJECT_CLASS.get(k.rstrip("0123456789"), k)
                self.struct(f"objects.{k}", cls, bytes.fromhex(oa["raw"]), bytes.fromhex(ob["raw"]))

    @staticmethod
    def poly_skip(name, blob):  # vertex/normal slots past m_vertexCount are dead storage
        if name.startswith(("m_vertices[", "m_normals[")):
            return int(name[name.index("[") + 1:name.index("]")]) >= struct.unpack_from("<i", blob, 0xa8)[0]
        return False

    @staticmethod
    def manifold_skip(name, blob):  # manifold points past pointCount are dead storage (all of it
        n = struct.unpack_from("<i", blob, 0xac)[0]  # when the contact isn't touching)
        if name.startswith("m_manifold.points["):
            return int(name[len("m_manifold.points["):][0]) >= n
        return n == 0 and name.startswith("m_manifold.") and name != "m_manifold.pointCount"

    def tree(self, a, b, free_head):
        """b2DynamicTreeNode array. Free nodes (on the free list) only carry the `next` link."""
        n = min(len(a), len(b)) // 0x28
        if len(a) != len(b):
            self.report("world.tree", f"capacity {len(a) // 0x28} vs {len(b) // 0x28}")
        free, i = set(), free_head
        while 0 <= i < n and i not in free:
            free.add(i)
            i = struct.unpack_from("<i", a, i * 0x28 + 0x18)[0]
        for k in range(n):
            na, nb = a[k * 0x28:(k + 1) * 0x28], b[k * 0x28:(k + 1) * 0x28]
            if k in free:
                if na[0x18:0x1c] != nb[0x18:0x1c]:
                    self.report(f"world.tree.node[{k}]", "free-list link differs")
            else:
                self.struct(f"world.tree.node[{k}]", "b2DynamicTreeNode", na, nb,
                            skip=lambda name, _: name == "next")


def diff(A, B, limit=40, objects=True, layouts=None, fresh=False):
    d = Differ(A, B, limit, layouts or load_layouts(), fresh, objects)
    d.run(A, B, objects)
    return d.count, d.out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--ignore-objects", action="store_true", help="pure physics: skip game objects/user data")
    ap.add_argument("--fresh", action="store_true", help="tick-0 dumps: skip never-solved joint temporaries")
    args = ap.parse_args()
    n, lines = diff(json.load(open(args.a)), json.load(open(args.b)), args.max, not args.ignore_objects,
                    fresh=args.fresh)
    print("\n".join(lines))
    print(f"{n} differences")
    sys.exit(1 if n else 0)
