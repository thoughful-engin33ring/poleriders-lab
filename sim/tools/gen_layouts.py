#!/usr/bin/env python3
"""Exact field layouts (offset, size, kind, name) of Box2D and game classes, from clang's
-fdump-record-layouts, for the typed dump diff (diff_dump.py). Padding is whatever no field covers.

    gen_layouts.py [out.json]      (default: sim/build/layouts.json)
"""
import json, os, re, subprocess, sys, tempfile

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CLASSES = ["b2World", "b2Body", "b2Fixture", "b2PolygonShape", "b2CircleShape", "b2Contact",
           "b2RevoluteJoint", "b2PrismaticJoint", "b2DistanceJoint", "b2WeldJoint", "b2DynamicTreeNode"]
GAME = ["PoleRiders::PlayState", "PoleRiders::Player", "PoleRiders::Pole", "PoleRiders::Ball", "PoleRiders::Stats"]

PRIM = {"float32": (4, "f32"), "float": (4, "f32"), "double": (8, "f64"), "int32": (4, "i32"), "int": (4, "i32"),
        "uint32": (4, "u32"), "unsigned int": (4, "u32"), "int16": (2, "i16"), "uint16": (2, "u16"),
        "short": (2, "i16"), "unsigned short": (2, "u16"), "int8": (1, "i8"), "uint8": (1, "u8"),
        "char": (1, "i8"), "unsigned char": (1, "u8"), "bool": (1, "bool"), "_Bool": (1, "bool"), "long": (8, "i64"), "unsigned long": (8, "u64"), "size_t": (8, "u64"), "int64_t": (8, "i64"),
        "uint64_t": (8, "u64"), "int32_t": (4, "i32"), "uint32_t": (4, "u32"), "int16_t": (2, "i16"),
        "uint16_t": (2, "u16"), "int8_t": (1, "i8"), "uint8_t": (1, "u8"), "float32_t": (4, "f32")}
ENUMS = {"b2BodyType", "b2LimitState", "b2JointType", "Type", "b2Shape::Type", "b2Manifold::Type",
         "enum b2BodyType", "enum b2LimitState", "enum b2JointType", "enum b2Shape::Type",
         "enum b2Manifold::Type"}

LINE = re.compile(r"^\s*(\d+)(?::(\d+)-(\d+))? \|(\s*)(.*)$")


def run_clang(game):
    src = "#include <Box2D/Box2D.h>\n"
    classes = list(CLASSES)
    if game and os.path.exists(os.path.join(SIM, "src/game/layout.h")):
        src += '#include "src/game/layout.h"\n'
        classes += GAME
    src += "".join(f"int s{i} = sizeof({c});\n" for i, c in enumerate(classes))
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(src)
    r = subprocess.run(["clang++", "-std=c++17", "-I", SIM, "-Xclang", "-fdump-record-layouts", "-fsyntax-only",
                        f.name], capture_output=True, text=True)
    os.unlink(f.name)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    return r.stdout, classes


def sections(text):
    """{record name: [(offset, depth, decl), ...], sizeof}"""
    out = {}
    for chunk in text.split("*** Dumping AST Record Layout")[1:]:
        rows, size = [], None
        for ln in chunk.split("\n"):
            m = LINE.match(ln)
            if m:
                rows.append((int(m.group(1)), len(m.group(4)), m.group(5).strip()))
            s = re.search(r"\[sizeof=(\d+)", ln)
            if s and size is None:
                size = int(s.group(1))
        if rows:
            name = re.sub(r"^(class|struct|union) ", "", rows[0][2])
            out.setdefault(name, (rows, size))
    return out


def leaves(name, secs, base=0, prefix=""):
    rows, size = secs[name]
    out, stack = [], []  # stack: (depth, name) of the enclosing nested fields
    for i, (off, depth, decl) in enumerate(rows[1:], 1):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        nxt = rows[i + 1][1] if i + 1 < len(rows) else -1
        path = prefix + "".join(n + "." for _, n in stack if n)
        if nxt > depth:  # nested struct or base class: its members follow, deeper
            m = re.match(r"^.*?(\w+)$", decl)
            stack.append((depth, "" if "base)" in decl else (m.group(1) if m else "")))
            continue
        o = base + off
        if "vtable pointer" in decl:
            out.append([o, 8, "ptr", path + "vtable"])
            continue
        m = re.match(r"^(?:(?:class|struct|union|enum) )?(.+?)\s*(\*+)?\s*(?:\[(\d+)\])?\s+(\w+)$", decl)
        if not m:
            if "(empty)" in decl or "base" in decl:
                continue
            raise ValueError(f"{name}: cannot parse {decl!r}")
        typ, star, count, field = m.group(1).strip(), m.group(2), m.group(3), m.group(4)
        n = int(count) if count else 1
        if star:
            elems = [(8, "ptr")]
        elif typ in PRIM:
            elems = [PRIM[typ]]
        elif typ in ENUMS or typ.split("::")[-1] in {"Type", "b2BodyType", "b2LimitState", "b2JointType"}:
            elems = [(4, "i32")]
        elif typ in secs or typ.split("::")[-1] in secs:
            key = typ if typ in secs else typ.split("::")[-1]
            esz = secs[key][1]
            for k in range(n):
                out += leaves(key, secs, o + k * esz, f"{path}{field}[{k}]." if n > 1 else f"{path}{field}.")
            continue
        else:
            raise ValueError(f"{name}: unknown type {typ!r} in {decl!r}")
        esz, kind = elems[0]
        for k in range(n):
            out.append([o + k * esz, esz, kind, f"{path}{field}[{k}]" if n > 1 else path + field])
    return out


def presentation_fields(path):
    """{struct name: {field names marked `// presentation` in layout.h}} — the diff skips them."""
    out, cur = {}, None
    for ln in open(path):
        m = re.match(r"^\s*(?:struct|class)\s+(\w+)\s*(?::[^{]*)?\{", ln)
        if m:
            cur = m.group(1)
            continue
        if re.match(r"^\s*\};", ln):
            cur = None
            continue
        if cur and "// presentation" in ln:
            f = re.match(r"^\s*[\w:<>\s\*]+?[\s\*]+(\w+)\s*(?:\[[^\]]*\])*\s*;", ln)
            if f:
                out.setdefault(cur, set()).add(f.group(1))
    return out


def main(dst):
    text, classes = run_clang(game=True)
    secs = sections(text)
    res = {}
    for c in classes:
        key = c if c in secs else c.split("::")[-1]
        if key not in secs:
            key = next((k for k in secs if k.endswith("::" + c.split("::")[-1]) or k == c), None)
        if key is None:
            print(f"warning: no layout for {c}", file=sys.stderr)
            continue
        res[c.split("::")[-1]] = {"size": secs[key][1], "fields": leaves(key, secs)}
    lay_h = os.path.join(SIM, "src/game/layout.h")
    pres = presentation_fields(lay_h) if os.path.exists(lay_h) else {}
    for cls, v in res.items():  # 5th element: 1 = presentation-only field
        names = pres.get(cls, set())
        for f in v["fields"]:
            f.append(1 if re.split(r"[.\[]", f[3])[0] in names else 0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, "w"), indent=0)
    print(f"{dst}: " + ", ".join(f"{k} {v['size']:#x}/{len(v['fields'])}" for k, v in res.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(SIM, "build", "layouts.json"))
