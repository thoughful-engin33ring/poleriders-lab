# Box2D: Sportsfriends vs stock 2.1.2

Verified 2026-09-30. The game's statically linked Box2D is **stock 2.1.2 code with 6 changed
`b2Settings.h` constants**. I found no code-level changes. All 6 are now applied in
`sim/Box2D/Common/b2Settings.h`, each marked `// SPORTSFRIENDS:`. `sim/third_party` is untouched.

## Method

1. I copied `sim/Box2D`, set `b2_maxPolygonVertices 9`, and compiled every `.cpp` with
   `clang++ -arch x86_64 -O0 -ffp-contract=off` (asserts on, like the game) into one dylib. Then I
   disassembled it with `objdump --macho`.
2. A script parsed both disassemblies (ours and `re/dis.txt`) into functions by mangled name.
   600 functions matched. The only names missing from the game are libc++ `std::sort` internals and
   static-init thunks. The only extra name in the game is `b2GetStats(int&,int&)`. For each function
   it compared:
   - **FP literals**: RIP-relative `ss`/`sd` loads resolved from each Mach-O. It also resolved the
     old clang's `movabsq $N; cvtsi2ss` pattern for integer-valued floats and the -0.0 sign masks.
   - **Integer immediates and memory displacements**, i.e. field offsets. Stack-frame offsets are
     excluded.
   - **Call sequence**, compared with difflib.
   - **Assert strings and assert line numbers** (`__LINE__` in `%edx`). These are **identical in
     every function**, so no source lines were added or removed in any Box2D file.
   - **FP arithmetic op sequence** (add/sub/mul/div/sqrt/cvt/ucomiss/floor, in order).
   - **Compare/branch condition sequence.**
3. I compared allocation sizes and static data separately.
4. After patching, I rebuilt from the patched `sim/Box2D` and re-ran everything. **All 600
   functions have identical FP-literal sets.** The remaining differences are all compiler codegen
   (listed at the end).

## Differences found (all applied)

| Constant | Stock | Game | Evidence (game address) |
|---|---|---|---|
| `b2_maxPolygonVertices` | 8 | **9** | `b2PolygonShape::Set` `cmpl $0x9` @0x100178a6e, assert text `"2 <= count && count <= 9"`; `m_vertexCount` @+0xa8; `b2PolygonShape::Clone` allocates 0xb0 (176) @0x1001784d8 |
| `b2_linearSlop` | 0.005f | **0.002f** | Every derived use scales. `b2TimeOfImpact` uses 0.002 / 0.006 (3·slop) / 0.0005 (0.25·slop) @0x10017f991..9d7. ContactSolver and TOISolver `SolvePositionConstraints` use 0.002 and -0.003 (-1.5·slop) @0x10016bd76/bfb5 and 0x10016d147/d38c. Revolute and Weld use 0.02 (10·slop) and 0.0004 ((10·slop)²). Prismatic and Line `InitVelocityConstraints` use 0.004 (2·slop) @0x100160b64. `b2PolygonShape()` sets `m_radius` = imm 0x3b83126f (0.004f = `b2_polygonRadius`) @0x100179f4b. The same 0.002 appears in Distance, Gear, Pulley, Weld, Revolute, Prismatic and Line. |
| `b2_angularSlop` | 2°·π/180 | **1°**: `(1.0f / 180.0f * b2_pi)` | Literal 0.01745329424738884 in Revolute/Weld/Prismatic/Line `SolvePositionConstraints` (e.g. @0x10016684c). This value is bit-exactly `1.0f/180.0f*b2_pi`; `b2_pi/180.0f` gives a different float, so the operation order matters. Revolute `InitVelocityConstraints` uses 2·angularSlop = 0.0349066 @0x10016595b. |
| `b2_maxLinearCorrection` | 0.2f | **10.0f** | ±10 in Distance (@0x100158f00/08), Prismatic and Line `SolvePositionConstraints`; -10 in ContactSolver (`movabsq $-0xa; cvtsi2ss` @0x10016bd63), TOISolver and Pulley. |
| `b2_maxAngularCorrection` | 8°·π/180 | **90°**: `(90.0f/180.0f*b2_pi)` = 1.5707964 | ±1.5708 in `b2RevoluteJoint::SolvePositionConstraints` @0x1001667c0/c8, its only user |
| `b2_maxTranslation` | 2.0f | **0.5f** | `b2Island::Solve` compares with `maxTranslationSquared` 0.25 @0x100171ff8 and scales by 0.5 @0x10017202a |

For each changed constant, every source site that uses it shows the change consistently. This
points to header edits, not per-site code edits. Revolute's ±π/2 could also be `b2_maxRotation`
used in code, but that is the same float value, so the result is identical.

**Unchanged and verified:** `b2_maxManifoldPoints` 2, `b2_aabbExtension` 0.1, `b2_aabbMultiplier` 2,
`b2_maxTOIContacts` 32, `b2_velocityThreshold` 1, `b2_maxRotation` π/2 (literals 1.5708 and 2.4674 in
Island), `b2_contactBaumgarte` 0.2, `b2_timeToSleep` 0.5, `b2_linearSleepTolerance` 0.01
(0.0001 squared), `b2_angularSleepTolerance` 2° (0.00121847 squared). `b2_polygonRadius` keeps its
formula `2*linearSlop`. The block allocator constants are stock (`s_blockSizes` = 16..640, chunk and
max block sizes). Stack allocator sizes, `b2_version` = {2,1,2}, TOI/GJK iteration limits and all
other integer immediates are stock.

**Struct sizes (identical once maxPolygonVertices = 9):** all contacts 184, b2PolygonShape 176,
b2CircleShape 24, b2Body 192, b2Fixture 80. Joints: Distance 216, Mouse 232, Prismatic 312,
Revolute 264, Pulley 264, Gear 280, Line 280, Weld 232, Friction 216.

**Code difference found later (2026-09-30, package B): `b2Fixture::SetFilterData`.** The game's build
has the `m_body == NULL` early-out inverted (@0x100171675: `cmpq $0,0x20(%rsi); setne; xorb $1; jne <loop>;
jmp <ret>`), so for any fixture attached to a body it returns right after storing `m_filter` and never
calls `FlagForFiltering()` on the body's contacts. Confirmed on the live game with lldb (no contact has
`e_filterFlag` after `Player::Land -> _SetBallCollisionFilters`). The game code compensates by setting
`e_filterFlag` on every world contact itself (Player::SetGhost/_OffPole, contact handlers). Applied in
`sim/Box2D/Dynamics/b2Fixture.cpp` (`// SPORTSFRIENDS:`). The branch-sequence comparison above missed it
because both builds compare `m_body` with 0 and branch once.

## Other game-only code (harmless, not replicated)

`b2Settings.cpp`: `b2Alloc`/`b2Free` update static `allocCount`/`allocSize` counters, and
`b2GetStats(int&,int&)` reads them. `b2Free` skips the decrement for NULL. There is no effect on
physics.

## Residual diffs judged to be compiler noise (old Apple clang vs current clang, both -O0)

- **Literals re-used or re-loaded.** Only the counts differ; the sets are equal. Examples:
  `ComputeMass`, `b2CollidePolygons` 0.98/0.001, ContactSolver ctor 100.
- **Integer-valued floats.** The game builds them with `movabsq $N; cvtsi2ss`, which gives the same
  value.
- **Negation.** Ours uses `xorps` sign-mask negation. The game's b2PulleyJoint ctor computes
  `C + (-2)*ratio` instead of `C - ratio*2`, which is exact and gives the same result.
- **libm vs inline instructions.**
  - `sqrtf` is a libm call in the game; current clang uses inline `sqrtss`. Both are correctly
    rounded.
  - `floorf` is a libm call in the game; current clang uses `roundss`. Both are exact.
  - `b2Max<float>`, `b2Min<float>` and `b2MixRestitution` compile to `maxss`/`minss`. These match
    `a > b ? a : b` exactly, including NaN and ±0 handling.
- **Code shape only.** Struct copies (8- vs 16-byte moves), `addq $imm` vs displacement addressing,
  jump-table data decoded as code, EH/terminate landing pads, virtual-call register choice,
  inverted branches and bool-test lowering.

## Float math calls made by the game's Box2D

- `sinf` + `cosf`: only in `b2Mat22::Set(float)` and `b2Mat22(float)`. **This is on the hot
  path**: `b2Body::SynchronizeTransform` and `b2Sweep::GetTransform` run every step and on every
  `SetTransform`. The Box2D sources call plain `cosf`/`sinf`, so on arm64 they must be routed to
  `x86_cosf`/`x86_sinf` (`sim/tools/libm`) to be bit-exact. I have **not** patched this in
  `sim/Box2D`, because it depends on the sim's libm shim. Do it where the shim lives, e.g.
  `#define`s before `b2Math.h`, or edit `b2Mat22::Set`.
- `sqrtf`: libm call in `b2Vec2::Length`, `b2MixFriction` and `b2CircleShape::RayCast`. Correctly
  rounded, so it is the same as `sqrtss` or arm64 `fsqrt`.
- `floorf`: `b2Sweep::Normalize`. Exact.
- `atan2f`: never called by Box2D itself. `b2Mat22::GetAngle` is not even emitted in the game; the
  29 game `atan2f` calls are PoleRiders code.

## Build notes and uncertainties

- The game is x86 SSE scalar code with no FMA. The sim must be built with `-ffp-contract=off`, as
  PORTING.md already says. Current clang otherwise contracts `a*b - c*d` even at -O0; I saw this in
  my first reference build.
- Not checked: whether two different literals were **swapped** within one function while the set
  and the op sequence stayed the same. That is very unlikely.
- Also not checked: integer/pointer logic beyond immediates, offsets, call order and branch
  conditions. All of these matched, apart from the codegen noise above.
