// Animation (package B): the tiny clip player the rider's kick/land state machine depends on,
// plus the static AnimationData tables (animation_data.h).
#include "animation_data.h"

namespace PoleRiders {
namespace Anims {
const AnimationData nullAnimation = {{}, 1, 0x1.0000000000000p+0f, false, {}};  // @0x10028dfe0  fps=1
const AnimationData kickAnimNE_hold = {{36, 37, 38, 30, 30, 30}, 6, 0x1.e000000000000p+4f, false, {}};  // @0x10028ed40  fps=30
const AnimationData kickAnimNE = {{36, 37, 38, 30, 30, 30, 38, 11}, 8, 0x1.e000000000000p+4f, false, {}};  // @0x10028ede0  fps=30
const AnimationData kickAnimE_hold = {{36, 37, 38, 39, 39, 39}, 6, 0x1.e000000000000p+4f, false, {}};  // @0x10028ee80  fps=30
const AnimationData kickAnimE = {{36, 37, 38, 39, 39, 39, 38, 11}, 8, 0x1.e000000000000p+4f, false, {}};  // @0x10028ef20  fps=30
const AnimationData kickAnimSE_hold = {{36, 37, 38, 31, 31, 31}, 6, 0x1.e000000000000p+4f, false, {}};  // @0x10028efc0  fps=30
const AnimationData kickAnimSE = {{36, 37, 38, 31, 31, 31, 38, 11}, 8, 0x1.e000000000000p+4f, false, {}};  // @0x10028f060  fps=30
const AnimationData kickAnimS = {{32, 33, 34, 35, 11}, 5, 0x1.e000000000000p+4f, false, {}};  // @0x10028f100  fps=30
const AnimationData kickAnimSW_hold = {{24, 25, 26, 29, 29, 29}, 6, 0x1.e000000000000p+4f, false, {}};  // @0x10028f1a0  fps=30
const AnimationData kickAnimSW = {{24, 25, 26, 29, 29, 29, 26, 11}, 8, 0x1.e000000000000p+4f, false, {}};  // @0x10028f240  fps=30
const AnimationData kickAnimW_hold = {{24, 25, 26, 27, 27, 27}, 6, 0x1.e000000000000p+4f, false, {}};  // @0x10028f2e0  fps=30
const AnimationData kickAnimW = {{24, 25, 26, 27, 27, 27, 26, 11}, 8, 0x1.e000000000000p+4f, false, {}};  // @0x10028f380  fps=30
const AnimationData kickAnimNW_hold = {{24, 25, 26, 28, 28, 28}, 6, 0x1.e000000000000p+4f, false, {}};  // @0x10028f420  fps=30
const AnimationData kickAnimNW = {{24, 25, 26, 28, 28, 28, 26, 11}, 8, 0x1.e000000000000p+4f, false, {}};  // @0x10028f4c0  fps=30
const AnimationData landingAnim = {{12, 13, 14, 15, 16}, 5, 0x1.e000000000000p+3f, false, {}};  // @0x10028f560  fps=15
const AnimationData turnAnim = {{9, 8, 16}, 3, 0x1.e000000000000p+3f, false, {}};  // @0x10028f600  fps=15
const AnimationData walkAnim = {{0, 1, 2, 3, 4, 5, 6, 7}, 8, 0x1.e000000000000p+4f, true, {}};  // @0x10028f6a0  fps=30
const AnimationData idleAnim = {{16}, 1, 0x0.0p+0f, false, {}};  // @0x10028f740  fps=0
const AnimationData poleIdleAnim = {{11}, 1, 0x0.0p+0f, false, {}};  // @0x10028f7e0  fps=0
const AnimationData fellAnim = {{21, 22, 23}, 3, 0x1.4000000000000p+3f, false, {}};  // @0x10028f880  fps=10
const AnimationData fallingAnim = {{17, 18, 19, 20}, 4, 0x1.e000000000000p+3f, false, {}};  // @0x10028f920  fps=15
const AnimationData smashAnimData = {{0, 1, 1, 2, 2, 3, 3, 3}, 8, 0x1.e000000000000p+4f, false, {}};  // @0x10028f9c0  fps=30
const AnimationData liftoffAnim = {{10, 11}, 2, 0x1.e000000000000p+3f, false, {}};  // @0x10028fa60  fps=15
const AnimationData gullTurn = {{0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1}, 23, 0x1.e000000000000p+4f, false, {}};  // @0x10028fcc0  fps=30
const AnimationData scoreArrow2 = {{4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7}, 24, 0x1.e000000000000p+3f, false, {}};  // @0x10028fd60  fps=15
const AnimationData scoreArrow1 = {{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}, 24, 0x1.e000000000000p+3f, false, {}};  // @0x10028fe00  fps=15
const AnimationData scoreArrow2idle = {{7}, 1, 0x1.e000000000000p+3f, false, {}};  // @0x10028fea0  fps=15
const AnimationData scoreArrow1idle = {{3}, 1, 0x1.e000000000000p+3f, false, {}};  // @0x10028ff40  fps=15
const AnimationData gullIdle = {{}, 1, 0x1.e000000000000p+3f, false, {}};  // @0x10028ffe0  fps=15
const AnimationData animHit = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11}, 23, 0x1.e000000000000p+3f, false, {}};  // @0x100290220  fps=15
const AnimationData animWait = {{}, 1, 0x1.e000000000000p+3f, false, {}};  // @0x1002902c0  fps=15
}  // namespace Anims

// PoleRiders::Animation::Animation() @0x100119b14
Animation::Animation() {
  data = &Anims::nullAnimation;
  playing = false;
}

// PoleRiders::Animation::Play(AnimationData const*) @0x100119b28
void Animation::Play(const AnimationData *d) {
  if (data != d) {
    data = d;
    position = 0.0f;
    rate = 1.0f;
    playing = true;
  }
}

// PoleRiders::Animation::Play(AnimationData const*, bool) @0x100119b48
void Animation::Play(const AnimationData *d, bool restart) {
  if (restart) {
    position = 0.0f;
    playing = true;
  }
  if (data != d) {
    data = d;
    position = 0.0f;
    rate = 1.0f;
    playing = true;
  }
}

// PoleRiders::Animation::Update(float) @0x100119b78
void Animation::Update(float dt) {
  float pos = position;
  if (1.0f > pos) {  // ucomiss pos, 1.0 ; jbe (NaN skips)
    // ((dt * rate) * fps) / (float)numFrames, all single precision
    float step = dt * rate;
    step = step * data->fps;
    step = step / (float)data->numFrames;
    pos = pos + step;
    position = pos;
  }
  while (pos >= 1.0f && data->loop) {
    pos = pos + -1.0f;
    position = pos;
  }
}

// PoleRiders::Animation::SetRate(float) @0x100119bea
void Animation::SetRate(float r) { rate = r; }

// PoleRiders::Animation::IsFinished() @0x100119c36
bool Animation::IsFinished() { return position >= 1.0f; }

// PoleRiders::Animation::GetData() @0x100119c4e
const AnimationData *Animation::GetData() { return data; }

// PoleRiders::Animation::SetPosition(float) @0x100119c58
void Animation::SetPosition(float p) {
  if (0.0f > p) {
    position = 0.0f;
    return;
  }
  position = p;
  if (p > 1.0f) position = 0.0f;
}

// PoleRiders::Animation::GetPosition() @0x100119c7e
float Animation::GetPosition() { return position; }

}  // namespace PoleRiders
