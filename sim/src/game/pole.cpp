// Pole helpers (package B). Pole ctor/dtor are package A.
#include <cstdio>
#include <cstdlib>
#include "layout.h"

namespace PoleRiders {

// PoleRiders::Pole::Hammer() @0x10013d660
b2Body *Pole::Hammer() { return isHammer ? hammerBody : nullptr; }

// PoleRiders::Pole::NumSegments() @0x10013d674
int Pole::NumSegments() { return 10; }

// PoleRiders::Pole::Detach(bool) @0x10013d7da
// Detaching: every fixture of every segment gets filter {category 0x40, mask 1, group 0}.
void Pole::Detach(bool detach) {
  if (!detach) return;
  b2Filter f;
  f.categoryBits = 0x40;
  f.maskBits = 1;
  f.groupIndex = 0;
  for (int i = 0; i < 10; i++)
    for (b2Fixture *fx = segments[i]->GetFixtureList(); fx; fx = fx->GetNext()) fx->SetFilterData(f);
}

// PoleRiders::Pole::Segment(int) @0x10013e596
b2Body *Pole::Segment(int i) {
  if ((unsigned)i >= 10u) {  // cmpl $0xa ; jb (unsigned)
    fprintf(stderr, "Grabbing invalid pole segment.\n");  // _Kernel_Panic
    abort();
  }
  return segments[i];
}

}  // namespace PoleRiders
