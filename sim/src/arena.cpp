#include "arena.h"

#include <sys/mman.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>

#include "game/engine.h"

namespace sim {

namespace {
constexpr size_t kAlign = 16, kHdr = 16;
constexpr int kSmallClasses = 256;  // 16 .. 4096 bytes in 16-byte steps: exact size-class free lists
struct FreeBig {
  size_t size;
  FreeBig *next;
};
struct Header {  // at the arena base: all allocator state lives inside the arena
  size_t bump;
  size_t overflow;  // allocations that did not fit (went to malloc)
  void *small[kSmallClasses];
  FreeBig *big;
};
constexpr size_t kFirst = (sizeof(Header) + 63) & ~size_t(63);

uint8_t *g_region = nullptr;  // kArenaSlots * kArenaCap reserved bytes
Arena g_arenas[kArenaSlots];
std::vector<int> g_free_slots;
int g_next_slot = 0;
std::mutex g_mu;
std::atomic<unsigned long long> g_n_arena{0}, g_n_heap{0};

Header *H(uint8_t *base) { return reinterpret_cast<Header *>(base); }
}  // namespace

Arena *Arena::Create() {
  std::lock_guard<std::mutex> l(g_mu);
  if (!g_region) {
    void *m = mmap(nullptr, kArenaCap * kArenaSlots, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    g_region = static_cast<uint8_t *>(m);
  }
  int s;
  if (!g_free_slots.empty()) {
    s = g_free_slots.back();
    g_free_slots.pop_back();
  } else if (g_next_slot < kArenaSlots) {
    s = g_next_slot++;
  } else {
    return nullptr;
  }
  uint8_t *base = g_region + kArenaCap * s;
  if (mprotect(base, kArenaCap, PROT_READ | PROT_WRITE) != 0) {
    g_free_slots.push_back(s);
    return nullptr;
  }
  Arena *a = &g_arenas[s];
  a->base_ = base, a->slot_ = s;
  memset(base, 0, kFirst);
  H(base)->bump = kFirst;
  return a;
}

void Arena::Destroy(Arena *a) {
  if (!a || !a->base_) return;
  // fresh zero pages (drops the physical memory), inaccessible until the slot is reused
  mmap(a->base_, kArenaCap, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
  std::lock_guard<std::mutex> l(g_mu);
  g_free_slots.push_back(a->slot_);
  a->base_ = nullptr, a->slot_ = 0;
}

Arena *Arena::Owner(const void *p) {
  const uint8_t *q = static_cast<const uint8_t *>(p);
  if (!g_region || q < g_region || q >= g_region + kArenaCap * kArenaSlots) return nullptr;
  return &g_arenas[(q - g_region) / kArenaCap];
}

size_t Arena::Used() const { return H(base_)->bump; }
bool Arena::Overflowed() const { return H(base_)->overflow != 0; }

void *Arena::Alloc(size_t n) {
  Header *h = H(base_);
  size_t sz = (n + kAlign - 1) & ~(kAlign - 1);
  if (sz == 0) sz = kAlign;
  if (sz <= kSmallClasses * kAlign) {
    int c = static_cast<int>(sz / kAlign) - 1;
    if (void *p = h->small[c]) {
      h->small[c] = *static_cast<void **>(p);
      return p;
    }
  } else {
    for (FreeBig **pp = &h->big; *pp; pp = &(*pp)->next)
      if ((*pp)->size == sz) {  // exact-size reuse (Box2D grows its buffers by doubling)
        FreeBig *f = *pp;
        *pp = f->next;
        return reinterpret_cast<uint8_t *>(f) + kHdr;
      }
  }
  size_t off = h->bump;
  if (off + kHdr + sz > kArenaCap) {  // never seen (a match uses ~190 KB): degrade instead of crashing
    h->overflow++;
    return malloc(n ? n : 1);
  }
  h->bump = off + kHdr + sz;
  *reinterpret_cast<size_t *>(base_ + off) = sz;
  return base_ + off + kHdr;
}

void Arena::Free(void *p) {
  Header *h = H(base_);
  uint8_t *q = static_cast<uint8_t *>(p);
  size_t sz = *reinterpret_cast<size_t *>(q - kHdr);
  if (sz <= kSmallClasses * kAlign) {
    int c = static_cast<int>(sz / kAlign) - 1;
    *static_cast<void **>(p) = h->small[c];
    h->small[c] = p;
  } else {
    FreeBig *f = reinterpret_cast<FreeBig *>(q - kHdr);
    f->size = sz;  // the size header's slot
    f->next = h->big;
    h->big = f;
  }
}

void Arena::Clear() {
  memset(base_, 0, Used());
  H(base_)->bump = kFirst;
}

void Arena::Save(std::vector<uint8_t> *out) const {
  out->resize(Used());
  memcpy(out->data(), base_, out->size());
}

void Arena::Restore(const std::vector<uint8_t> &snap) {
  const size_t cur = Used();
  memcpy(base_, snap.data(), snap.size());
  // bytes past the snapshot's bump pointer were still zero when it was taken (bump only grows): zero what
  // the state since then has used, so later allocations see exactly what they would have seen
  if (cur > snap.size()) memset(base_ + snap.size(), 0, cur - snap.size());
}

ArenaCounters GetArenaCounters() { return {g_n_arena.load(), g_n_heap.load()}; }

}  // namespace sim

// ---------------------------------------------------------------- hooks

extern "C" void *sim_arena_alloc(size_t n) {
  sim::EngineContext *e = sim::g_engine;
  if (e && e->arena) {
    sim::g_n_arena.fetch_add(1, std::memory_order_relaxed);
    return static_cast<sim::Arena *>(e->arena)->Alloc(n);
  }
  if (e) sim::g_n_heap.fetch_add(1, std::memory_order_relaxed);  // game code of a match without an arena
  return malloc(n ? n : 1);
}

extern "C" void sim_arena_free(void *p) {
  if (!p) return;
  if (sim::Arena *a = sim::Arena::Owner(p)) {
    if (a->Contains(p) && a->Base()) {
      a->Free(p);
      return;
    }
  }
  free(p);
}

// Global operator new / delete of this library (the game code's `new Player(...)` etc.). Not exported from
// libsprsim.dylib (Makefile: -unexported_symbol), so other libraries in the process (torch, Python) keep
// their own; without an active arena-backed match this is plain malloc / free.
void *operator new(size_t n) { return sim_arena_alloc(n); }
void *operator new[](size_t n) { return sim_arena_alloc(n); }
void operator delete(void *p) noexcept { sim_arena_free(p); }
void operator delete[](void *p) noexcept { sim_arena_free(p); }
void operator delete(void *p, size_t) noexcept { sim_arena_free(p); }
void operator delete[](void *p, size_t) noexcept { sim_arena_free(p); }
