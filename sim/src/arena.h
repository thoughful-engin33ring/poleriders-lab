// Per-match memory arena (snapshot / restore of a whole native match by memcpy).
//
// Every allocation the game code and Box2D make for an arena-backed match (operator new / delete, b2Alloc /
// b2Free) lands in that match's arena: one contiguous region whose allocator state (bump pointer, free lists)
// also lives inside the region. Routing is automatic: the allocation hooks use the arena of the match whose
// game code is running, i.e. `sim::g_engine->arena` (EngineContext::arena, set by Match::CreateIn; every code
// path that runs game code sets g_engine first). A snapshot is a copy of the used bytes [base, base + Used());
// a restore copies them back to the SAME address (no pointer fix-up) and zeroes the tail the restored state
// has not used yet, so the arena is byte-identical to its state at the snapshot.
//
// All arenas are slots of one reserved virtual range (kArenaSlots x kArenaCap), so "which arena owns p" is one
// subtraction. Physical memory is only what a match touches (~190 KB). If an arena ever fills up, further
// allocations fall back to malloc and the arena is flagged (Overflowed(): snapshots of it are refused).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sim {

constexpr size_t kArenaCap = size_t(4) << 20;  // virtual bytes per arena (a match uses ~190 KB)
constexpr int kArenaSlots = 16384;             // arenas alive at once (process-wide; 64 GB of address space)

class Arena {
 public:
  static Arena *Create();  // nullptr when all slots are in use (callers then fall back to the plain heap)
  static void Destroy(Arena *a);  // drops the memory (no destructors run)
  static Arena *Owner(const void *p);  // arena containing p, or nullptr

  uint8_t *Base() const { return base_; }
  size_t Used() const;
  bool Overflowed() const;
  bool Contains(const void *p) const { return (const uint8_t *)p >= base_ && (const uint8_t *)p < base_ + kArenaCap; }
  void *Alloc(size_t n);
  void Free(void *p);
  void Clear();  // forget everything (zeroes the used bytes): a fresh, empty arena

  // snapshot: the used bytes; restore: back to exactly that state (same address)
  void Save(std::vector<uint8_t> *out) const;
  void Restore(const std::vector<uint8_t> &snap);

 private:
  uint8_t *base_ = nullptr;
  int slot_ = 0;
};

// Allocation statistics (tests): calls routed to an arena / to malloc (no arena-backed match active)
struct ArenaCounters {
  unsigned long long arena_allocs, heap_allocs;
};
ArenaCounters GetArenaCounters();

}  // namespace sim

// the hooks (Box2D/Common/b2Settings.cpp calls these; weak malloc fallbacks there for tools without the game)
extern "C" void *sim_arena_alloc(size_t n);
extern "C" void sim_arena_free(void *p);
