# Native Pole Riders simulator (arm64). Float semantics must match the game: no FMA contraction,
# no fast-math (see PORTING.md).
CXX      = clang++
CC       = clang
FLAGS    = -arch arm64 -O2 -ffp-contract=off -fno-fast-math -g
CXXFLAGS = $(FLAGS) -std=c++17 -I. -Wall -Wno-unused-private-field
CFLAGS   = $(FLAGS) -Wall
# output directory: `make env B=build-dev` builds a test copy without touching build/ (live trainers)
B       ?= build

B2_SRC   = $(shell find Box2D -name '*.cpp')
B2_OBJ   = $(patsubst %.cpp,$(B)/%.o,$(B2_SRC))

GAME_SRC = $(wildcard src/game/*.cpp) src/match.cpp src/arena.cpp
GAME_OBJ = $(patsubst %.cpp,$(B)/%.o,$(GAME_SRC))

all: $(B)/libbox2d.a $(B)/physcheck

$(B)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Wno-everything -c $< -o $@

$(B)/src/x86math.o: src/x86math.c src/x86math.h src/x86_trig_consts.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(B)/libbox2d.a: $(B2_OBJ) $(B)/src/x86math.o
	rm -f $@ && ar rcs $@ $^

# the oracle-side dump format (bridge/farm.c `farm dump`) for Box2D worlds; -Dprivate=public so it
# can read Box2D internals
$(B)/src/b2dump.o: src/b2dump.cpp src/b2dump.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Dprivate=public -Dprotected=public -c $< -o $@

$(B)/physcheck: tools/physcheck.cpp $(B)/src/b2dump.o $(B)/libbox2d.a
	$(CXX) $(CXXFLAGS) -Dprivate=public -Dprotected=public $^ -o $@

# game code: -Dprivate=public is not used; -Wno-invalid-offsetof for layout.h static_asserts
$(B)/src/game/%.o: src/game/%.cpp src/game/layout.h src/game/engine.h src/game/assets.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Wno-invalid-offsetof -c $< -o $@

$(B)/src/match.o: src/match.cpp src/match.h src/arena.h src/game/layout.h src/game/engine.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Wno-invalid-offsetof -c $< -o $@

# per-match arenas (snapshot / restore) + this library's operator new / delete
$(B)/src/arena.o: src/arena.cpp src/arena.h src/game/engine.h src/game/layout.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Wno-invalid-offsetof -c $< -o $@

$(B)/gamecheck: tools/gamecheck.cpp $(GAME_OBJ) $(B)/src/b2dump.o $(B)/libbox2d.a
	$(CXX) $(CXXFLAGS) -Wno-invalid-offsetof $^ -o $@

clean:
	rm -rf $(B)
.PHONY: all clean

# package D: the input chain alone (tools/inputcheck.py compares it with oracle traces)
$(B)/inputcheck: tools/inputcheck.cpp src/game/input.cpp src/game/engine.h src/game/layout.h
	@mkdir -p $(B)
	$(CXX) $(CXXFLAGS) -Wno-invalid-offsetof tools/inputcheck.cpp src/game/input.cpp -o $@

$(B)/bench: tools/bench.cpp $(GAME_OBJ) $(B)/src/b2dump.o $(B)/libbox2d.a
	$(CXX) $(CXXFLAGS) -Wno-invalid-offsetof $^ -o $@

# batched RL env (src/env/) + C ABI shared library for bridge/native_env.py
ENV_SRC  = $(wildcard src/env/*.cpp)
# the library's own operator new / delete (src/arena.cpp) stay private: torch / Python keep theirs
UNEXPORT_NEW = $(foreach s,__Znwm __Znam __ZdlPv __ZdaPv __ZdlPvm __ZdaPvm,-Wl,-unexported_symbol,$(s))
ENV_OBJ  = $(patsubst %.cpp,$(B)/%.o,$(ENV_SRC))
$(ENV_OBJ): $(B)/src/env/%.o: src/env/%.cpp $(wildcard src/env/*.h) src/match.h src/arena.h src/game/layout.h src/game/engine.h src/game/assets.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Wno-invalid-offsetof -c $< -o $@

$(B)/libsprsim.dylib: $(ENV_OBJ) $(GAME_OBJ) $(B)/src/b2dump.o $(B)/libbox2d.a
	$(CXX) $(CXXFLAGS) -dynamiclib -install_name @rpath/libsprsim.dylib $(UNEXPORT_NEW) $^ -o $@.tmp && mv $@.tmp $@  # rename: live trainers keep their mapped copy

env: $(B)/libsprsim.dylib
.PHONY: env
