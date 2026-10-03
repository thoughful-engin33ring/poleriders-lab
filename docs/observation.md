# What the agents see (observation v2)

Every agent gets **62 numbers**, refreshed every 4 game frames (15 times a second). This applies to farm2, farm3,
the killers and the PBT members. Everything is from the agent's own point of view: mirrored so it always attacks
toward +x (right), with y pointing up. Units are metres, m/s and rad/s, and angles are given as sin/cos pairs.

Source: `observe2` in `bridge/farm_env.py` (Python) and `Observe2` in `sim/src/env/features.cpp` (native, identical).

## Itself (0–16)

| # | Field | Meaning |
|---|---|---|
| 0–1 | x, y | position in the arena (x from the centre; goal walls at ±20.3 m) |
| 2–3 | vx, vy | velocity |
| 4–5 | sin, cos of body angle | how the rider's body is tilted |
| 6 | w | body spin speed |
| 7 | on_pole | still holding its pole |
| 8 | on_ground | standing on the ground (not airborne) |
| 9 | out of play | dead (waiting to respawn) or in the brief invulnerable "ghost" state after spawning |
| 10–11 | sin, cos of pole angle | direction from the pole's base to its tip: 0° points at the goal it attacks, 90° straight up, 180° back toward its own goal |
| 12 | pole spin speed | angular velocity of the pole's middle segment (+ = rotating backward, over the top) |
| 13–14 | tip dx, dy | pole tip position relative to its body |
| 15 | hammer | pole is a sudden-death hammer (always 0 now that there's no sudden death) |
| 16 | respawn in | seconds until back in play ÷ 3 (death countdown, or what's left of the ghost state) |

## The opponent (17–33)

The same 17 fields for the opponent, in the agent's frame. While the opponent is dead or ghosted, its block is all
zeros except "out of play" = 1 and "respawn in". The corpse stays in the world but is hidden from the agent.

## Terrain around itself (34–51)

| # | Field | Meaning |
|---|---|---|
| 34–42 | floor | height of the ground relative to its body, sampled at −6, −4, −2, −1, 0, +1, +2, +4, +6 m ahead (negative = below it; shows pits and slopes), clipped to ±10 |
| 43–51 | overhead | height of the nearest ledge above it at the same 9 points (10 = open sky) |

## Ball and match (52–61)

| # | Field | Meaning |
|---|---|---|
| 52–55 | ball x, y, vx, vy | ball position and velocity (agent frame) |
| 56 | time left | seconds left ÷ 180 (up to 1.67 in 5-minute matches; the clock only starts on the first touch) |
| 57 | my score | |
| 58 | opponent's score | |
| 59 | sudden death | always 0 now |
| 60–61 | ball relative to me | ball x and y minus its own (+x = the ball is ahead, toward the goal it attacks): "which side of the ball am I on" in one input |

## Notes

- **Normalization:** before reaching the network, every input is normalized by a running mean and spread
  (`RunningNorm`), then clipped to ±10.
- **Hidden score and clock:** PBT members with `mask_match_info` (m1 and m3 in pbt1) have inputs 56–59 forced to
  zero, so they don't see the clock, the scores or the sudden-death flag. The mask is pinned in the normalizer.
- **No memory:** the networks have no memory and only see the current frame. Motion comes only from the
  velocities. There's no history, no record of the opponent's stick inputs, and no ball spin.
- **Worker (goal-following) agents** get 10 more numbers describing their current target: the body or tip
  target's position, and whether it's in the air (`train_worker.py`, `Goals.feats`).

## What they output

Three numbers, each in −1..1:

| # | Action | Meaning |
|---|---|---|
| 0 | run | left stick x (mirrored for the rider attacking left) |
| 1–2 | aim x, aim y | the right stick: the direction the arms drive the pole toward (the pole follows with a lag) |
