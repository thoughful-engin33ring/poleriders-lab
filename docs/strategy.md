# Pole Riders strategy notes (from the user — self-described best player in the world)

Frame: "you-ball-goal" = the ball is between you and the goal you score in = the good side.

## Normal play
- farm1 (flat PPO policy, 159M steps) is already an excellent player **as long as it's on the right
  side of the ball**. Its failures are the wrong-side situations below.
- Pole low before jumping is the normal carry (unless touching the ball directly without a jump).
- Trick farm1 doesn't use: jumping left -> right you swing the pole clockwise; swinging it
  **counter-clockwise during the jump** prolongs the jump and shifts your height a bit — fine control.

## Wrong side of the ball — situation A
Order along the field: [goal you attack] you — opp — ball — [goal opp attacks].
The opponent is on his good side, you're behind both. Two options:
1. A huge jump over to the right side of the ball: concede some ground, but you get to play normally again.
2. Walk up to the opponent and put your pole over his head: threatens a kill and limits his movement;
   he has to deal with you and can't play normally.

## Wrong side of the ball — situation B ("you're fucked")
Order: [goal you attack] opp — you — ball — [goal opp attacks].
- The opponent presses you down with his pole. If you press back he just lets go: you fly further to
  the wrong side and he has the tempo to score from almost anywhere.
- You can't kill him while his pole is on you. Walking back: he vaults and kicks the ball, recreating
  the same situation closer to your goal.
- Defence: keep your pole on him, **alternate pressing and releasing** (so "he does nothing and I fly"
  doesn't happen). Goal: get **his pole tip behind the ball**, so that when you manage to jump his tip
  comes up from under the ball and moves it your way — defusing (maybe reversing) the situation.

## More (user, round 2)
- **Recognising situations:** just relative positions (me / opponent / ball along x).
- **farm1's other weak spot:** from the moment you are **less than half a pole from the ball** — still on
  the right side — farm1 gets bad; that's roughly when you need to start backing up.
- **Tempo:** the map is short: 2–3 unanswered jumps at the ball are a goal, so letting one through
  already puts you in check.
- **Situation A:** the long jump over the ball only when you have free tempo. If you start it and the
  opponent (closer to the ball by definition) just jumps at the ball, you've only translated the
  situation, unfavourably.
- **Situation B timing:** fine motor, like fencing / BJJ timing — constant push/pull. In a replay it's
  probably simple: a rhythmical swing. With right = 0°, up = 90°, left = 180°: the pole moves between
  ~180° and 225°, 2–3 times a second (eyeballed, not a hard reference).
- **Vaulting = a slingshot:** the pole is loaded between two "vices": the ground under the tip and the
  rider's arms; the release direction sets the jump — pointed up = high, flatter = quicker and flatter.
  The engine seems to let you add energy by moving the pole. **Landing:** store kinetic + gravitational
  energy as elastic energy by falling smoothly onto the pole from a jump -> bigger next jump.
  (Counter-clockwise swing during a left->right jump prolongs it / fine height control — see above.)
- **Kill:** very often guarantees a goal — a very good outcome.
- **Ball handling — tip peeking:** get your pole tip to the opponent's side of the ball.
  - From below: a slight touch on the opponent as he vaults at the ball makes him miss and land on the
    wrong side (opp-(our tip)-ball-us -> (our tip)-ball-us-opp: GG), or pushes him back down from a vault.
    Works without being on the right side of the ball — an answer to the dreaded opp-us-ball.
  - From above — **the throw**: when the opponent steps on our pole we fling it and he flies (catapult;
    usually on the ground). When both jump at the ball the outcome is a coin flip (who hits first, where
    it lands, and the ball can glitch through riders/poles when the right-side kick triggers — a known
    game quirk). Instead: put the pole over the ball, let the opponent step on it trying to vault in, then
    catapult him to the wrong side of the ball: GG.

## Nuance: the pole tip decides the side near the ball (user, round 3)
Good/bad side breaks down right under the ball. Angles in your attack frame (0 = toward the goal you
attack, 90 = up, 180 = back toward your own goal):
- Pole between 0° and 89° (90 impossible, the ball is there): you can **push the ball forward**, quickly
  turning the awkward spot into one where you can comfortably vault into the ball (rule of thumb: the
  ball needs to be **more than half a pole in front of you** to vault into it). True even if your body
  is slightly on the wrong side — the pole is still hooked the right way.
- Pole between 91° and 180°: you can only **pull the ball backwards** — you're effectively in the
  dreaded situation.
Small difference in position, huge difference in how to assess it.
Data (farm1 self-play, 64 x 7 min, analysis/situations.py + ad-hoc; "hooked to push" = pole tip on the
far side of the ball from the goal you attack, i.e. pointing back-up; your 0-89° when attacking left):
- directly under the ball (|dx| < 1 m): hooked to push +0.11 goals/10 s vs reaching past it -0.07
- within half a pole, body on the right side: tip behind the ball +0.15 vs past it +0.01
- within half a pole, body slightly PAST the ball: tip behind +0.13 vs past -0.17  <- the tip decides

## Commitment, agency, energy, hitting the ball second (user, 2026-10-02)
(The user's notes, paraphrased; definitions and data are in concepts.py / analysis/concepts_report_v2.txt.)
- **No action is an action.** Attacking ->, situation [ball, you, opp] (x order): jumping at the opponent hoping
  he catches you with his pole is an overcommit. He can just do nothing: you fly past him and it becomes
  [ball, opp, you far away] — so much tempo that it's basically worth a goal.
  Same in deep defence: you're in your hole with the ball roughly above you, he hangs out in front of the hole
  threatening to score; you jump at him, overshoot, he does nothing, you land behind him: free goal.
- **Doing nothing is an expert decision, not a default.** You need to know exactly when to freeze — frozen at
  the wrong moment it's an autolose 99% of the time. (Why "opponents that randomly do nothing" was rejected as a
  training idea.)
- **Agency.** In the air you can barely affect the outcome until you touch something (a pole swing changes the
  trajectory a little). A fast jump at the ball can be a good gambit (pressure / brute force), but if the
  opponent can move the ball so you miss, kill you, or hit the ball right after you, you're at his mercy. Same
  when you land on his pole: he can fling you — worst case flung into the ball for an own goal.
- **Energy.** High total energy is generally good (rough guide). If you jump, faster is better.
- **Hitting the ball second** very often guarantees the best outcome: the ball goes the second kicker's way,
  regardless of energies (it glitches through the first player and his pole). Drawback: the second player is
  pushed away hard and may fly across the map and lose tempo. Countermeasures: map travel is fast anyway, and
  when flying the wrong way you can **impale the pole into the ground (lower back quadrant)** to shed energy
  quickly.
- **Rewards (the user's rule, now the training default):** no points between a goal and the respawn — a goal
  isn't worth less because we die after it. And a death right before scoring doesn't count: if you die and your
  team scores before you respawn, the death is fully refunded (the opponent's kill reward is taken back).
- Game mechanics behind "second" (from the sim, sim/src/game/contacts.cpp): a kick (PlayState::BallPlayerCollision)
  needs the rider's body — or the pole's handle segments, which the game tags as the rider — behind the ball;
  it gives the ball a fixed impulse (kickBallStrength 1400 / 64) toward the kicker's attack direction, pushes
  the kicker back by 7% of that, and switches the kicker's ball collisions off for 0.6 s (body + whole pole: the
  ball passes through him). At most one kick per tick in the whole match; the first kicker can't touch the ball
  again for 0.6 s, so a second kick in that time is the last one applied.
- Data (measured, not the user's words; analysis/concepts_report_v2.txt, 128 matches per line-up, farm2/farm3/search
  t2): a second KICK sends the ball the second kicker's way 90-97% of the time, and the kicker is thrown back ~8 m in
  1 s. Lunges at the opponent end with him between you and the ball 33-47% of the time; after that, a goal against
  within 5 s is 3-6x as likely as on average. Value grows with total energy and most clearly with the energy lead
  over the opponent; faster take-offs (top half) are worth more. Bots never "do nothing" (0-2% of overcommits).
