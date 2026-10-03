#!/usr/bin/env python3
"""Does the actor react to the ball? Plays a checkpoint in a throwaway farm host, then moves
ONLY the ball (5 m ahead vs 5 m behind each rider) in real observations and compares commands.

    .venv/bin/python analysis/ball_awareness.py [run]
"""
import os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RUN = sys.argv[1] if len(sys.argv) > 1 else 'farm1'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'bridge'))
from spr_env import launch_game, SOCK_DIR
from farm_env import FarmHost, OBS2_SIZE
from train_ppo import net_from_ckpt
ck = torch.load(os.path.join(ROOT, 'runs', RUN, 'ckpt.pt'), weights_only=False)
net, norm, _ = net_from_ckpt(ck)
sock = os.path.join(SOCK_DIR, 'spr-diag.sock'); proc = launch_game(sock, size=(320, 180)); time.sleep(16)
host = FarmHost(sock, n_matches=16, arena=2); host.start(); obs = host.reset_all()
O = []
for t in range(400):
    raw = np.array([o[a] for o in obs for a in (1, 2)], np.float32); O.append(raw)
    with torch.no_grad():
        a = net.sample_env(torch.as_tensor(norm(raw), dtype=torch.float32)).clamp(-1, 1).numpy()
    o2, rew, done, info = host.step([(tuple(a[2*k]), tuple(a[2*k+1])) for k in range(16)])
    for k in range(16):
        if done[k] or info[k]['truncated'] or o2[k] is None: o2[k] = host.reset(k)
    obs = o2
proc.kill()
X = np.concatenate(O); X = X[X[:, 9] < 0.5]          # live riders only
def mean_act(M):
    with torch.no_grad():
        d = net.dist(torch.as_tensor(norm(M.astype(np.float32)), dtype=torch.float32))
        return d.mean.numpy(), net.v(torch.as_tensor(norm(M.astype(np.float32)), dtype=torch.float32)).squeeze(-1).numpy()
ahead, behind = X.copy(), X.copy()
ahead[:, 52] = X[:, 0] + 5.0; behind[:, 52] = X[:, 0] - 5.0      # ball x in my own frame (I attack +x)
ahead[:, 54] = behind[:, 54] = 0.0                                  # ball at rest
if X.shape[1] > 60:                                                 # obs with ball-relative inputs
    ahead[:, 60], behind[:, 60] = 5.0, -5.0
ma, va = mean_act(ahead); mb, vb = mean_act(behind)
print(f"{len(X)} real observations; only the ball moved (5 m ahead vs 5 m behind me)")
for i, name in enumerate(["run (+ = toward the goal I attack)", "aim_x", "aim_y"]):
    print(f"  {name:36s} ahead {ma[:, i].mean():+.3f}   behind {mb[:, i].mean():+.3f}   mean |change| {np.abs(ma[:, i]-mb[:, i]).mean():.3f}")
print(f"  critic value                          ahead {va.mean():+.3f}   behind {vb.mean():+.3f}")
# for scale: how much does the run command move when we move ME by 5 m instead?
me_f, me_b = X.copy(), X.copy(); me_f[:, 0] += 5; me_b[:, 0] -= 5
if X.shape[1] > 60:
    me_f[:, 60] -= 5; me_b[:, 60] += 5
m1, _ = mean_act(me_f); m2, _ = mean_act(me_b)
print(f"  (scale) run |change| when moving myself ±5 m instead: {np.abs(m1[:,0]-m2[:,0]).mean():.3f}")
# does the actor ever push toward its own goal when the ball is behind?
print(f"  share of states where 'ball behind' makes it run backward (run < -0.2): {(mb[:,0] < -0.2).mean():.1%}  vs ahead {(ma[:,0] < -0.2).mean():.1%}")
