"""Manager/worker hierarchy on the native simulator: the trained policy (manager) issues a target
every `macro` steps and a frozen goal-conditioned worker (train_worker.py) turns it into sticks.

HierHost has NativeHost's interface (reset_all/step/out, json/state_json for the dashboard), so
train_farm.py can train the manager with its usual PPO/league machinery (--hier WORKER_CKPT):

    manager action (4 floats in [-1, 1], in the rider's own frame: x toward the goal it attacks, y up)
      a0  body target x   = my x + 16 * a0      (clipped to the field, |x| <= 19.8 m)
      a1  pole-tip target x = my x + 8 * a1
      a2  pole-tip target height = 5 + 4 * a2 m above the ground under me (1..9 m)
      a3  mode: < -1/3 body only, > 1/3 tip only, else both
    one manager step = `macro` worker steps (worker: frame skip 4, so macro 8 = 32 ticks = 0.53 s);
    rewards/terms are summed over the macro step. A match that ends mid-macro stops counting there
    (its reset / final / score / truncation are reported like a normal step); the worker keeps it
    standing still until the macro step is over.

Flat league opponents: set_flat(k, agent, (net, norm, name)) lets a flat stick policy (e.g. a farm1
snapshot) drive rider `agent` of match k directly every worker step instead of the worker — the
manager's action for that rider is ignored. That gives hierarchy-vs-flat games inside training.
"""
import os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import train_worker as tw  # noqa: E402
from farm_env import OBS2_SIZE  # noqa: E402
from native_env import NativeHost  # noqa: E402
from train_ppo import LeagueBank, net_from_ckpt  # noqa: E402

MANAGER_ACT = 4


def load_worker(path):
    ck = torch.load(path, weights_only=False)
    net, norm, _ = net_from_ckpt(ck)
    a = ck.get("args", {})
    # mirror mode: False, "dynamic" (older workers: frame re-decided every step) or "goal" (per target)
    mirror = a.get("flip", "dynamic") if a.get("mirror", True) else False
    return net, norm, mirror


def set_goals(goals, a, O):
    """Manager actions a [B, 4] (clipped to [-1, 1]) -> worker targets for rows at observations O."""
    a = np.clip(a, -1, 1)
    me_x, me_y = O[:, 0], O[:, 1]
    goals.body_on[:] = a[:, 3] < 1 / 3
    goals.tip_on[:] = a[:, 3] > -1 / 3
    goals.bx[:] = np.clip(me_x + 16.0 * a[:, 0], -tw.X_MAX, tw.X_MAX)
    goals.tx[:] = me_x + 8.0 * a[:, 1]
    goals.ty[:] = (me_y - 0.69) + 5.0 + 4.0 * a[:, 2]
    goals.left[:] = tw.LIMIT
    goals.d0[:] = goals.bx - me_x
    goals.set_flip(np.arange(len(O)), O)


def hold_still(goals, rows, O):
    """Neutral target (stay where you are) for rows whose match restarted mid-macro."""
    goals.body_on[rows], goals.tip_on[rows] = True, False
    goals.bx[rows] = O[rows, 0]
    goals.left[rows] = tw.LIMIT
    goals.set_flip(np.nonzero(rows)[0] if rows.dtype == bool else rows, O)


class HierHost:
    act_size = MANAGER_ACT

    def __init__(self, worker_ckpt, macro=8, n_hosts=1, matches=16, frame_skip=4, **kw):
        if frame_skip != 4:
            raise ValueError("the worker was trained at frame skip 4")
        self.env = NativeHost(n_hosts, matches, 4, **kw)
        self.n, self.macro = self.env.n, macro
        self.net, self.norm, self.mirror = load_worker(worker_ckpt)
        self.B = 2 * self.n
        self.goals = tw.Goals(self.B, np.random.default_rng(0))
        N, D = self.n, OBS2_SIZE
        self.out = {"obs": np.zeros((N, 2, D), np.float32), "final": np.zeros((N, 2, D), np.float32),
                    "rew": np.zeros((N, 2), np.float32), "terms": np.zeros((N, 2, 2), np.float32),
                    "reset": np.zeros(N, bool), "trunc": np.zeros(N, bool), "sd": np.zeros(N, bool),
                    "score": np.zeros((N, 2), np.int32), "sd_start": np.zeros(N, bool)}
        self.O = np.zeros((self.B, D), np.float32)
        self.flat_bank = LeagueBank(self.n, OBS2_SIZE, act_size=3)
        self.flat_agent = np.zeros(self.n, int)  # 0 = none, else which rider a flat policy drives

    def set_flat(self, k, agent, opp):
        """Match k: rider `agent` (1/2) is driven by flat policy opp = (net, norm, name); agent 0 = none."""
        if agent:
            self.flat_bank.load(k, opp)
        self.flat_agent[k] = agent

    # ---- NativeHost interface
    def reset_all(self):
        obs = self.env.reset_all()
        self.O = obs.reshape(self.B, OBS2_SIZE).copy()
        self.out["obs"][:] = obs
        self.out["sd_start"][:] = self.env.out["sd_start"]
        return self.out["obs"]

    def worker_act(self, O):
        x = np.concatenate([O, self.goals.feats(O)], 1)
        flip = tw.target_behind(self.goals, O, self.mirror == "goal") if self.mirror else np.zeros(len(O), bool)
        x = tw.mirror(x, flip)[:, :len(self.norm.mean)]  # older workers take fewer goal features
        with torch.no_grad():
            a = self.net.mean_env(torch.as_tensor(self.norm(x), dtype=torch.float32)).numpy()  # mean action
        a = np.clip(a, -1, 1)
        a[flip] *= tw.ACT_MIRROR
        return a

    def step(self, actions):
        N, B = self.n, self.B
        o = self.out
        set_goals(self.goals, np.asarray(actions, np.float32).reshape(B, MANAGER_ACT), self.O)
        o["rew"][:] = 0
        o["terms"][:] = 0
        o["reset"][:] = False
        o["trunc"][:] = False
        done = np.zeros(N, bool)
        for j in range(self.macro):
            wa = self.worker_act(self.O)
            fk = np.nonzero(self.flat_agent)[0]
            if len(fk):  # flat league opponents play their sticks themselves (frame skip 4, like the worker)
                rows = 2 * fk + self.flat_agent[fk] - 1
                wa[rows] = np.clip(self.flat_bank.act(list(fk), self.O[rows]), -1, 1)
            e = self.env.step(wa.reshape(N, 2, 3))
            live = ~done
            o["rew"][live] += e["rew"][live]
            o["terms"][live] += e["terms"][live]
            o["sd"][live] = e["sd"][live]
            o["score"][live] = e["score"][live]
            ended = live & e["reset"]
            if ended.any():
                o["final"][ended] = e["final"][ended]
                o["trunc"][ended] = e["trunc"][ended]
                o["reset"][ended] = True
                done |= ended
            self.O = e["obs"].reshape(B, OBS2_SIZE).copy()
            if ended.any():
                hold_still(self.goals, np.repeat(ended, 2), self.O)
            self.goals.left -= 1
        o["final"][~done] = e["final"][~done]
        o["obs"][:] = self.O.reshape(N, 2, OBS2_SIZE)
        o["sd_start"][:] = e["sd_start"]
        return o

    # ---- dashboard / inspection pass-throughs
    def json(self, k, which=0):
        return self.env.json(k, which)

    def state_json(self, k0=0, k1=None):
        return self.env.state_json(k0, k1)

    def pit_start(self, k):
        return self.env.pit_start(k)

    def close(self):
        self.env.close()
