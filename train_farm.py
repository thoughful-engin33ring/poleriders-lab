#!/usr/bin/env python3
"""Self-play PPO on the headless farm: H game processes ("hosts") x K shadow matches each.

    .venv/bin/python train_farm.py --run farm1 --hosts 10 --matches 16
    .venv/bin/python train_farm.py --run farm1 --resume
    .venv/bin/python train_farm.py --run farm1 --watch        # watch the policy in a real match

Every match is the real game (bit-identical shadows of PoleRiders::PlayState, see
bridge/test_farm_equivalence.py); one socket round trip steps all of a host's matches.
Observation v2 (bridge/farm_env.py) adds terrain heights, hammer and sudden-death flags.

League: every --pool-every updates the policy is frozen into runs/<run>/pool/; in --league-frac
of the matches one rider (random side, per episode) is a past snapshot and its transitions are
not trained on. `vs_pool` = current policy's zero-sum return per episode against snapshots.
Sudden death: --sd-frac of new matches start with the clock nearly out at 0-0, so the game drops
straight into sudden death (hammers); otherwise it only happens late in tied matches.
"""
import argparse, copy, math, csv, json, multiprocessing as mp, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))
from farm_env import OBS2_SIZE, observe2  # noqa: E402
from farm_worker import host_worker  # noqa: E402
from spr_env import SOCK_DIR, PoleRidersEnv, launch_game  # noqa: E402
from train_ppo import (ACT_SIZE, ActorCritic, LeagueBank, RunningNorm, net_from_ckpt, spec_from_args,  # noqa: E402
                       spec_from_state)
from rollout import Match, Pool, Workers, merge_norm  # noqa: E402

MANAGER_ACT = 4  # --hier: body target dx, tip target dx, tip target height, mode (see bridge/hier_env.py)


def train(args):
    run_dir = os.path.join(HERE, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    A = MANAGER_ACT if args.hier else ACT_SIZE  # a manager outputs worker targets, not sticks
    # network spec (--arch/--hidden/--layers/--res-expand/--action-head): a resumed run keeps its checkpoint's,
    # --init-from adopts the source's unless flags say otherwise; saved in the checkpoint's args
    resume_ck = (torch.load(ckpt_path, weights_only=False)
                 if args.resume and os.path.exists(ckpt_path) else None)
    explicit = {k: getattr(args, ak) for k, ak in (("arch", "arch"), ("hidden", "hidden"), ("layers", "layers"),
                                                   ("expand", "res_expand"), ("action_head", "action_head"))
                if getattr(args, ak) is not None}  # network flags given on the command line
    if resume_ck is not None:
        spec = spec_from_state(resume_ck["net"])[0]
        bad = {k: (v, spec[k]) for k, v in explicit.items() if spec[k] != v}
        if bad:
            raise SystemExit(f"--resume: the checkpoint's network differs from the flags {bad} (flag, checkpoint)")
    elif args.init_from:
        spec = spec_from_state(torch.load(args.init_from, weights_only=False)["net"])[0]
        if args.hier:
            spec["action_head"] = "gauss"
        spec.update(explicit)
    else:
        spec = spec_from_args(args)
    args.arch, args.hidden, args.layers = spec["arch"], spec["hidden"], spec["layers"]
    args.res_expand, args.action_head = spec["expand"], spec["action_head"]
    if args.action_head == "gated" and (args.hier or not args.native):
        raise SystemExit("--action-head gated: flat stick policies on --native only")
    first = "inp" if args.arch == "resmlp" else "0"  # the input layer's key: pi.<first>.weight / v.<first>.weight
    net, norm = ActorCritic(OBS2_SIZE, A, **spec), RunningNorm(OBS2_SIZE)
    if spec != spec_from_args({}):
        print(f"network: {spec}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    total_steps, update = 0, 0
    if resume_ck is not None:
        ck = resume_ck
        old_in = ck["net"][f"pi.{first}.weight"].shape[1]
        if old_in < OBS2_SIZE:
            # The observation gained features at the end: widen the first layers with zero weights
            # for the new inputs (the network computes exactly what it did before and learns to use
            # them), extend the normalizer, and restart Adam (its moments no longer match).
            sd = ck["net"]
            for head in ("pi", "v"):
                w = sd[f"{head}.{first}.weight"]
                sd[f"{head}.{first}.weight"] = torch.cat([w, torch.zeros(w.shape[0], OBS2_SIZE - old_in)], 1)
            net.load_state_dict(sd)
            ns = ck["norm"]
            norm.load({"mean": np.concatenate([ns["mean"], np.zeros(OBS2_SIZE - old_in)]),
                       "var": np.concatenate([ns["var"], np.ones(OBS2_SIZE - old_in)]),
                       "count": ns["count"]})
            print(f"observation grew {old_in} -> {OBS2_SIZE}: widened the network, optimizer restarted",
                  flush=True)
        else:
            net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); norm.load(ck["norm"])
        total_steps, update = ck["steps"], ck["update"]
        print(f"resumed at {total_steps} steps, update {update}", flush=True)
    elif args.init_from:  # a new run (typically a --hier manager) seeded from a flat game policy:
        # same observation and reward, so the critic carries over whole and the policy's hidden layers
        # (its reading of the game) too; only the output layer, which now means something else, is new
        src = torch.load(args.init_from, weights_only=False)
        sd, own = src["net"], net.state_dict()
        sspec, sobs, _ = spec_from_state(sd)
        if sobs != OBS2_SIZE:
            raise SystemExit(f"--init-from: observation size {sobs} != {OBS2_SIZE}")
        if any(sspec[k] != spec[k] for k in ("arch", "hidden", "layers", "expand")):
            raise SystemExit(f"--init-from: source network {sspec} != this run's {spec}")
        # flat -> flat with the same head (e.g. a specialist): the whole policy carries over; otherwise the
        # policy's output layer (and log_std) are new
        same_act = sd["log_std"].shape[0] == A and sspec["action_head"] == spec["action_head"]
        out_key = "pi.out." if spec["arch"] == "resmlp" else f"pi.{2 * spec['layers']}."
        take = [k for k in sd if k in own and own[k].shape == sd[k].shape
                and (same_act or not (k.startswith(out_key) or k == "log_std"))]
        for k in take:
            own[k] = sd[k]
        net.load_state_dict(own)
        norm.load(src["norm"])
        print(f"initialised from {args.init_from}: critic + policy " + ("(whole)" if same_act else
              f"hidden layers; new output layer ({A} actions)") + f" — {len(take)} tensors", flush=True)
    if args.mask_match_info and not norm.masked[56:60].all():  # hide time left, scores, sudden death (obs 56-59)
        norm.masked[56:60] = True
        norm.pin()
        print("match info (time, score) hidden from the policy", flush=True)
    if args.init_std:
        net.log_std.data.fill_(math.log(args.init_std))
        print(f"policy std set to {args.init_std}", flush=True)
    if args.min_std > 0:  # exploration floor (also lifts a resumed policy that is already below it)
        net.log_std.data.clamp_(min=math.log(args.min_std))
    pool = Pool(os.path.join(run_dir, "pool"))
    # --flat-league: some league games pit the manager against flat stick policies (e.g. farm1's pool):
    # hierarchy vs flat, head-to-head (logged as vs_flat)
    flat_pool = Pool(args.flat_league) if args.flat_league else None
    if flat_pool and not args.hier:
        raise SystemExit("--flat-league needs --hier")
    rng = random.Random(args.seed)

    H, K, D = args.hosts, args.matches, OBS2_SIZE
    cfg = {"matches": K, "arena": args.arena, "frame_skip": args.frame_skip, "sd_frac": args.sd_frac,
           "pit_frac": args.pit_frac, "reward": json.loads(args.reward) if args.reward else None,
           "end_on_sd": args.no_sudden_death, "match_format": args.match_format, "scramble": args.random_spawn,
           "random_pole": args.random_pole, "goal_spawn": args.goal_spawn, "goal_refund": args.goal_refund,
           "episode_steps": args.episode_steps}
    if not args.goal_refund:
        print("goal refund OFF (--no-goal-refund): the pre-2026-10-02 reward", flush=True)
    conns, workers, farm = [], [], None
    # The PPO update runs on a learner copy (lnet) on --ppo-device (Apple GPU: ~2.5x faster than the CPU).
    # --async-ppo overlaps it with the next rollout, which then acts one update behind: correct (the stored
    # behaviour log-probs), but it learned about half as much per step from scratch, so it is off by default.
    dev = torch.device(args.ppo_device)
    lnet = copy.deepcopy(net).to(dev)
    opt_state = opt.state_dict()
    opt = torch.optim.Adam(lnet.parameters(), lr=args.lr, eps=1e-5)
    opt.load_state_dict(opt_state)  # Adam moments move to the learner's device
    learner = ThreadPoolExecutor(1)
    rollers, events = None, []
    if args.native:
        # native simulator (sim/, bridge/native_env.py) played by rollout.py: the hosts split over --procs
        # worker processes (each: native env threads, policy copy, league bank), the learner here.
        # `farm state` sockets per host for the dashboard. No game processes are started or killed.
        # odd slots are league games with probability 2 * league_frac; above 0.5 the even slots join in
        # (league_frac 1 = every game vs the league, e.g. an exploiter against one frozen policy)
        league_flags = [[(k % 2 == 1 and args.league_frac > 0 and rng.random() < 2 * args.league_frac)
                         or (k % 2 == 0 and args.league_frac > 0.5 and rng.random() < 2 * args.league_frac - 1)
                         for k in range(K)] for h in range(H)]
        rollers = Workers(args, league_flags, A, cfg, args.procs, args.threads or os.cpu_count())
        socks = [os.path.join(SOCK_DIR, f"spr-native-{args.run}-{i}.sock") for i in range(H)]
        print(f"{H} hosts x {K} matches = {H * K} matches in the native simulator, {args.procs} rollout "
              f"processes ({sum(map(sum, league_flags))} league slots)", flush=True)
    if not args.native:
        # one worker process per game host (see bridge/farm_worker.py)
        subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
        time.sleep(1)
        subprocess.run([sys.executable, os.path.join(HERE, "prmod.py"), "exhibition", "on"], check=True,
                       capture_output=True)
        ctx = mp.get_context("spawn")
        socks = [os.path.join(SOCK_DIR, f"spr-farm-{i}.sock") for i in range(H)]
        for i in range(H):
            a, b = ctx.Pipe()
            w = ctx.Process(target=host_worker, args=(b, i, socks[i], cfg), daemon=True)
            w.start()
            conns.append(a)
            workers.append(w)
            obs = np.zeros((H, K, 2, D), np.float32)
        sd_start = np.zeros((H, K), bool)
        for h, c in enumerate(conns):
            _, msg = c.recv()
            obs[h], sd_start[h] = msg["obs"], msg["sd_start"]
        events = []
        M = [[Match(h, k, league=(k % 2 == 1 and args.league_frac > 0 and
                                     rng.random() < 2 * args.league_frac), rng=random.Random(1000 * h + k))
              for k in range(K)] for h in range(H)]
        flat = [m for hm in M for m in hm]
        bank = LeagueBank(len(flat), OBS2_SIZE, act_size=A)  # league opponents, one slot per match
        for m in flat:
            m.sd_start = bool(sd_start[m.h, m.k])
        # per-match episode state as arrays (row j = match h * K + k), so a step's bookkeeping is a few numpy
        # ops instead of a Python loop over every match; only episode ends and goals/kills loop
        NM = len(flat)
        opp_mask = np.ones(2 * NM, np.float32)  # 0 = that rider is a league opponent (not trained on)
        league_on = np.zeros(NM, bool)          # a league snapshot (LeagueBank) drives a rider of match j
        opp_row = np.zeros(NM, int)             # its rider (0/1)
        acc = {"t": np.zeros(NM, int), "ret": np.zeros((NM, 2)), "goal": np.zeros((NM, 2)), "kill": np.zeros((NM, 2)),
               "eg": np.zeros(NM), "ek": np.zeros(NM), "sd": np.zeros(NM, bool)}

        def new_episode(m):
            """Next episode's opponent for match m; flat opponents play inside the env (HierHost)."""
            m.new_episode(pool, flat_pool, args.flat_frac)
            if farm is not None and args.hier:
                farm.set_flat(m.h * K + m.k, m.opp_agent if m.opp_flat else 0, m.opp)
            j = m.h * K + m.k
            opp_mask[2 * j:2 * j + 2] = 1.0
            if m.opp and m.opp_agent:
                opp_mask[2 * j + m.opp_agent - 1] = 0.0
            league_on[j] = bool(m.opp) and not m.opp_flat  # flat opponents act inside the env, every worker step
            opp_row[j] = max(m.opp_agent - 1, 0)
            if league_on[j]:
                bank.load(j, m.opp)
            for v in acc.values():
                v[j] = 0
        print(f"{H} hosts x {K} matches = {len(flat)} matches in "
              f"{'the native simulator' if args.native else f'{H} worker processes'} "
              f"({sum(m.league for m in flat)} league slots)", flush=True)

    header = ["update", "steps", "sps", "episodes", "ep_len", "goals_per_ep", "kills_per_ep",
              "pool_size", "vs_pool", "vs_pool_goal_diff", "vs_pool_kill_diff", "vs_pool_n",
              "sd_episodes", "sd_decided", "pi_loss", "v_loss", "entropy", "std", "crashes",
              "good_side", "enemy_pit", "own_pit", "vs_flat", "vs_flat_goal_diff", "vs_flat_kill_diff",
              "vs_flat_n", "back_passes_per_min", "fwd_passes_per_min", "rotation_w"]
    log_path = os.path.join(run_dir, "log.csv")
    if os.path.exists(log_path) and os.path.getsize(log_path):  # resumed with new columns: upgrade header
        lines = open(log_path).read().splitlines()
        if lines[0] != ",".join(header):
            with open(log_path, "w") as f:
                f.write("\n".join([",".join(header)] + lines[1:]) + "\n")
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if os.path.getsize(log_path) == 0:
        log.writerow(header)
    ev_f = open(os.path.join(run_dir, "events.jsonl"), "a")

    gate_p = []  # --action-head gated: mean gate-on probability per stick over the last minibatch

    def ppo_update(data):
        """PPO epochs on lnet (runs in the learner thread). Returns (pi loss, v loss, entropy, skipped)."""
        fo, fa, flp, fadv, fret = (x.to(dev) for x in data)
        n = fo.shape[0]
        skipped, pi_loss, v_loss, ent = 0, torch.zeros(()), torch.zeros(()), torch.zeros(())
        for _ in range(args.epochs):
            perm = torch.randperm(n).to(dev)
            for s in range(0, n, args.minibatch):
                idx = perm[s:s + args.minibatch]
                d = lnet.dist(fo[idx])
                lp = d.log_prob(fa[idx]).sum(-1)
                # clamp the log-ratio: with a small std an action far from the updated mean overflows
                # exp() to inf, and a negative advantage then makes the loss -inf and the gradients NaN
                ratio = (lp - flp[idx]).clamp(-20.0, 20.0).exp()
                ad = fadv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pi_loss = -torch.min(ratio * ad, ratio.clamp(1 - args.clip, 1 + args.clip) * ad).mean()
                v_loss = 0.5 * (lnet.v(fo[idx]).squeeze(-1) - fret[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + args.vf_coef * v_loss - args.ent_coef * ent
                if not torch.isfinite(loss):  # never step on a non-finite loss
                    skipped += 1
                    continue
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(lnet.parameters(), 0.5)
                opt.step()
                if args.min_std > 0:  # exploration floor
                    lnet.log_std.data.clamp_(min=math.log(args.min_std))
        if lnet.n_gates and n:
            with torch.no_grad():
                gate_p[:] = d.gate_prob().mean(0).tolist()
        return pi_loss.item(), v_loss.item(), ent.item(), skipped

    def finish(fut, out=None):
        """Wait for a learner update (or take its result) and copy its weights into the acting net."""
        out = fut.result() if fut is not None else out
        with torch.no_grad():
            for pa, pl in zip(net.parameters(), lnet.parameters()):
                pa.copy_(pl.detach().cpu())
        return out

    def to_cpu(x):
        if torch.is_tensor(x):
            return x.cpu()
        if isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_cpu(v) for v in x]
        return x

    pending, last = None, (float("nan"),) * 3 + (0,)
    T, rot_w = args.rollout, None
    while total_steps < args.total_steps:
        t0 = time.time()
        if rollers is not None:
            # rollout r with the current acting weights; meanwhile PPO learns from rollout r - 1 here
            # (--async-ppo: acting is one update behind, the stored behaviour log-probs keep PPO correct)
            rot_w = args.rotation_bonus * max(0.0, 1.0 - total_steps / args.rotation_anneal) if args.rotation_bonus else None
            rollers.start({k: v.detach().clone() for k, v in net.state_dict().items()}, norm.state(), T, rot_w)
            if pending is not None:
                last, pending = finish(None, ppo_update(pending)), None
            R = rollers.collect()
            if len(R["obs_stats"]) == 1:  # one roller: its per-step normalizer is exactly the master's
                norm.load(R["norm"])
            else:
                for st in R["obs_stats"]:
                    merge_norm(norm, *st)
            ep_stats, vs_pool, vs_flat, sd_stats = R["ep_stats"], R["vs_pool"], R["vs_flat"], R["sd_stats"]
            events.extend(R["events"])
            pos, prof, crashes, match_status, env_steps = R["pos"], R["prof"], 0, R["matches"], R["env_steps"]
            passes = R["passes"]
            total_steps += env_steps
            data = tuple(torch.as_tensor(x) for x in R["data"])
            if args.async_ppo:
                pending = data
            else:
                last = finish(None, ppo_update(data))
        else:
            B = 2 * len(flat)
            buf_o = np.zeros((T, B, OBS2_SIZE), np.float32)
            buf_a = np.zeros((T, B, A), np.float32)
            buf_lp, buf_v, buf_r, buf_d, buf_m = (np.zeros((T, B), np.float32) for _ in range(5))
            ep_stats, vs_pool, vs_flat, sd_stats, crashes = [], [], [], [], 0
            prof = {"policy": 0.0, "league": 0.0, "wait": 0.0, "book": 0.0}  # per-phase seconds
            pos = np.zeros(4)  # positioning of the current policy's live riders: n, good side, enemy pit, own pit
            passes = np.zeros(2)  # (pole passes through vertical: counted by rollout.py only)
            def act(t, h0, h1):
                """Actions of hosts h0..h1-1 at step t (current policy + league opponents); fills the buffers."""
                _t0 = time.time()
                j0, j1 = h0 * K, h1 * K
                rows = slice(2 * j0, 2 * j1)
                raw = obs[h0:h1].reshape(-1, D)  # (host, match, agent) order == flat order x agent
                mask = opp_mask[2 * j0:2 * j1]
                norm.update(raw[mask > 0])
                # agent frame (attacking +x, y up): me.x = 0, me.y = 1, ghost = 9, ball.x = 52.
                # Pits are below the middle plateau (y up < 0) beyond |x| = 16.3 m.
                live = (mask > 0) & (raw[:, 9] < 0.5)
                if live.any():
                    r = raw[live]
                    pos[:] += [len(r), (r[:, 0] < r[:, 52]).sum(), ((r[:, 0] > 16.3) & (r[:, 1] < 0)).sum(),
                               ((r[:, 0] < -16.3) & (r[:, 1] < 0)).sum()]
                o = torch.as_tensor(norm(raw), dtype=torch.float32)
                with torch.no_grad():
                    a, lp, v = net.act(o)
                a_np = a.numpy()
                _t1 = time.time()
                js = np.nonzero(league_on[j0:j1])[0] + j0  # league opponents act with their own frozen nets
                if len(js):
                    idx = 2 * (js - j0) + opp_row[js]
                    a_np[idx] = bank.act(js, raw[idx])
                buf_o[t, rows], buf_a[t, rows], buf_lp[t, rows], buf_v[t, rows], buf_m[t, rows] = \
                    o.numpy(), a_np, lp.numpy(), v.numpy(), mask
                prof["policy"] += _t1 - _t0
                prof["league"] += time.time() - _t1
                return np.clip(a_np, -1, 1).reshape(h1 - h0, K, 2, A)

            def book(t, h, tag, out):
                """Host h's step-t results: rewards, episode ends, events, next observations."""
                nonlocal crashes
                _t0 = time.time()
                if tag == "crash":
                    crashes += 1
                    print(f"[host {h}] {out['why'][:150]}; relaunched", flush=True)
                    events.append({"t": time.time(), "instance": f"{h}", "type": "crash",
                                   "why": out["why"], "returncode": out["returncode"]})
                    for k, m in enumerate(M[h]):
                        j = h * K + k
                        buf_r[t, 2 * j] = buf_r[t, 2 * j + 1] = 0.0
                        buf_d[t, 2 * j] = buf_d[t, 2 * j + 1] = 1.0
                        new_episode(m)
                        m.sd_start = bool(out["sd_start"][k])
                    obs[h] = out["obs"]
                    prof["book"] += time.time() - _t0
                    return
                prof["book"] += time.time() - _t0
                book_rows(t, h * K, out)

            obs_rows = obs.reshape(-1, 2, D)  # view: row j = match h * K + k

            def book_rows(t, j0, out):
                """Step-t results of matches j0, j0 + 1, ... (one host or a whole host group): rewards,
                episode ends, events, next observations."""
                _t0 = time.time()
                rew, tm = out["rew"], out["terms"]
                n = len(rew)
                js = slice(j0, j0 + n)
                acc["t"][js] += 1
                acc["ret"][js] += rew
                acc["goal"][js] += tm[:, :, 0]
                acc["kill"][js] += tm[:, :, 1]
                acc["eg"][js] += np.abs(tm[:, 0, 0])
                acc["ek"][js] += np.abs(tm[:, 0, 1])
                acc["sd"][js] |= out["sd"].astype(bool)
                rr = rew.astype(np.float64)
                tr = np.nonzero(out["trunc"])[0]
                if len(tr):  # bootstrap the time limit from the final state's value
                    with torch.no_grad():
                        vb = net.v(torch.as_tensor(norm(out["final"][tr].reshape(-1, D)), dtype=torch.float32))
                    rr[tr] += args.gamma * vb.squeeze(-1).numpy().reshape(-1, 2)
                buf_r[t, 2 * j0:2 * (j0 + n)] = rr.reshape(-1)
                for k in np.nonzero((tm[:, 0, 0] != 0) | (tm[:, 0, 1] != 0))[0]:
                    j, g1, k1 = j0 + k, float(tm[k, 0, 0]), float(tm[k, 0, 1])
                    m = flat[j]
                    events.append({"t": time.time(), "instance": f"{m.h}.{m.k}",
                                   "type": "goal" if g1 else "kill",
                                   "winner": 1 if (g1 or k1) > 0 else 2,
                                   "score": out["score"][k].tolist(), "step": int(acc["t"][j]),
                                   "sudden_death": bool(out["sd"][k]),
                                   "opp": m.opp[2] if m.opp else None, "opp_agent": m.opp_agent})
                for k in np.nonzero(out["reset"])[0]:
                    j = j0 + k
                    m = flat[j]
                    buf_d[t, 2 * j] = buf_d[t, 2 * j + 1] = 1.0
                    ep_stats.append((int(acc["t"][j]), float(acc["eg"][j]), float(acc["ek"][j])))
                    if m.opp:
                        me = 2 - m.opp_agent  # the current policy's rider (0/1)
                        (vs_flat if m.opp_flat else vs_pool).append(
                            (float(acc["ret"][j, me]), float(acc["goal"][j, me]), float(acc["kill"][j, me])))
                    if acc["sd"][j]:
                        sd_stats.append(1.0 if acc["goal"][j, 0] != 0 else 0.0)
                    new_episode(m)
                    m.sd_start = bool(out["sd_start"][k])
                obs_rows[js] = out["obs"]
                prof["book"] += time.time() - _t0

            for t in range(T):
                clipped = act(t, 0, H)
                for h, c in enumerate(conns):  # all hosts step in parallel processes
                    c.send(("step", clipped[h]))
                for h, c in enumerate(conns):
                    _w0 = time.time()
                    tag, out = c.recv()
                    prof["wait"] += time.time() - _w0
                    book(t, h, tag, out)
            match_status = [{"host": m.h, "k": m.k, "league": m.league, "opp": m.opp[2] if m.opp else None,
                             "opp_agent": m.opp_agent, "opp_flat": bool(m.opp_flat), "sd_start": m.sd_start}
                            for m in flat]
            # env steps at frame skip 4 (15/s of game time) whatever the policy: a --hier manager decision
            # spans `macro` of them, so steps/sps stay comparable with flat runs
            env_steps = T * len(flat) * (args.macro if args.hier else 1)
            total_steps += env_steps

            # GAE
            with torch.no_grad():
                raw = obs.reshape(-1, D)
                last_v = net.v(torch.as_tensor(norm(raw), dtype=torch.float32)).squeeze(-1).numpy()
            adv = np.zeros((T, B), np.float32)
            gae = np.zeros(B, np.float32)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else buf_v[t + 1]
                nonterm = 1.0 - buf_d[t]
                delta = buf_r[t] + args.gamma * nv * nonterm - buf_v[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + buf_v

            # PPO update on the current policy's own transitions (the learner copy on --ppo-device)
            keep = torch.as_tensor(buf_m.reshape(-1) > 0)
            data = (torch.as_tensor(buf_o.reshape(-1, OBS2_SIZE))[keep], torch.as_tensor(buf_a.reshape(-1, A))[keep],
                    torch.as_tensor(buf_lp.reshape(-1))[keep], torch.as_tensor(adv.reshape(-1))[keep],
                    torch.as_tensor(ret.reshape(-1))[keep])
            if pending is not None:  # the previous update (run during this rollout) is done: act with it from now on
                last = finish(pending)
            if args.async_ppo:
                pending = learner.submit(ppo_update, data)
            else:
                last, pending = finish(None, ppo_update(data)), None
        pi_loss, v_loss, ent, skipped = last
        update += 1
        if skipped:
            print(f"update {update}: skipped {skipped} minibatches with a non-finite loss", flush=True)

        for e in events:
            ev_f.write(json.dumps(e) + "\n")
        events.clear()
        ev_f.flush()
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        el, eg, ek = (mean([e[i] for e in ep_stats]) for i in range(3))
        vp, vg, vk = (mean([e[i] for e in vs_pool]) for i in range(3))
        std = net.log_std.exp().mean().item()
        sps = env_steps / (time.time() - t0)  # the whole iteration: rollout, GAE, waiting for the learner
        log.writerow([update, total_steps, round(sps, 1), len(ep_stats), el, eg, ek, len(pool.files()),
                      vp, vg, vk, len(vs_pool), len(sd_stats), mean(sd_stats),
                      pi_loss, v_loss, ent, std, crashes,
                      pos[1] / max(pos[0], 1), pos[2] / max(pos[0], 1), pos[3] / max(pos[0], 1),
                      *(mean([e[i] for e in vs_flat]) for i in range(3)), len(vs_flat),
                      *(passes / max(pos[0] * 4 / 60 / 60, 1e-9) if pos[0] else [float("nan")] * 2),
                      rot_w or 0.0])
        log_f.flush()
        status = {"run": args.run, "mode": "farm", "backend": "native" if args.native else "game",
                  "arena": args.arena, "update": update, "steps": total_steps, "sps": sps,
                  "t": time.time(), "hosts": [{"idx": i, "sock": socks[i], "matches": K} for i in range(H)],
                  "matches": match_status}
        with open(os.path.join(run_dir, "status.json.tmp"), "w") as f:
            json.dump(status, f)
        os.replace(os.path.join(run_dir, "status.json.tmp"), os.path.join(run_dir, "status.json"))
        print(f"upd {update:4d} steps {total_steps:9d} sps {sps:5.0f} eps {len(ep_stats):3d} "
              f"goals/ep {eg:5.2f} kills/ep {ek:5.2f} | pool {len(pool.files()):3d} vs_pool {vp:+.3f} "
              f"(n={len(vs_pool)}) "
              + (f"vs_flat {mean([e[0] for e in vs_flat]):+.3f} goals {mean([e[1] for e in vs_flat]):+.2f} "
                 f"(n={len(vs_flat)}) " if flat_pool or args.league_fixed else "")
              + f"sd {len(sd_stats)} | pi {pi_loss:+.3f} v {v_loss:.4f} "
              f"std {std:.2f} | good side {pos[1] / max(pos[0], 1):.0%} enemy pit {pos[2] / max(pos[0], 1):.0%}"
              f" | ms/step policy {1e3 * prof['policy'] / T:.1f} league {1e3 * prof['league'] / T:.1f}"
              f" wait {1e3 * prof['wait'] / T:.1f} book {1e3 * prof['book'] / T:.1f}"
              + (" | gate on run {:.2f} aim {:.2f}".format(*gate_p) if gate_p else ""), flush=True)
        if update % args.save_every == 0:
            if pending is not None and rollers is None:  # the optimizer must not change while it is saved
                last, pending = finish(pending), None
            torch.save({"net": net.state_dict(), "opt": to_cpu(opt.state_dict()), "norm": norm.state(),
                        "steps": total_steps, "update": update, "args": vars(args)}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic: readers (dashboard) never see a partial file
        if update % args.pool_every == 0:
            pool.add(net, norm, update)
    if pending is not None and rollers is None:
        last, pending = finish(pending), None
    if update:  # --total-steps reached: the final weights (the loop may end between periodic saves)
        torch.save({"net": net.state_dict(), "opt": to_cpu(opt.state_dict()), "norm": norm.state(),
                    "steps": total_steps, "update": update, "args": vars(args)}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    if rollers is not None:
        rollers.close()


def watch(args):
    ck = torch.load(os.path.join(HERE, "runs", args.run, "ckpt.pt"), weights_only=False)
    net, norm, size = net_from_ckpt(ck)
    from farm_env import Terrain
    terrain = Terrain(args.arena)
    print(f"watching {args.run} at {ck['steps']} steps (real time; ctrl-c to stop)")
    sock = os.path.join(SOCK_DIR, "spr-watch.sock")
    proc = launch_game(sock, mute=False, size=(1280, 720))
    time.sleep(16)
    env = PoleRidersEnv(frame_skip=args.frame_skip, fast=0, sock=sock)
    env.start_match(arena=args.arena)
    env.reset()
    st = env.state
    try:
        while True:
            raw = np.array([observe2(st, 1, terrain), observe2(st, 2, terrain)], np.float32)[:, :size]
            with torch.no_grad():
                x = torch.as_tensor(norm(raw), dtype=torch.float32)
                a = (net.mean_env(x) if args.deterministic else net.sample_env(x)).clamp(-1, 1).numpy()
            obs, rew, done, info = env.step({1: tuple(a[0]), 2: tuple(a[1])})
            st = info["state"]
            tr = info["terms"][1]
            if tr["goal"] or tr["kill"]:
                print(f"score {st['score']}  terms {tr}")
            if done or not st.get("players"):
                env.reset()
                st = env.state
            time.sleep(args.frame_skip / 60)
    finally:
        env.close()
        proc.kill()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="farm1")
    p.add_argument("--hosts", type=int, default=10)
    p.add_argument("--matches", type=int, default=16, help="shadow matches per host")
    p.add_argument("--arena", type=int, default=2)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--episode-steps", type=int, default=None,
                   help="safety cap in env steps (default: twice the round time)")
    p.add_argument("--format", default="7,300", metavar="GOALS,SECONDS",
                   help="--native match format: goals to win, round time in s (user, 2026-09-30: 7 points, 5 min; "
                        "the game config the oracle checks is 3,180)")
    p.add_argument("--sd-frac", type=float, default=0.15, help="matches that start in sudden death")
    p.add_argument("--hier", default=None, metavar="WORKER_CKPT",
                   help="--native: train a MANAGER that issues targets to this frozen goal-conditioned worker "
                        "(train_worker.py) every --macro worker steps (bridge/hier_env.py)")
    p.add_argument("--macro", type=int, default=8, help="--hier: worker steps per manager decision")
    p.add_argument("--flat-league", default=None, metavar="POOL_DIR",
                   help="--hier: flat stick-policy snapshots (e.g. runs/farm1/pool) as league opponents")
    p.add_argument("--flat-frac", type=float, default=0.5,
                   help="--flat-league: share of league episodes played against a flat snapshot")
    p.add_argument("--arch", default=None, choices=("mlp", "resmlp"),
                   help="network: mlp (tanh MLP, the default) or resmlp (SimBa-style residual MLP: input linear, "
                        "--layers pre-LayerNorm residual blocks, final LayerNorm); separate policy and value towers. "
                        "--resume keeps the checkpoint's network, --init-from adopts the source's")
    p.add_argument("--hidden", type=int, default=None, help="network width (default 256)")
    p.add_argument("--layers", type=int, default=None, help="mlp hidden layers / resmlp blocks (default 2)")
    p.add_argument("--res-expand", type=int, default=None,
                   help="resmlp: inner width of a residual block = this x --hidden (default 4)")
    p.add_argument("--action-head", default=None, choices=("gauss", "gated"),
                   help="gauss (diagonal Gaussian, the default) or gated: per stick (left stick run; right stick "
                        "aim) a Bernoulli 'touch this stick' gate + the Gaussian of its value; gate 0 = that "
                        "stick exactly centred (train_ppo.GatedNormal). --native flat policies only")
    p.add_argument("--init-from", default=None, metavar="CKPT",
                   help="new run: copy a flat policy's critic and policy hidden layers (e.g. runs/farm1/ckpt.pt)")
    p.add_argument("--reward", default=None, metavar="JSON",
                   help='reward weights, e.g. \'{"goal": 0, "kill": 1, "ball_progress": 0, "position": 0}\' '
                        "(a kill-only specialist); default spr_env.DEFAULT_REWARD")
    p.add_argument("--no-sudden-death", action="store_true",
                   help="--native: a match tied when the clock runs out ends as a draw (no sudden death); "
                        "implies --sd-frac 0")
    p.add_argument("--min-std", type=float, default=0.0,
                   help="floor on the policy's action std (keeps exploring once entropy has collapsed)")
    p.add_argument("--goal-episodes", action="store_true",
                   help="--native: every goal ends a learning episode (goal-to-goal: no value across a goal); "
                        "the match itself goes on (user, 2026-10-02)")
    p.add_argument("--no-goal-refund", dest="goal_refund", action="store_false",
                   help="turn off the goal-refund reward rule (default ON since 2026-10-02; only to reproduce older "
                        "runs): no kill/death/shaping/rotation reward between a goal and the respawn, and a death is "
                        "refunded in full (the kill taken back) when the victim's team scores while it is dead "
                        "(spr_env.GoalRefund / sim GoalRefundState)")
    p.add_argument("--goal-spawn", type=float, default=0.0,
                   help="--native: probability that a goal is followed by a random re-spawn of both riders + "
                        "ball (after the game's own reset; never after kills)")
    p.add_argument("--mask-match-info", action="store_true",
                   help="hide time left, scores and the sudden-death flag from the policy (pinned in the normalizer)")
    p.add_argument("--league-fixed", default=None, metavar="NAME:CKPT:SHARE,...",
                   help="--native: frozen opponents (e.g. farm2:runs/farm2/ckpt.pt:0.25) in that share of the "
                        "league games; logged as vs_flat")
    p.add_argument("--random-pole", action="store_true",
                   help="--native: every (re)spawn builds the pole at a random angle from forward to back-up "
                        "(0-150 deg, attack frame) — training curriculum so backward poles get explored")
    p.add_argument("--rotation-bonus", type=float, default=0.0,
                   help="--native: reward per radian of backward pole rotation (counter-clockwise in the rider's "
                        "attack frame), fading linearly to 0 over --rotation-anneal env steps")
    p.add_argument("--rotation-anneal", type=float, default=100e6)
    p.add_argument("--random-spawn", action="store_true",
                   help="--native: after every goal and kill both riders re-spawn at random ground positions "
                        "(pole up) and the ball at a random zipline point (training curriculum)")
    p.add_argument("--init-std", type=float, default=None,
                   help="set the policy's std to this at the start (e.g. re-open exploration of an --init-from "
                        "or resumed policy); --min-std still floors it afterwards")
    p.add_argument("--pit-frac", type=float, default=0.0,
                   help="--native: non-sudden-death matches that start with player 1, 2 or both dropped into "
                        "the enemy pit, pole up (curriculum)")
    p.add_argument("--total-steps", type=int, default=2_000_000_000)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--procs", type=int, default=4,
                   help="--native: rollout worker processes (0 = roll out in a thread of the learner process)")
    p.add_argument("--torch-threads", type=int, default=2,
                   help="CPU threads for the acting policy's forward passes (the simulator has its own pool)")
    p.add_argument("--ppo-device", default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="device of the PPO update (the rollout always acts on the CPU)")
    p.add_argument("--async-ppo", action="store_true",
                   help="overlap each PPO update with the next rollout (acting one update behind). Faster per "
                        "step but learned ~2x slower per step from scratch (2026-09-30), so off by default")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=None,
                   help="per step (15/s): default 0.998, ~33 s look-ahead; with --hier 0.998**macro per manager step")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.001, help="keeps exploring (pit escapes)")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--league-frac", type=float, default=0.5,
                   help="share of games vs the league (pool snapshots / --league-fixed); 1 = no self-play")
    p.add_argument("--pool-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--native", action="store_true",
                   help="train on the native simulator (make -C sim env) instead of game processes")
    p.add_argument("--threads", type=int, default=0, help="--native: env worker threads (0 = all cores)")
    p.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                   help="--native: step all matches as one batch instead of two staggered host groups")
    args = p.parse_args()
    g, sec = args.format.split(",")
    args.match_format = (int(g), float(sec))
    if args.episode_steps is None:
        args.episode_steps = int(2 * args.match_format[1] * 60 / args.frame_skip)
    if args.no_sudden_death:
        args.sd_frac = 0.0
    if args.gamma is None:  # same look-ahead in seconds whatever the decision interval
        args.gamma = 0.998 ** (args.macro if args.hier else 1)
    if args.hier and not args.native:
        p.error("--hier needs --native")
    watch(args) if args.watch else train(args)


if __name__ == "__main__":
    main()
