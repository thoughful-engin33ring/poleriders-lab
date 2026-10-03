#!/usr/bin/env python3
"""Tests for the network options (train_ppo: --arch / --hidden / --layers / --action-head, net_from_ckpt,
LeagueBank with mixed opponents, GatedNormal).

    .venv/bin/python test_netopts.py          # all tests (a few seconds)
"""
import glob, math, os, sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bridge"))
import train_ppo as tp  # noqa: E402

D = 62
SPECS = [dict(), dict(hidden=128), dict(layers=3), dict(arch="resmlp", hidden=64, layers=2),
         dict(arch="resmlp", hidden=96, layers=3, expand=2), dict(action_head="gated"),
         dict(arch="resmlp", hidden=64, layers=1, action_head="gated")]


def ckpt_of(net, D=D):
    norm = tp.RunningNorm(D)
    rng = np.random.default_rng(0)
    norm.mean, norm.var = rng.normal(size=D), rng.uniform(0.5, 3, size=D)
    return {"net": net.state_dict(), "norm": norm.state()}


def test_default_net_is_the_original():
    """ActorCritic(62, 3) = the original construction, draw for draw (same seed -> same tensors, same keys)."""
    torch.manual_seed(123)
    ref_pi = nn.Sequential(nn.Linear(D, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 3))
    ref_v = nn.Sequential(nn.Linear(D, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
    log_std = torch.full((3,), -0.5)
    with torch.no_grad():
        ref_pi[-1].weight.mul_(0.01)
    after_ref = torch.rand(4)
    torch.manual_seed(123)
    net = tp.ActorCritic(D, 3)
    after_new = torch.rand(4)
    sd = net.state_dict()
    ref = {**{f"pi.{k}": v for k, v in ref_pi.state_dict().items()}, **{f"v.{k}": v for k, v in ref_v.state_dict().items()},
           "log_std": log_std}
    assert list(sd) == ["log_std", "pi.0.weight", "pi.0.bias", "pi.2.weight", "pi.2.bias", "pi.4.weight", "pi.4.bias",
                        "v.0.weight", "v.0.bias", "v.2.weight", "v.2.bias", "v.4.weight", "v.4.bias"]
    assert set(sd) == set(ref) and all(torch.equal(sd[k], ref[k]) for k in sd)
    assert torch.equal(after_ref, after_new), "RNG consumption changed"
    # sample_env / mean_env of the Gaussian head are exactly dist().sample() / pi()
    x = torch.randn(5, D)
    torch.manual_seed(7); a = net.dist(x).sample()
    torch.manual_seed(7); b = net.sample_env(x)
    assert torch.equal(a, b) and torch.equal(net.mean_env(x), net.pi(x)) and net.env_action(a) is a


def test_spec_roundtrip_and_net_from_ckpt():
    for spec in SPECS:
        full = dict(tp.DEFAULT_SPEC, **spec)
        for A in ((3, 4) if full["action_head"] == "gauss" else (3,)):
            net = tp.ActorCritic(D, A, **full)
            got, obs, act = tp.spec_from_state(net.state_dict())
            assert (got, obs, act) == (full if full["arch"] == "resmlp" else dict(full, expand=4), D, A), (spec, got)
            n2, norm, size = tp.net_from_ckpt(ckpt_of(net))
            assert size == D and n2.spec() == got
            x = torch.randn(7, D)
            assert torch.equal(n2.pi(x), net.pi(x)) and torch.equal(n2.v(x), net.v(x))
            assert net.stored_size == A + net.n_gates
    print(f"  {len(SPECS)} specs round-trip")


def test_old_checkpoints_load():
    """Real checkpoints / pool snapshots on disk (old 2x256 MLPs, some with a shorter observation) load as before."""
    runs = os.environ.get("TEST_RUNS", os.path.join(HERE, "runs"))
    paths = sorted(glob.glob(os.path.join(runs, "gauntlet", "ckpt", "*.pt")))[:6]
    paths += sorted(glob.glob(os.path.join(runs, "farm1", "pool", "*.pt")))[:1]
    for p in paths:
        ck = torch.load(p, weights_only=False)
        size = ck["net"]["pi.0.weight"].shape[1]
        torch.manual_seed(5)
        old = tp.ActorCritic(size, ck["net"]["log_std"].shape[0]); old.load_state_dict(ck["net"])
        r_old = torch.rand(3)
        torch.manual_seed(5)
        net, norm, obs = tp.net_from_ckpt(ck)
        assert torch.equal(torch.rand(3), r_old), "net_from_ckpt must draw like ActorCritic(size, act)"
        assert obs == size and net.spec()["arch"] == "mlp" and net.spec()["action_head"] == "gauss"
        x = torch.randn(4, size)
        assert torch.equal(net.pi(x), old.pi(x))
    print(f"  {len(paths)} real checkpoints load (2x256 tanh MLP, Gaussian)")


def brute_logp(mu, std, logit, a):
    """Independent float64 reference: p(g, x) = prod_k Bern(g_k; sigmoid(l_k)) * prod_{i: gate(i) on} N(x_i)."""
    out = np.zeros(len(a))
    for r in range(len(a)):
        lp = 0.0
        for k, dims in enumerate(tp.STICK_GROUPS):
            p = 1 / (1 + math.exp(-logit[r, k]))
            g = a[r, 3 + k]
            lp += math.log(p if g == 1 else 1 - p)
            if g == 1:
                for i in dims:
                    z = (a[r, i] - mu[r, i]) / std[i]
                    lp += -0.5 * z * z - math.log(std[i]) - 0.5 * math.log(2 * math.pi)
        out[r] = lp
    return out


def test_gated_logprob_entropy():
    torch.manual_seed(0)
    B = 64
    mu = torch.randn(B, 3, dtype=torch.float64) * 0.5
    std = torch.tensor([0.3, 0.7, 1.4], dtype=torch.float64)
    logit = torch.randn(B, 2, dtype=torch.float64) * 2
    d = tp.GatedNormal(mu, std, logit)
    a = d.sample()
    assert a.shape == (B, 5) and set(a[:, 3:].unique().tolist()) <= {0.0, 1.0}
    # every gate combination present: force a few
    a[0, 3:] = torch.tensor([0., 0.]); a[1, 3:] = torch.tensor([1., 0.]); a[2, 3:] = torch.tensor([0., 1.])
    lp = d.log_prob(a).sum(-1).numpy()
    ref = brute_logp(mu.numpy(), std.numpy(), logit.numpy(), a.numpy())
    assert np.allclose(lp, ref, atol=1e-10), np.abs(lp - ref).max()
    # a gate-0 stick's value does not matter
    a2 = a.clone(); a2[0, :3] += 5.0
    assert torch.allclose(d.log_prob(a2)[0], d.log_prob(a)[0])
    # normalisation: sum over gates of the integral over the touched dims = 1 (quadrature, 1-D per dim)
    r = 5
    xs = np.linspace(-12, 12, 20001)
    total = 0.0
    for g0 in (0, 1):
        for g1 in (0, 1):
            p = 1.0
            for k, g in enumerate((g0, g1)):
                pk = 1 / (1 + math.exp(-logit[r, k].item()))
                p *= pk if g else 1 - pk
                if g:
                    for i in tp.STICK_GROUPS[k]:
                        s, m = std[i].item(), mu[r, i].item()
                        p *= np.trapezoid(np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)), xs)
            total += p
    assert abs(total - 1) < 1e-6, total
    # entropy: exact formula vs enumeration + quadrature, and vs Monte Carlo -E[log p]
    H = d.entropy().sum(-1)
    for r in range(4):
        h = 0.0
        hn = [np.trapezoid(-(lambda p: p * np.log(np.maximum(p, 1e-300)))(
            np.exp(-0.5 * ((xs) / std[i].item()) ** 2) / (std[i].item() * math.sqrt(2 * math.pi))), xs) for i in range(3)]
        for k, dims in enumerate(tp.STICK_GROUPS):
            pk = 1 / (1 + math.exp(-logit[r, k].item()))
            h += -(pk * math.log(pk) + (1 - pk) * math.log(1 - pk)) + pk * sum(hn[i] for i in dims)
        assert abs(h - H[r].item()) < 1e-6, (h, H[r].item())
    N = 400_000
    dd = tp.GatedNormal(mu[:1].expand(N, 3), std, logit[:1].expand(N, 2))
    s = dd.sample()
    mc = -dd.log_prob(s).sum(-1)
    se = mc.std().item() / math.sqrt(N)
    assert abs(mc.mean().item() - H[0].item()) < 4 * se, (mc.mean().item(), H[0].item(), se)
    gm = s[:, 3:].mean(0)
    assert torch.allclose(gm, torch.sigmoid(logit[0]), atol=0.005)
    print(f"  log-prob = brute force (max err {np.abs(lp - ref).max():.1e}); entropy = quadrature; "
          f"MC {mc.mean().item():.4f} +- {se:.4f} vs {H[0].item():.4f}")


def test_gated_env_action_and_net():
    torch.manual_seed(1)
    net = tp.ActorCritic(D, 3, action_head="gated")
    x = torch.randn(2000, D)
    with torch.no_grad():
        a, lp, v = net.act(x)
        e = net.env_action(a)
    off_run, off_aim = a[:, 3] == 0, a[:, 4] == 0
    assert off_run.any() and off_aim.any() and (~off_run).any()
    assert (e[off_run, 0] == 0).all() and (e[off_aim, 1:] == 0).all()
    assert torch.equal(e[~off_run, 0], a[~off_run, 0]) and torch.equal(e[~off_aim, 1:], a[~off_aim, 1:3])
    assert torch.allclose(lp, net.dist(x).log_prob(a).sum(-1))
    m = net.mean_env(x)
    assert m.shape == (2000, 3)
    # PPO's gradient flows into gates and Gaussian
    net.zero_grad()
    net.dist(x).log_prob(a).sum().backward()
    assert out_grad(net).abs().sum() > 0 and net.log_std.grad.abs().sum() > 0


def out_grad(net):
    return tp.out_layer(net.pi).weight.grad


def test_neutral_stick_is_centre():
    """Gate 0 sends exactly 0 on the wire (PoleRidersEnv._stick / the native env's copy of it + "%.4f"): the game's
    centred stick (pad dead zone 0.1 -> 0; |run| <= 0.2 = no run force; right-stick magnitude <= 0.2 = pole servo off,
    i.e. both hand-joint motors at speed 0 and max torque 0: the pole swings freely). Native check when the library
    is there."""
    from spr_env import PoleRidersEnv
    for ag in (1, 2):
        assert all(v == 0.0 for v in PoleRidersEnv._stick(ag, (0.0, 0.0, 0.0)))
    try:
        import native_env
        for ag in (1, 2):
            assert all(v == 0.0 for v in native_env.stick(ag, (0.0, 0.0, 0.0)))  # (-0.0 == 0.0)
    except OSError:
        print("  (native library not loaded: wire check skipped)")


def test_league_bank():
    torch.manual_seed(2)
    S = 9
    bank = tp.LeagueBank(S, D, act_size=3)
    nets = {"p256": (tp.ActorCritic(D, 3), D), "p128": (tp.ActorCritic(D, 3, hidden=128), D),
            "p256_obs60": (tp.ActorCritic(60, 3), 60), "res": (tp.ActorCritic(D, 3, arch="resmlp", hidden=64), D),
            "deep": (tp.ActorCritic(D, 3, layers=3), D), "gated": (tp.ActorCritic(D, 3, action_head="gated"), D),
            "wide": (tp.ActorCritic(D, 3, hidden=320), D)}
    opps = {}
    for name, (n, size) in nets.items():
        with torch.no_grad():
            n.log_std.fill_(math.log(0.2))
        nm = tp.RunningNorm(size)
        r = np.random.default_rng(len(name))
        nm.mean, nm.var = r.normal(size=size), r.uniform(0.5, 3, size=size)
        opps[name] = (n, nm, name)
    order = ["p256", "p128", "p256_obs60", "res", "deep", "gated", "wide", "p128", "res"]
    for s, name in enumerate(order):
        bank.load(s, opps[name])
    plain = {"p256", "p128", "p256_obs60"}
    assert [bank.mods[s] is None for s in range(S)] == [n in plain for n in order]
    raw = np.random.default_rng(3).normal(size=(S, D)).astype(np.float32) * 3
    # padded bmm path == the module's own forward (normalised the bank's way, float32)
    ps = [s for s in range(S) if order[s] in plain]
    mu = bank.mu(ps, raw[ps])
    for i, s in enumerate(ps):
        n, nm, _ = opps[order[s]]
        k = len(nm.mean)
        x = torch.clamp((torch.as_tensor(raw[s:s + 1, :k]) - torch.as_tensor(nm.mean, dtype=torch.float32))
                        * torch.as_tensor(1.0 / np.sqrt(nm.var + 1e-8), dtype=torch.float32), -10, 10)
        ref = n.pi(x)[0].detach()
        err = (mu[i] - ref).abs().max().item()
        assert err < 2e-6, (order[s], err)
    # all-plain call: bit for bit the original algorithm
    torch.manual_seed(9); a = bank.act(np.array(ps), raw[ps])
    torch.manual_seed(9)
    s_ = torch.as_tensor(np.array(ps))
    x = torch.clamp((torch.as_tensor(raw[ps]) - bank.mean[s_]) * bank.inv_sd[s_], -10, 10)
    h = torch.tanh(torch.bmm(bank.w1[s_], x.unsqueeze(-1)).squeeze(-1) + bank.b1[s_])
    h = torch.tanh(torch.bmm(bank.w2[s_], h.unsqueeze(-1)).squeeze(-1) + bank.b2[s_])
    m = torch.bmm(bank.w3[s_], h.unsqueeze(-1)).squeeze(-1) + bank.b3[s_]
    assert np.array_equal(a, (m + bank.std[s_] * torch.randn_like(m)).numpy())
    # mixed call: right shape; module rows sample around the module's mean env action
    out = np.stack([bank.act(np.arange(S), raw) for _ in range(400)])
    assert out.shape == (400, S, 3) and np.isfinite(out).all()
    for s, name in enumerate(order):
        n, nm, _ = opps[name]
        xm = torch.as_tensor(nm(raw[s:s + 1, :len(nm.mean)]), dtype=torch.float32)
        if name == "gated":
            with torch.no_grad():
                d = n.dist(xm)
                p = d.gate_prob()[0].numpy()
            frac0 = (out[:, s, 0] == 0).mean(), (out[:, s, 1] == 0).mean()
            assert abs(frac0[0] - (1 - p[0])) < 0.1 and abs(frac0[1] - (1 - p[1])) < 0.1, (frac0, p)
            assert ((out[:, s, 1] == 0) == (out[:, s, 2] == 0)).all()  # the right stick is one gate
            continue
        ref = n.pi(xm)[0].detach().numpy()
        assert np.abs(out[:, s].mean(0) - ref).max() < 0.05, (name, out[:, s].mean(0), ref)
        assert np.abs(out[:, s].std(0) - 0.2).max() < 0.03, (name, out[:, s].std(0))
    # re-loading a slot with a plain opponent clears the module
    bank.names[3] = None
    bank.load(3, opps["p128"])
    assert bank.mods[3] is None
    print("  padded bmm path == module forward (max err < 2e-6); all-plain call bit-identical; mixed slots sample right")


def test_hier_flat_bank_accepts_any_opponent():
    """bridge/hier_env.HierHost.flat_bank is a LeagueBank called with a list of slots."""
    bank = tp.LeagueBank(4, D, act_size=3)
    bank.load(0, (tp.ActorCritic(D, 3), tp.RunningNorm(D), "a"))
    bank.load(2, (tp.ActorCritic(D, 3, arch="resmlp", hidden=32), tp.RunningNorm(D), "b"))
    out = bank.act([0, 2], np.zeros((2, D), np.float32))
    assert out.shape == (2, 3)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"{len(tests)} tests passed")
