#!/usr/bin/env python3
"""Throughput of the batched native env (bridge/native_env.py): env steps/s (matches x steps / s).

    bench_env.py [--n 224 512] [--threads 0] [--frame-skip 4] [--seconds 10]

Actions are random (resampled every 8 steps from a pre-generated pool, so Python overhead is ~0);
sd_frac 0.15 and max_steps 5400 like training.
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bridge"))
from native_env import NativeHost  # noqa: E402


def bench(n, threads, frame_skip, seconds, K=16):
    host = NativeHost(n // K, K, frame_skip, 0.15, 5400, seed=0, threads=threads)
    t0 = time.time()
    host.reset_all()
    t_reset = time.time() - t0
    pool = np.clip(np.random.default_rng(0).normal(0, 1, (16, n, 2, 3)), -1, 1).astype(np.float32)
    for i in range(20):  # warm up
        host.step(pool[i % 16])
    steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        host.step(pool[(steps // 8) % 16])
        steps += 1
    dt = time.time() - t0
    host.close()
    sps = steps * n / dt
    print(f"N={n:4d} ({n // K}x{K}) frame_skip {frame_skip} threads {threads or 'all'}: {sps:8.0f} env steps/s "
          f"({1e3 * dt / steps:.2f} ms/batch step, {sps * frame_skip:.0f} match ticks/s; reset_all {t_reset:.2f}s)",
          flush=True)
    return sps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[224, 512])
    ap.add_argument("--threads", type=int, nargs="+", default=[0])
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=10)
    args = ap.parse_args()
    print(f"load average {os.getloadavg()[0]:.1f} on {os.cpu_count()} cores")
    for n in args.n:
        for t in args.threads:
            bench(n, t, args.frame_skip, args.seconds)


if __name__ == "__main__":
    main()
