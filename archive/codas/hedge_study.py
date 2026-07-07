"""Hedging control study on real episodes: does a neural policy beat classical
hedges once transaction costs bite?

Position: short 1x the 1-day ATM straddle (real entry V0). Policies choose the ES
hedge h_j on a 5-min grid; costs c per unit |dh| (entry, rebalances, unwind).
  unhedged   h = 0
  bs_5m      Black-Scholes straddle delta every 5 min (over-trades under costs)
  bs_30m     same, every 30 min
  ww         Whalley-Wilmott no-trade band around BS delta, lambda tuned train+val
  deep_real  learned band nesting WW (below), trained on the 450 real train episodes
  deep_sim   same policy class, trained sim-to-real: ~6k simulated episodes built
             ONLY from train-year paths (episode resampling, BS-consistent scale
             jitter, fair-priced entry so the sim carries no premium edge), then
             evaluated once on real held-out years

Policy class: h = clip(h_prev, delta +- b), b = ww_band * exp(net(state)), net
zero-init -> starts EXACTLY at tuned WW and learns state-dependent band width.
(A residual-on-delta form h = delta + tanh(net) was tried first and LOST to WW at
every cost level: a smooth stateless map inherits BS turnover and cannot discover
band hysteresis by gradient descent. Parametrization is the lesson.)

Split: train 2021-23, val 2024 (all selection), TEST 2025-26 untouched.
Usage: python scripts/hedge_study.py [--costs 0.15,0.5,1.0] [--lam 1.0]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from volforecast.intraday import FOMC_DATES

STEPS = 78
TAU0 = 1.0 / 252.0


def bs_delta_gamma(S, K, sigma0, tau):
    """Straddle delta 2*Phi(d1)-1 and gamma 2*phi(d1)/(S sig sqrt(tau)), per step."""
    t = np.maximum(tau[None, :STEPS], 1e-9)
    st = sigma0[:, None] * np.sqrt(t)
    d1 = (np.log(S[:, :STEPS] / K[:, None]) + 0.5 * st ** 2) / st
    return 2 * norm.cdf(d1) - 1, 2 * norm.pdf(d1) / (S[:, :STEPS] * st)


def pnl(S, K, V0, h, c):
    gross = V0 + (h * np.diff(S, axis=1)).sum(1) - np.abs(S[:, -1] - K)
    cost = c * (np.abs(h[:, 0]) + np.abs(np.diff(h, axis=1)).sum(1) + np.abs(h[:, -1]))
    return gross - cost, cost


def cvar5(losses):
    k = max(1, int(np.ceil(0.05 * len(losses))))
    return np.mean(np.sort(losses)[-k:])


def ww_policy(delta, gamma, c, lam):
    band = (1.5 * c * gamma ** 2 / lam) ** (1 / 3)
    h = np.zeros_like(delta)
    prev = np.zeros(len(delta))
    for j in range(STEPS):
        prev = np.clip(prev, delta[:, j] - band[:, j], delta[:, j] + band[:, j])
        h[:, j] = prev
    return h


def make_dataset(S, K, V0, sigma0, fomc, c, lam_ww):
    delta, gamma = bs_delta_gamma(S, K, sigma0, np.array(TAU_GRID))
    bw = (1.5 * c * gamma ** 2 / lam_ww) ** (1 / 3)
    return dict(S=S, K=K, V0=V0, sigma0=sigma0, delta=delta, bw=bw, fomc=fomc)


def simulate(S, K, V0, sigma0, fomc, n_sim, rng):
    """Sim episodes from train paths only: resample + scale jitter, fair-priced V0."""
    r = np.diff(np.log(S), axis=1)
    i = rng.integers(0, len(S), n_sim)
    u = rng.uniform(0.65, 1.5, n_sim)
    S0 = S[i, 0]
    paths = S0[:, None] * np.exp(np.cumsum(u[:, None] * r[i], axis=1))
    Ssim = np.column_stack([S0, paths])
    sig = u * sigma0[i]
    V0s = np.sqrt(2 / np.pi) * S0 * sig * np.sqrt(TAU0)
    return Ssim, K[i], V0s, sig, fomc[i]


def train_deep(data_tr, data_va, tau, c, lam, seed=0, epochs=400, batch=1024):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tt = T(tau)

    def run(d, idx=None):
        sel = slice(None) if idx is None else idx
        s = T(d["S"][sel]); k = T(d["K"][sel]); v0 = T(d["V0"][sel])
        sig = T(d["sigma0"][sel]); db = T(d["delta"][sel]); bwid = T(d["bw"][sel])
        fm = T(d["fomc"][sel])
        n = len(s)
        anchor = s[:, 0] * sig * np.sqrt(TAU0)
        h_prev = torch.zeros(n)
        wealth = v0.clone()
        cost = torch.zeros(n)
        rets, H = [], []
        for j in range(STEPS):
            rv = (torch.stack(rets[-6:], 1).std(1) * np.sqrt(78 * 252)
                  if len(rets) >= 3 else sig)
            x = torch.stack([
                (s[:, j] - k) / anchor,
                torch.sqrt(tt[j] / tt[0]).expand(n),
                h_prev, db[:, j],
                torch.clamp(torch.log(rv / sig), -2, 2),
                torch.full((n,), j / STEPS), fm,
            ], 1)
            b = bwid[:, j] * torch.exp(torch.clamp(net(x).squeeze(1), -2.5, 2.5))
            h = torch.clamp(h_prev, db[:, j] - b, db[:, j] + b)
            cost = cost + c * (h - h_prev).abs()
            wealth = wealth + h * (s[:, j + 1] - s[:, j])
            rets.append(torch.log(s[:, j + 1] / s[:, j]))
            H.append(h)
            h_prev = h
        cost = cost + c * h_prev.abs()  # unwind at the close
        pl = wealth - (s[:, -1] - k).abs() - cost
        kk = max(1, int(np.ceil(0.05 * n)))
        return (-pl.mean() + lam * torch.topk(-pl, kk).values.mean(),
                torch.stack(H, 1))

    rng = np.random.default_rng(seed)
    n_tr = len(data_tr["S"])
    best, best_state = np.inf, {k2: v.clone() for k2, v in net.state_dict().items()}
    for ep in range(epochs):
        idx = rng.integers(0, n_tr, batch) if n_tr > batch else None
        opt.zero_grad()
        objv, _ = run(data_tr, idx)
        objv.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            with torch.no_grad():
                vobj, _ = run(data_va)
            if float(vobj) < best:
                best = float(vobj)
                best_state = {k2: v.clone() for k2, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    def h_of(data):
        with torch.no_grad():
            _, H = run(data)
        return H.numpy()

    return h_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", default="0.15,0.5,1.0", help="pts per unit |dh|, ladder")
    ap.add_argument("--lam", type=float, default=1.0, help="CVaR weight in EVAL objective")
    ap.add_argument("--nsim", type=int, default=6000)
    args = ap.parse_args()

    z = np.load("artifacts/hedge_episodes.npz")
    S, tau, K, V0, sigma0, year = z["S"], z["tau"], z["K"], z["V0"], z["sigma0"], z["year"]
    global TAU_GRID
    TAU_GRID = tau
    fomc = pd.to_datetime(z["date"]).isin(FOMC_DATES).astype(float)
    tr = np.where(year <= 2023)[0]
    va = np.where(year == 2024)[0]
    te = np.where(year >= 2025)[0]
    obj = lambda pl: -pl.mean() + args.lam * cvar5(-pl)
    print(f"episodes train/val/test: {len(tr)}/{len(va)}/{len(te)} | "
          f"eval objective: -mean + {args.lam}*CVaR5 (selection on train/val ONLY)")

    delta, gamma = bs_delta_gamma(S, K, sigma0, tau)
    tv = np.concatenate([tr, va])
    rng = np.random.default_rng(0)
    for c in [float(x) for x in args.costs.split(",")]:
        lams, scores = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], []
        for lam in lams:
            pl, _ = pnl(S[tv], K[tv], V0[tv], ww_policy(delta[tv], gamma[tv], c, lam), c)
            scores.append(obj(pl))
        lam_ww = lams[int(np.argmin(scores))]

        d_all = make_dataset(S, K, V0, sigma0, fomc, c, lam_ww)
        sub = lambda d, ix: {k2: v[ix] for k2, v in d.items()}
        Ss, Ks, V0s, sigs, fms = simulate(S[tr], K[tr], V0[tr], sigma0[tr],
                                          fomc[tr], args.nsim, rng)
        d_sim = make_dataset(Ss, Ks, V0s, sigs, fms, c, lam_ww)

        policies = {
            "unhedged": np.zeros((len(K), STEPS)),
            "bs_5m": delta,
            "bs_30m": np.repeat(delta[:, ::6], 6, axis=1)[:, :STEPS],
            "ww": ww_policy(delta, gamma, c, lam_ww),
        }
        sel_note = {}
        for name, d_train, ep_n in [("deep_real", sub(d_all, tr), 400),
                                    ("deep_sim", d_sim, 600)]:
            best_v, best_hof, best_lam = np.inf, None, None
            for lam_tr in (0.5, 1.0, 2.0):
                h_of = train_deep(d_train, sub(d_all, va), tau, c, lam_tr,
                                  seed=int(lam_tr * 10), epochs=ep_n)
                plv, _ = pnl(S[va], K[va], V0[va], h_of(sub(d_all, va)), c)
                if obj(plv) < best_v:
                    best_v, best_hof, best_lam = obj(plv), h_of, lam_tr
            policies[name] = np.zeros((len(K), STEPS))
            for ix in (tr, va, te):
                policies[name][ix] = best_hof(sub(d_all, ix))
            sel_note[name] = best_lam

        print(f"\n=== cost {c} pts/|dh| (ww lam {lam_ww}; train-lam real "
              f"{sel_note['deep_real']}, sim {sel_note['deep_sim']}) | TEST 2025-26 ===")
        print(f"    {'policy':9s} {'mean':>7s} {'std':>7s} {'CVaR5':>8s} {'cost':>6s} "
              f"{'turnover':>9s} {'objective':>10s}")
        res = {}
        for name, h in policies.items():
            pl, cost = pnl(S[te], K[te], V0[te], h[te], c)
            res[name] = pl
            turn = (np.abs(h[te, 0]) + np.abs(np.diff(h[te], axis=1)).sum(1)
                    + np.abs(h[te, -1]))
            print(f"    {name:9s} {pl.mean():+7.2f} {pl.std():7.2f} {-cvar5(-pl):+8.2f} "
                  f"{cost.mean():6.2f} {turn.mean():9.2f} {obj(pl):10.2f}")
        for nm in ("deep_real", "deep_sim"):
            d = res[nm] - res["ww"]
            print(f"    paired {nm}-vs-ww: mean {d.mean():+.3f} pts, "
                  f"t = {d.mean() / (d.std() / np.sqrt(len(d))):+.2f}")


if __name__ == "__main__":
    main()
