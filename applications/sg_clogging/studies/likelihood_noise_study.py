"""Does the random simplex weighting in the MCMC likelihood matter?

For the last cycle of each SG (first refinement iteration, prior = DoE prior), compares:

  * likelihood variants
      current : max + log(sum(exp(ll) - max)) - log(W)   (formula of the original mcmc_integration.py)
      avg     : logsumexp(ll) - log(W)                    (the intended average over the weights)
      max     : max(ll)                                   (profile over the weights)
  * weight schemes
      random  : fresh Dirichlet(1) draws at every likelihood call (production behaviour, W = 1000)
      fixed   : one scrambled-Sobol design on the simplex, reused at every call (W = 2^10, 2^13, 2^16)

Part 1: noise of the log-likelihood at fixed x.
Part 2: exact grid posteriors (uniform prior, 1-D active parameter) for every fixed design.
Part 3: real OpenTURNS random-walk MH runs over several seeds, random vs fixed design.

The 15 calibration members only depend on the active parameter, so their outputs are
precomputed on a grid and linearly interpolated -- the likelihood is then cheap.
Run it from the application root; outputs go to results_noise_study/.
"""
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..'))                       # applications/sg_clogging
sys.path.insert(0, APP)                                               # clogging_prognostics
sys.path.insert(0, os.path.abspath(os.path.join(APP, '..', '..')))    # bayes_fusion (repo root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from scipy.special import logsumexp
from scipy.stats import qmc, wasserstein_distance
from sklearn.model_selection import train_test_split

from clogging_prognostics import PipelineConfig
from clogging_prognostics.data_io import build_doe, build_per_variable_scalers, load_dataset
from bayes_fusion.screening import select_active_index
from bayes_fusion.surrogates import build_calibration_ensemble
from bayes_fusion.mcmc import BayesCalibrationMCMC, sample_positive_l1_sphere

warnings.filterwarnings("ignore")
OUT = "results_noise_study"
os.makedirs(OUT, exist_ok=True)

SGS = {"SGA": 2, "SGB": 1}                  # DATA/ prefix -> last cycle index
ALIAS = {"SGA": "SG A", "SGB": "SG B"}      # names used in printed output and files
W_RANDOM = 1000                            # nb_weights of the full run
W_FIXED = [2**10, 2**13, 2**16]
GRID = 400
N_NOISE = 200
N_SEEDS = 10
VARIANTS = ["current", "avg", "max"]


def combine(ll, variant):
    W = ll.size
    if variant == "current":
        m = ll.max()
        return m + np.log(np.sum(np.exp(ll) - m)) - np.log(W)
    if variant == "avg":
        return logsumexp(ll) - np.log(W)
    return ll.max()


def sobol_weights(W, n_members, seed):
    u = qmc.Sobol(d=n_members, scramble=True, seed=seed).random(W)
    z = -np.log(np.clip(u, 1e-12, 1.0))
    return z / z.sum(axis=1, keepdims=True)


class Problem:
    def __init__(self, sg_tag, cycle_index):
        t0 = time.time()
        self.sg_tag = sg_tag
        cfg = PipelineConfig(base_directory="DATA/")
        ds = load_dataset("DATA/", sg_tag)
        ot.RandomGenerator.SetSeed(42)
        X, y, _ = build_doe(ds, cfg)
        scalers = build_per_variable_scalers(ds, cfg)
        ft = ds.data_without_time_division[:, 0] * 24
        y_int = np.asarray([np.interp(ft, ds.simulation_time * 24, y[i]) for i in range(len(y))])
        X_tr, _, y_tr, _ = train_test_split(X, y_int, train_size=cfg.train_size, random_state=42)
        self.active = select_active_index(X, y)
        self.name = cfg.prior.names[self.active]
        self.lo, self.hi = cfg.prior.bounds[self.active]
        self.step = cfg.mcmc.step_sizes[self.active]
        print(f"[{ALIAS[sg_tag]}] cycle {cycle_index}, HSIC-active variable: {self.name}, "
              f"building 15-member ensemble ...", flush=True)
        mms = build_calibration_ensemble(X_tr, y_tr, cfg, ft, self.active)
        self.n_members = len(mms)
        self.data = ds.data_with_time_division[cycle_index]
        self.tidx = ds.time_indices[cycle_index]
        self.cal_funcs = [BayesCalibrationMCMC.CalibrationFunction(
            m, self.data, self.tidx, scalers[self.active]) for m in mms]

        self.xg = np.linspace(self.lo, self.hi, GRID)
        self.cal_grid = np.stack([
            np.stack([np.asarray(f([x])).reshape(-1) for f in self.cal_funcs]) for x in self.xg
        ])                                                     # (GRID, n_members, T_sel)
        print(f"[{ALIAS[sg_tag]}] ensemble + grid ready in {time.time() - t0:.0f}s "
              f"(members={self.n_members}, T_sel={self.cal_grid.shape[2]})", flush=True)

    def cal_at(self, x):
        j = np.clip(np.searchsorted(self.xg, x) - 1, 0, GRID - 2)
        a = (x - self.xg[j]) / (self.xg[j + 1] - self.xg[j])
        return (1 - a) * self.cal_grid[j] + a * self.cal_grid[j + 1]

    def ll_vec(self, cal, Wt):
        mo = Wt @ cal
        ll = np.zeros(Wt.shape[0])
        for d, idx in zip(self.data, self.tidx):
            diff = d[:, 1][None, :] - mo[:, idx]
            if diff.shape[1] == 0:
                continue
            ll += -np.log(np.sum(diff ** 2, axis=1)) * diff.shape[1] / 2.0
        return ll

    def check_against_production(self):
        real = BayesCalibrationMCMC.LikelihoodFunction(
            self.data, self.tidx, calibration_functions=self.cal_funcs, nb_weights=W_RANDOM)()
        x = float(self.xg[GRID // 2])
        np.random.seed(123)
        a = real([x])[0]
        np.random.seed(123)
        Wt = sample_positive_l1_sphere(self.n_members, W_RANDOM)
        b = combine(self.ll_vec(self.cal_grid[GRID // 2], Wt), "current")
        xo = 0.5 * (self.xg[GRID // 2] + self.xg[GRID // 2 + 1])
        np.random.seed(123)
        c = real([xo])[0]
        np.random.seed(123)
        d = combine(self.ll_vec(self.cal_at(xo), sample_positive_l1_sphere(self.n_members, W_RANDOM)), "current")
        print(f"[{ALIAS[self.sg_tag]}] check vs production likelihood: on-grid {a:.6f} vs {b:.6f}; "
              f"off-grid (interp) {c:.6f} vs {d:.6f}", flush=True)

    def grid_loglik(self, Wt, variant):
        return np.array([combine(self.ll_vec(self.cal_grid[g], Wt), variant) for g in range(GRID)])


def grid_posterior(L):
    p = np.exp(L - np.nanmax(L))
    p = np.where(np.isfinite(p), p, 0.0)
    return p / p.sum()


def post_stats_grid(xg, p):
    m = float(np.sum(xg * p))
    s = float(np.sqrt(np.sum((xg - m) ** 2 * p)))
    c = np.cumsum(p)
    q = lambda a: float(xg[np.searchsorted(c, a)])
    return dict(mean=m, std=s, q05=q(0.05), q50=q(0.5), q95=q(0.95))


def sample_stats(s):
    return dict(mean=float(s.mean()), std=float(s.std()), q05=float(np.quantile(s, .05)),
                q50=float(np.quantile(s, .5)), q95=float(np.quantile(s, .95)))


def run_rwmh(prob, variant, mode, W, seed, fixed_Wt=None):
    calls = [0]

    def f(x):
        calls[0] += 1
        cal = prob.cal_at(float(x[0]))
        Wt = sample_positive_l1_sphere(prob.n_members, W) if mode == "random" else fixed_Wt
        return [combine(prob.ll_vec(cal, Wt), variant)]

    lik = ot.PythonFunction(1, 1, f)
    ot.RandomGenerator.SetSeed(seed)
    np.random.seed(seed)
    support = ot.Interval([prob.lo], [prob.hi])
    x_init = ot.Uniform(prob.lo, prob.hi).getSample(3)
    proposal = ot.Uniform(-prob.step, prob.step)
    chains = []
    per_chain_calls = None
    for j in range(3):
        c0 = calls[0]
        sampler = ot.RandomWalkMetropolisHastings(lik, support, x_init[j], proposal)
        sampler.setBurnIn(100)
        chains.append(np.asarray(sampler.getSample(1000)).reshape(-1))
        per_chain_calls = calls[0] - c0
    s = np.concatenate(chains)
    stick = float(np.mean([np.mean(c[1:] == c[:-1]) for c in chains]))
    return s, stick, per_chain_calls


def study(sg_tag, cycle_index):
    prob = Problem(sg_tag, cycle_index)
    prob.check_against_production()
    res = dict(sg=ALIAS[sg_tag], active=prob.name)
    xg = prob.xg

    # fixed designs (seed 0) and the largest one as the per-variant reference
    designs = {W: sobol_weights(W, prob.n_members, seed=0) for W in W_FIXED}
    ref_post, ref_L = {}, {}
    for v in VARIANTS:
        ref_L[v] = prob.grid_loglik(designs[W_FIXED[-1]], v)
        ref_post[v] = grid_posterior(ref_L[v])

    # ---- Part 1: noise of the log-likelihood at fixed x ------------------------------------ #
    print(f"\n[{ALIAS[sg_tag]}] PART 1 -- std of the log-likelihood over {N_NOISE} random-weight draws "
          f"(W={W_RANDOM}) at fixed x", flush=True)
    st = post_stats_grid(xg, ref_post["avg"])
    picks = {"mode": xg[np.argmax(ref_post["avg"])], "mean-1sd": st["mean"] - st["std"],
             "mean+1sd": st["mean"] + st["std"]}
    part1 = []
    print(f"{'x label':>9} {'x':>11} {'variant':>8} {'mean':>10} {'std':>8} {'nan%':>5} {'fixed 2^16':>11}")
    for lab, x in picks.items():
        x = float(np.clip(x, prob.lo, prob.hi))
        cal = prob.cal_at(x)
        for v in VARIANTS:
            vals = np.array([combine(prob.ll_vec(cal, sample_positive_l1_sphere(prob.n_members, W_RANDOM)), v)
                             for _ in range(N_NOISE)])
            fx = combine(prob.ll_vec(cal, designs[W_FIXED[-1]]), v)
            row = dict(x_label=lab, x=x, variant=v, mean=float(np.nanmean(vals)),
                       std=float(np.nanstd(vals)), nan_frac=float(np.mean(~np.isfinite(vals))), fixed=float(fx))
            part1.append(row)
            print(f"{lab:>9} {x:11.4e} {v:>8} {row['mean']:10.3f} {row['std']:8.3f} "
                  f"{100 * row['nan_frac']:5.0f} {fx:11.3f}")
    res["part1"] = part1

    # ---- Part 2: exact grid posteriors for the fixed designs ------------------------------- #
    print(f"\n[{ALIAS[sg_tag]}] PART 2 -- exact grid posteriors, fixed Sobol designs (uniform prior)", flush=True)
    print(f"{'variant':>8} {'W':>6} {'mean':>11} {'std':>11} {'q05':>11} {'q95':>11} {'W1 to 2^16':>11}")
    part2 = []
    for v in VARIANTS:
        for W in W_FIXED:
            p = grid_posterior(prob.grid_loglik(designs[W], v)) if W != W_FIXED[-1] else ref_post[v]
            s = post_stats_grid(xg, p)
            w1 = wasserstein_distance(xg, xg, p, ref_post[v])
            part2.append(dict(variant=v, W=W, w1_to_ref=float(w1), **s))
            print(f"{v:>8} {W:6d} {s['mean']:11.4e} {s['std']:11.4e} {s['q05']:11.4e} {s['q95']:11.4e} {w1:11.3e}")
    # sensitivity of the fixed design to its scrambling seed (W=2^13)
    print(f"  design-seed sensitivity at W=2^13 (5 scrambles), posterior mean std across seeds:")
    for v in VARIANTS:
        means = [post_stats_grid(xg, grid_posterior(prob.grid_loglik(sobol_weights(2**13, prob.n_members, s), v)))["mean"]
                 for s in range(1, 6)]
        print(f"   {v:>8}: {np.std(means):.3e}")
    res["part2"] = part2

    # ---- Part 3: real RWMH runs over seeds ------------------------------------------------- #
    print(f"\n[{ALIAS[sg_tag]}] PART 3 -- OpenTURNS RWMH, 3 chains x 1000, {N_SEEDS} seeds each", flush=True)
    fixed13 = designs[2**13]
    configs = [("random W=1000, current", "current", "random", W_RANDOM),
               ("random W=1000, avg", "avg", "random", W_RANDOM),
               ("fixed 2^13, current", "current", "fixed", 2**13),
               ("fixed 2^13, avg", "avg", "fixed", 2**13),
               ("fixed 2^13, max", "max", "fixed", 2**13)]
    part3, kdes = [], {}
    print(f"{'config':>24} {'mean(mean)':>11} {'sd(mean)':>10} {'mean(std)':>10} {'sd(std)':>10} "
          f"{'W1->ref':>10} {'sticky':>7} {'calls/step':>10}")
    for name, v, mode, W in configs:
        stats, w1s, sticks, allsamp = [], [], [], []
        for r in range(N_SEEDS):
            s, stick, calls = run_rwmh(prob, v, mode, W, seed=1000 + r,
                                       fixed_Wt=fixed13 if mode == "fixed" else None)
            stats.append(sample_stats(s))
            w1s.append(wasserstein_distance(s, xg, v_weights=ref_post[v]))
            sticks.append(stick)
            allsamp.append(s)
        means = np.array([q["mean"] for q in stats]); stds = np.array([q["std"] for q in stats])
        row = dict(config=name, mean_of_mean=float(means.mean()), sd_of_mean=float(means.std()),
                   mean_of_std=float(stds.mean()), sd_of_std=float(stds.std()),
                   w1_to_ref=float(np.mean(w1s)), sticky=float(np.mean(sticks)), calls_per_step=calls / 1100)
        part3.append(row)
        kdes[name] = np.concatenate(allsamp)
        print(f"{name:>24} {row['mean_of_mean']:11.4e} {row['sd_of_mean']:10.2e} {row['mean_of_std']:10.3e} "
              f"{row['sd_of_std']:10.2e} {row['w1_to_ref']:10.2e} {row['sticky']:7.2f} {row['calls_per_step']:10.2f}",
              flush=True)
    res["part3"] = part3

    # ---- figure ---------------------------------------------------------------------------- #
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for v, c in zip(VARIANTS, ["tab:red", "tab:blue", "tab:green"]):
        axes[0].plot(xg, ref_post[v] / (xg[1] - xg[0]), color=c, label=f"fixed 2^16, {v}")
    axes[0].set_title(f"{ALIAS[sg_tag]}: exact posteriors of {prob.name}, fixed design")
    axes[0].legend(fontsize=7)
    for (name, _, _, _), c in zip(configs, ["tab:red", "tab:blue", "tab:orange", "tab:cyan", "tab:green"]):
        axes[1].plot(xg, gaussian_kde(kdes[name])(xg), color=c, label=name, lw=1.3)
    axes[1].set_title("RWMH samples pooled over seeds")
    axes[1].legend(fontsize=7)
    for a in axes:
        a.set_xlabel(prob.name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{sg_tag}_noise_study.png"), dpi=130)
    plt.close(fig)
    return res


if __name__ == "__main__":
    all_res = [study(sg, ci) for sg, ci in SGS.items()]
    with open(os.path.join(OUT, "noise_study.json"), "w") as fh:
        json.dump(all_res, fh, indent=1)
    print("\nDONE")
