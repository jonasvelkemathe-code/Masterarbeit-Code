import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

from w1lab.models import make_critic
from w1lab.data import make_provider_gauss_shift
from w1lab.train import (
    train_dual_constrained,
    train_dual_parseval_projected,
    train_dual_linf_projected,
)

#Parameter der Algorithmen, wir verwenden in diesem Beispiel 20 Theta
DIM = 64
THETAS = [2.0 * i / 19 for i in range(20)]  # 20 Werte von 0 bis 2 (inkl.)

STEPS = 100
BS = 512
LR = 1e-3
EVAL_MC = 10000
LOG_EVERY = 20

WIDTH = 128
DEPTH = 3

BJORCK_ITERS = 3
PARSEVAL_BETA = 0.5
PARSEVAL_PROJ_ITERS = 1
LINF_TAU = 1.0
LINF_PROJ_ITERS = 1

SEED = 123
OUTDIR = Path("runs/gauss/multiple_theta")
OUTDIR.mkdir(parents=True, exist_ok=True)

methods = ["bjorck", "spectral", "parseval", "zeilenp"]
labels = {"bjorck": "Björck", "spectral": "Spectral", "parseval": "Parseval", "zeilenp": "ZeilenP"}

XCOL = "step"
YCOL = "dual"


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

rows = []

for theta in THETAS:
    theta_dir = OUTDIR / f"theta_{theta:.6f}"
    theta_dir.mkdir(parents=True, exist_ok=True)

    provider = make_provider_gauss_shift(DIM, theta, device)
    true_w1 = abs(theta)

    curve_paths = {}


    for method in methods:
        if method == "bjorck":
            critic = make_critic(d=DIM, width=WIDTH, depth=DEPTH, method="bjorck", bjorck_iters=BJORCK_ITERS).to(device)
            log_path = theta_dir / "curve_bjoerck.csv"
            train_dual_constrained(
                critic, provider,
                steps=STEPS, bs=BS, lr=LR,
                eval_mc=EVAL_MC, log_every=LOG_EVERY,
                log_path=str(log_path),
            )
            curve_paths[method] = log_path

        elif method == "spectral":
            critic = make_critic(d=DIM, width=WIDTH, depth=DEPTH, method="spectral").to(device)
            log_path = theta_dir / "curve_spectral.csv"
            train_dual_constrained(
                critic, provider,
                steps=STEPS, bs=BS, lr=LR,
                eval_mc=EVAL_MC, log_every=LOG_EVERY,
                log_path=str(log_path),
            )
            curve_paths[method] = log_path

        elif method == "parseval":
            critic = make_critic(d=DIM, width=WIDTH, depth=DEPTH, method="parseval").to(device)
            log_path = theta_dir / "curve_parseval.csv"
            train_dual_parseval_projected(
                critic, provider,
                steps=STEPS, bs=BS, lr=LR,
                eval_mc=EVAL_MC, log_every=LOG_EVERY,
                beta=PARSEVAL_BETA, proj_iters=PARSEVAL_PROJ_ITERS,
                rows_if_out_ge_in=True,
                log_path=str(log_path),
            )
            curve_paths[method] = log_path

        elif method == "zeilenp":
            critic = make_critic(d=DIM, width=WIDTH, depth=DEPTH, method="parseval").to(device)
            log_path = theta_dir / "curve_zeilenp.csv"
            train_dual_linf_projected(
                critic, provider,
                steps=STEPS, bs=BS, lr=LR,
                eval_mc=EVAL_MC, log_every=LOG_EVERY,
                tau=LINF_TAU, proj_iters=LINF_PROJ_ITERS,
                log_path=str(log_path),
            )
            curve_paths[method] = log_path


    for method in methods:
        df = pd.read_csv(curve_paths[method])
        dual_last = float(df[YCOL].iloc[-1])  # letzter dual
        diff = dual_last - true_w1
        rows.append({
            "theta": float(theta),
            "method": method,
            "dual_last": dual_last,
            "true_w1": true_w1,
            "diff": diff,
            "abs_diff": abs(diff),
        })


res = pd.DataFrame(rows)
out_csv = OUTDIR / "theta_error.csv"
res.to_csv(out_csv, index=False)


plt.figure(figsize=(10, 6))

for method in methods:
    sub = res[res["method"] == method].sort_values("theta")
    plt.plot(sub["theta"], sub["abs_diff"], marker="o", label=labels[method])

plt.xlabel(r"$\theta$")
plt.ylabel(r"$|\hat d_D(\mathrm{last}) - W_1(\theta)|$")
plt.title(rf"gauss: Fehler der letzten Iteration vs. $\theta$ (d={DIM})")
plt.grid(True)
plt.legend()
plt.tight_layout()

out_png = OUTDIR / "error_abs_last_vs_theta.png"
plt.savefig(out_png, dpi=200)
plt.show()
