import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

from w1lab.models import make_critic
from w1lab.data import make_provider_cone_dirac_shell
from w1lab.train import (
    train_dual_constrained,
    train_dual_parseval_projected,
    train_dual_linf_projected,
)

#Wir setzen hier die Parameter für die Algorithmen und die Dimension der Wahrscheinlichkeitsverteilung
DIM = 16

STEPS = 40
BS = 512
LR = 1e-3
EVAL_MC = 10000
LOG_EVERY = 20

WIDTH = 256
DEPTH = 4

BJORCK_ITERS = 3
PARSEVAL_BETA = 0.5
PARSEVAL_PROJ_ITERS = 1
LINF_TAU = 1.0
LINF_PROJ_ITERS = 1

SEED = 123
OUTDIR = Path("runs/conedirac/single_theta")
OUTDIR.mkdir(parents=True, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)


provider = make_provider_cone_dirac_shell(DIM, device)

methods = ["bjorck", "spectral", "parseval", "zeilenp"]
labels = {"bjorck": "Björck", "spectral": "Spectral", "parseval": "Parseval", "zeilenp": "ZeilenP"}

curve_paths = {}


for method in methods:
    if method == "bjorck":
        critic = make_critic(d=DIM, width=WIDTH, depth=DEPTH, method="bjorck", bjorck_iters=BJORCK_ITERS).to(device)
        log_path = OUTDIR / "curve_bjoerck.csv"
        train_dual_constrained(
            critic, provider,
            steps=STEPS, bs=BS, lr=LR,
            eval_mc=EVAL_MC, log_every=LOG_EVERY,
            log_path=str(log_path),
        )
        curve_paths[method] = log_path

    elif method == "spectral":
        critic = make_critic(d=DIM, width=WIDTH, depth=DEPTH, method="spectral").to(device)
        log_path = OUTDIR / "curve_spectral.csv"
        train_dual_constrained(
            critic, provider,
            steps=STEPS, bs=BS, lr=LR,
            eval_mc=EVAL_MC, log_every=LOG_EVERY,
            log_path=str(log_path),
        )
        curve_paths[method] = log_path

    elif method == "parseval":
        critic = make_critic(d=DIM, width=WIDTH, depth=DEPTH, method="parseval").to(device)
        log_path = OUTDIR / "curve_parseval.csv"
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
        log_path = OUTDIR / "curve_zeilenp.csv"
        train_dual_linf_projected(
            critic, provider,
            steps=STEPS, bs=BS, lr=LR,
            eval_mc=EVAL_MC, log_every=LOG_EVERY,
            tau=LINF_TAU, proj_iters=LINF_PROJ_ITERS,
            log_path=str(log_path),
        )
        curve_paths[method] = log_path


XCOL = "step"
YCOL = "dual"

plt.figure(figsize=(10, 6))
for method in methods:
    df = pd.read_csv(curve_paths[method])
    plt.plot(df[XCOL], df[YCOL], label=labels[method])

plt.axhline(1.0, linestyle="--", linewidth=1)  # true_w1 = 1.0

plt.xlabel("Iteration")
plt.ylabel(r"Schätzer $\hat d_D$")
plt.title(rf":Hochdimensionale Kegelverteilung (Radius=1, d={DIM})")
plt.grid(True)
plt.ylim(0, 1.2)
plt.legend()
plt.tight_layout()

out_png = OUTDIR / "vergleich_methoden_conedirac_single_theta.png"
plt.savefig(out_png, dpi=200)
plt.show()

