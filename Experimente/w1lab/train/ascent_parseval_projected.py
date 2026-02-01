import time
import csv
from pathlib import Path
import torch
from w1lab.layers.parseval import parseval_project_module_

@torch.no_grad()
def _dual_estimate(critic, provider, n_mc: int):
    critic.eval()
    X, Y = provider.sample(n_mc)
    return (critic(X).mean() - critic(Y).mean()).item()

def train_dual_parseval_projected(
    critic,
    provider,
    steps=2000,
    bs=512,
    lr=1e-3,
    eval_mc=20000,
    log_every=20,
    beta=1e-4,
    proj_iters=1,
    rows_if_out_ge_in=True,
    log_path=None,
):
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    true = provider.true_w1()
    t0 = time.time()

    _rows = [] if log_path is not None else None

    for step in range(1, steps + 1):
        critic.train()
        X, Y = provider.sample(bs)
        loss = -(critic(X).mean() - critic(Y).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for _ in range(proj_iters):
            parseval_project_module_(critic, beta=beta, rows_if_out_ge_in=rows_if_out_ge_in)

        if step % log_every == 0 or step == 1 or step == steps:
            est = _dual_estimate(critic, provider, eval_mc)
            gap = abs(est - true)
            elapsed = time.time() - t0
            print(f"[parseval-proj] step {step:4d} | dual {est:.4f} | true {true:.4f} | gap {gap:.4f}")
            if _rows is not None:
                _rows.append({"step": step, "dual": est, "true": float(true), "gap": gap, "elapsed_s": elapsed})

    est = _dual_estimate(critic, provider, eval_mc)
    print(f"done in {time.time()-t0:.1f}s | final dual {est:.4f} | true {true:.4f}")

    if _rows is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "dual", "true", "gap", "elapsed_s"])
            writer.writeheader()
            writer.writerows(_rows)
        print(f"[log] wrote {len(_rows)} rows to {path}")

    return est, true
