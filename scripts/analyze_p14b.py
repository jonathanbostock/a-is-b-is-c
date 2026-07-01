"""Pull and summarize Pythia-1.4B sweep results, regardless of batch."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

RUN_ROOT = Path("runs/p14b_pulled")

def composable(s, t, train_edges):
    return any((s, m) in train_edges and (m, t) in train_edges for m in range(6))

def summarize(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        for cfg_dir in root.rglob("*_llm_*"):
            evf = cfg_dir / "eval_results.json"
            if not evf.exists(): continue
            cfg = cfg_dir.name.split("_llm_")[0]
            d = json.load(open(evf))
            rep0 = sorted([r for r in d if r["repeat_id"] == 0], key=lambda r: r["step"])
            if not rep0: continue
            last = rep0[-1]
            train_edges = set(tuple(e["edge"]) for e in last["train_edges"])
            te_cmp = [e["accuracy"] for e in last["test_edges"] if composable(*e["edge"], train_edges)]
            te_nc  = [e["accuracy"] for e in last["test_edges"] if not composable(*e["edge"], train_edges)]
            rows.append(dict(
                cfg=cfg, step=last["step"],
                train_acc=float(np.mean([e["accuracy"] for e in last["train_edges"]])),
                test_acc=float(np.mean([e["accuracy"] for e in last["test_edges"]])),
                test_cmp=float(np.mean(te_cmp)) if te_cmp else float("nan"),
                test_nc=float(np.mean(te_nc)) if te_nc else float("nan"),
                n_test=len(last["test_edges"]),
                path=str(cfg_dir),
            ))
    return rows


def main():
    if len(sys.argv) > 1:
        roots = [Path(p) for p in sys.argv[1:]]
    else:
        roots = [RUN_ROOT]
    all_rows = summarize(roots)
    all_rows.sort(key=lambda r: -r["test_acc"])
    print(f"{'config':<32}  {'step':>5}  {'TRAIN':>6}  {'TEST':>6}  {'cmp':>5}  {'nc':>5}")
    for r in all_rows:
        nc = "n/a" if np.isnan(r["test_nc"]) else f"{r['test_nc']:.2f}"
        print(f"{r['cfg']:<32}  {r['step']:>5}  {r['train_acc']:>6.2f}  {r['test_acc']:>6.2f}  {r['test_cmp']:>5.2f}  {nc:>5}")
    return all_rows


if __name__ == "__main__":
    main()
