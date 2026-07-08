"""
Configuration-sweep runner for the SHIFT interceptor.

Motivation
----------
When you run many engagements with varying configurations you do *not* want N
copies of a large monolithic sim-config.  This runner uses **delta storage**:

* one shared *base* master config (referenced once, content-hashed for
  provenance), and
* a small ``overrides`` dict per case (dotted ``fmu_id.param`` keys).

A case's full resolved configuration is therefore reproducible from
``base + delta`` at any time, so the on-disk cost of a 10 000-case campaign is a
single base file plus a compact override list, not 10 000 JSON blobs.

The sweep manifest (see ``config/shift_missile/sweep_intercept.json``) supplies:

* ``cases``  – an explicit list of ``{"id", "overrides"}`` deltas, and/or
* ``grid``   – a dict ``{dotted_param: [values...]}`` expanded into the Cartesian
               product and merged into the case list.

Outputs (under ``examples/runs/<sweep>_<timestamp>/``):

* ``manifest.json`` – base reference + base content hash + every case delta and
  its scalar result summary (the human-readable index of the campaign).
* ``results.jsonl`` – one JSON record per *run* (append-friendly, streaming;
  scales to millions of rows and is trivially loadable into pandas/DuckDB).

Usage::

    python examples/shift_missile/run_sweep.py \
        [config/shift_missile/sweep_intercept.json] [--montecarlo]

Note: cases are currently executed serially.  The results.jsonl / manifest
layout is designed so runs can be sharded across processes or hosts later
(see the scaling notes in docs/shift_missile.md).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compose_sim_config as composer  # noqa: E402
import engagement  # noqa: E402  (also puts the controller folder on sys.path)
import uncertainty as uq  # noqa: E402

RUNS_DIR = os.path.join(os.path.dirname(HERE), "runs")
DIVERGED_MISS_M = 1.0e5


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _expand_cases(sweep: dict) -> list[dict]:
    """Explicit cases + Cartesian product of ``grid`` -> deduplicated case list."""
    cases: list[dict] = [dict(c) for c in sweep.get("cases", [])]

    grid = sweep.get("grid", {})
    if grid:
        keys = list(grid.keys())
        for combo in itertools.product(*(grid[k] for k in keys)):
            overrides = dict(zip(keys, combo))
            cid = "grid__" + "__".join(
                f"{k.split('.')[-1]}={_slug(v)}" for k, v in overrides.items())
            cases.append({"id": cid, "overrides": overrides})

    # Deduplicate by resolved override signature (keep first id seen).
    seen: dict[str, dict] = {}
    for c in cases:
        sig = _sha256(c.get("overrides", {}))
        if sig not in seen:
            c.setdefault("overrides", {})
            seen[sig] = c
    return list(seen.values())


def _slug(v) -> str:
    return str(v).replace(".", "p").replace("-", "m").replace(" ", "")


def _run_case_single(base_params, overrides, sim_time_s, dt, seed) -> dict:
    res = engagement.run_engagement(
        overrides=overrides, params=base_params,
        sim_time_s=sim_time_s, dt=dt, seed=seed)
    miss = res["miss_distance_m"]
    if res["nonfinite"] or not np.isfinite(miss):
        miss = DIVERGED_MISS_M
    return {
        "miss_distance_m": float(miss),
        "intercept_time_s": float(res["intercept_time_s"]),
        "final_mach": float(res["final_mach"]),
        "min_margin_of_safety": float(res["final_margin_of_safety"]),
        "nonfinite": int(res["nonfinite"]),
    }


def _run_case_mc(base_params, case_overrides, mc, uncertain_spec,
                 sim_time_s, dt, seed0) -> dict:
    """Dispersed P_kill campaign for one case (case delta + per-run sample)."""
    lethal_r = float(mc.get("lethal_radius_m", 8.0))
    model = str(mc.get("damage_model", "carleton"))
    conf = float(mc.get("confidence", 0.95))
    n = int(mc.get("n_runs", 50))

    misses = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        ov = dict(case_overrides)
        ov.update(uq.sample_params(uncertain_spec, rng))
        res = engagement.run_engagement(overrides=ov, params=base_params,
                                        sim_time_s=sim_time_s, dt=dt, seed=seed0 + i)
        miss = res["miss_distance_m"]
        if res["nonfinite"] or not np.isfinite(miss):
            miss = DIVERGED_MISS_M
        misses.append(float(miss))

    misses = np.asarray(misses, dtype=float)
    pk = uq.sspk_monte_carlo(misses, lethal_r, model, conf)
    return {
        "n_runs": n,
        "pkill": pk["pkill"], "pkill_low": pk["low"], "pkill_high": pk["high"],
        "cep50_m": float(np.percentile(misses, 50)),
        "mean_miss_m": float(np.mean(misses)),
        "min_miss_m": float(np.min(misses)),
    }


def run_sweep(sweep_path: str, force_mc=None, verbose=True) -> str:
    with open(sweep_path, "r", encoding="utf-8") as fh:
        sweep = json.load(fh)

    base_master = os.path.join(os.path.dirname(sweep_path), sweep["base_master"])
    bundle = composer.load_scenario(base_master)
    base_params = composer._merged_params(bundle)
    base_hash = _sha256(base_params)

    mc_cfg = sweep.get("montecarlo", {})
    do_mc = mc_cfg.get("enabled", False) if force_mc is None else force_mc
    uncertain_spec = (bundle.get("monte_carlo") or {}).get("uncertain_params", {})

    sim_time_s = float(sweep.get("sim_time_s", 40.0))
    dt = float(sweep.get("dt_s", 0.01))
    seed0 = int(sweep.get("seed", 2024))

    cases = _expand_cases(sweep)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(RUNS_DIR, f"{sweep.get('name', 'sweep')}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    if verbose:
        print(f"[sweep] {sweep.get('name')}  base={os.path.basename(base_master)} "
              f"(sha {base_hash[:12]})")
        print(f"  {len(cases)} cases  mode={'montecarlo' if do_mc else 'single'}  "
              f"-> {out_dir}\n")

    t0 = time.time()
    manifest_cases = []
    with open(results_path, "w", encoding="utf-8") as jl:
        for idx, case in enumerate(cases):
            cid = case["id"]
            overrides = case.get("overrides", {})
            if do_mc:
                summary = _run_case_mc(base_params, overrides, mc_cfg,
                                       uncertain_spec, sim_time_s, dt, seed0)
                head = f"P_kill={summary['pkill']:.3f} CEP={summary['cep50_m']:.0f}m"
            else:
                summary = _run_case_single(base_params, overrides, sim_time_s, dt, seed0)
                head = f"miss={summary['miss_distance_m']:.0f}m mach={summary['final_mach']:.2f}"
            rec = {"case_id": cid, "case_index": idx, "overrides": overrides, **summary}
            jl.write(json.dumps(rec) + "\n")
            jl.flush()
            manifest_cases.append({"id": cid, "overrides": overrides, "summary": summary})
            if verbose:
                print(f"  [{idx + 1:3d}/{len(cases)}] {cid:42s} {head}")

    manifest = {
        "name": sweep.get("name"),
        "created_utc": stamp,
        "base_master": os.path.relpath(base_master, os.path.dirname(sweep_path)),
        "base_params_sha256": base_hash,
        "mode": "montecarlo" if do_mc else "single",
        "sim_time_s": sim_time_s,
        "dt_s": dt,
        "wall_time_s": round(time.time() - t0, 2),
        "n_cases": len(cases),
        "cases": manifest_cases,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if verbose:
        print(f"\n  wrote {len(cases)} cases in {manifest['wall_time_s']}s")
        print(f"  manifest -> {manifest_path}")
        print(f"  records  -> {results_path}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="SHIFT interceptor configuration sweep")
    ap.add_argument("sweep", nargs="?",
                    default=os.path.join(composer.MODULAR_DIR, "sweep_intercept.json"),
                    help="sweep manifest JSON")
    ap.add_argument("--montecarlo", action="store_true",
                    help="force a dispersed P_kill campaign per case")
    args = ap.parse_args(argv)
    sweep = args.sweep if os.path.isabs(args.sweep) else os.path.join(os.getcwd(), args.sweep)
    if not os.path.exists(sweep):
        sweep = os.path.join(composer.MODULAR_DIR, os.path.basename(args.sweep))
    run_sweep(sweep, force_mc=True if args.montecarlo else None)


if __name__ == "__main__":
    main()
