"""
Monte-Carlo P_kill campaign for the SHIFT interceptor.

Draws the uncertain parameters declared in the Monte-Carlo config from their
distributions, runs one closed-loop engagement per sample with the standalone
``engagement.run_engagement`` engine (faster than real time, no Kafka /
orchestrator), collects the miss-distance distribution and reports the
single-shot probability of kill (P_kill / SSPK) with a confidence interval.

Two lethality models are reported (see ``uncertainty.py``):

* ``carleton`` – diffuse-Gaussian warhead damage function; each run contributes
  a continuous conditional-kill probability and P_kill is their mean (CLT
  interval).  Also reported is the closed-form Rayleigh-miss/Carleton SSPK from
  the fitted miss statistics as a cross-check.
* ``cookie``   – cookie-cutter (hit iff miss <= lethal radius); each run is a
  Bernoulli trial and the Wilson score interval is reported.

Usage::

    python examples/shift_missile/run_monte_carlo.py \
        [config/shift_missile/master_intercept.json] [--runs N]

Note: runs are currently executed serially. See the scaling notes in
docs/shift_missile.md for parallelizing across cores/hosts.

Results are printed and written next to the config as ``<name>.results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compose_sim_config as composer  # noqa: E402
import engagement  # noqa: E402  (also puts the controller folder on sys.path)
import uncertainty as uq  # noqa: E402  (from shift_missile_controller_fmus)

# Diverged / non-finite runs are scored as clean misses (kill prob 0) rather
# than being silently dropped, so instability shows up as reduced P_kill.
DIVERGED_MISS_M = 1.0e5


def _one_run(base_params, uncertain_spec, seed, sim_time_s, dt):
    """Run a single dispersed engagement; return (miss_distance_m, record)."""
    rng = np.random.default_rng(seed)
    overrides = uq.sample_params(uncertain_spec, rng)
    res = engagement.run_engagement(
        overrides=overrides,
        params=base_params,
        sim_time_s=sim_time_s,
        dt=dt,
        seed=seed,
    )
    miss = res["miss_distance_m"]
    if res["nonfinite"] or not np.isfinite(miss):
        miss = DIVERGED_MISS_M
    record = {
        "seed": seed,
        "miss_distance_m": float(miss),
        "intercept_time_s": float(res["intercept_time_s"]),
        "final_mach": float(res["final_mach"]),
        "min_margin_of_safety": float(res["final_margin_of_safety"]),
        "nonfinite": int(res["nonfinite"]),
        "overrides": overrides,
    }
    return float(miss), record


def run_campaign(master_path: str, n_runs=None, verbose=True) -> dict:
    bundle = composer.load_scenario(master_path)
    mc = bundle.get("monte_carlo")
    if mc is None:
        raise ValueError(f"master config {master_path!r} has no monte_carlo section")

    base_params = composer._merged_params(bundle)
    uncertain_spec = mc.get("uncertain_params", {})
    n = int(n_runs if n_runs is not None else mc.get("n_runs", 100))
    seed0 = int(mc.get("seed", 12345))
    sim_time_s = float(mc.get("sim_time_s", 40.0))
    dt = float(mc.get("dt_s", 0.01))
    lethal_r = float(mc.get("lethal_radius_m", 8.0))
    model = str(mc.get("damage_model", "carleton"))
    conf = float(mc.get("confidence", 0.95))

    if verbose:
        print(f"[monte-carlo] {mc.get('name', master_path)}")
        print(f"  runs={n}  lethal_radius={lethal_r} m  damage_model={model}  "
              f"confidence={conf:.0%}")
        print(f"  dispersing {len(uncertain_spec)} parameters over {sim_time_s:.0f}s "
              f"engagements @ dt={dt*1000:.0f}ms\n")

    misses = []
    records = []
    t_start = time.time()
    for i in range(n):
        miss, rec = _one_run(base_params, uncertain_spec, seed0 + i, sim_time_s, dt)
        misses.append(miss)
        records.append(rec)
        if verbose and ((i + 1) % max(1, n // 20) == 0 or i == n - 1):
            done = i + 1
            rate = done / max(1e-9, time.time() - t_start)
            hits = int(np.sum(np.asarray(misses) <= lethal_r))
            print(f"  {done:4d}/{n}  miss={miss:8.1f} m  "
                  f"running Phit(cookie)={hits/done:5.1%}  ({rate:4.1f} runs/s)")

    misses = np.asarray(misses, dtype=float)

    carleton = uq.sspk_monte_carlo(misses, lethal_r, "carleton", conf)
    cookie = uq.sspk_monte_carlo(misses, lethal_r, "cookie", conf)
    closed_form = uq.sspk_rayleigh_carleton(float(np.std(misses)), lethal_r)

    summary = {
        "master": os.path.basename(master_path),
        "n_runs": int(n),
        "lethal_radius_m": lethal_r,
        "confidence": conf,
        "wall_time_s": round(time.time() - t_start, 2),
        "miss_stats_m": {
            "min": float(np.min(misses)),
            "cep50": float(np.percentile(misses, 50)),
            "mean": float(np.mean(misses)),
            "p90": float(np.percentile(misses, 90)),
            "max": float(np.max(misses)),
            "std": float(np.std(misses)),
        },
        "pkill_carleton": carleton,
        "pkill_cookie": cookie,
        "pkill_closed_form_rayleigh_carleton": closed_form,
    }

    if verbose:
        s = summary["miss_stats_m"]
        print("\n" + "=" * 64)
        print(f"  RESULTS  ({summary['n_runs']} runs, {summary['wall_time_s']} s)")
        print("=" * 64)
        print(f"  miss distance   min={s['min']:.1f}  CEP={s['cep50']:.1f}  "
              f"mean={s['mean']:.1f}  p90={s['p90']:.1f}  max={s['max']:.1f} m")
        c = carleton
        print(f"  P_kill (Carleton, r_L={lethal_r:.0f} m) = {c['pkill']:.3f}  "
              f"[{c['low']:.3f}, {c['high']:.3f}]  ({conf:.0%} CI)")
        k = cookie
        print(f"  P_kill (cookie-cutter)               = {k['pkill']:.3f}  "
              f"[{k['low']:.3f}, {k['high']:.3f}]  (Wilson {conf:.0%})")
        print(f"  P_kill (closed-form Rayleigh/Carleton)= {closed_form:.3f}")
        print("=" * 64)

    return {"summary": summary, "records": records}


def main(argv=None):
    ap = argparse.ArgumentParser(description="SHIFT interceptor P_kill Monte-Carlo")
    ap.add_argument("master", nargs="?",
                    default=os.path.join(composer.MODULAR_DIR, "master_intercept.json"),
                    help="master config JSON (references scenario + monte_carlo)")
    ap.add_argument("--runs", type=int, default=None, help="override n_runs")
    ap.add_argument("--out", default=None, help="results JSON output path")
    args = ap.parse_args(argv)

    master = args.master if os.path.isabs(args.master) else os.path.join(os.getcwd(), args.master)
    if not os.path.exists(master):
        master = os.path.join(composer.MODULAR_DIR, os.path.basename(args.master))

    out = run_campaign(master, n_runs=args.runs)

    out_path = args.out or os.path.join(
        composer.CONFIG_DIR, os.path.splitext(os.path.basename(master))[0] + ".results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out["summary"], fh, indent=4)
    print(f"\n  wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
