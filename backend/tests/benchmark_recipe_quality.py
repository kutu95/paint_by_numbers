"""
Benchmark and quality regression harness for recipe generation.

- Benchmark: time to generate recipes for 256 target colours (green-focused).
- Quality baseline: store ΔE per target; regression fails if mean ΔE increases > 0.2,
  max ΔE increases > 0.5, or greens subset worsens.

Run from repo root (with backend deps installed: pip install -r backend/requirements.txt):
  python -m backend.tests.benchmark_recipe_quality capture   # capture baseline
  python -m backend.tests.benchmark_recipe_quality benchmark # run benchmark only (timing)
  python -m backend.tests.benchmark_recipe_quality regression # run and check vs baseline
  python -m backend.tests.benchmark_recipe_quality profile  # cProfile + top time sinks (optional --quick)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Run from repo root or backend/
_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

import paint_manager

BASELINE_PATH = Path(__file__).resolve().parent / "baseline_recipe_quality.json"
LIBRARY_GROUP = "default"

# Quality gates (regression)
MAX_MEAN_DELTA_E_INCREASE = 0.2
MAX_SINGLE_DELTA_E_INCREASE = 0.5
GREENS_MUST_NOT_WORSEN = True  # green mean/max must be <= baseline (no tolerance)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
    )


def _is_green_rgb(r: int, g: int, b: int) -> bool:
    """True if green is dominant (G >= R and G >= B and not achromatic)."""
    if g < 10 and r < 10 and b < 10:
        return False
    return g >= r and g >= b


def generate_benchmark_targets(n: int = 256, green_focus: bool = True) -> list[dict]:
    """Build 256 target colours (green-focused). Each has index, hex, rgb, is_green."""
    import random
    rng = random.Random(42)
    targets = []
    n_greens = 200 if green_focus else 0
    # Greens: G >= R, G >= B. Sample deterministically.
    seen = set()
    while len(targets) < n_greens:
        g = rng.randint(40, 255)
        r = rng.randint(0, g)
        b = rng.randint(0, g)
        key = (r, g, b)
        if key in seen:
            continue
        seen.add(key)
        targets.append({
            "index": len(targets),
            "rgb": [r, g, b],
            "hex": _rgb_to_hex(r, g, b),
            "is_green": True,
        })
    # Remaining: mixed colours (including non-greens)
    while len(targets) < n:
        r = rng.randint(0, 255)
        g = rng.randint(0, 255)
        b = rng.randint(0, 255)
        targets.append({
            "index": len(targets),
            "rgb": [r, g, b],
            "hex": _rgb_to_hex(r, g, b),
            "is_green": _is_green_rgb(r, g, b),
        })
    return targets


def palette_from_targets(targets: list[dict]) -> list[dict]:
    """Convert to palette format expected by generate_recipes_for_palette (index, rgb)."""
    return [{"index": t["index"], "rgb": t["rgb"], "hex": t["hex"]} for t in targets]


def extract_delta_e(recipes: list[dict], targets: list[dict]) -> tuple[list[float | None], list[bool]]:
    """Extract ΔE per target from solver output. Returns (delta_es, is_green)."""
    by_idx = {r["palette_index"]: r for r in recipes if r.get("palette_index") is not None}
    delta_es = []
    is_green = []
    for t in targets:
        i = t["index"]
        is_green.append(t.get("is_green", False))
        rec = by_idx.get(i)
        if rec and rec.get("recipe") and isinstance(rec["recipe"].get("error"), (int, float)):
            delta_es.append(float(rec["recipe"]["error"]))
        else:
            delta_es.append(None)  # failed or no recipe
    return delta_es, is_green


def compute_summary(delta_es: list[float | None], is_green: list[bool]) -> dict:
    """Mean/max ΔE overall and for greens subset. Exclude None (failed)."""
    valid = [e for e in delta_es if e is not None]
    green_es = [e for e, g in zip(delta_es, is_green) if g and e is not None]
    return {
        "n_total": len(delta_es),
        "n_failed": sum(1 for e in delta_es if e is None),
        "n_greens": sum(1 for g in is_green if g),
        "mean_delta_e": sum(valid) / len(valid) if valid else None,
        "max_delta_e": max(valid) if valid else None,
        "green_mean_delta_e": sum(green_es) / len(green_es) if green_es else None,
        "green_max_delta_e": max(green_es) if green_es else None,
    }


def run_benchmark(targets: list[dict], library_group: str = LIBRARY_GROUP) -> tuple[list[dict], float]:
    """Run solver for the given targets; return (recipes, duration_sec)."""
    palette = palette_from_targets(targets)
    start = time.perf_counter()
    recipes = paint_manager.generate_recipes_for_palette(
        "benchmark",
        palette,
        library_group=library_group,
        progress_cb=None,
    )
    duration = time.perf_counter() - start
    return recipes, duration


def capture_baseline(targets: list[dict] | None = None, library_group: str = LIBRARY_GROUP) -> dict:
    """Run solver, record ΔE per target and summary, save to baseline JSON."""
    if targets is None:
        targets = generate_benchmark_targets(256, green_focus=True)
    recipes, duration = run_benchmark(targets, library_group)
    delta_es, is_green = extract_delta_e(recipes, targets)
    summary = compute_summary(delta_es, is_green)
    baseline = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": round(duration, 2),
        "n_targets": len(targets),
        "targets": [
            {
                "index": t["index"],
                "hex": t["hex"],
                "rgb": t["rgb"],
                "is_green": t.get("is_green", False),
                "baseline_delta_e": delta_es[i],
            }
            for i, t in enumerate(targets)
        ],
        "summary": summary,
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline captured: {BASELINE_PATH}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Mean ΔE: {summary.get('mean_delta_e')}")
    print(f"  Max ΔE: {summary.get('max_delta_e')}")
    print(f"  Green mean ΔE: {summary.get('green_mean_delta_e')}")
    print(f"  Green max ΔE: {summary.get('green_max_delta_e')}")
    return baseline


def run_regression(library_group: str = LIBRARY_GROUP) -> bool:
    """Load baseline, run solver, check quality gates. Return True if pass."""
    if not BASELINE_PATH.exists():
        print("No baseline file found. Run 'capture' first.")
        return False
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    targets_data = baseline["targets"]
    # Rebuild target list (index, hex, rgb, is_green)
    targets = [
        {
            "index": t["index"],
            "hex": t["hex"],
            "rgb": t["rgb"],
            "is_green": t.get("is_green", False),
        }
        for t in targets_data
    ]
    base_summary = baseline["summary"]
    recipes, duration = run_benchmark(targets, library_group)
    delta_es, is_green = extract_delta_e(recipes, targets)
    new_summary = compute_summary(delta_es, is_green)

    # Compare
    def _f(name: str, old: dict, new: dict):
        o = old.get(name)
        n = new.get(name)
        if o is None and n is None:
            return True, "-"
        if o is None or n is None:
            return False, f"{o} -> {n}"
        return True, f"{o:.4f} -> {n:.4f}"

    passed = True
    # Mean ΔE: new <= baseline + 0.2
    b_mean = base_summary.get("mean_delta_e")
    n_mean = new_summary.get("mean_delta_e")
    if b_mean is not None and n_mean is not None:
        if n_mean > b_mean + MAX_MEAN_DELTA_E_INCREASE:
            print(f"FAIL: mean ΔE {n_mean:.4f} > baseline {b_mean:.4f} + {MAX_MEAN_DELTA_E_INCREASE}")
            passed = False
        else:
            print(f"OK mean ΔE: {n_mean:.4f} (baseline {b_mean:.4f})")
    # Max ΔE: new <= baseline + 0.5
    b_max = base_summary.get("max_delta_e")
    n_max = new_summary.get("max_delta_e")
    if b_max is not None and n_max is not None:
        if n_max > b_max + MAX_SINGLE_DELTA_E_INCREASE:
            print(f"FAIL: max ΔE {n_max:.4f} > baseline {b_max:.4f} + {MAX_SINGLE_DELTA_E_INCREASE}")
            passed = False
        else:
            print(f"OK max ΔE: {n_max:.4f} (baseline {b_max:.4f})")
    # Greens must not worsen
    if GREENS_MUST_NOT_WORSEN:
        bg_mean = base_summary.get("green_mean_delta_e")
        ng_mean = new_summary.get("green_mean_delta_e")
        bg_max = base_summary.get("green_max_delta_e")
        ng_max = new_summary.get("green_max_delta_e")
        if bg_mean is not None and ng_mean is not None and ng_mean > bg_mean:
            print(f"FAIL: green mean ΔE {ng_mean:.4f} > baseline {bg_mean:.4f}")
            passed = False
        elif bg_max is not None and ng_max is not None and ng_max > bg_max:
            print(f"FAIL: green max ΔE {ng_max:.4f} > baseline {bg_max:.4f}")
            passed = False
        else:
            print("OK greens: within baseline")
    print(f"Duration: {duration:.2f}s (baseline {baseline.get('duration_sec')}s)")
    return passed


def run_profile(targets: list[dict], library_group: str = LIBRARY_GROUP, top_n: int = 30) -> None:
    """Run solver under cProfile and print top time sinks."""
    import cProfile
    import pstats
    import io
    pr = cProfile.Profile()
    pr.enable()
    paint_manager.generate_recipes_for_palette(
        "profile",
        palette_from_targets(targets),
        library_group=library_group,
    )
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n)
    print(s.getvalue())
    # Also by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats(pstats.SortKey.TIME)
    ps2.print_stats(top_n)
    print("--- By total time ---")
    print(s2.getvalue())


def main():
    import argparse
    p = argparse.ArgumentParser(description="Recipe generation benchmark and quality harness")
    p.add_argument("command", choices=["capture", "benchmark", "regression", "profile"], help="capture | benchmark | regression | profile")
    p.add_argument("--library", default=LIBRARY_GROUP, help="Library group (default: default)")
    p.add_argument("--quick", action="store_true", help="Use 16 targets only (for quick validation)")
    p.add_argument("--top", type=int, default=30, help="Profile: show top N functions (default 30)")
    args = p.parse_args()
    n_targets = 16 if args.quick else 256
    targets = generate_benchmark_targets(n_targets, green_focus=True)
    if args.command == "capture":
        capture_baseline(targets, args.library)
    elif args.command == "benchmark":
        _, duration = run_benchmark(targets, args.library)
        print(f"Benchmark ({len(targets)} targets, green-focused): {duration:.2f}s")
    elif args.command == "regression":
        ok = run_regression(args.library)
        sys.exit(0 if ok else 1)
    elif args.command == "profile":
        run_profile(targets, args.library, top_n=args.top)


def test_regression_quality_gates():
    """Pytest: run regression check if baseline exists; skip otherwise."""
    if not BASELINE_PATH.exists():
        import pytest
        pytest.skip("No baseline_recipe_quality.json; run 'capture' first")
    assert run_regression(LIBRARY_GROUP), "Quality regression: ΔE exceeded thresholds or greens worsened"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. No baseline or results saved.")
        sys.exit(130)
