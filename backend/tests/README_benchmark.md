# Recipe generation benchmark and quality harness

## Overview

- **Benchmark**: Time to generate recipes for 256 target colours (green-focused).
- **Quality baseline**: ΔE per target is stored; regression fails if:
  - mean ΔE increases by > 0.2,
  - max ΔE increases by > 0.5,
  - greens subset mean/max ΔE worsen (no tolerance).

## Commands (run from repo root)

**Option A – Use the helper script (recommended on macOS/Homebrew):**  
Creates/uses `.venv` and installs deps automatically.

```bash
./backend/run_benchmark.sh capture
./backend/run_benchmark.sh benchmark
./backend/run_benchmark.sh regression
./backend/run_benchmark.sh profile
```

**Option B – Manual venv:**  
Ensure backend dependencies are installed in a venv: `python3 -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt`

```bash
# After activating venv:
python -m backend.tests.benchmark_recipe_quality capture
python -m backend.tests.benchmark_recipe_quality benchmark
python -m backend.tests.benchmark_recipe_quality regression
python -m backend.tests.benchmark_recipe_quality profile
```

Use `--quick` for 16 targets (faster validation). Use `--library <group>` to choose library group.

Baseline is stored in `backend/tests/baseline_recipe_quality.json`. Run `capture` before `regression`; commit baseline when you want to lock quality for future changes.

## Expected time sinks (code inspection)

Before running the profiler, likely hot spots from the current implementation:

1. **Per-palette-color loop** in `generate_recipes_for_palette`: for each of 256 colours we run one- and multi-pigment search.
2. **`find_best_one_pigment_recipe`** called once per (colour, paint): grid over `test_ratio`, each step calls `interpolate_lab_from_calibration` and `delta_e_lab`.
3. **`find_best_multi_pigment_recipe`**: for each paint combination, `search()` does nested loops over ratio axes; each iteration calls `evaluate()` → `_predict_mix_lab_from_components` → `delta_e_lab`. Many ratio combinations per colour.
4. **`_predict_mix_lab_from_components`**: per call does `interpolate_lab_from_calibration` (or hex tint), `lab_to_rgb`, `_rgb_to_linear`, `_linear_to_rgb`, `rgb_to_lab`.
5. **`_load_calibration_cached`**: file read on first use per (paint_id, group); then cached. So first run per paint hits disk.
6. **`itertools.combinations`** for paint subsets: generates many 3- and 4-paint combinations; each triggers a full multi-pigment search.

Optimization order (per your rules): (a) caching/interpolation, (b) vectorize distance with numpy, (c) precompute candidate mixes, (d) optional coarse-to-fine only if quality gates pass.

## Profiling

`profile` uses cProfile and prints cumulative and total time by function. Focus on:

- `generate_recipes_for_palette` and callees
- `find_best_one_pigment_recipe` / `find_best_multi_pigment_recipe`
- `delta_e_lab`, `rgb_to_lab`, `_predict_mix_lab_from_components`
- File I/O (e.g. calibration load) and `itertools.combinations`

For a flame graph, run with pyinstrument (if installed):

```bash
pyinstrument -m backend.tests.benchmark_recipe_quality profile --quick
```

## Pytest

If baseline exists, you can run the regression as a test:

```bash
pytest backend/tests/benchmark_recipe_quality.py -v
```

(Skips if `baseline_recipe_quality.json` is missing.)
