import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import paint_manager


def _normalize_hex(hex_color: str) -> str:
    h = (hex_color or "").upper().lstrip("#")
    return f"#{h}"


class RecipeGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_percentages_and_grams_sum_to_targets(self):
        paints = [
            {"id": "white", "name": "Titanium White", "type": "base", "hex_approx": "#FFFFFF", "notes": ""},
            {"id": "red", "name": "Naphthol Red", "type": "base", "hex_approx": "#FF0000", "notes": ""},
        ]
        library = {"paints": paints, "coverage_mg_per_cm2": 0.008}
        solver_output = [
            {
                "palette_index": 0,
                "type": "one_pigment",
                "recipe": {
                    "white_ratio": 0.3333,
                    "pigment_id": "red",
                    "pigment_ratio": 0.6667,
                    "error": 1.1,
                },
            }
        ]
        cache_store = {}

        def fake_get_cached(group: str, hex_color: str):
            return cache_store.get((group, _normalize_hex(hex_color)))

        def fake_cache(group: str, hex_color: str, recipe: dict):
            cache_store[(group, _normalize_hex(hex_color))] = recipe

        palette = json.dumps([{"index": 0, "hex": "#AA1122", "target_grams": 10.0}])
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = Path(tmp)
            with patch.object(main, "load_library", return_value=library), \
                 patch.object(main, "generate_recipes_for_palette", return_value=solver_output), \
                 patch.object(main, "get_cached_recipe", side_effect=fake_get_cached), \
                 patch.object(main, "cache_recipe", side_effect=fake_cache), \
                 patch.object(main, "CALIBRATION_DIR", cal_dir):
                response = await main.generate_recipes_from_palette(
                    palette=palette,
                    library_group="default",
                    force_regenerate="true",
                )

        rec = response["recipes"][0]["recipe"]
        ingredients = rec["ingredients"]
        pct_sum = round(sum(float(i["percentage"]) for i in ingredients), 2)
        grams_sum = round(sum(float(i["grams"]) for i in ingredients), 2)
        self.assertEqual(pct_sum, 100.00)
        self.assertEqual(grams_sum, 10.00)
        self.assertEqual(response["recipes"][0]["type"], "deterministic")

    async def test_cache_fingerprint_invalidates_when_calibration_changes(self):
        paints = [
            {"id": "white", "name": "White", "type": "base", "hex_approx": "#FFFFFF", "notes": ""},
            {"id": "red", "name": "Red", "type": "base", "hex_approx": "#FF0000", "notes": ""},
        ]
        library = {"paints": paints, "coverage_mg_per_cm2": 0.008}
        cache_store = {}
        solver_calls = {"count": 0}

        def fake_solver(_session_id, palette_list, _group):
            solver_calls["count"] += 1
            idx = int(palette_list[0]["index"])
            return [{
                "palette_index": idx,
                "type": "one_pigment",
                "recipe": {
                    "white_ratio": 0.5,
                    "pigment_id": "red",
                    "pigment_ratio": 0.5,
                    "error": 1.5,
                },
            }]

        def fake_get_cached(group: str, hex_color: str):
            return cache_store.get((group, _normalize_hex(hex_color)))

        def fake_cache(group: str, hex_color: str, recipe: dict):
            cache_store[(group, _normalize_hex(hex_color))] = recipe

        palette = json.dumps([{"index": 0, "hex": "#AA1122", "target_grams": 6.0}])
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = Path(tmp)
            cal_dir.mkdir(parents=True, exist_ok=True)
            red_cal = cal_dir / "red.json"
            red_cal.write_text('{"samples":[{"ratio":1,"rgb":[255,0,0],"lab":[50,80,60]}]}')

            with patch.object(main, "load_library", return_value=library), \
                 patch.object(main, "generate_recipes_for_palette", side_effect=fake_solver), \
                 patch.object(main, "get_cached_recipe", side_effect=fake_get_cached), \
                 patch.object(main, "cache_recipe", side_effect=fake_cache), \
                 patch.object(main, "CALIBRATION_DIR", cal_dir):
                await main.generate_recipes_from_palette(
                    palette=palette,
                    library_group="default",
                    force_regenerate="false",
                )
                self.assertEqual(solver_calls["count"], 1)

                await main.generate_recipes_from_palette(
                    palette=palette,
                    library_group="default",
                    force_regenerate="false",
                )
                self.assertEqual(solver_calls["count"], 1)

                red_cal.write_text('{"samples":[{"ratio":1,"rgb":[254,0,0],"lab":[50,79,60]}]}')

                await main.generate_recipes_from_palette(
                    palette=palette,
                    library_group="default",
                    force_regenerate="false",
                )
                self.assertEqual(solver_calls["count"], 2)

    async def test_structured_recipe_merges_duplicate_components(self):
        paints = [
            {"id": "white", "name": "Titanium White", "type": "base", "hex_approx": "#FFFFFF", "notes": ""},
            {"id": "blue", "name": "Cool Blue", "type": "base", "hex_approx": "#2040C8", "notes": ""},
        ]
        library = {"paints": paints, "coverage_mg_per_cm2": 0.008}
        solver_output = [
            {
                "palette_index": 0,
                "type": "three_pigment",
                "recipe": {
                    "white_ratio": 0.4,
                    "pigment_ids": ["white", "blue", "blue"],
                    "pigment_ratios": [0.1, 0.3, 0.2],
                    "error": 2.0,
                },
            }
        ]
        cache_store = {}

        def fake_get_cached(group: str, hex_color: str):
            return cache_store.get((group, _normalize_hex(hex_color)))

        def fake_cache(group: str, hex_color: str, recipe: dict):
            cache_store[(group, _normalize_hex(hex_color))] = recipe

        palette = json.dumps([{"index": 0, "hex": "#4A6A9A", "target_grams": 9.0}])
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = Path(tmp)
            with patch.object(main, "load_library", return_value=library), \
                 patch.object(main, "generate_recipes_for_palette", return_value=solver_output), \
                 patch.object(main, "get_cached_recipe", side_effect=fake_get_cached), \
                 patch.object(main, "cache_recipe", side_effect=fake_cache), \
                 patch.object(main, "CALIBRATION_DIR", cal_dir):
                response = await main.generate_recipes_from_palette(
                    palette=palette,
                    library_group="default",
                    force_regenerate="true",
                )

        ingredients = response["recipes"][0]["recipe"]["ingredients"]
        self.assertEqual(len(ingredients), 2)
        ids = [i["paint_id"] for i in ingredients]
        self.assertEqual(sorted(ids), ["blue", "white"])
        pct_sum = round(sum(float(i["percentage"]) for i in ingredients), 2)
        grams_sum = round(sum(float(i["grams"]) for i in ingredients), 2)
        self.assertEqual(pct_sum, 100.00)
        self.assertEqual(grams_sum, 9.00)

    async def test_integration_real_endpoint_with_calibrated_library_no_fallback(self):
        group = f"integration_no_fallback_{uuid.uuid4().hex[:8]}"
        library_file = paint_manager.LIBRARIES_DIR / f"{group}.json"
        cache_file = paint_manager.RECIPES_CACHE_DIR / f"{group}_recipes.json"

        paints = [
            {"id": f"{group}_white", "name": "White", "type": "base", "hex_approx": "#FFFFFF", "notes": ""},
            {"id": f"{group}_red", "name": "Red", "type": "base", "hex_approx": "#C82020", "notes": ""},
            {"id": f"{group}_blue", "name": "Blue", "type": "base", "hex_approx": "#2040C8", "notes": ""},
            {"id": f"{group}_yellow", "name": "Yellow", "type": "base", "hex_approx": "#D8C020", "notes": ""},
        ]
        library_data = {
            "version": 1,
            "group": group,
            "name": "Integration No Fallback",
            "coverage_mg_per_cm2": 0.008,
            "paints": paints,
        }

        calibration_payloads = {
            f"{group}_white": {
                "paint_id": f"{group}_white",
                "ratios": [1.0, 0.5, 0.25, 0.125],
                "samples": [
                    {"ratio": 1.0, "rgb": [255, 255, 255], "lab": [100.0, 0.0, 0.0]},
                    {"ratio": 0.5, "rgb": [250, 250, 250], "lab": [97.5, 0.0, 0.0]},
                    {"ratio": 0.25, "rgb": [246, 246, 246], "lab": [95.0, 0.0, 0.0]},
                    {"ratio": 0.125, "rgb": [242, 242, 242], "lab": [92.5, 0.0, 0.0]},
                ],
            },
            f"{group}_red": {
                "paint_id": f"{group}_red",
                "ratios": [1.0, 0.5, 0.25, 0.125],
                "samples": [
                    {"ratio": 1.0, "rgb": [200, 32, 32], "lab": [50.0, 65.0, 45.0]},
                    {"ratio": 0.5, "rgb": [228, 120, 120], "lab": [67.0, 48.0, 28.0]},
                    {"ratio": 0.25, "rgb": [240, 176, 176], "lab": [79.0, 31.0, 16.0]},
                    {"ratio": 0.125, "rgb": [247, 214, 214], "lab": [88.0, 18.0, 8.0]},
                ],
            },
            f"{group}_blue": {
                "paint_id": f"{group}_blue",
                "ratios": [1.0, 0.5, 0.25, 0.125],
                "samples": [
                    {"ratio": 1.0, "rgb": [32, 64, 200], "lab": [42.0, 30.0, -62.0]},
                    {"ratio": 0.5, "rgb": [120, 140, 228], "lab": [60.0, 18.0, -42.0]},
                    {"ratio": 0.25, "rgb": [176, 186, 240], "lab": [73.0, 10.0, -25.0]},
                    {"ratio": 0.125, "rgb": [214, 220, 247], "lab": [84.0, 5.0, -12.0]},
                ],
            },
            f"{group}_yellow": {
                "paint_id": f"{group}_yellow",
                "ratios": [1.0, 0.5, 0.25, 0.125],
                "samples": [
                    {"ratio": 1.0, "rgb": [216, 192, 32], "lab": [77.0, -6.0, 66.0]},
                    {"ratio": 0.5, "rgb": [234, 218, 112], "lab": [86.0, -7.0, 48.0]},
                    {"ratio": 0.25, "rgb": [242, 232, 164], "lab": [91.0, -6.0, 34.0]},
                    {"ratio": 0.125, "rgb": [248, 242, 204], "lab": [95.0, -4.0, 20.0]},
                ],
            },
        }

        calibration_files = []
        try:
            paint_manager.atomic_write(library_file, library_data)
            for paint_id, payload in calibration_payloads.items():
                cal_path = paint_manager.CALIBRATION_DIR / f"{paint_id}.json"
                paint_manager.atomic_write(cal_path, payload)
                calibration_files.append(cal_path)

            palette = json.dumps([
                {"index": 0, "hex": "#7A6AA7", "target_grams": 12.0},
                {"index": 1, "hex": "#C88A4A", "target_grams": 8.5},
            ])
            response = await main.generate_recipes_from_palette(
                palette=palette,
                library_group=group,
                force_regenerate="true",
            )
            recipes = response.get("recipes", [])
            self.assertEqual(len(recipes), 2)
            for item, target_grams in zip(recipes, [12.0, 8.5]):
                self.assertEqual(item.get("type"), "deterministic")
                self.assertIsNotNone(item.get("recipe"))
                recipe = item["recipe"]
                self.assertFalse(bool(recipe.get("uncalibrated", True)))
                self.assertIn("ingredients", recipe)
                ingredients = recipe["ingredients"]
                self.assertGreater(len(ingredients), 0)
                self.assertTrue(all("grams" in ing for ing in ingredients))
                pct_sum = round(sum(float(i.get("percentage", 0.0)) for i in ingredients), 2)
                grams_sum = round(sum(float(i.get("grams", 0.0)) for i in ingredients), 2)
                self.assertEqual(pct_sum, 100.00)
                self.assertEqual(grams_sum, round(target_grams, 2))
                self.assertTrue(all(float(i.get("percentage", 0.0)) > 0 for i in ingredients))
                self.assertEqual(len({str(i.get("paint_id")) for i in ingredients}), len(ingredients))
                self.assertIsNotNone(recipe.get("error"))
                self.assertLess(float(recipe.get("error")), 80.0)
        finally:
            if cache_file.exists():
                cache_file.unlink()
            if library_file.exists():
                library_file.unlink()
            for cal_path in calibration_files:
                if cal_path.exists():
                    cal_path.unlink()


if __name__ == "__main__":
    unittest.main()
