# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import design_3d
from data_audit import audit_dataframe

from data_to_excel import (
    assess_candidate,
    build_output_tables,
    classify_water_role,
    extract_parameters_form_text,
    normalize_unit,
    preprocess_value,
)
from generator_3d import OpenSCADRenderError, find_openscad, generate_openscad_script, render_openscad
from model_training_saving_v2 import (
    evaluate_imputer,
    load_dataset,
    run_supervised_training_v2,
    train_imputer,
)
from parameter_schema import DESIGN_FIELDS, apply_design_defaults, validate_parameters


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.frame = pd.DataFrame(
            {
                "长": rng.normal(10, 1, 24),
                "宽": rng.normal(5, 0.4, 24),
                "高": rng.normal(1.5, 0.1, 24),
                "进水量": rng.normal(100, 8, 24),
                "水力停留时间": rng.normal(24, 2, 24),
                "水力负荷": rng.normal(0.3, 0.03, 24),
            }
        )
        cls.frame.loc[[2, 5, 9], "宽"] = np.nan
        cls.imputer, cls.features = train_imputer(cls.frame, n_estimators=4, max_iter=2)

    def test_value_and_unit_normalization(self):
        self.assertEqual(preprocess_value("1~3"), 2.0)
        self.assertEqual(preprocess_value("-5"), -5.0)
        self.assertEqual(normalize_unit(1000, "ug/L"), (1.0, "mg/L"))
        self.assertEqual(normalize_unit(2, "d"), (48.0, "h"))
        self.assertEqual(normalize_unit(2, "m³/h"), (48.0, "m³/d"))
        self.assertEqual(classify_water_role("Influent COD was 200 mg/L"), "influent")

    def test_extraction_keeps_source_context(self):
        records = extract_parameters_form_text(
            "paper.pdf", [{"page": 3, "text": "Influent COD was 200 mg/L and effluent COD was 30 mg/L."}]
        )
        self.assertTrue(records)
        self.assertTrue(all(item["review_status"] == "pending" for item in records))
        self.assertTrue(all(item["record_id"].startswith("paper.pdf:p3:") for item in records))
        cod_records = [item for item in records if item["parameter"] == "COD"]
        self.assertEqual([item["water_role"] for item in cod_records], ["influent", "effluent"])
        self.assertTrue(all(item["quality_status"] == "accepted_candidate" for item in cod_records))

    def test_low_quality_candidates_are_flagged(self):
        score, status, reason = assess_candidate("COD", 2020, "", 5, "COD study published in 2020")
        self.assertEqual(status, "rejected")
        self.assertIn("年份", reason)
        _, status, reason = assess_candidate("水力停留时间", 5, "mg/L", 4, "HRT 5 mg/L")
        self.assertEqual(status, "rejected")
        self.assertIn("单位不相容", reason)

    def test_training_table_excludes_ambiguous_and_rejected_records(self):
        records = extract_parameters_form_text(
            "paper.pdf",
            [{"page": 1, "text": "COD was 40 mg/L. Influent TN was 20 mg/L. HRT was 5 mg/L."}],
        )
        detail, _, training, review, quality = build_output_tables(records)
        self.assertIn("总氮__influent", training.columns)
        self.assertNotIn("COD", training.columns)
        self.assertGreaterEqual(len(review), 2)
        self.assertEqual(len(detail), int(quality["记录数"].sum()))

    def test_complex_hydraulic_load_unit_is_matched(self):
        records = extract_parameters_form_text(
            "paper.pdf", [{"page": 1, "text": "The hydraulic loading rate was 0.35 m³/(m²·d)."}]
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["parameter"], "水力负荷")
        self.assertEqual(records[0]["unit"], "m³/(m²·d)")
        self.assertEqual(records[0]["quality_status"], "accepted_candidate")

    def test_dataset_quality_and_review_filters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "review.csv"
            pd.DataFrame(
                [
                    {"x": 1, "y": 2, "quality_status": "accepted_candidate", "review_status": "approved"},
                    {"x": 3, "y": 4, "quality_status": "accepted_candidate", "review_status": "pending"},
                    {"x": 5, "y": 6, "quality_status": "rejected", "review_status": "approved"},
                ]
            ).to_csv(csv_path, index=False)
            filtered = load_dataset(str(csv_path), require_reviewed=True)
            self.assertEqual(len(filtered), 1)

    def test_supervised_training_uses_group_validation_and_report(self):
        rng = np.random.default_rng(11)
        rows = 30
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            frame = pd.DataFrame(
                {
                    "pdf_name": ["study_{:02d}.pdf".format(index // 3) for index in range(rows)],
                    "x1": rng.normal(size=rows),
                    "x2": rng.normal(size=rows),
                }
            )
            frame["target"] = 2 * frame["x1"] - 0.5 * frame["x2"]
            csv_path = temp / "grouped.csv"
            model_path = temp / "grouped.joblib"
            report_path = temp / "report.json"
            frame.to_csv(csv_path, index=False)
            result = run_supervised_training_v2(
                str(csv_path),
                "target",
                str(model_path),
                n_splits=5,
                model_type="random_forest",
                report_path=str(report_path),
            )
            self.assertEqual(result["validation_strategy"], "group_kfold")
            self.assertEqual(result["source_column"], "pdf_name")
            self.assertIsNotNone(result["group_holdout"])
            self.assertFalse(
                set(result["group_holdout"]["train_sources"])
                & set(result["group_holdout"]["test_sources"])
            )
            self.assertIn("model_improvement_ratio", result["baseline"])
            self.assertTrue(report_path.exists())
            saved = joblib.load(model_path)
            self.assertEqual(len(saved["data_sha256"]), 64)

    def test_data_audit_reports_source_and_duplicates(self):
        frame = pd.DataFrame(
            [
                {"pdf_name": "a.pdf", "x": 1.0, "constant": 5.0},
                {"pdf_name": "a.pdf", "x": 1.0, "constant": 5.0},
                {"pdf_name": "b.pdf", "x": None, "constant": 5.0},
            ]
        )
        report = audit_dataframe(frame)
        self.assertEqual(report["source_column"], "pdf_name")
        self.assertEqual(report["source_count"], 2)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertIn("constant", report["constant_numeric_columns"])

    def test_schema_defaults_and_constraints(self):
        params = apply_design_defaults({"长": 10, "宽": 5, "高": 1.5})
        converted, errors = validate_parameters(params, DESIGN_FIELDS)
        self.assertFalse(errors)
        self.assertEqual(converted["flow_type"], "horizontal")
        _, errors = validate_parameters({**params, "water_depth": 2}, DESIGN_FIELDS)
        self.assertTrue(any("water_depth" in error for error in errors))

    def test_imputer_evaluation_masks_subset(self):
        metrics = evaluate_imputer(self.imputer, self.frame, mask_ratio=0.2)
        self.assertIn("宽", metrics)
        self.assertGreaterEqual(metrics["宽"]["MAE"]["value"], 0)

    def test_end_to_end_2d_and_scad(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            model_path = temp / "model.joblib"
            joblib.dump(
                {
                    "schema_version": 1,
                    "type": "imputer",
                    "imputer": self.imputer,
                    "feature_names": self.features,
                    "input_features": self.features,
                },
                model_path,
            )
            params = {name: None for name in self.features}
            params.update({"长": 10, "宽": 5, "高": 1.5})
            result = design_3d.generate_from_params(params, str(model_path), str(temp / "wetland"), mode="2d")
            self.assertTrue(Path(result["2d_image"]).exists())
            completed = result["parameters"]
            scad = temp / "wetland.scad"
            generate_openscad_script(completed, str(scad))
            content = scad.read_text(encoding="utf-8")
            self.assertIn("floor_thickness", content)
            self.assertIn("过水口", content)

    def test_openscad_path_and_render_output_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            executable = temp / "openscad.exe"
            executable.write_bytes(b"fake")
            scad = temp / "model.scad"
            scad.write_text("cube([1,1,1]);", encoding="utf-8")
            output = temp / "model.stl"
            output.write_text("stale output", encoding="utf-8")
            self.assertEqual(find_openscad(str(executable)), str(executable.resolve()))

            def successful_run(*args, **kwargs):
                output.write_text("solid model\nendsolid model", encoding="utf-8")
                return mock.Mock(returncode=0, stdout="ok", stderr="")

            with mock.patch("generator_3d.subprocess.run", side_effect=successful_run):
                result = render_openscad(str(scad), output_stl=str(output), openscad_path=str(executable))
            self.assertGreater(output.stat().st_size, 0)
            self.assertTrue(Path(result["logs"][0]).exists())

            output.unlink()
            with mock.patch(
                "generator_3d.subprocess.run", return_value=mock.Mock(returncode=0, stdout="", stderr="")
            ):
                with self.assertRaises(OpenSCADRenderError):
                    render_openscad(str(scad), output_stl=str(output), openscad_path=str(executable))
            self.assertFalse(output.exists())

    def test_compute_parameters_without_model_uses_validated_defaults(self):
        import api

        result = api.compute_parameters({"长": 10, "宽": 5, "高": 1.5})
        self.assertEqual(result["flow_type"], "horizontal")
        self.assertEqual(result["compartments"], 1)

    def test_design_without_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = design_3d.generate_from_params(
                {"长": 10, "宽": 5, "高": 1.5}, None, str(Path(temp_dir) / "plain"), mode="2d"
            )
            self.assertTrue(Path(result["2d_image"]).exists())


if __name__ == "__main__":
    unittest.main()
