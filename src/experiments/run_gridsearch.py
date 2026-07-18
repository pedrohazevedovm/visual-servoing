import argparse
from datetime import datetime
import itertools
from pathlib import Path
import sys
from typing import Dict, Any, List
import yaml

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lightglue.utils import load_image

import src.steps
from src.core.pipeline import Pipeline
from src.core.registry import StepRegistry
from src.evaluation.metrics import compute_pipeline_metrics
from src.evaluation.reporter import Reporter


def build_pipeline_from_combo(combo: Dict[str, Any], defaults: Dict[str, Any]) -> Pipeline:
    """
    Constructs a Pipeline instance from a grid search combination dictionary.
    """
    steps = []

    # 1. Histogram Matching
    if combo.get("histogram_matching", False):
        steps.append(StepRegistry.create("histogram_matching", enabled=True))

    # 2. Bilateral Filter
    if combo.get("bilateral_filter", False):
        bf_params = defaults.get("bilateral_params", {})
        steps.append(StepRegistry.create("bilateral_filter", enabled=True, **bf_params))

    # 3. Edge Detection
    edge_method = combo.get("edge_detection_method", "none")
    if edge_method != "none":
        steps.append(StepRegistry.create("edge_detection", enabled=True, method=edge_method))

    # 4. Superpixel Reduction
    sp_algo = combo.get("superpixel_algorithm", "none")
    if sp_algo != "none":
        n_sp = combo.get("n_superpixels", 100)
        steps.append(
            StepRegistry.create(
                "superpixel_reduction",
                enabled=True,
                algorithm=sp_algo,
                n_superpixels=n_sp,
            )
        )

    # 5. ROI Crop
    if combo.get("roi_enabled", False):
        roi_params = defaults.get("roi_params", {})
        steps.append(StepRegistry.create("roi_crop", enabled=True, **roi_params))

    # 6. Feature Matching (always enabled)
    fm_params = defaults.get("feature_matching_params", {})
    steps.append(StepRegistry.create("feature_matching", enabled=True, **fm_params))

    return Pipeline(steps=steps)


def run_gridsearch_experiment(config_path: Path, output_dir: Path = None):
    print(f"Loading GridSearch configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    exp_name = config_data.get("experiment_name", "gridsearch_experiment")
    data_cfg = config_data.get("data", {})
    grid_cfg = config_data.get("grid", {})
    defaults = config_data.get("defaults", {})

    ref_path = Path(data_cfg.get("ref_image", "src/assets/vaso_1.jpeg"))
    cur_path = Path(data_cfg.get("cur_image", "src/assets/vaso_2.jpeg"))

    print(f"Loading reference image: {ref_path}")
    print(f"Loading current image:   {cur_path}")
    img_ref = load_image(ref_path)
    img_cur = load_image(cur_path)

    # Generate Cartesian product of all parameters in grid
    keys = list(grid_cfg.keys())
    value_lists = [grid_cfg[k] if isinstance(grid_cfg[k], list) else [grid_cfg[k]] for k in keys]
    combinations = [dict(zip(keys, v)) for v in itertools.product(*value_lists)]

    print(f"\n=> Total GridSearch Combinations to Evaluate: {len(combinations)}\n")

    if output_dir is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(f"runs/gridsearch_{run_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for idx, combo in enumerate(combinations, start=1):
        combo_id = f"config_{idx:03d}"
        print(f"[{idx}/{len(combinations)}] Evaluating {combo_id}: {combo}")

        pipeline = build_pipeline_from_combo(combo, defaults)
        context = pipeline.run(img_ref, img_cur)

        metrics = compute_pipeline_metrics(context)

        # Build summary row
        row = {"combo_id": combo_id}
        row.update(combo)
        row.update(
            {
                "matches_count": metrics["matches_count"],
                "inliers_count": metrics["inliers_count"],
                "inlier_ratio_pct": metrics["inlier_ratio_pct"],
                "stop_layer": metrics["stop_layer"],
                "servoing_error_norm": metrics["servoing_error_norm"],
                "total_time_sec": metrics["total_time_sec"],
            }
        )
        summary_rows.append(row)

        all_results.append({"combo_id": combo_id, "config": combo, "metrics": metrics})

        # Save plot for this run
        combo_dir = output_dir / combo_id
        combo_dir.mkdir(parents=True, exist_ok=True)
        fig = Reporter.render_visualization(
            context, title=f"{combo_id}: {combo.get('superpixel_algorithm','none')}-{combo.get('edge_detection_method','none')}"
        )
        fig.savefig(combo_dir / "plot.png", bbox_inches="tight", dpi=120)

    # Export consolidated summary CSV and JSON
    csv_path = output_dir / "gridsearch_summary.csv"
    Reporter.save_csv_summary(summary_rows, csv_path)
    print(f"\n=> Summary CSV saved to: {csv_path}")

    json_path = output_dir / "gridsearch_details.json"
    Reporter.save_json(all_results, json_path)
    print(f"=> Summary JSON saved to: {json_path}")
    print(f"\n=> GridSearch Complete! All artifacts saved under: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GridSearch experiment for Visual Servoing.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gridsearch_experiment.yaml",
        help="Path to GridSearch YAML config",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    run_gridsearch_experiment(
        config_path=Path(args.config),
        output_dir=Path(args.output) if args.output else None,
    )
