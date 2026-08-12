import argparse
from datetime import datetime
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from lightglue.utils import load_image

# Ensure steps are imported to trigger @register_step
import src.steps
from src.core.pipeline import Pipeline
from src.core.synthetic import SyntheticTransformGenerator
from src.evaluation.metrics import compute_pipeline_metrics
from src.evaluation.reporter import Reporter


def run_single_experiment(
    config_path: Path,
    ref_image_path: Path = None,
    cur_image_path: Path = None,
    single_image_path: Path = None,
    angle_deg: float = 10.0,
    scale: float = 1.0,
    tx: float = 20.0,
    ty: float = -15.0,
    output_dir: Path = None,
):
    print(f"Loading pipeline configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    pipeline_cfg = config_data.get("pipeline", [])
    pipeline = Pipeline.from_config(pipeline_cfg)

    print(f"Instantiated Pipeline:\n{pipeline}")

    ground_truth_H = None

    if single_image_path is not None:
        print(f"Single image mode selected: {single_image_path}")
        print(f"Applying synthetic transform: Rotation={angle_deg}°, Scale={scale}, Tx={tx}px, Ty={ty}px")
        img_raw = load_image(single_image_path)
        synth_gen = SyntheticTransformGenerator(
            angle_deg=angle_deg, scale=scale, tx=tx, ty=ty
        )
        img_ref, img_cur, ground_truth_H = synth_gen.apply(img_raw)
    else:
        if ref_image_path is None:
            ref_image_path = Path("src/assets/vaso_1.jpeg")
        if cur_image_path is None:
            cur_image_path = Path("src/assets/vaso_2.jpeg")

        print(f"Loading reference image: {ref_image_path}")
        print(f"Loading current image:   {cur_image_path}")
        img_ref = load_image(ref_image_path)
        img_cur = load_image(cur_image_path)

    print("Executing pipeline...")
    context = pipeline.run(img_ref, img_cur)

    metrics = compute_pipeline_metrics(context, ground_truth_H=ground_truth_H)

    print("\n" + "=" * 50)
    print("           PIPELINE EXECUTION METRICS")
    print("=" * 50)
    print(f"Matches count:        {metrics['matches_count']}")
    print(f"Inliers count:        {metrics['inliers_count']} ({metrics['inlier_ratio_pct']}%)")
    print(f"Stop Layer:           {metrics['stop_layer']}")
    if metrics.get('corner_error_px') is not None:
        print(f"Corner Error (px):    {metrics['corner_error_px']} px (Ground Truth)")
    print(f"Servoing Error Norm:  {metrics['servoing_error_norm']}")
    print(f"Total Time (sec):     {metrics['total_time_sec']}s")
    print("Step Breakdown:")
    for step_name, t_sec in metrics["step_times"].items():
        print(f"  - {step_name}: {t_sec:.4f}s")
    print("=" * 50 + "\n")

    if output_dir is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(f"runs/run_pipeline_{run_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fig = Reporter.render_visualization(context, title="Single Pipeline Run")
    fig_path = output_dir / "pipeline_result.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"Saved visualization to: {fig_path}")

    metrics_path = output_dir / "metrics.json"
    Reporter.save_json(metrics, metrics_path)
    print(f"Saved metrics to: {metrics_path}")

    return context, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single Visual Servoing pipeline experiment.")
    parser.add_argument("--config", type=str, default="config/pipeline_default.yaml", help="Path to YAML config file")
    parser.add_argument("--ref", type=str, default="src/assets/vaso_1.jpeg", help="Path to reference image")
    parser.add_argument("--cur", type=str, default="src/assets/vaso_2.jpeg", help="Path to current image")
    parser.add_argument("--single", type=str, default=None, help="Path to a single image for synthetic ground truth evaluation")
    parser.add_argument("--angle", type=float, default=10.0, help="Rotation angle in degrees for synthetic warp")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for synthetic warp")
    parser.add_argument("--tx", type=float, default=20.0, help="Horizontal translation (px) for synthetic warp")
    parser.add_argument("--ty", type=float, default=-15.0, help="Vertical translation (px) for synthetic warp")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()
    run_single_experiment(
        config_path=Path(args.config),
        ref_image_path=Path(args.ref) if args.ref else None,
        cur_image_path=Path(args.cur) if args.cur else None,
        single_image_path=Path(args.single) if args.single else None,
        angle_deg=args.angle,
        scale=args.scale,
        tx=args.tx,
        ty=args.ty,
        output_dir=Path(args.output) if args.output else None,
    )
