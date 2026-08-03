import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


class WandbLogger:
    """
    Handles logging visual servoing experiments, metrics, visualizations, and artifacts to Weights & Biases (wandb.ai).
    Supports graceful fallback if wandb is disabled or not installed.
    """

    def __init__(
        self,
        enabled: bool = True,
        project: str = "visual-servoing",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "eval",
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,  # "online", "offline", "disabled"
    ):
        self.enabled = enabled and HAS_WANDB
        self.run = None

        if not HAS_WANDB and enabled:
            print("[WandbLogger] Warning: 'wandb' package is not installed. W&B logging will be skipped.")
            return

        if self.enabled:
            # Mode resolution: CLI / init arg > environment variable > default "online"
            run_mode = mode or os.getenv("WANDB_MODE", "online")

            init_kwargs = {
                "project": project,
                "entity": entity,
                "group": group,
                "job_type": job_type,
                "name": name,
                "tags": tags,
                "config": config,
                "mode": run_mode,
                "reinit": True,
            }
            # Remove None values
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

            try:
                self.run = wandb.init(**init_kwargs)
                print(f"[WandbLogger] W&B run initialized: {self.run.name} (Project: {project}, Mode: {run_mode})")
            except Exception as e:
                print(f"[WandbLogger] Failed to initialize W&B run: {e}")
                self.enabled = False

    def log_run_results(
        self,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        fig: Optional[plt.Figure] = None,
        output_dir: Optional[Path] = None,
        upload_artifacts: bool = True,
    ):
        """
        Logs metrics, timing breakdown, figure visualization, and output artifacts to the active W&B run.
        """
        if not self.enabled or self.run is None:
            return

        # Update config if provided
        if config:
            self.run.config.update(config, allow_val_change=True)

        log_payload: Dict[str, Any] = {}

        # 1. Log scalar metrics
        scalar_keys = [
            "matches_count",
            "inliers_count",
            "inlier_ratio_pct",
            "stop_layer",
            "corner_error_px",
            "homography_matrix_error",
            "servoing_error_norm",
            "servoing_error_norm_gt",
            "servoing_error_diff_norm",
            "total_time_sec",
        ]
        for key in scalar_keys:
            if key in metrics and metrics[key] is not None:
                log_payload[key] = metrics[key]

        # 2. Log per-step timing breakdown under step_times/ namespace
        step_times = metrics.get("step_times", {})
        for step_name, t_sec in step_times.items():
            log_payload[f"step_times/{step_name}"] = t_sec

        # 3. Log visualization figure if provided
        if fig is not None:
            log_payload["visualization"] = wandb.Image(fig)

        # Log payload to wandb
        wandb.log(log_payload)

        # Update summary metrics for quick dashboard overview/sorting
        for k, v in log_payload.items():
            if not isinstance(v, wandb.Image):
                wandb.run.summary[k] = v

        # 4. Upload local directory contents as W&B Artifact if output_dir is given
        if upload_artifacts and output_dir is not None and output_dir.exists():
            try:
                artifact_name = f"run_output_{self.run.id}"
                artifact = wandb.Artifact(name=artifact_name, type="run_artifacts")
                artifact.add_dir(str(output_dir))
                self.run.log_artifact(artifact)
            except Exception as e:
                print(f"[WandbLogger] Could not upload W&B artifact: {e}")

    @staticmethod
    def log_gridsearch_summary_table(
        summary_rows: List[Dict[str, Any]],
        table_name: str = "gridsearch_summary_table",
    ):
        """
        Logs a structured wandb.Table containing all grid search parameter combinations and metrics.
        """
        if not HAS_WANDB or wandb.run is None or not summary_rows:
            return

        cols = list(summary_rows[0].keys())
        table = wandb.Table(columns=cols)
        for row in summary_rows:
            table.add_data(*[row.get(c) for c in cols])

        wandb.log({table_name: table})

    def finish(self):
        """
        Finishes the active W&B run.
        """
        if self.enabled and self.run is not None:
            self.run.finish()
            self.run = None
