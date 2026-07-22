from typing import List, Union, Dict, Any, Optional
import torch
from src.core.context import PipelineContext
from src.core.base_step import BaseStep
from src.core.registry import StepRegistry


class Pipeline:
    """
    Sequential execution engine for pipeline steps.
    Can be instantiated from a list of BaseStep instances, or built declaratively from a config dict/YAML.
    """

    def __init__(self, steps: List[BaseStep] = None, config: Optional[Any] = None):
        self.steps: List[BaseStep] = steps if steps is not None else []
        self.config = config

    def add_step(self, step: BaseStep) -> "Pipeline":
        self.steps.append(step)
        return self

    def get_config(self) -> Any:
        if self.config is not None:
            return self.config
        return [step.to_dict() for step in self.steps]

    def run(
        self,
        img_ref_raw: torch.Tensor,
        img_cur_raw: torch.Tensor,
        config: Optional[Any] = None,
    ) -> PipelineContext:
        """
        Executes all steps sequentially on the input images.
        """
        run_config = config if config is not None else self.get_config()
        context = PipelineContext(
            img_ref_raw=img_ref_raw,
            img_cur_raw=img_cur_raw,
            config=run_config,
        )

        for step in self.steps:
            context = step.run(context)

        return context

    @classmethod
    def from_config(
        cls, config: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> "Pipeline":
        """
        Builds a Pipeline instance from a list or dict of step configuration dicts.
        """
        if isinstance(config, dict) and "pipeline" in config:
            pipeline_cfg = config["pipeline"]
        elif isinstance(config, list):
            pipeline_cfg = config
        else:
            pipeline_cfg = []

        steps = []
        for step_cfg in pipeline_cfg:
            step_type = step_cfg.get("type")
            enabled = step_cfg.get("enabled", True)
            params = step_cfg.get("params", {})

            step_instance = StepRegistry.create(
                name=step_type, enabled=enabled, **params
            )
            steps.append(step_instance)

        return cls(steps=steps, config=config)

    def __repr__(self) -> str:
        step_str = "\n  ".join([str(s) for s in self.steps])
        return f"Pipeline([\n  {step_str}\n])"
