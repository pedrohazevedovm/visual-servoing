from typing import List, Union, Dict, Any
import torch
from src.core.context import PipelineContext
from src.core.base_step import BaseStep
from src.core.registry import StepRegistry


class Pipeline:
    """
    Sequential execution engine for pipeline steps.
    Can be instantiated from a list of BaseStep instances, or built declaratively from a config dict/YAML.
    """

    def __init__(self, steps: List[BaseStep] = None):
        self.steps: List[BaseStep] = steps if steps is not None else []

    def add_step(self, step: BaseStep) -> "Pipeline":
        self.steps.append(step)
        return self

    def run(self, img_ref_raw: torch.Tensor, img_cur_raw: torch.Tensor) -> PipelineContext:
        """
        Executes all steps sequentially on the input images.
        """
        context = PipelineContext(img_ref_raw=img_ref_raw, img_cur_raw=img_cur_raw)

        for step in self.steps:
            context = step.run(context)

        return context

    @classmethod
    def from_config(cls, config: List[Dict[str, Any]]) -> "Pipeline":
        """
        Builds a Pipeline instance from a list of step configuration dicts.
        Each dict should have format:
        {
            "type": "histogram_matching",
            "enabled": True,
            "params": { ... }
        }
        """
        steps = []
        for step_cfg in config:
            step_type = step_cfg.get("type")
            enabled = step_cfg.get("enabled", True)
            params = step_cfg.get("params", {})
            
            step_instance = StepRegistry.create(
                name=step_type,
                enabled=enabled,
                **params
            )
            steps.append(step_instance)

        return cls(steps=steps)

    def __repr__(self) -> str:
        step_str = "\n  ".join([str(s) for s in self.steps])
        return f"Pipeline([\n  {step_str}\n])"
