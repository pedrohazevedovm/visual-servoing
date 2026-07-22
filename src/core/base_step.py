from abc import ABC, abstractmethod
import time
from src.core.context import PipelineContext


class BaseStep(ABC):
    """
    Abstract base class for all pipeline steps.
    Each step modifies the PipelineContext and records its execution time.
    """

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        self.name = name
        self.enabled = enabled
        self.params = kwargs

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Executes step if enabled and measures timing.
        """
        if not self.enabled:
            return context

        start_time = time.perf_counter()
        context = self.process(context)
        elapsed = time.perf_counter() - start_time

        context.step_times[self.name] = elapsed
        return context

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Subclasses implement the actual processing logic here.
        """
        pass

    def to_dict(self) -> dict:
        """
        Serializes step to a dictionary representation.
        """
        return {
            "type": self.name,
            "enabled": self.enabled,
            "params": self.params,
        }

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"<{self.__class__.__name__}(name='{self.name}', status={status}, params={self.params})>"
