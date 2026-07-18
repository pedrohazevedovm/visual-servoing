from typing import Dict, Type
from src.core.base_step import BaseStep


class StepRegistry:
    """
    Registry for dynamic step lookup and instantiation.
    """

    _registry: Dict[str, Type[BaseStep]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a step class under a string key.
        """

        def decorator(step_cls: Type[BaseStep]):
            if name in cls._registry:
                pass
            cls._registry[name] = step_cls
            return step_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseStep]:
        """
        Retrieves a step class by name.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Step '{name}' is not registered. Available steps: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, enabled: bool = True, **kwargs) -> BaseStep:
        """
        Instantiates a registered step with parameters.
        """
        step_cls = cls.get(name)
        return step_cls(name=name, enabled=enabled, **kwargs)

    @classmethod
    def list_registered(cls):
        return list(cls._registry.keys())


register_step = StepRegistry.register
