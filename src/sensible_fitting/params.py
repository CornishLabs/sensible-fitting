from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    fixed: bool = False
    fixed_value: Optional[float] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    guess: Optional[float] = None
    # Stored for Bayesian backends (ignored by curve_fit in v1)
    prior: Optional[Tuple[str, Tuple[Any, ...]]] = None
    meta: Dict[str, Any] = None


@dataclass(frozen=True)
class DerivedSpec:
    """Post-fit derived parameter (v1).

    v1 restriction:
    - computed only AFTER fitting
    - depends only on fitted params, not on other derived params
    """
    name: str
    func: Any  # Callable[[Mapping[str, float]], float]
    doc: str = ""
    meta: Dict[str, Any] = None


@dataclass(frozen=True)
class ParamView:
    """A single parameter view."""
    name: str
    value: Any
    error: Any = None
    fixed: Any = False
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    derived: bool = False
    meta: Dict[str, Any] = None

    def __getitem__(self, key: str) -> Any:
        if key == "value":
            return self.value
        if key in ("error", "stderr"):
            return self.error
        if key == "fixed":
            return self.fixed
        if key == "bounds":
            return self.bounds
        if key == "derived":
            return self.derived
        raise KeyError(key)


class ParamsView(Mapping[str, ParamView]):
    """Mapping name -> ParamView."""

    def __init__(self, items: Mapping[str, ParamView]):
        self._items = dict(items)

    def __getitem__(self, key: str) -> ParamView:
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def items(self):
        return self._items.items()

    def as_dict(self) -> Dict[str, Any]:
        """Return name->value (extracting .value)."""
        return {k: v.value for k, v in self._items.items()}


class GuessState:
    """Mutable guess state passed to guessers.

    Supports:
        g.amplitude = 1.0
        g.is_unset("amplitude")
    """

    def __init__(self):
        object.__setattr__(self, "_d", {})

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if name in d:
            return d[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_d":
            object.__setattr__(self, name, value)
            return
        d = object.__getattribute__(self, "_d")
        d[name] = value

    def is_unset(self, name: str) -> bool:
        d = object.__getattribute__(self, "_d")
        return name not in d

    def to_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, "_d"))
