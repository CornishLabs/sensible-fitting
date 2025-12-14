from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from uncertainties import unumpy as unp
except Exception:  # pragma: no cover - optional at import time
    unp = None


__all__ = [
    "ParameterSpec",
    "DerivedSpec",
    "ParamView",
    "ParamsView",
    "MultiParamView",
    "GuessState",
]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    fixed: bool = False
    fixed_value: Optional[float] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    # Strong guess: overrides guessers
    guess: Optional[float] = None
    # Weak guess: used only if guessers don't provide a value
    weak_guess: Optional[float] = None
    # Stored for Bayesian backends (ignored by curve_fit in v1)
    prior: Optional[Tuple[str, Tuple[Any, ...]]] = None


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


@dataclass(frozen=True)
class ParamView:
    """A single parameter view."""

    name: str
    value: Any
    stderr: Any = None
    fixed: Any = False
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    derived: bool = False

    # Backwards-compatible alias
    @property
    def error(self) -> Any:  # type: ignore[override]
        return self.stderr

    @property
    def u(self):
        """Return an uncertainties uarray/ufloat if stderr is available."""
        if self.stderr is None:
            raise ValueError(f"No stderr available for parameter {self.name!r}.")
        if unp is None:
            raise RuntimeError("uncertainties package is not available.")
        return unp.uarray(self.value, self.stderr)

    def __getitem__(self, key: str) -> Any:
        if key == "value":
            return self.value
        if key in ("error", "stderr"):
            return self.stderr
        if key == "fixed":
            return self.fixed
        if key == "bounds":
            return self.bounds
        if key == "derived":
            return self.derived
        raise KeyError(key)


@dataclass(frozen=True)
class MultiParamView:
    """View over multiple parameters at once.

    value and stderr have shape batch_shape + (len(names),).
    """

    names: Tuple[str, ...]
    value: Any
    stderr: Any = None

    @property
    def u(self):
        if self.stderr is None:
            raise ValueError("No stderr available for MultiParamView.u.")
        if unp is None:
            raise RuntimeError("uncertainties package is not available.")
        return unp.uarray(self.value, self.stderr)


class ParamsView(Mapping[str, ParamView]):
    """Mapping name -> ParamView, with rich indexing."""

    def __init__(self, items: Mapping[str, ParamView]):
        self._items = dict(items)
        self._names = tuple(self._items.keys())

    def __getitem__(self, key):  # type: ignore[override]
        # Param by name
        if isinstance(key, str):
            return self._items[key]

        # Multi-param by names: ("frequency", "phase") or ["frequency", "phase"]
        if (
            isinstance(key, (tuple, list))
            and key
            and all(isinstance(k, str) for k in key)
        ):
            names = tuple(key)
            return self._multi_by_names(names)

        # Param by index
        if isinstance(key, int):
            name = self._names[key]
            return self._items[name]

        # Slice of params -> MultiParamView
        if isinstance(key, slice):
            names = self._names[key]
            return self._multi_by_names(names)

        # Explicit indices -> MultiParamView
        if (
            isinstance(key, (tuple, list))
            and key
            and all(isinstance(k, int) for k in key)
        ):
            names = tuple(self._names[i] for i in key)
            return self._multi_by_names(names)

        raise KeyError(key)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def items(self):
        return self._items.items()

    def as_dict(self) -> Dict[str, Any]:
        """Return name->value (extracting .value)."""
        return {k: v.value for k, v in self._items.items()}

    # ---- helpers ----
    def _multi_by_names(self, names: Sequence[str]) -> MultiParamView:
        names = tuple(names)
        if not names:
            raise ValueError("MultiParamView requires at least one parameter name.")

        values = []
        stderrs = []
        all_have_err = True
        for n in names:
            pv = self._items[n]
            v = np.asarray(pv.value)
            values.append(v)
            if pv.stderr is None:
                all_have_err = False
                stderrs.append(None)
            else:
                stderrs.append(np.asarray(pv.stderr))

        value_arr = np.stack(values, axis=-1)
        stderr_arr = None
        if all_have_err:
            stderr_arr = np.stack([np.asarray(e) for e in stderrs], axis=-1)  # type: ignore[arg-type]

        return MultiParamView(names=names, value=value_arr, stderr=stderr_arr)


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
