from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Iterable

import numpy as np

try:
    import uncertainties
    from uncertainties import unumpy as unp
except Exception:  # pragma: no cover - optional at import time
    uncertainties = None
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
    # For samplers with periodic/circular parameters (e.g. UltraNest wrapped_params)
    wrapped: bool = False


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


@dataclass
class _UncContext:
    values: Mapping[str, Any]
    stderrs: Mapping[str, Any]
    cov: Optional[np.ndarray]
    free_names: Tuple[str, ...]
    _cache_scalar: Optional[Dict[str, Any]] = field(
        default=None, init=False, repr=False
    )
    _cache_batch: Optional[Dict[str, np.ndarray]] = field(
        default=None, init=False, repr=False
    )

    def _normalized_cov(self) -> Optional[np.ndarray]:
        cov = self.cov
        if isinstance(cov, np.ndarray) and cov.dtype == object and cov.shape == ():
            cov = cov.item()
        return cov

    def _batch_shape(self, cov: Optional[np.ndarray]) -> Tuple[int, ...]:
        if cov is None or not isinstance(cov, np.ndarray):
            return ()
        if cov.dtype == object:
            return cov.shape
        if cov.ndim <= 2:
            return ()
        return cov.shape[:-2]

    def _value_at(self, name: str, idx: Optional[Tuple[int, ...]] = None) -> float:
        v = np.asarray(self.values[name])
        if idx is None:
            if v.shape == ():
                return float(v.item())
            return float(v)
        return float(v[idx])

    def _stderr_at(self, name: str, idx: Optional[Tuple[int, ...]] = None) -> Optional[float]:
        e = self.stderrs.get(name)
        if e is None:
            return None
        a = np.asarray(e)
        if idx is None:
            if a.shape == ():
                return float(a.item())
            return float(a)
        return float(a[idx])

    def _build_scalar_cache(self) -> None:
        if self._cache_scalar is not None or uncertainties is None:
            return
        cov = self._normalized_cov()
        if cov is None:
            return
        if isinstance(cov, np.ndarray) and cov.ndim >= 3:
            return
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2:
            return
        if cov_arr.shape[0] != cov_arr.shape[1]:
            return
        if cov_arr.shape[0] != len(self.free_names):
            return
        vals = [self._value_at(n) for n in self.free_names]
        try:
            corr = uncertainties.correlated_values(vals, cov_arr)
        except Exception:
            return
        self._cache_scalar = dict(zip(self.free_names, corr))

    def _build_batch_cache(self) -> None:
        if self._cache_batch is not None or uncertainties is None:
            return
        cov = self._normalized_cov()
        batch_shape = self._batch_shape(cov)
        if batch_shape == () or cov is None or not isinstance(cov, np.ndarray):
            return

        out = {n: np.empty(batch_shape, dtype=object) for n in self.free_names}
        for idx in np.ndindex(batch_shape):
            cov_i = cov[idx] if cov.dtype == object else cov[idx]
            if isinstance(cov_i, np.ndarray) and cov_i.shape == ():
                cov_i = cov_i.item()

            corr = None
            if cov_i is not None:
                cov_arr = np.asarray(cov_i, dtype=float)
                if cov_arr.shape == (len(self.free_names), len(self.free_names)):
                    vals = [self._value_at(n, idx) for n in self.free_names]
                    try:
                        corr = uncertainties.correlated_values(vals, cov_arr)
                    except Exception:
                        corr = None

            if corr is None:
                for name in self.free_names:
                    val = self._value_at(name, idx)
                    err = self._stderr_at(name, idx)
                    out[name][idx] = uncertainties.ufloat(val, np.nan if err is None else err)
            else:
                for j, name in enumerate(self.free_names):
                    out[name][idx] = corr[j]

        self._cache_batch = out

    def u_for(self, name: str) -> Optional[Any]:
        if uncertainties is None or name not in self.free_names:
            return None
        cov = self._normalized_cov()
        batch_shape = self._batch_shape(cov)
        if batch_shape == ():
            self._build_scalar_cache()
            if self._cache_scalar is None:
                return None
            return self._cache_scalar.get(name)
        self._build_batch_cache()
        if self._cache_batch is None:
            return None
        return self._cache_batch.get(name)

    def u_for_many(self, names: Sequence[str]) -> Optional[np.ndarray]:
        if uncertainties is None:
            return None
        names = tuple(names)
        if any(n not in self.free_names for n in names):
            return None
        cov = self._normalized_cov()
        batch_shape = self._batch_shape(cov)
        if batch_shape == ():
            self._build_scalar_cache()
            if self._cache_scalar is None:
                return None
            return np.array([self._cache_scalar[n] for n in names], dtype=object)
        self._build_batch_cache()
        if self._cache_batch is None:
            return None
        return np.stack([self._cache_batch[n] for n in names], axis=-1)


@dataclass(frozen=True)
class ParamView:
    """A single parameter view."""

    name: str
    value: Any
    stderr: Any = None
    fixed: Any = False
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    derived: bool = False
    _context: Optional[_UncContext] = field(
        default=None, repr=False, compare=False
    )

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

        e = np.asarray(self.stderr)
        # Support object arrays with per-batch missing values (None).
        if e.dtype == object:
            flat: Iterable[Any] = e.ravel().tolist()
            if any(v is None for v in flat):
                raise ValueError(
                    f"stderr for {self.name!r} contains missing entries; slice to a concrete batch element first."
                )
            e = e.astype(float)

        if not np.all(np.isfinite(e)):
            raise ValueError(
                f"stderr for {self.name!r} contains non-finite entries; slice to a valid batch element first."
            )
        if self._context is not None:
            correlated = self._context.u_for(self.name)
            if correlated is not None:
                return correlated
        return unp.uarray(self.value, e)

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
    _context: Optional[_UncContext] = field(
        default=None, repr=False, compare=False
    )

    @property
    def u(self):
        if self.stderr is None:
            raise ValueError("No stderr available for MultiParamView.u.")
        if unp is None:
            raise RuntimeError("uncertainties package is not available.")
        e = np.asarray(self.stderr)
        if e.dtype == object:
            flat = e.ravel().tolist()
            if any(v is None for v in flat):
                raise ValueError(
                    "MultiParamView.stderr contains missing entries; slice to a concrete batch element first."
                )
            e = e.astype(float)
        if not np.all(np.isfinite(e)):
            raise ValueError(
                "MultiParamView.stderr contains non-finite entries; slice to a valid batch element first."
            )
        if self._context is not None:
            correlated = self._context.u_for_many(self.names)
            if correlated is not None:
                return correlated
        return unp.uarray(self.value, e)


class ParamsView(Mapping[str, ParamView]):
    """Mapping name -> ParamView, with rich indexing."""

    def __init__(
        self,
        items: Mapping[str, ParamView],
        *,
        _context: Optional[_UncContext] = None,
    ):
        self._items = dict(items)
        self._names = tuple(self._items.keys())
        self._context = _context

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
                e = np.asarray(pv.stderr)
                # If this is an object array with per-batch None, treat as "no complete stderr".
                if e.dtype == object and any(vv is None for vv in e.ravel().tolist()):
                    all_have_err = False
                    stderrs.append(None)
                else:
                    stderrs.append(np.asarray(pv.stderr, dtype=float))

        value_arr = np.stack(values, axis=-1)
        stderr_arr = None
        if all_have_err:
            stderr_arr = np.stack([np.asarray(e, dtype=float) for e in stderrs], axis=-1)  # type: ignore[arg-type]

        return MultiParamView(
            names=names,
            value=value_arr,
            stderr=stderr_arr,
            _context=self._context,
        )


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
