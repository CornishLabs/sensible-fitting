from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .params import ParamView, ParamsView
from .util import level_to_conf_int, prod, sample_mvn


@dataclass(frozen=True)
class Band:
    low: np.ndarray
    high: np.ndarray
    median: Optional[np.ndarray] = None
    meta: Dict[str, Any] = None


@dataclass(frozen=True)
class Results:
    batch_shape: Tuple[int, ...]
    params: ParamsView
    cov: Optional[np.ndarray] = None
    backend: str = ""
    meta: Dict[str, Any] = None

    def __getitem__(self, idx) -> "Results":
        if self.batch_shape == ():
            raise IndexError("Scalar Results cannot be indexed; already squeezed.")

        def _slice(v):
            if v is None:
                return None
            a = np.asarray(v)
            if a.shape == ():
                return v
            return a[idx]

        new_items = {}
        for name, pv in self.params.items():
            new_items[name] = ParamView(
                name=name,
                value=_slice(pv.value),
                error=_slice(pv.error),
                fixed=_slice(pv.fixed) if isinstance(pv.fixed, np.ndarray) else pv.fixed,
                bounds=pv.bounds,
                derived=pv.derived,
                meta=pv.meta,
            )

        cov = self.cov
        if cov is not None and np.asarray(cov).ndim >= 3:
            cov = np.asarray(cov)[idx]

        new_batch_shape = ()
        for pv in new_items.values():
            a = np.asarray(pv.value)
            if a.shape != ():
                new_batch_shape = a.shape
                break

        return Results(
            batch_shape=tuple(new_batch_shape),
            params=ParamsView(new_items),
            cov=cov,
            backend=self.backend,
            meta=self.meta,
        )

    def summary(self, digits: int = 4) -> str:
        meta = self.meta or {}
        lines = [f"Results(backend={self.backend!r}, batch_shape={self.batch_shape})"]

        if self.batch_shape == ():
            for name, pv in self.params.items():
                v = pv.value
                e = pv.error
                tag = " (derived)" if pv.derived else ""
                if e is None:
                    lines.append(f"  {name:>12s}: {float(v):.{digits}g}{tag}")
                else:
                    lines.append(f"  {name:>12s}: {float(v):.{digits}g} ± {float(e):.{digits}g}{tag}")
            return "\n".join(lines)

        batch_size = prod(self.batch_shape)
        show = min(batch_size, 10)
        names = list(self.params.keys())

        header = "idx " + " ".join([f"{n:>14s}" for n in names])
        lines.append(header)
        lines.append("-" * len(header))

        for i in range(show):
            row = [f"{i:>3d}"]
            for n in names:
                pv = self.params[n]
                v = np.asarray(pv.value).reshape((batch_size,))[i]
                e = pv.error
                if e is None:
                    row.append(f"{float(v):>14.{digits}g}")
                else:
                    eflat = np.asarray(e).reshape((batch_size,))[i]
                    row.append(f"{float(v):>7.{digits}g}±{float(eflat):<6.{digits}g}")
            lines.append(" ".join(row))

        if batch_size > show:
            lines.append(f"... ({batch_size-show} more)")
        return "\n".join(lines)


@dataclass(frozen=True)
class Run:
    model: Any  # Model
    results: Results
    backend: str
    data_format: str
    meta: Dict[str, Any] = None
    data: Dict[str, Any] = None

    def squeeze(self) -> "Run":
        if self.results.batch_shape == ():
            return self
        batch_size = prod(self.results.batch_shape)
        if batch_size != 1:
            raise ValueError(f"run.squeeze() requires exactly one fit; got batch_size={batch_size}. Slice first.")
        idx = tuple(0 for _ in self.results.batch_shape)
        return self[idx]

    def __getitem__(self, idx) -> "Run":
        sub_results = self.results[idx]
        sub_data = None
        if self.data is not None:
            sub_data = {}
            for k, v in self.data.items():
                if isinstance(v, (list, tuple)):
                    sub_data[k] = v[idx]
                else:
                    sub_data[k] = v
        return Run(
            model=self.model,
            results=sub_results,
            backend=self.backend,
            data_format=self.data_format,
            meta=self.meta,
            data=sub_data,
        )

    def band(
        self,
        x: Any,
        *,
        nsamples: int = 400,
        level: Optional[float] = None,
        conf_int: Optional[Tuple[float, float]] = None,
        method: Literal["auto", "posterior", "covariance"] = "auto",
        rng: Optional[np.random.Generator] = None,
    ) -> Band:
        if self.results.batch_shape != ():
            raise ValueError("run.band() requires a scalar run. Slice first (e.g., run[i].band(...)).")

        if rng is None:
            rng = np.random.default_rng()

        if level is None and conf_int is None:
            level = 2.0
        if level is not None and conf_int is not None:
            raise ValueError("Provide only one of level= or conf_int=.")

        if conf_int is None:
            qlo, qhi = level_to_conf_int(float(level))
        else:
            qlo, qhi = conf_int

        # v1: covariance-only (posterior reserved)
        if method not in ("auto", "covariance"):
            raise NotImplementedError("v1: posterior-based band() is reserved.")

        cov = self.results.cov
        if cov is None:
            raise ValueError("No covariance available for band().")

        meta = self.results.meta or {}
        free_names = meta.get("free_param_names")
        if not free_names:
            raise ValueError("Results.meta['free_param_names'] missing; cannot compute band().")

        mean = np.array([float(self.results.params[n].value) for n in free_names], dtype=float)
        cov = np.asarray(cov, dtype=float)

        theta = sample_mvn(mean, cov, int(nsamples), rng)  # (S,P)

        preds = []
        for s in range(theta.shape[0]):
            p = {name: theta[s, j] for j, name in enumerate(free_names)}
            preds.append(np.asarray(self.model.eval(x, **p)))
        preds = np.stack(preds, axis=0)

        lo = np.quantile(preds, qlo, axis=0)
        hi = np.quantile(preds, qhi, axis=0)
        med = np.quantile(preds, 0.5, axis=0)

        return Band(low=lo, high=hi, median=med, meta={"method": "covariance", "q": (qlo, qhi)})
