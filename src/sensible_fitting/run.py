from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np

from .params import ParamView, ParamsView, _UncContext
from .util import level_to_conf_int, prod, sample_mvn


@dataclass(frozen=True)
class Band:
    low: np.ndarray
    high: np.ndarray
    median: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SuggestNextXResult:
    x: np.ndarray
    candidates: np.ndarray
    score: np.ndarray
    method: str
    objective: str


@dataclass(frozen=True)
class Results:
    batch_shape: Tuple[int, ...]
    params: ParamsView
    seed: Optional[ParamsView] = None
    cov: Optional[np.ndarray] = None
    backend: str = ""
    # Backend-specific extras (e.g. UltraNest logZ / posterior samples)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        """Return ParamView(s) or a batch-sliced Results view."""
        # ---- Parameter access sugar ----------------------------------------
        # res["frequency"]           -> ParamView (all batches)
        # res["frequency", "phase"]  -> MultiParamView
        # res[["frequency", "phase"]] same as above
        if isinstance(key, str):
            return self.params[key]

        if (
            isinstance(key, (tuple, list))
            and key
            and all(isinstance(k, str) for k in key)
        ):
            return self.params[key]

        # ---- Batch slicing --------------------------------------------------
        idx = key
        if self.batch_shape == ():
            raise IndexError("Scalar Results cannot be indexed; already squeezed.")
        
        # Slice backend stats in a conservative way.
        def _slice_stats(stats: Dict[str, Any], idx: Any) -> Dict[str, Any]:
            """Slice only per-batch stats while preserving global keys."""
            if not stats:
                return {}
            if "per_batch" not in stats:
                return dict(stats)

            out: Dict[str, Any] = {k: v for k, v in stats.items() if k != "per_batch"}
            pb = stats.get("per_batch")
            if isinstance(pb, np.ndarray):
                if pb.shape == ():
                    pb = pb.item()
                else:
                    pb = pb[idx]
                if isinstance(pb, np.ndarray) and pb.shape == ():
                    pb = pb.item()

            if isinstance(pb, dict) and not out:
                return pb

            out["per_batch"] = pb
            return out

        def _slice(v):
            """Slice an array-like along the batch index."""
            if v is None:
                return None
            a = np.asarray(v)
            if a.shape == ():
                return v
            return a[idx]

        cov = self.cov
        # cov may be:
        #  - None
        #  - ndarray (P,P) for scalar
        #  - ndarray object array shaped batch_shape containing (P,P) arrays / None
        #  - dense stacked array (B,P,P) for legacy/other backends
        if isinstance(cov, np.ndarray):
            if cov.dtype == object and cov.shape[: len(self.batch_shape)] == self.batch_shape:
                cov = cov[idx]
                if isinstance(cov, np.ndarray) and cov.shape == ():
                    cov = cov.item()
            elif cov.ndim >= 3:
                cov = cov[idx]

        old_ctx = getattr(self.params, "_context", None)
        free_names = ()
        if old_ctx is not None:
            free_names = tuple(old_ctx.free_names)
        elif isinstance(self.stats, dict):
            free_names = tuple(self.stats.get("free_names", ()))

        values_map: Dict[str, Any] = {}
        stderr_map: Dict[str, Any] = {}
        ctx = _UncContext(
            values=values_map,
            stderrs=stderr_map,
            cov=cov,
            free_names=free_names,
        )

        new_items: Dict[str, ParamView] = {}
        for name, pv in self.params.items():
            v = _slice(pv.value)
            e = _slice(pv.stderr)
            values_map[name] = v
            stderr_map[name] = e
            new_items[name] = ParamView(
                name=name,
                value=v,
                stderr=e,
                fixed=_slice(pv.fixed)
                if isinstance(pv.fixed, np.ndarray)
                else pv.fixed,
                bounds=pv.bounds,
                derived=pv.derived,
                _context=ctx,
            )

        new_seed = None
        if self.seed is not None:
            seed_items: Dict[str, ParamView] = {}
            for name, pv in self.seed.items():
                seed_items[name] = ParamView(
                    name=name,
                    value=_slice(pv.value),
                    stderr=_slice(pv.stderr),
                    fixed=_slice(pv.fixed)
                    if isinstance(pv.fixed, np.ndarray)
                    else pv.fixed,
                    bounds=pv.bounds,
                    derived=pv.derived,
                )
            new_seed = ParamsView(seed_items)

        new_batch_shape = ()
        for pv in new_items.values():
            a = np.asarray(pv.value)
            if a.shape != ():
                new_batch_shape = a.shape
                break

        return Results(
            batch_shape=tuple(new_batch_shape),
            params=ParamsView(new_items, _context=ctx),
            seed=new_seed,
            cov=cov,
            backend=self.backend,
            stats=_slice_stats(self.stats, idx),
        )

    def summary(self, digits: int = 4) -> str:
        """Return a human-readable summary string for the results."""
        lines = [f"Results(backend={self.backend!r}, batch_shape={self.batch_shape})"]
        if self.batch_shape == () and self.stats:
            if "logz" in self.stats:
                try:
                    lines.append(
                        f"  {'logZ':>12s}: {float(self.stats['logz']):.{digits}g} ± {float(self.stats.get('logzerr', float('nan'))):.{digits}g}"
                    )
                except Exception:
                    pass
 

        if self.batch_shape == ():
            for name, pv in self.params.items():
                v = pv.value
                e = pv.error
                tag = " (derived)" if pv.derived else ""
                if e is None:
                    lines.append(f"  {name:>12s}: {float(v):.{digits}g}{tag}")
                else:
                    lines.append(
                        f"  {name:>12s}: {float(v):.{digits}g} ± {float(e):.{digits}g}{tag}"
                    )
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
                    eflat = np.asarray(e, dtype=object).reshape((batch_size,))[i]
                    if eflat is None:
                        row.append(f"{float(v):>14.{digits}g}")
                    else:
                        try:
                            ef = float(eflat)
                            if not np.isfinite(ef):
                                row.append(f"{float(v):>14.{digits}g}")
                            else:
                                row.append(f"{float(v):>7.{digits}g}±{ef:<6.{digits}g}")
                        except Exception:
                            row.append(f"{float(v):>14.{digits}g}")
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
    data: Dict[str, Any] = None
    success: Any = True
    message: Any = ""

    def squeeze(self) -> "Run":
        """Return a scalar Run if batch size is 1; otherwise raise."""
        if self.results.batch_shape == ():
            return self
        batch_size = prod(self.results.batch_shape)
        if batch_size != 1:
            raise ValueError(
                f"run.squeeze() requires exactly one fit; got batch_size={batch_size}. Slice first."
            )
        idx = tuple(0 for _ in self.results.batch_shape)
        return self[idx]

    def __getitem__(self, idx) -> "Run":
        """Return a Run for a batch slice."""
        sub_results = self.results[idx]

        sub_data = None
        if self.data is not None:
            sub_data = {}
            bs = self.results.batch_shape  # batch shape *before* slicing

            for k, v in self.data.items():
                # Ragged batch stores list-of-datasets; index directly.
                if isinstance(v, list):
                    sub_data[k] = _index_ragged_list(v, idx)

                # Structured payload tuples like (y, sigma), (n, k), (alpha, beta), etc:
                # slice any numpy array that carries the batch dimension.
                elif isinstance(v, tuple):
                    parts = []
                    for part in v:
                        if (
                            isinstance(part, np.ndarray)
                            and bs != ()
                            and part.shape[: len(bs)] == bs
                        ):
                            parts.append(part[idx])
                        else:
                            parts.append(part)
                    sub_data[k] = tuple(parts)

                else:
                    sub_data[k] = v

        return Run(
            model=self.model,
            results=sub_results,
            backend=self.backend,
            data_format=self.data_format,
            data=sub_data,
            success=_slice_like(self.success, idx),
            message=_slice_like(self.message, idx),
        )

    def predict(
        self,
        x: Any,
        *,
        which: Literal["fit", "seed"] = "fit",
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Evaluate the model at `x` using fitted or seed parameters.

        which="fit"  -> use results.params
        which="seed" -> use results.seed (if available)
        params=...   -> explicit param mapping; 'which' must be "fit"
        """
        if params is not None and which != "fit":
            raise ValueError("Cannot pass explicit params when which != 'fit'.")

        if params is None:
            if which == "fit":
                p = self.results.params
            elif which == "seed":
                if self.results.seed is None:
                    raise ValueError("No seed parameters available on this Run.")
                p = self.results.seed
            else:
                raise ValueError(f"Unknown value for 'which': {which!r}")
        else:
            p = params

        return self.model.eval(x, params=p)

    def plot(
        self,
        *,
        ax: Optional[Any] = None,
        axs: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Plot this run using sensible defaults.

        This is a convenience wrapper around `sensible_fitting.viz.plot_run`.
        """
        from .viz import plot_run

        return plot_run(run=self, ax=ax, axs=axs, **kwargs)

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
        """Compute a predictive band for the model at x."""
        if self.results.batch_shape != ():
            raise ValueError(
                "run.band() requires a scalar run. Slice first (e.g., run[i].band(...))."
            )

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

        stats = getattr(self.results, "stats", {}) or {}

        if method == "auto":
            method = "posterior" if ("posterior_samples" in stats) else "covariance"

        if method == "posterior":
            samples = np.asarray(stats.get("posterior_samples", None))
            if samples is None or samples.size == 0:
                raise ValueError("No posterior_samples available for posterior band().")
            if samples.ndim != 2:
                raise ValueError("posterior_samples must have shape (S, P).")

            free_names = list(stats.get("free_names", ()))
            if not free_names:
                free_names = [p.name for p in getattr(self.model, "params") if not p.fixed]

            if samples.shape[1] != len(free_names):
                raise ValueError(
                    f"posterior_samples has P={samples.shape[1]} columns but free_names has {len(free_names)}."
                )

            S = samples.shape[0]
            take = min(int(nsamples), int(S))
            if take <= 0:
                raise ValueError("nsamples must be >= 1.")
            if take < S:
                idx = rng.choice(S, size=take, replace=False)
                theta = samples[idx]
            else:
                theta = samples

            preds = []
            for s in range(theta.shape[0]):
                p = {name: float(theta[s, j]) for j, name in enumerate(free_names)}
                preds.append(np.asarray(self.model.eval(x, **p)))
            preds = np.stack(preds, axis=0)

            lo = np.quantile(preds, qlo, axis=0)
            hi = np.quantile(preds, qhi, axis=0)
            med = np.quantile(preds, 0.5, axis=0)
            return Band(low=lo, high=hi, median=med)

        if method != "covariance":
            raise ValueError(f"Unknown band() method: {method!r}")

        cov = self.results.cov
        if cov is None:
            raise ValueError("No covariance available for band().")

        # Derive free parameter names from the model (no meta required).
        free_names = [p.name for p in getattr(self.model, "params") if not p.fixed]
        if not free_names:
            raise ValueError("No free parameters; cannot compute band().")

        mean = np.array(
            [float(self.results.params[n].value) for n in free_names], dtype=float
        )
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

        return Band(low=lo, high=hi, median=med)

    def suggest_next_x(
        self,
        candidates: Optional[Any] = None,
        *,
        bounds: Optional[Tuple[float, float]] = None,
        n_candidates: int = 800,
        n: int = 1,
        method: Literal["auto", "posterior", "covariance"] = "auto",
        objective: str = "auto",
        nsamples: int = 400,
        level: Optional[float] = None,
        conf_int: Optional[Tuple[float, float]] = None,
        avoid_existing: bool = True,
        min_separation: float = 0.0,
        noise: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        return_details: bool = False,
    ) -> Any:
        """Suggest the next x point(s) to measure.

        Scores candidate x locations using the current parameter uncertainty and
        returns the best point(s) by a simple objective (default: "auto" ->
        "info_gain" when a noise scale is known, else "max_width").
        """
        if self.results.batch_shape != ():
            raise ValueError(
                "suggest_next_x requires a scalar run. Slice first (e.g., run[i].suggest_next_x(...))."
            )

        if rng is None:
            rng = np.random.default_rng()

        obj = str(objective).lower().strip()
        if obj in ("width", "maxband", "max_band", "maxwidth"):
            obj = "max_width"
        if obj in ("var", "variance", "maxvariance"):
            obj = "max_var"

        if level is None and conf_int is None:
            level = 2.0
        if level is not None and conf_int is not None:
            raise ValueError("Provide only one of level= or conf_int=.")
        if conf_int is None:
            qlo, qhi = level_to_conf_int(float(level))
        else:
            qlo, qhi = conf_int

        def _infer_noise_from_run() -> Optional[Any]:
            if noise is not None:
                return noise
            data = getattr(self, "data", None) or {}
            if getattr(self, "data_format", None) != "normal":
                return None
            payload = data.get("data", None)
            if isinstance(payload, tuple) and len(payload) == 2:
                return payload[1]
            if isinstance(payload, tuple) and len(payload) == 3:
                try:
                    lo = np.asarray(payload[1], dtype=float)
                    hi = np.asarray(payload[2], dtype=float)
                    return 0.5 * (lo + hi)
                except Exception:
                    return None
            return None

        # ---- build candidate set ------------------------------------------
        if candidates is None:
            if bounds is None:
                data = getattr(self, "data", None) or {}
                if "x" not in data:
                    raise ValueError(
                        "No candidates/bounds provided and no stored x data to infer bounds."
                    )
                x_data = np.asarray(data["x"], dtype=float)
                if x_data.ndim != 1:
                    raise ValueError(
                        "suggest_next_x can only infer bounds from 1D x data."
                    )
                xmin = float(np.min(x_data))
                xmax = float(np.max(x_data))
            else:
                xmin, xmax = bounds
                xmin = float(xmin)
                xmax = float(xmax)
            if not (np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin):
                raise ValueError("Invalid bounds; require finite xmax > xmin.")
            n_candidates = int(n_candidates)
            if n_candidates < 2:
                raise ValueError("n_candidates must be >= 2.")
            x_cand = np.linspace(xmin, xmax, n_candidates, dtype=float)
        else:
            x_cand = np.asarray(candidates, dtype=float).reshape((-1,))

        if x_cand.size == 0:
            raise ValueError("No candidate x points to score.")

        # Avoid existing measured x by default (including exact endpoints in linspace).
        if avoid_existing:
            data = getattr(self, "data", None) or {}
            if "x" in data:
                x_exist = np.asarray(data["x"], dtype=float)
                if x_exist.ndim == 1 and x_exist.size:
                    tol = 1e-12 * (
                        float(np.max(x_cand)) - float(np.min(x_cand)) + 1.0
                    )
                    mask = np.ones_like(x_cand, dtype=bool)
                    for xv in x_exist.reshape((-1,)).tolist():
                        if min_separation > 0.0:
                            mask &= (
                                np.abs(x_cand - float(xv)) > float(min_separation)
                            )
                        else:
                            mask &= ~np.isclose(
                                x_cand, float(xv), rtol=0.0, atol=tol
                            )
                    x_cand = x_cand[mask]

        if x_cand.size == 0:
            raise ValueError(
                "All candidates were removed (avoid_existing/min_separation too strict)."
            )

        if obj == "auto":
            # If we have a noise scale, prefer an information-gain heuristic.
            obj = "info_gain" if (_infer_noise_from_run() is not None) else "max_width"

        # ---- parameter samples --------------------------------------------
        stats = getattr(self.results, "stats", {}) or {}
        samples = stats.get("posterior_samples", None)
        has_posterior = (
            isinstance(samples, np.ndarray) and samples.size and samples.ndim == 2
        )

        method_use = str(method)
        if method_use == "auto":
            method_use = "posterior" if has_posterior else "covariance"

        free_names = list(stats.get("free_names", ()))
        if not free_names:
            ctx = getattr(self.results.params, "_context", None)
            if ctx is not None:
                try:
                    free_names = list(getattr(ctx, "free_names", ()))
                except Exception:
                    free_names = []
        if not free_names:
            free_names = [
                p.name
                for p in getattr(getattr(self, "model", None), "params")
                if not getattr(p, "fixed", False)
            ]

        if method_use == "posterior":
            if not has_posterior:
                raise ValueError(
                    "No posterior samples available for method='posterior'."
                )
            samples_arr = np.asarray(samples, dtype=float)
            if samples_arr.shape[1] != len(free_names):
                raise ValueError(
                    f"posterior_samples has P={samples_arr.shape[1]} columns but free_names has {len(free_names)}."
                )
            S = int(samples_arr.shape[0])
            take = min(int(nsamples), S)
            if take <= 0:
                raise ValueError("nsamples must be >= 1.")
            if take < S:
                # Deterministic downsample for stability.
                idx = np.linspace(0, S - 1, take, dtype=int)
                theta = samples_arr[idx]
            else:
                theta = samples_arr
        elif method_use == "covariance":
            cov = self.results.cov
            if cov is None:
                raise ValueError("No covariance available for method='covariance'.")
            mean = np.array(
                [float(self.results.params[n].value) for n in free_names], dtype=float
            )
            cov = np.asarray(cov, dtype=float)
            theta = sample_mvn(mean, cov, int(nsamples), rng)
        else:
            raise ValueError(f"Unknown method: {method!r}.")

        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 2 or theta.shape[0] == 0:
            raise ValueError("Could not build parameter samples for scoring.")

        # ---- predict on candidates ----------------------------------------
        S = int(theta.shape[0])
        N = int(x_cand.shape[0])
        preds = np.empty((S, N), dtype=float)
        for i in range(S):
            p = {name: float(theta[i, j]) for j, name in enumerate(free_names)}
            y = np.asarray(self.predict(x_cand, params=p), dtype=float)
            if y.shape != x_cand.shape:
                try:
                    y = np.broadcast_to(y, x_cand.shape)
                except Exception as exc:
                    raise ValueError(
                        f"Model output shape {y.shape} not broadcastable to candidates shape {x_cand.shape}."
                    ) from exc
            preds[i] = y

        # ---- score ---------------------------------------------------------
        if obj == "max_width":
            lo = np.quantile(preds, qlo, axis=0)
            hi = np.quantile(preds, qhi, axis=0)
            score = np.asarray(hi - lo, dtype=float)
        else:
            var = np.var(preds, axis=0, ddof=1 if preds.shape[0] >= 2 else 0)
            if obj == "max_var":
                score = np.asarray(var, dtype=float)
            elif obj == "info_gain":
                sigma = _infer_noise_from_run()
                if sigma is None:
                    sigma2 = 1.0
                else:
                    s = np.asarray(sigma, dtype=float)
                    if s.shape not in ((), x_cand.shape):
                        try:
                            s = np.broadcast_to(s, x_cand.shape)
                        except Exception as exc:
                            raise ValueError(
                                "noise must be scalar or broadcastable to candidates."
                            ) from exc
                    sigma2 = s * s
                score = 0.5 * np.log1p(np.asarray(var, dtype=float) / sigma2)
            else:
                raise ValueError(f"Unknown objective: {objective!r}.")

        if not np.all(np.isfinite(score)):
            score = np.where(np.isfinite(score), score, -np.inf)

        # ---- choose best points (greedy) -----------------------------------
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1.")

        chosen: list[float] = []
        available = np.ones_like(x_cand, dtype=bool)

        def _exclude_near(x0: float) -> None:
            nonlocal available
            if min_separation > 0.0:
                available &= (np.abs(x_cand - float(x0)) > float(min_separation))
            else:
                tol = 1e-12 * (
                    float(np.max(x_cand)) - float(np.min(x_cand)) + 1.0
                )
                available &= ~np.isclose(x_cand, float(x0), rtol=0.0, atol=tol)

        for _ in range(n):
            if not np.any(available):
                break
            idx = int(np.argmax(np.where(available, score, -np.inf)))
            if not np.isfinite(score[idx]):
                break
            x_best = float(x_cand[idx])
            chosen.append(x_best)
            _exclude_near(x_best)

        x_out = np.asarray(chosen, dtype=float)
        if x_out.size == 0:
            raise ValueError("Could not select a next x point (all scores non-finite).")

        if return_details:
            return SuggestNextXResult(
                x=x_out,
                candidates=x_cand,
                score=score,
                method=str(method_use),
                objective=str(obj),
            )

        if x_out.size == 1:
            return float(x_out[0])
        return x_out


def _normalize_ragged_index(idx: Any) -> Any:
    """Normalize ragged batch indices to 1D indexing."""
    # Ragged batches are 1D (list-of-datasets). squeeze() uses idx=(0,).
    if isinstance(idx, tuple):
        if len(idx) == 1:
            return idx[0]
        raise IndexError("Ragged batches support only 1D indexing.")
    return idx


def _index_ragged_list(v: list[Any], idx: Any) -> Any:
    """Index a ragged list with int/slice/bool/array indices."""
    idx = _normalize_ragged_index(idx)

    if isinstance(idx, (int, slice)):
        return v[idx]

    a = np.asarray(idx)
    if a.dtype == bool:
        if a.ndim != 1 or a.shape[0] != len(v):
            raise IndexError("Boolean mask has wrong shape for ragged batch.")
        return [vv for vv, keep in zip(v, a.tolist()) if keep]

    inds = [int(i) for i in a.ravel().tolist()]
    return [v[i] for i in inds]


def _slice_like(v: Any, idx: Any) -> Any:
    """Slice success/message arrays similarly to Results slicing."""
    idx = _normalize_ragged_index(idx)
    if isinstance(v, (bool, str)) or v is None:
        return v
    try:
        a = np.asarray(v)
        if a.shape == ():
            return v
        return a[idx]
    except Exception:
        return v
