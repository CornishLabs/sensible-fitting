from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from warnings import warn

from .backends import get_backend
from .data import Dataset, prepare_datasets
from .inputs import FitData
from .params import DerivedSpec, GuessState, ParameterSpec, ParamView, ParamsView, _UncContext
from .run import Results, Run
from .util import (
    infer_param_names,
    prod,
    unflatten_batch,
)

Guesser = Callable[[Any, Any, GuessState], None]


@dataclass
class Model:
    """A model wraps a callable and parameter metadata."""

    name: str
    func: Callable[..., Any]
    param_names: Tuple[str, ...]
    params: Tuple[ParameterSpec, ...]
    guessers: Tuple[Guesser, ...] = ()
    derived: Tuple[DerivedSpec, ...] = ()

    # ---- constructor ----
    @staticmethod
    def from_function(
        func: Callable[..., Any], *, name: Optional[str] = None
    ) -> "Model":
        """Construct a Model from a plain function signature."""
        names = infer_param_names(func)

        # Treat numeric defaults in the function signature as default guesses.
        sig = inspect.signature(func)
        specs = []
        for n in names:
            p = sig.parameters[n]
            g = None
            if p.default is not inspect._empty:
                d = p.default
                if isinstance(d, (int, float, np.number)) and not isinstance(d, bool):
                    g = float(d)
            # Function defaults should be weak guesses: let guessers override them.
            specs.append(ParameterSpec(name=n, weak_guess=g))
        specs = tuple(specs)
        return Model(
            name=name or getattr(func, "__name__", "model"),
            func=func,
            param_names=names,
            params=specs,
        )

    # ---- evaluation ----
    def eval(
        self, x: Any, *, params: Optional[Mapping[str, Any]] = None, **kwargs
    ) -> Any:
        """Evaluate the model function at x with given parameters."""
        values: Dict[str, Any] = {}

        if params is not None:
            for k, v in params.items():
                if isinstance(v, ParamView):
                    values[k] = v.value
                else:
                    # If someone passes a dict-like with ["value"], allow it.
                    try:
                        if hasattr(v, "__getitem__"):
                            values[k] = v["value"]  # type: ignore[index]
                        else:
                            values[k] = v
                    except Exception:
                        values[k] = v

        values.update(kwargs)

        # Fill fixed values if absent
        for spec in self.params:
            if spec.fixed and spec.name not in values:
                values[spec.name] = spec.fixed_value

        missing = [n for n in self.param_names if n not in values]
        if missing:
            raise TypeError(f"Missing parameter values for: {missing}")

        args = [x] + [values[n] for n in self.param_names]
        return self.func(*args)

    # ---- builders (pure; return new model) ----
    def fix(self, **fixed: float) -> "Model":
        """Return a new Model with parameters fixed to values."""
        m = {p.name: p for p in self.params}
        for k, v in fixed.items():
            if k not in m:
                raise KeyError(k)
            m[k] = replace(m[k], fixed=True, fixed_value=float(v))
        return replace(self, params=tuple(m[n] for n in self.param_names))

    def bound(self, **bounds: Tuple[Optional[float], Optional[float]]) -> "Model":
        """Return a new Model with parameter bounds applied."""
        m = {p.name: p for p in self.params}
        for k, b in bounds.items():
            if k not in m:
                raise KeyError(k)
            lo, hi = b
            m[k] = replace(m[k], bounds=(lo, hi))
        return replace(self, params=tuple(m[n] for n in self.param_names))

    def guess(self, **guesses: float) -> "Model":
        """Return a new Model with strong parameter guesses."""
        m = {p.name: p for p in self.params}
        for k, g in guesses.items():
            if k not in m:
                raise KeyError(k)
            m[k] = replace(m[k], guess=float(g))
        return replace(self, params=tuple(m[n] for n in self.param_names))

    def weak_guess(self, **guesses: float) -> "Model":
        """Set weak (low-precedence) guesses.

        Weak guesses are used only if guessers don't provide a value for that parameter.
        Strong guesses set via .guess(...) override guessers.
        """
        m = {p.name: p for p in self.params}
        for k, g in guesses.items():
            if k not in m:
                raise KeyError(k)
            m[k] = replace(m[k], weak_guess=float(g))
        return replace(self, params=tuple(m[n] for n in self.param_names))

    def prior(self, **priors: Tuple[str, Any]) -> "Model":
        """Return a new Model with Bayesian priors set for parameters."""
        m = {p.name: p for p in self.params}
        for k, p in priors.items():
            if k not in m:
                raise KeyError(k)
            if not isinstance(p, tuple) or len(p) < 1:
                raise TypeError("prior must be like ('normal', 0, 1) etc.")
            kind = str(p[0])
            args = tuple(p[1:])
            m[k] = replace(m[k], prior=(kind, args))
        return replace(self, params=tuple(m[n] for n in self.param_names))
    
    def wrap(self, **wrapped: bool) -> "Model":
        """Mark parameters as circular/periodic (for samplers like UltraNest)."""
        m = {p.name: p for p in self.params}
        for k, v in wrapped.items():
            if k not in m:
                raise KeyError(k)
            m[k] = replace(m[k], wrapped=bool(v))
        return replace(self, params=tuple(m[n] for n in self.param_names))


    def derive(
        self, name: str, func: Callable[[Mapping[str, float]], float], *, doc: str = ""
    ) -> "Model":
        """Return a new Model with a post-fit derived parameter."""
        if name in self.param_names:
            raise ValueError(
                f"Derived name {name!r} conflicts with an existing parameter."
            )
        return replace(
            self, derived=self.derived + (DerivedSpec(name=name, func=func, doc=doc),)
        )

    def with_guesser(self, fn: Guesser) -> "Model":
        """Return a new Model with `fn` appended to the guesser list."""
        return replace(self, guessers=self.guessers + (fn,))

    def seed(
        self,
        x: Any,
        data: Any = None,
        *,
        seed_override: Optional[Mapping[str, float]] = None,
        data_format: Optional[str] = None,
        parallel: Optional[Literal[None, "auto"]] = None,
        strict: bool = False,
        backend_options: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> ParamsView:
        """Compute parameter seeds without running the optimiser.

        This is equivalent to a `fit(..., optimise=False)` call, but returns the
        seed parameter view directly.

        Data inference rules (non-strict mode warns on ambiguity):
        - tuples are payloads: (y, sigma), (y, lo, hi), (n, k), (alpha, beta)
        - ragged batches require list inputs for both x and data
        - lists otherwise are treated as array data when possible
        - pass strict=True to raise on ambiguous inputs
        """
        if isinstance(x, FitData):
            if data is not None:
                raise TypeError("If x is FitData, do not also pass data=...")
            fd = x
            x = fd.x
            data = fd.data
            if data_format is None:
                data_format = fd.data_format

        # Pick a sensible backend label (even though optimise=False won't call it).
        if data_format in ("binomial", "beta"):
            seed_backend: Literal[
                "scipy.curve_fit", "scipy.minimize", "ultranest"
            ] = "scipy.minimize"
        else:
            seed_backend = "scipy.curve_fit"

        run = self.fit(
            x,
            data,
            backend=seed_backend,
            data_format=data_format,
            parallel=parallel,
            seed_override=seed_override,
            strict=strict,
            optimise=False,
            backend_options=backend_options,
            rng=rng,
        )
        res = run.results
        if res.seed is not None:
            return res.seed
        return res.params

    # ---- fitting ----
    def fit(
        self,
        x: Any,
        data: Any = None,
        *,
        backend: str | Sequence[str] = "scipy.curve_fit",
        data_format: Optional[str] = None,
        parallel: Optional[Literal[None, "auto"]] = None,
        seed_override: Optional[Mapping[str, float]] = None,
        strict: bool = False,
        optimise: bool = True,
        backend_options: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Run:
        """Fit the model to data and return a Run.

        Data inference rules (non-strict mode warns on ambiguity):
        - tuples are payloads: (y, sigma), (y, lo, hi), (n, k), (alpha, beta)
        - ragged batches require list inputs for both x and data
        - lists otherwise are treated as array data when possible
        - pass strict=True to raise on ambiguous inputs

        Backend notes:
        - scipy.minimize supports cov_method="auto" (use hess_inv if available, else numdiff)
        - cov_method="numdiff" is more robust for non-Gaussian likelihoods
        - scipy.differential_evolution is a global optimiser (requires finite bounds)
        - backend="auto" tries a fast local solver first, then falls back to a global solver
        - backend=("a", "b", ...) runs a backend pipeline, passing results forward as seeds
        """
        from .params import ParamView

        meta: Optional[Dict[str, Any]] = None
        if isinstance(x, FitData):
            if data is not None:
                raise TypeError("If x is FitData, do not also pass data=...")
            fd = x
            x = fd.x
            data = fd.data
            if data_format is None:
                data_format = fd.data_format
            elif fd.data_format is not None and data_format != fd.data_format:
                raise ValueError(
                    f"Conflicting data_format: FitData has {fd.data_format!r} but call passed {data_format!r}."
                )

            meta = dict(fd.meta)
            if fd.x_label is not None:
                meta["x_label"] = fd.x_label
            if fd.y_label is not None:
                meta["y_label"] = fd.y_label
            if fd.label is not None:
                meta["label"] = fd.label
        elif isinstance(data, ParamView):
            # Convenience: allow fits-of-fits without manually repackaging
            # ParamView(value, stderr) into (y, sigma).
            if data_format is None:
                data_format = "normal"
            elif data_format != "normal":
                raise ValueError("ParamView payloads are only supported for data_format='normal'.")
            yerr = data.stderr
            if yerr is not None:
                try:
                    e = np.asarray(yerr, dtype=object)
                    if e.dtype == object and any(v is None for v in e.ravel().tolist()):
                        yerr = None
                    else:
                        ef = np.asarray(yerr, dtype=float)
                        if not np.all(np.isfinite(ef)):
                            yerr = None
                except Exception:
                    yerr = None
            data = data.value if yerr is None else (data.value, yerr)
        elif data is None:
            raise TypeError("fit() missing required argument: data")

        if rng is None:
            rng = np.random.default_rng()
        backend_options = dict(backend_options or {})

        if data_format is None:
            data_format = "normal"

        # ---- backend normalization -----------------------------------------
        backend_mode = "single"
        backend_steps: list[str] = []

        if isinstance(backend, str):
            backend_name = backend
            if backend_name == "auto":
                backend_mode = "auto"
            else:
                backend_steps = [backend_name]
        elif isinstance(backend, (tuple, list)):
            backend_mode = "pipeline"
            backend_steps = [str(b) for b in backend]
            if not backend_steps:
                raise ValueError("backend pipeline cannot be empty.")
        else:
            raise TypeError("backend must be a string or a sequence of strings.")

        # Automatic backend switching for non-Gaussian (old default behaviour).
        if backend_mode == "single" and data_format in ("binomial", "beta") and backend_steps == ["scipy.curve_fit"]:
            backend_steps = ["scipy.minimize"]

        # Enforce current capability boundaries clearly.
        allowed_normal = {"scipy.curve_fit", "scipy.differential_evolution", "ultranest"}
        allowed_binomial = {"scipy.minimize", "ultranest"}
        allowed_beta = {"scipy.minimize"}

        if backend_mode == "auto":
            if data_format == "beta":
                backend_steps = ["scipy.minimize"]
                backend_mode = "single"
            elif data_format == "binomial":
                backend_steps = ["scipy.minimize"]
                backend_mode = "single"
            else:  # normal
                # Auto mode is handled per-dataset below.
                backend_steps = []

        if data_format == "normal":
            if backend_steps and any(b not in allowed_normal for b in backend_steps):
                raise NotImplementedError(
                    "v1: normal data currently supports backend in "
                    f"{tuple(sorted(allowed_normal))}."
                )
        elif data_format == "binomial":
            if backend_steps and any(b not in allowed_binomial for b in backend_steps):
                raise NotImplementedError(
                    "v1: binomial data currently requires backend in "
                    f"{tuple(sorted(allowed_binomial))}."
                )
        elif data_format == "beta":
            if backend_steps and any(b not in allowed_beta for b in backend_steps):
                raise NotImplementedError(
                    "v1: beta data currently requires backend in "
                    f"{tuple(sorted(allowed_beta))}."
                )
        else:
            raise NotImplementedError(f"Unknown data_format {data_format!r}.")

        datasets, batch_shape = prepare_datasets(x, data, data_format, strict)

        free_names, fixed_map = _free_and_fixed(self.params)
        # used for Results/Run labels; may differ from the user's backend spec.
        backend_label = backend if isinstance(backend, str) else "pipeline"

        # Allocate storage in flattened-batch form
        B = len(datasets)
        values = {n: np.empty((B,), dtype=float) for n in self.param_names}
        seed_values = {n: np.empty((B,), dtype=float) for n in self.param_names}
        errors = {n: np.empty((B,), dtype=object) for n in self.param_names}
        for n in self.param_names:
            errors[n][:] = None

        covs = np.empty((B,), dtype=object)
        covs[:] = None
        successes = np.empty((B,), dtype=bool)
        messages = np.empty((B,), dtype=object)
        stats_list = np.empty((B,), dtype=object)
        stats_list[:] = None

        for i, ds in enumerate(datasets):
            p0_map = _compute_seed_map(
                self,
                ds.x,
                np.asarray(ds.y_for_seed, dtype=float),
                free_names,
                rng=rng,
                seed_override=seed_override,
            )
            p0 = np.asarray([float(p0_map[n]) for n in free_names], dtype=float)
            bounds = _bounds_for_free(self.params, free_names)

            # Record seed used
            for j, n in enumerate(free_names):
                seed_values[n][i] = float(p0[j])
            for n, fv in fixed_map.items():
                seed_values[n][i] = float(fv)

            if not optimise:
                successes[i] = True
                messages[i] = "optimise=False (seed only)"
                stats_list[i] = {}
                for j, n in enumerate(free_names):
                    values[n][i] = float(p0[j])
                for n, fv in fixed_map.items():
                    values[n][i] = float(fv)
                covs[i] = None
                continue

            r_use, used_backend_name = _run_backend_plan(
                model=self,
                dataset=ds,
                free_names=free_names,
                fixed_map=fixed_map,
                p0=p0,
                bounds=bounds,
                options=backend_options,
                mode=backend_mode,
                steps=backend_steps,
                data_format=data_format,
            )

            successes[i] = bool(r_use.success)
            messages[i] = str(r_use.message)
            stats_list[i] = dict(r_use.stats or {})

            theta = np.asarray(r_use.theta, dtype=float)
            for j, n in enumerate(free_names):
                values[n][i] = float(theta[j])
            for n, fv in fixed_map.items():
                values[n][i] = float(fv)

            covs[i] = r_use.cov
            if r_use.cov is not None:
                pcov = np.asarray(r_use.cov, dtype=float)
                perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
                for j, n in enumerate(free_names):
                    errors[n][i] = float(perr[j])
            # Keep a useful backend label per-dataset (best backend in a pipeline).
            if isinstance(stats_list[i], dict):
                stats_list[i].setdefault("backend", used_backend_name)

        # Build param views + cov
        if batch_shape == ():
            items: Dict[str, ParamView] = {}
            seed_items: Dict[str, ParamView] = {}
            cov0 = covs[0] if covs.shape[0] else None
            cov = None if cov0 is None else np.asarray(cov0, dtype=float)
            values_map: Dict[str, Any] = {}
            stderr_map: Dict[str, Any] = {}
            ctx = _UncContext(
                values=values_map,
                stderrs=stderr_map,
                cov=cov,
                free_names=tuple(free_names),
            )
            for n in self.param_names:
                spec = _spec_by_name(self.params, n)
                v = float(values[n][0])
                sv = float(seed_values[n][0])
                e = errors[n][0]
                stderr_val = None if (spec.fixed or cov0 is None) else float(e)
                values_map[n] = v
                stderr_map[n] = stderr_val
                items[n] = ParamView(
                    name=n,
                    value=v,
                    stderr=stderr_val,
                    fixed=spec.fixed,
                    bounds=spec.bounds,
                    derived=False,
                    _context=ctx,
                )
                seed_items[n] = ParamView(
                    name=n,
                    value=sv,
                    stderr=None,
                    fixed=spec.fixed,
                    bounds=spec.bounds,
                    derived=False,
                )
            results = Results(
                batch_shape=(),
                params=ParamsView(items, _context=ctx),
                seed=ParamsView(seed_items),
                cov=cov,
                backend=str(stats_list[0].get("backend", backend_label))
                if isinstance(stats_list[0], dict)
                else str(backend_label),
                stats=(dict(stats_list[0]) if stats_list[0] is not None else {}),
            )
        else:
            items = {}
            seed_items = {}
            cov_obj = unflatten_batch(np.asarray(covs, dtype=object), batch_shape)
            values_map = {}
            stderr_map = {}
            ctx = _UncContext(
                values=values_map,
                stderrs=stderr_map,
                cov=cov_obj,
                free_names=tuple(free_names),
            )
            for n in self.param_names:
                spec = _spec_by_name(self.params, n)
                v = unflatten_batch(values[n], batch_shape)
                e_obj = unflatten_batch(np.asarray(errors[n], dtype=object), batch_shape)
                e = _maybe_float_stderr(e_obj)
                sv = unflatten_batch(seed_values[n], batch_shape)
                stderr_val = None if spec.fixed else e
                values_map[n] = v
                stderr_map[n] = stderr_val
                items[n] = ParamView(
                    name=n,
                    value=v,
                    stderr=stderr_val,
                    fixed=spec.fixed,
                    bounds=spec.bounds,
                    derived=False,
                    _context=ctx,
                )
                seed_items[n] = ParamView(
                    name=n,
                    value=sv,
                    stderr=None,
                    fixed=spec.fixed,
                    bounds=spec.bounds,
                    derived=False,
                )
            stats: Dict[str, Any] = {
                "per_batch": unflatten_batch(np.asarray(stats_list, dtype=object), batch_shape)
            }
            backends = []
            for s in np.asarray(stats_list, dtype=object).ravel().tolist():
                if isinstance(s, dict) and isinstance(s.get("backend"), str):
                    backends.append(str(s["backend"]))
            backend_used = str(backend_label)
            if backends:
                uniq = sorted(set(backends))
                backend_used = uniq[0] if len(uniq) == 1 else "mixed"
            results = Results(
                batch_shape=batch_shape,
                params=ParamsView(items, _context=ctx),
                seed=ParamsView(seed_items),
                cov=cov_obj,  # object array, per-batch cov matrices (or None)
                backend=backend_used,
                stats=stats,
            )

        # Post-fit derived params (v1): depend only on fitted params
        if self.derived:
            try:
                import uncertainties  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                uncertainties = None

            def _can_use_uncertainties() -> bool:
                if uncertainties is None:
                    return False
                for spec in self.params:
                    if spec.fixed:
                        continue
                    pv = results.params[spec.name]
                    err = pv.stderr
                    if err is None:
                        return False
                    a = np.asarray(err)
                    if a.dtype == object:
                        if any(v is None for v in a.ravel().tolist()):
                            return False
                        a = a.astype(float)
                    if not np.all(np.isfinite(a)):
                        return False
                return True

            use_uncertainties = _can_use_uncertainties()
            if results.batch_shape == ():
                base = {n: float(results.params[n].value) for n in self.param_names}
                base_unc = None
                if use_uncertainties:
                    base_unc = {}
                    for spec in self.params:
                        pv = results.params[spec.name]
                        if spec.fixed:
                            base_unc[spec.name] = float(pv.value)
                        else:
                            try:
                                base_unc[spec.name] = pv.u
                            except Exception:
                                base_unc = None
                                use_uncertainties = False
                                break
                extra = {}
                for d in self.derived:
                    if use_uncertainties and base_unc is not None:
                        dv = d.func(base_unc)
                        if hasattr(dv, "nominal_value") and hasattr(dv, "std_dev"):
                            dv_val = float(dv.nominal_value)
                            dv_err = float(dv.std_dev)
                        else:
                            dv_val = float(dv)
                            dv_err = None
                    else:
                        dv_val = float(d.func(base))
                        dv_err = None
                    extra[d.name] = ParamView(
                        name=d.name,
                        value=dv_val,
                        stderr=dv_err,
                        fixed=True,
                        derived=True,
                    )
                results = replace(
                    results,
                    params=ParamsView(
                        {**dict(results.params.items()), **extra},
                        _context=getattr(results.params, "_context", None),
                    ),
                )
            else:
                # flatten again
                batch_size = prod(results.batch_shape)
                flat_vals = {
                    n: np.asarray(results.params[n].value).reshape((batch_size,))
                    for n in self.param_names
                }
                flat_unc = None
                if use_uncertainties:
                    flat_unc = {}
                    for spec in self.params:
                        pv = results.params[spec.name]
                        if spec.fixed:
                            flat_unc[spec.name] = np.asarray(pv.value).reshape((batch_size,))
                        else:
                            try:
                                uarr = pv.u
                            except Exception:
                                flat_unc = None
                                use_uncertainties = False
                                break
                            flat_unc[spec.name] = np.asarray(uarr, dtype=object).reshape(
                                (batch_size,)
                            )
                extra_items = {}
                for d in self.derived:
                    out = np.empty((batch_size,), dtype=float)
                    err_out = None
                    if use_uncertainties:
                        err_out = np.empty((batch_size,), dtype=object)
                    for i in range(batch_size):
                        if use_uncertainties and flat_unc is not None:
                            base = {n: flat_unc[n][i] for n in self.param_names}
                            dv = d.func(base)
                            if hasattr(dv, "nominal_value") and hasattr(dv, "std_dev"):
                                out[i] = float(dv.nominal_value)
                                err_out[i] = float(dv.std_dev)
                            else:
                                out[i] = float(dv)
                                err_out[i] = None
                        else:
                            base = {n: float(flat_vals[n][i]) for n in self.param_names}
                            out[i] = float(d.func(base))
                    stderr_val = None
                    if use_uncertainties and err_out is not None:
                        if any(v is None for v in err_out.tolist()):
                            stderr_val = unflatten_batch(err_out, results.batch_shape)
                        else:
                            stderr_val = unflatten_batch(
                                err_out.astype(float), results.batch_shape
                            )
                    extra_items[d.name] = ParamView(
                        name=d.name,
                        value=unflatten_batch(out, results.batch_shape),
                        stderr=stderr_val,
                        fixed=True,
                        derived=True,
                    )
                results = replace(
                    results,
                    params=ParamsView(
                        {**dict(results.params.items()), **extra_items},
                        _context=getattr(results.params, "_context", None),
                    ),
                )

        run = Run(
            model=self,
            results=results,
            backend=str(getattr(results, "backend", backend_label)),
            data_format=data_format,
            data=(
                {"x": x, "data": data}
                if not meta
                else {"x": x, "data": data, "meta": meta}
            ),
            success=(
                bool(successes[0])
                if batch_shape == ()
                else unflatten_batch(np.asarray(successes, dtype=bool), batch_shape)
            ),
            message=(
                str(messages[0])
                if batch_shape == ()
                else unflatten_batch(np.asarray(messages, dtype=object), batch_shape)
            ),
        )

        return run


def _spec_by_name(params: Tuple[ParameterSpec, ...], name: str) -> ParameterSpec:
    """Return the ParameterSpec with a matching name."""
    for p in params:
        if p.name == name:
            return p
    raise KeyError(name)


def _free_and_fixed(
    params: Tuple[ParameterSpec, ...]
) -> Tuple[List[str], Dict[str, float]]:
    """Split parameters into free names and fixed name->value mapping."""
    free: List[str] = []
    fixed: Dict[str, float] = {}
    for p in params:
        if p.fixed:
            if p.fixed_value is None:
                raise ValueError(f"Parameter {p.name} is fixed but has no fixed_value.")
            fixed[p.name] = float(p.fixed_value)
        else:
            free.append(p.name)
    return free, fixed


def _bounds_for_free(
    params: Tuple[ParameterSpec, ...], free_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) arrays of bounds for free parameters."""
    pmap = {p.name: p for p in params}
    lo: List[float] = []
    hi: List[float] = []
    for n in free_names:
        b = pmap[n].bounds
        if b is None:
            lo.append(-np.inf)
            hi.append(np.inf)
        else:
            lo.append(-np.inf if b[0] is None else float(b[0]))
            hi.append(np.inf if b[1] is None else float(b[1]))
    return (np.array(lo, dtype=float), np.array(hi, dtype=float))


def _all_finite_bounds(bounds: Tuple[np.ndarray, np.ndarray]) -> bool:
    """Return True if every parameter has finite (lo, hi) bounds."""
    lo, hi = bounds
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return bool(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo))


def _maybe_float_stderr(stderr: Any) -> Any:
    """Cast object stderr arrays to float when fully-populated.

    v1 stores batched stderrs as dtype=object so we can represent per-batch missing
    entries as None. When all entries are present, returning a float array is more
    ergonomic (plots, downstream fits-of-fits, etc).
    """
    if not isinstance(stderr, np.ndarray) or stderr.dtype != object:
        return stderr
    flat = stderr.ravel().tolist()
    if any(v is None for v in flat):
        return stderr
    try:
        out = stderr.astype(float)
    except Exception:
        return stderr
    return out


def _run_backend_plan(
    *,
    model: Any,
    dataset: Any,
    free_names: List[str],
    fixed_map: Dict[str, float],
    p0: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    options: Dict[str, Any],
    mode: str,
    steps: Sequence[str],
    data_format: str,
) -> Tuple[Any, str]:
    """Run a single backend, a backend pipeline, or auto fallback for one dataset."""
    from .backends.common import BackendResult

    def _record(backend_name: str, r: BackendResult) -> Dict[str, Any]:
        return {
            "backend": backend_name,
            "success": bool(r.success),
            "message": str(r.message),
            "stats": dict(r.stats or {}),
        }

    if mode == "single":
        if len(steps) != 1:
            raise ValueError("single backend mode requires exactly one step.")
        name = str(steps[0])
        r = get_backend(name).fit_one(
            model=model,
            dataset=dataset,
            free_names=free_names,
            fixed_map=fixed_map,
            p0=p0,
            bounds=bounds,
            options=options,
        )
        stats = dict(r.stats or {})
        stats["backend"] = name
        return BackendResult(
            theta=np.asarray(r.theta, dtype=float),
            cov=r.cov,
            success=bool(r.success),
            message=str(r.message),
            stats=stats,
        ), name

    if mode == "pipeline":
        if not steps:
            raise ValueError("pipeline backend mode requires at least one step.")

        pipeline: list[Dict[str, Any]] = []
        best: Optional[BackendResult] = None
        best_name: str = str(steps[0])
        current_p0 = np.asarray(p0, dtype=float)
        last: Optional[BackendResult] = None
        last_name: str = best_name

        for name in steps:
            name = str(name)
            last_name = name
            r = get_backend(name).fit_one(
                model=model,
                dataset=dataset,
                free_names=free_names,
                fixed_map=fixed_map,
                p0=current_p0,
                bounds=bounds,
                options=options,
            )
            last = r
            pipeline.append(_record(name, r))
            if bool(r.success):
                best = r
                best_name = name
            try:
                theta = np.asarray(r.theta, dtype=float)
                if theta.shape == current_p0.shape and np.all(np.isfinite(theta)):
                    current_p0 = theta
            except Exception:
                pass

        use = best if best is not None else last
        use_name = best_name if best is not None else last_name
        if use is None:
            raise RuntimeError("pipeline produced no result (this should be impossible).")

        stats = dict(use.stats or {})
        stats["backend"] = use_name
        stats["pipeline_mode"] = "pipeline"
        stats["pipeline_best_backend"] = use_name
        stats["pipeline"] = pipeline

        return BackendResult(
            theta=np.asarray(use.theta, dtype=float),
            cov=use.cov,
            success=bool(use.success),
            message=str(use.message),
            stats=stats,
        ), use_name

    if mode != "auto":
        raise ValueError(f"Unknown backend mode: {mode!r}")

    # ---- auto fallback ------------------------------------------------------
    if data_format != "normal":
        raise ValueError("auto backend mode is only valid for data_format='normal'.")

    pipeline = []
    best: Optional[BackendResult] = None
    best_name: str = "scipy.curve_fit"
    last: BackendResult
    last_name: str = "scipy.curve_fit"

    # Try fast local least squares first.
    r_cf = get_backend("scipy.curve_fit").fit_one(
        model=model,
        dataset=dataset,
        free_names=free_names,
        fixed_map=fixed_map,
        p0=p0,
        bounds=bounds,
        options=options,
    )
    pipeline.append(_record("scipy.curve_fit", r_cf))
    last = r_cf
    if bool(r_cf.success):
        best = r_cf
        best_name = "scipy.curve_fit"

    # If that fails and bounds are finite, fall back to DE and (optionally) re-run curve_fit.
    if best is None and _all_finite_bounds(bounds):
        r_de = get_backend("scipy.differential_evolution").fit_one(
            model=model,
            dataset=dataset,
            free_names=free_names,
            fixed_map=fixed_map,
            p0=p0,
            bounds=bounds,
            options=options,
        )
        pipeline.append(_record("scipy.differential_evolution", r_de))
        last = r_de
        last_name = "scipy.differential_evolution"
        # Treat DE as the best available fallback even if it hits maxiter.
        best = r_de
        best_name = "scipy.differential_evolution"

        # Refine with curve_fit from the DE result (often improves covariance).
        try:
            p0_refine = np.asarray(r_de.theta, dtype=float)
        except Exception:
            p0_refine = np.asarray(p0, dtype=float)
        r_cf2 = get_backend("scipy.curve_fit").fit_one(
            model=model,
            dataset=dataset,
            free_names=free_names,
            fixed_map=fixed_map,
            p0=p0_refine,
            bounds=bounds,
            options=options,
        )
        pipeline.append(_record("scipy.curve_fit", r_cf2))
        last = r_cf2
        last_name = "scipy.curve_fit"
        if bool(r_cf2.success):
            best = r_cf2
            best_name = "scipy.curve_fit"

    use = best if best is not None else last
    stats = dict(use.stats or {})
    stats["backend"] = best_name if best is not None else last_name
    stats["pipeline_mode"] = "auto"
    stats["pipeline_best_backend"] = stats["backend"]
    stats["pipeline"] = pipeline

    return BackendResult(
        theta=np.asarray(use.theta, dtype=float),
        cov=use.cov,
        success=bool(use.success),
        message=str(use.message),
        stats=stats,
    ), str(stats["backend"])


def _default_seed_engine(
    model: Model,
    x: Any,
    y: np.ndarray,
    free_names: Sequence[str],
) -> Dict[str, float]:
    """Built-in seeding strategy.

    Order (before per-call seed overlay):
    1) model-level strong .guess(...)
    2) user guessers (cannot override strong guesses)
    3) model-level weak_guess (used only if still missing)
    """
    pmap = {p.name: p for p in model.params}
    seeds: Dict[str, float] = {}
    strong: set[str] = set()

    # 1) strong guesses
    for n in free_names:
        spec = pmap[n]
        if spec.guess is not None:
            seeds[n] = float(spec.guess)
            strong.add(n)

    # 2) user guessers (only fill if not strong, and not already set)
    if model.guessers:
        gs = GuessState()
        for fn in model.guessers:
            fn(x, y, gs)
        for n, v in gs.to_dict().items():
            if n in free_names and n not in strong and n not in seeds:
                seeds[n] = float(v)

    # 3) weak guesses (only fill if still missing)
    for n in free_names:
        if n in seeds:
            continue
        spec = pmap[n]
        if getattr(spec, "weak_guess", None) is not None:
            seeds[n] = float(spec.weak_guess)  # type: ignore[union-attr]

    return seeds


def _compute_seed_map(
    model: Model,
    x: Any,
    y: np.ndarray,
    free_names: List[str],
    rng: np.random.Generator,
    seed_override: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute initial seeds for the given dataset.

    Precedence per free parameter:

      1) per-call `seed_override` (fit(..., seed=...))
      2) strong guess via model.guess(...)
      3) model guessers (with_guesser)
      4) weak guess via model.weak_guess(...) (and function defaults)
      5) midpoint of finite bounds (with a warning)
      6) else: raise ValueError
    """
    free_names = list(free_names)
    pmap = {p.name: p for p in model.params}

    # 2+3: model guesses + guessers
    seeds = _default_seed_engine(model, x, y, free_names)

    # keep only known free names
    seeds = {n: float(v) for n, v in seeds.items() if n in free_names}

    # 1) overlay per-call seed (highest precedence)
    if seed_override is not None:
        for n, v in seed_override.items():
            if n in free_names:
                seeds[n] = float(v)

    # 4) fill missing from bounds midpoint if possible
    filled_from_bounds: List[str] = []
    for n in free_names:
        if n in seeds:
            continue
        spec = pmap[n]
        b = spec.bounds
        if b is not None:
            lo, hi = b
            if (
                lo is not None
                and hi is not None
                and np.isfinite(lo)
                and np.isfinite(hi)
            ):
                seeds[n] = float(0.5 * (float(lo) + float(hi)))
                filled_from_bounds.append(n)

    if filled_from_bounds:
        warn(
            "Using mid-point of bounds as seed for parameters: "
            + ", ".join(filled_from_bounds),
            UserWarning,
        )

    # Clip any out-of-bounds seeds into the feasible range (saves lots of backend pain).
    clipped: List[str] = []
    for n in free_names:
        if n not in seeds:
            continue
        b = pmap[n].bounds
        if b is None:
            continue
        lo, hi = b
        v = float(seeds[n])
        v0 = v
        if lo is not None and np.isfinite(lo):
            v = max(v, float(lo))
        if hi is not None and np.isfinite(hi):
            v = min(v, float(hi))
        if v != v0:
            seeds[n] = v
            clipped.append(n)
    if clipped:
        warn("Clipped seed values into bounds for: " + ", ".join(clipped), UserWarning)


    # 5) final check
    missing = [n for n in free_names if n not in seeds]
    if missing:
        raise ValueError(
            "Could not determine initial seeds for parameters: "
            + ", ".join(missing)
            + ". Provide seed=..., model.guess(...), a guesser, or finite bounds."
        )

    return seeds
