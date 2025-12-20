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
        m = {p.name: p for p in self.params}
        for k, v in fixed.items():
            if k not in m:
                raise KeyError(k)
            m[k] = replace(m[k], fixed=True, fixed_value=float(v))
        return replace(self, params=tuple(m[n] for n in self.param_names))

    def bound(self, **bounds: Tuple[Optional[float], Optional[float]]) -> "Model":
        m = {p.name: p for p in self.params}
        for k, b in bounds.items():
            if k not in m:
                raise KeyError(k)
            lo, hi = b
            m[k] = replace(m[k], bounds=(lo, hi))
        return replace(self, params=tuple(m[n] for n in self.param_names))

    def guess(self, **guesses: float) -> "Model":
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
        data: Any,
        *,
        seed_override: Optional[Mapping[str, float]] = None,
        data_format: Optional[str] = None,
        parallel: Optional[Literal[None, "auto"]] = None,
        backend_options: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> ParamsView:
        """Compute parameter seeds without running the optimiser.

        This is equivalent to a `fit(..., optimise=False)` call, but returns the
        seed parameter view directly.
        """

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
        data: Any,
        *,
        backend: Literal[
            "scipy.curve_fit", "scipy.minimize", "ultranest"
        ] = "scipy.curve_fit",
        data_format: Optional[str] = None,
        parallel: Optional[Literal[None, "auto"]] = None,
        seed_override: Optional[Mapping[str, float]] = None,
        optimise: bool = True,
        backend_options: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Run:
        if rng is None:
            rng = np.random.default_rng()
        backend_options = dict(backend_options or {})

        if data_format is None:
            data_format = "normal"

        # Automatic backend switching for non-Gaussian (old default behaviour).
        if data_format in ("binomial", "beta") and backend == "scipy.curve_fit":
            backend = "scipy.minimize"

        # Enforce current capability boundaries clearly.
        if data_format in ("binomial", "beta") and backend != "scipy.minimize":
            raise NotImplementedError(
                "v1: non-Gaussian data currently requires backend='scipy.minimize'."
            )

        datasets, batch_shape = prepare_datasets(x, data, data_format)

        free_names, fixed_map = _free_and_fixed(self.params)

        backend_impl = get_backend(backend)

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

            r = backend_impl.fit_one(
                model=self,
                dataset=ds,
                free_names=free_names,
                fixed_map=fixed_map,
                p0=p0,
                bounds=bounds,
                options=backend_options,
            )

            successes[i] = bool(r.success)
            messages[i] = str(r.message)
            stats_list[i] = dict(r.stats or {})

            theta = np.asarray(r.theta, dtype=float)
            for j, n in enumerate(free_names):
                values[n][i] = float(theta[j])
            for n, fv in fixed_map.items():
                values[n][i] = float(fv)

            covs[i] = r.cov
            if r.cov is not None:
                pcov = np.asarray(r.cov, dtype=float)
                perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
                for j, n in enumerate(free_names):
                    errors[n][i] = float(perr[j])

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
                backend=backend,
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
                e = unflatten_batch(np.asarray(errors[n], dtype=object), batch_shape)
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
            results = Results(
                batch_shape=batch_shape,
                params=ParamsView(items, _context=ctx),
                seed=ParamsView(seed_items),
                cov=cov_obj,  # object array, per-batch cov matrices (or None)
                backend=backend,
                stats=stats,
            )

        # Post-fit derived params (v1): depend only on fitted params
        if self.derived:
            if results.batch_shape == ():
                base = {n: float(results.params[n].value) for n in self.param_names}
                extra = {}
                for d in self.derived:
                    dv = float(d.func(base))
                    extra[d.name] = ParamView(
                        name=d.name, value=dv, stderr=None, fixed=True, derived=True
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
                extra_items = {}
                for d in self.derived:
                    out = np.empty((batch_size,), dtype=float)
                    for i in range(batch_size):
                        base = {n: float(flat_vals[n][i]) for n in self.param_names}
                        out[i] = float(d.func(base))
                    extra_items[d.name] = ParamView(
                        name=d.name,
                        value=unflatten_batch(out, results.batch_shape),
                        stderr=None,
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
            backend=backend,
            data_format=data_format,
            data={"x": x, "data": data},
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
    for p in params:
        if p.name == name:
            return p
    raise KeyError(name)


def _free_and_fixed(
    params: Tuple[ParameterSpec, ...]
) -> Tuple[List[str], Dict[str, float]]:
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
