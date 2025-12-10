from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .backends.scipy_curve_fit import fit_curve_fit
from .params import DerivedSpec, GuessState, ParameterSpec, ParamView, ParamsView
from .run import Results, Run
from .util import flatten_batch, infer_param_names, is_ragged_batch, prod, unflatten_batch


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
    autoguess_names: Tuple[str, ...] = ()
    meta: Dict[str, Any] = None

    # ---- constructor ----
    @staticmethod
    def from_function(func: Callable[..., Any], *, name: Optional[str] = None) -> "Model":
        names = infer_param_names(func)
        specs = tuple(ParameterSpec(name=n) for n in names)
        return Model(name=name or getattr(func, "__name__", "model"), func=func, param_names=names, params=specs)

    # ---- evaluation ----
    def eval(self, x: Any, *, params: Optional[Mapping[str, Any]] = None, **kwargs) -> Any:
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

    def autoguess(self, *names: str) -> "Model":
        for n in names:
            if n not in self.param_names:
                raise KeyError(n)
        merged = tuple(dict.fromkeys(self.autoguess_names + tuple(names)).keys())
        return replace(self, autoguess_names=merged)

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

    def derive(self, name: str, func: Callable[[Mapping[str, float]], float], *, doc: str = "") -> "Model":
        if name in self.param_names:
            raise ValueError(f"Derived name {name!r} conflicts with an existing parameter.")
        return replace(self, derived=self.derived + (DerivedSpec(name=name, func=func, doc=doc),))

    # ---- guesser registration (side-effect, v1 ergonomic) ----
    def guesser(self, fn: Optional[Guesser] = None):
        """Decorator to register a custom guesser.

        NOTE: This mutates `self` by appending the guesser, and returns the function.
        This supports the ergonomic pattern:

            @model.guesser
            def g(x, y, gs): ...

        Builder methods remain pure (return new models).
        """
        def decorator(f: Guesser) -> Guesser:
            self.guessers = self.guessers + (f,)
            return f

        return decorator(fn) if fn is not None else decorator

    # ---- fitting ----
    def fit(
        self,
        *,
        x: Any,
        y: Any,
        backend: Literal["scipy.curve_fit", "scipy.minimize", "ultranest"] = "scipy.curve_fit",
        data_format: Optional[str] = None,
        parallel: Optional[Literal[None, "auto"]] = None,
        return_run: bool = False,
        backend_options: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[Results, Run]:
        if rng is None:
            rng = np.random.default_rng()
        backend_options = backend_options or {}

        # v1: only Gaussian inference (data_format None or 'normal')
        if data_format not in (None, "normal"):
            raise NotImplementedError("v1 MVP supports only Gaussian default data inference.")

        datasets: List[Dict[str, Any]] = []
        batch_shape: Tuple[int, ...] = ()

        if is_ragged_batch(x, y):
            for xi, yi in zip(x, y):
                yobs, sigma = _infer_gaussian_payload(yi)
                datasets.append({"x": xi, "y": yobs, "sigma": sigma})
            batch_shape = (len(datasets),)
        else:
            yobs, sigma = _infer_gaussian_payload(y)
            yobs = np.asarray(yobs)
            if yobs.ndim == 1:
                datasets.append({"x": x, "y": yobs, "sigma": sigma})
                batch_shape = ()
            else:
                yflat, batch_shape = flatten_batch(yobs)
                # broadcast sigma if needed
                if sigma is None:
                    sflat = [None] * yflat.shape[0]
                else:
                    sarr = np.asarray(sigma)
                    if sarr.shape == ():
                        sflat = [float(sarr)] * yflat.shape[0]
                    else:
                        sb = np.broadcast_to(sarr, yobs.shape)
                        sb_flat, _ = flatten_batch(sb)
                        sflat = [sb_flat[i] for i in range(sb_flat.shape[0])]
                for i in range(yflat.shape[0]):
                    datasets.append({"x": x, "y": yflat[i], "sigma": sflat[i]})

        free_names, fixed_map = _free_and_fixed(self.params)

        # allocate storage (flattened batch)
        B = len(datasets)
        values = {n: np.empty((B,), dtype=float) for n in self.param_names}
        errors = {n: np.full((B,), np.nan, dtype=float) for n in self.param_names}
        covs: List[Optional[np.ndarray]] = []

        meta: Dict[str, Any] = {
            "free_param_names": list(free_names),
            "success": [],
            "message": [],
        }

        for i, ds in enumerate(datasets):
            xi = ds["x"]
            yi = np.asarray(ds["y"])
            si = ds["sigma"]
            si_arr = None if si is None else np.asarray(si, dtype=float)

            p0_map = _initial_guess(self, xi, yi, free_names, rng=rng)
            p0 = np.array([float(p0_map[n]) for n in free_names], dtype=float)
            bounds = _bounds_for_free(self.params, free_names)

            if backend != "scipy.curve_fit":
                raise NotImplementedError("v1 MVP implements only backend='scipy.curve_fit'.")

            f_wrapped = _wrap_free_params(self, fixed_map, free_names)
            r = fit_curve_fit(
                f_wrapped, xi, yi,
                sigma=si_arr, p0=p0, bounds=bounds,
                maxfev=backend_options.get("maxfev"),
            )
            meta["success"].append(r.success)
            meta["message"].append(r.message)

            # store free values
            for j, n in enumerate(free_names):
                values[n][i] = r.popt[j]
            # fixed values
            for n, fv in fixed_map.items():
                values[n][i] = float(fv)

            if r.pcov is not None:
                pcov = np.asarray(r.pcov, dtype=float)
                covs.append(pcov)
                perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
                for j, n in enumerate(free_names):
                    errors[n][i] = perr[j]
            else:
                covs.append(None)

        # Build param views + cov
        if batch_shape == ():
            items: Dict[str, ParamView] = {}
            for n in self.param_names:
                spec = _spec_by_name(self.params, n)
                v = float(values[n][0])
                e = float(errors[n][0])
                items[n] = ParamView(
                    name=n,
                    value=v,
                    error=(None if np.isnan(e) else e),
                    fixed=spec.fixed,
                    bounds=spec.bounds,
                    derived=False,
                )
            cov = covs[0] if covs and covs[0] is not None else None
            results = Results(batch_shape=(), params=ParamsView(items), cov=cov, backend=backend, meta=meta)
        else:
            items = {}
            for n in self.param_names:
                spec = _spec_by_name(self.params, n)
                v = unflatten_batch(values[n], batch_shape)
                e = unflatten_batch(errors[n], batch_shape)
                items[n] = ParamView(
                    name=n,
                    value=v,
                    error=e,
                    fixed=spec.fixed,
                    bounds=spec.bounds,
                    derived=False,
                )
            if all(c is not None for c in covs):
                cov = np.stack([c for c in covs], axis=0)
                cov = unflatten_batch(cov, batch_shape)
            else:
                cov = None
            results = Results(batch_shape=batch_shape, params=ParamsView(items), cov=cov, backend=backend, meta=meta)

        # Post-fit derived params (v1): depend only on fitted params
        if self.derived:
            if results.batch_shape == ():
                base = {n: float(results.params[n].value) for n in self.param_names}
                extra = {}
                for d in self.derived:
                    dv = float(d.func(base))
                    extra[d.name] = ParamView(name=d.name, value=dv, error=None, fixed=True, derived=True)
                results = replace(results, params=ParamsView({**dict(results.params.items()), **extra}))
            else:
                # flatten again
                batch_size = prod(results.batch_shape)
                flat_vals = {n: np.asarray(results.params[n].value).reshape((batch_size,)) for n in self.param_names}
                extra_items = {}
                for d in self.derived:
                    out = np.empty((batch_size,), dtype=float)
                    for i in range(batch_size):
                        base = {n: float(flat_vals[n][i]) for n in self.param_names}
                        out[i] = float(d.func(base))
                    extra_items[d.name] = ParamView(
                        name=d.name,
                        value=unflatten_batch(out, results.batch_shape),
                        error=None,
                        fixed=True,
                        derived=True,
                    )
                results = replace(results, params=ParamsView({**dict(results.params.items()), **extra_items}))

        run = Run(
            model=self,
            results=results,
            backend=backend,
            data_format=(data_format or "normal"),
            meta=meta,
            data={"x": x, "y": y},
        )

        return run if return_run else results


def _spec_by_name(params: Tuple[ParameterSpec, ...], name: str) -> ParameterSpec:
    for p in params:
        if p.name == name:
            return p
    raise KeyError(name)


def _free_and_fixed(params: Tuple[ParameterSpec, ...]) -> Tuple[List[str], Dict[str, float]]:
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


def _bounds_for_free(params: Tuple[ParameterSpec, ...], free_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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


def _wrap_free_params(model: Model, fixed_map: Dict[str, float], free_names: List[str]):
    def f(x, *theta_free):
        kwargs = dict(fixed_map)
        for j, n in enumerate(free_names):
            kwargs[n] = theta_free[j]
        return model.eval(x, **kwargs)
    return f


def _infer_gaussian_payload(y: Any):
    # v1 default inference:
    # y -> unweighted
    # (y, yerr) -> symmetric absolute errors
    # (y, yerr_low, yerr_high) -> asymmetric; approximate to mean sigma for curve_fit
    if isinstance(y, (tuple, list)) and len(y) == 2:
        yobs, yerr = y
        return np.asarray(yobs), yerr
    if isinstance(y, (tuple, list)) and len(y) == 3:
        yobs, ylo, yhi = y
        sigma = 0.5 * (np.asarray(ylo) + np.asarray(yhi))
        return np.asarray(yobs), sigma
    return np.asarray(y), None


def _initial_guess(model: Model, x: Any, y: np.ndarray, free_names: List[str], rng: np.random.Generator) -> Dict[str, float]:
    pmap = {p.name: p for p in model.params}
    g: Dict[str, float] = {}

    # manual guesses
    for n in free_names:
        if pmap[n].guess is not None:
            g[n] = float(pmap[n].guess)

    # built-in heuristics for autoguess names (only if unset)
    if model.autoguess_names:
        g2 = _builtin_autoguess(x, y, model.autoguess_names)
        for n, v in g2.items():
            if n in free_names and n not in g:
                g[n] = float(v)

    # user guessers
    if model.guessers:
        gs = GuessState()
        for fn in model.guessers:
            fn(x, y, gs)
        for n, v in gs.to_dict().items():
            if n in free_names and n not in g:
                g[n] = float(v)

    # final fill
    for n in free_names:
        if n not in g:
            g[n] = 0.0
    return g


def _builtin_autoguess(x: Any, y: np.ndarray, names: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y = np.asarray(y)

    # If x is a container, use first entry for slope-type heuristics.
    x0 = x[0] if isinstance(x, (tuple, list)) and len(x) > 0 else x
    x0 = np.asarray(x0)

    for n in names:
        if n in ("b", "c", "offset", "intercept"):
            out[n] = float(np.median(y))
        elif n in ("m", "slope"):
            if x0.size >= 2:
                out[n] = float((y[-1] - y[0]) / (x0[-1] - x0[0] + 1e-12))
            else:
                out[n] = 0.0
        elif n in ("amplitude", "amp", "A"):
            out[n] = float(0.5 * (np.nanmax(y) - np.nanmin(y)))
        elif n in ("mu", "mean", "center"):
            out[n] = float(np.nanmean(x0))
        elif n in ("sigma", "width"):
            out[n] = float(0.1 * (np.nanmax(x0) - np.nanmin(x0) + 1e-12))
        else:
            # unknown name: no-op
            pass

    # Special case: if both m and b requested, attempt polyfit
    if (("m" in names) or ("slope" in names)) and (("b" in names) or ("intercept" in names)):
        if x0.ndim == 1 and x0.size == y.size:
            try:
                m, b = np.polyfit(x0, y, deg=1)
                out.setdefault("m", float(m))
                out.setdefault("b", float(b))
                out.setdefault("slope", float(m))
                out.setdefault("intercept", float(b))
            except Exception:
                pass

    return out
