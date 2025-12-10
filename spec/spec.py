"""sensible_fitting v1 API SPEC (executable-looking pseudocode)

Goal
----
A small, productive fitting API with:
- ergonomic model definition (from plain python functions)
- sensible defaults for common Gaussian fitting
- optional power backends (e.g., nested sampling) when needed
- batch fitting (many independent datasets) with clean slicing semantics
- model-owned seed/guess logic
- derived parameters computed *after* fitting (v1), with no dependency chains

This file is a *spec*, not an implementation. It is intended to be:
- readable as a single source of truth
- runnable only in the sense that it is valid Python syntax

Design principles
-----------------
- Models are immutable builder objects: .fix/.bound/.guess return new Models.
- Parameters are explicit objects: value/error plus metadata.
- Defaults reduce boilerplate for common cases.
- Shapes are first-class: batch dims are preserved; slicing works.
- Provided measurement errors are always treated as absolute measurement errors.

Non-goals (v1)
--------------
- parameter constraints/ties that affect the fit (lmfit-style expr during optimization)
- correlated noise / full covariance Gaussian likelihood
- robust likelihoods (Student-t)
- globally-coupled multi-dataset fits (shared parameters across datasets)

See “V2 considerations” at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


# -----------------------------------------------------------------------------
# Core public classes (stubs): Model, Run, Results
# -----------------------------------------------------------------------------

Array = np.ndarray


@dataclass(frozen=True)
class Band:
    """Uncertainty band over a grid of x values."""

    low: Array
    high: Array
    # Optional extras
    median: Optional[Array] = None
    meta: Dict[str, Any] = None


@dataclass(frozen=True)
class ParamView:
    """A single parameter view.

    Semantics
    ---------
    - value/error are numpy arrays of shape == Results.batch_shape
    - For squeezed single-fit Results, value/error are scalars (0-d arrays) or Python floats.
    - Supports both attribute and mapping-style access:
        p.value  == p["value"]
        p.error  == p["error"]
    """

    name: str
    value: Any
    error: Any
    fixed: Any = False
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    derived: bool = False
    meta: Dict[str, Any] = None

    # Mapping-like shorthand
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
    """Mapping of parameter name -> ParamView.

    v1 requirements:
    - params["m"].value exists
    - params["m"]["value"] exists and matches
    - slicing a Results slices the values/errors inside ParamViews
    """

    def __getitem__(self, key: str) -> ParamView: ...
    def __iter__(self): ...
    def __len__(self): ...


@dataclass(frozen=True)
class Results:
    """Backend-agnostic fit results.

    Attributes
    ----------
    batch_shape:
        Tuple of batch dimensions. Empty tuple means scalar/single fit.

    params:
        Mapping-like object with ParamViews.
        params includes both fitted parameters and post-fit derived parameters.

    cov:
        Optional covariance matrix for *free* fitted parameters.
        For batched fits, cov may be an array of shape batch_shape + (P,P).

    backend:
        Backend identifier string (e.g., "scipy.curve_fit", "ultranest").

    meta:
        Free-form backend metadata (success flags, messages, nfev, etc.).
    """

    batch_shape: Tuple[int, ...]
    params: ParamsView
    cov: Optional[Array] = None
    backend: str = ""
    meta: Dict[str, Any] = None

    # ---- slicing semantics ----
    def __getitem__(self, idx) -> "Results":
        """Slice along the batch axes.

        Examples
        --------
        res[0] returns a scalar Results (batch_shape=())
        res[0:2] returns Results with batch_shape=(2,)

        Also supports chained slicing:
            res[0:2]["m"]["value"]  -> array shape (2,)
        """
        raise NotImplementedError

    def summary(self, digits: int = 4) -> str:
        """Return a human-friendly table.

        v1 requirements:
        - For scalar results: a compact table of parameters + errors + bounds/fixed flags.
        - For batched results: a row per batch element, or a condensed table.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Run:
    """A Run bundles the model, data, backend info, and results.

    Run is the primary object for post-fit utilities:
    - band() uncertainty bands
    - slicing to get a sub-run
    - exposing any posterior samples, if backend provides them
    """

    model: "Model"
    results: Results
    backend: str
    data_format: str
    meta: Dict[str, Any] = None

    # optional: store normalized data representation for plotting/reuse
    data: Dict[str, Any] = None

    def squeeze(self) -> "Run":
        """Remove batch axes of length 1.

        v1 strictness:
        - If total batch size > 1, calling squeeze() with no axis raises.
          (This makes single-fit intent explicit.)

        Example
        -------
        run = model.fit(..., return_run=True)
        run = run.squeeze()  # ok only if exactly one fit is present
        """
        raise NotImplementedError

    def __getitem__(self, idx) -> "Run":
        """Slice a batched run to a sub-run."""
        raise NotImplementedError

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
        """Compute an uncertainty band for model predictions.

        Parameters
        ----------
        level:
            Shorthand for a symmetric Normal-equivalent sigma level.
            Example: level=2 means a central interval with quantiles
                (Phi(-2), Phi(+2)) ~= (0.02275, 0.97725).

        conf_int:
            Explicit central interval quantiles, e.g. (0.05, 0.95).

        method:
            - "auto": prefer posterior samples if present, else covariance
            - "posterior": require posterior samples (else raise)
            - "covariance": require covariance (else raise)

        Returns
        -------
        Band(low, high)

        Notes
        -----
        - For covariance-based bands, we sample parameters from N(theta_hat, cov).
        - For posterior-based bands, we sample parameters from posterior.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    fixed: bool = False
    fixed_value: Optional[float] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    guess: Optional[float] = None
    # v1: priors are reserved for Bayesian backends, bounds imply uniform prior
    prior: Optional[Tuple[str, Tuple[Any, ...]]] = None
    meta: Dict[str, Any] = None


class Guesser(Protocol):
    """Guesser signature.

    v1: user-defined guessers are pure functions; they mutate a GuessState.
    """

    def __call__(self, x: Any, y: Any, g: "GuessState") -> None: ...


@dataclass
class GuessState:
    """Mutable guess container passed to guessers."""

    # Implementations may store guesses in a dict internally.
    def is_unset(self, name: str) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class DerivedSpec:
    """Post-fit derived parameter spec (v1).

    Important v1 restriction:
    - derived parameters are computed only AFTER fitting
    - derived parameters depend ONLY on fitted params, not on other derived params
      (no dependency graph; simple one-pass computation)
    """

    name: str
    func: Callable[[Mapping[str, float]], float]
    doc: str = ""
    meta: Dict[str, Any] = None


@dataclass(frozen=True)
class Model:
    """A model wraps a callable and parameter metadata."""

    name: str
    func: Callable[..., Any]
    param_names: Tuple[str, ...]
    params: Tuple[ParameterSpec, ...]
    guessers: Tuple[Guesser, ...] = ()
    derived: Tuple[DerivedSpec, ...] = ()
    meta: Dict[str, Any] = None

    # --- constructors ---
    @staticmethod
    def from_function(func: Callable[..., Any], *, name: Optional[str] = None) -> "Model":
        """Create a Model from a python function.

        Signature conventions
        ---------------------
        def f(x, p1, p2, ...): ...

        - The first argument is the independent variable container ("x").
          It may be an array or a tuple/list of arrays.
        - Remaining positional arguments are treated as model parameters.
        - Keyword-only parameters are allowed only if they are also treated as parameters.
          (Implementations may choose to forbid *args/**kwargs in v1 for clarity.)
        """
        raise NotImplementedError

    # --- evaluation ---
    def eval(self, x: Any, *, params: Optional[Mapping[str, Any]] = None, **kwargs) -> Any:
        """Evaluate model.

        Usage
        -----
        model.eval(x, m=2.0, b=-1.0)
        model.eval(x, params=res.params)

        Notes
        -----
        - If params is provided, it may be:
            * a dict of name->float
            * a ParamsView/ParamView mapping (where value is extracted)
        - kwargs may override params entries.
        """
        raise NotImplementedError

    # --- builders (immutability) ---
    def fix(self, **fixed: float) -> "Model":
        """Fix parameters to constant values."""
        raise NotImplementedError

    def bound(self, **bounds: Tuple[Optional[float], Optional[float]]) -> "Model":
        """Set hard bounds for parameters.

        v1 behavior:
        - For Bayesian backends, bounds imply a Uniform prior if no explicit prior is provided.
        """
        raise NotImplementedError

    def guess(self, **guesses: float) -> "Model":
        """Set manual initial guesses."""
        raise NotImplementedError

    def autoguess(self, *names: str) -> "Model":
        """Enable built-in autoguess strategies for the named parameters."""
        raise NotImplementedError

    def prior(self, **priors: Tuple[str, Any]) -> "Model":
        """Attach explicit priors for Bayesian backends.

        Examples
        --------
        model.prior(m=("normal", 0.0, 5.0), b=("cauchy", 0.0, 2.0))

        Notes
        -----
        - v1 may implement priors only for ultranest.
        - For non-Bayesian backends priors are stored but ignored.
        """
        raise NotImplementedError

    def derive(self, name: str, func: Callable[[Mapping[str, float]], float], *, doc: str = "") -> "Model":
        """Add a post-fit derived parameter.

        v1 restrictions:
        - derived params computed AFTER fit
        - derived func must depend only on fitted params (not other derived)

        Example
        -------
        model = model.derive("fwhm", lambda p: 2.354820045 * p["sigma"], doc="Gaussian FWHM")
        """
        raise NotImplementedError

    def guesser(self, fn: Optional[Guesser] = None):
        """Decorator to register a custom guesser.

        Usage
        -----
        @model.guesser
        def my_guess(x, y, g):
            if g.is_unset("amplitude"):
                g.amplitude = ...
        """
        raise NotImplementedError

    # --- fitting ---
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
        """Fit the model.

        Data inference rules (v1)
        -------------------------
        Unless data_format is specified, y is interpreted as Gaussian:

        - y.shape == (...):                 y is observed values (unweighted)
        - y is a tuple/list of length 2:    (y, yerr) symmetric absolute errors
        - y is a tuple/list of length 3:    (y, yerr_low, yerr_high) asymmetric absolute errors

        If you want a different meaning for length-1/2/3, you MUST set data_format.

        Absolute measurement errors (v1)
        --------------------------------
        If yerr is provided, it is ALWAYS treated as absolute measurement error.
        (Equivalent to always using absolute_sigma=True where applicable.)

        Batch semantics (v1)
        --------------------
        - If y is an array with shape (B, N) and x is (N,), we treat as B independent fits.
        - If x differs per dataset, x and y may be provided as lists of datasets.

        return_run
        ----------
        - If False: returns Results
        - If True:  returns Run

        backend_options
        ---------------
        Backend-specific knobs.

        SciPy curve_fit:
            - absolute_sigma is implied True when yerr is provided
            - bounds pulled from Model.bound
            - p0 constructed from guesses/autoguess/custom guessers

        SciPy minimize:
            - uses negative log-likelihood for selected data_format

        UltraNest:
            - requires priors OR bounds for all free params
            - bounds imply uniform priors
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Convenience module: sensible_fitting.models (prebuilt models)
# -----------------------------------------------------------------------------


class models:
    """Namespace for prebuilt models.

    v1 includes only a small set of high-utility models with sensible defaults.

    Each factory returns a Model that can still be customized with .fix/.bound/.guess/etc.
    """

    @staticmethod
    def straight_line(*, name: str = "straight line") -> Model:
        """Return y = m*x + b."""
        raise NotImplementedError

    @staticmethod
    def sinusoid(*, name: str = "sinusoid") -> Model:
        """Return offset + amplitude*sin(2π*frequency*x + phase)."""
        raise NotImplementedError


# =============================================================================
# USAGE EXAMPLES (canonical)
# =============================================================================


def example_01_basic_line_curve_fit():
    """Basic straight-line fit using scipy.curve_fit."""

    import matplotlib.pyplot as plt

    # 1) define model from plain function signature (params inferred from args)
    def line(x, m, b):
        return m * x + b

    model = Model.from_function(line, name="straight line")

    # 2) make fake data
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y_true = line(x, 2.0, -1.0)
    sigma = 0.6
    y = y_true + rng.normal(0, sigma, size=x.size)

    # 3) fit: tuple-length inference => Gaussian likelihood with absolute errors
    run = model.fit(
        x=x,
        y=(y, sigma),
        backend="scipy.curve_fit",
        return_run=True,
        # v1: always absolute measurement errors; backend_options can be omitted.
        # backend_options={"absolute_sigma": True},
    ).squeeze()

    res = run.results

    # 4) pythonic param access
    m = res.params["m"]
    b = res.params["b"]
    print(m.value, "±", m.error)
    print(b.value, "±", b.error)

    # mapping-style also works
    assert m.value == res.params["m"]["value"]

    print(res.summary(digits=4))

    # 5) plot data + fit line + band
    fig, ax = plt.subplots()

    ax.errorbar(x, y, yerr=sigma, fmt="o", ms=4, capsize=2, label="data")

    xg = np.linspace(x.min(), x.max(), 400)
    yg = run.model.eval(xg, params=res.params)
    ax.plot(xg, yg, label="fit")

    # level=2 => central quantiles ~ (0.02275, 0.97725)
    band = run.band(xg, nsamples=400, level=2, method="auto")
    ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ")

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def example_02_prebuilt_line_model():
    """Prebuilt model with bounds."""

    import matplotlib.pyplot as plt

    model = models.straight_line().bound(m=(-10, 10))

    rng = np.random.default_rng(1)
    x = np.linspace(-2, 2, 60)
    sigma = 0.15
    y = 0.7 * x + 0.2 + rng.normal(0, sigma, size=x.size)

    run = model.fit(x=x, y=(y, sigma), backend="scipy.curve_fit", return_run=True).squeeze()
    res = run.results

    print(res.summary(digits=6))

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=sigma, fmt=".", label="data")

    xg = np.linspace(x.min(), x.max(), 400)
    ax.plot(xg, run.model.eval(xg, params=res.params), label="fit")

    band = run.band(xg, nsamples=500, level=2)
    ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ")

    ax.legend()
    plt.show()


def example_03_sinusoid_with_guessers_and_fixes():
    """Sinusoid with fixed params, bounds, manual + auto guesses, and custom guesser."""

    import matplotlib.pyplot as plt

    base = models.sinusoid(name="sinusoid")

    model = (
        base
        .fix(offset=0.0, phase=np.pi / 3)
        .bound(amplitude=(0.2, 5.0), frequency=(1.0, 10.0))
        .guess(frequency=3.1)
        .autoguess("amplitude")
    )

    @model.guesser
    def smart_init(x, y, g):
        # Keep user guess if provided; otherwise infer.
        if g.is_unset("frequency"):
            # v1 may implement a simple frequency estimate; lomb-scargle is optional.
            g.frequency = 3.0

        # robust-ish amplitude estimate
        g.amplitude = np.quantile(y, 0.95) - np.quantile(y, 0.05)

    # Fake data
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 200)
    sigma = 0.15
    y = model.eval(x, amplitude=1.8, frequency=3.3) + rng.normal(0, sigma, size=x.size)

    run = model.fit(x=x, y=(y, sigma), backend="scipy.curve_fit", return_run=True).squeeze()
    res = run.results

    print(res.summary(digits=5))

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=sigma, fmt=".", label="data")

    xg = np.linspace(x.min(), x.max(), 400)
    ax.plot(xg, run.model.eval(xg, params=res.params), label="fit")

    band = run.band(xg, nsamples=600, level=2)
    ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ")

    ax.legend()
    plt.show()


def example_04_batch_fit_common_x():
    """Batch fit with common x across datasets."""

    import matplotlib.pyplot as plt

    model = (
        models.sinusoid(name="wave")
        .fix(offset=0.0, phase=np.pi / 3)
        .bound(amplitude=(0.2, 5.0), frequency=(1.0, 6.0))
        .guess(frequency=2.8)
        .autoguess("amplitude")
    )

    # make 4 datasets
    rng = np.random.default_rng(2)
    N_SYSTEMS, N = 4, 250
    x = np.linspace(0, 1, N)

    A0, F0 = 2.0, 3.0
    A = A0 * (1 + 0.05 * rng.normal(size=N_SYSTEMS))
    F = F0 * (1 + 0.02 * rng.normal(size=N_SYSTEMS))

    sigma = 0.2
    y_clean = np.stack([model.eval(x, amplitude=A[i], frequency=F[i]) for i in range(N_SYSTEMS)])
    y = y_clean + rng.normal(0, sigma, size=y_clean.shape)

    # Batch fit: y has shape (systems, N); sigma scalar broadcasts
    run = model.fit(
        x=x,
        y=(y, sigma),
        backend="scipy.curve_fit",
        parallel="auto",
        return_run=True,
    )

    res = run.results
    print(res.summary(digits=4))

    # slicing semantics
    print(res[0:2].params["frequency"].value)  # (2,)

    # plot on grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axs = np.ravel(axs)

    xg = np.linspace(x.min(), x.max(), 500)

    for i, ax in enumerate(axs):
        ax.errorbar(x, y[i], yerr=sigma, fmt=".", ms=3, label=f"data {i}")

        yi = run.model.eval(xg, params=res[i].params)
        ax.plot(xg, yi, label="fit")

        band = run[i].band(xg, nsamples=300, level=2)
        ax.fill_between(xg, band.low, band.high, alpha=0.2)

        ax.set_title(f"system {i}")
        ax.legend()

    plt.show()


def example_05_batch_fit_ragged_x():
    """Batch fit where each dataset has its own x grid.

    Spec: if x differs, user passes lists of datasets.
    """

    model = models.straight_line()

    rng = np.random.default_rng(123)

    xs: List[Array] = []
    ys: List[Tuple[Array, Array]] = []

    for i in range(3):
        n = 30 + 10 * i
        x = np.sort(rng.uniform(-2, 2, size=n))
        sigma = 0.1 + 0.05 * rng.random(size=n)
        y = 0.5 * x - 0.1 + rng.normal(0, sigma)

        xs.append(x)
        ys.append((y, sigma))

    run = model.fit(
        x=xs,
        y=ys,
        backend="scipy.curve_fit",
        parallel="auto",
        return_run=True,
    )

    res = run.results
    print(res.summary(digits=4))


def example_06_backend_swap_quick_vs_ultranest():
    """Same model, same data, two backends.

    v1 intent:
    - quick: scipy.curve_fit
    - full posterior: ultranest (requires bounds/priors)
    """

    import matplotlib.pyplot as plt

    def line(x, m, b):
        return m * x + b

    model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10))

    rng = np.random.default_rng(5)
    x = np.linspace(0, 4, 50)
    sigma = 0.3
    y = line(x, 1.7, -0.4) + rng.normal(0, sigma, size=x.size)

    run_cf = model.fit(x=x, y=(y, sigma), backend="scipy.curve_fit", return_run=True).squeeze()

    # Bayesian backend requires priors OR bounds for all free params.
    run_ns = model.fit(x=x, y=(y, sigma), backend="ultranest", return_run=True).squeeze()

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=sigma, fmt=".", label="data")

    xg = np.linspace(x.min(), x.max(), 400)

    y_cf = run_cf.model.eval(xg, params=run_cf.results.params)
    ax.plot(xg, y_cf, label="curve_fit")

    y_ns = run_ns.model.eval(xg, params=run_ns.results.params)
    ax.plot(xg, y_ns, label="ultranest MAP/median")

    band_cf = run_cf.band(xg, level=2, method="covariance")
    ax.fill_between(xg, band_cf.low, band_cf.high, alpha=0.15, label="curve_fit ~2σ")

    # For ultranest, method="auto" should prefer posterior
    band_ns = run_ns.band(xg, level=2, method="auto")
    ax.fill_between(xg, band_ns.low, band_ns.high, alpha=0.15, label="ultranest 2σ")

    ax.legend()
    plt.show()


def example_07_derived_params_post_fit():
    """Derived parameters computed AFTER fitting (v1).

    v1: derived params do NOT affect optimization/sampling.
    They are computed in Results after fit completes.
    """

    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    model = (
        Model.from_function(gaussian)
        .bound(amp=(0, None), sigma=(1e-6, None))
        .derive(
            "fwhm",
            lambda p: 2.354820045 * p["sigma"],
            doc="Full-width at half maximum",
        )
    )

    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 200)
    sigma_y = 0.05
    y = model.eval(x, amp=1.0, mu=0.2, sigma=0.7) + rng.normal(0, sigma_y, size=x.size)

    run = model.fit(x=x, y=(y, sigma_y), backend="scipy.curve_fit", return_run=True).squeeze()
    res = run.results

    print(res.params["sigma"].value, res.params["fwhm"].value)
    assert res.params["fwhm"].derived is True


# =============================================================================
# Defaults & standards (v1)
# =============================================================================

"""
1) Default likelihood / data_format
----------------------------------
If data_format is None:
- y is array-like:
    -> Gaussian, unweighted
- y is (y, yerr):
    -> Gaussian with symmetric absolute errors
- y is (y, yerr_low, yerr_high):
    -> Gaussian with asymmetric absolute errors

If user wants a different meaning for y shapes 1/2/3, they MUST set data_format.

2) Absolute measurement errors
------------------------------
Whenever yerr is provided, it is treated as absolute measurement error.

If yerr is omitted:
- fit is unweighted
- reported parameter errors may use an estimated residual scale

3) Asymmetric errors ordering
-----------------------------
We adopt Matplotlib convention:
- (y, yerr_low, yerr_high)
Where yerr_low is downward error (y - low), yerr_high is upward error (high - y).

4) Parameter access
-------------------
- res.params["m"].value and res.params["m"]["value"] both work and match.
- res.params["m"].error == res.params["m"]["error"]

5) Band API
-----------
- band(level=2) means Normal-equivalent ±2σ central interval
- band(conf_int=(qlo,qhi)) supported
- method='auto' prefers posterior if available, else covariance

6) Squeeze semantics
--------------------
- run.squeeze() raises unless there is exactly one fit present.

7) Batch semantics
------------------
- x common: x shape (N,), y shape (B,N)
- x ragged: x is list of arrays; y is list of matching payloads

8) Derived parameters (v1)
--------------------------
- derived params computed after fit only
- derived params depend only on fitted params
- derived params do not influence fitting
"""


# =============================================================================
# V2 considerations (reserved names / ideas)
# =============================================================================

"""V2 ideas to keep doors open (not in v1 scope)

A) Parameter constraints/ties DURING fitting
--------------------------------------------
Competitors like lmfit allow algebraic constraints: param.expr depends on other params.
In sensible_fitting, a v2 feature could allow:
- model.tie(name=...) / model.expr(...) / model.constrain(...)
- ties affect the free parameterization and thus the fit

Reserved builder names (suggestion):
- tie, expr, constrain, derive (already used for post-fit derived)

B) Shared/global parameters across datasets
-------------------------------------------
Coupled multi-dataset fits (global + per-dataset params) could be expressed with:
- model.share(...), model.link(...), model.per_batch(...), model.expand(...)

C) Additional likelihoods
------------------------
- Binomial / Poisson / lognormal / Student-t
- Full covariance Gaussian: (y, C)

D) Predictive utilities
-----------------------
- run.predict(x, kind='mean'|'median')
- posterior predictive sampling

E) Diagnostics
--------------
- residual plots
- goodness-of-fit summaries
- corner plots (posterior)

F) Multi-output models
----------------------
- y is dict of observables, or stacked outputs
"""
