from typing import Any, Dict, Optional, Tuple
import tempfile

import numpy as np 
from ..inference import build_gaussian_loglike, build_prior_transform, neg_loglike_binomial
from .common import BackendResult

class UltraNestBackend:
    name = "ultranest"

    def fit_one(
        self,
        *,
        model: Any,
        dataset: Any,
        free_names: list[str],
        fixed_map: dict[str, float],
        p0: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        options: dict[str, Any],
    ) -> BackendResult:
        fmt = getattr(dataset, "format", None)
        if fmt not in ("normal", "binomial"):
            raise NotImplementedError(
                "ultranest backend supports data_format in {'normal','binomial'}."
            )

        payload: Dict[str, Any] = dict(getattr(dataset, "payload"))
        x = getattr(dataset, "x")
        y = np.asarray(payload["y"], dtype=float) if fmt == "normal" else None
        sigma = payload.get("sigma", None) if fmt == "normal" else None
        sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)

        # Backend options parsing/merging lives HERE (not in Model.fit).
        backend_options = dict(options or {})
        vectorized = bool(backend_options.pop("vectorized", False))
        log_dir = backend_options.pop("log_dir", None)
        resume = backend_options.pop("resume", "subfolder")
        sampler_kwargs = dict(backend_options.pop("sampler_kwargs", {}) or {})
        run_kwargs = dict(backend_options.pop("run_kwargs", {}) or {})

        # Any leftover keys get treated as UltraNest run() kwargs, unless already set.
        for k, v in list(backend_options.items()):
            run_kwargs.setdefault(k, v)

        sampler_kwargs.setdefault("vectorized", vectorized)

        wrapped_params = [
            bool(getattr(spec, "wrapped", False))
            for spec in getattr(model, "params")
            if spec.name in free_names
        ]

        transform = build_prior_transform(getattr(model, "params"), free_names)
        if fmt == "normal":
            loglike = build_gaussian_loglike(
                model=model,
                x=x,
                y=y,
                sigma=sigma_arr,
                fixed_map=fixed_map,
                free_names=free_names,
                vectorized=vectorized,
            )
        else:
            n = np.asarray(payload["n"], dtype=float)
            k = np.asarray(payload["k"], dtype=float)

            def _one(theta_free: np.ndarray) -> float:
                kw = dict(fixed_map)
                for j, name in enumerate(free_names):
                    kw[name] = float(theta_free[j])
                p = np.asarray(model.eval(x, **kw), dtype=float)
                p = np.broadcast_to(p, k.shape)
                return float(-neg_loglike_binomial(p, n, k))

            if not vectorized:
                loglike = _one
            else:
                def _many(thetas: np.ndarray) -> np.ndarray:
                    thetas = np.asarray(thetas, dtype=float)
                    if thetas.ndim == 1:
                        return np.asarray(_one(thetas), dtype=float)
                    out = np.empty((thetas.shape[0],), dtype=float)
                    for i in range(thetas.shape[0]):
                        out[i] = _one(thetas[i])
                    return out
                loglike = _many

        fallback_theta = np.asarray(p0, dtype=float)

        temp_dir = None
        if log_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="ultranest_")
            log_dir = temp_dir.name

        try:
            import ultranest  # local import (optional dependency)

            sampler = ultranest.ReactiveNestedSampler(
                list(free_names),
                loglike,
                transform=transform,
                wrapped_params=wrapped_params,
                log_dir=log_dir,
                resume=resume,
                **sampler_kwargs,
            )

            result = sampler.run(**run_kwargs)

            samples = np.asarray(result.get("samples", []), dtype=float)
            if samples.ndim != 2 or samples.shape[1] != len(free_names) or samples.shape[0] == 0:
                center = fallback_theta
                cov = None
            else:
                # Use a robust point estimate for the Run "main line".
                # For multimodal posteriors (e.g. Ramsey fringe ambiguity), the mean
                # can sit in a low-likelihood region between modes.
                # Use a non-interpolating median to avoid landing between modes
                # for fringe-ambiguous (effectively discrete) posteriors.
                center = np.quantile(samples, 0.5, axis=0, method="nearest")
                cov = np.cov(samples.T, ddof=1) if samples.shape[0] >= 2 else None

            stats: Dict[str, Any] = {
                "backend": "ultranest",
                "free_names": tuple(free_names),
                "logz": float(result.get("logz", np.nan)),
                "logzerr": float(result.get("logzerr", np.nan)),
                "posterior_samples": samples,
                "ultranest_result": result,
            }

            return BackendResult(
                theta=np.asarray(center, dtype=float),
                cov=None if cov is None else np.asarray(cov, dtype=float),
                success=True,
                message="ok",
                stats=stats,
            )

        except Exception as e:
            return BackendResult(
                theta=np.asarray(fallback_theta, dtype=float),
                cov=None,
                success=False,
                message=str(e),
                stats={"backend": "ultranest", "error": str(e), "free_names": tuple(free_names)},
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
