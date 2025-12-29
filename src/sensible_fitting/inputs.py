from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FitData:
    """Container for fit inputs plus lightweight plotting metadata.

    This is an optional convenience layer: the library still accepts raw arrays
    and tuples/lists as usual.
    """

    x: Any
    data: Any
    data_format: Optional[str] = None

    # Plotting metadata (optional)
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    label: Optional[str] = None  # legend label for the data

    # Extra user metadata (stored on Run.data["meta"] if present)
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def normal(
        *,
        x: Any,
        y: Any,
        yerr: Optional[Any] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        label: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "FitData":
        """Create FitData for data_format='normal'."""
        data = y if yerr is None else (y, yerr)
        return FitData(
            x=x,
            data=data,
            data_format="normal",
            x_label=x_label,
            y_label=y_label,
            label=label,
            meta=dict(meta or {}),
        )

    @staticmethod
    def binomial(
        *,
        x: Any,
        n: Any,
        k: Any,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        label: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "FitData":
        """Create FitData for data_format='binomial'."""
        return FitData(
            x=x,
            data=(n, k),
            data_format="binomial",
            x_label=x_label,
            y_label=y_label,
            label=label,
            meta=dict(meta or {}),
        )

    @staticmethod
    def beta(
        *,
        x: Any,
        alpha: Any,
        beta: Any,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        label: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "FitData":
        """Create FitData for data_format='beta'."""
        return FitData(
            x=x,
            data=(alpha, beta),
            data_format="beta",
            x_label=x_label,
            y_label=y_label,
            label=label,
            meta=dict(meta or {}),
        )

    @staticmethod
    def from_param(
        *,
        x: Any,
        param: Any,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        label: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "FitData":
        """Create FitData for data_format='normal' from a ParamView-like object."""
        name = getattr(param, "name", None)
        y = getattr(param, "value", None)
        yerr = getattr(param, "stderr", None)
        if y is None:
            raise TypeError("param must provide .value")
        if y_label is None and isinstance(name, str) and name:
            y_label = name
        if label is None and isinstance(name, str) and name:
            label = name
        return FitData.normal(
            x=x,
            y=y,
            yerr=yerr,
            x_label=x_label,
            y_label=y_label,
            label=label,
            meta=meta,
        )

    def with_labels(
        self,
        *,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "FitData":
        """Return a copy with updated label fields."""
        return replace(
            self,
            x_label=self.x_label if x_label is None else x_label,
            y_label=self.y_label if y_label is None else y_label,
            label=self.label if label is None else label,
        )

    def with_meta(self, **meta: Any) -> "FitData":
        """Return a copy with additional meta merged in."""
        merged = dict(self.meta)
        merged.update(meta)
        return replace(self, meta=merged)

    def append(
        self,
        *,
        x: Any,
        y: Any = None,
        yerr: Optional[Any] = None,
        n: Any = None,
        k: Any = None,
        alpha: Any = None,
        beta: Any = None,
    ) -> "FitData":
        """Append new measurements to this FitData (v1: 1D x only)."""
        fmt = self.data_format
        if fmt is None:
            raise ValueError("FitData.append requires data_format to be set.")

        x0 = np.asarray(self.x, dtype=float).reshape((-1,))
        x1 = np.asarray(x, dtype=float).reshape((-1,))
        if x1.size == 0:
            return self

        if fmt == "normal":
            if y is None:
                raise TypeError("FitData.append(normal) requires y=...")

            # Unpack existing
            payload0 = self.data
            if isinstance(payload0, tuple) and len(payload0) == 2:
                y0, e0 = payload0
            elif isinstance(payload0, tuple) and len(payload0) == 3:
                # Stored as (y, lo, hi). Convert to symmetric sigma for appending.
                y0, lo0, hi0 = payload0
                e0 = 0.5 * (np.asarray(lo0, dtype=float) + np.asarray(hi0, dtype=float))
            else:
                y0, e0 = payload0, None

            y0 = np.asarray(y0, dtype=float).reshape((-1,))
            y1 = np.asarray(y, dtype=float).reshape((-1,))
            if y0.shape != x0.shape:
                raise ValueError("Existing normal payload y must match x shape.")
            if y1.shape != x1.shape:
                raise ValueError("New normal payload y must match x shape.")

            x_out = np.concatenate([x0, x1])
            y_out = np.concatenate([y0, y1])

            # Handle yerr merging.
            if e0 is None and yerr is None:
                return replace(self, x=x_out, data=y_out)

            def _as_err_array(err: Any, npts: int) -> Optional[np.ndarray]:
                if err is None:
                    return None
                a = np.asarray(err, dtype=float)
                if a.shape == ():
                    return np.full((npts,), float(a), dtype=float)
                a = a.reshape((-1,))
                if a.shape != (npts,):
                    raise ValueError("yerr must be scalar or same length as x.")
                return a

            e0_arr = _as_err_array(e0, x0.size) if e0 is not None else None

            if yerr is None:
                if e0 is None:
                    e1_arr = None
                else:
                    # If existing yerr is scalar, reuse it for new points.
                    e0_scalar = np.asarray(e0).shape == ()
                    if not e0_scalar:
                        raise ValueError(
                            "Appending without yerr requires existing yerr to be scalar."
                        )
                    e1_arr = np.full((x1.size,), float(np.asarray(e0)), dtype=float)
            else:
                e1_arr = _as_err_array(yerr, x1.size)

            if e0_arr is None:
                if e1_arr is None:
                    return replace(self, x=x_out, data=y_out)
                # If we didn't previously have yerr, assume new scalar applies to old points.
                if np.asarray(yerr).shape != ():
                    raise ValueError(
                        "Existing data has no yerr; appending array yerr is ambiguous."
                    )
                e0_arr = np.full((x0.size,), float(np.asarray(yerr)), dtype=float)

            if e1_arr is None:
                # Existing yerr must have been scalar (handled above).
                e1_arr = np.full((x1.size,), float(e0_arr[0]), dtype=float)

            e_out = np.concatenate([e0_arr, e1_arr])

            # Preserve scalar yerr if it is constant.
            if e_out.size and np.allclose(e_out, e_out[0], rtol=0.0, atol=0.0):
                return replace(self, x=x_out, data=(y_out, float(e_out[0])))
            return replace(self, x=x_out, data=(y_out, e_out))

        if fmt == "binomial":
            if n is None or k is None:
                raise TypeError("FitData.append(binomial) requires n=... and k=...")
            if not (isinstance(self.data, tuple) and len(self.data) == 2):
                raise TypeError("Existing binomial payload must be (n, k).")
            n0, k0 = self.data
            n0 = np.asarray(n0).reshape((-1,))
            k0 = np.asarray(k0).reshape((-1,))
            n1 = np.asarray(n).reshape((-1,))
            k1 = np.asarray(k).reshape((-1,))
            if n0.shape != x0.shape or k0.shape != x0.shape:
                raise ValueError("Existing binomial payload must match x shape.")
            if n1.shape != x1.shape or k1.shape != x1.shape:
                raise ValueError("New binomial payload must match x shape.")
            return replace(
                self,
                x=np.concatenate([x0, x1]),
                data=(np.concatenate([n0, n1]), np.concatenate([k0, k1])),
            )

        if fmt == "beta":
            if alpha is None or beta is None:
                raise TypeError("FitData.append(beta) requires alpha=... and beta=...")
            if not (isinstance(self.data, tuple) and len(self.data) == 2):
                raise TypeError("Existing beta payload must be (alpha, beta).")
            a0, b0 = self.data
            a0 = np.asarray(a0).reshape((-1,))
            b0 = np.asarray(b0).reshape((-1,))
            a1 = np.asarray(alpha).reshape((-1,))
            b1 = np.asarray(beta).reshape((-1,))
            if a0.shape != x0.shape or b0.shape != x0.shape:
                raise ValueError("Existing beta payload must match x shape.")
            if a1.shape != x1.shape or b1.shape != x1.shape:
                raise ValueError("New beta payload must match x shape.")
            return replace(
                self,
                x=np.concatenate([x0, x1]),
                data=(np.concatenate([a0, a1]), np.concatenate([b0, b1])),
            )

        raise NotImplementedError(f"FitData.append does not support data_format={fmt!r}.")

    def extend(self, *args: Any, **kwargs: Any) -> "FitData":
        """Alias for append(...); accepts multiple points at once."""
        return self.append(*args, **kwargs)

    def sorted(self) -> "FitData":
        """Return a copy sorted by x (v1: 1D x only)."""
        if self.data_format is None:
            raise ValueError("FitData.sorted requires data_format to be set.")

        x = np.asarray(self.x, dtype=float).reshape((-1,))
        order = np.argsort(x)
        if np.all(order == np.arange(order.size)):
            return self

        fmt = self.data_format
        payload = self.data

        if fmt == "normal":
            if isinstance(payload, tuple) and len(payload) == 2:
                y, e = payload
                y = np.asarray(y, dtype=float).reshape((-1,))[order]
                if np.asarray(e).shape == ():
                    return replace(self, x=x[order], data=(y, e))
                e = np.asarray(e, dtype=float).reshape((-1,))[order]
                return replace(self, x=x[order], data=(y, e))
            if isinstance(payload, tuple) and len(payload) == 3:
                y, lo, hi = payload
                y = np.asarray(y, dtype=float).reshape((-1,))[order]
                lo = np.asarray(lo, dtype=float).reshape((-1,))[order]
                hi = np.asarray(hi, dtype=float).reshape((-1,))[order]
                return replace(self, x=x[order], data=(y, lo, hi))
            y = np.asarray(payload, dtype=float).reshape((-1,))[order]
            return replace(self, x=x[order], data=y)

        if fmt in ("binomial", "beta"):
            if not (isinstance(payload, tuple) and len(payload) == 2):
                raise TypeError(f"Existing {fmt} payload must be a 2-tuple.")
            a, b = payload
            a = np.asarray(a).reshape((-1,))[order]
            b = np.asarray(b).reshape((-1,))[order]
            return replace(self, x=x[order], data=(a, b))

        raise NotImplementedError(f"FitData.sorted does not support data_format={fmt!r}.")

    def plot(self, *args: Any, **kwargs: Any):
        """Plot this data using sensible defaults (no fit line/band)."""
        from .viz import plot_data

        return plot_data(self, *args, **kwargs)
