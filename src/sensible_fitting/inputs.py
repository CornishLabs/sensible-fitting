from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple


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

    def plot(self, *args: Any, **kwargs: Any):
        """Plot this data using sensible defaults (no fit line/band)."""
        from .viz import plot_data

        return plot_data(self, *args, **kwargs)
