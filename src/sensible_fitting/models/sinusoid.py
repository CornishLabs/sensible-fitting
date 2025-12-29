from __future__ import annotations

import numpy as np

from ..model import Model


def sinusoid_func(x, amplitude, offset, frequency, phase):
    """Module-level sinusoid: offset + amplitude * sin(2π f x + phase)."""
    return offset + amplitude * np.sin(2 * np.pi * frequency * x + phase)


def _guess_sinusoid(x, y, g) -> None:
    """Sensible default guesser for sinusoids.

    Strategy:
    - estimate frequency via FFT peak (requires ~uniform sampling),
    - then estimate offset/amplitude/phase via linear least squares at that frequency.
    """
    try:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
    except Exception:
        return

    if x_arr.size < 6 or y_arr.size != x_arr.size:
        return
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        return

    # Ensure increasing x (FFT assumes regular spacing; this at least prevents negative dx).
    if np.any(np.diff(x_arr) < 0):
        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]

    # Fallback offset/amplitude guesses (used if FFT/LS fails).
    if g.is_unset("offset"):
        g.offset = float(np.mean(y_arr))
    if g.is_unset("amplitude"):
        amp0 = 0.5 * float(np.max(y_arr) - np.min(y_arr))
        g.amplitude = 1.0 if not np.isfinite(amp0) or amp0 <= 0 else amp0

    # Determine a frequency to use for phase/amplitude refinement.
    f0 = None
    if not g.is_unset("frequency"):
        try:
            f0 = float(g.frequency)
        except Exception:
            f0 = None

    if f0 is None:
        dx = np.diff(x_arr)
        if not np.all(dx > 0):
            return
        dx_med = float(np.median(dx))
        if not np.isfinite(dx_med) or dx_med <= 0:
            return
        # Rough uniformity check: FFT is only sensible for near-uniform x.
        rel = float(np.std(dx) / (np.mean(dx) + 1e-15))
        if rel > 1e-2:
            return

        y0 = y_arr - float(np.mean(y_arr))
        yf = np.fft.rfft(y0)
        freqs = np.fft.rfftfreq(y0.size, d=dx_med)
        if freqs.size < 2:
            return

        mag = np.abs(yf)
        k = int(np.argmax(mag[1:]) + 1)
        df = float(freqs[1] - freqs[0])
        f_peak = float(freqs[k])

        # Sub-bin refinement via quadratic interpolation around the FFT peak.
        if 0 < k < (mag.size - 1):
            y1, y2, y3 = float(mag[k - 1]), float(mag[k]), float(mag[k + 1])
            denom = (y1 - 2.0 * y2 + y3)
            if denom != 0.0:
                delta = 0.5 * (y1 - y3) / denom
                if np.isfinite(delta) and abs(delta) <= 1.0:
                    f_peak = float(freqs[k] + delta * df)

        # Small local search: pick f that best explains y via linear LS.
        def _solve_at_freq(freq: float):
            w = 2.0 * np.pi * float(freq)
            s = np.sin(w * x_arr)
            c = np.cos(w * x_arr)
            A = np.stack([np.ones_like(x_arr), s, c], axis=1)
            coef, resid, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
            if resid.size:
                sse = float(resid[0])
            else:
                yhat = A @ coef
                r = y_arr - yhat
                sse = float(np.sum(r * r))
            return sse, coef

        f_best = f_peak
        sse_best, coef_best = _solve_at_freq(f_best)
        if np.isfinite(df) and df > 0:
            for frac in np.linspace(-1.0, 1.0, 21):
                f_try = f_peak + float(frac) * df
                if not np.isfinite(f_try) or f_try <= 0:
                    continue
                sse, coef = _solve_at_freq(f_try)
                if np.isfinite(sse) and sse < sse_best:
                    sse_best, coef_best = sse, coef
                    f_best = f_try

        f0 = float(f_best)
        if g.is_unset("frequency"):
            g.frequency = f0

        # Reuse the best LS coefficients below.
        off, a_sin, a_cos = [float(v) for v in coef_best]
        amp = float(np.hypot(a_sin, a_cos))
        phase = float(np.arctan2(a_cos, a_sin))

        if g.is_unset("offset") and np.isfinite(off):
            g.offset = off
        if g.is_unset("amplitude") and np.isfinite(amp) and amp > 0:
            g.amplitude = amp
        if g.is_unset("phase") and np.isfinite(phase):
            g.phase = phase
        return

    # Refine offset/amplitude/phase via linear least squares at f0.
    w = 2.0 * np.pi * float(f0)
    s = np.sin(w * x_arr)
    c = np.cos(w * x_arr)
    A = np.stack([np.ones_like(x_arr), s, c], axis=1)  # (N, 3)
    try:
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    except Exception:
        return

    off, a_sin, a_cos = [float(v) for v in coef]
    amp = float(np.hypot(a_sin, a_cos))
    phase = float(np.arctan2(a_cos, a_sin))

    if g.is_unset("offset") and np.isfinite(off):
        g.offset = off
    if g.is_unset("amplitude") and np.isfinite(amp) and amp > 0:
        g.amplitude = amp
    if g.is_unset("phase") and np.isfinite(phase):
        g.phase = phase


def sinusoid(*, name: str = "sinusoid") -> Model:
    """Return a sinusoid Model with sensible default seeding."""
    return Model.from_function(sinusoid_func, name=name).with_guesser(_guess_sinusoid)
