from __future__ import annotations

import numpy as np

from ..model import Model


def rabi_func(x, amplitude, offset, t_period, phase, tau, t_dead):
    """Decaying cosine with dead time for Rabi oscillations."""
    x_arr = np.asarray(x, dtype=float)
    t_dead = np.asarray(t_dead, dtype=float)
    t_period = np.asarray(t_period, dtype=float)
    tau = np.asarray(tau, dtype=float)
    t = x_arr - t_dead
    t_eff = np.where(t < 0.0, 0.0, t)
    env = np.exp(-t_eff / tau)
    return offset + amplitude * env * np.cos(2 * np.pi * t_eff / t_period + phase)


def rabi_oscillation(*, name: str = "rabi") -> Model:
    """Return a Rabi oscillation Model with decay, phase, and sensible seeding.

    Parameters in the model
    -----------------------
    amplitude : oscillation amplitude
    offset    : baseline offset
    t_period  : oscillation period (> 0)
    phase     : phase offset (radians)
    tau       : exponential decay time (> 0)
    t_dead    : dead time before oscillations start (>= 0)

    Derived parameters
    ------------------
    t_pi      : approximate pi time (t_dead + t_period / 2)
    rabi_freq : 1 / t_period
    omega     : 2 * pi / t_period
    """
    base = (
        Model.from_function(rabi_func, name=name)
        .bound(
            amplitude=(0.0, 1.0),
            offset=(0.0, 1.0),
            phase=(-np.pi, np.pi),
            t_period=(1e-12, None),
            tau=(1e-12, None),
            t_dead=(0.0, None),
        )
        .wrap(phase=True)
    )

    def init_rabi(x, y, g):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return

        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        t_range = x_max - x_min
        if t_range <= 0.0:
            t_range = 1.0

        y_mean = float(np.mean(y_arr))
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        amp = 0.5 * (y_max - y_min)
        if not np.isfinite(amp) or amp == 0.0:
            amp = 1.0

        if g.is_unset("offset"):
            g.offset = y_mean
        if g.is_unset("amplitude"):
            g.amplitude = amp
        if g.is_unset("t_dead"):
            g.t_dead = 0.0
        if g.is_unset("tau"):
            g.tau = t_range

        if g.is_unset("t_period"):
            period = None
            try:
                from scipy.signal import lombscargle

                base_freq = np.pi / t_range
                nfreq = max(10, 2 * int(x_arr.size))
                freqs = np.linspace(0.1 * base_freq, 10.0 * base_freq, nfreq)
                y_detrend = y_arr - y_mean
                pgram = lombscargle(x_arr, y_detrend, freqs, precenter=False)
                best = float(freqs[int(np.argmax(pgram))])
                if np.isfinite(best) and best > 0.0:
                    period = 2 * np.pi / best
            except Exception:
                period = None

            if period is None or not np.isfinite(period) or period <= 0.0:
                period = 2.0 * t_range
            g.t_period = float(period)

        if g.is_unset("phase"):
            idx = int(np.argmin(x_arr))
            t0 = x_arr[idx] - float(g.t_dead)
            t0 = float(t0) if t0 > 0.0 else 0.0
            amp_guess = float(g.amplitude)
            offset_guess = float(g.offset)
            tau_guess = float(g.tau)
            denom = amp_guess * np.exp(-t0 / tau_guess) if tau_guess > 0 else amp_guess
            if denom != 0.0:
                cos_arg = (float(y_arr[idx]) - offset_guess) / denom
                cos_arg = float(np.clip(cos_arg, -1.0, 1.0))
                phase = np.arccos(cos_arg) - 2 * np.pi * t0 / float(g.t_period)
                g.phase = float(phase)
            else:
                g.phase = 0.0

    model = base.with_guesser(init_rabi)
    model = model.derive(
        "t_pi",
        lambda p: p["t_dead"] + 0.5 * p["t_period"],
        doc="Approx pi time (t_dead + t_period/2)",
    )
    model = model.derive(
        "rabi_freq",
        lambda p: 1.0 / p["t_period"],
        doc="Rabi frequency (1 / t_period)",
    )
    model = model.derive(
        "omega",
        lambda p: 2.0 * np.pi / p["t_period"],
        doc="Angular frequency (2*pi / t_period)",
    )

    return model
