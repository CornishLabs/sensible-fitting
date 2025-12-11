"""
Example: fits of fits (hierarchical use).

1) For each realisation r:
   - Generate 5 sinusoids with frequencies f_i = a_true + b_true * i, i=0..4.
   - Batch-fit the 5 sinusoids to extract the frequencies f_i (with errors).
   - Fit a straight line f(i) = a + b*i to those 5 frequencies.
   - Store (a_hat, b_hat).

2) Plot the cloud of (a_hat, b_hat) over several realisations.
"""

import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import Model, models


def main() -> None:
    rng = np.random.default_rng(123)

    # Sinusoid model for the first-level fits
    sin_model = (
        models.sinusoid(name="wave")
        .fix(offset=0.0, phase=0.0)
        .bound(amplitude=(0.5, 3.0), frequency=(0.5, 6.0))
        .guess(frequency=3.0)
    )

    # Straight line model: f(i) = a + b * i
    def line(i, a, b):
        return a + b * i

    line_model = Model.from_function(line, name="linear frequency")

    N_REAL = 3       # number of realisations of (a_true, b_true)
    N_SYSTEMS = 5    # number of sinusoids per realisation
    N = 300          # points per sinusoid

    x = np.linspace(0.0, 1.0, N)
    sigma_y = 0.2

    a_hats = []
    b_hats = []

    for r in range(N_REAL):
        # True linear relation for this realisation
        a_true = 2.5 + 0.2 * rng.normal()
        b_true = 0.5 + 0.05 * rng.normal()

        idx = np.arange(N_SYSTEMS, dtype=float)
        freq_true = a_true + b_true * idx

        # Generate 5 sinusoids with those frequencies, batched
        y_clean = []
        for i in range(N_SYSTEMS):
            y_clean.append(sin_model.eval(x, amplitude=1.5, frequency=freq_true[i]))
        y_clean = np.stack(y_clean, axis=0)

        y = y_clean + rng.normal(0, sigma_y, size=y_clean.shape)

        # First-level batch fit: get frequency per system
        run_sin = sin_model.fit(x=x, y=(y, sigma_y))
        res_sin = run_sin.results

        freqs = res_sin["frequency"].value
        freq_err = res_sin["frequency"].stderr

        # Second-level fit: line through frequencies vs. index
        run_line = line_model.fit(x=idx, y=(freqs, freq_err)).squeeze()
        res_line = run_line.results

        a_hat = res_line["a"].value
        b_hat = res_line["b"].value

        a_hats.append(a_hat)
        b_hats.append(b_hat)

        print(
            f"realisation {r}: "
            f"true (a,b)=({a_true:.3f}, {b_true:.3f}), "
            f"fit (a,b)=({a_hat:.3f}, {b_hat:.3f})"
        )

    a_hats = np.asarray(a_hats)
    b_hats = np.asarray(b_hats)

    # Scatter of the (a_hat, b_hat) pairs
    fig, ax = plt.subplots()
    ax.errorbar(a_hats, b_hats, fmt="o")
    ax.set_xlabel("a_hat")
    ax.set_ylabel("b_hat")
    ax.set_title("Fits-of-fits: (a, b) from each realisation")
    plt.show()


if __name__ == "__main__":
    main()
