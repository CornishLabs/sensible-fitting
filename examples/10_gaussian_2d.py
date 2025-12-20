"""
Example: fit a 2D Gaussian (image-like data) by flattening to 1D.

We pass x as a tuple of flattened (X, Y) grids and y as a flattened image.
"""

import numpy as np

from sensible_fitting import Model


def gauss2d(xy, amp, x0, y0, sx, sy, offset):
    if isinstance(xy, (tuple, list)):
        x, y = xy
    else:
        xy_arr = np.asarray(xy)
        if xy_arr.shape[0] != 2:
            raise ValueError("gauss2d expects xy with shape (2, N).")
        x, y = xy_arr[0], xy_arr[1]
    dx = x - x0
    dy = y - y0
    return offset + amp * np.exp(-(dx * dx / (2 * sx * sx) + dy * dy / (2 * sy * sy)))


def main() -> None:
    rng = np.random.default_rng(0)

    nx, ny = 60, 50
    x = np.linspace(-3.0, 3.0, nx)
    y = np.linspace(-2.5, 2.5, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    true = dict(amp=2.0, x0=0.3, y0=-0.2, sx=1.1, sy=0.7, offset=0.1)
    sigma = 0.05
    z = gauss2d((X, Y), **true) + rng.normal(0, sigma, size=X.shape)

    model = (
        Model.from_function(gauss2d, name="gauss2d")
        .bound(amp=(0.0, None), sx=(0.1, 5.0), sy=(0.1, 5.0))
        .guess(amp=1.5, x0=0.0, y0=0.0, sx=1.0, sy=1.0, offset=0.0)
    )

    # Tuple x is fine for ND inputs; lists are reserved for ragged batches.
    xy = (X.ravel(), Y.ravel())
    z_flat = z.ravel()

    run = model.fit(xy, (z_flat, sigma)).squeeze()
    res = run.results

    print("fit params:")
    for name in ("amp", "x0", "y0", "sx", "sy", "offset"):
        pv = res[name]
        print(f"  {name:>6s}: {pv.value:.4g} ± {pv.stderr:.3g}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is not None:
        z_fit = run.predict((X.ravel(), Y.ravel())).reshape(X.shape)
        resid = z - z_fit

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        axs[0].imshow(z, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
        axs[0].set_title("data")
        axs[1].imshow(z_fit, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
        axs[1].set_title("fit")
        axs[2].imshow(resid, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
        axs[2].set_title("residual")
        plt.show()


if __name__ == "__main__":
    main()
