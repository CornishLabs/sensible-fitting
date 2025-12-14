import numpy as np
from sensible_fitting import Model


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


model = (
    Model.from_function(gaussian)
    .bound(amp=(0, None), sigma=(1e-6, None))
    .guess(mu=0.0, amp=1.0, sigma=1)  # or whatever you like as typical
    .derive("fwhm", lambda p: 2.354820045 * p["sigma"], doc="Full-width at half maximum")
)


rng = np.random.default_rng(0)
x = np.linspace(-3, 3, 200)
sigma_y = 0.05
y = model.eval(x, amp=1.0, mu=0.2, sigma=0.7) + rng.normal(0, sigma_y, size=x.size)

run = model.fit(x, (y, sigma_y)).squeeze()
res = run.results

print("sigma:", res["sigma"].value)
print("fwhm :", res["fwhm"].value, "(derived:", res["fwhm"].derived, ")")
