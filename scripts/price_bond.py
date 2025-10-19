import argparse
import numpy as np
from bondviz.pricing import pv_continuous, discount_factors_continuous
from bondviz.plots import plot_discount_factors

p = argparse.ArgumentParser()
p.add_argument("--face", type=float, default=1000)
p.add_argument("--coupon", type=float, required=True, help="annual coupon rate, e.g. 0.05")
p.add_argument("--yield", dest="y", type=float, required=True, help="continuous yield, e.g. 0.04")
p.add_argument("--years", type=float, required=True)
args = p.parse_args()

pv = pv_continuous(args.face, args.coupon, args.y, args.years)
print(f"PV: {pv:.4f}")

tenors = np.linspace(0, args.years, num=11)
points = discount_factors_continuous(args.y, tenors)
fig = plot_discount_factors(points)
fig.savefig("discount_factors.png", dpi=144, bbox_inches="tight")
print("Saved: discount_factors.png")
