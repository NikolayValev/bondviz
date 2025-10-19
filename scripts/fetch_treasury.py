import pandas as pd
from bondviz.treasury import latest_par_yields
from bondviz.plots import plot_yield_curve

row = latest_par_yields()
print(row.to_string())

fig = plot_yield_curve(row)
fig.savefig("yield_curve.png", dpi=144, bbox_inches="tight")
print("Saved: yield_curve.png")
