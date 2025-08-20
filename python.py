# 24f2007860@ds.study.iitm.ac.in
# Marimo notebook (Python script with cell markers) demonstrating variable dependencies,
# interactive widgets, dynamic markdown output, and documented data flow between cells.

# %% [markdown]
# # Interactive Data Relationship Demo (matplotlib-free fallback)
# This Marimo notebook demonstrates a linear relationship between two variables (x and y)
# using a synthetic dataset. It includes an interactive slider widget to adjust noise,
# and avoids hard dependency on `matplotlib` by using an inline SVG renderer when needed.

# %%
# Cell 1 — Data generation (base variables)
import math
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, HTML, clear_output
from ipywidgets import FloatSlider, VBox, interactive_output, Label

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def make_data(n=200, noise_std=1.0):
    x = np.linspace(0, 10, n)
    y = 2.5 * x + 1.0 + np.random.normal(0, noise_std, size=n)
    return pd.DataFrame({"x": x, "y": y})

df_raw = make_data()
print("Preview of df_raw (first 5 rows):")
print(df_raw.head())

# %%
# Cell 2 — Derived features

def derive_features(df):
    df = df.copy()
    df["x_norm"] = (df["x"] - df["x"].mean()) / (df["x"].std() or 1)
    df["y_roll"] = df["y"].rolling(window=10, center=True, min_periods=1).mean()
    stats = {
        "mean_x": float(df["x"].mean()),
        "mean_y": float(df["y"].mean()),
        "std_y": float(df["y"].std()),
        "correlation": float(df["x"].corr(df["y"]) if len(df) > 1 else 0.0)
    }
    return df, stats

# %%
# Cell 3 — SVG plotting fallback (no matplotlib required)

def _map_to_canvas(val, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2.0
    return out_min + (val - vmin) * (out_max - out_min) / (vmax - vmin)

def make_svg_scatter(df, title=None, width=600, height=300, point_r=2):
    padding = 40
    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    yroll = df["y_roll"].to_numpy()
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(min(ys.min(), yroll.min())), float(max(ys.max(), yroll.max()))

    def mx(xv): return _map_to_canvas(xv, xmin, xmax, padding, width - padding)
    def my(yv): return _map_to_canvas(yv, ymin, ymax, height - padding, padding)

    circles = [f'<circle cx="{mx(x):.2f}" cy="{my(y):.2f}" r="{point_r}" fill="blue" opacity="0.5" />' for x,y in zip(xs,ys)]
    pts = [f"{mx(x):.2f},{my(y):.2f}" for x,y in zip(xs,yroll) if not math.isnan(y)]
    path = f"<polyline fill='none' stroke='red' stroke-width='2' points='{' '.join(pts)}' />" if pts else ""

    svg = f"<svg width='{width}' height='{height}'>{''.join(circles)}{path}</svg>"
    if title:
        return f"<div><h4>{title}</h4>{svg}</div>"
    return svg

# %%
# Cell 4 — Interactive slider widget and dynamic markdown

def render(noise_std=1.0):
    df = make_data(noise_std=noise_std)
    df2, s = derive_features(df)
    clear_output(wait=True)
    html_plot = make_svg_scatter(df2, title=f"Scatter of y vs x (noise_std={noise_std:.2f})")
    display(HTML(html_plot))
    md = (
        f"**Noise (std)**: `{noise_std:.2f}`  \n"
        f"**Mean y**: `{s['mean_y']:.3f}`  \n"
        f"**Std of y**: `{s['std_y']:.3f}`  \n"
        f"**Correlation (x,y)**: `{s['correlation']:.3f}`  \n"
    )
    display(Markdown(md))

# Interactive slider widget (canonical variable name `slider`)
slider = FloatSlider(value=1.0, min=0.0, max=5.0, step=0.1, description='Noise')
out = interactive_output(render, {"noise_std": slider})
ui = VBox([Label("Interactive controls:"), slider, out])
display(ui)

# %%
# Cell 5 — Final remarks
print("Finished: Use the slider above to explore how noise changes the relationship between x and y.")
