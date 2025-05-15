#!/usr/bin/env python3

import argparse
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BAD = 1e9
MIN_POINTS = 20

# --------------------------------------------------------------------
#   Жуковский профиль (порт Mathematica)
# --------------------------------------------------------------------

def zhukovsky_profile(a: float, Z0: float, beta: float, z0: float, N: int = 150):
    t = -beta + (np.arange(1, N + 1) * 2 * math.pi / N)
    Z2 = a * np.exp(1j * t)
    ZZ = (Z0 - a * np.exp(-1j * beta) + Z2)

    sigma = z0 / Z0
    with np.errstate(divide="ignore", invalid="ignore"):
        Zsigma1 = (ZZ - Z0) / (ZZ + Z0)
        Zsigma = np.concatenate([Zsigma1[:-1] ** sigma, [0]])
        zz = z0 * (1 + Zsigma) / (1 - Zsigma)

    Re_shift = np.real(zz - z0)
    min_re = Re_shift.min()
    if min_re >= 0 or not np.isfinite(min_re):
        raise ValueError("Invalid profile: min(Re) non‑negative or NaN")

    x = -Re_shift / min_re + 1
    y = np.imag(zz - z0)
    return np.column_stack([x, y])

# --------------------------------------------------------------------
#   Least‑squares objective
# --------------------------------------------------------------------

def objective(params, data, N=150):
    a, Z0, beta, z0 = params
    try:
        model = zhukovsky_profile(a, Z0, beta, z0, N)
    except Exception:
        return np.full(data.shape[0], BAD)

    if not np.isfinite(model).all():
        return np.full(data.shape[0], BAD)

    dists, _ = cKDTree(model).query(data)
    return dists

# --------------------------------------------------------------------
#   Fit one CSV profile
# --------------------------------------------------------------------

def fit_one_profile(points: np.ndarray, name: str, out_dir: Path, debug=False):
    xmax, xmin = points[:, 0].max(), points[:, 0].min()
    chord = xmax - xmin
    if chord <= 0 or not np.isfinite(chord):
        logging.warning("%s: invalid chord", name)
        return None

    a0 = 0.225 * chord
    Z00 = 0.24 * chord
    beta0 = 0.075
    z00 = (xmax + xmin) / 2

    p0 = np.array([a0, Z00, beta0, z00])
    bounds_lo = np.array([0.02 * chord, 0.05 * chord, 0.001, 0.0])
    bounds_hi = np.array([0.4 * chord,  1.0 * chord, math.pi / 2, 1.2 * chord])

    res = least_squares(
        objective,
        p0,
        bounds=(bounds_lo, bounds_hi),
        kwargs=dict(data=points, N=150),
        max_nfev=30000,
        loss='soft_l1',
        verbose=0,
    )

    if not res.success:
        logging.warning("%s: optimisation failed → %s", name, res.message)
        return None

    a, Z0, beta, z0 = res.x
    rms = math.sqrt(np.mean(res.fun ** 2))
    logging.info(
        "%s → a=%.5f, Z0=%.5f, β=%.3f°, z0=%.5f, RMS=%.2e",
        name, a, Z0, math.degrees(beta), z0, rms,
    )

    # Overlay plot
    if debug or rms > 5e-3:
        model = zhukovsky_profile(a, Z0, beta, z0)
        plt.figure(figsize=(4, 3))
        plt.plot(points[:, 0], points[:, 1], "o", ms=3, label="data")
        plt.plot(model[:, 0], model[:, 1], "-", lw=1.4, label="fit")  # ASCII minus
        plt.gca().set_aspect("equal", "box")
        plt.legend()
        plt.title(name)
        plt.savefig(out_dir / f"{name}_fit.png", dpi=200, bbox_inches="tight")
        plt.close()

    return dict(file=name, a=a, Z0=Z0, beta=beta, z0=z0, rms=rms)

# --------------------------------------------------------------------
#   CSV reader with ;‑separator & comma decimal
# --------------------------------------------------------------------

def read_profile_csv(path: Path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as err:
        logging.warning("%s: %s", path.name, err)
        return None

    df.columns = [c.lower().strip() for c in df.columns]

    # Replace decimal comma
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.replace(",", ".", regex=False))

    # Identify X/Y columns
    x_candidates = [c for c in df.columns if re.fullmatch(r"x(_final)?", c)]
    y_candidates = [c for c in df.columns if re.fullmatch(r"y(_final)?", c)]

    if not (x_candidates and y_candidates):
        num_cols = df.select_dtypes(include=[float, int]).columns
        if len(num_cols) < 2:
            return None
        xcol, ycol = num_cols[:2]
    else:
        xcol, ycol = x_candidates[0], y_candidates[0]

    x = pd.to_numeric(df[xcol], errors="coerce").values
    y = pd.to_numeric(df[ycol], errors="coerce").values
    mask = np.isfinite(x) & np.isfinite(y)
    pts = np.column_stack([x[mask], y[mask]])
    return pts if pts.shape[0] >= MIN_POINTS else None

# --------------------------------------------------------------------
#   CLI
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="final", help="input folder with *_final.csv")
    ap.add_argument("-o", "--output", default=None, help="output folder (default=input)")
    ap.add_argument("--debug", action="store_true", help="always save overlay PNGs")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output or args.input)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for csv_path in sorted(in_dir.glob("*_final.csv")):
        pts = read_profile_csv(csv_path)
        if pts is None:
            logging.warning("%s: cannot identify X/Y columns or not enough points", csv_path.name)
            continue

        res = fit_one_profile(pts, csv_path.stem.replace("_final", ""), out_dir, debug=args.debug)
        if res:
            results.append(res)

    if results:
        pd.DataFrame(results).to_csv(out_dir / "zhukovsky_params.csv", index=False)
        logging.info("Done. Processed %d files.", len(results))
    else:
        logging.warning("No profiles processed.")


if __name__ == "__main__":
    main()
