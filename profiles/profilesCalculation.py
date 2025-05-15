
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------
# 1. Конформное отображение Жуковского Z → z
# ---------------------------------------------------------------

def zxzy(Z: complex, a: float, Z0: float, z0: float) -> complex:
    """Оригинальная формула из Лойцянского–Петрова."""
    sigma = z0 / Z0
    zsigma = (Z - Z0) / (Z + Z0)
    return z0 * (1 + zsigma ** sigma) / (1 - zsigma ** sigma)

zxzy_vec = np.vectorize(zxzy)

# ---------------------------------------------------------------
# 2. Модуль скорости |v(t)| и Cp
# ---------------------------------------------------------------

def velocity_mag(
    a: float,
    Z0: float,
    z0: float,
    beta: float,
    alpha: float,
    delta_t: float,
    t: np.ndarray,
    Vinf: float = 1.0,
) -> np.ndarray:
    """|v(t)| по формуле (A.3) с безопасным делением."""
    Gamma = -4 * math.pi * Vinf * a * math.sin(beta + alpha + delta_t)

    Z = Z0 - a * np.exp(-1j * beta) + a * np.exp(1j * (t - beta))
    z = zxzy_vec(Z, a, Z0, z0)

    num = np.abs(Z ** 2 - Z0 ** 2)
    den = np.abs(z ** 2 - z0 ** 2)
    den = np.clip(den, 1e-12, None)  # избегаем нулевого знаменателя

    v = 2 * num / den * (
        Vinf * np.sin(t - beta - alpha)
        - Gamma / (4 * math.pi * a)
    )

    # заменяем нелегальные значения ограниченными
    v = np.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)
    return v


def cp_distribution(v_mag: np.ndarray) -> np.ndarray:
    """Коэффициент давления Cp при нормировке V∞=1."""
    return 1.0 - v_mag ** 2

# ---------------------------------------------------------------
# 3. Интеграл Cd из Cp (формула 4)
# ---------------------------------------------------------------

def cd_integral(cp: np.ndarray, t: np.ndarray, z: np.ndarray, theta: float = 0.0) -> float:
    dz_dt = np.gradient(z, t)
    integrand = cp * np.exp(-1j * theta) * (1j * dz_dt.real - dz_dt.imag)
    return float(np.real(np.trapz(integrand, t)))

# ---------------------------------------------------------------
# 4. Решатель Δt по условию Хоурата
# ---------------------------------------------------------------

def delta_t_solver(a: float, Z0: float, z0: float, beta: float, alpha: float) -> float:
    def f(dt):
        t_cr = math.pi + 2 * (beta + alpha) + dt
        return velocity_mag(a, Z0, z0, beta, alpha, dt, np.array([t_cr]))[0]

    try:
        sol = root_scalar(f, bracket=[-0.6, 0.6], method="bisect", maxiter=60)
        return sol.root if sol.converged else 0.0
    except ValueError:
        return 0.0

# ---------------------------------------------------------------
# 5. Витерна
# ---------------------------------------------------------------

def viterna_extend(alpha_core, cl_core, cd_core, cdmax=1.3, step=1.0):
    a_left, a_right = alpha_core[0], alpha_core[-1]
    cl_left, cl_right = cl_core[0], cl_core[-1]
    cd_left, cd_right = cd_core[0], cd_core[-1]

    theta_s = math.radians(a_right)
    A2 = (cl_right - cdmax * math.sin(theta_s) * math.cos(theta_s)) * math.sin(theta_s) / (
        math.cos(theta_s) ** 2
    )
    B2 = (cd_right - cdmax * math.sin(theta_s) ** 2) / math.cos(theta_s)

    alphas = np.arange(-180.0, 180.0 + step, step)
    cl = np.zeros_like(alphas)
    cd = np.zeros_like(alphas)

    f_cl = interp1d(alpha_core, cl_core, kind="cubic", fill_value="extrapolate")
    f_cd = interp1d(alpha_core, cd_core, kind="cubic", fill_value="extrapolate")

    for i, a_deg in enumerate(alphas):
        if a_left <= a_deg <= a_right:
            cl[i] = f_cl(a_deg)
            cd[i] = f_cd(a_deg)
        else:
            th = math.radians(a_deg)
            cl[i] = 0.5 * cdmax * math.sin(2 * th) + A2 * (math.cos(th) ** 2 / math.sin(th))
            cd[i] = cdmax * (math.sin(th) ** 2) + B2 * math.cos(th)

    return alphas, cl, cd

# ---------------------------------------------------------------
# 6. Ядро поляр (±20°)
# ---------------------------------------------------------------

def build_core(row, step, chi, cd0, k, npts):
    a, Z0, beta, z0 = map(float, (row["a"], row["Z0"], row["beta"], row["z0"]))
    # сетка t без сингулярных точек 0,π,2π
    t_vals = np.linspace(0.0, 2 * math.pi, npts, endpoint=False) + 1e-4

    alphas = np.arange(-20.0, 20.0 + step, step)
    cl_list, cd_list = [], []

    for a_deg in alphas:
        alpha = math.radians(a_deg)
        dt = delta_t_solver(a, Z0, z0, beta, alpha)

        v_mag = velocity_mag(a, Z0, z0, beta, alpha, dt, t_vals)
        cp = cp_distribution(v_mag)
        Z = Z0 - a * np.exp(-1j * beta) + a * np.exp(1j * (t_vals - beta))
        z_vals = zxzy_vec(Z, a, Z0, z0)
        cd_val = cd_integral(cp, t_vals, z_vals)

        cl_val = 4 * math.pi * math.sin(beta + alpha + dt) * chi
        cd_val += cd0 + k * cl_val * cl_val

        cl_list.append(cl_val)
        cd_list.append(cd_val)

    return np.array(alphas), np.array(cl_list), np.array(cd_list)

# ---------------------------------------------------------------
# 7. Сохранение (Bound.+Viterna)
# ---------------------------------------------------------------

def radius_id(row) -> str:
    for key in ("radius", "r", "r_mm", "id"):
        if key in row.index:
            try:
                val = float(row[key])
                return str(int(round(val)))
            except Exception:
                continue
    return str(int(row.name))


def save_polar(out_dir: Path, fname: str, alphas, cl, cd, re, clip):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / fname).open("w", encoding="utf-8") as f:
        f.write("// Bound.+Viterna\n")
        f.write(f"// Re = {re:.2E}\n")
        f.write("// (alpha_deg cl cd)\n")
        for a, clv, cdv in zip(alphas, cl, cd):
            clv = float(np.clip(clv, -clip, clip))
            cdv = float(np.clip(cdv, -clip, clip))
            f.write(f"({a:8.2f} {clv: .5f} {cdv: .5f})\n")

# ---------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Zhukovsky polar builder (Cl/Cd vs alpha)")
    ap.add_argument("--params", default="final/zhukovsky_params.csv")
    ap.add_argument("--outdir", default="polars")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--re", type=float, default=1e5)
    ap.add_argument("--chi", type=float, default=1.05)
    ap.add_argument("--cd0", type=float, default=0.008)
    ap.add_argument("--k", type=float, default=0.02)
    ap.add_argument("--cdmax", type=float, default=1.3)
    ap.add_argument("--npts", type=int, default=400)
    ap.add_argument("--clip", type=float, default=1e3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.params)
    processed = 0
    out_dir = Path(args.outdir)

    for _, row in df.iterrows():
        pid = radius_id(row)

        a_core, cl_core, cd_core = build_core(
            row, args.step, args.chi, args.cd0, args.k, args.npts
        )
        alp, cl, cd = viterna_extend(a_core, cl_core, cd_core, args.cdmax, args.step)
        save_polar(out_dir, pid, alp, cl, cd, args.re, args.clip)
        processed += 1

        if args.debug:
            plt.figure(figsize=(6, 4))
            plt.plot(a_core, cl_core, "o", label="core Cl")
            plt.plot(alp, cl, "-", label="full Cl")
            plt.title(f"Section {pid}")
            plt.xlabel("α, °")
            plt.ylabel("Cl")
            plt.grid(True)
            plt.legend()
            plt.savefig(out_dir / f"{pid}_debug.png", dpi=120)
            plt.close()

        print(f"[OK] section {pid} done")

    print(f"Finished: {processed} files → {out_dir}")


if __name__ == "__main__":
    main()
