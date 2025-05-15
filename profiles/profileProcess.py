#!/usr/bin/env python3
"""profilesProcess.py — единый конвейер обработки профилей винта
=================================================================
Этот скрипт полностью заменяет четыре вспомогательные утилиты:
    – развёртка цилиндрических координат (theta·r vs Y)
    – выделение одного кластер‑профиля
    – переориентация (ось X вдоль хорды)
    – поворот и нормировка хорды к единице

На вход принимает .csv с колонками `Points_0, Points_1, Points_2` в папке initial/.
На выход кладёт файлы "<name>_final.csv" и "<name>_preview.png" в папку final/.

---------------------------------------------------------------
USAGE (по умолчанию):
    python profileProcess.py          # берёт ./initial, пишет ./final

С ключами:
    python profileProcess.py \
        --input  data/raw_profiles \
        --output results/normalized \
        --gap    0.03               # порог разрыва (доля от диапазона X_unwrapped)

Все зависимости: `numpy  pandas  matplotlib`.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. ПАРСЕР АРГУМЕНТОВ
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Integrated pipeline → normalized blade profile")
    p.add_argument("--input",  "-i", default="initial", help="Folder with raw CSV files")
    p.add_argument("--output", "-o", default="final",   help="Folder for normalized CSV/PNG")
    p.add_argument("--gap",    "-g", type=float, default=0.03,
                   help="Fraction of X_unwrapped‑range treated as a gap between stripes")
    p.add_argument("--plot",   action="store_true", help="Save step‑by‑step debug plot")
    return p

# ---------------------------------------------------------------------------
# 2. ШАГ 1 – РАЗВЁРТКА ЦИЛИНДРИЧЕСКОЙ ПОВЕРХНОСТИ
# ---------------------------------------------------------------------------

def unwrap_cylinder(df_xyz: pd.DataFrame) -> pd.DataFrame:
    """Convert (X,Y,Z) -> (theta, r) and build unwrapped coordinates."""
    df = df_xyz.copy()
    df.rename(columns={
        "Points_0": "X",
        "Points_1": "Y",
        "Points_2": "Z",
    }, inplace=True)
    df["theta"] = np.arctan2(df["Z"], df["X"])
    df["r"]     = np.sqrt(df["X"]**2 + df["Z"]**2)
    df["X_unwrapped"] = df["r"] * df["theta"]
    df["Y_unwrapped"] = df["Y"]
    return df[["X_unwrapped", "Y_unwrapped"]]

# ---------------------------------------------------------------------------
# 3. ШАГ 2 – ВЫДЕЛЯЕМ ОДИН КЛАСТЕР (ОДНУ «ПОЛОСУ» ПРОФИЛЯ)
# ---------------------------------------------------------------------------

def find_clusters_by_gap(df: pd.DataFrame, gap_fraction: float) -> list[pd.DataFrame]:
    """Группируем точки вдоль X_unwrapped: если соседние X отличаются
    > gap_threshold – считаем, что это разрыв (конец полосы)."""
    df_sorted = df.sort_values("X_unwrapped").reset_index(drop=True)
    x = df_sorted["X_unwrapped"].values
    gap_threshold = gap_fraction * (x.max() - x.min())
    # индексы, где разрыв: dist > threshold
    breaks = np.where(np.diff(x) > gap_threshold)[0]
    # формируем срезы между разрывами
    segments: list[pd.DataFrame] = []
    start = 0
    for idx in breaks:
        segments.append(df_sorted.iloc[start:idx+1].copy())
        start = idx + 1
    segments.append(df_sorted.iloc[start:].copy())
    return segments


def choose_best_cluster(segments: list[pd.DataFrame]) -> pd.DataFrame:
    """Ищем сегмент, чей *средний* X_unwrapped ближе к нулю.
    В большинстве сканов именно такой сегмент соответствует реальному контуру."""
    if not segments:
        raise ValueError("No clusters found")
    best = min(segments, key=lambda seg: abs(seg["X_unwrapped"].mean()))
    return best

# ---------------------------------------------------------------------------
# 4. ШАГ 3 – ПЕРЕОРИЕНТАЦИЯ К СИСТЕМЕ (X_along_chord, Y_from_chord)
# ---------------------------------------------------------------------------

def reorder_and_flip(df: pd.DataFrame) -> pd.DataFrame:
    """Map (X_unwrapped,Y_unwrapped) → (X_new,Y_new) and order points CCW."""
    df2 = pd.DataFrame({
        "X_new": df["Y_unwrapped"],
        "Y_new": -df["X_unwrapped"],  # инвертируем ось Y
    })
    # сортируем по углу вокруг геом. центра
    cx, cy = df2["X_new"].mean(), df2["Y_new"].mean()
    df2["theta"] = np.arctan2(df2["Y_new"] - cy, df2["X_new"] - cx)
    df_sorted = df2.sort_values("theta").reset_index(drop=True)
    # сдвигаем так, чтобы самой левой точке соответствовал первый индекс
    left_idx = df_sorted["X_new"].idxmin()
    df_shifted = pd.concat([df_sorted.iloc[left_idx:], df_sorted.iloc[:left_idx]],
                            ignore_index=True)
    return df_shifted[["X_new", "Y_new"]]

# ---------------------------------------------------------------------------
# 5. ШАГ 4 – ПОВОРОТ ХОРДЫ В ГОРИЗОНТАЛЬ + НОРМИРОВКА ДЛИНЫ ДО 1.0
# ---------------------------------------------------------------------------

def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[ c, -s], [s,  c]])
    return points @ R.T  # (N,2) · (2,2)


def align_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    pts = df[["X_new", "Y_new"]].values
    # крайние по X точки как концы хорды
    i_left, i_right = np.argmin(pts[:,0]), np.argmax(pts[:,0])
    p_left, p_right = pts[i_left], pts[i_right]
    dx, dy = p_right - p_left
    angle = math.atan2(dy, dx)
    pts_rot = rotate(pts, -angle)  # хорда горизонтальна
    # сдвиг: левая точка → (0,0)
    pts_shift = pts_rot - pts_rot[i_left]
    # длина хорды
    chord = math.hypot(*(p_right - p_left))
    if chord < 1e-8:
        raise ValueError("Degenerate chord length ≈ 0")
    pts_norm = pts_shift / chord  # X in [0..1]
    df_out = pd.DataFrame(pts_norm, columns=["X_final", "Y_final"])
    return df_out, angle, chord

# ---------------------------------------------------------------------------
# 6. ОСНОВНОЙ ЦИКЛ
# ---------------------------------------------------------------------------

def process_file(csv_path: Path, output_dir: Path, gap_fraction: float, save_plot: bool=False) -> None:
    name = csv_path.stem
    df_xyz = pd.read_csv(csv_path)
    if not {"Points_0","Points_1","Points_2"}.issubset(df_xyz.columns):
        print(f"[skip] {name}: missing required columns")
        return
    # 1) unwrap
    df_unw = unwrap_cylinder(df_xyz)
    # 2) clusters
    clusters = find_clusters_by_gap(df_unw, gap_fraction)
    profile = choose_best_cluster(clusters)
    # 3) reorder
    profile_reor = reorder_and_flip(profile)
    # 4) align + normalize
    df_final, angle_rad, chord = align_and_normalize(profile_reor)

    # save csv
    out_csv = output_dir / f"{name}_final.csv"
    df_final.to_csv(out_csv, index=False)

    # optional plot
    if save_plot:
        theta = np.linspace(0, 2*math.pi, 400)
        circle_x = 0.5*np.cos(theta)
        circle_y = 0.5*np.sin(theta)
        plt.figure(figsize=(6,6))
        plt.plot(circle_x, circle_y, color="orange", label="0.5‑радиус")
        plt.scatter(df_final["X_final"], df_final["Y_final"], s=8, color="green", label="profile")
        plt.axis("equal")
        plt.title(f"{name}: chord=1.0  (orig chord={chord:.3f} → angle={math.degrees(angle_rad):.1f}°)")
        plt.grid(True)
        plt.legend()
        out_png = output_dir / f"{name}_preview.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    print(f"[ok] {name}: points={len(df_final)}, chord={chord:.3f}, angle={math.degrees(angle_rad):.1f}°")


# ---------------------------------------------------------------------------
# 7. ENTRY POINT
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    inp_dir  = Path(args.input)
    out_dir  = Path(args.output)
    gap_frac = args.gap
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(inp_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {inp_dir.resolve()}")
        return

    for csv_file in csv_files:
        try:
            process_file(csv_file, out_dir, gap_frac, save_plot=True)
        except Exception as e:
            print(f"[error] {csv_file.stem}: {e}")

if __name__ == "__main__":
    main()
