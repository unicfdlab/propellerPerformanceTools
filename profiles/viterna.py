import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splprep, splev


# ===== Методы для расчёта по Витерне =====

def viterna_method(alpha_deg, cl_stall, cd_stall, cd_max, alpha_stall_deg):
    """
    Реализация метода Витерны:
      alpha_deg       : угол атаки (в градусах)
      cl_stall        : Cl на угле сваливания alpha_stall_deg
      cd_stall        : Cd на угле сваливания alpha_stall_deg
      cd_max          : максимально возможный Cd (параметр для Витерны)
      alpha_stall_deg : угол сваливания (в градусах), вокруг которого оцениваются параметры
    """
    alpha_rad = np.radians(alpha_deg)
    alpha_stall_rad = np.radians(alpha_stall_deg)

    A1 = cd_max / 2.0
    A2 = ((cl_stall - cd_max * np.sin(alpha_stall_rad) * np.cos(alpha_stall_rad))
          * np.sin(alpha_stall_rad) / (np.cos(alpha_stall_rad) ** 2))
    B1 = cd_max
    B2 = (cd_stall - cd_max * (np.sin(alpha_stall_rad) ** 2)) / np.cos(alpha_stall_rad)

    cl = A1 * np.sin(2.0 * alpha_rad) + A2 * (np.cos(alpha_rad) ** 2) / np.sin(alpha_rad)
    cd = B1 * (np.sin(alpha_rad) ** 2) + B2 * np.cos(alpha_rad)

    return cl, cd


def parse_bound_viterna_file(filename):
    """
    Считывает файл, в котором каждая строка имеет формат:
      (alpha  Cl  Cd)
    Пример:
      ( -180  0.0000  0.0825 )
    Возвращает numpy-массивы: alphas, cls, cds
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден.")

    alphas = []
    cls = []
    cds = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                content = line[1:-1].strip()
                parts = content.split()
                if len(parts) == 3:
                    try:
                        alpha_val = float(parts[0])
                        cl_val = float(parts[1])
                        cd_val = float(parts[2])
                        alphas.append(alpha_val)
                        cls.append(cl_val)
                        cds.append(cd_val)
                    except ValueError:
                        pass
    return np.array(alphas), np.array(cls), np.array(cds)


def remove_duplicates(alphas, cls, cds):
    """
    Удаляет дублирующиеся значения углов атаки, усредняя Cl и Cd для каждого повторяющегося alpha.
    """
    unique_alphas, inv_indices = np.unique(alphas, return_inverse=True)
    avg_cls = []
    avg_cds = []
    for i in range(len(unique_alphas)):
        mask = (inv_indices == i)
        avg_cls.append(np.mean(cls[mask]))
        avg_cds.append(np.mean(cds[mask]))
    return unique_alphas, np.array(avg_cls), np.array(avg_cds)


def build_interpolators(alphas_data, cls_data, cds_data):
    """
    Создаёт интерполяционные функции Cl(alpha) и Cd(alpha) по исходным данным.
    """
    cl_interp = interp1d(alphas_data, cls_data, kind='cubic', fill_value="extrapolate")
    cd_interp = interp1d(alphas_data, cds_data, kind='cubic', fill_value="extrapolate")
    return cl_interp, cd_interp


def compute_cl_cd(alpha_deg, alpha_left, alpha_right,
                  cl_interp_func, cd_interp_func,
                  cl_stall_left, cd_stall_left,
                  cl_stall_right, cd_stall_right,
                  cd_max):
    """
    Сшивка интерполяции и метода Витерны:
      - если alpha_deg внутри [alpha_left, alpha_right], берём интерполяцию;
      - если alpha_deg < alpha_left, используем метод Витерны слева;
      - если alpha_deg > alpha_right, используем метод Витерны справа.
    """
    if alpha_left <= alpha_deg <= alpha_right:
        cl = cl_interp_func(alpha_deg)
        cd = cd_interp_func(alpha_deg)
    elif alpha_deg < alpha_left:
        cl, cd = viterna_method(alpha_deg, cl_stall_left, cd_stall_left, cd_max, alpha_left)
    else:  # alpha_deg > alpha_right
        cl, cd = viterna_method(alpha_deg, cl_stall_right, cd_stall_right, cd_max, alpha_right)
    return cl, cd


def save_to_bound_viterna_format(output_file, alphas, cls, cds):
    """
    Сохраняет данные в файл в требуемом формате:
      // Bound.+Viterna
      // Re = 1.00E+05
      // (alpha_deg cl cd)
      (alpha cl cd)
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("// Bound.+Viterna\n")
        f.write("// Re = 1.00E+05\n")
        f.write("// (alpha_deg cl cd)\n")
        for alpha, cl_val, cd_val in zip(alphas, cls, cds):
            f.write(f"({alpha:.2f} {cl_val:.4f} {cd_val:.4f})\n")


def main_bound_viterna_processing(
        input_file,
        output_file,
        plot_file="comparison_plot.png",
        alpha_min=-180,
        alpha_max=180,
        alpha_step=0.5,
        AR=11 / 2.2
):
    """
    Основной рабочий процесс:
      1) Считать исходные данные из input_file.
      2) Удалить дубликаты и отсортировать данные.
      3) Определить границы срыва:
         - для отрицательных углов – выбрать локальный минимум, ближайший к -10°,
         - для положительных углов – выбрать локальный максимум, ближайший к 10°.
      4) Вычислить витерну для сетки углов от alpha_min до alpha_max с шагом alpha_step,
         используя интерполяцию внутри [alpha_left, alpha_right] и метод Витерны вне этих границ.
      5) Заменить строки для углов -180 и 180 значениями, полученными для -179.5 и 179.5 соответственно.
      6) Сохранить результат в формате «Bound.+Viterna».
      7) Построить и сохранить сравнительные графики. Для графика Cl установить диапазон по y от -2 до 2.
    """
    # Считываем измеренные данные
    alphas_data, cls_data, cds_data = parse_bound_viterna_file(input_file)
    if len(alphas_data) < 2:
        print(f"В файле {input_file} слишком мало корректных данных для обработки.")
        return

    alphas_data, cls_data, cds_data = remove_duplicates(alphas_data, cls_data, cds_data)
    sort_indices = np.argsort(alphas_data)
    alphas_data = alphas_data[sort_indices]
    cls_data = cls_data[sort_indices]
    cds_data = cds_data[sort_indices]

    # Определяем границу для отрицательных углов: локальный минимум, ближайший к -10°
    left_candidates = []
    for i in range(1, len(alphas_data) - 1):
        if alphas_data[i] < 0 and cls_data[i] < cls_data[i - 1] and cls_data[i] < cls_data[i + 1]:
            left_candidates.append(i)
    if left_candidates:
        left_stall_index = min(left_candidates, key=lambda i: abs(alphas_data[i] - (-10)))
    else:
        neg_indices = [i for i in range(len(alphas_data)) if alphas_data[i] < 0]
        left_stall_index = min(neg_indices, key=lambda i: abs(alphas_data[i] - (-10))) if neg_indices else 0

    # Определяем границу для положительных углов: локальный максимум, ближайший к 10°
    right_candidates = []
    for i in range(1, len(alphas_data) - 1):
        if alphas_data[i] > 0 and cls_data[i] > cls_data[i - 1] and cls_data[i] > cls_data[i + 1]:
            right_candidates.append(i)
    if right_candidates:
        right_stall_index = min(right_candidates, key=lambda i: abs(alphas_data[i] - 10))
    else:
        pos_indices = [i for i in range(len(alphas_data)) if alphas_data[i] > 0]
        right_stall_index = min(pos_indices, key=lambda i: abs(alphas_data[i] - 10)) if pos_indices else len(
            alphas_data) - 1

    # Границы срыва и соответствующие значения Cl и Cd
    alpha_left = alphas_data[left_stall_index]
    alpha_right = alphas_data[right_stall_index]
    cl_stall_left = cls_data[left_stall_index]
    cd_stall_left = cds_data[left_stall_index]
    cl_stall_right = cls_data[right_stall_index]
    cd_stall_right = cds_data[right_stall_index]

    print(f"Определены границы срыва: левая = {alpha_left:.2f}°, правая = {alpha_right:.2f}°")

    # Создаем интерполяционные функции по измеренным данным
    cl_interp_func, cd_interp_func = build_interpolators(alphas_data, cls_data, cds_data)

    # Параметр cd_max (из метода Витерны)
    cd_max = 1.27

    # Вычисляем витерну для всей сетки углов
    final_alphas = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)
    final_cls = []
    final_cds = []
    for alpha_deg in final_alphas:
        cl_val, cd_val = compute_cl_cd(
            alpha_deg,
            alpha_left,
            alpha_right,
            cl_interp_func,
            cd_interp_func,
            cl_stall_left,
            cd_stall_left,
            cl_stall_right,
            cd_stall_right,
            cd_max
        )
        final_cls.append(cl_val)
        final_cds.append(cd_val)

    final_alphas = np.array(final_alphas)
    final_cls = np.array(final_cls)
    final_cds = np.array(final_cds)

    # Замена крайних точек:
    # Заменяем значение для -180 на значение, соответствующее -179.5,
    # и значение для 180 на значение, соответствующее 179.5.
    mask_neg1795 = np.isclose(final_alphas, -179.5)
    mask_pos1795 = np.isclose(final_alphas, 179.5)
    if mask_neg1795.any():
        ref_cls_neg = final_cls[mask_neg1795][0]
        ref_cds_neg = final_cds[mask_neg1795][0]
        final_cls[0] = ref_cls_neg
        final_cds[0] = ref_cds_neg
    if mask_pos1795.any():
        ref_cls_pos = final_cls[mask_pos1795][0]
        ref_cds_pos = final_cds[mask_pos1795][0]
        final_cls[-1] = ref_cls_pos
        final_cds[-1] = ref_cds_pos

    # Сохранение результатов в файл
    save_to_bound_viterna_format(output_file, final_alphas, final_cls, final_cds)

    # Построение сравнительных графиков
    plt.figure(figsize=(12, 5))

    # График Cl vs. α с ограничением по y от -2 до 2
    plt.subplot(1, 2, 1)
    plt.plot(alphas_data, cls_data, 'ro', label='Измеренные данные')
    plt.plot(final_alphas, final_cls, 'b-', label='Сшитые (Витерна+интерп.)')
    plt.ylim(-2, 2)
    plt.title('Cl vs. Alpha')
    plt.xlabel('Alpha (deg)')
    plt.ylabel('Cl')
    plt.legend()
    plt.grid(True)
    # Отмечаем границы срыва
    plt.axvline(alpha_left, color='g', linestyle='--', label='Срыв (левая)')
    plt.axvline(alpha_right, color='m', linestyle='--', label='Срыв (правая)')

    # График Cd vs. α
    plt.subplot(1, 2, 2)
    plt.plot(alphas_data, cds_data, 'ro', label='Измеренные данные')
    plt.plot(final_alphas, final_cds, 'b-', label='Сшитые (Витерна+интерп.)')
    plt.title('Cd vs. Alpha')
    plt.xlabel('Alpha (deg)')
    plt.ylabel('Cd')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"Результаты сохранены в файл: {output_file}")
    print(f"График сохранён в файл: {plot_file}")


# ===== Функция преобразования файлов Calculation.py =====

def convert_calc_results_to_bound_format(calc_file, bound_file):
    """
    Преобразует файл, сохранённый Calculation.py (с заголовками вида "# ..."),
    в формат, требуемый для метода Витерны:
      // Bound.+Viterna
      // Re = 1.00E+05
      // (alpha_deg cl cd)
      (alpha cl cd)
    """
    data_lines = []
    with open(calc_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.lstrip().startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    alpha = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    data_lines.append(f"({alpha:.2f} {cl:.4f} {cd:.4f})\n")
                except Exception as e:
                    print(f"Ошибка преобразования строки '{line.strip()}': {e}")

    with open(bound_file, 'w', encoding='utf-8') as f:
        f.write("// Bound.+Viterna\n")
        f.write("// Re = 1.00E+05\n")
        f.write("// (alpha_deg cl cd)\n")
        for line in data_lines:
            f.write(line)


# ===== Основной блок =====

if __name__ == "__main__":
    # Папки с исходными и результирующими файлами
    results_folder = "results"
    converted_folder = "converted_results"
    viterna_output_folder = "viterna_output"

    os.makedirs(converted_folder, exist_ok=True)
    os.makedirs(viterna_output_folder, exist_ok=True)

    # Перебираем все файлы результатов (с суффиксом _results.txt)
    for filename in os.listdir(results_folder):
        if filename.endswith("_results.txt"):
            calc_file = os.path.join(results_folder, filename)
            bound_filename = filename.replace("_results.txt", "_bound.txt")
            bound_file = os.path.join(converted_folder, bound_filename)

            print(f"Преобразуем {calc_file} -> {bound_file}")
            convert_calc_results_to_bound_format(calc_file, bound_file)

            output_filename = filename.replace("_results.txt", "_viterna.txt")
            plot_filename = filename.replace("_results.txt", "_viterna_plot.png")
            output_file = os.path.join(viterna_output_folder, output_filename)
            plot_file = os.path.join(viterna_output_folder, plot_filename)

            print(f"Запуск обработки по Витерне для {bound_file}")
            main_bound_viterna_processing(
                input_file=bound_file,
                output_file=output_file,
                plot_file=plot_file,
                alpha_min=-180,
                alpha_max=180,
                alpha_step=0.5
            )
            print(f"Результат сохранён: {output_file}\n")
