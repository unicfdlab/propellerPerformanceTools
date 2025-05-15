import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.interpolate import interp1d


def load_geometry_file():
    """
    Открывает диалоговое окно для выбора файла с данными геометрии сечений.
    Ожидаемый формат (текстовый, разделитель – пробел или табуляция):
      r       c       theta   aero_file
      0.01    0.003   40      C:/path/to/section1.txt
      0.012   0.003   38      C:/path/to/section2.txt
      ...
      0.025   0.003   20      C:/path/to/section15.txt

    Если заголовки отсутствуют, функция загрузит данные без заголовка и задаст имена столбцов:
    ['r', 'c', 'theta', 'aero_file'].
    """
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Выберите файл с данными геометрии сечений (текстовый формат)",
        filetypes=[("Text files", "*.txt *.dat"), ("All files", "*.*")]
    )
    root.destroy()
    if not file_path:
        raise Exception("Файл не выбран.")
    try:
        geometry_df = pd.read_csv(file_path, delim_whitespace=True)
        if 'r' not in geometry_df.columns:
            raise ValueError("Отсутствует столбец 'r'.")
    except Exception as e:
        print("Не удалось прочитать заголовки из файла. Попытка загрузки без заголовка.")
        geometry_df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        geometry_df.columns = ['r', 'c', 'theta', 'aero_file']
    return geometry_df


def load_aero_data(file_path):
    """
    Загружает аэродинамические данные из текстового файла.
    Ожидаемый формат файла:

    // Bound.+Viterna
    // Re = 1.00E+05
    // (alpha_deg cl cd)
    (-180.00 -0.4621 -0.0316)
    (-179.50 -0.4621 -0.0316)
    (-179.00 -0.2144 -0.0313)
    (-178.50 -0.1244 -0.0308)
    ...

    Функция игнорирует строки, начинающиеся с '//' или '#' и пустые строки.
    Для строк с данными удаляются скобки и производится разбиение по пробелам.
    """
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(('#', '//')):
                continue
            # Удаляем круглые скобки, если они присутствуют
            if s.startswith('(') and s.endswith(')'):
                s = s[1:-1]
            # Разбиваем по пробелам (любым пробельным символам)
            parts = s.split()
            try:
                # Преобразуем каждую часть в число с плавающей точкой
                values = [float(val) for val in parts]
                if len(values) >= 3:
                    data_list.append(values[:3])
            except Exception as e:
                print(f"Ошибка преобразования строки: {s}\n{e}")
    data = np.array(data_list)
    # Предполагаем, что данные имеют три колонки: alpha, Cl, Cd
    alpha = data[:, 0]
    Cl = data[:, 1]
    Cd = data[:, 2]
    return alpha, Cl, Cd


def blade_element_analysis_from_files(geometry_df, omega, V_axial, rho=1000):
    """
    Расчёт распределения элементарной тяги и крутящего момента по сечениям лопасти.

    geometry_df : DataFrame с данными по сечениям (столбцы: r, c, theta, aero_file)
    omega       : угловая скорость (рад/с)
    V_axial     : осевая скорость (м/с)
    rho         : плотность воздуха (по умолчанию 1.225 кг/м³)

    Возвращает:
      T_total   : суммарная тяга (Н) для одной лопасти
      Q_total   : суммарный крутящий момент (Н·м) для одной лопасти
      r_list    : список радиусов сечений (м)
      dT_list   : список элементарных вкладов тяги (Н) по сечениям
      dQ_list   : список элементарных вкладов крутящего момента (Н·м) по сечениям
    """
    # Сортируем данные по возрастанию радиуса (от корня к законцовке)
    geometry_df = geometry_df.sort_values(by='r')

    dT_list = []
    dQ_list = []
    r_list = []
    T_total = 0.0
    Q_total = 0.0

    # Вычисляем шаг по радиусу для каждого сечения
    r_values = geometry_df['r'].values
    N = len(r_values)
    dr_arr = []
    for i in range(N - 1):
        dr_arr.append(r_values[i + 1] - r_values[i])
    if N > 1:
        dr_arr.append(dr_arr[-1])
    else:
        dr_arr.append(0.0)
    geometry_df['dr'] = dr_arr

    for idx, row in geometry_df.iterrows():
        r_i = row['r']
        c_i = row['c']
        theta_deg = row['theta']
        aero_file = row['aero_file']
        dr_i = row['dr']

        # Загружаем аэродинамические данные для данного сечения
        alpha_arr, Cl_arr, Cd_arr = load_aero_data(aero_file)
        # Интерполяция коэффициентов по углу атаки (угол в градусах)
        f_Cl = interp1d(alpha_arr, Cl_arr, kind='linear', fill_value="extrapolate")
        f_Cd = interp1d(alpha_arr, Cd_arr, kind='linear', fill_value="extrapolate")

        # Расчёт локальной скорости
        V_tangential = omega * r_i
        V_local = math.sqrt(V_tangential ** 2 + V_axial ** 2)

        # Угол набегающего потока phi
        if abs(V_axial) < 1e-9:
            phi = math.pi / 2.0
        else:
            phi = math.atan(V_axial / V_tangential)

        # Перевод угла установки в радианы
        theta_rad = math.radians(theta_deg)
        # Определяем угол атаки: alpha = theta - phi (в радианах)
        alpha_rad = phi - theta_rad
        alpha_deg = math.degrees(alpha_rad)

        # Получаем коэффициенты по интерполяции
        Cl = f_Cl(alpha_deg)
        Cd = f_Cd(alpha_deg)

        # Расчёт элементарных сил
        dL = 0.5 * rho * (V_local ** 2) * Cl * c_i * dr_i
        dD = 0.5 * rho * (V_local ** 2) * Cd * c_i * dr_i

        # Проекция на осевую: элементарная тяга
        dT = dL * math.cos(phi) - dD * math.sin(phi)
        # Элементарный крутящий момент относительно оси
        dQ = r_i * (dL * math.sin(phi) + dD * math.cos(phi))

        T_total += dT
        Q_total += dQ

        dT_list.append(dT)
        dQ_list.append(dQ)
        r_list.append(r_i)

    return T_total, Q_total, r_list, dT_list, dQ_list


def plot_distribution(r_list, dT_list):
    """
    Строит графики распределения элементарной тяги и накопленной тяги вдоль лопасти.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(r_list, dT_list, marker='o', label='dT (элементарная тяга)')
    plt.xlabel('Радиус, м')
    plt.ylabel('dT, Н')
    plt.title('Распределение элементарной тяги по лопасти')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    T_cumulative = np.cumsum(dT_list)
    plt.figure(figsize=(7, 5))
    plt.plot(r_list, T_cumulative, marker='o', color='red', label='Тяга')
    plt.xlabel('Радиус, м')
    plt.ylabel('Тяга, Н')
    plt.title('Тяга по лопасти')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Выберите файл с данными геометрии сечений (текстовый формат).")
    geometry_df = load_geometry_file()

    omega = 52.36
    V_axial = 0.9
    B = 4

    T_one_blade, Q_one_blade, r_list, dT_list, dQ_list = blade_element_analysis_from_files(
        geometry_df, omega, V_axial
    )

    print("\nРезультаты для одной лопасти:")
    print(f"Суммарная тяга, Н: {T_one_blade:.4f}")
    print(f"Суммарный момент, Н·м: {Q_one_blade:.6f}")

    print("\nРезультаты для всего винта:")
    print(f"Тяга, Н: {B * T_one_blade:.4f}")
    print(f"Момент, Н·м: {B * Q_one_blade:.6f}")

    plot_distribution(r_list, dT_list)
