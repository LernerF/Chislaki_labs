"""
ЛАБОРАТОРНАЯ РАБОТА: ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ
=================================================
Сравнение различных разностных схем для вычисления первой и второй производных
Функция: y = (3e^(-2x) + 5)^(1/3) / (4x² + e^(x/4) + 1)^(1/2)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # Отключаем предупреждения для чистоты вывода


def f(x):
    """
    Исходная функция: f(x) = (3e^(-2x) + 5)^(1/3) / sqrt(4x² + e^(x/4) + 1)
    
    Параметры:
        x: число или массив чисел
    Возвращает:
        значение функции в точке(ах) x
    """
    # Числитель: кубический корень из (3e^(-2x) + 5)
    numerator = (3 * np.exp(-2 * x) + 5) ** (1 / 3)
    # Знаменатель: квадратный корень из (4x² + e^(x/4) + 1)
    denominator = np.sqrt(4 * x ** 2 + np.exp(x / 4) + 1)
    
    # Обработка массивов и скаляров
    if isinstance(x, np.ndarray):
        # Если на вход подан массив, вычисляем для каждого элемента
        result = np.zeros_like(x, dtype=float)
        for idx, xi in enumerate(x):
            num = (3 * np.exp(-2 * xi) + 5) ** (1 / 3)
            denom = np.sqrt(4 * xi ** 2 + np.exp(xi / 4) + 1)
            result[idx] = num / denom
        return result
    else:
        # Если на вход подан скаляр, возвращаем одно значение
        return numerator / denominator


def f_prime_analytical(x, h=1e-6):
    """
    Вычисление первой производной с помощью 5-точечной формулы
    (используется как аналитический эталон)
    
    Формула: f'(x) ≈ [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h)
    Порядок точности: O(h^4)
    """
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def f_double_prime_analytical(x, h=1e-6):
    """
    Вычисление второй производной с помощью 5-точечной формулы
    (используется как аналитический эталон)
    
    Формула: f''(x) ≈ [-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)] / (12h²)
    Порядок точности: O(h^4)
    """
    return (-f(x - 2 * h) + 16 * f(x - h) - 30 * f(x) + 16 * f(x + h) - f(x + 2 * h)) / (12 * h ** 2)


# ==================== РАЗНОСТНЫЕ СХЕМЫ ДЛЯ ПЕРВОЙ ПРОИЗВОДНОЙ ====================
def scheme1_right_diff(y, h, i):
    """
    Схема 1: ПРАВАЯ РАЗНОСТЬ (односторонняя)
    Формула: f'(x_i) ≈ (y_{i+1} - y_i) / h
    Порядок точности: O(h)
    """
    if i == len(y) - 1:  # Для последней точки формула не работает
        return np.nan
    return (y[i + 1] - y[i]) / h


def scheme2_left_diff(y, h, i):
    """
    Схема 2: ЛЕВАЯ РАЗНОСТЬ (односторонняя)
    Формула: f'(x_i) ≈ (y_i - y_{i-1}) / h
    Порядок точности: O(h)
    """
    if i == 0:  # Для первой точки формула не работает
        return np.nan
    return (y[i] - y[i - 1]) / h


def scheme3_central_diff(y, h, i):
    """
    Схема 3: ЦЕНТРАЛЬНАЯ РАЗНОСТЬ
    Формула: f'(x_i) ≈ (y_{i+1} - y_{i-1}) / (2h)
    Порядок точности: O(h²)
    """
    if i == 0 or i == len(y) - 1:  # Для граничных точек формула не работает
        return np.nan
    return (y[i + 1] - y[i - 1]) / (2 * h)


def scheme4_3point_left(y, h, i):
    """
    Схема 4: АСИММЕТРИЧНАЯ ЛЕВАЯ 3-ТОЧЕЧНАЯ
    Формула: f'(x_i) ≈ (y_{i+1} - y_{i-1}) / (2h)
    (фактически то же, что и центральная, для внутренних точек)
    Порядок точности: O(h²)
    """
    if i == 0 or i == len(y) - 1:
        return np.nan
    return (y[i + 1] - y[i - 1]) / (2 * h)


def scheme5_3point_right(y, h, i):
    """
    Схема 5: АСИММЕТРИЧНАЯ ПРАВАЯ 3-ТОЧЕЧНАЯ
    Для первой точки: f'(x₀) ≈ (-3y₀ + 4y₁ - y₂) / (2h)
    Для последней точки: f'(x_n) ≈ (3y_n - 4y_{n-1} + y_{n-2}) / (2h)
    Для внутренних точек: центральная разность
    Порядок точности: O(h²)
    """
    if i == 0:
        # Левая граница: используем три точки справа
        if len(y) < 3:
            return np.nan
        return (-3 * y[i] + 4 * y[i + 1] - y[i + 2]) / (2 * h)
    elif i == len(y) - 1:
        # Правая граница: используем три точки слева
        if len(y) < 3:
            return np.nan
        return (3 * y[i] - 4 * y[i - 1] + y[i - 2]) / (2 * h)
    else:
        # Внутренние точки: центральная разность
        return (y[i + 1] - y[i - 1]) / (2 * h)


def scheme6_4point_central(y, h, i):
    """
    Схема 6: 4-ТОЧЕЧНАЯ ЦЕНТРАЛЬНАЯ
    Формула: f'(x_i) ≈ (-2y_{i-1} - 3y_i + 6y_{i+1} - y_{i+2}) / (6h)
    Порядок точности: O(h³)
    """
    if i == 0 or i > len(y) - 3:  # Нужны точки i-1, i, i+1, i+2
        return np.nan
    return (-2 * y[i - 1] - 3 * y[i] + 6 * y[i + 1] - y[i + 2]) / (6 * h)


# Список всех схем для первой производной
schemes_first = [
    ("Правая разность", scheme1_right_diff, 1),
    ("Левая разность", scheme2_left_diff, 1),
    ("Центральная разность", scheme3_central_diff, 2),
    ("Асимм. левая", scheme4_3point_left, 2),
    ("Асимм. правая", scheme5_3point_right, 2),
    ("4-точечная", scheme6_4point_central, 3),
]


# ==================== РАЗНОСТНЫЕ СХЕМЫ ДЛЯ ВТОРОЙ ПРОИЗВОДНОЙ ====================
def scheme_2d_1_3point_right(y, h, i):
    """
    Схема 1: АСИММЕТРИЧНАЯ ПРАВАЯ 3-ТОЧЕЧНАЯ
    Формула: f''(x_i) ≈ (y_i - 2y_{i+1} + y_{i+2}) / h²
    Порядок точности: O(h)
    """
    if i > len(y) - 3:  # Нужны точки i, i+1, i+2
        return np.nan
    return (y[i] - 2 * y[i + 1] + y[i + 2]) / (h ** 2)


def scheme_2d_2_3point_left(y, h, i):
    """
    Схема 2: АСИММЕТРИЧНАЯ ЛЕВАЯ 3-ТОЧЕЧНАЯ
    Формула: f''(x_i) ≈ (y_{i-2} - 2y_{i-1} + y_i) / h²
    Порядок точности: O(h)
    """
    if i < 2:  # Нужны точки i-2, i-1, i
        return np.nan
    return (y[i - 2] - 2 * y[i - 1] + y[i]) / (h ** 2)


def scheme_2d_3_3point_central(y, h, i):
    """
    Схема 3: ЦЕНТРАЛЬНАЯ 3-ТОЧЕЧНАЯ
    Формула: f''(x_i) ≈ (y_{i+1} - 2y_i + y_{i-1}) / h²
    Порядок точности: O(h²)
    """
    if i == 0 or i == len(y) - 1:  # Для границ формула не работает
        return np.nan
    return (y[i + 1] - 2 * y[i] + y[i - 1]) / (h ** 2)


def scheme_2d_4_5point_central(y, h, i):
    """
    Схема 4: 5-ТОЧЕЧНАЯ ЦЕНТРАЛЬНАЯ
    Формула: f''(x_i) ≈ [-y_{i+2} + 16y_{i+1} - 30y_i + 16y_{i-1} - y_{i-2}] / (12h²)
    Порядок точности: O(h⁴)
    """
    if i < 2 or i > len(y) - 3:  # Нужны точки i-2, i-1, i, i+1, i+2
        return np.nan
    return (-y[i + 2] + 16 * y[i + 1] - 30 * y[i] + 16 * y[i - 1] - y[i - 2]) / (12 * h ** 2)


# Список всех схем для второй производной
schemes_second = [
    ("Асимм. правая", scheme_2d_1_3point_right, 1),
    ("Асимм. левая", scheme_2d_2_3point_left, 1),
    ("Центральная", scheme_2d_3_3point_central, 2),
    ("5-точечная", scheme_2d_4_5point_central, 4),
]


def print_full_table(x_points, analytical_first, analytical_second, 
                     results_first, results_second, h):
    """
    Выводит полную таблицу всех производных для всех точек
    
    Параметры:
        x_points: массив узлов сетки
        analytical_first: аналитические значения первой производной
        analytical_second: аналитические значения второй производной
        results_first: словарь с результатами схем для первой производной
        results_second: словарь с результатами схем для второй производной
        h: шаг сетки
    """
    print(f"\n{'='*120}")
    print(f"ПОЛНАЯ ТАБЛИЦА ПРОИЗВОДНЫХ ДЛЯ ШАГА h = {h}")
    print(f"{'='*120}\n")
    
    # Выводим информацию о функции
    print("Исходная функция: f(x) = (3e^(-2x) + 5)^(1/3) / sqrt(4x² + e^(x/4) + 1)")
    print(f"Интервал: [{x_points[0]:.2f}, {x_points[-1]:.2f}]")
    print(f"Количество точек: {len(x_points)}")
    print(f"Шаг сетки: {h}")
    print("\n" + "-"*120)
    
    # Спрашиваем пользователя, хочет ли он видеть все точки
    print("Внимание: будет выведено", len(x_points), "точек.")
    response = input("Вывести полную таблицу? (y/n, по умолчанию n): ").strip().lower()
    
    if response != 'y':
        print("Пропускаем полную таблицу. Используйте другие таблицы для анализа.")
        return
    
    # Для каждой точки выводим таблицу
    for idx in range(len(x_points)):
        x_val = x_points[idx]
        
        print(f"\n{'='*120}")
        print(f"ТОЧКА {idx + 1}: x = {x_val:.6f}")
        print(f"{'='*120}")
        
        # Значение функции
        f_val = f(x_val)
        print(f"\nЗначение функции: f({x_val:.6f}) = {f_val:.10f}")
        
        # Первая производная
        print(f"\n--- ПЕРВАЯ ПРОИЗВОДНАЯ ---")
        print(f"{'Схема':<25} | {'Значение':<15} | {'Абсолютная ошибка':<18} | {'Относительная ошибка':<20} | {'Порядок':<8}")
        print("-" * 95)
        
        # Аналитическое значение
        analytical_f_prime = analytical_first[idx]
        print(f"{'АНАЛИТИЧЕСКАЯ':<25} | {analytical_f_prime:15.10f} | {'---':<18} | {'---':<20} | {'---':<8}")
        
        # Значения по разностным схемам
        for name, _, order in schemes_first:
            val = results_first[name][idx]
            if not np.isnan(val):
                abs_err = abs(analytical_f_prime - val)
                rel_err = (abs_err / abs(analytical_f_prime)) * 100 if analytical_f_prime != 0 else 0
                print(f"{name:<25} | {val:15.10f} | {abs_err:18.10f} | {rel_err:19.6f}% | O(h^{order})")
            else:
                print(f"{name:<25} | {'N/A':15} | {'N/A':18} | {'N/A':20} | {'N/A':<8}")
        
        # Вторая производная
        print(f"\n--- ВТОРАЯ ПРОИЗВОДНАЯ ---")
        print(f"{'Схема':<25} | {'Значение':<15} | {'Абсолютная ошибка':<18} | {'Относительная ошибка':<20} | {'Порядок':<8}")
        print("-" * 95)
        
        # Аналитическое значение
        analytical_f_double_prime = analytical_second[idx]
        print(f"{'АНАЛИТИЧЕСКАЯ':<25} | {analytical_f_double_prime:15.10f} | {'---':<18} | {'---':<20} | {'---':<8}")
        
        # Значения по разностным схемам
        for name, _, order in schemes_second:
            val = results_second[name][idx]
            if not np.isnan(val):
                abs_err = abs(analytical_f_double_prime - val)
                rel_err = (abs_err / abs(analytical_f_double_prime)) * 100 if analytical_f_double_prime != 0 else 0
                print(f"{name:<25} | {val:15.10f} | {abs_err:18.10f} | {rel_err:19.6f}% | O(h^{order})")
            else:
                print(f"{name:<25} | {'N/A':15} | {'N/A':18} | {'N/A':20} | {'N/A':<8}")
        
        # Пауза после каждой точки для удобства просмотра
        if idx < len(x_points) - 1:
            input("\nНажмите Enter для продолжения к следующей точке...")
    
    print(f"\n{'='*120}")
    print("КОНЕЦ ТАБЛИЦЫ")
    print(f"{'='*120}\n")


def print_summary_table(x_points, analytical_first, analytical_second,
                        results_first, results_second, h):
    """
    Выводит сводную таблицу погрешностей для всех схем
    
    Параметры:
        x_points: массив узлов сетки
        analytical_first: аналитические значения первой производной
        analytical_second: аналитические значения второй производной
        results_first: словарь с результатами схем для первой производной
        results_second: словарь с результатами схем для второй производной
        h: шаг сетки
    """
    print(f"\n{'='*100}")
    print(f"СВОДНАЯ ТАБЛИЦА ПОГРЕШНОСТЕЙ ДЛЯ ШАГА h = {h}")
    print(f"{'='*100}\n")
    
    # Сводка для первой производной
    print("--- ПЕРВАЯ ПРОИЗВОДНАЯ ---")
    print(f"{'Схема':<25} | {'Средняя абс. ошибка':<20} | {'Макс. абс. ошибка':<20} | {'Ср. отн. ошибка (%)':<20}")
    print("-" * 85)
    
    for name, _, order in schemes_first:
        errors = []
        rel_errors = []
        for idx in range(len(x_points)):
            val = results_first[name][idx]
            analytical = analytical_first[idx]
            if not np.isnan(val):
                abs_err = abs(analytical - val)
                errors.append(abs_err)
                if analytical != 0:
                    rel_errors.append((abs_err / abs(analytical)) * 100)
        
        if errors:
            avg_err = np.mean(errors)
            max_err = np.max(errors)
            avg_rel_err = np.mean(rel_errors) if rel_errors else 0
            print(f"{name:<25} | {avg_err:19.10f} | {max_err:19.10f} | {avg_rel_err:19.6f}")
        else:
            print(f"{name:<25} | {'N/A':20} | {'N/A':20} | {'N/A':20}")
    
    # Сводка для второй производной
    print(f"\n--- ВТОРАЯ ПРОИЗВОДНАЯ ---")
    print(f"{'Схема':<25} | {'Средняя абс. ошибка':<20} | {'Макс. абс. ошибка':<20} | {'Ср. отн. ошибка (%)':<20}")
    print("-" * 85)
    
    for name, _, order in schemes_second:
        errors = []
        rel_errors = []
        for idx in range(len(x_points)):
            val = results_second[name][idx]
            analytical = analytical_second[idx]
            if not np.isnan(val):
                abs_err = abs(analytical - val)
                errors.append(abs_err)
                if analytical != 0:
                    rel_errors.append((abs_err / abs(analytical)) * 100)
        
        if errors:
            avg_err = np.mean(errors)
            max_err = np.max(errors)
            avg_rel_err = np.mean(rel_errors) if rel_errors else 0
            print(f"{name:<25} | {avg_err:19.10f} | {max_err:19.10f} | {avg_rel_err:19.6f}")
        else:
            print(f"{name:<25} | {'N/A':20} | {'N/A':20} | {'N/A':20}")
    
    print(f"\n{'='*100}\n")


def plot_all(x_points, y_points, analytical_first, analytical_second,
             results_first, results_second, h, suffix):
    """
    Построение графиков всех разностных схем
    
    Параметры:
        x_points: массив узлов сетки
        y_points: значения функции в узлах
        analytical_first: аналитические значения первой производной
        analytical_second: аналитические значения второй производной
        results_first: словарь с результатами схем для первой производной
        results_second: словарь с результатами схем для второй производной
        h: шаг сетки
        suffix: суффикс для имени файла
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Левый график: первая производная
    axes[0].plot(x_points, analytical_first, 'k-', label='Аналитическая', linewidth=3)
    colors_first = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    for idx, (name, _, _) in enumerate(schemes_first):
        axes[0].plot(x_points, results_first[name], color=colors_first[idx % len(colors_first)], 
                    label=name, linewidth=1.5, alpha=0.7)
    axes[0].set_title(f"Первые производные (h={h})")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel("f'(x)")
    axes[0].legend(fontsize=8, loc='best')
    axes[0].grid(True, alpha=0.35)
    
    # Правый график: вторая производная
    axes[1].plot(x_points, analytical_second, 'k-', label='Аналитическая', linewidth=3)
    colors_second = ['red', 'green', 'purple', 'brown']
    for idx, (name, _, _) in enumerate(schemes_second):
        axes[1].plot(x_points, results_second[name], color=colors_second[idx % len(colors_second)], 
                    label=name, linewidth=1.5, alpha=0.7)
    axes[1].set_title(f"Вторые производные (h={h})")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("f''(x)")
    axes[1].legend(fontsize=8, loc='best')
    axes[1].grid(True, alpha=0.35)
    
    plt.suptitle(f"Графики производных для шага h={h}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f'graphs_derivatives_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_comparison_table(x_points, analytical_first, analytical_second,
                          results_first, results_second, h):
    """
    Выводит сравнительную таблицу для выбранных точек (границы и середина)
    
    Параметры:
        x_points: массив узлов сетки
        analytical_first: аналитические значения первой производной
        analytical_second: аналитические значения второй производной
        results_first: словарь с результатами схем для первой производной
        results_second: словарь с результатами схем для второй производной
        h: шаг сетки
    """
    # Выбираем ключевые точки: начало, середина, конец
    idx_points = [0, len(x_points)//2, -1]
    point_names = ["Левая граница", "Середина интервала", "Правая граница"]
    
    print(f"\n{'='*100}")
    print(f"СРАВНИТЕЛЬНАЯ ТАБЛИЦА ДЛЯ КЛЮЧЕВЫХ ТОЧЕК (h = {h})")
    print(f"{'='*100}\n")
    
    for idx, point_name in zip(idx_points, point_names):
        x_val = x_points[idx]
        print(f"\n--- {point_name}: x = {x_val:.6f} ---")
        
        # Первая производная
        print(f"\nПервая производная:")
        print(f"{'Схема':<25} | {'Значение':<15} | {'Ошибка':<15}")
        print("-" * 55)
        print(f"{'АНАЛИТИЧЕСКАЯ':<25} | {analytical_first[idx]:15.10f} | {'---':<15}")
        
        for name, _, _ in schemes_first:
            val = results_first[name][idx]
            if not np.isnan(val):
                err = abs(analytical_first[idx] - val)
                print(f"{name:<25} | {val:15.10f} | {err:15.10f}")
            else:
                print(f"{name:<25} | {'N/A':15} | {'N/A':15}")
        
        # Вторая производная
        print(f"\nВторая производная:")
        print(f"{'Схема':<25} | {'Значение':<15} | {'Ошибка':<15}")
        print("-" * 55)
        print(f"{'АНАЛИТИЧЕСКАЯ':<25} | {analytical_second[idx]:15.10f} | {'---':<15}")
        
        for name, _, _ in schemes_second:
            val = results_second[name][idx]
            if not np.isnan(val):
                err = abs(analytical_second[idx] - val)
                print(f"{name:<25} | {val:15.10f} | {err:15.10f}")
            else:
                print(f"{name:<25} | {'N/A':15} | {'N/A':15}")
        
        print("-" * 55)


def main():
    """
    Основная функция программы.
    Выполняет расчеты для двух шагов сетки: h=0.3 и h=0.15
    """
    # Перебираем два шага сетки
    for h, suffix in [(0.30, "h0.3"), (0.15, "h0.15")]:
        # Определяем интервал [-2, 4]
        x_start, x_end = -2.0, 4.0
        
        # Создаем сетку с шагом h
        x_points = np.arange(x_start, x_end + h/2, h)
        
        # Вычисляем значения функции в узлах
        y_points = np.array([f(x) for x in x_points])
        
        # Вычисляем аналитические производные (эталон)
        analytical_first = np.array([f_prime_analytical(x) for x in x_points])
        analytical_second = np.array([f_double_prime_analytical(x) for x in x_points])
        
        # Вычисляем численные производные для всех схем (шаг h)
        results_first = {}
        results_second = {}
        
        for name, scheme_func, _ in schemes_first:
            results_first[name] = np.array([scheme_func(y_points, h, i)
                                           for i in range(len(x_points))])
        
        for name, scheme_func, _ in schemes_second:
            results_second[name] = np.array([scheme_func(y_points, h, i)
                                            for i in range(len(x_points))])
        
        # Выводим результаты
        print(f"\n{'='*30}\nШАГ СЕТКИ h = {h}\n{'='*30}\n")
        
        # 1. Сравнительная таблица для ключевых точек
        print_comparison_table(x_points, analytical_first, analytical_second,
                              results_first, results_second, h)
        
        # 2. Сводная таблица погрешностей
        print_summary_table(x_points, analytical_first, analytical_second,
                           results_first, results_second, h)
        
        # 3. Полная таблица для всех точек (с запросом подтверждения)
        print_full_table(x_points, analytical_first, analytical_second,
                        results_first, results_second, h)
        
        # 4. Строим графики
        plot_all(x_points, y_points, analytical_first, analytical_second,
                results_first, results_second, h, suffix)
        
        # Пауза для удобства просмотра
        if h == 0.30:
            print("\n" + "="*50)
            input("Нажмите Enter для продолжения со следующим шагом h=0.15...")
            print("="*50)


# Точка входа в программу
if __name__ == "__main__":
    main()