import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3 * math.cos(3 * x) - 2 * (x + 1) ** 2 + 7 * x + 3


def fp(x):
    return -9 * math.sin(3 * x) - 4 * x + 3


def fpp(x):
    return -27 * math.cos(3 * x) - 4


eps = 1e-6  # Точность


# Функция для автоматического поиска интервалов, где функция меняет знак
def find_sign_change_intervals(a, b, num_points=1000):
    """Находит интервалы, где функция меняет знак"""
    x_vals = np.linspace(a, b, num_points)
    y_vals = np.array([f(x) for x in x_vals])

    intervals = []
    for i in range(len(x_vals) - 1):
        if y_vals[i] * y_vals[i + 1] <= 0:  # Знак меняется или есть ноль
            intervals.append((x_vals[i], x_vals[i + 1]))

    # Объединяем пересекающиеся интервалы
    merged_intervals = []
    if intervals:
        merged_intervals = [list(intervals[0])]
        for interval in intervals[1:]:
            if interval[0] <= merged_intervals[-1][1]:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
            else:
                merged_intervals.append(list(interval))

    return merged_intervals


# Проверка достаточного условия сходимости для методов Ньютона, секущих, хорд
def check_newton_secant_chord(a, b):
    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        return False, "Функция имеет одинаковый знак на концах отрезка", None

    fp_a = fp(a)
    fp_b = fp(b)
    fpp_a = fpp(a)
    fpp_b = fpp(b)

    if fp_a * fp_b <= 0:
        return False, "Первая производная меняет знак", None

    if fpp_a * fpp_b <= 0:
        return False, "Вторая производная меняет знак", None

    lhs_a = abs(fa * fpp_a)
    rhs_a = fp_a ** 2
    check_a = lhs_a < rhs_a

    lhs_b = abs(fb * fpp_b)
    rhs_b = fp_b ** 2
    check_b = lhs_b < rhs_b

    start_point = None
    if fa * fpp_a > 0:
        start_point = 'a'
    elif fb * fpp_b > 0:
        start_point = 'b'

    if check_a and check_b:
        if start_point:
            return True, "Условие сходимости выполнено", start_point
        else:
            return False, "Нет подходящей начальной точки", None
    elif check_a and not check_b:
        if start_point == 'a':
            return True, "Условие выполнено в точке a", start_point
        else:
            return False, "Не выполнено условие начальной точки", None
    elif not check_a and check_b:
        if start_point == 'b':
            return True, "Условие выполнено в точке b", start_point
        else:
            return False, "Не выполнено условие начальной точки", None
    else:
        return False, "Условие сходимости не выполнено", None


# Метод дихотомии
def bisection(a, b, eps):
    if f(a) * f(b) >= 0:
        return None, 0

    iterations = 0
    while (b - a) >= eps:
        iterations += 1
        mid = (a + b) / 2
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    root = (a + b) / 2
    return root, iterations


# Метод Ньютона
def newton(a, b, eps):
    ok, msg, start_point = check_newton_secant_chord(a, b)
    if not ok or start_point is None:
        # Попробуем начать с середины интервала
        x0 = (a + b) / 2
    else:
        x0 = a if start_point == 'a' else b

    iterations = 0
    max_iter = 1000
    x1 = x0
    while iterations < max_iter:
        iterations += 1
        if abs(fp(x0)) < 1e-12:
            return None, iterations
        x1 = x0 - f(x0) / fp(x0)
        if abs(x1 - x0) < eps:
            if a <= x1 <= b:  # Проверяем, что корень в интервале
                return x1, iterations
            else:
                return None, iterations
        x0 = x1
    return None, iterations


# Метод хорд
def chord(a, b, eps):
    ok, msg, start_point = check_newton_secant_chord(a, b)
    if not ok or start_point is None:
        # Начинаем с середины и используем оба конца
        z = a if f(a) * fpp(a) > 0 else b
        x0 = b if z == a else a
    else:
        z = a if start_point == 'a' else b
        x0 = b if z == a else a

    iterations = 0
    max_iter = 1000
    x1 = x0
    while iterations < max_iter:
        iterations += 1
        denominator = f(z) - f(x0)
        if abs(denominator) < 1e-12:
            return None, iterations
        x1 = x0 - f(x0) * (z - x0) / denominator
        if abs(x1 - x0) < eps:
            if a <= x1 <= b:  # Проверяем, что корень в интервале
                return x1, iterations
            else:
                return None, iterations
        x0 = x1
    return None, iterations


# Основная программа
def main():
    print("=" * 70)
    print("ПОИСК КОРНЕЙ УРАВНЕНИЯ")
    print("f(x) = 3*cos(3x) - 2*(x+1)^2 + 7x + 3")
    print("=" * 70)

    # Ввод интервала
    while True:
        try:
            a = float(input("Введите начало интервала A: "))
            b = float(input("Введите конец интервала B: "))
            if a >= b:
                print("Ошибка: A должно быть меньше B")
                continue
            break
        except ValueError:
            print("Ошибка: введите числовые значения")

    print(f"\nИщем корни на интервале [{a}, {b}]")
    print("=" * 70)

    # Автоматический поиск интервалов смены знака
    intervals = find_sign_change_intervals(a, b)

    if not intervals:
        print(f"На интервале [{a}, {b}] не найдено интервалов смены знака функции.")
        print("Возможно, корней нет или функция не пересекает ось X.")

        # Проверяем значения на концах
        fa = f(a)
        fb = f(b)
        print(f"\nf({a}) = {fa:.6f}")
        print(f"f({b}) = {fb:.6f}")

        # Проверяем, есть ли корень на границах
        if abs(fa) < eps:
            print(f"\nНайден корень на левой границе: x = {a:.6f}")
            intervals = [(a, a)]
        elif abs(fb) < eps:
            print(f"\nНайден корень на правой границе: x = {b:.6f}")
            intervals = [(b, b)]
        else:
            # Показываем график
            plot_function(a, b, [], [])
            return
    else:
        print(f"Найдено {len(intervals)} интервал(ов) смены знака:")
        for i, (start, end) in enumerate(intervals, 1):
            print(f"  Интервал {i}: [{start:.4f}, {end:.4f}]")

    print("\n" + "=" * 70)
    print("УТОЧНЕНИЕ КОРНЕЙ")
    print("=" * 70)

    methods = {
        "Метод дихотомии": bisection,
        "Метод Ньютона": newton,
        "Метод хорд": chord
    }

    all_roots = []
    for i, (interval_start, interval_end) in enumerate(intervals, 1):
        print(f"\nИнтервал {i}: [{interval_start:.6f}, {interval_end:.6f}]")
        print(f"f({interval_start:.6f}) = {f(interval_start):.6e}")
        print(f"f({interval_end:.6f}) = {f(interval_end):.6e}")

        # Для каждого интервала применяем методы
        interval_roots = []
        for method_name, method_func in methods.items():
            root, iters = method_func(interval_start, interval_end, eps)
            if root is not None:
                # Проверяем, не нашли ли мы уже этот корень
                if not any(abs(root - r) < eps for r in interval_roots):
                    interval_roots.append(root)
                    print(f"  {method_name}: x = {root:.10f}, итераций = {iters}")

        # Если методы не сошлись, используем середину интервала как приближение
        if not interval_roots:
            approx_root = (interval_start + interval_end) / 2
            if abs(f(approx_root)) < 0.1:  # Если значение функции близко к нулю
                interval_roots.append(approx_root)
                print(f"  Приблизительный корень: x ≈ {approx_root:.10f}")

        all_roots.extend(interval_roots)

    if all_roots:
        # Удаляем дубликаты (корни, близкие друг к другу)
        unique_roots = []
        for root in sorted(all_roots):
            if not unique_roots or abs(root - unique_roots[-1]) > eps:
                unique_roots.append(root)

        print(f"\nНайдено {len(unique_roots)} уникальных корней:")
        for i, root in enumerate(unique_roots, 1):
            print(f"  Корень {i}: x = {root:.10f}, f(x) = {f(root):.2e}")
    else:
        print("\nНе удалось найти корни с заданной точностью.")
        print("Попробуйте увеличить точность или изменить интервал.")

    # Построение графика
    plot_function(a, b, intervals, unique_roots if 'unique_roots' in locals() else [])


def plot_function(a, b, intervals, roots):
    """Построение графика функции с корнями"""
    x = np.linspace(a, b, 1000)
    y = np.array([f(xi) for xi in x])

    plt.figure(figsize=(12, 7))

    # График функции
    plt.plot(x, y, 'b-', label='f(x) = 3cos(3x) - 2(x+1)² + 7x + 3', linewidth=2)

    # Ось X
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Интервалы смены знака
    for i, (start, end) in enumerate(intervals):
        plt.axvspan(start, end, alpha=0.2, color='yellow',
                    label='Интервалы смены знака' if i == 0 else "")

    # Найденные корни
    if roots:
        y_roots = [f(r) for r in roots]
        plt.scatter(roots, y_roots, color='red', s=100, zorder=5,
                    label='Найденные корни')

        # Подписи к корням
        for i, (root, y_root) in enumerate(zip(roots, y_roots)):
            plt.annotate(f'x{i + 1} = {root:.4f}',
                         xy=(root, y_root),
                         xytext=(10, 20 * (1 if i % 2 == 0 else -1)),
                         textcoords='offset points',
                         ha='left',
                         va='bottom' if i % 2 == 0 else 'top',
                         fontsize=10,
                         arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

    # Настройки графика
    plt.title(f'График функции f(x) на интервале [{a}, {b}]', fontsize=14, pad=20)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)

    # Автоматическое масштабирование по Y
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    if y_range > 0:
        plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    # Сохранение и отображение
    output_file = 'function_roots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранен в файл: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()