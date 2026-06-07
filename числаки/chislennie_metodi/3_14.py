import math
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """Подынтегральная функция: (2 * cos(x/4) * exp(x^2 - 2)) / sqrt(2x^2 + 3)"""
    numerator = 2 * math.cos(x / 4) * math.exp(x ** 2 - 2)
    denominator = math.sqrt(2 * x ** 2 + 3)
    return numerator / denominator


def df(x):
    """Первая производная f'(x) - аналитически вычисленная"""
    u = 2 * math.cos(x / 4) * math.exp(x ** 2 - 2)
    v = math.sqrt(2 * x ** 2 + 3)
    
    du_dx = 2 * (-math.sin(x / 4) * (1 / 4) * math.exp(x ** 2 - 2) +
                 math.cos(x / 4) * math.exp(x ** 2 - 2) * 2 * x)
    
    dv_dx = (1 / 2) * (2 * x ** 2 + 3) ** (-1 / 2) * 4 * x
    
    return (du_dx * v - u * dv_dx) / (v ** 2)


def d2f_approx(x, h=1e-5):
    """Численное вычисление второй производной"""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


def d4f_approx(x, h=1e-5):
    """Численное вычисление четвертой производной"""
    return (f(x + 2 * h) - 4 * f(x + h) + 6 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (h ** 4)


def estimate_max_derivatives(a, b, n_samples=1000):
    """Оценка максимумов производных на отрезке"""
    x_samples = np.linspace(a, b, n_samples)
    
    max_d2f = max(abs(d2f_approx(x)) for x in x_samples)
    max_d4f = max(abs(d4f_approx(x)) for x in x_samples)
    
    return max_d2f, max_d4f


def plot_integration(a, b, n, method_name, integral_value, exact_value=None):
    """Визуализация метода интегрирования"""
    h = (b - a) / n
    x_nodes = [a + i * h for i in range(n + 1)]
    y_nodes = [f(x) for x in x_nodes]
    
    x_plot = np.linspace(a, b, 1000)
    y_plot = [f(x) for x in x_plot]
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='$f(x)$')
    
    if method_name == "Прямоугольники (средние)":
        for i in range(n):
            x_left = x_nodes[i]
            x_right = x_nodes[i + 1]
            x_mid = (x_left + x_right) / 2
            y_mid = f(x_mid)
            
            rect_x = [x_left, x_right, x_right, x_left, x_left]
            rect_y = [0, 0, y_mid, y_mid, 0]
            plt.fill(rect_x, rect_y, 'lightblue', alpha=0.5)
            plt.plot([x_left, x_right], [y_mid, y_mid], 'b-', linewidth=1)
            plt.plot([x_mid, x_mid], [0, y_mid], 'b--', linewidth=0.5)
        
        plt.scatter([(x_nodes[i] + x_nodes[i + 1]) / 2 for i in range(n)],
                    [f((x_nodes[i] + x_nodes[i + 1]) / 2) for i in range(n)],
                    color='blue', s=40, zorder=5, label='Средние точки')
    
    elif method_name == "Трапеции":
        for i in range(n):
            x_left = x_nodes[i]
            x_right = x_nodes[i + 1]
            y_left = f(x_left)
            y_right = f(x_right)
            
            trap_x = [x_left, x_right, x_right, x_left, x_left]
            trap_y = [0, 0, y_right, y_left, 0]
            plt.fill(trap_x, trap_y, 'lightgreen', alpha=0.5)
            plt.plot([x_left, x_right], [y_left, y_right], 'g-', linewidth=1)
        
        plt.scatter(x_nodes, y_nodes, color='darkgreen', s=50, zorder=5, label='Узлы')
    
    elif method_name == "Симпсона":
        if n % 2 == 0:
            for i in range(0, n, 2):
                if i + 2 <= n:
                    x0, x1, x2 = x_nodes[i], x_nodes[i + 1], x_nodes[i + 2]
                    y0, y1, y2 = y_nodes[i], y_nodes[i + 1], y_nodes[i + 2]
                    
                    x_parabola = np.linspace(x0, x2, 50)
                    y_parabola = []
                    for x in x_parabola:
                        L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
                        L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
                        L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
                        y = L0 * y0 + L1 * y1 + L2 * y2
                        y_parabola.append(y)
                    
                    fill_x = list(x_parabola) + [x2, x0]
                    fill_y = list(y_parabola) + [0, 0]
                    plt.fill(fill_x, fill_y, 'orange', alpha=0.3)
                    plt.plot(x_parabola, y_parabola, 'orange', linewidth=1.5)
        
        plt.scatter(x_nodes, y_nodes, color='darkorange', s=50, zorder=5, label='Узлы')
    
    # Заголовок с информацией об ошибке
    if exact_value is not None:
        error = abs(integral_value - exact_value)
        title = f'Метод {method_name}\n$n={n},\\ F \\approx {integral_value:.8f}$\nОшибка: {error:.2e}'
    else:
        title = f'Метод {method_name}\n$n={n},\\ F \\approx {integral_value:.8f}$'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$f(x)$', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def midpoint_rule(a, b, n):
    """Метод средних прямоугольников (2-й порядок точности)"""
    h = (b - a) / n
    integral = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        integral += f(x_mid)
    return h * integral


def trapezoidal_rule(a, b, n):
    """Классический метод трапеций (2-й порядок точности)"""
    h = (b - a) / n
    x_nodes = [a + i * h for i in range(n + 1)]
    y_nodes = [f(x) for x in x_nodes]
    
    integral = y_nodes[0] + y_nodes[-1]
    for i in range(1, n):
        integral += 2 * y_nodes[i]
    integral *= h / 2
    
    return integral


def simpsons_rule(a, b, n):
    """Метод Симпсона (4-й порядок точности)"""
    if n % 2 != 0:
        n += 1  # Делаем n четным
    
    h = (b - a) / n
    x_nodes = [a + i * h for i in range(n + 1)]
    y_nodes = [f(x) for x in x_nodes]
    
    integral = y_nodes[0] + y_nodes[-1]
    
    # Нечетные узлы (с коэффициентом 4)
    for i in range(1, n, 2):
        integral += 4 * y_nodes[i]
    
    # Четные узлы (с коэффициентом 2)
    for i in range(2, n - 1, 2):
        integral += 2 * y_nodes[i]
    
    return h * integral / 3


def runge_romberg(Fh, Fkh, k, p):
    """Уточнение по Рунге-Ромбергу"""
    correction = (Fh - Fkh) / (k ** p - 1)
    return Fh + correction


def get_exact_integral():
    """
    Получение точного значения интеграла
    Используем метод Симпсона с очень маленьким шагом как эталон
    """
    a, b = -1.5, 1.5
    n_exact = 10000  # Очень большое количество разбиений
    return simpsons_rule(a, b, n_exact)


def main():
    a, b = -1.5, 1.5
    n1, n2 = 8, 16  # Начальное и удвоенное количество разбиений
    
    print("=" * 100)
    print(f"ВЫЧИСЛЕНИЕ ИНТЕГРАЛА: F = ∫[{-1.5},{1.5}] (2·cos(x/4)·e^(x²-2))/√(2x²+3) dx")
    print("=" * 100)
    
    # Получаем точное значение интеграла
    exact_integral = get_exact_integral()
    print(f"\nЭталонное значение интеграла (n=10000): {exact_integral:.12f}")
    
    # Вычисление интегралов для двух шагов
    F_mid1 = midpoint_rule(a, b, n1)
    F_mid2 = midpoint_rule(a, b, n2)
    
    F_trap1 = trapezoidal_rule(a, b, n1)
    F_trap2 = trapezoidal_rule(a, b, n2)
    
    F_simp1 = simpsons_rule(a, b, n1)
    F_simp2 = simpsons_rule(a, b, n2)
    
    # Уточнение по Рунге-Ромбергу
    k = 2  # Коэффициент уменьшения шага
    
    F_mid_rr = runge_romberg(F_mid2, F_mid1, k, 2)
    F_trap_rr = runge_romberg(F_trap2, F_trap1, k, 2)
    F_simp_rr = runge_romberg(F_simp2, F_simp1, k, 4)
    
    # Оценка максимальных производных для теоретических погрешностей
    max_d2f, max_d4f = estimate_max_derivatives(a, b)
    h1 = (b - a) / n1
    h2 = (b - a) / n2
    
    # Теоретические оценки погрешностей
    R_mid1 = (h1 ** 2 * (b - a) * max_d2f) / 24
    R_mid2 = (h2 ** 2 * (b - a) * max_d2f) / 24
    
    R_trap1 = (h1 ** 2 * (b - a) * max_d2f) / 12
    R_trap2 = (h2 ** 2 * (b - a) * max_d2f) / 12
    
    R_simp1 = (h1 ** 4 * (b - a) * max_d4f) / 180
    R_simp2 = (h2 ** 4 * (b - a) * max_d4f) / 180
    
    # Фактические ошибки
    err_mid1 = abs(F_mid1 - exact_integral)
    err_mid2 = abs(F_mid2 - exact_integral)
    
    err_trap1 = abs(F_trap1 - exact_integral)
    err_trap2 = abs(F_trap2 - exact_integral)
    
    err_simp1 = abs(F_simp1 - exact_integral)
    err_simp2 = abs(F_simp2 - exact_integral)
    
    # Вывод результатов
    print("\n" + "=" * 120)
    print("РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ:")
    print("=" * 120)
    
    print(f"\n{'Метод':<25} {'n':<4} {'Значение':<15} {'Теор. погр.':<12} {'Факт. погр.':<12} "
          f"{'Уточненное':<15} {'Порядок':<8}")
    print("-" * 120)
    
    # Метод прямоугольников
    print(f"{'Прямоугольники':<25} {n1:<4} {F_mid1:<15.8f} {R_mid1:<12.2e} {err_mid1:<12.2e} "
          f"{'-':<15} {'O(h²)':<8}")
    print(f"{'':<25} {n2:<4} {F_mid2:<15.8f} {R_mid2:<12.2e} {err_mid2:<12.2e} "
          f"{F_mid_rr:<15.8f} {'O(h²)':<8}")
    
    # Метод трапеций
    print(f"\n{'Трапеции':<25} {n1:<4} {F_trap1:<15.8f} {R_trap1:<12.2e} {err_trap1:<12.2e} "
          f"{'-':<15} {'O(h²)':<8}")
    print(f"{'':<25} {n2:<4} {F_trap2:<15.8f} {R_trap2:<12.2e} {err_trap2:<12.2e} "
          f"{F_trap_rr:<15.8f} {'O(h²)':<8}")
    
    # Метод Симпсона
    print(f"\n{'Симпсона':<25} {n1:<4} {F_simp1:<15.8f} {R_simp1:<12.2e} {err_simp1:<12.2e} "
          f"{'-':<15} {'O(h⁴)':<8}")
    print(f"{'':<25} {n2:<4} {F_simp2:<15.8f} {R_simp2:<12.2e} {err_simp2:<12.2e} "
          f"{F_simp_rr:<15.8f} {'O(h⁴)':<8}")
    
    print("-" * 120)
    
    # Сравнение порядков точности
    print("\n" + "=" * 100)
    print("АНАЛИЗ ПОРЯДКОВ ТОЧНОСТИ:")
    print("=" * 100)
    
    print(f"\nУменьшение фактической ошибки при удвоении числа разбиений (n: {n1} → {n2}):")
    print("-" * 70)
    
    ratio_mid = err_mid1 / err_mid2 if err_mid2 != 0 else 0
    ratio_trap = err_trap1 / err_trap2 if err_trap2 != 0 else 0
    ratio_simp = err_simp1 / err_simp2 if err_simp2 != 0 else 0
    
    print(f"{'Прямоугольники:':<25} {err_mid1:.2e} → {err_mid2:.2e}  "
          f"Уменьшение в {ratio_mid:.2f} раз (теоретически в 4 раза)")
    print(f"{'Трапеции:':<25} {err_trap1:.2e} → {err_trap2:.2e}  "
          f"Уменьшение в {ratio_trap:.2f} раз (теоретически в 4 раза)")
    print(f"{'Симпсона:':<25} {err_simp1:.2e} → {err_simp2:.2e}  "
          f"Уменьшение в {ratio_simp:.2f} раз (теоретически в 16 раз)")
    
    print("\n" + "=" * 100)
    print("ВЫВОДЫ:")
    print("=" * 100)
    
    print(f"\n1. Метод прямоугольников (средних) имеет 2-й порядок точности O(h²)")
    print(f"   - При удвоении n ошибка уменьшается примерно в {ratio_mid:.1f} раз")
    
    print(f"\n2. Метод трапеций имеет 2-й порядок точности O(h²)")
    print(f"   - При удвоении n ошибка уменьшается примерно в {ratio_trap:.1f} раз")
    
    print(f"\n3. Метод Симпсона имеет 4-й порядок точности O(h⁴)")
    print(f"   - При удвоении n ошибка уменьшается примерно в {ratio_simp:.1f} раз")
    
    print(f"\n4. Самый точный метод - Симпсона (ошибка: {err_simp2:.2e})")
    print(f"   Прямоугольники: {err_mid2:.2e}, Трапеции: {err_trap2:.2e}")
    
    print("\n" + "=" * 100)
    
    # Визуализация
    print("\nПостроение графиков...")
    plot_integration(a, b, n2, "Прямоугольники (средние)", F_mid2, exact_integral)
    plot_integration(a, b, n2, "Трапеции", F_trap2, exact_integral)
    plot_integration(a, b, n2, "Симпсона", F_simp2, exact_integral)
    
    # Дополнительный график сравнения ошибок
    plt.figure(figsize=(10, 6))
    n_values = [n1, n2]
    errors_mid = [err_mid1, err_mid2]
    errors_trap = [err_trap1, err_trap2]
    errors_simp = [err_simp1, err_simp2]
    
    plt.loglog(n_values, errors_mid, 'bo-', linewidth=2, markersize=8, label='Прямоугольники (O(h²))')
    plt.loglog(n_values, errors_trap, 'gs-', linewidth=2, markersize=8, label='Трапеции (O(h²))')
    plt.loglog(n_values, errors_simp, 'r^-', linewidth=2, markersize=8, label='Симпсона (O(h⁴))')
    
    # Теоретические линии для сравнения
    n_theor = np.array([8, 16, 32, 64])
    plt.loglog(n_theor, err_mid1 * (n_theor/n1)**(-2), 'b--', alpha=0.5, label='Теория O(h²)')
    plt.loglog(n_theor, err_simp1 * (n_theor/n1)**(-4), 'r--', alpha=0.5, label='Теория O(h⁴)')
    
    plt.xlabel('Число разбиений n', fontsize=12)
    plt.ylabel('Абсолютная ошибка', fontsize=12)
    plt.title('Сравнение точности методов численного интегрирования\n(логарифмический масштаб)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('errors_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nГрафик сравнения ошибок сохранен как 'errors_comparison.png'")


if __name__ == "__main__":
    main()