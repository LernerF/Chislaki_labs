"""
МЕТОД СТРЕЛЬБЫ И КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД ДЛЯ РЕШЕНИЯ КРАЕВОЙ ЗАДАЧИ
Уравнение: xy'' + 2y' - xy = e^x
Граничные условия: y(0.5)=5.33, y(4.5)=65.01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import sparse
from scipy.sparse.linalg import spsolve
import warnings

warnings.filterwarnings("ignore")


def exact_solution(x):
    """
    АНАЛИТИЧЕСКОЕ РЕШЕНИЕ (эталон для сравнения)
    
    Формула получена аналитически из исходного уравнения:
    y(x) = e^x/2 + (e^x + e^(-x))/x
    
    Параметры:
        x - точка или массив точек
    
    Возвращает:
        значение точного решения
    """
    x = np.asarray(x)
    term1 = np.exp(x) / 2
    term2 = (np.exp(x) + np.exp(-x)) / x
    return term1 + term2


def ode_system(x, Y):
    """
    ПРЕОБРАЗОВАНИЕ ОДУ 2-ГО ПОРЯДКА В СИСТЕМУ 1-ГО ПОРЯДКА
    
    Исходное уравнение: xy'' + 2y' - xy = e^x
    Выражаем y'': y'' = (e^x - 2y' + xy)/x
    
    Вводим замену:
        y1 = y  (сама функция)
        y2 = y' = y1'  (первая производная)
    
    Получаем систему:
        y1' = y2
        y2' = (e^x - 2*y2 + x*y1)/x
    
    Параметры:
        x - аргумент (точка)
        Y - вектор [y, y'] в текущей точке
    
    Возвращает:
        вектор производных [y', y'']
    """
    y, z = Y
    
    # Защита от деления на ноль
    if abs(x) < 1e-12:
        x = 1e-12
    
    y_prime_prime = np.exp(x) / x - 2 * z / x + y
    return np.array([z, y_prime_prime])


def rk4_step(x, Y, h, ode_func):
    """
    ОДИН ШАГ МЕТОДА РУНГЕ-КУТТА 4-ГО ПОРЯДКА (RK4)
    
    Формулы:
        k1 = f(x, Y)
        k2 = f(x + h/2, Y + h*k1/2)
        k3 = f(x + h/2, Y + h*k2/2)
        k4 = f(x + h, Y + h*k3)
        Y_new = Y + h*(k1 + 2*k2 + 2*k3 + k4)/6
    """
    k1 = ode_func(x, Y)
    k2 = ode_func(x + h / 2, Y + h * k1 / 2)
    k3 = ode_func(x + h / 2, Y + h * k2 / 2)
    k4 = ode_func(x + h, Y + h * k3)
    
    return Y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solve_ivp(a, b, ya, slope, h, ode_func):
    """
    РЕШЕНИЕ ЗАДАЧИ КОШИ МЕТОДОМ РУНГЕ-КУТТА 4-ГО ПОРЯДКА
    
    Решаем от точки a до точки b с начальными условиями:
        y(a) = ya
        y'(a) = slope
    """
    n = int(np.round((b - a) / h)) + 1
    h_actual = (b - a) / (n - 1)
    
    x = np.zeros(n)
    y = np.zeros(n)
    
    x[0] = a
    y[0] = ya
    z = slope
    
    for i in range(n - 1):
        Y = np.array([y[i], z])
        Y_new = rk4_step(x[i], Y, h_actual, ode_func)
        y[i + 1], z = Y_new
        x[i + 1] = x[i] + h_actual
    
    return x, y, y[-1]


def find_bracket_adaptive(a, b, ya, yb, h, initial_width=20.0, max_attempts=5, verbose=True):
    """
    АДАПТИВНЫЙ ПОИСК ИНТЕРВАЛА ДЛЯ НАЧАЛЬНОГО НАКЛОНА
    
    Ищем такие два наклона alpha1 и alpha2, чтобы F(alpha1) и F(alpha2) имели
    разные знаки (F(alpha) = y(b, alpha) - yb - невязка).
    """
    slope0 = (yb - ya) / (b - a)
    
    print("1. Поиск интервала (вилки) для начального наклона")
    print(f" Начальное приближение наклона: {slope0:.4f}")
    print("-" * 60)
    
    for attempt in range(1, max_attempts + 1):
        width = initial_width * attempt
        
        if attempt > 1:
            print(f"\nПопытка {attempt}: Расширяем диапазон поиска (width = {width})")
            print("-" * 60)
        
        offsets = np.linspace(-width, width, 11)
        try_vals = slope0 + offsets
        
        results = []
        signs = []
        
        for idx, alpha in enumerate(try_vals, 1):
            try:
                _, _, yb_calc = solve_ivp(a, b, ya, alpha, h, ode_system)
                nev = yb_calc - yb
                sign = np.sign(nev)
                
                results.append((alpha, nev))
                signs.append(sign)
                
                if verbose:
                    print(f"Попытка {idx:2d}: Наклон = {alpha:10.4f}, Невязка = {nev:12.6f}")
            except:
                if verbose:
                    print(f"Попытка {idx:2d}: Наклон = {alpha:10.4f}, Ошибка при интегрировании")
                continue
        
        for i in range(1, len(signs)):
            if signs[i - 1] * signs[i] < 0:
                print(f"\n+ Вилка найдена! Интервал наклонов: [{results[i - 1][0]:.4f}, {results[i][0]:.4f}]")
                print(f" Невязки: [{results[i - 1][1]:.6f}, {results[i][1]:.6f}]\n")
                return results[i - 1][0], results[i][0]
    
    print("\n- Вилка не найдена после всех попыток.")
    print("Попробуем использовать метод Ньютона для поиска корня...")
    return find_bracket_newton(a, b, ya, yb, h, slope0, verbose)


def find_bracket_newton(a, b, ya, yb, h, slope0, verbose=True):
    """
    ПОИСК КОРНЯ МЕТОДОМ НЬЮТОНА (через scipy.optimize)
    """
    def residual(alpha):
        _, _, yb_calc = solve_ivp(a, b, ya, alpha, h, ode_system)
        return yb_calc - yb
    
    try:
        initial_guesses = [slope0, 0, -slope0, 2 * slope0, -2 * slope0]
        
        for guess in initial_guesses:
            try:
                alpha_opt = fsolve(residual, guess, full_output=False)[0]
                _, _, yb_calc = solve_ivp(a, b, ya, alpha_opt, h, ode_system)
                nev = yb_calc - yb
                
                if abs(nev) < 1e-4:
                    print(f"\n+ Найдено решение методом Ньютона!")
                    print(f" Оптимальный наклон: {alpha_opt:.6f}")
                    print(f" Невязка: {nev:.8f}\n")
                    
                    delta = 0.1
                    return alpha_opt - delta, alpha_opt + delta
            except:
                continue
        
        print("\n- Не удалось найти решение методом Ньютона.")
        return None, None
    
    except Exception as e:
        print(f"\n- Ошибка при использовании метода Ньютона: {e}")
        return None, None


def binary_search_method(a, b, ya, yb, h, bracket, tol=1e-6, maxIter=30, verbose=True):
    """
    МЕТОД ДИХОТОМИИ (БИНАРНОГО ПОИСКА) ДЛЯ УТОЧНЕНИЯ НАКЛОНА
    """
    alpha_left, alpha_right = bracket
    history = []
    
    print("2. Уточнение решения методом дихотомии...")
    print("-" * 60)
    
    for i in range(1, maxIter + 1):
        alpha_mid = (alpha_left + alpha_right) / 2
        xArr, yArr, yb_mid = solve_ivp(a, b, ya, alpha_mid, h, ode_system)
        nev = yb_mid - yb
        
        if verbose:
            print(f"Итерация {i:2d}: Наклон = {alpha_mid:10.6f}, Невязка = {nev:12.8f}")
        
        history.append((xArr.copy(), yArr.copy(), alpha_mid))
        
        if abs(nev) < tol:
            print(f"\n+ Достигнута требуемая точность на итерации {i}")
            break
        
        if abs(alpha_right - alpha_left) < 1e-10:
            print(f"\n+ Интервал стал достаточно малым на итерации {i}")
            break
        
        _, _, yb_left = solve_ivp(a, b, ya, alpha_left, h, ode_system)
        nev_left = yb_left - yb
        
        if nev_left * nev < 0:
            alpha_right = alpha_mid
        else:
            alpha_left = alpha_mid
    else:
        print(f"\n+ Достигнуто максимальное число итераций ({maxIter})")
    
    print(f"\nОптимальный начальный наклон: {alpha_mid:.8f}\n")
    return xArr, yArr, history, alpha_mid


# ============================================================================
# КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД
# ============================================================================

def finite_difference_method(a, b, ya, yb, n, verbose=True):
    """
    КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД РЕШЕНИЯ КРАЕВОЙ ЗАДАЧИ
    
    Уравнение: xy'' + 2y' - xy = e^x
    Граничные условия: y(a)=ya, y(b)=yb
    
    Используем центральные разности:
        y'(x_i) ≈ (y_{i+1} - y_{i-1})/(2h)
        y''(x_i) ≈ (y_{i+1} - 2y_i + y_{i-1})/h^2
    
    Параметры:
        a, b - границы отрезка
        ya, yb - граничные условия
        n - количество узлов сетки (включая границы)
        verbose - печатать ли детали
    
    Возвращает:
        x - массив узлов
        y - численное решение
        A - матрица СЛАУ (для анализа)
        b_vec - вектор правой части
    """
    # Шаг сетки
    h = (b - a) / (n - 1)
    
    # Создаем сетку
    x = np.linspace(a, b, n)
    
    # Количество внутренних узлов
    n_inner = n - 2
    
    print("=" * 80)
    print("КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД")
    print("=" * 80)
    print(f"Количество узлов: {n}")
    print(f"Шаг сетки: h = {h:.6f}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"Граничные условия: y({a})={ya}, y({b})={yb}")
    print("-" * 80)
    
    # Инициализируем матрицу СЛАУ (разреженную для эффективности)
    # Для уравнения вида: A_i * y_{i-1} + B_i * y_i + C_i * y_{i+1} = F_i
    A_coeff = np.zeros(n_inner)  # Коэффициенты при y_{i-1}
    B_coeff = np.zeros(n_inner)  # Коэффициенты при y_i
    C_coeff = np.zeros(n_inner)  # Коэффициенты при y_{i+1}
    F = np.zeros(n_inner)        # Правая часть
    
    # Заполняем коэффициенты для внутренних узлов
    for i in range(1, n - 1):
        xi = x[i]
        
        # Коэффициенты разностной схемы:
        # (xi/h^2) * (y_{i+1} - 2y_i + y_{i-1}) + (1/h) * (y_{i+1} - y_{i-1}) - xi*y_i = e^{xi}
        
        # Приводим к виду: A*y_{i-1} + B*y_i + C*y_{i+1} = F
        
        # Коэффициент при y_{i-1}
        A = xi / h**2 - 1 / h
        
        # Коэффициент при y_i
        B = -2 * xi / h**2 - xi
        
        # Коэффициент при y_{i+1}
        C = xi / h**2 + 1 / h
        
        # Правая часть
        F_val = np.exp(xi)
        
        # Сохраняем коэффициенты для внутреннего узла
        idx = i - 1  # Индекс в массиве внутренних узлов
        A_coeff[idx] = A
        B_coeff[idx] = B
        C_coeff[idx] = C
        F[idx] = F_val
        
        if verbose and i <= 3:
            print(f"Узел {i}: x[{i}]={xi:.4f}")
            print(f"  {A:.6f} * y_{i-1} + {B:.6f} * y_{i} + {C:.6f} * y_{i+1} = {F_val:.6f}")
    
    if verbose and n > 5:
        print("  ...")
        print(f"Узел {n-2}: x[{n-2}]={x[n-2]:.4f}")
        print(f"  {A_coeff[-1]:.6f} * y_{n-3} + {B_coeff[-1]:.6f} * y_{n-2} + {C_coeff[-1]:.6f} * y_{n-1} = {F[-1]:.6f}")
    
    # Учитываем граничные условия
    # Для i=1 (первый внутренний узел): y0 = ya
    # Для i=n-2 (последний внутренний узел): y_{n-1} = yb
    
    # Корректируем первое уравнение (i=1)
    # Исходное: A1*y0 + B1*y1 + C1*y2 = F1
    # Переносим известное y0 в правую часть: B1*y1 + C1*y2 = F1 - A1*ya
    F[0] = F[0] - A_coeff[0] * ya
    A_coeff[0] = 0  # Обнуляем коэффициент при y0 (он перенесен)
    
    # Корректируем последнее уравнение (i=n-2)
    # Исходное: A_{n-2}*y_{n-3} + B_{n-2}*y_{n-2} + C_{n-2}*y_{n-1} = F_{n-2}
    # Переносим известное y_{n-1} в правую часть: A_{n-2}*y_{n-3} + B_{n-2}*y_{n-2} = F_{n-2} - C_{n-2}*yb
    F[-1] = F[-1] - C_coeff[-1] * yb
    C_coeff[-1] = 0  # Обнуляем коэффициент при y_{n-1} (он перенесен)
    
    # Строим трехдиагональную матрицу
    # Для системы: B_i * y_i + C_i * y_{i+1} = F_i - A_i * y_{i-1}
    # Но так как A_i уже учтены, матрица имеет вид:
    # Для i=0: B0*y0 + C0*y1 = F0
    # Для i=1: A1*y0 + B1*y1 + C1*y2 = F1
    # ...
    # Для i=n-3: A_{n-3}*y_{n-4} + B_{n-3}*y_{n-3} + C_{n-3}*y_{n-2} = F_{n-3}
    # Для i=n-2: A_{n-2}*y_{n-3} + B_{n-2}*y_{n-2} = F_{n-2}
    
    # Создаем разреженную трехдиагональную матрицу
    main_diag = B_coeff.copy()
    upper_diag = C_coeff[:-1].copy()  # Наддиагональ (без последнего элемента)
    lower_diag = A_coeff[1:].copy()    # Поддиагональ (без первого элемента)
    
    # Строим матрицу в формате CSR для эффективного решения
    diagonals = [main_diag, upper_diag, lower_diag]
    offsets = [0, 1, -1]
    A_matrix = sparse.diags(diagonals, offsets, format='csr')
    
    # Решаем СЛАУ
    y_inner = spsolve(A_matrix, F)
    
    # Формируем полное решение (включая граничные узлы)
    y = np.zeros(n)
    y[0] = ya
    y[1:-1] = y_inner
    y[-1] = yb
    
    if verbose:
        print("-" * 80)
        print("Матрица СЛАУ (трехдиагональная):")
        print(f"  Размер: {n_inner} x {n_inner}")
        print(f"  Ненулевых элементов: {3*n_inner - 2}")
        print("-" * 80)
    
    return x, y, A_matrix, F


def runge_romberg_error(y_h, y_h2, p=4):
    """
    ОЦЕНКА ПОГРЕШНОСТИ МЕТОДОМ РУНГЕ-РОМБЕРГА
    
    Формула: R = (y_{h/2} - y_h) / (2^p - 1)
    
    Параметры:
        y_h - решение с шагом h
        y_h2 - решение с шагом h/2
        p - порядок точности метода (для RK4 и конечно-разностного p=2)
    
    Возвращает:
        оценку погрешности
    """
    # Интерполируем y_h на сетку y_h2 для сравнения в одинаковых узлах
    if len(y_h) != len(y_h2):
        # Создаем интерполяцию
        from scipy.interpolate import interp1d
        x_h = np.linspace(a, b, len(y_h))
        x_h2 = np.linspace(a, b, len(y_h2))
        y_h_interp = interp1d(x_h, y_h, kind='linear')(x_h2)
        return np.abs(y_h2 - y_h_interp) / (2**p - 1)
    else:
        return np.abs(y_h2 - y_h) / (2**p - 1)


def finite_difference_convergence(a, b, ya, yb, n_values, verbose=True):
    """
    ИССЛЕДОВАНИЕ СХОДИМОСТИ КОНЕЧНО-РАЗНОСТНОГО МЕТОДА
    
    Решает задачу с разными шагами и оценивает погрешность.
    
    Параметры:
        a, b - границы
        ya, yb - граничные условия
        n_values - список количества узлов
        verbose - печатать ли детали
    
    Возвращает:
        результаты для каждого n
    """
    results = []
    
    print("\n" + "=" * 80)
    print("ИССЛЕДОВАНИЕ СХОДИМОСТИ КОНЕЧНО-РАЗНОСТНОГО МЕТОДА")
    print("=" * 80)
    print(f"{'N':>6} | {'h':>10} | {'max error':>12} | {'mean error':>12} | {'order':>8}")
    print("-" * 80)
    
    prev_error = None
    
    for n in n_values:
        x, y, _, _ = finite_difference_method(a, b, ya, yb, n, verbose=False)
        y_exact = exact_solution(x)
        errors = np.abs(y - y_exact)
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        h = (b - a) / (n - 1)
        
        # Оценка порядка сходимости
        order = None
        if prev_error is not None:
            order = np.log(prev_error / max_error) / np.log(2)
        
        results.append({
            'n': n,
            'h': h,
            'max_error': max_error,
            'mean_error': mean_error,
            'order': order,
            'x': x,
            'y': y,
            'errors': errors
        })
        
        order_str = f"{order:.2f}" if order is not None else "---"
        print(f"{n:6d} | {h:10.6f} | {max_error:12.8f} | {mean_error:12.8f} | {order_str:>8}")
        
        prev_error = max_error
    
    print("=" * 80)
    
    return results


# ============================================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ============================================================================

def plot_solution(history, a, b, ya, yb):
    """
    График решения с показом итераций метода стрельбы
    """
    plt.figure(figsize=(12, 8))
    plt.grid(True, alpha=0.3)
    plt.title(f"Метод стрельбы: y({a})={ya}, y({b})={yb}", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y(x)", fontsize=12)
    
    # Точное решение
    xExact = np.linspace(a, b, 200)
    yExact = exact_solution(xExact)
    plt.plot(xExact, yExact, 'k-', lw=2, label="Аналитическое решение")
    
    # Численные решения
    if len(history) > 10:
        indices = np.linspace(0, len(history) - 1, min(10, len(history)), dtype=int)
        selected_history = [history[i] for i in indices]
    else:
        selected_history = history
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_history)))
    
    for idx, ((xArr, yArr, slopeVal), color) in enumerate(zip(selected_history, colors)):
        if idx < len(selected_history) - 1:
            label = f'Итерация {idx + 1}'
            style = '--'
            alpha = 0.5
            marker = 'o'
            markersize = 3
        else:
            label = f'Финальное решение (α={slopeVal:.5f})'
            style = '-'
            alpha = 1.0
            marker = 's'
            markersize = 5
        
        plt.plot(xArr, yArr, style, marker=marker, markersize=markersize,
                 label=label, alpha=alpha, color=color)
    
    plt.plot([a, b], [ya, yb], 'ro', markersize=10, label="Граничные условия", zorder=5)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_errors(x, yNum, yExact, errors, title="Численное решение"):
    """
    Графики сравнения решений и погрешностей
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Сравнение численного и аналитического решений ({title})", fontsize=14)
    ax1.plot(x, yExact, 'k-', lw=2, label="Аналитическое решение")
    ax1.plot(x, yNum, 'ro--', markersize=6, label="Численное решение")
    ax1.set_ylabel("y(x)", fontsize=12)
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Абсолютная погрешность численного решения", fontsize=14)
    ax2.bar(x, errors, width=0.01, alpha=0.7, color='blue')
    ax2.plot(x, errors, 'bo-', markersize=6)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("|y_числ - y_точн|", fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_convergence(history, a, b, ya, yb):
    """
    График сходимости метода стрельбы
    """
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.title("Сходимость метода стрельбы", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y(x)", fontsize=12)
    
    xExact = np.linspace(a, b, 200)
    yExact = exact_solution(xExact)
    plt.plot(xExact, yExact, 'k-', lw=3, label="Аналитическое решение")
    
    num_iterations = len(history)
    if num_iterations > 4:
        key_iterations = [0, num_iterations // 4, num_iterations // 2, num_iterations - 1]
    else:
        key_iterations = list(range(num_iterations))
    
    colors = ['red', 'orange', 'green', 'blue']
    
    for idx, color in zip(key_iterations, colors[:len(key_iterations)]):
        xArr, yArr, slopeVal = history[idx]
        if idx == num_iterations - 1:
            label = f'Финальное (α={slopeVal:.5f})'
            lw = 2.5
        else:
            label = f'Итерация {idx + 1}'
            lw = 1.5
        
        plt.plot(xArr, yArr, color=color, linestyle='--', lw=lw, label=label)
    
    plt.plot([a, b], [ya, yb], 'ko', markersize=10, label="Граничные условия", zorder=5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_comparison_shooting_fd(x_shooting, y_shooting, x_fd, y_fd, a, b):
    """
    Сравнение методов стрельбы и конечно-разностного
    """
    plt.figure(figsize=(12, 8))
    plt.grid(True, alpha=0.3)
    plt.title("Сравнение методов стрельбы и конечно-разностного", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y(x)", fontsize=12)
    
    # Точное решение
    xExact = np.linspace(a, b, 200)
    yExact = exact_solution(xExact)
    plt.plot(xExact, yExact, 'k-', lw=2, label="Аналитическое решение")
    
    # Численные решения
    plt.plot(x_shooting, y_shooting, 'bo-', markersize=4, label="Метод стрельбы (RK4)", alpha=0.7)
    plt.plot(x_fd, y_fd, 'rs--', markersize=4, label="Конечно-разностный метод", alpha=0.7)
    
    # Граничные условия
    plt.plot([a, b], [y_shooting[0], y_shooting[-1]], 'ko', markersize=8, label="Граничные условия")
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_fd_convergence(convergence_results):
    """
    График сходимости конечно-разностного метода
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Погрешность в зависимости от шага
    h_values = [r['h'] for r in convergence_results]
    max_errors = [r['max_error'] for r in convergence_results]
    
    ax1.loglog(h_values, max_errors, 'bo-', linewidth=2, markersize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Шаг сетки h', fontsize=12)
    ax1.set_ylabel('Максимальная погрешность', fontsize=12)
    ax1.set_title('Зависимость погрешности от шага', fontsize=14)
    
    # Добавляем линию теоретического порядка (h^2)
    h_theory = np.array([h_values[0], h_values[-1]])
    error_theory = max_errors[0] * (h_theory / h_values[0])**2
    ax1.loglog(h_theory, error_theory, 'r--', linewidth=2, label='Теоретический порядок O(h²)')
    ax1.legend()
    
    # График 2: Порядок сходимости
    n_values = [r['n'] for r in convergence_results[1:]]
    orders = [r['order'] for r in convergence_results[1:]]
    
    ax2.plot(n_values, orders, 'gs-', linewidth=2, markersize=8)
    ax2.axhline(y=2, color='r', linestyle='--', label='Теоретический порядок (2)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Количество узлов N', fontsize=12)
    ax2.set_ylabel('Порядок сходимости', fontsize=12)
    ax2.set_title('Порядок сходимости метода', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================================

def main():
    """
    ГЛАВНАЯ ФУНКЦИЯ: РЕШЕНИЕ КРАЕВОЙ ЗАДАЧИ ДВУМЯ МЕТОДАМИ
    """
    # ===== ПАРАМЕТРЫ ЗАДАЧИ =====
    a = 0.5        # Левая граница
    b = 4.5        # Правая граница
    ya = 5.33      # Значение на левой границе y(a)
    yb = 65.01     # Значение на правой границе y(b)
    h = 0.4        # Шаг интегрирования для метода стрельбы
    
    print("=" * 80)
    print("РЕШЕНИЕ КРАЕВОЙ ЗАДАЧИ")
    print("=" * 80)
    print(f"Уравнение: xy'' + 2y' - xy = e^x")
    print(f"Граничные условия: y({a}) = {ya}, y({b}) = {yb}")
    print("=" * 80)
    
    # ========================================================================
    # ЧАСТЬ 1: МЕТОД СТРЕЛЬБЫ
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 1: МЕТОД СТРЕЛЬБЫ (SHOOTING METHOD)")
    print("=" * 80)
    
    # Поиск интервала с адаптивным расширением диапазона
    bracket = find_bracket_adaptive(a, b, ya, yb, h, initial_width=30.0, max_attempts=3, verbose=True)
    
    if bracket[0] is not None:
        # Уточнение решения
        x_shooting, y_shooting, history, alpha_opt = binary_search_method(
            a, b, ya, yb, h, bracket, tol=1e-6, maxIter=30, verbose=True
        )
        
        # Вычисление погрешностей
        errors_shooting, maxError_shooting, meanError_shooting, yExact_shooting = compute_errors(x_shooting, y_shooting)
        
        # Вывод результатов
        print("=" * 80)
        print("РЕЗУЛЬТАТЫ МЕТОДА СТРЕЛЬБЫ")
        print("=" * 80)
        print(f"Количество точек: {len(x_shooting)}")
        print(f"Первая точка: x={x_shooting[0]:.4f}, последняя: x={x_shooting[-1]:.4f}")
        print("-" * 80)
        print("{:<8} | {:<12} | {:<12} | {:<12}".format(
            "x", "y_числ", "y_точн", "погрешность"))
        print("-" * 80)
        
        for xi, yi, ye, err in zip(x_shooting, y_shooting, yExact_shooting, errors_shooting):
            print("{:<8.4f} | {:<14.8f} | {:<14.8f} | {:<20.15f}".format(xi, yi, ye, err))
        
        print("=" * 80)
        print(f"\nСТАТИСТИКА ПОГРЕШНОСТЕЙ (МЕТОД СТРЕЛЬБЫ):")
        print(f"Максимальная погрешность: {maxError_shooting:.15f}")
        print(f"Средняя погрешность: {meanError_shooting:.15f}")
        print(f"Оптимальный начальный наклон: {alpha_opt:.8f}")
        
        # Построение графиков для метода стрельбы
        plot_solution(history, a, b, ya, yb)
        plot_errors(x_shooting, y_shooting, yExact_shooting, errors_shooting, "Метод стрельбы")
        plot_convergence(history, a, b, ya, yb)
    
    # ========================================================================
    # ЧАСТЬ 2: КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД (FINITE DIFFERENCE METHOD)")
    print("=" * 80)
    
    # Выбираем количество узлов, соответствующее шагу h из метода стрельбы
    n_fd = int((b - a) / h) + 1
    x_fd, y_fd, A_matrix, F_vec = finite_difference_method(a, b, ya, yb, n_fd, verbose=True)
    
    # Вычисление погрешностей
    errors_fd, maxError_fd, meanError_fd, yExact_fd = compute_errors(x_fd, y_fd)
    
    # Вывод результатов
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ КОНЕЧНО-РАЗНОСТНОГО МЕТОДА")
    print("=" * 80)
    print(f"Количество узлов: {len(x_fd)}")
    print(f"Шаг сетки: h = {(b-a)/(len(x_fd)-1):.6f}")
    print("-" * 80)
    print("{:<8} | {:<12} | {:<12} | {:<12}".format(
        "x", "y_числ", "y_точн", "погрешность"))
    print("-" * 80)
    
    for xi, yi, ye, err in zip(x_fd, y_fd, yExact_fd, errors_fd):
        print("{:<8.4f} | {:<14.8f} | {:<14.8f} | {:<20.15f}".format(xi, yi, ye, err))
    
    print("=" * 80)
    print(f"\nСТАТИСТИКА ПОГРЕШНОСТЕЙ (КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД):")
    print(f"Максимальная погрешность: {maxError_fd:.15f}")
    print(f"Средняя погрешность: {meanError_fd:.15f}")
    
    # ========================================================================
    # ЧАСТЬ 3: СРАВНЕНИЕ МЕТОДОВ
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 3: СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 80)
    print(f"{'Метод':<25} | {'Макс. погрешность':<20} | {'Ср. погрешность':<20}")
    print("-" * 80)
    print(f"{'Метод стрельбы (RK4)':<25} | {maxError_shooting:<20.15f} | {meanError_shooting:<20.15f}")
    print(f"{'Конечно-разностный':<25} | {maxError_fd:<20.15f} | {meanError_fd:<20.15f}")
    print("=" * 80)
    
    # Сравнительный график
    plot_comparison_shooting_fd(x_shooting, y_shooting, x_fd, y_fd, a, b)
    
    # ========================================================================
    # ЧАСТЬ 4: ИССЛЕДОВАНИЕ СХОДИМОСТИ КОНЕЧНО-РАЗНОСТНОГО МЕТОДА
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 4: ИССЛЕДОВАНИЕ СХОДИМОСТИ КОНЕЧНО-РАЗНОСТНОГО МЕТОДА")
    print("=" * 80)
    
    # Исследуем сходимость для разных сеток
    n_values = [11, 21, 41, 81, 161]  # Разные количества узлов
    convergence_results = finite_difference_convergence(a, b, ya, yb, n_values, verbose=True)
    
    # Построение графика сходимости
    plot_fd_convergence(convergence_results)
    
    # ========================================================================
    # ЧАСТЬ 5: ОЦЕНКА ПОГРЕШНОСТИ МЕТОДОМ РУНГЕ-РОМБЕРГА
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 5: ОЦЕНКА ПОГРЕШНОСТИ МЕТОДОМ РУНГЕ-РОМБЕРГА")
    print("=" * 80)
    
    # Для метода стрельбы (RK4, p=4)
    print("\nМЕТОД СТРЕЛЬБЫ (RK4, порядок p=4):")
    print("-" * 60)
    
    # Решаем с разными шагами
    h1 = 0.4
    h2 = 0.2
    
    # Получаем решения для разных шагов
    bracket_h1 = find_bracket_adaptive(a, b, ya, yb, h1, initial_width=30.0, max_attempts=3, verbose=False)
    if bracket_h1[0] is not None:
        _, y_h1, _, _ = binary_search_method(a, b, ya, yb, h1, bracket_h1, tol=1e-6, maxIter=30, verbose=False)
        
        bracket_h2 = find_bracket_adaptive(a, b, ya, yb, h2, initial_width=30.0, max_attempts=3, verbose=False)
        if bracket_h2[0] is not None:
            x_h2, y_h2, _, _ = binary_search_method(a, b, ya, yb, h2, bracket_h2, tol=1e-6, maxIter=30, verbose=False)
            
            # Интерполируем y_h1 на сетку y_h2
            from scipy.interpolate import interp1d
            x_h1 = np.linspace(a, b, int((b - a)/h1) + 1)
            y_h1_interp = interp1d(x_h1, y_h1, kind='linear')(x_h2)
            
            # Оценка погрешности по Рунге-Ромбергу
            runge_error = np.abs(y_h2 - y_h1_interp) / (2**4 - 1)
            max_runge_error = np.max(runge_error)
            mean_runge_error = np.mean(runge_error)
            
            print(f"Шаг h1 = {h1}, шаг h2 = {h2}")
            print(f"Максимальная оценка погрешности по Рунге-Ромбергу: {max_runge_error:.15f}")
            print(f"Средняя оценка погрешности: {mean_runge_error:.15f}")
            print(f"Фактическая максимальная погрешность (сравнение с точным): {maxError_shooting:.15f}")
            print(f"Отношение оценки к фактической: {max_runge_error / maxError_shooting:.4f}")
    
    # Для конечно-разностного метода (p=2)
    print("\nКОНЕЧНО-РАЗНОСТНЫЙ МЕТОД (порядок p=2):")
    print("-" * 60)
    
    # Берем результаты из исследования сходимости
    if len(convergence_results) >= 2:
        # Сравниваем решения с разными шагами
        y_coarse = convergence_results[0]['y']
        y_fine = convergence_results[1]['y']
        x_coarse = convergence_results[0]['x']
        x_fine = convergence_results[1]['x']
        
        # Интерполируем грубое решение на сетку мелкого
        from scipy.interpolate import interp1d
        y_coarse_interp = interp1d(x_coarse, y_coarse, kind='linear')(x_fine)
        
        # Оценка погрешности по Рунге-Ромбергу (p=2)
        runge_error_fd = np.abs(y_fine - y_coarse_interp) / (2**2 - 1)
        max_runge_error_fd = np.max(runge_error_fd)
        mean_runge_error_fd = np.mean(runge_error_fd)
        
        # Фактическая погрешность для мелкой сетки
        y_exact_fine = exact_solution(x_fine)
        actual_error_fine = np.abs(y_fine - y_exact_fine)
        max_actual_fine = np.max(actual_error_fine)
        
        print(f"Сетка 1: N={convergence_results[0]['n']}, h={convergence_results[0]['h']:.6f}")
        print(f"Сетка 2: N={convergence_results[1]['n']}, h={convergence_results[1]['h']:.6f}")
        print(f"Максимальная оценка погрешности по Рунге-Ромбергу: {max_runge_error_fd:.15f}")
        print(f"Средняя оценка погрешности: {mean_runge_error_fd:.15f}")
        print(f"Фактическая максимальная погрешность (мелкая сетка): {max_actual_fine:.15f}")
        print(f"Отношение оценки к фактической: {max_runge_error_fd / max_actual_fine:.4f}")
    
    print("\n" + "=" * 80)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


# Функция compute_errors (добавляем глобально для доступности)
def compute_errors(x, yNum):
    """
    Вычисление абсолютной погрешности
    """
    yExact = exact_solution(x)
    errors = np.abs(yNum - yExact)
    maxError = np.max(errors)
    meanError = np.mean(errors)
    return errors, maxError, meanError, yExact


if __name__ == "__main__":
    main()