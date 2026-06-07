"""
ЛАБОРАТОРНАЯ РАБОТА: ЧИСЛЕННЫЕ МЕТОДЫ РЕШЕНИЯ ОДУ
==================================================
Реализация методов Эйлера, Рунге-Кутты и Адамса 4-го порядка
для решения задачи Коши.

Уравнение: x³·y' - y² = x⁴
Начальное условие: y(1) = 1 - 1/ln(2)
Интервал: x ∈ [1, 4]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")  # Отключаем предупреждения для чистоты вывода


#ОПРЕДЕЛЕНИЕ ЗАДАЧИ

def exact_solution(x_list):
    """
    Аналитическое (точное) решение дифференциального уравнения.
    
    Вывод аналитического решения:
    Исходное уравнение: x³·y' - y² = x⁴
    Это уравнение Бернулли, решение которого имеет вид:
    y(x) = x² - x² / ln(2x)
    
    Параметры:
    ----------
    x_list : list или array
        Список значений x, для которых вычисляется решение
        
    Возвращает:
    -----------
    list : значения точного решения y(x) в каждой точке
    """
    return [x ** 2 - x ** 2 / np.log(2 * x) for x in x_list]


def f(x, y):
    """
    Правая часть дифференциального уравнения, приведенного к нормальной форме.
    
    Исходное уравнение: x³·y' - y² = x⁴
    Выражаем y': y' = (x⁴ + y²) / x³
    
    Параметры:
    ----------
    x : float
        Независимая переменная
    y : float
        Искомая функция
        
    Возвращает:
    -----------
    float : значение y' = f(x, y)
    """
    if x == 0:
        return 0  # Избегаем деления на ноль (хотя в нашей задаче x >= 1)
    return (x ** 4 + y ** 2) / x ** 3


#ПАРАМЕТРЫ ЗАДАЧИ И СОЗДАНИЕ СЕТКИ

x0 = 1.0                    # Начальная точка
y0 = 1.0 - 1.0 / np.log(2)  # Начальное условие y(1) = 1 - 1/ln(2)
x_end = 4.0                 # Конечная точка
h = 0.20                    # Шаг сетки

# Создание равномерной сетки узлов
x = []          # Список узлов сетки
val = x0
while val < x_end + 0.5 * h:  # +0.5*h для учета погрешности округления
    x.append(val)
    val += h

n = len(x)      # Количество узлов


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def prod(b, K):
    """
    Вычисляет скалярное произведение векторов b и K.
    Используется для вычисления взвешенной суммы стадий в методах Рунге-Кутты.
    
    Формула: Σ(b[j] * K[j])
    
    Параметры:
    ----------
    b : list
        Вектор весов (коэффициентов) метода
    K : list
        Вектор стадий (промежуточных значений)
        
    Возвращает:
    -----------
    float : взвешенная сумма
    """
    return sum(b[j] * K[j] for j in range(len(b)))


# 4. ЯВНЫЕ МЕТОДЫ РУНГЕ-КУТТЫ

def explicit_runge_kutta(butcher_table, name=""):
    """
    Универсальная реализация явных методов Рунге-Кутты.
    
    Принцип работы:
    1. На каждом шаге вычисляем p стадий K_j:
       K_j = h * f(x_i + c_j·h, y_i + Σ(a_jk·K_k))
    2. Новое значение: y_{i+1} = y_i + Σ(b_j·K_j)
    
    Параметры:
    ----------
    butcher_table : tuple (A, b, c)
        A : матрица коэффициентов (p×p) - нижняя треугольная для явных методов
        b : вектор весов (p)
        c : вектор узлов (p)
    name : str
        Название метода (для отладки)
        
    Возвращает:
    -----------
    list : численное решение во всех узлах сетки
    """
    A, b, c = butcher_table
    p = len(b)                  # Количество стадий

    y = [0.0 for _ in range(n)]  # Инициализация массива решения
    y[0] = y0                    # Начальное условие

    for i in range(n - 1):       # Проход по всем узлам
        K = [0.0 for _ in range(p)]  # Стадии для текущего шага

        # Вычисление всех стадий
        for j in range(p):
            sum_AK = 0
            # Суммируем a_jk * K_k для k < j (явный метод)
            for k in range(j):
                sum_AK += A[j][k] * K[k]

            # Вычисляем j-ю стадию
            K[j] = h * f(x[i] + c[j] * h, y[i] + sum_AK)

        # Вычисляем новое значение y_{i+1}
        y[i + 1] = y[i] + prod(b, K)

    return y


# ТАБЛИЦЫ БУТЧЕРА ДЛЯ РАЗЛИЧНЫХ МЕТОДОВ

# Метод Эйлера (1-й порядок)
# Формула: y_{n+1} = y_n + h·f(x_n, y_n)
euler_table = (
    [[0]],      # A - нет зависимостей
    [1],        # b - один вес
    [0]         # c - узел в начале
)

# Метод Хойна 2-го порядка (улучшенный метод Эйлера)
# 2 стадии: k1 = h·f(x_n, y_n)
#           k2 = h·f(x_n + h/2, y_n + k1/2)
#           y_{n+1} = y_n + k2
heun2_table = (
    [[0, 0],
     [1 / 2, 0]],   # A: k2 зависит от k1
    [0, 1],         # b: используется только k2
    [0, 1 / 2]      # c: узлы в 0 и 0.5
)

# Метод Хойна 3-го порядка (3 стадии)
heun3_table = (
    [[0, 0, 0],
     [1 / 3, 0, 0],
     [0, 2 / 3, 0]],
    [1 / 4, 0, 3 / 4],
    [0, 1 / 3, 2 / 3]
)

# Классический метод Рунге-Кутты 4-го порядка (4 стадии)
# Самый популярный метод: RK4
# k1 = h·f(x_n, y_n)
# k2 = h·f(x_n + h/2, y_n + k1/2)
# k3 = h·f(x_n + h/2, y_n + k2/2)
# k4 = h·f(x_n + h, y_n + k3)
# y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6
rk4_table = (
    [[0, 0, 0, 0],
     [1 / 2, 0, 0, 0],
     [0, 1 / 2, 0, 0],
     [0, 0, 1, 0]],
    [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    [0, 1 / 2, 1 / 2, 1]
)


# 6. ВЛОЖЕННЫЙ МЕТОД МЕРСОНА (3(4) порядок)

def embedded_runge_kutta(butcher_table, name=""):
    """
    Реализация вложенных методов Рунге-Кутты (используется для контроля ошибки).
    
    Вложенные методы вычисляют два решения разного порядка
    с минимальными дополнительными затратами.
    
    Параметры аналогичны explicit_runge_kutta, но таблица Бутчера
    содержит два вектора b: для основного и вложенного метода.
    """
    A, b, b_star, c = butcher_table  # b_star - веса для вложенного метода
    p = len(b)

    y = [0.0 for _ in range(n)]
    y[0] = y0

    for i in range(n - 1):
        K = [0.0 for _ in range(p)]

        for j in range(p):
            sum_AK = 0
            for k in range(j):
                sum_AK += A[j][k] * K[k]

            K[j] = h * f(x[i] + c[j] * h, y[i] + sum_AK)

        # В текущей реализации используем только основной метод
        y[i + 1] = y[i] + prod(b, K)

    return y


# Таблица Бутчера для метода Мерсона 3(4)
# Используется для адаптивного выбора шага
merson_table = (
    [[0, 0, 0, 0, 0],
     [1 / 3, 0, 0, 0, 0],
     [1 / 6, 1 / 6, 0, 0, 0],
     [1 / 8, 0, 3 / 8, 0, 0],
     [1 / 2, 0, -3 / 2, 2, 0]],
    [1 / 10, 0, 3 / 10, 2 / 5, 1 / 5],   # b - основной метод (4-й порядок)
    [1 / 6, 0, 0, 2 / 3, 1 / 6],         # b* - вложенный метод (3-й порядок)
    [0, 1 / 3, 1 / 3, 1 / 2, 1]          # c - узлы
)


# 7. НЕЯВНЫЕ МЕТОДЫ РУНГЕ-КУТТЫ

def fixed_point_K(A, b, c, f, x, y, h, s, i, K, tol=1e-10, max_iter=100):
    """
    Метод простой итерации для решения нелинейной системы уравнений
    при вычислении стадий K в неявных методах.
    
    Неявные методы имеют стадии, зависящие от неизвестных K_j,
    что приводит к системе уравнений: K = h·f(x + c·h, y + A·K)
    
    Параметры:
    ----------
    A, b, c : таблица Бутчера
    f : правая часть уравнения
    x, y : массивы узлов и решений
    h : шаг
    s : количество стадий
    i : индекс текущего узла
    K : начальное приближение
    tol : допустимая погрешность
    max_iter : максимальное число итераций
    
    Возвращает:
    -----------
    list : найденные стадии K
    """
    for iteration in range(max_iter):
        K_new = []

        # Вычисляем новые значения стадий
        for j in range(s):
            sum_AK = 0
            for k in range(s):
                sum_AK += A[j][k] * K[k]

            val = h * f(x[i] + c[j] * h, y[i] + sum_AK)
            K_new.append(val)

        # Проверка сходимости
        if max(abs(K_new[j] - K[j]) for j in range(s)) < tol:
            return K_new

        K = K_new

    raise Exception("Стадии K не сошлись за max_iter итераций")


def implicit_runge_kutta(butcher_table, name=""):
    """
    Универсальная реализация неявных методов Рунге-Кутты.
    
    Отличие от явных: стадии K зависят друг от друга, требуется
    решение системы нелинейных уравнений.
    
    Параметры:
    ----------
    butcher_table : tuple (A, b, c)
        A : полная матрица коэффициентов (не обязательно треугольная)
        b, c : как в явных методах
    """
    A, b, c = butcher_table
    s = len(b)          # Количество стадий

    y = [0.0 for _ in range(n)]
    y[0] = y0

    for i in range(n - 1):
        # Начальное приближение: явный шаг Эйлера
        K0 = [h * f(x[i] + c[j] * h, y[i]) for j in range(s)]

        # Решаем нелинейную систему методом простой итерации
        K_sol = fixed_point_K(A, b, c, f, x, y, h, s, i, K0)

        y[i + 1] = y[i] + prod(b, K_sol)

    return y


# Метод Гаусса 2-го порядка (неявный, одностадийный)
# Формула: y_{n+1} = y_n + h·f(x_n + h/2, (y_n + y_{n+1})/2)
# Это метод средней точки (трапеций)
gauss2_table = (
    [[1 / 2]],      # A = [1/2]
    [1],            # b = [1]
    [1 / 2]         # c = [1/2]
)

# Метод Гаусса 4-го порядка (неявный, двухстадийный)
# Самый точный метод из рассматриваемых
sqrt3 = 3 ** 0.5
gauss4_table = (
    [[1 / 4, 1 / 4 - sqrt3 / 6],
     [1 / 4 + sqrt3 / 6, 1 / 4]],
    [1 / 2, 1 / 2],
    [1 / 2 - sqrt3 / 6, 1 / 2 + sqrt3 / 6]
)


#МЕТОДЫ АДАМСА (МНОГОШАГОВЫЕ)

def adams_moulton_simple_iter(i, x, y, h, coefficients, f, tol=1e-6, max_iter=100):
    """
    Метод простой итерации для неявного метода Адамса-Моултона.
    
    Неявная формула: y_{i+1} = y_i + h·[β₀·f(x_{i+1}, y_{i+1}) + Σβ_j·f(x_{i+1-j}, y_{i+1-j})]
    Решаем относительно y_{i+1} методом простой итерации.
    
    Параметры:
    ----------
    i : индекс текущего узла
    x, y : массивы узлов и решений
    h : шаг
    coefficients : коэффициенты метода Адамса
    f : правая часть
    tol, max_iter : параметры сходимости
    """
    y_next = y[i]  # Начальное приближение

    for _ in range(max_iter):
        sum_coeff = 0

        # Вычисляем сумму коэффициентов
        for j in range(len(coefficients)):
            if j == 0:
                # Неизвестное значение на новом шаге
                sum_coeff += coefficients[j] * f(x[i + 1], y_next)
            else:
                # Известные значения с предыдущих шагов
                sum_coeff += coefficients[j] * f(x[i + 1 - j], y[i + 1 - j])

        y_new = y[i] + h * sum_coeff

        if abs(y_new - y_next) < tol:
            return y_new

        y_next = y_new

    raise Exception("Не сошлось")


def adams_method(coefficients, implicit=False, name=""):
    """
    Реализация методов Адамса (многошаговых методов).
    
    Явный метод Адамса-Башфорта: использует только известные значения
    Неявный метод Адамса-Моултона: использует также неизвестное значение
    
    Особенность: для старта нужны значения в нескольких первых точках.
    Используем метод Рунге-Кутты 4-го порядка для разгона.
    
    Параметры:
    ----------
    coefficients : list
        Коэффициенты метода (например, [55/24, -59/24, 37/24, -9/24] для AB4)
    implicit : bool
        True - неявный метод (Адамса-Моултона)
        False - явный метод (Адамса-Башфорта)
    name : str
        Название метода (для отладки)
    """
    k = len(coefficients) - 1  # Количество предыдущих точек (порядок метода)

    y = [0.0 for _ in range(n)]
    
    # Разгон: получаем первые k+1 значений методом RK4
    y_rk4 = explicit_runge_kutta(rk4_table)
    y[:k + 1] = y_rk4[:k + 1]

    # Основной цикл
    for i in range(k, n - 1):
        if implicit:
            # Неявный метод Адамса-Моултона
            y[i + 1] = adams_moulton_simple_iter(i, x, y, h, coefficients, f)
        else:
            # Явный метод Адамса-Башфорта
            sum_coeff = 0
            for j in range(k + 1):
                sum_coeff += coefficients[j] * f(x[i - j], y[i - j])

            y[i + 1] = y[i] + h * sum_coeff

    return y


# Коэффициенты методов Адамса 4-го порядка
# Явный метод Адамса-Башфорта 4-го порядка
# Формула: y_{n+1} = y_n + h·(55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})/24
adams_bashforth_coeff = [55 / 24, -59 / 24, 37 / 24, -9 / 24]

# Неявный метод Адамса-Моултона 4-го порядка (метод коррекции)
# Формула: y_{n+1} = y_n + h·(9f_{n+1} + 19f_n - 5f_{n-1} + f_{n-2})/24
adams_moulton_coeff = [9 / 24, 19 / 24, -5 / 24, 1 / 24]


# 9. ВЫЧИСЛЕНИЕ РЕШЕНИЙ ВСЕМИ МЕТОДАМИ
# Словарь всех методов и их решений
methods = {
    'Явный метод Эйлера': explicit_runge_kutta(euler_table),
    'Метод Хойна 2 порядка': explicit_runge_kutta(heun2_table),
    'Метод Хойна 3 порядка': explicit_runge_kutta(heun3_table),
    'Классический RK4': explicit_runge_kutta(rk4_table),
    'Мерсон 3(4)': embedded_runge_kutta(merson_table),
    'Неявный метод Гаусса 2 порядка': implicit_runge_kutta(gauss2_table),
    'Неявный метод Гаусса 4 порядка': implicit_runge_kutta(gauss4_table),
    'Адамс-Башфорт 4': adams_method(adams_bashforth_coeff, implicit=False),
    'Адамс-Моултон 4': adams_method(adams_moulton_coeff, implicit=True),
}

# Вычисляем аналитическое решение для сравнения
exact_sol = exact_solution(x)



print("ЗАДАЧА КОШИ")
print("=" * 70)
print(f"Уравнение: x³y' - y² = x⁴")
print(f"y({x0}) = {y0:.6f}, x ∈ [{x0}; {x_end}], h = {h}")
print()

print("СВОДНАЯ ТАБЛИЦА МАКСИМАЛЬНЫХ ПОГРЕШНОСТЕЙ")
print("=" * 70)
print(f"{'Метод':<40} | {'Порядок':^8} | {'Макс. погр.':>12}")
print("-" * 70)

# Словарь порядков точности методов
orders = {
    'Явный метод Эйлера': 1,
    'Метод Хойна 2 порядка': 2,
    'Метод Хойна 3 порядка': 3,
    'Классический RK4': 4,
    'Мерсон 3(4)': 4,
    'Неявный метод Гаусса 2 порядка': 2,
    'Неявный метод Гаусса 4 порядка': 4,
    'Адамс-Башфорт 4': 4,
    'Адамс-Моултон 4': 4
}

max_errors = {}  # Словарь для хранения максимальных погрешностей

for name, solution in methods.items():
    # Вычисляем погрешность в каждой точке
    error = [abs(a - b) for a, b in zip(solution, exact_sol)]
    max_error = max(error)
    max_errors[name] = max_error

    order = orders[name]
    method_name = name.ljust(40)
    order_str = str(order).center(8)
    error_str = f"{max_error:.10f}".rjust(18)

    print(f"{method_name} | {order_str} | {error_str}")

print("=" * 70)
print()




plt.figure(figsize=(14, 7))

# График 1: Сравнение численных решений с аналитическим
plt.subplot(1, 2, 1)
plt.plot(x, exact_sol, 'k-', linewidth=2.5, label='Аналитическое решение', zorder=5)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
color_idx = 0

for name, solution in methods.items():
    plt.plot(x, solution, 'o-', color=colors[color_idx], markersize=3,
             label=name, alpha=0.7, linewidth=1.2)
    color_idx += 1

plt.xlabel('x', fontsize=11)
plt.ylabel('y(x)', fontsize=11)
plt.title('Решение задачи Коши: x³y\' - y² = x⁴, y(1) = 1 - 1/ln(2)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)

# График 2: Погрешность в логарифмическом масштабе
plt.subplot(1, 2, 2)
for name, solution in methods.items():
    error = [abs(a - b) for a, b in zip(solution, exact_sol)]
    # Защита от логарифма нуля
    error = [max(e, 1e-16) for e in error]
    plt.semilogy(x, error, 'o-', label=name, markersize=3, linewidth=1.2)

plt.xlabel('x', fontsize=11)
plt.ylabel('|Погрешность| (логарифм)', fontsize=11)
plt.title('Погрешность численных методов', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ПОГРЕШНОСТЕЙ")
print("=" * 85)
print(f"{'Метод':<40} | {'Средняя погр.':>13} | {'Погр. в конце':>13}")
print("-" * 85)

for name, solution in methods.items():
    error = [abs(a - b) for a, b in zip(solution, exact_sol)]
    mean_error = np.mean(error)      # Средняя погрешность на всем интервале
    final_error = error[-1]          # Погрешность в конечной точке

    method_name = name.ljust(40)
    mean_str = f"{mean_error:.10f}".center(18)
    final_str = f"{final_error:.10f}".center(18)

    print(f"{method_name} | {mean_str} | {final_str}")

print("=" * 85)