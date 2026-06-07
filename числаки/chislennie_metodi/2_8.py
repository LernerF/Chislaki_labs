import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def f1(x, y):
    return 4 * x ** 2 * y + 3 * y ** 2 + 5 * x + y - 4


def f2(x, y):
    return 3 * x ** 2 + 2 * x * y - 2 * y ** 2 + 5 * x - 3 * y + 2


def df1dx(x, y):
    return 8 * x * y + 5


def df1dy(x, y):
    return 4 * x ** 2 + 6 * y + 1


def df2dx(x, y):
    return 6 * x + 2 * y + 5


def df2dy(x, y):
    return 2 * x - 4 * y - 3


initials = [
    (0.2, 0.8),  # Для корня около (0.228, 0.796)
    (-0.7, -2.2),  # Для корня около (-0.748, -2.234)
    (-2.2, 0.6),  # Для корня около (-2.235, 0.661)
    (1.5, -2.8)  # Для корня около (1.473, -2.831)
]

eps = 0.0001
max_iter = 10000


def check_convergence_iteration_methods(x, y, lambda1, lambda2):
    dphi1_dx = 1 + lambda1 * df1dx(x, y)
    dphi1_dy = lambda1 * df1dy(x, y)
    dphi2_dx = lambda2 * df2dx(x, y)
    dphi2_dy = 1 + lambda2 * df2dy(x, y)

    # Проверка по СТРОКАМ
    sum_row1 = abs(dphi1_dx) + abs(dphi1_dy)  # Сумма модулей 1-й строки
    sum_row2 = abs(dphi2_dx) + abs(dphi2_dy)  # Сумма модулей 2-й строки
    norm_rows = max(sum_row1, sum_row2)

    # Проверка по СТОЛБЦАМ
    sum_col1 = abs(dphi1_dx) + abs(dphi2_dx)  # Сумма модулей 1-го столбца
    sum_col2 = abs(dphi1_dy) + abs(dphi2_dy)  # Сумма модулей 2-го столбца
    norm_cols = max(sum_col1, sum_col2)

    # Проверяем условие: норма < 1 ИЛИ по строкам, ИЛИ по столбцам
    check_rows = norm_rows < 1
    check_cols = norm_cols < 1

    if check_rows or check_cols:
        # Хотя бы одно условие выполнено
        if check_rows and check_cols:
            message = f"Выполнено оба условия: (строки) = {norm_rows:.4f} < 1 и (столбцы) = {norm_cols:.4f} < 1"
        elif check_rows:
            message = f"Выполнено условие по СТРОКАМ: = {norm_rows:.4f} < 1 (по столбцам: {norm_cols:.4f})"
        else:
            message = f"Выполнено условие по СТОЛБЦАМ: = {norm_cols:.4f} < 1 (по строкам: {norm_rows:.4f})"

        return True, norm_rows, norm_cols, message, dphi1_dx, dphi1_dy, dphi2_dx, dphi2_dy
    else:
        # Ни одно условие не выполнено
        message = f"НЕ выполнено: (строки) = {norm_rows:.4f} ≥ 1 И (столбцы) = {norm_cols:.4f} ≥ 1"
        return False, norm_rows, norm_cols, message, dphi1_dx, dphi1_dy, dphi2_dx, dphi2_dy


def check_newton_convergence(x, y):
    det_J = df1dx(x, y) * df2dy(x, y) - df1dy(x, y) * df2dx(x, y)

    if abs(det_J) < 1e-10:
        return False, "✗ Якобиан вырожден (det J ≈ 0)"

    delta = 0.01
    points = [
        (x + delta, y), (x - delta, y),
        (x, y + delta), (x, y - delta),
        (x + delta, y + delta), (x - delta, y - delta)
    ]

    sign_det = np.sign(det_J)
    for px, py in points:
        det_test = df1dx(px, py) * df2dy(px, py) - df1dy(px, py) * df2dx(px, py)
        if abs(det_test) < 1e-10 or np.sign(det_test) != sign_det:
            return False, f"✗ Якобиан меняет знак в окрестности (det_J = {det_J:.6f})"

    return True, f"✓ Условия сходимости выполнены (det_J = {det_J:.6f})"


def check_solution(x, y):
    f1_val = f1(x, y)
    f2_val = f2(x, y)
    residual = np.sqrt(f1_val ** 2 + f2_val ** 2)
    return f1_val, f2_val, residual


def newton_method(x0, y0):
    x = x0
    y = y0
    it = 0

    while True:
        it += 1

        f1_val = f1(x, y)
        f2_val = f2(x, y)

        df1_dx_val = df1dx(x, y)
        df1_dy_val = df1dy(x, y)
        df2_dx_val = df2dx(x, y)
        df2_dy_val = df2dy(x, y)

        det_J = df1_dx_val * df2_dy_val - df1_dy_val * df2_dx_val

        if abs(det_J) < 1e-10:
            return None, it, "Сингулярный якобиан (det J ≈ 0)"

        det_A1 = f1_val * df2_dy_val - f2_val * df1_dy_val
        det_A2 = df1_dx_val * f2_val - df2_dx_val * f1_val

        delta_x = -det_A1 / det_J
        delta_y = -det_A2 / det_J

        x_new = x + delta_x
        y_new = y + delta_y

        diff = np.sqrt(delta_x ** 2 + delta_y ** 2)

        x = x_new
        y = y_new

        if diff < eps:
            return (x, y), it, "Сошлось"

        if it > max_iter:
            return None, it, "Превышено максимальное количество итераций"


def simple_iteration(x0, y0, lambda1, lambda2):
    x = x0
    y = y0
    it = 0

    while True:
        it += 1

        f1_val = f1(x, y)
        f2_val = f2(x, y)

        x_new = x + lambda1 * f1_val
        y_new = y + lambda2 * f2_val

        diff = np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2)

        x = x_new
        y = y_new

        if diff < eps:
            return (x, y), it, "Сошлось"

        if it > max_iter:
            return None, it, "Превышено максимальное количество итераций"

        if abs(x) > 1e10 or abs(y) > 1e10:
            return None, it, "Обнаружено переполнение (расходимость)"


def seidel_method(x0, y0, lambda1, lambda2):
    x = x0
    y = y0
    it = 0

    while True:
        it += 1

        f1_val = f1(x, y)
        x_new = x + lambda1 * f1_val

        f2_val = f2(x_new, y)
        y_new = y + lambda2 * f2_val

        diff = np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2)

        x = x_new
        y = y_new

        if diff < eps:
            return (x, y), it, "Сошлось"

        if it > max_iter:
            return None, it, "Превышено максимальное количество итераций"

        if abs(x) > 1e10 or abs(y) > 1e10:
            return None, it, "Обнаружено переполнение (расходимость)"


roots = []
all_initials = []

for idx, (x0, y0) in enumerate(initials, 1):
    print(f"\n{'=' * 70}")
    print(f"Начальное приближение {idx}: x₀ = {x0}, y₀ = {y0}")
    print(f"{'=' * 70}")
    all_initials.append((x0, y0))

    diag1 = df1dx(x0, y0)
    diag2 = df2dy(x0, y0)

    if abs(diag1) < 1e-3:
        lambda1 = -0.01
    else:
        lambda1 = -1 / (2 * diag1)

    if abs(diag2) < 1e-3:
        lambda2 = -0.01
    else:
        lambda2 = -1 / (2 * diag2)

    conv_check, norm_rows, norm_cols, message, dphi1_dx, dphi1_dy, dphi2_dx, dphi2_dy = check_convergence_iteration_methods(
        x0, y0, lambda1, lambda2)

    print(f"\n--- Метод простой итерации ---")
    print(f"Проверка условия сходимости: {message}")
    root, it, status = simple_iteration(x0, y0, lambda1, lambda2)
    if root:
        x, y = root
        print(f"Результат: x = {x:.6f}, y = {y:.6f}, итераций: {it}")
    else:
        print(f"Результат: {status}, итераций: {it}")

    print(f"\n--- Метод Зейделя ---")
    print(f"Проверка условия сходимости: {message}")
    root, it, status = seidel_method(x0, y0, lambda1, lambda2)
    if root:
        x, y = root
        print(f"Результат: x = {x:.6f}, y = {y:.6f}, итераций: {it}")
    else:
        print(f"Результат: {status}, итераций: {it}")

    print(f"\n--- Метод Ньютона ---")
    conv_newton, msg_newton = check_newton_convergence(x0, y0)
    print(f"Проверка условия сходимости: {msg_newton}")

    root, it, status = newton_method(x0, y0)
    if root:
        x, y = root
        print(f"Результат: x = {x:.6f}, y = {y:.6f}, итераций: {it}")
        roots.append((x, y))
    else:
        print(f"Результат: {status}, итераций: {it}")

print(f"\n{'=' * 70}")

x_range = np.linspace(-4, 3, 300)
y_range = np.linspace(-4, 2, 300)
X, Y = np.meshgrid(x_range, y_range)
F1 = f1(X, Y)
F2 = f2(X, Y)

fig = plt.figure(figsize=(14, 10))
ax = fig.gca()

contour1 = plt.contour(X, Y, F1, levels=[0], colors='red', linewidths=2)
contour2 = plt.contour(X, Y, F2, levels=[0], colors='blue', linewidths=2)

if all_initials:
    init_x, init_y = zip(*all_initials)
    plt.scatter(init_x, init_y, color='purple', marker='x', s=150,
                linewidths=3, label='Начальные приближения', zorder=5)
    for i, (ix, iy) in enumerate(all_initials, 1):
        plt.annotate(f'x₀⁽{i}⁾', (ix, iy), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color='purple')

if roots:
    roots_x, roots_y = zip(*roots)
    plt.scatter(roots_x, roots_y, color='green', marker='o', s=200,
                edgecolors='black', linewidths=2, label='Найденные корни', zorder=6)
    for i, (rx, ry) in enumerate(roots, 1):
        plt.annotate(f'({rx:.3f}, {ry:.3f})', (rx, ry),
                     xytext=(10, -10), textcoords='offset points',
                     fontsize=9, color='green', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Графики уравнений f₁(x,y)=0 и f₂(x,y)=0\nс начальными приближениями и найденными корнями',
          fontsize=14, fontweight='bold')
plt.legend(['f₁(x,y) = 0', 'f₂(x,y) = 0', 'Начальные приближения', 'Найденные корни'],
           fontsize=11, loc='best')
plt.grid(True, alpha=0.3)

# Жирные оси координат
plt.axhline(y=0, color='black', linewidth=3, alpha=0.8, zorder=4)
plt.axvline(x=0, color='black', linewidth=3, alpha=0.8, zorder=4)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.tick_params(width=2, length=6)

plt.tight_layout()

output_file = '2_8.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"График сохранен в файл: {output_file}")

plt.show()

print(f"\n{'=' * 70}")
print(f"ИТОГОВАЯ СВОДКА")
print(f"{'=' * 70}")
print(f"Найдено корней: {len(roots)}")
if roots:
    print(f"\nКорни системы (метод Ньютона):")
    for i, (x, y) in enumerate(roots, 1):
        print(f"  {i}. x = {x:.6f}, y = {y:.6f}")
print(f"{'=' * 70}")