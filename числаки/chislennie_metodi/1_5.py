# Расширенная матрица [A|b] - исходные данные
matrix = [
    [-3, -4, -5, 15, 49],
    [-3, 14, 2, -7, 33],
    [13, -3, 4, -5, 44],
    [6, -5, 15, -3, 94]
]

n = len(matrix)  # размерность системы
eps = 0.0001     # точность вычислений

# ------------------------------------------------------------
# БЛОК 1: ПРОВЕРКА УСЛОВИЯ СХОДИМОСТИ
# Проверяем диагональное преобладание: |a_ii| > sum|a_ij| для j≠i
# ------------------------------------------------------------
def check_convergence(mat):
    print("Проверка достаточного условия сходимости:\n")
    
    convergent = True
    for i in range(n):
        diagonal = abs(mat[i][i])
        sum_others = sum(abs(mat[i][j]) for j in range(n) if j != i)
        
        print(f"Строка {i + 1}: |{mat[i][i]}| = {diagonal} > {sum_others} : {diagonal > sum_others}")
        
        if diagonal <= sum_others:
            convergent = False
    
    return convergent

# ------------------------------------------------------------
# БЛОК 2: ВЫЧИСЛЕНИЕ НОРМЫ МАТРИЦЫ
# Используем норму для оценки сходимости
# ------------------------------------------------------------
def calculate_norm(matrix):
    """Вычисляем строчную норму матрицы (максимальную сумму по строкам)"""
    norms = []
    for i in range(len(matrix)):
        row_sum = sum(abs(matrix[i][j]) for j in range(len(matrix)))
        norms.append(row_sum)
        print(f"Сумма модулей строки {i+1}: {row_sum:.4f}")
    
    norm = max(norms)
    print(f"Норма матрицы = {norm:.4f}")
    return norm

# ------------------------------------------------------------
# БЛОК 3: ПЕРЕСТАНОВКА СТРОК ДЛЯ УЛУЧШЕНИЯ СХОДИМОСТИ
# Ставим максимальные элементы на диагональ
# ------------------------------------------------------------
def rearrange_for_convergence(mat):
    A = [row[:] for row in mat]  # копируем матрицу
    
    for i in range(n):
        max_row = i
        max_val = abs(A[i][i])
        
        # ищем строку с максимальным элементом в i-ом столбце
        for k in range(n):
            if abs(A[k][i]) > max_val:
                max_val = abs(A[k][i])
                max_row = k
        
        # меняем строки местами
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
    
    return A

# ------------------------------------------------------------
# БЛОК 4: ПОДГОТОВКА ИТЕРАЦИОННЫХ ФОРМУЛ
# Преобразуем Ax = b к виду x = αx + β
# ------------------------------------------------------------
def prepare_iteration_form(mat):
    alpha = [[0.0] * n for _ in range(n)]  # матрица коэффициентов α
    beta = [0.0] * n                         # вектор свободных членов β
    
    for i in range(n):
        a_ii = mat[i][i]  # диагональный элемент
        for j in range(n):
            if i != j:
                # α[i][j] = -a[i][j]/a[i][i] для недиагональных элементов
                alpha[i][j] = -mat[i][j] / a_ii
            else:
                alpha[i][j] = 0  # диагональные элементы обнуляем
        beta[i] = mat[i][n] / a_ii  # β[i] = b[i]/a[i][i]
    
    # Выводим формулы для наглядности
    print("\nИтерационные формулы метода простых итераций:")
    for i in range(n):
        formula = f"x{i + 1}^(k+1) = "
        terms = []
        
        # в методе простых итераций все значения берутся с предыдущей итерации
        for j in range(n):
            if j != i and alpha[i][j] != 0:
                if alpha[i][j] > 0:
                    terms.append(f"{alpha[i][j]:.4f}*x{j + 1}^(k)")
                else:
                    terms.append(f"({alpha[i][j]:.4f})*x{j + 1}^(k)")
        
        if terms:
            formula += " + ".join(terms)
        else:
            formula += "0"
            
        if beta[i] >= 0:
            formula += f" + {beta[i]:.4f}"
        else:
            formula += f" - {abs(beta[i]):.4f}"
        print(formula)
    
    print("\nИтерационные формулы метода Зейделя:")
    for i in range(n):
        formula = f"x{i + 1}^(k+1) = "
        terms = []
        
        # добавляем члены с новыми значениями (j < i)
        for j in range(i):
            if alpha[i][j] != 0:
                if alpha[i][j] > 0:
                    terms.append(f"{alpha[i][j]:.4f}*x{j + 1}^(k+1)")
                else:
                    terms.append(f"({alpha[i][j]:.4f})*x{j + 1}^(k+1)")
        
        # добавляем члены со старыми значениями (j > i)
        for j in range(i + 1, n):
            if alpha[i][j] != 0:
                if alpha[i][j] > 0:
                    terms.append(f"{alpha[i][j]:.4f}*x{j + 1}^(k)")
                else:
                    terms.append(f"({alpha[i][j]:.4f})*x{j + 1}^(k)")
        
        formula += " + ".join(terms)
        if beta[i] >= 0:
            formula += f" + {beta[i]:.4f}"
        else:
            formula += f" - {abs(beta[i]):.4f}"
        print(formula)
    
    # Вычисляем и выводим норму матрицы α
    print("\nПроверка достаточного условия сходимости по норме матрицы α:")
    norm_alpha = calculate_norm(alpha)
    if norm_alpha < 1:
        print(f"Норма матрицы α = {norm_alpha:.4f} < 1 -> сходимость гарантирована")
    else:
        print(f"Норма матрицы α = {norm_alpha:.4f} >= 1 -> сходимость не гарантирована")
    print()
    
    return alpha, beta

# ------------------------------------------------------------
# БЛОК 5: МЕТОД ПРОСТЫХ ИТЕРАЦИЙ
# ------------------------------------------------------------
def simple_iteration_method(alpha, beta, epsilon):
    """Метод простых итераций - все значения берутся с предыдущей итерации"""
    x = beta[:]  # начальное приближение
    
    print(f"\n{'='*60}")
    print("МЕТОД ПРОСТЫХ ИТЕРАЦИЙ")
    print(f"{'='*60}")
    print(f"\nНачальное приближение x^(0):")
    for i in range(n):
        print(f"x{i + 1}^(0) = {x[i]:.6f}")
    
    iteration = 0
    max_iterations = 1000
    
    print(f"\nИтерационный процесс (ε = {epsilon}):\n")
    
    while iteration < max_iterations:
        x_new = [0.0] * n  # новое приближение
        
        # в методе простых итераций используем ТОЛЬКО старые значения
        for i in range(n):
            x_new[i] = beta[i]
            for j in range(n):
                x_new[i] += alpha[i][j] * x[j]  # все x[j] с предыдущей итерации
        
        iteration += 1
        
        # вычисляем нормы разности
        diff_max = max(abs(x_new[i] - x[i]) for i in range(n))
        diff_euclid = sum((x_new[i] - x[i])**2 for i in range(n))**0.5
        diff_l1 = sum(abs(x_new[i] - x[i]) for i in range(n))
        
        # выводим первые 5 итераций и последние перед сходимостью
        if iteration <= 5 or diff_max < epsilon:
            print(f"Итерация {iteration}:")
            for i in range(n):
                print(f"  x{i + 1} = {x_new[i]:.6f}")
            print(f"  Нормы разности:")
            print(f"    L∞ (max): {diff_max:.6f}")
            print(f"    L2 (евклидова): {diff_euclid:.6f}")
            print(f"    L1: {diff_l1:.6f}")
            print()
        
        x = x_new  # обновляем значения для следующей итерации
        
        # проверка условия остановки
        if diff_max < epsilon:
            print(f"Сходимость достигнута за {iteration} итераций\n")
            return x, iteration
    
    print(f"Предупреждение: достигнуто максимальное число итераций ({max_iterations})")
    return x, iteration

# ------------------------------------------------------------
# БЛОК 6: МЕТОД ЗЕЙДЕЛЯ
# ------------------------------------------------------------
def seidel_method(alpha, beta, epsilon):
    """Метод Зейделя - используем новые значения сразу"""
    x = beta[:]  # начальное приближение
    
    print(f"\n{'='*60}")
    print("МЕТОД ЗЕЙДЕЛЯ")
    print(f"{'='*60}")
    print(f"\nНачальное приближение x^(0):")
    for i in range(n):
        print(f"x{i + 1}^(0) = {x[i]:.6f}")
    
    iteration = 0
    max_iterations = 1000
    
    print(f"\nИтерационный процесс (ε = {epsilon}):\n")
    
    while iteration < max_iterations:
        x_old = x[:]  # запоминаем предыдущее приближение
        
        # вычисляем новое приближение
        for i in range(n):
            x[i] = beta[i]
            for j in range(n):
                # для j < i используем уже вычисленные x[j] (новые)
                # для j > i используем x[j] из предыдущей итерации (старые)
                x[i] += alpha[i][j] * x[j]
        
        iteration += 1
        
        # вычисляем нормы разности
        diff_max = max(abs(x[i] - x_old[i]) for i in range(n))
        diff_euclid = sum((x[i] - x_old[i])**2 for i in range(n))**0.5
        diff_l1 = sum(abs(x[i] - x_old[i]) for i in range(n))
        
        # выводим первые 5 итераций и последние перед сходимостью
        if iteration <= 5 or diff_max < epsilon:
            print(f"Итерация {iteration}:")
            for i in range(n):
                print(f"  x{i + 1} = {x[i]:.6f}")
            print(f"  Нормы разности:")
            print(f"    L∞ (max): {diff_max:.6f}")
            print(f"    L2 (евклидова): {diff_euclid:.6f}")
            print(f"    L1: {diff_l1:.6f}")
            print()
        
        # проверка условия остановки
        if diff_max < epsilon:
            print(f"Сходимость достигнута за {iteration} итераций\n")
            return x, iteration
    
    print(f"Предупреждение: достигнуто максимальное число итераций ({max_iterations})")
    return x, iteration

# ------------------------------------------------------------
# БЛОК 7: ОСНОВНАЯ ПРОГРАММА
# Последовательность выполнения
# ------------------------------------------------------------

# Шаг 1: Проверяем сходимость исходной системы
if not check_convergence(matrix):
    print("\nУсловие сходимости не выполнено для исходной системы.")
    print("Выполняем перестановку строк...\n")
    matrix = rearrange_for_convergence(matrix)
    
    # Шаг 2: Проверяем сходимость после перестановки
    if check_convergence(matrix):
        print("\nПосле перестановки условие сходимости выполнено!\n")
    else:
        print("\nВнимание: условие сходимости может не выполняться!")

# Шаг 3: Готовим итерационные формулы
alpha, beta = prepare_iteration_form(matrix)

# Шаг 4: Решаем систему двумя методами для сравнения
solution_simple, iter_simple = simple_iteration_method(alpha, beta, eps)
solution_seidel, iter_seidel = seidel_method(alpha, beta, eps)

# Шаг 5: Сравниваем результаты
print(f"\n{'='*60}")
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print(f"{'='*60}")

print("\nМетод простых итераций:")
for i in range(n):
    print(f"  x{i + 1} = {solution_simple[i]:.6f} ==> {solution_simple[i]:.1f}")
print(f"Количество итераций: {iter_simple}")

print("\nМетод Зейделя:")
for i in range(n):
    print(f"  x{i + 1} = {solution_seidel[i]:.6f} ==> {solution_seidel[i]:.1f}")
print(f"Количество итераций: {iter_seidel}")

# Шаг 6: Сравниваем скорость сходимости
if iter_simple < iter_seidel:
    print(f"\nМетод простых итераций быстрее на {iter_seidel - iter_simple} итераций")
elif iter_seidel < iter_simple:
    print(f"\nМетод Зейделя быстрее на {iter_simple - iter_seidel} итераций")
else:
    print("\nОба метода сошлись за одинаковое число итераций")