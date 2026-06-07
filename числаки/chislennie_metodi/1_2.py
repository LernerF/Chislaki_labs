import math

# Расширенная матрица [A|b] - первые 5 столбцов: матрица A, последний: вектор b
matrix = [
    [2, -5, -4, 7, 4, 13],
    [5, 2, 1, -5, 2, 43],
    [4, 9, 7, -4, -5, 24],
    [2, 5, -4, 9, 3, 30],
    [8, 3, -5, -6, -2, 81]
]

n = len(matrix)  # размерность системы (5)


def build_augmented_from_matrix(mat):
    """Создаёт расширенную матрицу [A | b | I] для одновременного получения решения и обратной матрицы"""
    A_only = [row[:n] for row in mat]      # отделяем матрицу A (первые n столбцов)
    b_vec = [row[n] for row in mat]        # отделяем вектор b (последний столбец)

    augmented = []
    for i in range(n):
        row = [float(A_only[i][j]) for j in range(n)]  # копируем строку A
        row.append(float(b_vec[i]))                     # добавляем правую часть b
        row += [1.0 if j == i else 0.0 for j in range(n)]  # добавляем единичную матрицу I
        augmented.append(row)

    return augmented  # возвращаем матрицу размера n × (n + 1 + n) = n × (2n+1)


def process_lu(aug):
    """LU-разложение с компактной схемой хранения (L и U в одной матрице)"""
    total_cols = len(aug[0])  # общее количество столбцов (2n+1)
    
    for i in range(n):           # i - номер шага (диагональный элемент)
        for k in range(i + 1, n): # k - строки ниже текущей
            # Проверка на вырожденность
            if abs(aug[i][i]) < 1e-15:
                raise ZeroDivisionError("Матрица вырождена")
            
            # Вычисляем множитель для обнуления элемента в строке k, столбце i
            factor = aug[k][i] / aug[i][i]
            
            # КЛЮЧЕВОЙ МОМЕНТ: сохраняем множитель в матрицу (это будет элемент L)
            aug[k][i] = factor
            
            # Обновляем остальные элементы строки (начиная со следующего столбца)
            for j in range(i + 1, total_cols):
                aug[k][j] -= factor * aug[i][j]
    
    return aug  # теперь в матрице: ниже диагонали - L, на диагонали и выше - U


def det_from_lu(aug):
    """Вычисление определителя из LU-разложения"""
    det = 1.0
    for i in range(n):
        det *= aug[i][i]  # определитель = произведение диагональных элементов U
    return det


def backward_substitution_for_aug(aug):
    """Обратная подстановка (метод Гаусса-Жордана) - превращает левую часть в единичную матрицу"""
    total_cols = len(aug[0])

    # Идём СНИЗУ ВВЕРХ по строкам
    for i in range(n - 1, -1, -1):
        pivot = aug[i][i]  # диагональный элемент
        
        # Проверка на вырожденность
        if abs(pivot) < 1e-15:
            raise ZeroDivisionError("Нулевой диагональный элемент")
        
        # ШАГ 1: Нормализуем текущую строку (делим на диагональный элемент)
        for j in range(i, total_cols):
            aug[i][j] /= pivot
        
        # ШАГ 2: Обнуляем элементы ВЫШЕ диагонали в текущем столбце
        for r in range(i):  # r - строки выше текущей
            factor = aug[r][i]  # элемент в строке r, столбце i
            if abs(factor) > 0.0:  # если не ноль
                for j in range(i, total_cols):
                    aug[r][j] -= factor * aug[i][j]
    
    return aug  # теперь левая часть (первые n столбцов) - единичная матрица


def extract_solution_and_inverse(aug):
    """Извлекает решение и обратную матрицу из обработанной расширенной матрицы"""
    # Решение - это столбец с индексом n (после матрицы A)
    x = [aug[i][n] for i in range(n)]
    
    # Обратная матрица - это последние n столбцов (начиная с n+1)
    inv = [[aug[i][n + 1 + j] for j in range(n)] for i in range(n)]
    
    return x, inv


def multiply_matrices(A, B):
    """Умножение двух квадратных матриц"""
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]  # строка на столбец
            C[i][j] = s
    return C


def multiply_matrix_vector(A, x):
    """Умножение матрицы на вектор"""
    res = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]  # сумма по строке
        res[i] = s
    return res


def print_matrix(M):
    """Красивый вывод матрицы с 3 знаками после запятой"""
    for row in M:
        print("[" + ", ".join(f"{val: .3f}" for val in row) + "]")


def print_vector(v):
    """Красивый вывод вектора с 3 знаками после запятой"""
    print("[" + ", ".join(f"{val: .3f}" for val in v) + "]")


# ========== ОСНОВНАЯ ПРОГРАММА ==========

# 1. Выводим исходную матрицу
print("Исходная расширенная матрица [A | b]:")
print_matrix(matrix)
print()

# 2. Строим расширенную матрицу [A | b | I]
augmented = build_augmented_from_matrix(matrix)
print("Начальная расширенная матрица [A | b | I]:")
print_matrix(augmented)
print()

# 3. Выполняем LU-разложение
augmented = process_lu(augmented)
print("Матрица после LU-разложения (L в нижней части, U в верхней):")
print_matrix(augmented)
print()

# 4. Вычисляем определитель (произведение диагональных элементов U)
detA = det_from_lu(augmented)
print(f"Определитель матрицы A: det(A) = {detA:.3f}")
print()

# 5. Выполняем обратную подстановку (превращаем левую часть в единичную)
augmented = backward_substitution_for_aug(augmented)
print("Матрица после обратной подстановки (в правой части — [x | A^{-1}]):")
print_matrix(augmented)
print()

# 6. Извлекаем решение и обратную матрицу
solution, inverse_a = extract_solution_and_inverse(augmented)
print("Решение СЛАУ x:")
print_vector(solution)
print()

# 7. ПРОВЕРКА: A * x должна равняться b
A_only = [row[:n] for row in matrix]  # берём только матрицу A
Ax = multiply_matrix_vector(A_only, solution)
print("Проверка A * x (должно совпадать с b):")
print_vector(Ax)
print()

# 8. Выводим обратную матрицу
print("Обратная матрица A^(-1):")
print_matrix(inverse_a)
print()

# 9. ПРОВЕРКА: A * A^(-1) должна равняться единичной матрице I
prod = multiply_matrices(A_only, inverse_a)
print("Проверка A * A^(-1) = I:")
print_matrix(prod)