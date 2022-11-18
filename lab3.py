import numpy as np
from numpy.polynomial import Polynomial


def Faddeev_LeVerrier_algorithm(A: np.array):
    """
    Находит коэффициенты характеристического уравнения матрицы A
    https://en.wikipedia.org/wiki/Faddeev%E2%80%93LeVerrier_algorithm
    """
    poly_degree = A.shape[0]
    M = np.eye(poly_degree)
    # Список коэффициентов инициализируем единицами
    # Первый будет всегда 1, остальные перезапишутся в цикле
    coeffs = np.ones(poly_degree+1)
    for i in np.arange(poly_degree) + 1:
        AM = np.dot(A, M)
        coeffs[i] = -1 / i * np.trace(AM)
        # Прибавляем коэффициент к главной диагонали
        M = AM + np.eye(poly_degree) * coeffs[i]
    
    return coeffs

# ======================================================================
# Функции для численного поиска корней уравнения
# ======================================================================

# Используется для инициализации корней
def random_uniform_complex(radius: float, k: int):
    """
    Возвращает массив из k случайных комплексных чисел
    Равномерное распределение на [-radius, radius]
    """
    a = np.random.uniform(-radius, radius, k)
    b = np.random.uniform(-radius, radius, k)
    return a + 1j * b


# Ошибка для алгоритмов поиска корней
class RootsNotFoundError(Exception):
    def __init__(self):
        default_message = "Algorithm for finding roots has not converged"
        super().__init__(default_message)

        
# Декоратор для перезапуска поиска
# Пока не используется
def retry(times):
    def decorator(func):
        
        def new_func(*args, **kwargs):
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except RootsNotFoundError:
                    pass
            return func(*args, **kwargs)
        
        return new_func
    return decorator


def Durand_Kerner_method(poly_coeff: np.array, 
                         max_steps=100,
                         atol=1e-7):
    """
    Находит корни полинома с коэффициентами poly_coeff
      Если не найдет корни за max_steps с точностью atol:
      Бросает исключение RootsNotFoundError
    https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
    """
    
    poly_degree = len(poly_coeff) - 1
    f = Polynomial(poly_coeff[::-1])
    
    # Инициализируем корни случайным образом
    init_radius = 1 + np.abs(poly_coeff).max()
    roots = random_uniform_complex(init_radius, poly_degree)

    for _ in range(max_steps):
        for var_index in range(poly_degree):
            num = f(roots[var_index])
            # Произведение (xi-xj), i != j
            diff = roots[var_index] - np.delete(roots, var_index)
            den = np.prod(diff)
            roots[var_index] -= num / den
            
        if np.allclose(f(roots), 0, atol=atol):
            return roots
        
    raise RootsNotFoundError()
    
    
def Aberth_method(poly_coeff: np.array, 
                  max_steps=100,
                  atol=1e-7):
    
    """
    Находит корни полинома с коэффициентами poly_coeff
      Если не найдет корни за max_steps с точностью atol:
      Бросает исключение RootsNotFoundError
    https://en.wikipedia.org/wiki/Aberth_method
    """
    
    poly_degree = len(poly_coeff) - 1
    f = Polynomial(poly_coeff[::-1])
    
    # Инициализируем корни случайным образом
    init_radius = 1 + np.abs(poly_coeff).max()
    roots = random_uniform_complex(init_radius, poly_degree)

    for iteration in range(max_steps):
        
        num = f(roots) / f.deriv()(roots)
        s = np.array([
            np.sum(np.reciprocal(
                # (xi-xj), i != j
                roots[i] - np.delete(roots, i)
            ))
            for i in range(poly_degree)
        ])

        den = 1 - num * s
        
        roots -= num / den

        if np.allclose(f(roots), 0, atol=atol):
            return roots
    
    raise RootsNotFoundError()
    
    
# !! Примечание для нахождения корней полиномов
# Алгоритм может застревать в одной точке, если она близко к корню
# (у меня начинает застревать примерно около 1e-8)
# Для увеличения точности можно добавить, например, метод Ньютона
# с инициализацией в точках, найденных реализованными методами
# Но в данной задаче хватает такой точности


# ======================================================================
# Функции для поиска собственных чисел и векторов
# ======================================================================

    
def get_eigen_values(matrix: np.array):
    char_poly_coeffs = Faddeev_LeVerrier_algorithm(matrix)
    char_poly_roots = Aberth_method(char_poly_coeffs)
    return char_poly_roots  
    
    
def get_eigen(matrix: np.array):
    eigen_values = get_eigen_values(matrix).real
    
    size = matrix.shape[0]
    vectors = np.zeros((size, size))
    for i, eigen_value in enumerate(eigen_values):
        # Вычитаем собственное число из главной диагонали
        r = matrix - np.eye(size) * eigen_value
        r = r.real
        
        # Система имеет бесконечное число решений
        # Пусть последний корень равен -1
        # Решим уравнения, кроме последнего
        
        sub_sol = np.linalg.solve(r[:-1, :-1], r[:-1, -1])
        sub_sol = np.append(sub_sol, -1)
        vectors[i, :] = sub_sol
        
    # Нормализуем векторы
    norm = np.linalg.norm(vectors, axis=1)
    vectors = (vectors.T / norm).T
    # Сортируем по собственным числам
    sort = np.argsort(eigen_values)[::-1]
    return (eigen_values[sort], vectors[sort])
    

# Подробнее по получение собственных векторов
# [a b c] [x]   [ax + by + cz]   [0]
# [d e f] [y] = [dx + ey + fz] = [0]
# [g h i] [z]   [gx + hy + iz]   [0]
# 
# пусть z=-1, тогда
# [ax + by - c] = [0]
# [dx + ey - f]   [0]
#   
# то есть
# [a b] [x] = [-c]
# [d e] [y]   [-f]
# 

