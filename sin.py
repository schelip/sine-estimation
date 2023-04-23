import math, timeit, sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, pow, factorial
from typing import List

MACLAURIN_TERMS = 6 # Taylor order 12
PADE_ORDER = (7, 4)
coefs_maclaurin = [1, -1/6, 1/120, -1/5040, 1/362880, -1/39916800]
coefs_pade_num = [1, 0, -241/1650, 0, 601/118800, 0, -121/2268000]
coefs_pade_den = [0, 17/825, 0, 19/118800]

def horner(x: float, coefs: List[float], n_terms: int, extra: bool = True) -> float:
    """
    Calcula o valor de um polinômio usando o método de Horner.

    Args:
        x: Valor para o qual o polinômio será avaliado.
        coefs: Lista de coeficientes do polinômio.
        n_terms: Número de termos (ou grau do polinômio).
        extra: Se True, multiplica o resultado final por x.

    Returns:
        O valor do polinômio avaliado em x.
    """
    x_2 = x * x
    result = coefs[n_terms - 1]
    for i in range(n_terms - 2, -1, -1):
        if coefs[i] != 0:
            result *= x_2
            result += coefs[i]
    if coefs[0] == 0: result *= x_2
    if extra: result *= x
    return result

def sin_taylor_unopt(x: float) -> float:
    """Versão simples da função para estimar sin x usando
    uma Série de Maclaurin.

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """
    result = 0
    for i in range(MACLAURIN_TERMS):
        result += (pow(-1, i) * pow(x, 2 * i + 1) / factorial(2 * i + 1))
    return result

def sin_taylor_opt(x: float) -> float:
    """Versão otimizada com multiplicações reduzidas com o método de Horner
    da função para estimar sin x usando uma Série de Maclaurin.

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """
    return horner(x, coefs_maclaurin, MACLAURIN_TERMS)

def sin_pade_unopt(x: float) -> float:
    """Versão simples da função para estimar sin x usando
    a aproximação de Padè.

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """
    num = sum([coefs_pade_num[i] * pow(x, i + 1) for i in range(PADE_ORDER[0])])
    den = 1 + sum([coefs_pade_den[i] * pow(x, i + 1) for i in range(PADE_ORDER[1])])
    return num / den

def sin_pade_opt(x: float) -> float:
    """Versão otimizada com com multiplicações reduzidas o método de Horner
    da função para estimar sin x usando a aproximação de Padè.

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """ 
    num = horner(x, coefs_pade_num, PADE_ORDER[0])
    den = 1 + horner(x, coefs_pade_den, PADE_ORDER[1], False)
    return num / den

def estimate() -> tuple:
    """Solicita ao usuário o valor inicial, o valor final e o incremento para o cálculo dos valores de x.
    Em seguida, calcula os valores exatos de sin x para cada valor de x e os
    valores estimados por meio das expansões de Taylor e Padé.

    Returns:
        tuple: valores de x, seno exato, seno por taylor, seno por padè
    """
    start = eval(input('Initial value [-pi/4]: ') or "-(pi / 4)")
    end = eval(input('Final value [pi/4]: ') or "pi / 4")
    step = eval(input('Step [0.1]: ') or "0.1")

    x_vals = np.arange(start, end, step)

    y_exact = np.array([math.sin(x) for x in x_vals])
    y_taylor = np.array([sin_taylor_opt(x) for x in x_vals])
    y_pade = np.array([sin_pade_opt(x) for x in x_vals])

    return x_vals, y_exact, y_taylor, y_pade

def export(estimate: tuple):
    """Se o usuário fornecer um nome de arquivo, cria um arquivo de texto e escreve as informações dos valores de x,
    os valores exatos de sin x, os valores estimados por meio das expansões de Taylor e Padé e os erros correspondentes em relação ao valor exato de sin x.
    Se o usuário não fornecer um nome de arquivo, não executa nenhuma ação

    Args:
        estimate (tuple): valores de x, seno exato, seno por Taylor, seno por Padè
    """
    values, exact, taylor, pade = estimate
    filename = input("File to export the error of the estimations [Skip]: ")

    if not filename:
        return
    
    with open(filename, "w") as file:
        file.write(f'''Comparison between the real and estimated values of sin x
Start: {values[0]} \t| End: {values[-1]}\n
Angle Value (rad)\t\t| Exact sin x\t\t\t\t| Taylor (to 11 power)\t\t| Taylor Error\t\t\t\t| Pade approximate [7\\4]\t| Pade Error\n{'-'*156}\n''')
                   
        taylor_error = [abs(exact[i] - taylor[i]) for i in range(len(exact))]
        pade_error = [abs(exact[i] - pade[i]) for i in range(len(exact))]

        for i in range(len(exact)):
            file.write(f'{values[i]:.16f}\t\t| {exact[i]:.16f}\t\t| {taylor[i]:.16f}\t\t| {taylor_error[i]:.16f}\t\t| {pade[i]:.16f}\t\t| {pade_error[i]:.16f}\n')

def time_profile():
    """Solicita ao usuário um valor de x, calcula o valor exato de sin x e calcula o resultado e o tempo de execução
    para cada um dos métodos de estimativa implementados no código.
    """
    x = eval(input('Value of x [pi/8]: ') or "pi / 8")
    exact = math.sin(x)
    print(f'''
Function \t\t| Result \t\t| Exec Time (10000x) \t| Error
--
unoptimized taylor \t| {sin_taylor_unopt(x)} \t| {timeit.timeit(lambda: sin_taylor_unopt(x), number=10000)} \t| {abs(sin_taylor_unopt(x) - exact)}
optimized taylor \t| {sin_taylor_opt(x)} \t| {timeit.timeit(lambda: sin_taylor_opt(x), number=10000)} \t| {abs(sin_taylor_opt(x) - exact)}
unoptmized padè \t| {sin_pade_unopt(x)} \t| {timeit.timeit(lambda: sin_pade_unopt(x), number=10000)} \t| {abs(sin_pade_unopt(x) - exact)}
optmized padè   \t| {sin_pade_opt(x)} \t| {timeit.timeit(lambda: sin_pade_opt(x), number=10000)} \t| {abs(sin_pade_opt(x) - exact)}
math.sin \t\t| {math.sin(x)} \t| {timeit.timeit(lambda: math.sin(x), number=10000)} \t| {abs(math.sin(x) - exact)}
    ''')

def plot_values(estimate: tuple, method: int = -1):
    """Plota os valores de sin x estimados pelos métodos de expansão de Taylor e Padé, juntamente com o valor exato de sin x.
    Também exporta os resultados para um arquivo de texto, opcionalmente

    Args:
        estimate (tuple): valores de x, seno exato, seno por Taylor, seno por Padè
        method (int): -1 | ambos, 0 | Maclaurin, 1 | Padè
    """
    x_vals, y_exact, y_taylor, y_pade = estimate
    plt.plot(x_vals, y_exact, label='Exact')
    if method == -1 or method == 0: plt.plot(x_vals, y_taylor, label='Taylor')
    if method == -1 or method == 1: plt.plot(x_vals, y_pade, label='Padè')
    plt.legend()

    export(estimate)

    plt.show()

def plot_error(estimate: tuple):
    """plota o erro absoluto de cada método de estimativa em relação ao valor exato de sin x.
    Também exporta os resultados para um arquivo de texto, opcionalmente

    Args:
        estimate (tuple): valores de x, seno exato, seno por Taylor, seno por Padè
    """
    x_vals, y_exact, y_taylor, y_pade = estimate
    plt.plot(x_vals, abs(y_taylor - y_exact), label='Taylor')
    plt.plot(x_vals, abs(y_pade - y_exact), label='Padè')
    plt.legend()

    export(estimate)

    plt.show()

def print_usage():
    """Imprime informações de uso para o usuário, mostrando as opções disponíveis no programa."""
    print('''
sin estimation comparison
    
Usage:
-t | --time-profile: prints the results, execution time and error of the implemented estimation functions
-v | --plot-values: plots the results of the implemented estimation functions
-e | --plot-error: plots the error of the implemented estimation functions
-x | --export: exports the results of the implemented estimation functions and error to a text file
-h | --help: shows this usage info
''')

if __name__ == '__main__':
    if len(sys.argv) <= 1 or sys.argv[1] in ['-h', '--help']:
        print_usage()
    elif sys.argv[1] in ['-t', '--time-profile']:
        time_profile()
    elif sys.argv[1] in ['-v', '--plot-values']:
        if len(sys.argv) == 3:
            if sys.argv[2] in ['-m', '--maclaurin']:
                plot_values(estimate(), 0)
            elif sys.argv[2] in ['-p', '--pade']:
                plot_values(estimate(), 1)
        else:
            plot_values(estimate(), -1)
    elif sys.argv[1] in ['-e', '--plot-error']:
        plot_error(estimate())
    elif sys.argv[1] in ['-x', '--export']:
        export(estimate())
    else:
        print_usage()