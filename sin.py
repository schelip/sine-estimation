import math, timeit, sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi

global taylor_11, pade_7_4_11
taylor_11 = [
        1.00000000e+00, -1.66666667e-01,
        8.33333333e-03,  -1.98412698e-04,
        2.75573192e-06,  -2.50521084e-08]
pade_7_4_11 = [
        2.06060606e-02, 1.59932660e-04,
        1.00000000e+00, -1.46060606e-01,
        5.05892256e-03, -5.33509700e-05]

def sin_taylor_unopt(x: float, n: int) -> float:
    """Versão simples da função para estimar sin x usando
    uma Série de Maclaurin.

    A iteração i realiza:
        i multiplicações para positivar/negativar o termo;
        2i multiplicações para calcular o fatorial;
        2i multiplicações para calcular a potência;
        1 multiplicação entre os fatores do termo;
        1 divisão entre os fatores do termo;
    = 5i + 2 operações

    Args:
        x (float): valor em rad do ângulo para estimar o seno
        n (int): ordem da Série de Maclaurin

    Returns:
        float: uma estimativa de sin x
    """
    result = 0
    for i in range(math.ceil(n / 2)):
        result += (((-1) ** i) / math.factorial(2 * i + 1)) * (x ** (2 * i + 1))
    return result

def sin_taylor_opt(x: float, n: int) -> float:
    """Versão otimizada com multiplicações reduzidas
    da função para estimar sin x usando uma Série de Maclaurin.

    Os valores da última iteração são salvos em memória e reutilizados
    na próxima iteração. Assim, toda iteração i realiza exatamente
    6 operações de multiplicação/divisão.

    Args:
        x (float): valor em rad do ângulo para estimar o seno
        n (int): ordem da Série de Maclaurin

    Returns:
        float: uma estimativa de sin x
    """
    result = 0
    factorial = 1 
    factor = 1
    positive = 1
    power = x
    for _ in range(math.ceil(n / 2)):
        result += (positive / factorial) * power
        positive = -positive
        factor += 1
        factorial *= factor
        factor += 1
        factorial *= factor
        power *= x * x
    return result

def sin_taylor_predef(x: float) -> float:
    """Versão utilizando coeficientes predefinidos
    da função para estimar sin x usando uma Série de Maclaurin de ordem 11 (6 termos).

    Realiza 3 * 6 multiplicações, totalizando 18 operações.

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """
    global taylor_11
    power = x
    result = 0
    for f in taylor_11:
        result += power * f
        power *= x * x
    return result

def sin_pade_unopt(x: float, n: int, m: int, p: int) -> float:
    """Versão simples da função para estimar sin x usando
    a aproximação de Padè.

    Calcula os coeficientes pra equação utilizando um sistema linear,
    igualando a uma série de Maclaurin.

    Args:
        x (float): valor em rad do ângulo para estimar o seno
        n (int): ordem da Série de Maclaurin

    Returns:
        float: uma estimativa de sin x
    """
    p_coef = np.zeros(p + 1)
    for i in range(math.ceil(p / 2)):
        p_coef[2 * i + 1] = ((-1) ** i) / math.factorial(2 * i + 1)
    A = np.zeros((p + 1, n + m + 1))
    for i in range(p + 1):
        for j in range(min(i, m)):
            A[i, j] = - p_coef[i - j - 1]
        if m + i < p + 1:
            A[i, m + i] = 1
    coef = np.linalg.solve(A, p_coef)
    a_coef = coef[m:]
    b_coef = coef[:m]
    a = 0
    b = 1
    for i in range(n + 1):
        a += a_coef[i] * (x ** i)
    for i in range(m):
        b += b_coef[i] * (x ** i)
    return a / b

def sin_pade_predef(x: float) -> float:
    """Versão utilizando coeficientes predefinidos
    da função para estimar sin x usando uma estimativa de Padè de ordem 7, 4
    igualando com uma Série de Maclaurin de ordem 11 (6 termos).

    Realiza 2 + 3 * 4 multiplicações e 1 divisão, totalizando 15 operações.

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """
    global pade_7_4_11
    a_coef = pade_7_4_11[2:]
    b_coef = pade_7_4_11[:2]
    a = 0
    b = 1
    power = x
    for i in range(4):
        if i < 2:
            b += b_coef[i] * power
        a += a_coef[i] * power
        power *= x * x
    return a / b

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
    y_taylor = np.array([sin_taylor_predef(x) for x in x_vals])
    y_pade = np.array([sin_pade_predef(x) for x in x_vals])

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
unoptimized taylor \t| {sin_taylor_unopt(x, 11)} \t| {timeit.timeit(lambda: sin_taylor_unopt(x, 11), number=10000)} \t| {abs(sin_taylor_unopt(x, 11) - exact)}
optimized taylor \t| {sin_taylor_opt(x, 11)} \t| {timeit.timeit(lambda: sin_taylor_opt(x, 11), number=10000)} \t| {abs(sin_taylor_opt(x, 11) - exact)}
predefined taylor \t| {sin_taylor_predef(x)} \t| {timeit.timeit(lambda: sin_taylor_predef(x), number=10000)} \t| {abs(sin_taylor_predef(x) - exact)}
unoptmized padè \t| {sin_pade_unopt(x, 7, 4, 11)} \t| {timeit.timeit(lambda: sin_pade_unopt(x, 7, 4, 11), number=10000)} \t| {abs(sin_pade_unopt(x, 7, 4, 11) - exact)}
predefined padè \t| {sin_pade_predef(x)} \t| {timeit.timeit(lambda: sin_pade_predef(x), number=10000)} \t| {abs(sin_pade_predef(x) - exact)}
math.sin \t\t| {math.sin(x)} \t| {timeit.timeit(lambda: math.sin(x), number=10000)} \t| {abs(math.sin(x) - exact)}
    ''')

def plot_values(estimate: tuple):
    """Plota os valores de sin x estimados pelos métodos de expansão de Taylor e Padé, juntamente com o valor exato de sin x.
    Também exporta os resultados para um arquivo de texto, opcionalmente

    Args:
        estimate (tuple): valores de x, seno exato, seno por Taylor, seno por Padè
    """
    x_vals, y_exact, y_taylor, y_pade = estimate
    plt.plot(x_vals, y_taylor, label='Taylor')
    plt.plot(x_vals, y_exact, label='Exact')
    plt.plot(x_vals, y_pade, label='Padè')
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
        plot_values(estimate())
    elif sys.argv[1] in ['-e', '--plot-error']:
        plot_error(estimate())
    elif sys.argv[1] in ['-x', '--export']:
        export(estimate())
    else:
        print_usage()