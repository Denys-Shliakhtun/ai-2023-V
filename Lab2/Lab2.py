import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.metrics import mean_absolute_error

def plot_func(variable, functions, labels, title):
    for i in range(len(functions)):
        plt.plot(variable, functions[i], label=labels[i] if labels else '')
    plt.title(title)
    if labels:
        plt.legend()
    plt.show()

def max_mf_num(value, mf_funcs_lists, real_values_list):
    index_in_list = min(range(len(real_values_list)), key=lambda i: abs(real_values_list[i] - value))
    values = [mf[index_in_list] for mf in mf_funcs_lists]
    return values.index(max(values))

x_min = 1.8
x_max = 2.4
step = 0.001
X = np.linspace(x_min, x_max, round((x_max-x_min)/step+1))
Y = X*np.cos(2*X) + np.sin(X/2)
Z = np.sin(Y)+ np.cos(X/2)
input_mf_number = 6
output_mf_number = 12

plot_func(X, [Y], [], 'y(x)')
plot_func(X, [Z], [], 'z(x)')


# координати вершин функцій 
X_means = np.linspace(min(X), max(X), input_mf_number)
Y_means = np.linspace(min(Y), max(Y), input_mf_number)
Z_means = np.linspace(min(Z), max(Z), output_mf_number)

# трикутні функції приналежності
triangle_X_mf = [fuzz.trimf(X, [mean-0.3, mean, mean+0.3]) for mean in X_means]
triangle_Y_mf = [fuzz.trimf(Y, [mean-1, mean, mean+1]) for mean in Y_means]
triangle_Z_mf = [fuzz.trimf(Z, [mean-0.5, mean, mean+0.5]) for mean in Z_means]
plot_func(X, triangle_X_mf, [], 'Трикутні функції приналежності для Х')
plot_func(Y, triangle_Y_mf, [], 'Трикутні функції приналежності для Y')
plot_func(Z, triangle_Z_mf, [], 'Трикутні функції приналежності для Z')

rules = []
row = ['\033[91m' + "y\\x " + '\033[0m']
for i in range(1, input_mf_number + 1):
    row.append('\033[93m' + "mx" + str(i).ljust(4))
print(''.join(row))
for i in range(input_mf_number):
    row = ['\033[92m' + "my" + str(i + 1) + '\033[0m' + ' ']
    for j in range(input_mf_number):
        z = np.sin(Y_means[i])+ np.cos(X_means[j]/2)
        best_func = max_mf_num(z, triangle_Z_mf, Z)
        row.append("mf" + str(best_func + 1))
        rules.append([j, i, best_func])
    print(''.join(["{:<6}".format(elem) for elem in row]))

print("\nВсі правила:")
for rule in rules:
    print(f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) then (z is mf{rule[2] + 1})")

Z_modelled = []
for i in range(len(X)):
    best_X = max_mf_num(X[i], triangle_X_mf, X)
    best_Y = max_mf_num(Y[i], triangle_Y_mf, Y)
    best_Z = None
    for rule in rules:
        if rule[0] == best_X and rule[1] == best_Y:
            best_Z = rule[2]
    Z_modelled.append(Z_means[best_Z])
relative_error = round(mean_absolute_error(Z, Z_modelled) * len(Z) / sum(abs(Z)) * 100, 2)
plot_func(X, [Z_modelled, Z], ["Triangular Z", "Z"], f"Фактичні значення функції Z та змодельовані\nВідносна помилка={relative_error}%")

# параметр sigma для побудови функцій Гауса
X_sigma = 0.1
Y_sigma = 0.3
Z_sigma = 0.1
# функції приналежності Гауса
gaussian_X_mf = [fuzz.gaussmf(X, mean, X_sigma) for mean in X_means]
gaussian_Y_mf = [fuzz.gaussmf(Y, mean, Y_sigma) for mean in Y_means]
gaussian_Z_mf = [fuzz.gaussmf(Z, mean, Z_sigma) for mean in Z_means]
plot_func(X, gaussian_X_mf, [], 'Гаусівські функції приналежності для Х')
plot_func(Y, gaussian_Y_mf, [], 'Гаусівські функції приналежності для Y')
plot_func(Z, gaussian_Z_mf, [], 'Гаусівські функції приналежності для Z')

rules = []
row = ['\033[91m' + "y\\x " + '\033[0m']
for i in range(1, input_mf_number + 1):
    row.append('\033[93m' + "mx" + str(i).ljust(4))
print(''.join(row))
for i in range(input_mf_number):
    row = ['\033[92m' + "my" + str(i + 1) + '\033[0m' + ' ']
    for j in range(input_mf_number):
        z = np.sin(Y_means[i])+ np.cos(X_means[j]/2)
        best_func = max_mf_num(z, gaussian_Z_mf, Z)
        row.append("mf" + str(best_func + 1))
        rules.append([j, i, best_func])
    print(''.join(["{:<6}".format(elem) for elem in row]))

print("\nВсі правила:")
for rule in rules:
    print(f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) then (z is mf{rule[2] + 1})")

Z_modelled = []
for i in range(len(X)):
    best_X = max_mf_num(X[i], gaussian_X_mf, X)
    best_Y = max_mf_num(Y[i], gaussian_Y_mf, Y)
    best_Z = None
    for rule in rules:
        if rule[0] == best_X and rule[1] == best_Y:
            best_Z = rule[2]
    Z_modelled.append(Z_means[best_Z])

relative_error = round(mean_absolute_error(Z, Z_modelled) * len(Z) / sum(abs(Z)) * 100, 2)
plot_func(X, [Z_modelled, Z], ["Gaussian Z", "Z"], f"Фактичні значення функції Z та змодельовані\nВідносна помилка={relative_error}%")