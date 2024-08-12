import numpy as np
import skfuzzy as fuzzy
import matplotlib.pyplot as plt


# 1. Побудувати трикутну і трапецієподібну функцію приналежності

x = np.arange(-5, 5, 1)

# Трикутна функція приналежності
y = fuzzy.trimf(x, [-3, 1, 3])
plt.subplot(121)
plt.plot(x, y)
plt.title('Трикутна функція приналежності')

# Трапецієподібна функція приналежності
y = fuzzy.trapmf(x, [-4, -1, 2, 3])
plt.subplot(122)
plt.plot(x, y)
plt.title('Трапецієподібна функція приналежності')

plt.show()


# 2. Побудувати просту і двосторонню функцію приналежності Гаусса, 
# утворену за допомогою різних функцій розподілу

x = np.arange(-10, 20, 1)

# Проста функція приналежності Гаусса
y = fuzzy.gaussmf(x, 5, 2)
plt.subplot(121)
plt.plot(x, y)
plt.title('Проста функція приналежності Гаусса')

# Двостороння функція приналежності Гаусса
y = fuzzy.gauss2mf(x, 1, 2, 3, 5)
plt.subplot(122)
plt.plot(x, y)
plt.title('Двостороння функція приналежності Гаусса')
plt.show()


# 3. Побудувати функцію приналежності "узагальнений дзвін", 
# яка дозволяє представляти нечіткі суб'єктивні переваги

x = np.arange(-5, 5, 0.1)

# Функція приналежності "узагальнений дзвін"
y = fuzzy.gbellmf(x, 1, 2, -1)
plt.plot(x, y)
plt.title('Функція приналежності "узагальнений дзвін"')
plt.show()


# 4. Побудувати набір сігмоїдних функцій: 
# основну односторонню, яка відкрита зліва чи справа; 
# додаткову двосторонню; 
# додаткову несиметричну

x = np.arange(-5, 5, 0.1)

y = fuzzy.sigmf(x, 0, -2)
plt.subplot(131)
plt.plot(x, y)

y = fuzzy.dsigmf(x, -1, 3, 1, 4)
plt.subplot(132)
plt.plot(x, y)

y = fuzzy.psigmf(x, 1, 1, 3, 4)
plt.subplot(133)
plt.plot(x, y)

plt.suptitle('Сигмоїдальні функції приналежності')
plt.show()

# 5. Побудувати набір поліноміальних функцій приналежності (Z-, PI- і S-функцій)

x = np.arange(0, 8, 0.1)

# Z-функція приналежності
y = fuzzy.zmf(x, 1, 3)
plt.subplot(131)
plt.plot(x, y)
plt.title("Z-функція приналежності")

# PI-функція приналежності
y = fuzzy.pimf(x, 1, 2, 3, 5)
plt.subplot(132)
plt.plot(x, y)
plt.title('PI-функція приналежності')

# S-функція приналежності
y = fuzzy.smf(x, 1, 3)
plt.subplot(133)
plt.plot(x, y)
plt.title('S-функція приналежності')

plt.show()


# 6. Побудувати мінімаксну інтерпретацію логічних операторів 
# з використанням операцій пошуку мінімуму і максимуму

x = np.arange(-5, 5, 0.1)

# Функції приналежності "узагальнений дзвін"
y1 = fuzzy.gbellmf(x, 1, 2, -1)
y2 = fuzzy.gbellmf(x, 1, 2, 1)

# Кон'юнкція (min)
min_func = np.fmin(y1, y2)
plt.subplot(121)
plt.plot(x, y1, linestyle='--')
plt.plot(x, y2, linestyle='--')
plt.plot(x, min_func)
plt.title('Кон\'юнкція (min)')

# Диз'юнкція (max)
max_func = np.fmax(y1, y2)
plt.subplot(122)
plt.plot(x, y1, linestyle='--')
plt.plot(x, y2, linestyle='--')
plt.plot(x, max_func)
plt.title('Диз\'юнкція (max)')

plt.suptitle("Мінімаксна інтерпретація логічних операторів")
plt.show()


# 7. Побудувати вірогідну інтерпретацію кон'юнктивну і диз'юнктивних операторів

x = np.arange(-5, 5, 0.1)

# Функції приналежності "узагальнений дзвін"
y1 = fuzzy.gbellmf(x, 1, 2, -1)
y2 = fuzzy.gbellmf(x, 1, 2, 1)

# Кон'юнкція (min)
min_func = y1 * y2
plt.subplot(121)
plt.plot(x, y1, linestyle='--')
plt.plot(x, y2, linestyle='--')
plt.plot(x, min_func)
plt.title('Кон\'юнкція (min)')

# Диз'юнкція (max)
max_func = y1 + y2 - y1 * y2
plt.subplot(122)
plt.plot(x, y1, linestyle='--')
plt.plot(x, y2, linestyle='--')
plt.plot(x, max_func, label='Max function')
plt.title('Диз\'юнкція (max)')

plt.suptitle("Вірогідна інтерпретація кон'юнктивних і диз'юнктивних операторів")
plt.show()


# 8. Побудувати доповнення нечіткої множини, 
# яке описує деяке розмите судження і представляє собою 
# математичний опис вербального вираження, 
# який заперечує це нечітка множина

x = np.arange(-5, 5, 0.1)

# Трапецієподібна функція приналежності та її заперечення
y0 = fuzzy.trapmf(x, [-2, 0, 2, 4])
y1 = 1 - y0
plt.plot(x, y0, label='Функція')
plt.plot(x, y1, linestyle='--', label='Заперечення функції')
plt.title('Трапецієподібна функція приналежності та її заперечення')
plt.legend()
plt.show()
