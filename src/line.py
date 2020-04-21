import numpy as np
import math
import matplotlib.pyplot as plt

xi = 3
yi = 2


angle = -math.pi/4

if abs(angle) == math.pi/2:
    x = xi * np.ones(100)
    y = np.linspace(-5,5,100)
elif angle == 0 or abs(angle) ==math.pi:
    x = np.linspace(-5,5,100)
    y = yi * np.ones(100)

else:
    x = np.linspace(-5,5,100)
    a = math.tan(angle)
    b = yi - (a*xi)
    y = (a*x) + b
plt.plot(x, y, '-r', label='y=')
plt.title('Graph of y=')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
