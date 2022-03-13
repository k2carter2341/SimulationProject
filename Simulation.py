import random
import numpy as np
import matplotlib.pyplot as plt

N = 100  # number of cells
rho = 0.8  # density
# vol = rho/N
L = np.power(N / rho, 1 / 3)  # length of the simulation
print("L is", L)


def x_rand(L):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    z = random.uniform(0, L)
    return x, y, z


x, y, z = x_rand(L)
# print(x,y,z)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)


def cell_loop():
    for i in range(N):
        x[i], y[i], z[i] = x_rand(L)


cell_loop()
print(x, y, z)

fp = open("point.txt", mode="w")
for i in range(N):
    line = str(x[i]) + " " + str(y[i]) + " " + str(z[i])
    fp.write("%s\n" % (line))
fp.close()

plt.scatter(x, y)
plt.show()
