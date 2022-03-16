import random
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt

N = 100  # number of cells
minDistance = 0.8 # the minimum distance each 
rho = 0.8  # density
# vol = rho/N
L = np.power(N / rho, 1 / 3)  # length of the simulation
print("L is", L)


def x_rand(L):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    z = random.uniform(0, L)
    return x, y, z

"""Actual x, y, z"""
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
"""Trial x, y, z"""
x_t = np.zeros(N)
y_t = np.zeros(N)
z_t = np.zeros(N)


def distanceLoop():
    for i in range(N):
        x_t[i], y_t[i], z_t[i] = x_rand(L)
        #print("x_t is ", x_t)
        if i > 0:
            for j in range(i):
                """d_x,y,z are all place holders equations to find dx,dy,dz"""
                d_x = x[i] - x_t
                dx = d_x - L * np.round(d_x/L)
                #print("np.round(d_x)/L =", np.round(d_x/L))
                #print("d_x is: ", d_x)
                #print("dx is :", dx)
                d_y = y[i] - y_t
                dy = d_y - L * np.round(d_y/L)
                d_z = z[i] - z_t
                dz = d_z - L * np.round(d_z/L)
                distance = dx**2 + dy**2 + dz**2
                #print("distance is ", distance)
                if distance.any() < 0.8:
                    x_rand()
                else:
                    x[i] = x_t[i]
                    y[i] = y_t[i]
                    z[i] = z_t[i]


def cell_loop():
    for i in range(N):
        x[i], y[i], z[i] = x_rand(L)
    #print("x is :", x)
    distanceLoop()



#distanceLoop()

cell_loop()

print(x, y, z)

fp = open("point.txt", mode="w")
for i in range(N):
    line = str(x[i]) + " " + str(y[i]) + " " + str(z[i])
    fp.write("%s\n" % (line))
fp.close()

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(x, y, z)
plt.show()
