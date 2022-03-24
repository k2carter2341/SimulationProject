from ast import Or
from cmath import log, sqrt
import random
from re import T
from statistics import variance
from turtle import distance
import numpy as np                   
import matplotlib.pyplot as plt

N = 100  # number of cells
rho = 0.8  # density
vol = N/rho # volume
L = np.power(vol, 1 / 3)  # length of the simulation
Lx = L
Ly = L
Lz = L
#global epsilon, sigma, sigma12, sigma6, rcut, rcut2, u
epsilon = 1
sigma = 1
sigma12 = sigma ** 12
sigma6 = sigma ** 6
rcut = 2.5
rcut2 = rcut * rcut
u = 3
print("L = ", L)
T = 1
size = int((N * (N-1))/2)
r = np.zeros(size)
pot = np.zeros(size)

steps = 100 #how many steps 
deltaT = 0.005
kb = (1.38064852e-23)       #this is 1.381 * 10**-23 refer to power check
K=1 # for temperature control
m = 0.0001  #temporary 

minDist = 0.9 # the minimum distance each 
iseed = 10

random.seed(iseed)

#--------------------------------------------------------------------------
#  creating of Force
#--------------------------------------------------------------------------
def Force(u):
    fp = open("potential.txt", mode="w")
    #tempory
    epsilon = 1
    sigma = 1
    sigma12 = sigma ** 12
    sigma6 = sigma ** 6
    rcut = 2.5
    rcut2 = rcut * rcut
    u = 0
    k = 0
    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dx = dx - Lx * np.round(dx/Lx) # minimum image distance

            dy = y[i] - y[j]
            dy = dy - Ly * np.round(dy/Ly)

            dz = z[i] - z[j]
            dz = dz - Lz * np.round(dz/Lz)

            dr2 = dx**2 + dy**2 + dz**2
            #inv = inverse
            if dr2 < rcut2:
                dr2inv = 1/dr2
                dr6inv = dr2inv * dr2inv * dr2inv
                dr12inv = dr6inv * dr6inv

                du = 4 * epsilon * ((sigma12 * dr12inv) - (sigma6 * dr6inv))
                line = str(np.sqrt(dr2)) + " " + str(du)
                #fp.write('%s  %s \n'%(np.sqrt(dr2), du))
                fp.write("%s \n" %(line))
                #print(du, i, j)
                k = k + 1
                r[k] = np.sqrt(dr2)
                pot[k] = du
                u = u + du

                wij = 4 * epsilon * (12 * sigma12 * dr12inv - 6 * sigma6 * dr6inv)
                fx[i] = fx[i] + wij * dx
                fy[i] = fy[i] + wij * dy
                fz[i] = fz[i] + wij * dz

                fx[j] = fx[j] - wij * dx
                fy[j] = fy[j] - wij * dy
                fz[j] = fz[j] - wij * dz

                
    fp.close()
    return u/N
    #print(u) -- figure this out



#--------------------------------------------------------------------------
#  generate 3 random numbers between 0 and L
#--------------------------------------------------------------------------
def x_rand(L):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    z = random.uniform(0, L)
    return x, y, z

""" Initialize array x, y, z"""
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

#--------------------------------------------------------------------------
#  function to generate randomc ocnfiguration s.t. distance between 2 particles > minDist
#--------------------------------------------------------------------------
def InitConf(minDist):
    x[0], y[0], z[0] = x_rand(L) # choose an arbitrary position for the very 1st particle
    i = 0
    while i < N:
            x_t, y_t, z_t = x_rand(L) # trial position
            iflag = 1 # flag for accepting trial position in x, y, z list if dist > minDist
            for j in range(i): # look for all possible pairs
                dx = x[j] - x_t
                dx = dx - L * np.round(dx/L) # minimun image distance

                dy = y[j] - y_t
                dy = dy - L * np.round(dy/L)

                dz = z[j] - z_t
                dz = dz - L * np.round(dz/L)

                dr2 = dx**2 + dy**2 + dz**2
                if(dr2 < minDist*minDist):
                    iflag = 0 # iflag=0 means don't accept the trial position: see later lines
                    break
            if(iflag==1): # this line will reach (i) by above break statement or (ii) after finishing above for loop
                x[i] = x_t; y[i] = y_t; z[i] = z_t; i = i + 1

#--------------------------------------------------------------------------
#  function to calculate distance of 2 particles (x1,y1,z1) and (x2,y2,z2)
#--------------------------------------------------------------------------
def dist(x1, y1, z1, x2, y2, z2, Lx, Ly, Lz):   #Use these variables in gaussian def
    dx = x1 - x2
    dx = dx - Lx * np.round(dx/Lx) # minimum image distance

    dy = y1 - y2
    dy = dy - Ly * np.round(dy/Ly)

    dz = z1 - z2
    dz = dz - Lz * np.round(dz/Lz)

    dr2 = dx**2 + dy**2 + dz**2
    dr = np.sqrt(dr2)

    return dr
#=========================================================================
#=========================================================================
#  MAIN fnction to call all functiona as required
#=========================================================================
#=========================================================================

Initial_pos();      #unsure/dont know what variable you named the initial positions and velocities under or where
Initial_velocity();     
Force();            #not sure if this is the old force or new force??
Fxold=Fx[i];Fyold=Fy[i]; Fzold=Fz[i];   
for i in range(steps):      #We just decide how many steps we want --> made a variable so we can change it in one place

mu = 0
sigma = np.sqrt(T)
vx = np.zeros(N)
vy = np.zeros(N)
vz = np.zeros(N)
    
for i in range(N): 
    temp = random.gauss(mu, sigma)      # is this the force? 
    temp2 = random.gauss(mu, sigma)
    temp3 = random.gauss(mu, sigma)
    vx[i] = temp
    vy[i] = temp2
    vz[i] = temp3
    #print(vx[i])
    
# plotting a graph 

plt.figure()
#plt.plot(vx) 
#plt.plot(vy) 
#plt.plot(vz) 
plt.hist(vx, bins = 30)
plt.show()
"""Calculating velocities center of mass"""
avg_vx = 0
avg_vy = 0
avg_vz = 0
for i in range(N):
    avg_vx = avg_vx + vx[i]
    avg_vy = avg_vy + vy[i]
    avg_vz = avg_vz + vz[i]

vxCoM = avg_vx/N
vyCoM = avg_vy/N
vzCoM = avg_vz/N


"""Subracting velocities center of mass"""

for i in range(N):
    vx[i] = vx[i] - vxCoM
    vy[i] = vy[i] - vyCoM
    vz[i] = vz[i] - vzCoM
    


avg_vx = 0
avg_vy = 0
avg_vz = 0
for i in range(N):
    avg_vx = avg_vx + vx[i]
    avg_vy = avg_vy + vy[i]
    avg_vz = avg_vz + vz[i]

SubtractedCoMx = avg_vx/N
SubtractedCoMy = avg_vy/N
SubtractedCoMz = avg_vz/N


print("Center of mass", vxCoM)
print("Updated Center of Mass", SubtractedCoMx, SubtractedCoMy, SubtractedCoMz)





InitConf(minDist) # calling function to generate initial configuration
value = Force(u)
print(u)
print(value)

plt.figure()
plt.plot(r, pot, "o")
plt.show()
#exit()
# Call dist() to calculate all the diatances. Just for checking 
fp = open("dist.txt", mode="w")
npair = int(N*(N-1)/2)
print(" \n")
print("npair= %s" %(npair))

r = np.zeros(npair)
index = np.zeros(npair)
ind = -1
for i in range(N):
    for j in range(i+1, N, 1):
        ind = ind + 1
        #r1 = dist(x[i], y[i], z[i], x[j], y[j], z[j], L, L, L)
        #r[ind] = r1
        fp.write("%s  %s\n" %(ind, r[ind]))
        index[ind] = ind 
fp.close()
#--------------------------------------------------------------------------
# Plot scattered plot
#--------------------------------------------------------------------------
fp = open("point.txt", mode="w")
for i in range(N):
    line = str(x[i]) + " " + str(y[i]) + " " + str(z[i])
    fp.write("%s\n" % (line))
fp.close()

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(x, y, z)
plt.show()
#--------------------------------------------------------------------------
# Plot distances among all particles
#--------------------------------------------------------------------------
fig = plt.figure()
plt.plot(index, r, "o")
plt.axhline(y=minDist, color='r', linestyle='--')
plt.show()


#Gaussian Distribution
mu = 0
sigma = sqrt(T)
vx = []
vy = []
vz = []
    
for i in range(1000): 
    temp = random.gauss(mu, sigma)
    temp2 = random.gauss(mu, sigma)
    temp3 = random.gauss(mu, sigma)
    vx.append(temp)
    vy.append(temp2)
    vz.append(temp3) 
    
     
        
# plotting a graph 
#plt.plot(vx) 
#plt.plot(vy) 
#plt.plot(vz) 
#plt.hist(vx, density= True, bins = 30)

sum = 0
sum1 = 0
sum2 = 0
for i in range(N):
    sum = sum + vx[i]

    
    vxCoM = sum/N

for i in range(N):
    vx[i] = vx[i] - vxCoM
    
    sum1 = sum1 + vx[i]
    UpdatedCoM = sum1/N

print(UpdatedCoM)


print(vxCoM)
#above is from Friday. An we apparently found the force so not sure how to put this in the defs below

#----------------------------------------------------------------------
# Brute Force/ Velocity Verlet Alg.
#----------------------------------------------------------------------
def Integration():
    # this find the position of the particles when the force acts upon it
    Force();    
    for i in range(N):
        # the positions are being overridden in each component
        x[i]= x[i] + vx[i]*deltaT + (1/2)*(Fxold[i]/m)*(deltaT*deltaT)    
        y[i]= y[i] + vy[i]*deltaT + (1/2)*(Fyold[i]/m)*(deltaT*deltaT)   
        z[i]= z[i] + vz[i]*deltaT + (1/2)*(Fzold[i]/m)*(deltaT*deltaT)

    Force();    

    for i in range(N):
        vx[i] = vx[i] + ((Fxold[i]+Fx[i])/2*m)*(deltaT*deltaT)       # the velocities are being overridden in each component
        vy[i] = vy[i] + ((Fyold[i]+Fy[i])/2*m)*(deltaT*deltaT)       #the velocities are being overridden
        vz[i] = vz[i] + ((Fzold[i]+Fz[i])/2*m)*(deltaT*deltaT)     #dont know why an error show up with the () because they are closed

#This will be used in Temperature control def
#Kinetic energy = (1/2)mv^2
def KE():
    sum = 0.0
    for i in range(N):
        sum = sum + (1/2)*m[i]*vx[i]*vx[i]               #we have a problem with m because we dont have mass initialized yet
        sum = sum + (1/2)*m[i]*vy[i]*vy[i]
        sum = sum + (1/2)*m[i]*vz[i]*vz[i]
        

#---------------------
#Temperature Control -- since the particels are moving the velocities are scaling (T is directly proportional to v^2) if we dont do this we wont know the temperature and cant keep it fixed. It will be an E, V, N simulation
#---------------------
def velscaling():
    #KE(K) 
    #K=(3N-4)(1/2)kbT   (3N-4) comes from degrees of freedom and maing sure that the plots dont shift
    Tk = (2*K)/(kb*(3*N-4))     #kb is the boltzmann constant

    #simulation for temperature=T0
    fact = sqrt(T0/Tk)      #sorry forget what T0 is 
    for i in range(N):
        vx[i]=vx[i]*fact
        vy[i]=vy[i]*fact
        vz[i]=vz[i]*fact
